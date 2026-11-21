import os
import logging

logger = logging.getLogger(__name__)

# Đường dẫn tới 2 file trọng số
MODEL_HELMET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "weights_helmet", "helmet_best.pt"))
MODEL_PLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "weights_plate", "plate_best.pt"))

# Khởi tạo mô hình ở chế độ lazy loading để tránh crash/treo ứng dụng khi import
model_helmet = None
model_plate = None


def _load_models():
    """Khởi tạo mô hình YOLO an toàn khi cần sử dụng."""
    global model_helmet, model_plate
    if model_helmet is not None and model_plate is not None:
        return True

    try:
        from ultralytics import YOLO

        if model_helmet is None:
            if os.path.exists(MODEL_HELMET_PATH):
                model_helmet = YOLO(MODEL_HELMET_PATH)
                logger.info(f"Loaded Helmet model: {model_helmet.names}")
            else:
                logger.warning(f"File trọng số không tồn tại: {MODEL_HELMET_PATH}")

        if model_plate is None:
            if os.path.exists(MODEL_PLATE_PATH):
                model_plate = YOLO(MODEL_PLATE_PATH)
                logger.info(f"Loaded Plate model: {model_plate.names}")
            else:
                logger.warning(f"File trọng số không tồn tại: {MODEL_PLATE_PATH}")

        return model_helmet is not None and model_plate is not None
    except Exception as e:
        logger.error(f"Lỗi khởi tạo mô hình YOLO: {e}")
        return False

# Giữ lại tên cũ: dùng làm ngưỡng mặc định cho các class không có ngưỡng riêng
# (class 0, 2, 3, 4). Giá trị đổi từ 0.3 -> 0.40 theo yêu cầu.
CONF_THRESHOLD = 0.40

# Ngưỡng riêng theo class_id. GIÁ TRỊ KHỞI ĐẦU, chưa được tối ưu bằng dữ liệu đánh giá.
#   1: bike                     -> 0.35 (giữ độ nhạy, vì xe là "mỏ neo" để ghép cặp)
#   5: driver_without_helmet    -> 0.45 (class gây phạt nên cần chắc chắn hơn)
#   6: passenger_without_helemt -> 0.45
CLASS_CONF_THRESHOLDS = {
    1: 0.35,
    5: 0.45,
    6: 0.45,
}

# Ngưỡng conf truyền cho model_helmet.track(). Đặt tường minh để không phụ thuộc
# giá trị mặc định của Ultralytics (0.25 ở các bản hiện hành) và phải <= mọi
# ngưỡng theo class ở trên, để việc lọc theo class diễn ra SAU tracker.
TRACK_CONF = 0.25


def _threshold_for(class_id):
    return CLASS_CONF_THRESHOLDS.get(int(class_id), CONF_THRESHOLD)


def reset_tracker():
    """Reset trạng thái ByteTrack trước mỗi phiên video mới để tránh giữ track_id cũ."""
    if model_helmet is not None and getattr(model_helmet, "predictor", None) is not None:
        model_helmet.predictor = None


def track_vehicles_and_helmets(image):
    """
    Theo dõi phương tiện, nón và lỗi không đội nón bằng ByteTrack.
    """
    if model_helmet is None:
        _load_models()

    if model_helmet is None:
        return []

    results = model_helmet.track(
        image,
        persist=True,
        tracker="bytetrack.yaml",
        conf=TRACK_CONF,
        verbose=False,
    )

    detections = []
    if not results or len(results) == 0:
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    track_ids = None
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)

    for i, (bbox, class_id, conf) in enumerate(zip(xyxy, class_ids, confs)):
        if conf > _threshold_for(class_id):
            tid = int(track_ids[i]) if track_ids is not None else None
            detections.append({
                "bbox": [float(x) for x in bbox],
                "track_id": tid,
                "class_id": int(class_id),
                "conf": float(conf),
            })

    return detections


def verify_violation_in_crop(crop_img):
    """
    Xác minh giai đoạn 2 trên ảnh crop cận cảnh (người + xe):
    Khi chạy trên toàn khung hình (4K/1080p thu về 640), đầu người ở xa rất nhỏ nên
    mũ tối màu/nửa đầu dễ bị nhầm thành không mũ. Hàm này chạy model_helmet trực tiếp
    trên ảnh crop cận cảnh tại thời điểm đủ chu kỳ debouncing để kiểm tra lại.
    """
    if crop_img is None or getattr(crop_img, "size", 0) == 0:
        return False

    if model_helmet is None:
        _load_models()

    if model_helmet is None:
        return False

    results = model_helmet.predict(crop_img, conf=TRACK_CONF, verbose=False)
    if not results or len(results) == 0:
        return False

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False

    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    no_helmets = []
    helmets = []
    for bbox, cls_id, conf in zip(xyxy, class_ids, confs):
        c = float(conf)
        cid = int(cls_id)
        if c <= _threshold_for(cid):
            continue
        b = [float(x) for x in bbox]
        if cid in (5, 6):
            no_helmets.append((b, c))
        elif cid in (0, 3):
            helmets.append((b, c))

    if not no_helmets:
        return False

    from src.features.processing import _iou, HELMET_CONFLICT_IOU
    for nh_box, nh_conf in no_helmets:
        if not any(h_conf >= nh_conf and _iou(nh_box, h_box) >= HELMET_CONFLICT_IOU for h_box, h_conf in helmets):
            return True

    return False


# Ngưỡng lọc biển số xe máy trong vùng crop xe: loại biển 1 dòng của ô tô (w/h > 2.2),
# biển ở phần trên/rìa nền sau xe (cy < 35% cao hoặc cx sát rìa < 15% ngang) và biển quá nhỏ ở xa.
PLATE_CONF_THRESHOLD = 0.45
PLATE_MIN_ASPECT = 0.80
PLATE_MAX_ASPECT = 2.20
PLATE_MIN_Y_RATIO = 0.35
PLATE_SIDE_MARGIN_RATIO = 0.15
PLATE_MIN_W_RATIO = 0.12


def detect_plate_in_crop(vehicle_crop_img):
    """
    Dùng model_plate phát hiện vị trí biển số trên ảnh crop của xe máy.
    """
    if vehicle_crop_img is None or vehicle_crop_img.size == 0:
        return None

    if model_plate is None:
        _load_models()

    if model_plate is None:
        return None

    h_crop, w_crop = vehicle_crop_img.shape[:2]
    if h_crop <= 0 or w_crop <= 0:
        return None

    results = model_plate.predict(vehicle_crop_img, conf=PLATE_CONF_THRESHOLD, verbose=False)
    if not results or len(results) == 0:
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    best_bbox = None
    best_conf = -1.0
    min_x = PLATE_SIDE_MARGIN_RATIO * w_crop
    max_x = (1.0 - PLATE_SIDE_MARGIN_RATIO) * w_crop
    for bbox, conf in zip(xyxy, confs):
        c = float(conf)
        if c < PLATE_CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        aspect = bw / bh
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if not (PLATE_MIN_ASPECT <= aspect <= PLATE_MAX_ASPECT):
            continue
        if cy < PLATE_MIN_Y_RATIO * h_crop or not (min_x <= cx <= max_x) or bw < PLATE_MIN_W_RATIO * w_crop:
            continue
        if c > best_conf:
            best_conf = c
            best_bbox = [x1, y1, x2, y2]

    return best_bbox