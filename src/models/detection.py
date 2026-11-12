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

    results = model_plate.predict(vehicle_crop_img, verbose=False)
    if not results or len(results) == 0:
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Lấy bounding box có confidence cao nhất
    best_idx = boxes.conf.argmax().item()
    best_bbox = boxes.xyxy[best_idx].cpu().numpy().tolist()
    return [float(x) for x in best_bbox]