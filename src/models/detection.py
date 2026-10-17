import os
from ultralytics import YOLO

# Đường dẫn tới 2 file trọng số
MODEL_HELMET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "weights_helmet", "helmet_best.pt"))
MODEL_PLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "weights_plate", "plate_best.pt"))

# Load 2 model YOLOv8 riêng biệt
model_helmet = YOLO(MODEL_HELMET_PATH)
print(f"\n[INFO] Danh sách nhãn của Model Helmet: {model_helmet.names}\n")
model_plate = YOLO(MODEL_PLATE_PATH)

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

    Class dự kiến từ model_helmet:
        0: driver_with_helmet
        1: bike
        2: driver
        3: passenger_with_helemt
        4: passenger
        5: driver_without_helmet
        6: passenger_without_helemt

    Ngưỡng confidence được áp dụng theo từng class (CLASS_CONF_THRESHOLDS,
    mặc định CONF_THRESHOLD) sau khi tracker đã chạy.

    Trả về: list[dict], mỗi phần tử gồm:
        {
            "bbox": [x1, y1, x2, y2],
            "track_id": int hoặc None,
            "class_id": int,
            "conf": float
        }
    """
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

    Trả về:
        [x1, y1, x2, y2] (tọa độ Bounding Box của biển số theo hệ tọa độ của vehicle_crop_img)
        hoặc None nếu không phát hiện được biển số.
    """
    if vehicle_crop_img is None or vehicle_crop_img.size == 0:
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