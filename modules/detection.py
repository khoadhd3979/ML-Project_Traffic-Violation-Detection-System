from ultralytics import YOLO

model = YOLO("models/best_second.pt")

CONF_THRESHOLD = 0.5


def detect_license_plate(image):
    """
    Theo dõi đa đối tượng bằng ByteTrack (thay cho detect đơn lẻ trước đây).

    Trả về: list[dict], mỗi phần tử tương ứng 1 phương tiện/biển số phát hiện được:
        {
            "bbox": [x1, y1, x2, y2],   # tọa độ theo hệ tọa độ của `image` đầu vào
            "track_id": int,             # ID theo dõi ổn định qua các frame (ByteTrack)
            "class_id": int,             # class phát hiện được (theo model.names)
        }

    Nếu ByteTrack chưa gán được track_id cho frame này (VD: frame đầu tiên,
    hoặc object vừa mất track), trả về list rỗng cho các box đó — main loop
    sẽ tự bỏ qua và chờ frame tiếp theo.
    """
    results = model.track(
        image,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        verbose=False,
    )

    detections = []
    boxes = results[0].boxes

    # boxes.id là None khi chưa có track nào được khởi tạo/duy trì cho frame này
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    for bbox, track_id, class_id, conf in zip(xyxy, track_ids, class_ids, confs):
        if conf > CONF_THRESHOLD:
            detections.append({
                "bbox": bbox,
                "track_id": int(track_id),
                "class_id": int(class_id),
            })

    return detections
