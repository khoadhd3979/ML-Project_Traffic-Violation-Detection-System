import cv2
import numpy as np
import os
import math
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số cho việc ghép người (class 5/6) với xe (class 1).
# Các giá trị là điểm khởi đầu, chưa được tối ưu bằng dữ liệu đánh giá.
# ---------------------------------------------------------------------------
CLASS_BIKE = 1
NO_HELMET_CLASSES = (5, 6)     # driver_without_helmet, passenger_without_helemt
HELMET_CLASSES = (0, 3)        # driver_with_helmet, passenger_with_helemt

# Mức mở rộng bbox xe (giữ nguyên giá trị cũ: top 35%, mỗi bên 10%, đáy 10%)
EXPAND_SIDE = 0.10
EXPAND_TOP = 0.35
EXPAND_BOTTOM = 0.10

# Tỷ lệ diện tích bbox NGƯỜI nằm trong vùng xe mở rộng (containment, không phải IoU).
# Theo yêu cầu: ghép cặp khi (tâm người trong vùng xe mở rộng) HOẶC (containment >= 0.20).
MIN_CONTAINMENT = 0.20

# True  = đúng yêu cầu (center HOẶC overlap). Cho phép ghép chỉ dựa vào overlap khi tâm
#         người nằm NGOÀI vùng xe mở rộng; đây là đường có nguy cơ ghép nhầm người đi bộ.
# False = chỉ ghép khi tâm người nằm trong vùng xe mở rộng.
ALLOW_OVERLAP_ONLY_MATCH = True

# Phạt cost cho cặp chỉ ghép được nhờ overlap (tâm ngoài vùng) để cặp có tâm trong vùng luôn thắng.
OVERLAP_ONLY_COST_PENALTY = 1.0

# Giới hạn tỷ lệ chiều cao người / chiều cao xe để loại người quá nhỏ (xa, nền)
# hoặc quá lớn (gần camera). Đặt MIN=0 và MAX=inf để tắt.
MIN_HEIGHT_RATIO = 0.15
MAX_HEIGHT_RATIO = 3.0

# Nếu có nhiều xe ứng viên và xe tốt thứ hai có cost chỉ hơn xe tốt nhất < ngưỡng này
# (đơn vị: khoảng cách tâm chuẩn hóa theo kích thước xe) thì bỏ qua người đó (không đủ bằng chứng).
AMBIGUITY_MARGIN = 0.10

# Hai bbox xe có IoU >= ngưỡng này coi là cùng một xe bị detect đôi, không tính là ứng viên cạnh tranh.
DUPLICATE_BIKE_IOU = 0.70

# Nếu bbox "không nón" trùng (IoU) với bbox "có nón" từ mức này trở lên và bbox
# "có nón" có conf >= conf của "không nón" thì bỏ bbox "không nón" (mâu thuẫn).
HELMET_CONFLICT_IOU = 0.50

# Padding (tỷ lệ theo w/h của vùng evidence) khi crop ảnh bằng chứng.
EVIDENCE_PADDING_RATIO = 0.05


def process_plate(image, bbox):
    h_img, w_img = image.shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w_img, x_max), min(h_img, y_max)

    plate_crop = image[y_min:y_max, x_min:x_max]

    if plate_crop.size == 0:
        return None

    h_crop, w_crop = plate_crop.shape[:2]
    ratio = 200.0 / float(h_crop)
    plate_crop = cv2.resize(plate_crop, (int(w_crop * ratio), 200), interpolation=cv2.INTER_CUBIC)

    # Chuyển sang hệ màu YUV để chỉ tăng độ sáng (kênh Y) mà không làm hỏng màu (U, V)
    img_yuv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2YUV)

    # Dùng CLAHE để cân bằng sáng (khử bóng râm, chống chói)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    # Chuyển ngược lại về ảnh BGR (ảnh màu chuẩn)
    enhanced_plate = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return enhanced_plate


def _normalize_bbox(bbox):
    """Trả về (x1, y1, x2, y2) dạng float, hoặc None nếu bbox không hợp lệ."""
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def _inter_area(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(a, b):
    inter = _inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _expand_bike_bbox(bike_bbox):
    bx1, by1, bx2, by2 = bike_bbox
    bw = bx2 - bx1
    bh = by2 - by1
    return (
        bx1 - EXPAND_SIDE * bw,
        by1 - EXPAND_TOP * bh,
        bx2 + EXPAND_SIDE * bw,
        by2 + EXPAND_BOTTOM * bh,
    )


def _match_person_to_bike(person_bbox, bike_bbox):
    """
    Kiểm tra một bbox người có thể thuộc về một bbox xe hay không.

    Trả về (cost, reason):
        cost  : float (càng nhỏ càng khớp) nếu hợp lệ, None nếu bị loại.
        reason: "ok" hoặc lý do loại (dùng cho log debug).

    Điều kiện:
        1. Tâm bbox người nằm trong vùng xe mở rộng,
           HOẶC containment = inter(người, vùng xe mở rộng) / diện tích người >= MIN_CONTAINMENT
           (đường thứ hai có thể tắt bằng ALLOW_OVERLAP_ONLY_MATCH).
        2. Tỷ lệ chiều cao người / chiều cao xe nằm trong [MIN_HEIGHT_RATIO, MAX_HEIGHT_RATIO].
    """
    bx1, by1, bx2, by2 = bike_bbox
    bw = bx2 - bx1
    bh = by2 - by1
    exp = _expand_bike_bbox(bike_bbox)
    ex1, ey1, ex2, ey2 = exp

    px1, py1, px2, py2 = person_bbox
    cx = (px1 + px2) / 2.0
    cy = (py1 + py2) / 2.0

    center_in = (ex1 <= cx <= ex2) and (ey1 <= cy <= ey2)
    containment = _inter_area(person_bbox, exp) / _area(person_bbox)

    if not center_in:
        if not ALLOW_OVERLAP_ONLY_MATCH:
            return None, "center_outside"
        if containment < MIN_CONTAINMENT:
            return None, "center_outside_low_containment(%.2f)" % containment

    height_ratio = (py2 - py1) / bh
    if not (MIN_HEIGHT_RATIO <= height_ratio <= MAX_HEIGHT_RATIO):
        return None, "height_ratio(%.2f)" % height_ratio

    # Chi phí để chọn xe khi một người rơi vào vùng của nhiều xe:
    # khoảng cách tâm người - tâm xe, chuẩn hóa theo kích thước từng xe.
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    cost = math.hypot((cx - bcx) / bw, (cy - bcy) / bh)
    if not center_in:
        cost += OVERLAP_ONLY_COST_PENALTY
    return cost, "ok"


def crop_with_padding(frame, bbox, padding_ratio=EVIDENCE_PADDING_RATIO):
    """
    Crop `bbox` (x1, y1, x2, y2 theo hệ tọa độ của `frame`) kèm padding nhỏ,
    clamp vào giới hạn ảnh. Trả về None nếu frame rỗng hoặc bbox không hợp lệ /
    nằm hoàn toàn ngoài frame (không lưu ảnh rỗng hoặc crop sai tọa độ).
    Nếu bbox nhô ra ngoài frame, trả về phần lớn nhất nằm trong frame.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    nb = _normalize_bbox(bbox)
    if nb is None:
        return None

    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = nb
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio

    cx1 = int(math.floor(max(0.0, x1 - pad_x)))
    cy1 = int(math.floor(max(0.0, y1 - pad_y)))
    cx2 = int(math.ceil(min(float(w_img), x2 + pad_x)))
    cy2 = int(math.ceil(min(float(h_img), y2 + pad_y)))

    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


def check_helmet_violation(detections):
    """
    Kiểm tra vi phạm không đội nón bảo hiểm từ danh sách detections.
    Class 0: driver_with_helmet
    Class 1: bike
    Class 2: driver
    Class 3: passenger_with_helemt
    Class 4: passenger
    Class 5: driver_without_helmet
    Class 6: passenger_without_helemt

    Quy trình:
        1. Bỏ bbox "không nón" (5/6) nếu bị bbox "có nón" (0/3) mâu thuẫn
           (IoU >= HELMET_CONFLICT_IOU và conf của "có nón" >= conf của "không nón").
        2. Mỗi người "không nón" được gán cho TỐI ĐA MỘT xe: xe hợp lệ có cost nhỏ nhất.
           Nếu xe tốt thứ hai (không phải bản detect đôi của xe tốt nhất) chỉ kém
           dưới AMBIGUITY_MARGIN thì bỏ qua người đó.
        3. Xe có ít nhất một người được gán, và có track_id, là xe vi phạm.

    Trả về danh sách các xe máy vi phạm, mỗi phần tử:
        {
            "track_id": int,
            "bbox": [x1, y1, x2, y2],                 # bbox xe gốc (không mở rộng)
            "full_violator_bbox": [x1, y1, x2, y2],   # hợp nhất bbox xe gốc + bbox mọi người vi phạm đã ghép
        }
    """
    if not detections:
        return []

    motorcycles = []
    no_helmets = []
    helmets = []

    for det in detections:
        if not isinstance(det, dict):
            continue
        bbox = _normalize_bbox(det.get("bbox"))
        if bbox is None:
            continue

        cls_id = det.get("class_id")
        conf = det.get("conf")
        try:
            conf = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None

        item = {"det": det, "bbox": bbox, "conf": conf}
        if cls_id == CLASS_BIKE:
            motorcycles.append(item)
        elif cls_id in NO_HELMET_CLASSES:
            no_helmets.append(item)
        elif cls_id in HELMET_CLASSES:
            helmets.append(item)

    if not motorcycles or not no_helmets:
        return []

    # Bước 1: loại "không nón" bị "có nón" mâu thuẫn (ví dụ nón tối màu bị nhận nhầm).
    # Dùng IoU vì đây là so sánh hai bbox cùng một đối tượng (kích thước tương đương).
    # Nếu thiếu conf thì không loại (giữ hành vi cũ).
    candidates = []
    for nh in no_helmets:
        contradicted = False
        if nh["conf"] is not None:
            for h in helmets:
                if h["conf"] is None:
                    continue
                if h["conf"] >= nh["conf"] and _iou(nh["bbox"], h["bbox"]) >= HELMET_CONFLICT_IOU:
                    contradicted = True
                    break
        if contradicted:
            logger.debug("no_helmet bỏ do mâu thuẫn với bbox có nón: cls=%s conf=%s bbox=%s",
                         nh["det"].get("class_id"), nh["conf"], nh["bbox"])
            continue
        candidates.append(nh)

    # Bước 2: gán mỗi người cho tối đa một xe.
    assigned = {}  # idx xe -> list bbox người đã ghép
    for nh in candidates:
        scored = []
        for idx, bike in enumerate(motorcycles):
            cost, reason = _match_person_to_bike(nh["bbox"], bike["bbox"])
            if cost is None:
                logger.debug("loại cặp người-xe: bike_track=%s reason=%s person=%s",
                             bike["det"].get("track_id"), reason, nh["bbox"])
                continue
            scored.append((cost, idx))
        if not scored:
            continue
        scored.sort()
        best_cost, best_idx = scored[0]

        # Xe thứ hai không tính nếu thực chất là bản detect đôi của xe tốt nhất.
        rivals = [(c, i) for c, i in scored[1:]
                  if _iou(motorcycles[i]["bbox"], motorcycles[best_idx]["bbox"]) < DUPLICATE_BIKE_IOU]
        if rivals and (rivals[0][0] - best_cost) < AMBIGUITY_MARGIN:
            logger.debug("bỏ qua người vì mơ hồ giữa xe track=%s và track=%s (cost %.3f vs %.3f) person=%s",
                         motorcycles[best_idx]["det"].get("track_id"),
                         motorcycles[rivals[0][1]]["det"].get("track_id"),
                         best_cost, rivals[0][0], nh["bbox"])
            continue

        assigned.setdefault(best_idx, []).append(nh["bbox"])

    # Bước 3: xuất kết quả. Xe chưa có track_id vẫn tham gia gán ở bước 2 (để không
    # bị gán nhầm cho xe khác) nhưng chưa thể báo cáo vì live_stream cần track_id.
    violating_vehicles = []
    violating_track_ids = set()
    for idx in sorted(assigned):
        bike = motorcycles[idx]
        bike_track_id = bike["det"].get("track_id")
        if bike_track_id is None:
            continue
        if bike_track_id in violating_track_ids:
            continue
        violating_track_ids.add(bike_track_id)

        # Evidence: hợp nhất bbox xe GỐC (không phải bbox mở rộng) với bbox mọi người vi phạm đã ghép.
        boxes = [bike["bbox"]] + assigned[idx]
        full_violator_bbox = [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]
        violating_vehicles.append({
            "track_id": bike_track_id,
            "bbox": bike["det"]["bbox"],
            "full_violator_bbox": full_violator_bbox,
        })

    return violating_vehicles

