# onfig.py — Cấu hình tập trung cho toàn bộ hệ thống LPR Smart Parking
# Chỉnh thông số TẠI ĐÂY — không cần đụng vào các file khác

import os

# CAMERA & YOLO
VIDEO_SOURCE: int | str = 0 # Nguồn video: 0 = webcam mặc định, hoặc đường dẫn file / RTSP URL

DISPLAY_SIZE: tuple[int, int] = (640, 480) # Kích thước cửa sổ hiển thị OpenCV (width, height)

YOLO_INTERVAL: int = 2 # Chạy YOLO mỗi N frame (giảm tải CPU, 1 = mỗi frame)

YOLO_INPUT_WIDTH: int = 640 # Resize frame về chiều rộng này (px) trước khi đưa vào YOLO

BBOX_MISS_THRESHOLD: int = 5 # Số frame miss liên tiếp trước khi xóa bbox (tăng để tránh nháy)


# OCR
OCR_COOLDOWN: float = 0.1 # 0.3(old) Giây — cooldown tối thiểu giữa 2 lần gửi crop vào OCR queue

BBOX_JUMP_THRESHOLD: int = 60 # Pixel — dịch chuyển tâm bbox > ngưỡng này → coi là biển số MỚI

VERIFY_INTERVAL: float = 0.1 # Giây — sau thời gian này, force re-verify để phát hiện xe vào/ra

# Regex tối thiểu để coi biển số là hợp lệ (Vietnam plate)
# Ví dụ hợp lệ: 29A-111.11, 51G1-23456, 30K-999.99
PLATE_REGEX: str = r"^\d{2}[A-Z]\d?[-–]?\d{3,5}\.?\d{0,2}$" 


# DATABASE

# Đường dẫn file SQLite (tương đối từ thư mục gốc project hoặc tuyệt đối)
DB_PATH: str = os.environ.get(
    "PARKING_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "parking.db"),
)


# BẢNG GIÁ (VNĐ)

PRICE_FLOOR: int      = 5_000    # Giá sàn (≤ 30 phút đầu)
PRICE_HALF_HOUR: int  = 8_000    # 30 phút < t < 60 phút
PRICE_PER_HOUR: int   = 10_000   # Mỗi giờ trọn vẹn
PRICE_EXTRA_HALF: int = 5_000    # Phút lẻ sau giờ trọn: ≤ 30 phút
PRICE_EXTRA_FULL: int = 8_000    # Phút lẻ sau giờ trọn: > 30 phút

MAX_CAPACITY: int = 100 # Sức chứa


# STREAMLIT DASHBOARD
AUTO_REFRESH_SEC: int = 5 # Tự động làm mới dashboard sau N giây (0 = tắt auto-refresh)

HISTORY_DISPLAY_LIMIT: int = 100 # Số bản ghi lịch sử hiển thị tối đa trong tab Lịch sử
