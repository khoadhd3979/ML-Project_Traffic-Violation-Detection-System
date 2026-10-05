C:\Project\N3_HK1\MH\ML\
├── app/                        # 🖥️ PROJECT 2: FRONTEND UI (Streamlit)
│   ├── __init__.py
│   ├── main.py                 # File entry point khởi chạy giao diện Web (trước đây là tmp/app.py).
│   └── components/             # Các module cấu thành nên các trang/chức năng UI.
│       ├── __init__.py
│       ├── chatbot_ui.py       # Giao diện Chatbot hỗ trợ giải đáp thông tin cho người dùng.
│       ├── dashboard.py        # Trang tổng quan thống kê dữ liệu, biểu đồ vi phạm, lượng xe.
│       ├── live_monitor.py     # Giao diện giám sát luồng camera trực tiếp và kết quả AI trả về.
│       └── violation_table.py  # Trang hiển thị bảng danh sách các vi phạm và bộ lọc tra cứu.
│
├── src/                        # ⚙️ PROJECT 1 (BACKEND) & CORE ML SYSTEM
│   ├── __init__.py
│   ├── api/                    # Không gian dành cho Backend API (FastAPI/Flask) trong tương lai.
│   │   └── __init__.py
│   ├── data/                   # Layer xử lý kết nối Database & Data pipelines.
│   │   ├── __init__.py
│   │   ├── database_manager.py # Lớp DatabaseManager chứa các phương thức tương tác với SQLite.
│   │   ├── init_db.py          # Script để tạo mới cơ sở dữ liệu dựa trên file schema.sql.
│   │   ├── schema.sql          # File SQL định nghĩa kiến trúc bảng (tables) của Database.
│   │   └── seed_data.py        # Script dùng để nạp dữ liệu giả/mẫu (mock data) vào hệ thống.
│   ├── features/               # Các pipeline trích xuất đặc trưng, tiền xử lý dữ liệu.
│   │   ├── __init__.py
│   │   └── processing.py       # Code xử lý hình ảnh, crop khung hình biển số/xe trước khi đưa vào ML.
│   ├── models/                 # Chứa Core logic của Machine Learning Models.
│   │   ├── __init__.py
│   │   ├── detection.py        # Logic load và chạy model YOLO (phát hiện xe, biển số, mũ bảo hiểm).
│   │   ├── ocr_engine.py       # Lớp PlateOCR đảm nhiệm trích xuất ký tự (chữ/số) từ ảnh biển số đã crop.
│   │   └── ocr_worker_proc.py  # Tiến trình chạy ngầm (worker) để xử lý hàng đợi OCR song song giúp giảm tải.
│   └── utils/                  # Tiện ích dùng chung cho toàn bộ dự án.
│       ├── __init__.py
│       └── config.py           # File lưu trữ toàn bộ các hằng số, thông số (DB_PATH, ngưỡng YOLO/OCR...).
│
├── notebooks/                  # 📓 PROJECT 3: JUPYTER NOTEBOOKS
│   └── (trống)                 # Nơi chứa các file .ipynb dùng để EDA, Test model, Training, phân tích dữ liệu.
│
├── data/                       # 🗄️ THƯ MỤC LƯU TRỮ DỮ LIỆU & DATABASE
│   ├── evidence/               # Thư mục chứa hình ảnh/video bằng chứng (crop từ camera) ghi nhận vi phạm.
│   ├── processed/              # Nơi lưu trữ dữ liệu đã qua xử lý chuẩn bị cho training.
│   ├── raw/                    # Nơi lưu dữ liệu/video thô thu thập được từ camera thực tế.
│   └── traffic.db/parking.db   # File Database chính thức của toàn hệ thống (được config.py trỏ tới đây).
│
├── docs/                       # 📚 TÀI LIỆU DỰ ÁN
│   └── evidence_naming_convention.md # Quy chuẩn đặt tên cho các tệp tin bằng chứng vi phạm.
│
├── tests/                      # 🧪 UNIT TEST & INTEGRATION TEST
│   └── integration/
│       └── live_test.py        # Script chạy test tích hợp toàn bộ pipeline (từ Video -> YOLO -> OCR -> DB).
│
└── modules/                    # 📦 (Thư mục cũ)
    └── helmet_detection_v1/... # Chứa các file artifact, biểu đồ, labels, weights của đợt training mô hình cũ.
