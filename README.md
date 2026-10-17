# 🚦 Hệ thống phát hiện vi phạm giao thông  
**Không đội mũ bảo hiểm & Nhận diện biển số**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-red?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

---

## 📖 Giới thiệu

Hệ thống được xây dựng nhằm **tự động phát hiện vi phạm không đội mũ bảo hiểm** của người điều khiển xe máy và **nhận diện biển số xe** từ các nguồn video/camera.  
Mục tiêu chính:

- Ổn định luồng nhận diện xe và ghép cặp hành vi không đội mũ bảo hiểm.
- Tách việc ghi nhận vi phạm khỏi kết quả OCR để giảm nguy cơ mất bản ghi.
- Thống nhất luồng xử lý video/camera, dữ liệu SQLite và giao diện Streamlit.

Hệ thống hỗ trợ nhiều nguồn đầu vào: webcam, video upload, RTSP stream, và image folder (theo cấu hình schema). Kết quả vi phạm được lưu trữ kèm ảnh bằng chứng (evidence) và hiển thị trực quan trên giao diện web.

---

## ✨ Tính năng nổi bật

- **🎯 Phát hiện & Tracking đa đối tượng**  
  Sử dụng hai model YOLOv8 riêng biệt: một cho xe máy/người đội mũ/không đội mũ, một cho biển số. Tích hợp ByteTrack với `persist=True` để gán `track_id` ổn định qua nhiều frame.

- **📐 Thuật toán ghép cặp mở rộng BBox**  
  Vùng BBox của xe được mở rộng linh hoạt: thêm 35% chiều cao phía trên, 20% chiều rộng hai bên, 10% chiều cao phía dưới. Ngưỡng overlap giảm xuống 20%, kết hợp điều kiện tâm BBox không mũ nằm trong vùng mở rộng → giảm đáng kể bỏ sót khi đầu người nằm lệch hoặc nhô khỏi BBox xe.

- **⚡ Asynchronous OCR Decoupling – Không mất vi phạm**  
  Khi phát hiện vi phạm, hệ thống **ghi ngay bản ghi với biển số UNKNOWN** và lưu ảnh evidence. Crop biển số được đưa vào hàng đợi OCR xử lý bất đồng bộ. Nếu OCR thành công, bản ghi được cập nhật `vehicle_id` tương ứng. Nếu OCR thất bại, vi phạm vẫn tồn tại để xác minh thủ công.

- **💾 Lưu trữ & Quản lý dữ liệu**  
  SQLite lưu trữ `violations`, `vehicles`, `processing_sessions`, `violation_rules`. Hỗ trợ cập nhật liên kết phương tiện sau OCR và reset dữ liệu demo (giữ nguyên `violation_rules`).

- **🖥️ Giao diện Streamlit trực quan**  
  Hiển thị live stream, lịch sử vi phạm, ảnh evidence dạng thumbnail, thống kê. Hỗ trợ tải video, chạy webcam/RTSP, và thao tác reset DB với xác nhận `XACNHAN`.

- **🔗 Tích hợp liền mạch**  
  Dùng chung generator `generate_live_stream()` cho cả camera và video upload, giảm trùng lặp logic. UI đọc dữ liệu trực tiếp từ SQLite thay vì mock data.

---

## 🏗️ Kiến trúc & Luồng xử lý

### Sơ đồ luồng dữ liệu

```
Video Input (Webcam / Video File / RTSP)
          │
          ▼
┌─────────────────────────────────────┐
│  Detection (YOLOv8 + ByteTrack)     │
│  - Model Helmet: xe máy, có mũ,     │
│    không mũ                         │
│  - Model Plate: biển số             │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Processing (ghép cặp BBox mở rộng) │
│  - Xác định vi phạm không đội mũ    │
│  - Lọc trùng theo track_id          │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Ghi nhận vi phạm NGAY LẬP TỨC      │
│  - Tạo violation với UNKNOWN        │
│  - Lưu ảnh evidence (crop xe)       │
│  - Gửi crop biển số vào OCR Queue   │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  OCR Worker (đa tiến trình)         │
│  - Xử lý crop biển số               │
│  - Trả về text biển số (nếu có)     │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Cập nhật Database                  │
│  - Nếu OCR thành công: tạo/tìm      │
│    vehicle, cập nhật vehicle_id     │
│  - Nếu thất bại: giữ UNKNOWN        │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Streamlit UI                       │
│  - Live monitor, lịch sử, evidence  │
│  - Thao tác reset, thống kê         │
└─────────────────────────────────────┘
```

### Luồng xử lý chính

1. **Đọc frame** từ camera hoặc video.
2. **Model Helmet** phát hiện xe máy / người có mũ / người không mũ; tracker gán `track_id`.
3. **`processing.py`** ghép detection không mũ với xe máy bằng BBox mở rộng và ngưỡng overlap 20%.
4. Khi xác định vi phạm, **`live_stream.py`** ghi sự kiện ngay với biển số `UNKNOWN`, lưu ảnh evidence vào `data/evidence/YYYYMMDD/`.
5. Crop biển số được gửi qua **OCR queue** để xử lý riêng.
6. Kết quả OCR (nếu có) được dùng để **cập nhật `vehicle_id`** cho violation đã ghi.
7. UI Streamlit đọc dữ liệu từ SQLite, hiển thị realtime và lịch sử.

---

## 🛠️ Công nghệ sử dụng

| Lĩnh vực | Công nghệ |
|----------|-----------|
| **AI / Computer Vision** | YOLOv8 (Ultralytics), ByteTrack, OpenCV |
| **Backend** | Python, Multiprocessing, Pathlib |
| **Cơ sở dữ liệu** | SQLite |
| **Giao diện** | Streamlit |
| **OCR** | OCR Engine (tùy chỉnh) + Worker đa tiến trình |
| **Thư viện hỗ trợ** | Pandas (hiển thị DataFrame) |

---

## 📁 Cấu trúc thư mục

```
ML/
├── app/
│   ├── main.py                  # Entry point Streamlit, điều phối UI
│   └── components/
│       ├── live_monitor.py      # Hiển thị live stream
│       └── violation_table.py   # Bảng lịch sử vi phạm
├── src/
│   ├── data/
│   │   ├── schema.sql           # Định nghĩa bảng & ràng buộc SQLite
│   │   └── database_manager.py  # CRUD, cập nhật vehicle_id, reset DB
│   ├── models/
│   │   ├── detection.py         # Load YOLO, tracking, phát hiện biển số
│   │   ├── ocr_engine.py        # Xử lý OCR
│   │   └── ocr_worker_proc.py   # Worker OCR đa tiến trình
│   └── features/
│       ├── processing.py        # Xử lý ảnh biển số & logic ghép cặp
│       └── live_stream.py       # Điều phối pipeline, nguồn video, queue
├── tests/
│   └── integration/
│       └── live_test.py         # Kiểm thử tích hợp pipeline
├── scratch/
│   └── reset_db.py              # Script dọn dẹp DB/evidence
└── data/
    ├── traffic.db               # SQLite database
    └── evidence/
        └── YYYYMMDD/            # Ảnh bằng chứng theo ngày
```

### Phân chia trách nhiệm

| Thành phần | Trách nhiệm |
|------------|-------------|
| `src/models/detection.py` | Nạp model YOLO, phát hiện đối tượng, tracking, phát hiện vùng biển số |
| `src/features/processing.py` | Xử lý ảnh biển số và quy tắc ghép cặp để xác định xe không đội mũ |
| `src/features/live_stream.py` | Điều phối pipeline, nguồn video, OCR queue, ghi nhận vi phạm & evidence |
| `src/models/ocr_engine.py` + `ocr_worker_proc.py` | Đọc biển số và xử lý OCR qua worker/đa tiến trình |
| `src/data/database_manager.py` | Tạo/đọc/cập nhật dữ liệu, liên kết vi phạm với phương tiện, reset |
| `src/data/schema.sql` | Định nghĩa bảng, quan hệ và ràng buộc SQLite |
| `app/main.py` + `app/components/` | Giao diện Streamlit, hiển thị live, lịch sử, thống kê, quản trị |
| `tests/integration/live_test.py` | Kiểm tra pipeline tích hợp ngoài giao diện chính |
| `scratch/reset_db.py` | Dọn dữ liệu test/workspace trong giai đoạn phát triển |

---

## ⚙️ Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd ML
```

### 2. Tạo môi trường ảo & cài đặt thư viện

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# hoặc: venv\Scripts\activate   # Windows
```

Cài đặt các thư viện cần thiết:

```bash
pip install ultralytics opencv-python streamlit pandas
```

> **Lưu ý:** Các thư viện khác như `sqlite3`, `multiprocessing`, `pathlib` đã có sẵn trong Python chuẩn.

### 3. Tải Model Weights

Tải các file trọng số YOLOv8 và đặt đúng vị trí:

```
models/
├── weights_helmet/
│   └── helmet_best.pt
└── weights_plate/
    └── plate_best.pt
```

> Đảm bảo mapping class ID đúng với dữ liệu huấn luyện:  
> `0 = xe máy`, `1 = có mũ`, `2 = không mũ`.

### 4. Khởi tạo Cơ sở dữ liệu

Database sẽ được tạo tự động khi chạy ứng dụng lần đầu (thông qua `DatabaseManager` dựa trên `schema.sql`).  
Nếu cần reset thủ công:

```bash
python scratch/reset_db.py
```

---

## 🚀 Hướng dẫn sử dụng

### Khởi chạy ứng dụng

```bash
streamlit run app/main.py
```

### Thao tác trên giao diện

1. **Chọn nguồn đầu vào**  
   - Webcam (mặc định index `0` nếu để trống)  
   - Upload video  
   - RTSP stream (nhập URL)

2. **Bắt đầu xử lý**  
   Nhấn nút bắt đầu, hệ thống sẽ hiển thị live stream kèm BBox, track ID, biển số và nhãn vi phạm.

3. **Xem lịch sử vi phạm**  
   - Bảng lịch sử hiển thị thông tin vi phạm, biển số, thời gian.  
   - Ảnh evidence dạng thumbnail, click để xem kích thước lớn.

4. **Reset dữ liệu demo**  
   - Vào mục quản trị trong sidebar.  
   - Nhập chính xác `XACNHAN` để xác nhận reset.  
   - Hệ thống xóa dữ liệu vi phạm, xe, session và evidence (giữ lại `violation_rules`).

---

## 📝 Ghi chú

- Hệ thống đã chạy thành công các kiểm tra import, cú pháp, unit test cho logic ghép cặp và thao tác cập nhật DB.  
- Tuy nhiên, để đánh giá chất lượng thực tế, cần test end-to-end với video/camera đa dạng góc độ và đo precision/recall.  
- Cơ chế decoupling giúp **giảm đáng kể nguy cơ mất bản ghi do OCR thất bại**, nhưng không loại bỏ hoàn toàn mọi điểm lỗi trong pipeline (model bỏ sót, mất frame, lỗi DB, queue dừng...).  
- Trước khi merge/demo, nên kiểm tra kỹ model weights, `DB_PATH` thống nhất, whitelist `source_type`, và chạy lại toàn bộ test.

---

**© 2026 – Nhóm phát triển Hệ thống phát hiện vi phạm giao thông**