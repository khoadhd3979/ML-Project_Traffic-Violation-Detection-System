### TUẦN 2: Xây dựng Backend API & Logic Vi Phạm (Thực Tế Hóa Dữ Liệu)

**Tình trạng Sprint:** Đang ở Sprint 2. Hoàn thành các công việc dưới đây sẽ đạt 100% mục tiêu Sprint 2 và tạo đà kết nối toàn hệ thống.
**Mục tiêu cốt lõi:** Loại bỏ dữ liệu giả (mock data), triển khai mô hình nhận diện vào luồng xử lý không gian và kết nối Frontend với Backend qua API.

1. **Khoa (AI/ML) — Nhánh `feature/ml-core-v2**`
* Đưa các file trọng số mô hình (`.pt`) vào `src/models/` và kích hoạt tính năng Object Tracking (vd: ByteTrack) bên trong `detection.py`.


* Lập trình logic không gian (Violation Engine) trong `src/features/processing.py` để phát hiện lỗi "Không đội mũ bảo hiểm" và "Sai làn đường".


* Kết nối luồng `tests/integration/live_test.py` để tự động gọi `DatabaseManager` lưu sự kiện vi phạm xuống SQLite.




2. **Phúc (Backend & Data) — Nhánh `feature/backend-api-v2**`
* Khởi tạo ứng dụng FastAPI tại `src/api/main.py`.


* Xây dựng 2 RESTful endpoints nền tảng: `GET /api/violations` (truy vấn danh sách vi phạm từ view cơ sở dữ liệu) và `GET /api/stats` (lấy dữ liệu thống kê tổng quan).


* Đảm bảo API trả về định dạng JSON chuẩn xác thông qua Pydantic schemas.


3. **Nguyên (Frontend UI) — Nhánh `feature/frontend-ui-v2**`
* Gỡ bỏ hoàn toàn hàm `_mock_initial_data()` trong `app/main.py`.
* Sử dụng thư viện `requests` để gọi API từ Backend (`http://localhost:8000/api/...`) và đổ dữ liệu JSON thực tế lên Dashboard và bảng vi phạm.
* Thỏa mãn Sprint 2 khi: F5 trình duyệt, UI hiển thị đúng dữ liệu lỗi mà file `live_test.py` vừa quét được.



---

### TUẦN 3: Tích hợp OCR & Quản Lý Bằng Chứng (Evidence)

**Tình trạng Sprint:** Bước vào Sprint 3. Hoàn thành tuần này sẽ có một luồng End-to-End hoàn chỉnh (từ video đến hình ảnh bằng chứng trên Web).
**Mục tiêu cốt lõi:** Đọc biển số xe chuẩn xác, lưu hình ảnh bằng chứng cắt từ camera và hiển thị chúng trên giao diện Web.

1. **Khoa (AI/ML)**
* Tích hợp và tối ưu PaddleOCR bên trong `src/models/ocr_engine.py` để đọc biển số.


* Bổ sung hàm cắt (crop) ảnh khung hình vi phạm (rõ biển số, rõ lỗi) và lưu vào `data/evidence/` theo đúng chuẩn đặt tên.




2. **Phúc (Backend & Data)**
* Cấu hình `StaticFiles` trên FastAPI để Frontend có thể truy cập ảnh từ thư mục `data/evidence/`.
* Xây dựng endpoint hỗ trợ Chatbot: Nhận câu hỏi từ UI, truy vấn dữ liệu theo biển số/loại lỗi từ DB và trả về câu trả lời.


3. **Nguyên (Frontend UI)**
* Cập nhật `app/components/violation_table.py`: Khi người dùng click vào một hàng, giao diện sẽ tải và hiển thị ảnh bằng chứng từ Static URL của Backend.


* Điều hướng khung nhập liệu trong `chatbot_ui.py` để gọi tới API Chatbot của Phúc thay vì tự xử lý logic nội bộ.



---

### TUẦN 4: Live Monitor & Xử Lý Thời Gian Thực (Bất Đồng Bộ)

**Tình trạng Sprint:** Bước vào Sprint 4. Hoàn thành tuần này hệ thống sẽ đạt tiêu chuẩn Real-time.
**Mục tiêu cốt lõi:** Loại bỏ tình trạng giật lag khi nhận diện, đẩy dữ liệu vi phạm lên UI theo thời gian thực mà không cần tải lại trang.

1. **Khoa (AI/ML)**
* Đẩy tiến trình OCR và ghi Database vào hàng đợi chạy ngầm (Background Worker trong `ocr_worker_proc.py`) để không chặn luồng đọc khung hình chính.
* Tối ưu hóa FPS xử lý video cho module Live Monitor.


2. **Phúc (Backend & Data)**
* Cấu hình WebSocket hoặc Server-Sent Events (SSE) trên FastAPI.
* Bắn sự kiện (Push event) tức thời lên Frontend ngay khi `DatabaseManager` ghi nhận một dòng vi phạm mới.


3. **Nguyên (Frontend UI)**
* Bắt kết nối WebSocket trong `app/components/live_monitor.py`.
* Cập nhật bảng "Sự kiện gần nhất" động trên màn hình camera ngay khi AI bắt được lỗi mới mà không làm đơ video đang phát.



---

### TUẦN 5: Đánh giá Mô Hình, QA & Đóng Gói (MVP Release)

**Tình trạng Sprint:** Bước vào Sprint 5. Hoàn thành tuần này sẽ chốt toàn bộ đồ án và sẵn sàng báo cáo.
**Mục tiêu cốt lõi:** Đắp phần lõi Data Science đang thiếu, dọn dẹp lỗi vặt và xuất file báo cáo 학 thuật.

1. **Khoa (AI/ML)**
* Sử dụng thư mục `notebooks/` để viết script đánh giá model (Model Evaluation).
* Chạy tập test để lấy các chỉ số mAP, Precision, Recall và vẽ Confusion Matrix cho YOLOv8, xuất đồ thị phục vụ báo cáo.




2. **Phúc & Nguyên (Backend/Frontend)**
* Đóng băng mã nguồn (Code Freeze): Không phát triển thêm tính năng mới.
* Bắt lỗi vặt (QA), bắt các trường hợp API sập do sai định dạng (Pydantic validation).
* Dọn dẹp code rác, hoàn thiện Swagger UI Docs tại `localhost:8000/docs` và tinh chỉnh lại CSS.
* Quay video Demo kịch bản toàn hệ thống.
