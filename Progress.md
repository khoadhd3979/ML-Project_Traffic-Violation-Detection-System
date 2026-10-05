**Tình trạng Sprint hiện tại:** Hệ thống đã hoàn thành Sprint 1 (Thiết lập cấu trúc Monorepo và Cơ sở dữ liệu). Tuần 2 này, nhóm sẽ tập trung giải quyết dứt điểm **Sprint 2** (Tích hợp Tracking & OCR) và bước đầu hoàn thành một nửa mục tiêu của **Sprint 3** (Xây dựng Backend API và tháo gỡ dữ liệu giả trên Frontend). Nếu hoàn thành xuất sắc các task của Tuần 2, nhóm sẽ đạt đủ điều kiện để chốt Sprint 2 và sẵn sàng cho đợt review API đầu tiên.

### 📅 KẾ HOẠCH LÀM VIỆC TUẦN 2: KẾT NỐI API & BẬT TRACKING

Mục tiêu cốt lõi của Tuần 2 là đưa các mô hình tĩnh vào luồng phân tích động, xây dựng lớp Backend API bằng FastAPI, và nâng cấp Frontend để hiển thị 100% dữ liệu thật từ cơ sở dữ liệu.

**1. Khoa (AI/ML Lead) — Nhánh `feature/ml-core**`

* **Kích hoạt Object Tracking:** Chuyển đổi từ Object Detection sang Tracking bằng cách cấu hình `model.track(persist=True)` trong `src/models/detection.py` để lấy `track_id` duy nhất cho từng xe, chống trùng lặp dữ liệu vi phạm.
* **Viết Violation Engine:** Xây dựng logic phát hiện vi phạm Không đội mũ và Sai làn/Ngược chiều (dựa trên OpenCV Polygon và Bounding Box) tại `src/features/processing.py`.
* **Tích hợp Database mới:** Cập nhật lại class `ViolationPipeline` để gọi đúng các hàm `create_vehicle()` và `create_violation()` từ `src/data/database_manager.py` khi phát hiện vi phạm.
* **Sản phẩm đầu ra:** Script `tests/integration/live_test.py` chạy mượt mà một video mẫu, tự động nhận diện và đẩy chính xác các dòng dữ liệu vi phạm vào file SQLite.

**2. Phúc (Backend & Data) — Nhánh `feature/backend-api**`

* **Dựng FastAPI Skeleton:** Khởi tạo ứng dụng FastAPI tại `src/api/main.py`. Cấu hình CORS để cho phép Frontend (chạy cổng 8501) gọi API (cổng 8000) mà không bị chặn.
* **Xây dựng Endpoints cốt lõi:** Viết 2 API quan trọng nhất: `GET /api/violations` (truy vấn danh sách lỗi từ View `v_violation_details`) và `GET /api/stats` (đếm số lượng vi phạm phục vụ Dashboard).
* **Tích hợp Pydantic:** Tạo file `src/api/schemas.py` định nghĩa các model Pydantic để chuẩn hóa dữ liệu JSON trả về cho Frontend.
* **Sản phẩm đầu ra:** Swagger UI hoạt động tại `http://localhost:8000/docs`, trả về cục dữ liệu JSON chuẩn xác từ SQLite khi query.

**3. Nguyên (Frontend UI) — Nhánh `feature/frontend-ui**`

* **Xóa bỏ Mock Data:** Loại bỏ hoàn toàn hàm `_mock_initial_data()` trong `app/main.py`.
* **Fetch dữ liệu từ Backend:** Sử dụng thư viện `requests` để gọi tới các endpoint FastAPI mà Phúc vừa viết. Nạp dữ liệu JSON nhận được vào Pandas DataFrame.
* **Cập nhật Component:** Đổ dữ liệu thật vào các KPI Cards trên `dashboard.py` và bảng danh sách trên `violation_table.py`.
* **Sản phẩm đầu ra:** Giao diện Streamlit hiển thị dữ liệu thật đang có trong SQLite. Khi Khoa chạy script ML nhận diện ra lỗi mới, Nguyên chỉ cần F5 trang là thấy số lượng tự nhảy.

---

### 🚀 LỘ TRÌNH CÁC TUẦN TIẾP THEO (TUẦN 3 -> TUẦN 5)

**TUẦN 3: Quản lý Bằng chứng (Evidence) & Hoàn thiện Sprint 3**

* **Mục tiêu:** Xử lý triệt để việc chụp, lưu trữ ảnh vi phạm và hiển thị lên giao diện.
* **Khoa (`feature/ml-core`):** Hoàn thiện PaddleOCR trong `ocr_engine.py`. Viết hàm crop ảnh sắc nét tại thời điểm vi phạm và lưu vào thư mục `data/evidence/` theo chuẩn định dạng.
* **Phúc (`feature/backend-api`):** Dùng `StaticFiles` của FastAPI để serve thư mục `data/evidence/`. Viết thêm API logic cho Chatbot.
* **Nguyên (`feature/frontend-ui`):** Cập nhật `violation_table.py` để khi click vào một hàng, UI sẽ hiển thị ảnh bằng chứng load từ URL của Backend.

**TUẦN 4: Real-time Streaming & Chatbot (Chốt Sprint 4)**

* **Mục tiêu:** Giảm độ trễ, biến hệ thống thành Real-time Live Monitor.
* **Khoa (`feature/ml-core`):** Đưa `ocr_worker_proc.py` vào hoạt động để xử lý biển số chạy ngầm (Background Worker), giúp FPS video không bị tụt.
* **Phúc (`feature/backend-api`):** Thiết lập WebSockets hoặc Server-Sent Events (SSE) để tự động push sự kiện mới lên Frontend ngay khi có vi phạm được ghi vào DB.
* **Nguyên (`feature/frontend-ui`):** Lắng nghe WebSocket trong `live_monitor.py` để cập nhật danh sách "Sự kiện gần nhất" theo thời gian thực mà không cần reload trang. Tích hợp giao diện Chatbot.

**TUẦN 5: Đánh giá Mô hình & MVP Release (Chốt Sprint 5)**

* **Mục tiêu:** Hoàn thiện báo cáo, QA và nghiệm thu đồ án.
* **Khoa (`feature/ml-core`):** Viết các script trong thư mục `notebooks/` để đo đạc mAP50, Precision, Recall, vẽ Confusion Matrix xuất ra biểu đồ cho báo cáo cuối kỳ.
* **Phúc & Nguyên (`feature/backend-api`, `feature/frontend-ui`):** Code freeze (Không thêm tính năng mới). Bắt bug, xử lý các lỗi sập giao diện, dọn dẹp mã nguồn, comment giải thích code và hoàn thiện Swagger Docs. Toàn đội chạy kịch bản quay video demo.
