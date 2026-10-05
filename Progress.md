### **Tuần 2: Xây dựng Backend API & Trừ khử Mock Data (Ưu tiên Cao nhất)**

*Mục tiêu: Lấp đầy khoảng trống ở `src/api/main.py` và kết nối giao diện Frontend với dữ liệu thực tế từ cơ sở dữ liệu.*

* **Phúc (Backend):** Khởi tạo ứng dụng FastAPI tại `src/api/main.py`. Viết ngay 2 RESTful endpoints nền tảng: `GET /api/violations` (truy vấn danh sách vi phạm từ view `v_violation_details`) và `GET /api/stats` (lấy thống kê đếm số lượng).
* **Nguyên (Frontend):** Xóa bỏ hoàn toàn hàm `_mock_initial_data()` trong `app/main.py`. Thay thế bằng thư viện `requests` để gọi các API Phúc vừa viết. Đổ dữ liệu thật JSON lên Streamlit Dataframe và các thẻ KPI.
* **Tôi (ML Lead):** Chạy `live_test.py` với một video mẫu để liên tục nhồi dữ liệu thật vào SQLite, tạo môi trường cho Phúc và Nguyên có data để test API và UI.
* **Thỏa mãn Sprint khi:** Giao diện Dashboard và Bảng vi phạm hiển thị 100% dữ liệu lấy từ SQLite thông qua API, không còn bất kỳ dòng code hard-code giả lập nào.

### **Tuần 3: Quản lý Bằng chứng (Evidence) & Đồng bộ Live Monitor**

*Mục tiêu: Hình ảnh vi phạm sinh ra từ ML phải hiển thị sắc nét trên UI, và tính năng Live Monitor bám sát luồng xử lý trung tâm.*

* **Phúc (Backend):** Cấu hình `StaticFiles` trong FastAPI để mở luồng truy cập vào thư mục `data/evidence/`. Viết API endpoint độc lập phục vụ Chatbot (nhận câu hỏi -> query DB -> trả đáp án logic).
* **Nguyên (Frontend):** Chỉnh sửa `app/components/violation_table.py` để khi click vào một dòng vi phạm, UI sẽ fetch hình ảnh từ URL Static của Backend. Điều hướng logic chat trong `chatbot_ui.py` gọi tới API Chatbot thay vì xử lý nội bộ.
* **Tôi (ML Lead):** Tối ưu hóa `src/models/processing.py`. Kiểm tra lại thuật toán crop ảnh bằng chứng để đảm bảo biển số và khuôn mặt/mũ bảo hiểm được lưu rõ nét, độ phân giải tốt trước khi đẩy đường dẫn vào DB.
* **Thỏa mãn Sprint khi:** Luồng end-to-end thông suốt: Camera bắt xe -> ML lưu ảnh vào `data/evidence/` -> Backend serve file -> Frontend bấm vào bảng hiện lên đúng ảnh chụp vi phạm đó.

### **Tuần 4: Xử lý Bất đồng bộ & Streaming Thời gian thực**

*Mục tiêu: Đưa hệ thống lên trạng thái Real-time chuyên nghiệp, xử lý dứt điểm tình trạng nghẽn cổ chai khi chạy video.*

* **Phúc (Backend):** Cấu hình WebSocket hoặc Server-Sent Events (SSE) trên FastAPI. Mục đích là push sự kiện ngay lập tức (real-time) lên UI mỗi khi `DatabaseManager` ghi nhận một vi phạm mới, thay vì bắt UI phải liên tục tải lại (polling).
* **Nguyên (Frontend):** Bắt kết nối WebSocket từ Backend trong trang `live_monitor.py`. Cập nhật mảng "Sự kiện gần nhất" ngay khi luồng stream AI bắt được lỗi mới mà không làm đơ khung hình video đang phát.
* **Tôi (ML Lead):** Đưa toàn bộ tiến trình nhận diện `detection.py` và `ocr_worker_proc.py` vào hàng đợi (Queue) chạy ngầm (Background Worker) độc lập với Main Thread. Chạy stress-test (ép tải) pipeline với video 1080p để đo đạc và tinh chỉnh FPS.
* **Thỏa mãn Sprint khi:** Video chạy mượt mà trên UI, có xe vượt đèn đỏ là khung thông báo tự động nảy số ngay lập tức. Hệ thống chịu tải tốt, không crash khi chạy video dài.

### **Tuần 5: Thực nghiệm Model (Jupyter) & Đóng gói MVP**

*Mục tiêu: Đắp phần lõi Data Science đang thiếu và hoàn thiện báo cáo minh chứng học thuật.*

* **Tôi (ML Lead):** Khởi tạo và code file `notebooks/model_evaluation.ipynb`. Load tập test dataset vào, tính toán các chỉ số mAP50, Precision, Recall, và vẽ Confusion Matrix cho YOLOv8. Trực quan hóa dữ liệu hiệu năng phục vụ báo cáo.
* **Phúc & Nguyên (Backend/Frontend):** Đóng băng tính năng mới (Code Freeze). Bắt lỗi vặt (QA), bắt các case API sập do sai định dạng (Pydantic validation). Hoàn thiện Swagger UI Docs tại `localhost:8000/docs` và tinh chỉnh CSS cho giao diện.
* **Thỏa mãn Sprint khi:** Các biểu đồ đánh giá model được đính kèm vào báo cáo, hệ thống khởi chạy trơn tru qua `setup_env.ps1`, nhóm có sẵn video record kịch bản Demo hoàn chỉnh.