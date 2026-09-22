Mục tiêu cốt lõi của Tuần 1 là hoàn thiện nền tảng (Foundation) bao gồm: Video/Camera truyền vào YOLO Detection, theo dõi đối tượng (Tracking) để lấy Track ID, lưu dữ liệu vào SQLite và hiển thị lên giao diện. Cả 3 thành viên cần nắm vững kiến trúc tổng thể của luồng dữ liệu này. 

1. Khoa (AI/CV) — Nhánh feature/khoa-ml-cv
•	Khảo sát và chuẩn hóa lại các đoạn code cũ từ detection.py, processing.py và ocr_engine.py. 

•	Xử lý Dataset: Xác định các class mục tiêu, chuẩn hóa nhãn, loại bỏ ảnh lỗi và chia tập train/validation/test. 

•	Train mô hình nhận diện phương tiện (Vehicle model) bằng YOLO, đánh giá kết quả trên tập validation và lưu lại mô hình. 

•	Tích hợp tính năng tracking (model.track()) vào pipeline, kiểm tra độ ổn định của Track ID và lưu lại quỹ đạo (trajectory) của xe. 

•	Sản phẩm đầu ra: Mô hình YOLO, file tracking.py, video test và báo cáo đánh giá. Vì file live test chưa hoàn thiện, bạn nên ưu tiên viết một script test độc lập chạy mượt phần Detection + Tracking trước. 



2. Thành viên 2 (Backend) — Nhánh feature/member2-backend
•	Thiết kế cấu trúc cơ sở dữ liệu với các bảng bắt buộc: vehicles, violations, violation_rules, và processing_sessions. 

•	Khởi tạo file schema.sql và xây dựng cơ sở dữ liệu SQLite. 

•	Viết module database_manager.py chứa các hàm CRUD cơ bản (ví dụ: create_vehicle(), get_vehicle(), create_violation()). 

•	Định nghĩa thư mục data/evidence/ và thiết lập quy tắc đặt tên cho các file ảnh bằng chứng. 

•	Sản phẩm đầu ra: File schema.sql, module quản lý DB và một file traffic.db mẫu có sẵn dữ liệu giả để test. 




3. Thành viên 3 (UI) — Nhánh feature/member3-ui-integration
•	Xây dựng bộ khung (skeleton) cho ứng dụng Streamlit bao gồm các trang/tab: Dashboard, Live Monitor, Violation Log, Chatbot và Settings. 

•	Tích hợp tính năng tải video bằng st.file_uploader kèm thanh tiến trình (progress bar) và khu vực hiển thị frame video. 

•	Thiết kế layout tổng quan gồm các khu vực: Số liệu phương tiện/vi phạm, màn hình Video và Bảng lịch sử. 

•	Sản phẩm đầu ra: Giao diện Streamlit có thể chạy độc lập, sẵn sàng nhận dữ liệu từ backend. 

Vào cuối Tuần 1, nhóm sẽ tiến hành gộp (merge) 3 nhánh này vào nhánh chính để kiểm thử toàn bộ luồng tích hợp: từ Video qua Detection, lấy Track ID, lưu DB và hiển thị UI. 

