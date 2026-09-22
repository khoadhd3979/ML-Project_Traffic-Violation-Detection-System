Quy trình Cài đặt Môi trường (Dành cho nhóm) 
1.Clone dự án và chuyển nhánh:
Git.
Kéo mã nguồn mới nhất từ GitHub về máy và chuyển sang đúng nhánh làm việc được phân công, tuyệt đối không commit trực tiếp trên nhánh main:
•	Khoa (ML/CV): git checkout ml-cv.
•	Thành viên 2 (Backend): git checkout Backend-Data.
•	Thành viên 3 (UI): git checkout Application-UI.
2.Tạo và kích hoạt môi trường ảo (venv):Python 3.10 - 3.11.
Mở Terminal hoặc PowerShell tại thư mục gốc của dự án và chạy lệnh tạo không gian độc lập để tránh xung đột hệ thống:
•	Trên Windows:
python -m venv venv
.\venv\Scripts\activate

3.Cài đặt thư viện đồng loạt:
pip install -r requirements.txt
Sau khi chữ (venv) xuất hiện ở đầu dòng lệnh, chạy lệnh sau để cài đặt toàn bộ danh sách thư viện từ file cấu hình:
pip install -r requirements.txt

4.Kiểm tra hệ thống:
Chạy đoạn script sau để xác nhận tất cả các module đã được nạp thành công mà không gặp lỗi DLL:
Bash
python -c "import torch, cv2, ultralytics, paddle, streamlit, pandas; print('Môi trường nhóm đã sẵn sàng!')"

