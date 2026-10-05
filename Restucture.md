**Bước 1: Xóa bỏ dự án cũ**

* Nếu còn code nào đang viết dở chưa push, hãy copy tạm ra Notepad.
* Tắt VSCode, xóa thẳng tay thư mục dự án cũ (`ML-Project_Traffic-Violation-Detection-System`) trên máy tính của mọi người.

**Bước 2: Clone lại dự án mới**

* Mở Terminal (Git Bash hoặc PowerShell) tại thư mục chứa code của bạn và gõ:
```bash
git clone https://github.com/khoadhd3979/ML-Project_Traffic-Violation-Detection-System.git

```


* Di chuyển vào thư mục dự án:
```bash
cd ML-Project_Traffic-Violation-Detection-System

```



**Bước 3: Chạy Script cài đặt môi trường tự động**

* Mở dự án bằng VSCode. Mở Terminal mới trong VSCode (PowerShell).
* Cấp quyền chạy script tạm thời và chạy file setup tự động (nó sẽ tự tạo `.venv` mới tinh và cài đặt đủ thư viện từ `requirements.txt`):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\setup_env.ps1

```



**Bước 4: Trỏ VSCode vào môi trường ảo mới**

* Nhấn tổ hợp phím **`Ctrl + Shift + P`**.
* Gõ và chọn **`Python: Select Interpreter`**.
* Chọn đường dẫn có đuôi `\.venv\Scripts\python.exe` (thường có chữ *Recommended*).
* Nhấn nút thùng rác góc phải để tắt Terminal cũ đi, nhấn `Ctrl + \`` để mở Terminal mới. Đảm bảo thấy chữ **`(.venv)` màu xanh ở đầu dòng.

**Bước 5: Tạo nhánh mới để làm việc (BẮT BUỘC)**

* Không ai được code trực tiếp trên nhánh `main`. Mỗi người hãy tạo một nhánh làm việc mới tinh (có đuôi `v2`) từ cấu trúc mới này:
* **Khoa (ML):** `git checkout -b feature/ml-core-v2`
* **Phúc (Backend):** `git checkout -b feature/backend-api-v2`
* **Nguyên (Frontend):** `git checkout -b feature/frontend-ui-v2`



**Bước 6: Khởi chạy kiểm tra chéo**

* **Để test giao diện (Phần của Nguyên):**
```powershell
streamlit run app/main.py

```


* **Để test luồng ML & Database (Phần của Khoa & Phúc):**
```powershell
python tests/integration/live_test.py