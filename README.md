# Traffic Violation Detection System (Hệ Thống Phát Hiện Vi Phạm Giao Thông)

## 1. Giới thiệu dự án
Hệ thống nhận diện biển số và phát hiện vi phạm giao thông (vượt đèn đỏ, đi sai làn đường, không đội mũ bảo hiểm) sử dụng các mô hình học sâu hiện đại (YOLOv8, PaddleOCR) kết hợp với giao diện giám sát thời gian thực. Dự án được cấu trúc theo chuẩn Monorepo, giúp dễ dàng mở rộng và tách biệt trách nhiệm giữa các thành phần.

## 2. Kiến trúc 3 phân hệ (Monorepo)
Dự án được chia thành 3 khối lõi độc lập:
- **Project 1 (Backend API - FastAPI):** Đóng vai trò là cầu nối xử lý logic và truy xuất cơ sở dữ liệu (`src/api`, `src/data`).
- **Project 2 (Frontend UI - Streamlit):** Giao diện Web tương tác trực quan cho người dùng, hiển thị dashboard và luồng camera trực tiếp (`app/`).
- **Project 3 (Jupyter/YOLOv8 Models):** Hệ thống Core AI, tiến hành huấn luyện, phân tích và trích xuất đặc trưng (`src/models`, `src/features`, `notebooks/`).

## 3. Phân công nhóm (Team 3 người)
- **AI/ML & Tech Lead:** Đảm nhận khối lượng công việc cốt lõi bao gồm thiết kế kiến trúc Monorepo tổng thể, huấn luyện và tối ưu các mô hình AI (YOLOv8, PaddleOCR), xây dựng luồng xử lý Computer Vision và pipeline phát hiện vi phạm.
- **Backend & Data Engineer:** Xây dựng hệ thống cơ sở dữ liệu SQLite, quản lý luồng dữ liệu (Data Pipeline), lưu trữ bằng chứng và phát triển các endpoint API (FastAPI) để phục vụ tương tác dữ liệu.
- **Frontend UI Developer:** Thiết kế và phát triển toàn bộ ứng dụng người dùng cuối bằng Streamlit, bao gồm Dashboard thống kê, Live Monitor theo dõi camera trực tiếp và giao diện Chatbot tra cứu.

## 4. Lộ trình phát triển (5 Sprints)
- **Sprint 1:** Thu thập dữ liệu, annotation, thiết lập cấu trúc Monorepo và setup cơ sở dữ liệu.
- **Sprint 2:** Huấn luyện mô hình YOLOv8, tích hợp OCR engine cơ bản và worker process.
- **Sprint 3:** Phát triển Backend API và xây dựng Frontend UI (Dashboard, Bảng vi phạm).
- **Sprint 4:** Tích hợp pipeline luồng camera trực tiếp (Live Monitor) với Frontend và Backend.
- **Sprint 5:** Kiểm thử tích hợp (Integration Tests), QA, tối ưu hóa FPS và hoàn thiện tài liệu.

## 5. Cây thư mục hiện hành
```text
.
├── app/                  # Frontend UI (Project 2)
├── data/                 # Thư mục chứa cơ sở dữ liệu và hình ảnh bằng chứng
├── docs/                 # Tài liệu quy chuẩn dự án
├── notebooks/            # Môi trường Data Science (Project 3)
├── src/                  # Core ML và Backend API (Project 1)
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/                # Test suites
├── requirements.txt      # Thư viện phụ thuộc
└── setup_env.ps1         # Script cài đặt môi trường
```

## 6. Hướng dẫn cài đặt và khởi chạy

### Cài đặt môi trường
Mở PowerShell dưới quyền Administrator (nếu cần thiết) và chạy lệnh:
```powershell
.\setup_env.ps1
```

### Khởi chạy hệ thống
Sau khi cài đặt xong, kích hoạt môi trường ảo:
```powershell
.\.venv\Scripts\Activate.ps1
```

- **Khởi chạy Frontend (UI):**
  ```powershell
  streamlit run app/main.py
  ```
- **Chạy Test Pipeline (Kiểm thử Live Camera):**
  ```powershell
  python tests/integration/live_test.py
  ```

