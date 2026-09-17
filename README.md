# ML-Project_Traffic-Violation-Detection-System# ML-Project_Traffic-Violation-Detection-System

> Hệ thống nhận diện và quản lý vi phạm giao thông bằng Computer Vision, Machine Learning và OCR.
>
> **Thời gian triển khai MVP:** 4 tuần / 8 tuần học  
> **Quy mô nhóm:** 3 thành viên  
> **Mục tiêu:** Xây dựng một prototype có thể nhận video hoặc nguồn camera trực tiếp, phát hiện phương tiện, theo dõi phương tiện theo thời gian, nhận diện một số hành vi vi phạm, đọc biển số, lưu bằng chứng vào SQLite và cung cấp Dashboard + Chatbot tra cứu thông tin vi phạm.

---

## 📑 Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. Mục tiêu dự án](#2-mục-tiêu-dự-án)
- [3. Tính năng chính](#3-tính-năng-chính)
- [4. Phạm vi MVP trong 4 tuần](#4-phạm-vi-mvp-trong-4-tuần)
- [5. Công nghệ sử dụng](#5-công-nghệ-sử-dụng)
- [6. Các kiến thức được ứng dụng từ môn Học Máy](#6-các-kiến-thức-được-ứng-dụng-từ-môn-học-máy)
- [7. Kiến trúc tổng thể](#7-kiến-trúc-tổng-thể)
- [8. Cấu trúc thư mục](#8-cấu-trúc-thư-mục)
- [9. Phân công thành viên và Git Branch](#9-phân-công-thành-viên-và-git-branch)
- [10. Chi tiết module](#10-chi-tiết-module)
- [11. Cài đặt](#11-cài-đặt)
- [12. Chạy chương trình](#12-chạy-chương-trình)
- [13. Lộ trình phát triển 4 tuần](#13-lộ-trình-phát-triển-4-tuần)
- [14. Pipeline xử lý](#14-pipeline-xử-lý)
- [15. Thiết kế Database](#15-thiết-kế-database)
- [16. Quy tắc phát hiện vi phạm](#16-quy-tắc-phát-hiện-vi-phạm)
- [17. Chatbot tra cứu](#17-chatbot-tra-cứu)
- [18. Kiểm thử và đánh giá](#18-kiểm-thử-và-đánh-giá)
- [19. Git Workflow và hướng dẫn đóng góp](#19-git-workflow-và-hướng-dẫn-đóng-góp)
- [20. Xử lý sự cố](#20-xử-lý-sự-cố)
- [21. Giới hạn của dự án](#21-giới-hạn-của-dự-án)
- [22. Tài liệu tham khảo](#22-tài-liệu-tham-khảo)

---

# 1. Giới thiệu

**ML-Project_Traffic-Violation-Detection-System** là dự án môn **Học Máy (Machine Learning)** nhằm xây dựng một hệ thống prototype phát hiện vi phạm giao thông từ video hoặc camera.

Dự án được phát triển dựa trên nền tảng của một hệ thống nhận diện biển số xe (LPR) đã có sẵn. Thay vì chỉ tập trung vào việc:

```text
Phát hiện biển số
        ↓
OCR
        ↓
Đọc biển số
```

hệ thống mới mở rộng thành:

```text
Video / Camera
      ↓
Vehicle Detection
      ↓
Multi-Object Tracking
      ↓
Violation Analysis
      ↓
License Plate Detection + OCR
      ↓
Evidence Capture
      ↓
SQLite Database
      ↓
Streamlit Dashboard
      ↓
Chatbot / API
```

Hệ thống hướng tới việc nhận diện một số lỗi giao thông trong môi trường có camera tương đối cố định, sau đó tạo một **violation event** chứa:

- Mã phương tiện/tracking ID
- Loại phương tiện
- Biển số xe nếu đọc được
- Loại vi phạm
- Thời gian phát hiện
- Confidence của mô hình liên quan
- Ảnh bằng chứng
- Video/frame tham chiếu nếu cần
- Trạng thái xử lý

> **Lưu ý:** Đây là dự án học tập và prototype Machine Learning/Computer Vision, không phải hệ thống xử phạt giao thông thực tế. Kết quả nhận diện cần được xem là kết quả hỗ trợ, không thay thế quyết định của cơ quan có thẩm quyền.

---

# 2. Mục tiêu dự án

## 2.1. Mục tiêu chính

Xây dựng một hệ thống có khả năng:

1. Nhận video từ file.
2. Nhận nguồn camera trực tiếp.
3. Phát hiện phương tiện bằng YOLO.
4. Tracking từng phương tiện qua nhiều frame.
5. Xác định các hành vi vi phạm dựa trên:
   - Object detection.
   - Tracking.
   - Vùng ROI/polygon.
   - Stop line.
   - Hướng di chuyển.
   - Trạng thái đèn giao thông.
6. Nhận diện biển số xe.
7. OCR biển số.
8. Chụp ảnh bằng chứng khi phát hiện vi phạm.
9. Lưu sự kiện vào SQLite.
10. Hiển thị kết quả trên Streamlit.
11. Cho phép tra cứu thông tin vi phạm bằng chatbot/API.
12. Đánh giá hiệu quả hệ thống bằng các chỉ số phù hợp.

## 2.2. Mục tiêu Machine Learning

Dự án phải thể hiện rõ kiến thức của môn Học Máy thông qua:

- Dataset preparation.
- Data labeling.
- Train/validation/test split.
- Supervised learning.
- Transfer learning.
- Object detection.
- Model training.
- Hyperparameter tuning.
- Loss monitoring.
- Precision/Recall.
- mAP.
- Confusion analysis.
- Model inference.
- Error analysis.
- Đánh giá ảnh hưởng của chất lượng dữ liệu và điều kiện môi trường.

---

# 3. Tính năng chính

## 3.1. Phát hiện phương tiện

Phát hiện các loại phương tiện mục tiêu, ví dụ:

- `car`
- `motorcycle`
- `bus`
- `truck`

Tùy dataset thực tế, nhóm có thể giảm số class để phù hợp với thời gian 4 tuần.

---

## 3.2. Multi-Object Tracking

Mỗi phương tiện được gán một `track_id` để theo dõi xuyên suốt video.

Ví dụ:

```text
Frame 001:
Car      → ID 7
Motorbike → ID 12

Frame 020:
Car      → ID 7
Motorbike → ID 12

Frame 050:
Car      → ID 7
Motorbike → ID 12
```

Tracking giúp hệ thống biết rằng các bounding box ở các frame khác nhau thuộc cùng một phương tiện.

Dự án có thể sử dụng tracker được tích hợp trong Ultralytics thay vì tự xây dựng thuật toán tracking từ đầu.

---

## 3.3. Nhận diện không đội nón bảo hiểm

Pipeline:

```text
Motorcycle Detection
        ↓
Rider / Helmet Analysis
        ↓
Helmet / No Helmet
        ↓
Violation Event
```

Kết quả:

```text
Violation Type: NO_HELMET
Vehicle Type: MOTORCYCLE
Track ID: 15
Confidence: 0.91
```

---

## 3.4. Nhận diện đi ngược chiều

Không nhất thiết phải huấn luyện một model riêng.

Hệ thống sử dụng:

```text
Tracking
   +
Movement Vector
   +
Allowed Direction
```

Ví dụ:

```text
Hướng hợp lệ:
LEFT → RIGHT

Phương tiện:
RIGHT → LEFT

=> WRONG_WAY
```

---

## 3.5. Nhận diện sai làn đường

Khu vực giao thông được mô tả bằng polygon/ROI.

Ví dụ:

```text
+----------------------------------------+
|                CAMERA                  |
|                                        |
|  ----------- Lane 1 ----------------   |
|                                        |
|  ----------- Lane 2 ----------------   |
|                                        |
|  ----------- Lane 3 ----------------   |
|                                        |
+----------------------------------------+
```

Trajectory của mỗi `track_id` được kiểm tra với vùng làn đường.

MVP có thể sử dụng cấu hình thủ công cho từng camera:

```python
LANES = {
    "lane_1": [...],
    "lane_2": [...],
    "lane_3": [...]
}
```

---

## 3.6. Nhận diện vượt đèn đỏ

Hệ thống sử dụng ba thành phần:

```text
Traffic Light State
       +
Vehicle Tracking
       +
Stop Line
```

Logic cơ bản:

```text
Traffic Light = RED
        ↓
Vehicle crosses Stop Line
        ↓
Vehicle not authorized to cross
        ↓
RED_LIGHT_VIOLATION
```

MVP có thể sử dụng trạng thái đèn được xác định bằng ROI hoặc model riêng, tùy khả năng triển khai của nhóm.

---

## 3.7. Nhận diện biển số

Hệ thống sử dụng pipeline LPR đã có/được mở rộng:

```text
Vehicle
   ↓
Plate Detection
   ↓
Plate Crop
   ↓
Image Processing
   ↓
OCR
   ↓
Post-processing
   ↓
Plate Number
```

---

## 3.8. Chụp bằng chứng

Khi một violation event được xác nhận:

```text
frame_2026xxxx_xxxxxx.jpg
```

được lưu vào:

```text
data/evidence/
```

Tên file nên chứa các thông tin chính:

```text
{timestamp}_{track_id}_{violation_type}.jpg
```

Ví dụ:

```text
20260917_143520_15_NO_HELMET.jpg
```

---

## 3.9. Dashboard

Dashboard Streamlit dự kiến có:

### Dashboard tổng quan

- Số phương tiện đang được tracking.
- Số vi phạm trong phiên.
- Số vi phạm theo loại.
- Số biển số đọc được.
- Số sự kiện cần kiểm tra.

### Live Monitor

- Video/frame hiện tại.
- Bounding box.
- Track ID.
- Violation label.
- Biển số nếu có.

### Violation Log

- Thời gian.
- Biển số.
- Loại xe.
- Loại vi phạm.
- Confidence.
- Đường dẫn bằng chứng.

---

## 3.10. Chatbot

Chatbot cung cấp giao diện tra cứu:

- Vi phạm là gì?
- Hệ thống phát hiện vi phạm đó như thế nào?
- Thông tin xử lý/tra cứu theo dữ liệu đã cấu hình.
- Tra cứu theo loại vi phạm.
- Tra cứu theo loại phương tiện.
- Giải thích một violation record.

Chatbot **không nên tự suy đoán mức phạt**. Dữ liệu pháp luật nên được quản lý từ nguồn được nhóm xác định, có ngày hiệu lực và nguồn tham chiếu.

---

# 4. Phạm vi MVP trong 4 tuần

Để đảm bảo khả năng hoàn thành, MVP ưu tiên:

### Bắt buộc

- [x] Vehicle detection.
- [x] Multi-object tracking.
- [x] License Plate Detection/OCR.
- [x] NO_HELMET.
- [x] WRONG_WAY.
- [x] RED_LIGHT.
- [x] SQLite.
- [x] Evidence image.
- [x] Streamlit Dashboard.
- [x] Video input.
- [x] Camera input ở mức prototype.
- [x] Chatbot/API cơ bản.
- [x] Test case.
- [x] Báo cáo Machine Learning.

### Có thể làm thêm nếu còn thời gian

- [ ] WRONG_LANE nâng cao.
- [ ] CHE_COVERED_PLATE.
- [ ] Traffic light detection bằng model riêng.
- [ ] Nhiều camera.
- [ ] RTSP hoàn chỉnh.
- [ ] Re-identification nâng cao.
- [ ] Xuất báo cáo CSV/PDF.
- [ ] Deploy online/server.

> **Quy tắc quản lý scope:** Chỉ thêm tính năng mới khi toàn bộ chức năng MVP đã chạy được.

---

# 5. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ chính |
| PyTorch | Deep Learning framework |
| Ultralytics YOLO | Object Detection + Tracking |
| OpenCV | Xử lý ảnh/video |
| PaddleOCR | OCR biển số |
| PaddlePaddle | Backend cho PaddleOCR |
| Streamlit | Dashboard |
| NumPy | Xử lý mảng/ảnh |
| SQLite3 | Database cục bộ |
| OpenVINO | Tối ưu inference khi phù hợp |
| Plotly | Trực quan hóa |
| Git/GitHub | Version Control |
| Roboflow | Dataset/annotation nếu sử dụng |
| API LLM | Chatbot |

---

# 6. Các kiến thức được ứng dụng từ môn Học Máy

## 6.1. Supervised Learning

Bài toán nhận diện phương tiện/helmet/plate dựa trên dữ liệu đã gán nhãn.

Ví dụ:

```text
Image
  ↓
Label:
  car
  motorcycle
  helmet
  no_helmet
  license_plate
```

Mô hình học ánh xạ:

```text
Input Image → Predicted Class + Bounding Box
```

---

## 6.2. Dataset

Nhóm cần quản lý:

```text
Raw Data
   ↓
Cleaning
   ↓
Annotation
   ↓
Train / Validation / Test
```

Dataset phải có sự đa dạng:

- Ngày/đêm.
- Sáng/tối.
- Xa/gần.
- Góc nghiêng.
- Xe máy/ô tô.
- Các tình huống có che khuất.
- Nhiều loại background.

---

## 6.3. Data Augmentation

Có thể áp dụng:

- Resize.
- Horizontal flip nếu phù hợp.
- Brightness adjustment.
- Contrast adjustment.
- Blur.
- Noise.
- Rotation nhỏ.
- Crop.

Mục tiêu:

```text
Tăng độ đa dạng dữ liệu
        ↓
Giảm overfitting
        ↓
Tăng khả năng generalization
```

Không áp dụng augmentation nếu nó làm thay đổi ngữ nghĩa của nhãn.

---

## 6.4. Transfer Learning

Thay vì train một model từ đầu:

```text
Pretrained Model
      ↓
Fine-tuning
      ↓
Traffic Dataset
      ↓
Traffic Detection Model
```

Đây là hướng chính để giảm thời gian training trong phạm vi môn học.

---

## 6.5. Object Detection

YOLO được sử dụng để dự đoán:

```text
Class
Bounding Box
Confidence
```

Ví dụ:

```text
motorcycle
x1 = 100
y1 = 200
x2 = 400
y2 = 600
confidence = 0.93
```

---

## 6.6. Multi-Object Tracking

Tracking không đơn thuần trả lời:

> Có xe không?

Mà trả lời:

> Đây có phải cùng chiếc xe ở frame trước hay không?

Kết quả:

```text
Detection + Tracking
        ↓
track_id
        ↓
Trajectory
        ↓
Behavior Analysis
```

Đây là cơ sở để xử lý WRONG_WAY, WRONG_LANE và RED_LIGHT.

---

## 6.7. Classification / Detection cho Helmet

Bài toán helmet có thể được triển khai theo hai hướng:

### Hướng A

Một detector nhiều class:

```text
helmet
no_helmet
motorcycle
person
```

### Hướng B

Tách thành:

```text
Motorcycle/Person Detection
          ↓
Head/Rider ROI
          ↓
Helmet Classifier/Detector
```

Trong phạm vi 4 tuần, nhóm ưu tiên phương án đơn giản hơn và phù hợp với dataset thực tế.

---

## 6.8. Model Evaluation

Các chỉ số cần quan tâm:

```text
Precision
Recall
F1-score
mAP@50
mAP@50:95
Confidence threshold
Inference speed / FPS
```

### Precision

Trong số các dự đoán là vi phạm, có bao nhiêu dự đoán đúng.

### Recall

Trong số các vi phạm thực tế, hệ thống phát hiện được bao nhiêu.

### F1-score

Cân bằng giữa Precision và Recall.

### mAP

Dùng để đánh giá hiệu năng object detection.

---

## 6.9. Overfitting và Generalization

Một model có thể có kết quả tốt trên training/validation nhưng kém khi gặp:

- Camera khác.
- Góc nhìn khác.
- Ánh sáng khác.
- Xe khác.
- Khoảng cách khác.

Do đó nhóm phải test trên dữ liệu mà model chưa nhìn thấy.

---

## 6.10. Error Analysis

Sau khi test, không chỉ ghi:

```text
Accuracy = ...
```

mà cần phân tích:

```text
False Positive
False Negative
Missed detection
Wrong class
Wrong tracking ID
OCR error
Violation logic error
```

Điều này giúp chứng minh nhóm thực sự hiểu Machine Learning thay vì chỉ gọi API/model.

---

## 6.11. Inference và Deployment

Sau training:

```text
Trained Model
      ↓
Export / Load
      ↓
Inference
      ↓
Prediction
      ↓
Application
```

OpenVINO có thể được dùng để tối ưu inference trong trường hợp phù hợp với môi trường triển khai.

---

# 7. Kiến trúc tổng thể

```text
                         ┌─────────────────────┐
                         │    VIDEO INPUT      │
                         │                     │
                         │ 1. Uploaded Video   │
                         │ 2. Live Camera      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   FRAME CAPTURE     │
                         │       OpenCV        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  VEHICLE DETECTION  │
                         │        YOLO         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      TRACKING       │
                         │  track_id per car   │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                     ▼              ▼              ▼
               Helmet Check     Lane Check     Direction Check
                     │              │              │
                     └──────────────┼──────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   RED LIGHT LOGIC   │
                         │ Stop Line + Signal  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   PLATE DETECTION   │
                         │       + OCR         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ VIOLATION ENGINE    │
                         │ Create Event        │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
                Evidence        SQLite DB        Dashboard
                 Image                             Streamlit
                                                     │
                                                     ▼
                                                   Chatbot
```

---

# 8. Cấu trúc thư mục

```text
ML-Project_Traffic-Violation-Detection-System/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── config.py
│   ├── camera_config.yaml
│   └── violation_config.yaml
│
├── models/
│   ├── vehicle/
│   │   └── best.pt
│   ├── helmet/
│   │   └── best.pt
│   └── plate/
│       └── best.pt
│
├── modules/
│   ├── detection.py
│   ├── tracking.py
│   ├── violation_engine.py
│   ├── lane_detector.py
│   ├── red_light_detector.py
│   ├── plate_detector.py
│   ├── ocr_engine.py
│   ├── image_processing.py
│   ├── evidence_manager.py
│   ├── database_manager.py
│   ├── chatbot.py
│   └── utils.py
│
├── ui/
│   ├── dashboard.py
│   ├── live_monitor.py
│   ├── violation_table.py
│   └── chatbot_ui.py
│
├── database/
│   ├── traffic.db
│   └── schema.sql
│
├── data/
│   ├── videos/
│   ├── test_images/
│   └── evidence/
│
├── datasets/
│   ├── vehicle/
│   ├── helmet/
│   └── plate/
│
├── scripts/
│   ├── train_vehicle.py
│   ├── train_helmet.py
│   ├── evaluate_model.py
│   └── test_pipeline.py
│
└── docs/
    ├── architecture.md
    ├── ml_report.md
    ├── database.md
    ├── testing.md
    └── api.md
```

---

# 9. Phân công thành viên và Git Branch

Nhóm sử dụng một branch riêng cho từng thành viên.

## Branch 1 — Tôi

```text
feature/khoa-ml-cv
```

### Phụ trách

- Dataset.
- Annotation.
- Training YOLO.
- Model evaluation.
- Vehicle detection.
- Helmet detection.
- Plate detection.
- OCR integration.
- Tracking integration ở tầng Computer Vision.
- Phân tích kết quả model.

### Module chính

```text
modules/detection.py
modules/tracking.py
modules/plate_detector.py
modules/ocr_engine.py
scripts/train_*.py
scripts/evaluate_model.py
models/
datasets/
```

---

## Branch 2 — Người 2

```text
feature/member2-backend
```

### Phụ trách

- Database.
- SQL.
- SQLite schema.
- Violation event.
- Evidence storage.
- Business logic.
- Violation rule configuration.
- Chatbot API integration.
- Backend validation.

### Module chính

```text
modules/database_manager.py
modules/violation_engine.py
modules/evidence_manager.py
modules/chatbot.py
database/
config/violation_config.yaml
```

---

## Branch 3 — Người 3

```text
feature/member3-ui-integration
```

### Phụ trách

- Streamlit.
- Dashboard.
- Upload video.
- Live monitor.
- Violation table.
- Evidence viewer.
- Visualization.
- Integration giữa backend và UI.
- Testing UI.

### Module chính

```text
app.py
ui/dashboard.py
ui/live_monitor.py
ui/violation_table.py
ui/chatbot_ui.py
```

---

# 10. Chi tiết module

## `app.py`

Điểm khởi động của ứng dụng.

Không nên chứa toàn bộ logic AI.

Nhiệm vụ:

```text
Load config
    ↓
Initialize modules
    ↓
Select input mode
    ↓
Run pipeline
    ↓
Display result
```

---

## `detection.py`

Chịu trách nhiệm:

- Load YOLO.
- Inference.
- Bounding boxes.
- Confidence.
- Class IDs.

Ví dụ kết quả:

```python
{
    "class_name": "motorcycle",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.94
}
```

---

## `tracking.py`

Chịu trách nhiệm:

- `track_id`.
- Persistent tracking.
- Track history.
- Center points.
- Trajectory.

Output:

```python
{
    "track_id": 15,
    "bbox": [...],
    "center": [x, y],
    "vehicle_type": "motorcycle"
}
```

---

## `violation_engine.py`

Đây là module logic trung tâm.

Input:

```text
Tracked Vehicle
+
Detection results
+
Camera configuration
+
Traffic state
```

Output:

```text
Violation Event
```

Ví dụ:

```python
{
    "track_id": 15,
    "violation_type": "WRONG_WAY",
    "confidence": 0.93
}
```

Module này phải tránh đặt code train model vào bên trong.

---

## `lane_detector.py`

Phụ trách:

- Lane polygons.
- Point-in-polygon.
- Lane assignment.
- Lane transition.

---

## `red_light_detector.py`

Phụ trách:

- Traffic light state.
- Stop line.
- Vehicle crossing.
- Red-light rule.

---

## `plate_detector.py`

Phụ trách:

- Plate bounding box.
- Crop plate.

---

## `ocr_engine.py`

Pipeline:

```text
Plate Crop
    ↓
Resize
    ↓
Contrast Enhancement
    ↓
OCR
    ↓
Text Cleanup
    ↓
Regex / Format Validation
```

---

## `evidence_manager.py`

Phụ trách:

- Chụp frame.
- Crop vùng cần thiết.
- Lưu evidence.
- Generate filename.

---

## `database_manager.py`

Phụ trách:

- Connect SQLite.
- Insert vehicle.
- Insert violation.
- Query history.
- Update status.
- Query statistics.

SQLite phù hợp cho prototype này vì là database dạng file và không yêu cầu một server database riêng.

---

## `chatbot.py`

Phụ trách:

```text
User question
     ↓
Intent / Entity extraction
     ↓
Database / knowledge lookup
     ↓
LLM API
     ↓
Answer
```

Chatbot không nên tự phát sinh dữ liệu pháp luật nếu không có dữ liệu nguồn tương ứng.

---

## `ui/`

Chỉ hiển thị và nhận input người dùng.

UI không nên chứa:

- YOLO training.
- SQL query phức tạp.
- OCR algorithm.
- Violation algorithm.

---

# 11. Cài đặt

## 11.1. Yêu cầu

Khuyến nghị:

```text
Python 3.10 - 3.11
Git
VS Code
NVIDIA GPU (nếu có, cho inference/training nhanh hơn)
```

`venv` được khuyến nghị để cô lập môi trường Python của project.

---

## 11.2. Tạo virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 11.3. Nếu các thư viện đã được cài đặt

Các thư viện chính:

```text
opencv-python
ultralytics
paddlepaddle
paddleocr
streamlit
numpy
torch
openvino
plotly
```

Không cần cài SQLite server riêng cho dự án.

Kiểm tra SQLite:

```bash
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

Kiểm tra các thư viện:

```bash
python -c "import cv2, torch, numpy, streamlit, ultralytics, openvino; print('Core packages OK')"
```

Kiểm tra GPU của PyTorch:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 11.4. Nếu chưa cài môi trường

```bash
pip install -r requirements.txt
```

Sau khi cài, kiểm tra:

```bash
pip list
```

### Lưu ý PyTorch

Cài PyTorch nên chọn đúng build tương ứng với:

- Operating System
- Python version
- CPU/GPU
- CUDA version nếu dùng NVIDIA GPU

Không nên copy một lệnh CUDA bất kỳ từ máy khác.

---

# 12. Chạy chương trình

Hệ thống thiết kế 2 chế độ chính.

---

## Mode 1 — Upload Video

### Mục đích

Dùng video đã quay sẵn để:

- kiểm thử model;
- kiểm thử tracking;
- kiểm thử violation logic;
- đánh giá kết quả;
- demo ổn định.

### Luồng

```text
Upload video
      ↓
Read frames
      ↓
Vehicle Detection
      ↓
Tracking
      ↓
Violation Detection
      ↓
Plate OCR
      ↓
Evidence
      ↓
SQLite
      ↓
Dashboard
```

### Chạy

```bash
streamlit run app.py
```

Sau đó:

```text
Upload Video
    ↓
Select violation types
    ↓
Run Detection
    ↓
View Result
```

---

## Mode 2 — Live Camera

### Mục đích

Nhận diện trực tiếp từ camera.

Nguồn camera có thể là:

```text
Camera USB
Camera laptop
Camera điện thoại
Camera điện thoại + gimbal
RTSP/HTTP stream
Virtual Camera
```

### Với camera điện thoại + gimbal

Về kiến trúc:

```text
Phone Camera
     ↓
Video Stream
     ↓
Computer
     ↓
OpenCV / YOLO
     ↓
Violation Detection
```

Trong MVP, nên bắt đầu bằng camera nguồn mà máy tính đọc được ổn định. Sau khi pipeline ổn định mới tích hợp stream từ điện thoại.

> `st.camera_input()` của Streamlit phù hợp với việc nhận ảnh từ camera; đối với live video liên tục, pipeline OpenCV/stream riêng thường phù hợp hơn.

---

# 13. Lộ trình phát triển trong 4 tuần

## Tổng quan

```text
WEEK 1
Foundation + Architecture + Tracking + Database

WEEK 2
Violation Detection + Model Development

WEEK 3
Integration + Dashboard + Evidence + OCR

WEEK 4
Chatbot + Testing + Optimization + Demo + Report
```

---

# WEEK 1 — FOUNDATION

## Mục tiêu

Cuối tuần 1 phải có:

```text
Video / Camera
      ↓
YOLO Detection
      ↓
Tracking
      ↓
Track ID
      ↓
SQLite
```

và cả 3 thành viên hiểu được kiến trúc tổng thể.

---

## Tôi — `feature/khoa-ml-cv`

### Task 1: Khảo sát code cũ

- Đọc `detection.py`.
- Đọc `processing.py`.
- Đọc `ocr_engine.py`.
- Xác định model hiện tại.
- Xác định format output.
- Xác định pipeline frame hiện tại.
- Ghi lại các điểm có thể tái sử dụng.

### Task 2: Chuẩn hóa dataset

- Xác định class.
- Chuẩn hóa label.
- Kiểm tra ảnh lỗi.
- Loại ảnh trùng.
- Kiểm tra imbalance.
- Tách train/validation/test.
- Tạo tài liệu dataset.

### Task 3: Vehicle model

- Load YOLO.
- Test detection.
- Fine-tune nếu cần.
- Đánh giá trên validation.
- Lưu model.

### Task 4: Tracking

- Tích hợp `model.track()`.
- Test track ID.
- Kiểm tra ID switch.
- Lưu trajectory.
- Kiểm tra nhiều phương tiện cùng frame.

### Deliverable

```text
vehicle model
tracking.py
test video
evaluation note
```

---

## Người 2 — `feature/member2-backend`

### Task 1: Thiết kế database

Tạo:

```text
vehicles
violations
violation_rules
processing_sessions
```

### Task 2: SQLite

- Create DB.
- Create tables.
- Insert.
- Select.
- Update.
- Delete/test cleanup.

### Task 3: Database Manager

Viết:

```python
create_vehicle()
get_vehicle()
create_violation()
get_violations()
get_violation_statistics()
```

### Task 4: Evidence

Định nghĩa:

```text
data/evidence/
```

và naming convention.

### Deliverable

```text
database/schema.sql
database_manager.py
sample traffic.db
```

---

## Người 3 — `feature/member3-ui-integration`

### Task 1: Streamlit skeleton

Tạo:

```text
Dashboard
Live Monitor
Violation Log
Chatbot
Settings
```

### Task 2: Upload video

- `st.file_uploader`.
- Hiển thị video.
- Hiển thị frame.
- Progress bar.

### Task 3: Layout

Thiết kế dashboard:

```text
+-------------------------------------+
| Traffic Violation Detection System  |
+-------------------------------------+
| Vehicles | Violations | Plates      |
+-------------------------------------+
|             Video                   |
|                                     |
+-------------------------------------+
| Violation Table                     |
+-------------------------------------+
```

### Deliverable

Streamlit UI chạy được độc lập.

---

## WEEK 1 — Integration

Ba người merge:

```text
feature/khoa-ml-cv
feature/member2-backend
feature/member3-ui-integration
        ↓
main / develop
```

Test:

```text
Video
 → Detection
 → Track ID
 → Save DB
 → Show UI
```

---

# WEEK 2 — VIOLATION DETECTION

## Mục tiêu

Hoàn thành:

```text
NO_HELMET
WRONG_WAY
RED_LIGHT
```

---

## Tôi

### NO_HELMET

- Dataset.
- Label.
- Train/finetune.
- Evaluate.
- Chọn confidence threshold.
- Test false positive.
- Test false negative.

### WRONG_WAY

- Lưu track history.
- Tính displacement.
- Xác định movement vector.
- Cấu hình allowed direction.
- Kiểm tra nhiễu tracking.
- Debounce để tránh tạo nhiều event cho một xe.

---

## Người 2

### Violation Engine

Tạo:

```python
check_no_helmet()
check_wrong_way()
check_red_light()
```

### Event deduplication

Không được tạo:

```text
15 NO_HELMET
15 NO_HELMET
15 NO_HELMET
15 NO_HELMET
```

cho cùng một lần vi phạm.

Thay vào đó:

```text
track_id = 15
violation = NO_HELMET
event_id = 1001
```

### State

```text
DETECTED
CONFIRMED
REVIEWED
IGNORED
```

---

## Người 3

### Violation UI

Hiển thị:

```text
🔴 NO HELMET
🟠 WRONG WAY
🔴 RED LIGHT
```

Có filter:

```text
All
No Helmet
Wrong Way
Red Light
```

Click event:

```text
→ Timestamp
→ Plate
→ Track ID
→ Evidence
```

---

# WEEK 3 — SYSTEM INTEGRATION

## Mục tiêu

Cuối tuần:

```text
Video/Camera
      ↓
Detection
      ↓
Tracking
      ↓
Violation
      ↓
Plate OCR
      ↓
Evidence
      ↓
SQLite
      ↓
Dashboard
```

---

## Tôi

### Plate pipeline

- Plate detection.
- Crop.
- Resize.
- CLAHE/processing.
- PaddleOCR.
- Text normalization.
- Regex validation.

### Voting

Một biển số nên được xác nhận qua nhiều frame thay vì tin một OCR duy nhất.

Ví dụ:

```text
Frame 100 → 51G12345
Frame 101 → 51G12345
Frame 102 → 51G12345

=> CONFIRMED
```

---

## Người 2

### Evidence manager

Khi violation confirmed:

```text
Capture frame
      ↓
Crop relevant area
      ↓
Save
      ↓
Store path in SQLite
```

### Database query

Tạo API nội bộ:

```python
get_recent_violations()
get_violation_by_id()
get_violations_by_plate()
get_violations_by_type()
get_statistics()
```

### Configuration

Ví dụ:

```yaml
camera_id: CAM_01

allowed_direction: left_to_right

stop_line:
  - [100, 400]
  - [800, 400]
```

---

## Người 3

### Dashboard hoàn chỉnh

Trang 1:

```text
Overview
```

Trang 2:

```text
Live Monitor
```

Trang 3:

```text
Violation History
```

Trang 4:

```text
Analytics
```

Trang 5:

```text
Chatbot
```

Plotly:

- Vi phạm theo loại.
- Vi phạm theo thời gian.
- Phương tiện theo loại.

---

# WEEK 4 — CHATBOT + TESTING + DEMO

## Mục tiêu

Đóng gói MVP.

---

## Tôi

### Model optimization

- Test confidence threshold.
- Test image size.
- Test FPS.
- Test CPU.
- Test GPU.
- Test video 720p.
- Test video 1080p.
- Record inference time.

### Error analysis

Tạo bảng:

```text
Case
Expected
Predicted
Correct?
Cause
Solution
```

---

## Người 2

### Chatbot backend

Pipeline:

```text
Question
  ↓
Identify violation type
  ↓
Query local rules/data
  ↓
Build context
  ↓
Call API
  ↓
Return answer
```

Ví dụ dữ liệu chatbot:

```text
Violation:
WRONG_WAY

Vehicle:
MOTORCYCLE

Source:
<configured legal source>

Effective date:
<date>
```

### Quy tắc

- Không hard-code thông tin pháp luật trong prompt.
- Không để chatbot tự tạo mức phạt.
- Hiển thị nguồn dữ liệu.
- Hiển thị ngày cập nhật.
- Có fallback nếu không tìm thấy dữ liệu.

---

## Người 3

### Final UI

- Loading state.
- Error state.
- Empty state.
- Video controls.
- Filter.
- Search.
- Evidence viewer.
- Chatbot UI.

### Demo flow

```text
1. Upload video
2. Start detection
3. Show tracking
4. Detect violation
5. Show plate
6. Save evidence
7. Show DB record
8. Ask chatbot
```

---

# WEEK 4 — FINAL TESTING

## Test Group A: Detection

```text
Vehicle detected?
Plate detected?
Helmet detected?
```

## Test Group B: Tracking

```text
Same vehicle keeps same ID?
Two vehicles collide in image?
Occlusion?
Vehicle enters/exits frame?
```

## Test Group C: Violation

```text
Wrong way?
No helmet?
Red light?
Repeated event?
False positive?
```

## Test Group D: OCR

```text
Day
Night
Far
Near
Tilted
Blurred
```

## Test Group E: System

```text
Database write?
Dashboard update?
Evidence saved?
Chatbot response?
```

---

# 14. Pipeline xử lý

```text
┌──────────────────────────┐
│ Video / Camera Input     │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ OpenCV Frame Capture     │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ YOLO Vehicle Detection   │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Multi-Object Tracking    │
│ Track ID + Trajectory    │
└─────────────┬────────────┘
              ▼
     ┌────────┴─────────┐
     │                  │
     ▼                  ▼
Helmet Analysis    Movement Analysis
     │                  │
     ▼                  ▼
NO_HELMET          WRONG_WAY
                         │
                         ▼
                  Lane / ROI Analysis
                         │
                         ▼
                    WRONG_LANE
                         │
                         ▼
                Red Light / Stop Line
                         │
                         ▼
                    RED_LIGHT
     │                  │
     └────────┬─────────┘
              ▼
┌──────────────────────────┐
│ Plate Detection          │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Image Processing         │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ PaddleOCR                │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Regex + Voting           │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Violation Event          │
└─────────────┬────────────┘
              ▼
       ┌──────┴───────┐
       │              │
       ▼              ▼
 Evidence           SQLite
       │              │
       └──────┬───────┘
              ▼
┌──────────────────────────┐
│ Streamlit Dashboard      │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Chatbot / API            │
└──────────────────────────┘
```

---

# 15. Thiết kế Database

## Bảng `vehicles`

```text
id
track_id
plate_number
vehicle_type
first_seen
last_seen
```

## Bảng `violations`

```text
id
track_id
plate_number
vehicle_type
violation_type
detected_at
confidence
evidence_path
status
```

## Bảng `violation_rules`

```text
id
violation_type
vehicle_type
description
source
effective_date
reference
```

## Bảng `processing_sessions`

```text
id
source_type
source_name
started_at
ended_at
total_frames
total_vehicles
total_violations
```

---

# 16. Quy tắc phát hiện vi phạm

## 16.1. NO_HELMET

```text
Motorcycle
    ↓
Rider/Head ROI
    ↓
Helmet = false
    ↓
Check confidence
    ↓
Temporal confirmation
    ↓
NO_HELMET
```

---

## 16.2. WRONG_WAY

```text
Track history
    ↓
p(t) = center point
    ↓
p(t) - p(t-k)
    ↓
Movement Vector
    ↓
Compare Allowed Direction
    ↓
WRONG_WAY
```

---

## 16.3. WRONG_LANE

```text
Vehicle trajectory
       ↓
Lane polygon
       ↓
Current Lane
       ↓
Allowed Lane
       ↓
If mismatch for N frames
       ↓
WRONG_LANE
```

---

## 16.4. RED_LIGHT

```text
Traffic light = RED
       ↓
Vehicle approaching stop line
       ↓
Vehicle center crosses line
       ↓
Crossing confirmed
       ↓
RED_LIGHT
```

---

## 16.5. CHE_COVERED_PLATE

Tính năng này được đánh dấu **optional**.

Một phương án là:

```text
Plate Detection
       ↓
Plate visibility/quality
       ↓
Normal / Occluded / Invalid
```

Không xem:

```text
OCR failed
```

đồng nghĩa với:

```text
Plate intentionally covered
```

vì OCR thất bại còn có nhiều nguyên nhân khác như blur, thiếu sáng, góc nhìn và khoảng cách.

---

# 17. Chatbot tra cứu

## Kiến trúc

```text
User
 ↓
Streamlit Chat UI
 ↓
chatbot.py
 ↓
Identify intent
 ↓
Query local database/rules
 ↓
Build context
 ↓
LLM API
 ↓
Response
```

## Ví dụ câu hỏi

```text
Xe máy đi ngược chiều là loại vi phạm nào?

Hệ thống đã ghi nhận những lỗi nào trong video?

Biển số 51G12345 có những violation event nào?

Sự kiện ID 102 được phát hiện lúc nào?

Hệ thống dùng Machine Learning ở phần nào?
```

## Nguyên tắc dữ liệu pháp luật

Thông tin xử phạt phải:

```text
Source
+
Effective Date
+
Version
```

Không dùng chatbot như nguồn pháp luật duy nhất.

---

# 18. Kiểm thử và đánh giá

## 18.1. ML Metrics

### Object Detection

```text
Precision
Recall
mAP@50
mAP@50:95
```

### OCR

```text
Character Accuracy
Plate-level Accuracy
```

### Tracking

Theo dõi:

```text
ID switch
Track loss
Tracking stability
```

### System

```text
FPS
Average inference time
End-to-end latency
```

---

## 18.2. Violation Detection Metrics

Tạo ground truth thủ công trên một tập video test.

Ví dụ:

```text
Expected Violations = 20
Detected Violations = 18
Correct Violations = 16
False Positives = 3
False Negatives = 4
```

Từ đó tính:

```text
Precision
Recall
F1
```

cho violation engine.

---

# 19. Git Workflow và hướng dẫn đóng góp

## Branches

```text
main
│
├── feature/khoa-ml-cv
├── feature/member2-backend
└── feature/member3-ui-integration
```

## Quy tắc

Không commit trực tiếp vào `main`.

### Quy trình

```bash
git checkout main
git pull
```

Tạo/update branch cá nhân:

```bash
git checkout feature/khoa-ml-cv
git pull origin main
```

Sau khi hoàn thành:

```bash
git add .
git commit -m "feat: add vehicle tracking"
git push origin feature/khoa-ml-cv
```

Sau đó tạo Pull Request.

---

## Commit convention

```text
feat:     tính năng mới
fix:      sửa lỗi
refactor: thay đổi cấu trúc code
docs:     tài liệu
test:     test
chore:    công việc hệ thống
```

Ví dụ:

```text
feat: add wrong-way detection
feat: add violation database
fix: prevent duplicate violation event
docs: update week 2 roadmap
test: add OCR validation test
```

---

## Quy tắc merge

Mỗi PR nên có:

```text
1. Mô tả thay đổi
2. Files thay đổi
3. Cách test
4. Kết quả
5. Screenshot nếu có UI
```

Không merge code nếu:

- Không chạy được.
- Không biết test thế nào.
- Có secret/API key.
- Làm hỏng pipeline của main.

---

# 20. Xử lý sự cố

## YOLO không detect

Kiểm tra:

```text
Model path
Confidence
Image size
Dataset
Lighting
Camera angle
```

---

## Tracking ID bị đổi

Kiểm tra:

```text
Tracker
Confidence threshold
Occlusion
FPS
Camera movement
```

Với video liên tục, tracking phải được duy trì giữa các frame của cùng stream.

---

## OCR sai

Kiểm tra:

```text
Plate crop
Resolution
CLAHE
Thresholding
Deskew
OCR confidence
Voting
Regex
```

---

## Database locked

Kiểm tra:

```text
Connection lifecycle
Commit
Close
Concurrent writes
```

Nếu cần đồng thời nhiều tiến trình, cân nhắc cấu hình SQLite phù hợp và kiểm soát transaction.

---

## Streamlit chậm

Kiểm tra:

```text
Model inference trong UI
Frame rate
Resolution
OCR frequency
Database query frequency
Caching
```

Không nên chạy OCR trên toàn bộ frame nếu không cần.

---

## Camera điện thoại không kết nối

Kiểm tra theo thứ tự:

```text
1. Phone và PC cùng mạng.
2. Stream URL đúng.
3. OpenCV đọc được stream.
4. Test stream bằng một script nhỏ.
5. Sau khi stream ổn định mới ghép YOLO.
```

---

## CUDA không hoạt động

Chạy:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu trả:

```text
False
```

kiểm tra:

- NVIDIA driver.
- PyTorch build.
- CUDA compatibility.
- Environment đang active.

---

# 21. Giới hạn của dự án

MVP 4 tuần có các giới hạn:

- Camera ưu tiên góc cố định.
- Số loại vi phạm giới hạn.
- Tracking có thể giảm chất lượng khi vật thể che khuất mạnh.
- OCR phụ thuộc chất lượng biển số.
- Camera chuyển động mạnh làm bài toán khó hơn.
- Vi phạm thực tế cần xác minh bởi con người.
- Chatbot không phải nguồn pháp luật độc lập.
- Mức độ chính xác phụ thuộc trực tiếp vào dataset.
- Không tuyên bố hệ thống có độ tin cậy đủ cho xử phạt ngoài thực tế.

---

# 22. Tài liệu tham khảo

## Python

- Python Documentation
- Python `venv`
- Python `sqlite3`
- Python `multiprocessing`
- Python `datetime`
- Python `logging`

## Machine Learning

- Machine Learning fundamentals: supervised learning, train/validation/test, overfitting, generalization.
- PyTorch Documentation.
- Ultralytics YOLO Documentation.
- OpenCV Documentation.
- PaddleOCR Documentation.
- OpenVINO Documentation.
- Streamlit Documentation.

## Các chủ đề cần nghiên cứu

### Object Detection

- Bounding Box
- IoU
- NMS
- Precision
- Recall
- mAP

### Object Tracking

- Track ID
- Motion prediction
- Data association
- ByteTrack
- BoT-SORT

### Model Training

- Epoch
- Batch size
- Learning rate
- Augmentation
- Transfer learning
- Loss

### Computer Vision

- ROI
- Polygon
- Contour
- Perspective
- Optical/trajectory analysis

### OCR

- Text detection
- Text recognition
- Preprocessing
- Confidence
- Post-processing

---

# 📌 Kết quả kỳ vọng sau 4 tuần

Nếu hoàn thành đúng MVP, hệ thống có thể trình diễn:

```text
               TRAFFIC VIDEO
                     │
                     ▼
              ┌──────────────┐
              │ YOLO Detect  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │   Tracking   │
              └──────┬───────┘
                     ▼
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    NO HELMET    WRONG WAY    RED LIGHT
        │            │            │
        └────────────┼────────────┘
                     ▼
             PLATE + OCR
                     │
                     ▼
              Evidence Image
                     │
                     ▼
                SQLite DB
                     │
              ┌──────┴──────┐
              ▼             ▼
         Streamlit       Chatbot
          Dashboard        API
```

## Demo tối thiểu

```text
1. Người dùng mở Streamlit.
2. Chọn Upload Video hoặc Live Camera.
3. Hệ thống phát hiện xe.
4. Tracking tạo ID.
5. Hệ thống phát hiện vi phạm.
6. Hệ thống đọc biển số nếu có thể.
7. Hệ thống lưu evidence.
8. SQLite lưu violation event.
9. Dashboard hiển thị kết quả.
10. Người dùng hỏi chatbot để tra cứu/giải thích dữ liệu.
```

---

## Trạng thái dự án

```text
Project Type: Machine Learning / Computer Vision
Duration: 4 weeks MVP
Team Size: 3
Status: In Development
```

> **Ưu tiên của dự án:** hoàn thành một pipeline nhỏ nhưng chạy ổn định và đo lường được, thay vì xây quá nhiều loại vi phạm nhưng không có đủ thời gian kiểm thử.
