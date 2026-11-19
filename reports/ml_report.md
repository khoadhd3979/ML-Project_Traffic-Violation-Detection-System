# BÁO CÁO ĐỒ ÁN: HỆ THỐNG PHÁT HIỆN VI PHẠM GIAO THÔNG (TVDS)
**Đề tài:** Tự động phát hiện vi phạm không đội mũ bảo hiểm và nhận diện biển số xe máy  
**Trạng thái tài liệu:** Đã hoàn thiện Phần 1–4 với đầy đủ số liệu thực nghiệm E1 (Baseline, EXP-1, EXP-2, EXP-3) và số đo Latency/FPS trên cùng máy; chờ Khoa chốt chính thức mô hình cuối (trưa 21/11) và cập nhật tỷ lệ đóng góp thành viên.

---

## 1. Mô tả bài toán & Kiến trúc hệ thống

### 1.1. Mục tiêu và phạm vi
Hệ thống **TVDS (Traffic Violation Detection System)** được xây dựng nhằm tự động phát hiện hành vi **không đội mũ bảo hiểm** của người đi xe máy (người điều khiển và người ngồi sau) từ nguồn webcam, tệp video hoặc luồng RTSP, đồng thời **nhận diện biển số xe** và lưu trữ bằng chứng phục vụ tra cứu, thống kê.

Phạm vi triển khai hiện tại tập trung vào lỗi `NO_HELMET` (không đội mũ bảo hiểm) kết hợp OCR biển số Việt Nam. Các loại vi phạm mở rộng trong schema như `WRONG_LANE`, `WRONG_WAY`, `RED_LIGHT` và giao diện hỏi đáp là hướng phát triển tiếp theo.

### 1.2. Kiến trúc tổng thể và luồng xử lý
Hệ thống gồm 3 tầng tách biệt:

1. **Tầng AI & Xử lý luồng (`src/models/`, `src/features/`):**
   - **Phát hiện & Theo dõi (`src/models/detection.py`):** Sử dụng mô hình nhận diện mũ/xe (`models/weights_helmet/helmet_best.pt`) kết hợp thuật toán theo dõi **ByteTrack** (`persist=True`, `TRACK_CONF = 0.25`) để duy trì `track_id` ổn định qua các khung hình. Sau tracking, hệ thống áp dụng ngưỡng lọc riêng theo từng lớp (`CLASS_CONF_THRESHOLDS`: `bike = 0.35`, `driver_without_helmet = 0.45`, `passenger_without_helemt = 0.45`; mặc định `CONF_THRESHOLD = 0.40` cho các lớp còn lại).
   - **Ghép cặp người–xe & lọc nhiễu (`src/features/processing.py`):** Mở rộng bounding box của xe (`EXPAND_SIDE = 0.10`, `EXPAND_TOP = 0.35`, `EXPAND_BOTTOM = 0.10`) để ghép người vi phạm (class 5/6) vào xe (class 1) dựa trên tâm bounding box hoặc tỷ lệ chứa `MIN_CONTAINMENT = 0.20`, kết hợp lọc xung đột mũ (`HELMET_CONFLICT_IOU = 0.50`), loại bỏ đối tượng quá nhỏ/quá lớn (`MIN_HEIGHT_RATIO = 0.15`, `MAX_HEIGHT_RATIO = 3.0`), và bỏ qua trường hợp nhập nhằng giữa hai xe gần nhau (`AMBIGUITY_MARGIN = 0.10`).
   - **Xác nhận theo chuỗi & OCR bất đồng bộ (`src/features/live_stream.py`, `src/models/ocr_engine.py`, `src/models/ocr_worker_proc.py`):** Áp dụng cơ chế **debouncing** yêu cầu cùng một `track_id` xe vi phạm liên tiếp trong `VIOLATION_CONFIRM_FRAMES = 3` chu kỳ trước khi ghi nhận vi phạm. Khi xác nhận, hệ thống lưu ảnh bằng chứng (padding `EVIDENCE_PADDING_RATIO = 0.05`) và ghi ngay bản ghi vào DB với biển số tạm `UNKNOWN`, đồng thời đẩy crop vùng xe sang tiến trình nền dùng mô hình biển số (`models/weights_plate/plate_best.pt`) + **PaddleOCR** (`PlateOCR`) để cập nhật lại biển số khi đọc thành công.

2. **Tầng Dữ liệu & Backend REST API (`src/data/`, `src/api/`):**
   - **Cơ sở dữ liệu SQLite (`src/data/schema.sql`, `src/data/database_manager.py`):** Gồm 4 bảng chuẩn hóa (`violation_rules`, `processing_sessions`, `vehicles`, `violations`) và view tổng hợp `v_violation_details`. `DatabaseManager` tích hợp cơ chế `_migrate()` tự động nâng cấp DB cũ lên schema hiện tại mà không làm mất dữ liệu.
   - **FastAPI Backend (`src/api/main.py`, `src/api/schemas.py`):** Chạy độc lập trên cổng `8000`, cấu hình CORS, mount thư mục tĩnh `/evidence`, và cung cấp các endpoint chính: `/health`, `/api/v1/violations` (lọc và phân trang, `page_size` `1..100`), `/api/v1/stats/overview`, `/api/v1/vehicles/{license_plate}/history`, và `POST /api/v1/detect/image`.

3. **Tầng Giao diện (`app/main.py`, `app/services/api_client.py`):**
   - Giao diện **Streamlit** (cổng `8501`) hiển thị luồng phát hiện trực tiếp, danh sách lịch sử vi phạm kèm ảnh bằng chứng, và bảng điều khiển thống kê thông qua `APIClient` kết nối tới FastAPI.

---

## 2. Xác định và phân tích dữ liệu (Phase A)

> *[Chờ bổ sung output 2 script Phase A (`phase_a_dataset_audit.py` và `phase_a_raw_predictions.py`) từ Khoa để điền số liệu thật.]*

### 2.1. Quy mô và cấu trúc bộ dữ liệu gốc
- **Nguồn cấu hình:** `data_helmet.yaml` — tổng cộng **7 lớp** (giữ nguyên tên lớp gốc trong bộ nhãn, gồm cả lỗi chính tả `helemt` ở lớp 3 và 6 để khớp trọng số đã huấn luyện):
  - `0`: `driver_with_helmet`
  - `1`: `bike`
  - `2`: `driver`
  - `3`: `passenger_with_helemt` *(lỗi chính tả nguyên bản trong dataset/weights)*
  - `4`: `passenger`
  - `5`: `driver_without_helmet`
  - `6`: `passenger_without_helemt` *(lỗi chính tả nguyên bản trong dataset/weights)*
- **Phân chia tập gốc:** **368 ảnh train** (kích thước trung vị `416 × 416`) và **65 ảnh val** (kích thước trung vị `197 × 300`, tổng `271` instances trên val).

### 2.2. Kết quả kiểm tra chất lượng dữ liệu (Audit Phase A)
1. **Ứng viên trùng lặp gần (Near-duplicate qua dHash):**
   - Kiểm tra dHash (`Hamming <= 6`) giữa train và val ghi nhận **4 cặp ứng viên gần trùng** (`Hamming = 2, 0, 0, 2`), tương ứng **4/65 ảnh val (~6.15%)** bị ảnh hưởng; trong tập train ước lượng có khoảng **358 cụm ứng viên** ở ngưỡng `Hamming <= 4`.
   - Đây là **các ứng viên near-duplicate** theo perceptual hash (chưa đối chiếu kiểm tra thủ công từng cặp ảnh gốc nên chưa kết luận là leakage hay trùng byte tuyệt đối), nhưng là giới hạn cần công bố khi diễn giải điểm val mAP.
2. **Phân bố nguồn theo tên tệp & cấu hình người/xe:**
   - Theo heuristic tên tệp, tập train có **90 nhóm** và tập val có **27 nhóm**, trong đó **1 nhóm (`BikesHelmets`) dùng chung** giữa hai tập (`190/368` ảnh train ≈ `51.6%` và `16/65` ảnh val ≈ `24.6%`), tạo rủi ro phụ thuộc nguồn dữ liệu.
   - Theo phép ghép xấp xỉ head–bike: tập train có khoảng **500 bike instances** (`289` xe chưa ghép được head box, `104` `driver_with`, `34` `driver_without`, `31` `driver_with × 2`, `15` `driver_without × 2`); tập val có **63 bike instances** (`32` chưa ghép được head box, `15` `driver_with`, `7` `driver_without`). Các cấu hình có hành khách (class 3, 4, 6) chiếm tỷ lệ thấp.
3. **Thống kê head box và độ sáng (`HSV V < 70`):**

| Class | Số head box (audit) | Median bbox w | Median bbox h | Tỷ lệ crop có median HSV V < 70 |
|---|---:|---:|---:|---:|
| `0` `driver_with_helmet` | 439 | 0.123 | 0.188 | 27% |
| `3` `passenger_with_helemt` | 65 | 0.130 | 0.212 | 22% |
| `5` `driver_without_helmet` | 129 | 0.096 | 0.167 | 12% |
| `6` `passenger_without_helemt` | 74 | 0.101 | 0.170 | 28% |

   - Lớp `3` và `6` có số mẫu ít hơn rõ rệt so với lớp `0` (mất cân bằng lớp). Tỷ lệ crop có độ sáng trung vị thấp (`V < 70`) ở lớp `0`, `3`, `6` là đáng kể (22%–28%), do đó quy tắc gán nhãn quy định **không dùng màu tối làm tiêu chí kết luận không đội mũ**.

### 2.3. Chẩn đoán dự đoán thô trên video quay màn hình & Cổng Fine-tune (NO-GO)
- Chạy dự đoán thô (`helmet_best.pt`, `imgsz = 640`, bước lấy mẫu `STEP = 2`, ngưỡng ghi `conf >= 0.45`, không tracking/debouncing) trên **3 video quay màn hình (screen recording)**:
  - `5pc` (`1920×1080`): **281** dòng dự đoán class 5/6 (median conf `0.76`, `21%` có box đối thủ class 0/3 đạt `IoU >= 0.5`).
  - `1pc` (`2160×3840`): **78** dòng dự đoán class 5/6 (median conf `0.58`, `35%` có box đối thủ class 0/3 đạt `IoU >= 0.5`, median chiều cao head box quy đổi ~`34 px`).
  - `7pc` (`1920×1080`): **25** dòng dự đoán class 5/6 (median conf `0.66`, `32%` có box đối thủ class 0/3 đạt `IoU >= 0.5`).
- **Lưu ý diễn đạt:** Do cả 3 video đều là *screen recording* (chịu ảnh hưởng nén, crop, scaling) và không có ground truth theo frame/track, các dòng dự đoán class 5/6 trên **không được kết luận là false positives** và không khẳng định *domain gap* trên video gốc.
- **Quyết định Fine-tune (`NO-GO`):** Căn cứ Labeling Policy (`APPROVED`), nhóm thống nhất **đóng nhánh fine-tuning** do chưa có video gốc và holdout độc lập (chưa đạt điều kiện tối thiểu 30 full-frame gốc được gán nhãn & QA; 28 ảnh crop `diag_out` từ screen recording không đủ điều kiện làm dữ liệu train).

---

## 3. Các mô hình thử nghiệm (Bảng so sánh E1)

### 3.1. Giao thức thử nghiệm chung
Để so sánh công bằng giữa các kiến trúc khác nhau theo yêu cầu thí nghiệm, cả 3 thử nghiệm E1 đều chạy trên **cùng bộ dữ liệu gốc (368 train / 65 val)**, cùng kích thước ảnh **`imgsz = 640`**, **`batch = 16`**, **`seed = 0`**, và cùng số epoch đã chốt là **50 epochs**.

**Lưu ý bất đối xứng về số epoch:** Mô hình **Baseline YOLOv8n** hiện tại trong hệ thống được huấn luyện trước đó với cấu hình `epochs = 100`, `patience = 25` và dừng sớm ở **epoch 95** (`time = 609.92s`), trong khi 3 mô hình thử nghiệm E1 được huấn luyện với **50 epochs**. Khi đối chiếu số liệu, cần xét đến sự khác biệt số epoch này.

### 3.2. Mô tả 4 mô hình đối chiếu
1. **Baseline — YOLOv8n (Anchor-free, Nano):** Mô hình hiện đang tích hợp tại `models/weights_helmet/helmet_best.pt`. Kiến trúc anchor-free với decoupled head, gọn nhẹ, làm mốc chuẩn để đối chiếu.
2. **EXP-1 — YOLOv8s (Anchor-free, Small — Khoa):** Tăng dung lượng mạng từ `n` lên `s` để kiểm tra khả năng cải thiện nhận diện các lớp yếu/ít mẫu (lớp 3, 5, 6) và đánh đổi về tốc độ/kích thước.
3. **EXP-2 — YOLOv5n (Anchor-based, Nano — Phúc):** Dùng repo chính thức `ultralytics/yolov5` (`v7.0-574-g1e7cf74c`, 68 layers, 1,768,636 tham số, 4.2 GFLOPs, với anchor boxes truyền thống) nhằm so sánh trực tiếp kiến trúc **anchor-based** thế hệ trước với **anchor-free** (`YOLOv8n`) trên cùng dữ liệu.
4. **EXP-3 — RT-DETR-L (Transformer, NMS-free — Nguyên):** Họ mô hình DETR dựa trên Transformer, loại bỏ bước hậu xử lý NMS truyền thống, nhằm đánh giá lợi thế trong cảnh đông xe/che khuất.

### 3.3. Bảng tổng hợp kết quả thực nghiệm (E1)

| Mô hình | Kiến trúc | Epochs | P (All) | R (All) | mAP50 (All) | mAP50-95 (All) | Class 3 (P / R) | Class 5 (P / R) | Class 6 (P / R) | Latency (ms/ảnh) | FPS | Size (MB) | Thời gian train |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| **Baseline: YOLOv8n** | Anchor-free | 95 *(early-stop)* | 0.754 | 0.753 | 0.797 | 0.416 | 0.641 / 0.571 | 0.730 / 0.736 | 0.806 / 0.568 | 94.59 | 10.57 | 5.95 | 10.17 phút *(609.9s)* |
| **EXP-1: YOLOv8s** | Anchor-free | 50 | 0.814 | 0.692 | 0.808 | 0.413 | 0.633 / 0.571 | 0.960 / 0.726 | 0.809 / 0.591 | 162.19 | 6.17 | 21.46 | 6.05 phút |
| **EXP-2: YOLOv5n** | Anchor-based | 50 | 0.556 | 0.713 | 0.680 | 0.336 | 0.424 / 0.160 | 0.440 / 0.758 | 0.402 / 0.727 | 82.24 | 12.16 | 3.67 | 6.90 phút |
| **EXP-3: RT-DETR-L** | Transformer | 50 | 0.831 | 0.741 | 0.816 | 0.432 | 0.765 / 0.643 | 0.959 / 0.712 | 0.923 / 0.682 | 1191.02 | 0.84 | 63.18 | 27.87 phút |

> *Ghi chú đo đạc:*
> - **Độ chính xác (`val`):** Toàn bộ 4 mô hình được đánh giá thống nhất trên cùng tập `valid` (`65` ảnh, `271` instances, `imgsz = 640`). Riêng `Baseline YOLOv8n` được đánh giá trực tiếp từ tệp trọng số `helmet_best.pt` (thay vì lấy dòng epoch 95 trong `results.csv` vốn cho `mAP50 = 0.7954`, `mAP50-95 = 0.3977`).
> - **Mốc thăm dò 30 epochs của `EXP-2 YOLOv5n`:** Đạt `P=0.578`, `R=0.619`, `mAP50=0.550`, `mAP50-95=0.261` (`4.38 phút`), lớp hiếm Class 3 có `R=0.000`; khi nâng lên mốc chốt `50` epochs, Class 3 bắt đầu hội tụ (`R=0.160`, `mAP50=0.303`) và Class 5/6 cải thiện rõ (`mAP50`: `0.545→0.646` và `0.332→0.648`).
> - **Tốc độ suy luận (Latency/FPS):** Đo tập trung bằng `scripts/benchmark_e1_latency.py` trên cùng máy chạy demo (`Windows 10`, Intel CPU, `CUDA: False`, `20` ảnh từ `data/train_raw_frames`, `warmup = 3`, `imgsz = 640`).

### 3.4. Phân tích đối chiếu 4 mô hình
- **EXP-3 (`RT-DETR-L`, Transformer, `32.82M` tham số):** Dẫn đầu về độ chính xác (`mAP50 = 0.816`, `mAP50-95 = 0.432`, Class 3/5/6 đều cao nhất), cho thấy lợi thế của cơ chế Transformer & bỏ NMS trên các đối tượng nhỏ/che khuất. Tuy nhiên, chi phí tính toán quá lớn trên CPU (`1191.02 ms/ảnh`, chỉ đạt `0.84 FPS`, kích thước `63.18 MB`), không thể đáp ứng luồng video real-time + ByteTrack trên máy demo không có GPU.
- **EXP-1 (`YOLOv8s`, Anchor-free Small, `11.14M` tham số):** Tăng `mAP50` tổng từ `0.797` lên `0.808` (`+0.011`) và đẩy `Precision` của Class 5 lên `0.960`, nhưng `mAP50-95` (`0.413`) tương đương Baseline (`0.416`) và `Recall` tổng giảm (`0.692` so với `0.753`). Đổi lại, kích thước gấp `3.6×` (`21.46 MB`) và tốc độ CPU giảm từ `10.57 FPS` xuống `6.17 FPS`.
- **EXP-2 (`YOLOv5n`, Anchor-based Nano, `1.77M` tham số):** Nhẹ và nhanh nhất (`3.67 MB`, `12.16 FPS`), nhưng với kiến trúc anchor-based truyền thống trên tập dữ liệu nhỏ ở `50` epochs, độ chính xác thấp hơn đáng kể (`mAP50 = 0.680`, Class 3 `R = 0.160`), không phù hợp để thay thế mô hình hiện tại.
- **Baseline (`YOLOv8n`, Anchor-free Nano, `3.01M` tham số):** Cân bằng tốt nhất giữa độ chính xác (`mAP50 = 0.797`, `mAP50-95 = 0.416`, `Recall` cao nhất `0.753`) và tốc độ thực tế trên CPU (`94.59 ms/ảnh`, `10.57 FPS`, `5.95 MB`).

---

## 4. Chọn mô hình cuối

> *[Số liệu E1 và latency đã đầy đủ — phần quyết định dưới đây là đề xuất căn cứ số liệu, chờ Khoa chủ trì xác nhận chính thức tại mốc chốt trưa 21/11.]*

### 4.1. Tiêu chí lựa chọn
Việc lựa chọn mô hình tích hợp cuối cùng dựa trên 6 tiêu chí:
1. Hiệu quả tổng thể (`mAP50`, `mAP50-95`, cân bằng `Precision` / `Recall`).
2. Độ chính xác trên các lớp vi phạm và lớp phụ quan trọng (class 3, 5, 6).
3. Tốc độ suy luận (Latency / FPS thực tế trên máy chạy demo) và kích thước weights.
4. Tính ổn định khi ghép cặp người–xe và tích hợp với ByteTrack + debouncing.
5. Khả năng tích hợp và kiểm thử ổn định trước mốc đóng băng code.
6. Tính tái lập (lưu đủ `args.yaml`/`opt.yaml`, `results.csv`, trọng số khớp hash MD5).

### 4.2. Quyết định và lý do lựa chọn (Đề xuất giữ `Baseline YOLOv8n`)
Căn cứ 6 tiêu chí trên và bảng số liệu tại Mục 3.3, phương án phù hợp nhất cho hệ thống TVDS là **giữ nguyên mô hình `YOLOv8n` (`models/weights_helmet/helmet_best.pt`)** làm mô hình tích hợp cuối cùng:
1. **Cân bằng tốt nhất cho triển khai real-time trên CPU:** `YOLOv8n` đạt `10.57 FPS` (`94.59 ms/ảnh`, `5.95 MB`), đủ nhanh để ByteTrack duy trì `track_id` liên tục qua `VIOLATION_CONFIRM_FRAMES = 3` chu kỳ mà không làm nghẽn tiến trình UI/OCR.
2. **Không có ứng viên nào thắng áp đảo cả độ chính xác lẫn tốc độ:** `RT-DETR-L` chỉ nhỉnh nhẹ `+0.019` `mAP50` (`0.816` so với `0.797`) nhưng chậm hơn `12.6×` trên CPU (`0.84 FPS`); `YOLOv8s` chỉ tăng `+0.011` `mAP50` (`0.808`) nhưng giảm `Recall` tổng (`0.692` so với `0.753`) và chậm hơn ~`42%` (`6.17 FPS`); `YOLOv5n` nhẹ hơn nhưng `mAP50` thấp hơn rõ rệt (`0.680`).
3. **Ổn định và rủi ro tích hợp thấp nhất:** `helmet_best.pt` đã tích hợp sẵn trong `src/models/detection.py`, đã xác minh hash và vượt qua toàn bộ `69/69` test trong `tests/`.

> *Lưu ý:* Mô hình `YOLOv8n` hiện tại được train `95` epochs (early-stop), nhiều hơn mốc `50` epochs của E1 — do đó báo cáo ghi nhận rõ đây là lựa chọn thực dụng dựa trên hiệu năng thực tế và độ ổn định tích hợp, không khẳng định `YOLOv8n` ưu việt tuyệt đối về mặt học thuật khi chưa train lại các đối thủ ở cùng `95` epochs.

---

## 5. Kết luận, Hạn chế & Đóng góp thành viên

### 5.1. Kết luận & Hạn chế hệ thống
- **Quy mô và độ phủ dữ liệu:** Tập gốc còn nhỏ (`368` train / `65` val), mất cân bằng lớp (các lớp hành khách `3` và `6` ít mẫu hơn đáng kể so với `0`), nhóm tên nguồn `BikesHelmets` chiếm tỷ lệ lớn và tồn tại `4` cặp ứng viên near-duplicate giữa train–val.
- **Chưa có video gốc độc lập (holdout) để đánh giá track-level:** Các video kiểm thử hiện có đều là *screen recording*, chưa đủ điều kiện mở fine-tuning một cách kiểm chứng được nên nhánh fine-tune giữ ở trạng thái `NO-GO`.
- **Kết luận cuối:** `[Chờ bổ sung sau khi hoàn thành bảng E1 và chốt mô hình ngày 21/11.]`

### 5.2. Bảng phân công và tỷ lệ đóng góp thực tế

| Thành viên | Vai trò chính | Đầu việc thực tế đã thực hiện | Tỷ lệ đóng góp (%) |
|---|---|---|---:|
| **Khoa** | AI/ML Lead | `[xác nhận khi chốt báo cáo]` | `[%]` |
| **Phúc** | Backend & Data | DB schema/migration, FastAPI REST API, EXP-2 YOLOv5n, tổng hợp báo cáo Phần 1–4, requirements & README | `[%]` |
| **Nguyên** | Frontend UI | `[xác nhận khi chốt báo cáo]` | `[%]` |
