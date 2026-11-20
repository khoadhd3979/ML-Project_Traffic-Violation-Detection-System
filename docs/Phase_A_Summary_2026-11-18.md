# PHASE A SUMMARY — TRAFFIC VIOLATION DETECTION
**Ngày:** 18/11/2026  
**Người thực hiện:** Khoa  
**Nguồn kiểm tra:** `ML-Traffic-FULL-Package-Khoa-2026-11-18.zip`  
**Phạm vi:** Helmet detection dataset audit, raw prediction diagnostics, dữ liệu và điều kiện mở hard-negative fine-tuning.

## 1. Kết luận điều hành

### Phán quyết

- **Dataset audit:** HOÀN THÀNH — đã có output của bốn nhóm kiểm tra.
- **Raw prediction diagnostics:** HOÀN THÀNH VỀ MẶT CHẠY SCRIPT — nhưng chỉ là chẩn đoán model trên ba video quay màn hình, không phải đánh giá lỗi có ground truth.
- **Video provenance / holdout:** CHƯA ĐẠT — manifest chưa có; cả ba video có raw-prediction output (`5pc`, `1pc`, `7pc`) được xác nhận là screen recording. Nhóm báo cáo cả bảy video test đều là screen recording và chưa có file nguồn.
- **Phase A:** Có thể ghi **`DATA AUDIT COMPLETE — HOLDOUT UNAVAILABLE`** sau khi hoàn thành manifest ghi rõ tình trạng của cả bảy video. Không được ghi Phase A là đã xác nhận được domain gap trên video gốc.
- **Fine-tune gate:** **NO-GO / ĐÓNG tại thời điểm 18/11/2026.** Không có bằng chứng về tối thiểu 30 frame gốc full-frame đã gán nhãn và QA; không có holdout video gốc độc lập; các crop trong `diag_out` không đủ điều kiện làm training data theo labeling policy APPROVED.

**Khuyến nghị thực thi:** không mở fine-tune với gói dữ liệu hiện tại. Tiếp tục ba thử nghiệm thuật toán E1 trên dataset gốc theo giao thức chung. Chỉ xem xét mở lại fine-tune nếu trước cut-line nhóm tìm được video gốc có provenance rõ, có train frames tách khỏi holdout và hoàn thành toàn bộ QA nhãn.

---

## 2. Nguồn và cấu hình đã kiểm tra

### 2.1. Dataset và YAML

`data_helmet.yaml` khai báo:

- `nc = 7`.
- Class order:
  0. `driver_with_helmet`
  1. `bike`
  2. `driver`
  3. `passenger_with_helemt`
  4. `passenger`
  5. `driver_without_helmet`
  6. `passenger_without_helemt`

Đường dẫn được YAML ghi:
- Train: `/content/dataset/content/smart-helmet-detection-using-dl/data/train`
- Val: `/content/dataset/content/smart-helmet-detection-using-dl/data/valid`

Dataset audit output ghi nhận:
- Train: **368 ảnh**.
- Validation: **65 ảnh**.
- Median kích thước ảnh train: **416 × 416**.
- Median kích thước ảnh validation: **197 × 300**.

Class mapping trong `class_mapping.txt` phù hợp với thứ tự 7 class nêu trên. Tên `helemt` là typo hiện diện trong `model.names`; phải giữ nguyên trong weights và code tương thích.

### 2.2. Baseline YOLOv8n

`args_helmet.yaml` ghi nhận:
- Model khởi tạo: `yolov8n.pt`.
- Epoch cấu hình: 100; `results.csv` có 95 epoch được ghi.
- Batch: 16.
- Image size: 640.
- Seed: 0.
- LR0: 0.01.
- Patience: 25.
- Pretrained: true.

Dòng epoch 95 của `results.csv`:
- Precision: **0.79321**.
- Recall: **0.70982**.
- mAP50: **0.79540**.
- mAP50-95: **0.39770**.

Lưu ý quan trọng:
- Đây là metrics ở **dòng epoch 95**, không mặc nhiên là metrics khi đánh giá file `helmet_best.pt`.
- Trong `results.csv`, mAP50-95 cao nhất là **0.41649 ở epoch 70**; mAP50 cao nhất là **0.80635 ở epoch 77**. Các cực đại này thuộc các epoch khác nhau.
- Vì vậy, bảng E1 nên chạy validation thống nhất trực tiếp trên weights baseline đang dùng (`helmet_best.pt`) và weights của ba thử nghiệm, thay vì lấy metrics epoch cuối làm đại diện cho best checkpoint mà không xác minh.

Gói ZIP có `weights_hashes.txt` ghi các hash của bốn weights đã được Muse xác minh. Tuy nhiên ZIP không chứa binary `.pt`, nên lần kiểm tra gói này chỉ đối chiếu được manifest hash, không tự tính lại MD5/SHA-256 từ weights.

---

## 3. Kết quả Phase A — Dataset audit

### 3.1. Near-duplicate train ↔ validation

Script `phase_a_dataset_audit.py` dùng dHash với ngưỡng Hamming distance `<= 6` cho so sánh train–val.

Kết quả:
- **4 cặp** train–val bị đánh dấu gần trùng.
- **4/65 ảnh validation** bị ảnh hưởng, tương đương khoảng **6.15%** validation.
- Khoảng cách Hamming của bốn cặp: **2, 0, 0, 2**.
- Trong train, script ước tính 368 ảnh tạo thành khoảng 358 cụm gần trùng ở ngưỡng Hamming `<= 4`.

Diễn giải:
- Đây là **các ứng viên near-duplicate cần kiểm tra hình ảnh và nguồn**, chưa phải kết luận xác nhận bốn ảnh validation chắc chắn bị leakage.
- dHash distance 0 chỉ có nghĩa hai ảnh có cùng perceptual hash theo phép đo này; không chứng minh hai file byte-identical. Muốn kết luận trùng file cần đối chiếu hash file hoặc kiểm tra ảnh gốc.
- Số cụm gần trùng trong train là kết quả heuristic của script, không nên coi là số cụm ground-truth đã xác nhận.

Hệ quả:
- E1 vẫn có thể sử dụng cùng split hiện tại cho cả ba model để giữ protocol so sánh nhất quán, nhưng phải công bố rủi ro near-duplicate.
- Không được diễn giải validation mAP như bằng chứng tổng quát hóa hoàn toàn độc lập.
- Nếu có thời gian kiểm tra thủ công bốn cặp, lưu phán quyết và nguồn ảnh. Không tự ý thay split riêng cho từng model.

### 3.2. Nhóm nguồn theo tên file

Kết quả heuristic:
- Train có **90 nhóm tên file**.
- Validation có **27 nhóm tên file**.
- Có **1 nhóm tên file dùng chung** giữa train và validation.
- Nhóm `BikesHelmets`: **190/368 train** (khoảng 51.6%) và **16/65 validation** (khoảng 24.6%).

Diễn giải:
- Dataset có sự tập trung đáng kể vào nhóm tên nguồn `BikesHelmets`.
- Có dấu hiệu nguồn được đặt tên chung xuất hiện ở cả train và validation.
- Phép nhóm chỉ dựa trên quy tắc xử lý tên file; nó **không xác minh được nguồn video/camera thực tế** và không chứng minh hai split chứa cùng video.
- Do đó, kết luận đúng là **rủi ro phụ thuộc nguồn dữ liệu cần lưu ý**, không phải đã chứng minh source leakage.

### 3.3. Cấu hình người/xe và trường hợp hai người

Script gán các head box vào bike box bằng một vùng bike được mở rộng; đây là phép ghép **xấp xỉ theo tâm box**, không phải track-level ground truth.

Kết quả:
- Train: khoảng **500 bike instances** được xét.
  - 289 xe không ghép được head box theo heuristic.
  - 104 cấu hình `driver_with`.
  - 34 cấu hình `driver_without`.
  - 31 cấu hình `driver_with × 2`.
  - 15 cấu hình `driver_without × 2`.
  - Các cấu hình có passenger được script ghi nhận là rất ít.
- Validation: khoảng **63 bike instances** được xét.
  - 32 xe không ghép được head box theo heuristic.
  - 15 cấu hình `driver_with`.
  - 7 cấu hình `driver_without`.
  - Có một số cấu hình driver + passenger, nhưng số lượng thấp.

Diễn giải:
- Các trường hợp hai người trên một xe, đặc biệt các kết hợp passenger và trạng thái mũ khác nhau, có độ phủ thấp trong dataset.
- Đây là một **rủi ro thiếu đại diện dữ liệu** đối với bài toán hai người trên xe.
- Các số trên phụ thuộc vùng mở rộng bike box và tâm head box; không được diễn giải thành số lượng người/xe đã xác minh bằng nhãn thủ công.
- Không dùng các con số này để tuyên bố model có tỷ lệ nhận diện passenger cụ thể.

### 3.4. Kích thước head box và độ sáng

Script tính median HSV V trên vùng crop trung tâm của head box; thống kê `V < 70` là tỷ lệ crop có median brightness dưới 70, không phải nhãn màu mũ.

| Class | Số head box | Median bbox width | Median bbox height | Tỷ lệ crop có median HSV V < 70 |
|---|---:|---:|---:|---:|
| 0 `driver_with_helmet` | 439 | 0.123 | 0.188 | **27%** |
| 3 `passenger_with_helemt` | 65 | 0.130 | 0.212 | **22%** |
| 5 `driver_without_helmet` | 129 | 0.096 | 0.167 | **12%** |
| 6 `passenger_without_helemt` | 74 | 0.101 | 0.170 | **28%** |

Diễn giải:
- Một phần đáng kể crop thuộc class 0, 3 và 6 có độ sáng trung vị thấp theo ngưỡng chẩn đoán này.
- Điều này hỗ trợ việc duy trì quy tắc labeling: **màu tối không phải tiêu chí để kết luận không đội mũ**.
- Đây không phải phép phân loại chất liệu/vỏ mũ; V thấp có thể chịu ảnh hưởng của ánh sáng, tóc, bóng đổ, nền hoặc chất lượng ảnh.
- Các class passenger 3 và 6 có số lượng box thấp hơn đáng kể so với class 0, cho thấy dữ liệu passenger cũng hạn chế.

---

## 4. Kết quả Phase A — Raw prediction diagnostics

### 4.1. Phạm vi và giới hạn phép đo

Script `phase_a_raw_predictions.py`:
- Dùng weights `helmet_best.pt`.
- `imgsz = 640`.
- Chạy mỗi 2 frame (`STEP = 2`).
- Confidence floor 0.05, nhưng chỉ ghi hàng class 5/6 có confidence `>= 0.45`.
- Không dùng tracker.
- Không chạy debouncing.
- Không gán nhãn tự động.
- Chỉ tìm box đối thủ class 0/3 nếu IoU với box class 5/6 `>= 0.5`.

**Cả ba video 5pc, 1pc, 7pc trong output đều là screen recording từ YouTube/Pinterest.** Vì vậy, kết quả có thể chịu ảnh hưởng của độ phân giải, crop, scaling, nén, moiré và các biến đổi do quay màn hình. Chúng không phải phép đo model thuần trên file nguồn.

### 4.2. Kết quả theo video

| Video | Kích thước | Số prediction class 5/6 >= 0.45 | Median confidence | Tỷ lệ >= 0.6 | Tỷ lệ >= 0.8 | Có rival class 0/3 đạt IoU >= 0.5 |
|---|---|---:|---:|---:|---:|---:|
| 5pc | 1920×1080 | **281** | **0.76** | 83% | 35% | 21% |
| 1pc | 2160×3840 | **78** | **0.58** | 45% | 13% | 35% |
| 7pc | 1920×1080 | **25** | **0.66** | 64% | 20% | 32% |

Khi có box rival class 0/3 đạt IoU >= 0.5:
- 5pc: median confidence rival **0.14**.
- 1pc: median confidence rival **0.33**.
- 7pc: median confidence rival **0.33**.

Median chiều cao box dự đoán sau quy đổi về ảnh 640:
- 5pc: khoảng **48 px**.
- 1pc: khoảng **34 px**.
- 7pc: khoảng **46 px**.

### 4.3. Diễn giải đúng mức

Có thể kết luận:
- Model tạo nhiều prediction class 5/6 trên các frame screen recording, đặc biệt 5pc.
- Một phần prediction có confidence cao và xuất hiện qua nhiều frame được lấy mẫu.
- Box dự đoán head ở 1pc nhỏ hơn hai video còn lại sau quy đổi về 640, nên kích thước đối tượng là một yếu tố cần lưu ý.
- Tần suất prediction và confidence cho thấy đây không đơn thuần là các detection confidence rất thấp.

**Không thể kết luận từ output này rằng có 281 false positives trên 5pc hoặc model chắc chắn phân loại sai.** Lý do:
- Không có ground-truth annotation theo frame/track cho ba video.
- Script không tracking, nên 281 hàng không phải 281 người hoặc 281 vi phạm duy nhất.
- `rival_helmet_conf = 0` nghĩa là không tìm thấy box class 0/3 chồng lấp IoU >= 0.5; không chứng minh không có dự đoán helmet ở vị trí khác.
- Confidence là độ tin cậy của prediction theo model, không phải xác suất prediction đúng.
- Video đầu vào đều là screen recording, nên không thể dùng kết quả để kết luận lỗi này có xảy ra tương tự trên file video gốc.

Vì vậy, cách ghi đúng trong báo cáo là:

> Raw-prediction diagnostics ghi nhận các prediction class 5/6 có confidence tương đối cao trên ba video quay màn hình. Tuy nhiên, do không có file video gốc và ground truth theo track, nhóm chưa thể xác nhận đây là false positives của model trên dữ liệu nguồn. Kết quả được sử dụng để định hướng điều tra, không được dùng làm metric đánh giá hoặc căn cứ xác nhận hiệu quả fine-tuning.

### 4.4. Các crop trong `diag_out`

ZIP có 28 ảnh crop chẩn đoán:
- 5pc: 16 crop.
- 1pc: 9 crop.
- 7pc: 3 crop.

Các crop này:
- Được trích từ screen recording.
- Chỉ phục vụ xem xét nguyên nhân prediction.
- Không phải full-frame.
- Không có bộ annotation toàn bộ object.
- Không thay thế video gốc.

Theo Labeling Policy APPROVED, **không đưa các crop này vào hard-negative training set**.

---

## 5. Đánh giá Labeling Policy và tình trạng dữ liệu hard-negative

Gói ZIP có `labeling_policy.md` trạng thái **APPROVED**, được cả Khoa, Phúc và Nguyên xác nhận ngày 18/11/2026.

Các điểm đã được khóa:
- Giữ nguyên 7 class và tên class trong weights, gồm typo `helemt`.
- Class 0/3/5/6 dùng head box; class 1 `bike` dùng bike box theo annotation gốc; class 2/4 dùng person box theo annotation gốc.
- Không nhìn thấy quai không tự động kết luận không đội mũ.
- Vỏ cứng rõ ràng vẫn được gán nhóm đội mũ, kể cả mũ màu tối.
- Ảnh không rõ cấu trúc mũ chuyển REVIEW.
- REVIEW không phải class train; ảnh REVIEW không được đưa vào training.
- Khoa là người phân xử cuối theo policy.
- Kiểm tra độc lập tối thiểu 20% frame; class agreement tối thiểu 90% trước phân xử.
- Fine-tune phải trộn ảnh mới đạt chuẩn với toàn bộ train gốc, giữ class mapping và không ghi đè weights cũ.

Tuy nhiên, package hiện tại **chưa có**:
- Ít nhất 30 full-frame từ file video gốc được chọn làm hard-negative.
- Bộ label YOLO đầy đủ cho tất cả box cần thiết trên các frame mới.
- Hard-negative manifest ghi nguồn video và frame index.
- Kết quả review độc lập 20%.
- Bảng tính class agreement 90%.
- Biên bản phân xử các object bất đồng.

Do đó, việc policy đã APPROVED **không đồng nghĩa bộ nhãn đã đạt nghiệm thu**.

---

## 6. Video manifest và holdout

Hiện còn thiếu video manifest.

Thông tin đã được nhóm báo cáo:
- Bảy video test hiện có đều là screen recording từ YouTube/Pinterest.
- Ba video được dùng trong raw diagnostics (`5pc`, `1pc`, `7pc`) cũng đều là screen recording.
- Chưa tải được file video nguồn.
- Chưa có holdout video gốc được phán quyết theo track.

Manifest nên ghi đủ bảy video với các trường:
- Tên file chính xác.
- Nguồn nền tảng hoặc URL/ID nếu còn biết.
- Trạng thái: `SCREEN_RECORDING`, `ORIGINAL_SOURCE_FOUND`, `SOURCE_UNKNOWN`.
- Có được dùng chẩn đoán hay không.
- Có được dùng train hay không.
- Có được dùng evaluation/holdout hay không.
- Lý do và người xác nhận.

Với trạng thái hiện tại:
- Bảy screen recording: chỉ dùng chẩn đoán/quan sát hạn chế.
- Không dùng làm train, validation, holdout hoặc căn cứ chọn model.
- Chưa xác định được video nào đủ điều kiện làm E2 holdout.

Nếu không tìm được video gốc độc lập trước cut-line, báo cáo phải ghi rõ E2 không có video holdout track-level. Validation set cũ có thể dùng cho kiểm tra regression trên ảnh, nhưng không thay thế được đánh giá video nguồn của lỗi mục tiêu.

---

## 7. Phán quyết cổng fine-tune

### 7.1. Bảng điều kiện

| Điều kiện đã chốt | Bằng chứng hiện có | Trạng thái |
|---|---|---|
| Phase A dataset audit đủ output | Có dataset audit output gồm dHash, source grouping, two-rider heuristic, brightness | **PASS** |
| Raw-prediction diagnostic đã chạy | Có CSV và output cho 5pc/1pc/7pc | **PASS có giới hạn** — chỉ screen recording, không có GT |
| Video manifest đủ bảy video | Chưa có manifest | **BLOCKED** |
| Có video gốc đủ điều kiện | Nhóm báo cáo hiện chỉ có screen recording, chưa có file nguồn | **FAIL hiện tại** |
| Có holdout độc lập, không trùng train | Chưa xác định | **FAIL hiện tại** |
| Ít nhất 30 full-frame gốc, ngoài holdout, gán đủ box | ZIP chỉ có diagnostic crops từ screen recording; không có hard-negative label set | **FAIL** |
| Review độc lập tối thiểu 20% | Chưa có label set/review log | **NOT MET** |
| Class agreement >= 90% trước phân xử | Chưa có số liệu review | **NOT MET** |
| Mọi bất đồng đã phân xử hoặc REVIEW | Chưa có annotation manifest | **NOT MET** |
| Train mới trộn toàn bộ train gốc và giữ 7 class | Đây là quy định trong policy, chưa có training set mới | **NOT STARTED** |
| Có dữ liệu độc lập đủ điều kiện đánh giá E2 | Chưa có video gốc/track-level holdout; val có 4 cặp dHash train–val cần kiểm tra | **NOT ESTABLISHED** |

### 7.2. Quyết định

# NO-GO — KHÔNG MỞ FINE-TUNE

Lý do quyết định:
1. Không có tối thiểu 30 full-frame gốc được gán nhãn và QA.
2. 28 crop chẩn đoán không đáp ứng yêu cầu full-frame và đều đến từ screen recording.
3. Không có holdout video gốc độc lập.
4. Không có kết quả review 20% hoặc class agreement 90%.
5. Không có ground truth theo track để xác nhận các prediction class 5/6 là false positives và đo mức cải thiện sau fine-tune.

**Không nên hạ tiêu chuẩn để kịp train.** Làm như vậy sẽ biến fine-tune thành một thử nghiệm không thể đánh giá đáng tin cậy, trái với Labeling Policy APPROVED.

### 7.3. Điều kiện duy nhất để xem xét mở lại

Chỉ xem xét lại trước cut-line nếu nhóm tìm được file video gốc và hoàn thành đồng thời:
- Manifest nguồn video đã xác minh.
- Hard-negative train frames tách biệt với holdout.
- Ít nhất 30 full-frame được gán đầy đủ box.
- Review độc lập tối thiểu 20%.
- Class agreement >= 90% trước phân xử.
- Tất cả bất đồng đã phân xử hoặc chuyển REVIEW.
- Holdout có ground truth theo track và đủ để đo FP/FN mục tiêu.

Nếu thiếu một điều kiện, giữ NO-GO.

---

## 8. Phase A status đề xuất cho tracker

Đề xuất ghi:

**`PHASE_A_DATA_AUDIT: COMPLETE`**

**`RAW_PREDICTION_DIAGNOSTICS: COMPLETE_WITH_SCREEN_RECORDING_LIMITATION`**

**`VIDEO_MANIFEST: PENDING — ALL 7 CURRENT TEST VIDEOS REPORTED AS SCREEN RECORDINGS; NO ORIGINAL HOLDOUT IDENTIFIED`**

**`FINE_TUNE_GATE: NO-GO`**

Không ghi:
- “Domain gap đã được chứng minh trên video gốc.”
- “281 false positives trên 5pc.”
- “Fine-tune đã đủ dữ liệu.”
- “E2 holdout đã sẵn sàng.”

## 9. Chất liệu chuyển vào báo cáo đồ án

### Phần 2 — Xác định dữ liệu
- Dataset có 368 train và 65 validation.
- Có 4 cặp train–val được dHash gắn cờ gần trùng, tương ứng 4 ảnh validation.
- Nguồn `BikesHelmets` chiếm 190 ảnh train và 16 ảnh validation theo heuristic tên file.
- Các cấu hình passenger/two-rider ít xuất hiện theo phép ghép head–bike xấp xỉ.
- Một số class head có crop tối, nhưng độ sáng không được dùng làm tiêu chí nhãn.

### Phần 3 — Các mô hình / phân tích baseline
- Mô hình hiện tại YOLOv8n.
- Cấu hình train và số liệu epoch 95.
- Phân tích giới hạn dataset, imbalance và near-duplicate.
- Raw predictions chỉ trình bày như chẩn đoán trên screen recording, không phải evaluation.

### Phần 4 — Lựa chọn model
- Dùng E1 để so sánh YOLOv8n baseline với YOLOv8s, YOLOv5n, RT-DETR.
- Không dùng các raw prediction trên screen recording làm bằng chứng chọn model.
- Không tuyên bố fine-tune cải thiện vì nhánh fine-tune bị đóng.

### Phần 5 — Kết luận và hạn chế
- Nêu rõ chưa có video gốc độc lập để xác nhận lỗi và đánh giá track-level.
- Nêu lý do không thực hiện fine-tune: thiếu dữ liệu hard-negative hợp lệ và holdout.
- Đây là quyết định kiểm soát chất lượng thực nghiệm, không phải bỏ qua vấn đề model.

---

## 10. Việc cần làm tiếp theo

1. **Khoa:** tạo video manifest cho đủ bảy video, ghi rõ `SCREEN_RECORDING`, nguồn đã biết và `NO_VALID_HOLDOUT`.
2. **Khoa:** lưu Phase A summary cùng output CSV gốc trong repo.
3. **Phúc:** đưa các kết quả dataset audit vào Phần 2; không viết các candidate dHash thành duplicate đã xác nhận.
4. **Cả nhóm:** giữ fine-tune ở trạng thái NO-GO, trừ khi có đủ file video gốc và hoàn tất mọi điều kiện trước cut-line.
5. **Cả ba:** tiếp tục E1 trên dataset gốc theo giao thức thống nhất; giữ nguyên weights baseline và không dùng hard-negative từ `diag_out`.

**Kết luận cuối:** Phase A đã cung cấp đủ bằng chứng để mô tả tình trạng dataset và các dấu hiệu cần điều tra. Nhưng bằng chứng hiện tại chưa đủ để xác nhận false positives trên dữ liệu video gốc hoặc mở fine-tuning một cách có thể đánh giá. Với deadline hiện tại, lựa chọn có trách nhiệm là đóng nhánh fine-tune và tập trung hoàn thành ba thử nghiệm E1.
