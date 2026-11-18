# KẾ HOẠCH SPRINT NÉN v3 — HẠN CHÓT CHỦ NHẬT 22/11/2026

**Thay thế** bản `ML-Traffic-Sprint-CN22Nov-2026-v2.md`. Lý do ra v3 (tối 18/11): (1) hai việc lớn của v2 đã xong sớm — **PR #2 đã merge** và **md5 weights đã xác minh xong**; (2) phát sinh **yêu cầu bắt buộc của giảng viên** mà v2 chưa có: tối thiểu **3 thử nghiệm với thuật toán khác** (≥ số thành viên), mỗi thử nghiệm có một ý riêng, kết quả là căn cứ chọn thuật toán cuối — đây là hạng mục chấm điểm tuần 12.

**Giữ nguyên từ v2:** hạn thật CN 22/11/2026; **freeze 20:00 Thứ 7, 21/11**; phạm vi chỉ không đội mũ + OCR biển số (`WRONG_LANE`, `RED_LIGHT`, Chatbot = future work); không deploy; UI đọc SQLite trực tiếp là bản nộp (FastAPI nối UI cắt); cut-line model trưa Thứ 7; các điểm vá C (fine-tune trộn train gốc, LR 0.001–0.002, chỉ dùng frame đầy đủ đã gán nhãn toàn bộ box) và D (holdout phải có phán quyết theo track, chỉ dùng **file video gốc** — 7 video quay màn hình **không** được dùng làm tập đánh giá).

**Thay đổi cốt lõi của v3 — thứ tự ưu tiên mới:**

1. Hạ cánh code + chạy thật end-to-end (nền tảng cho mọi việc khác).
2. **Track THỬ NGHIỆM THUẬT TOÁN (mới, BẮT BUỘC, không được cắt).**
3. Đóng Phase A (2 script audit còn thiếu output).
4. Gán nhãn hard-negative → fine-tune (**điều kiện**: chỉ làm khi 1–3 đã xong; vẫn chịu cut-line cũ).

---

## 1. Track thử nghiệm thuật toán (mới — bắt buộc)

**Yêu cầu của giảng viên (Nguyễn Thanh Phước):** có thêm thử nghiệm về các thuật toán khác, số thử nghiệm ít nhất bằng số thành viên (3), mỗi thử nghiệm có 1 ý riêng, dựa vào kết quả để nói lý do chọn thuật toán làm mô hình cuối. Báo cáo có % đóng góp của từng thành viên — **trung thực**.

**Bộ 3 thử nghiệm (mỗi người 1 thử nghiệm, đồng thời là phần đóng góp ghi trong báo cáo):**

| # | Thuật toán | Chủ trì | Ý của thử nghiệm |
|---|---|---|---|
| EXP-1 | **YOLOv8s** | Khoa | Tăng dung lượng model (n → s) có cải thiện các class yếu 3/5/6 không, và cái giá về tốc độ/kích thước là bao nhiêu? |
| EXP-2 | **YOLOv5n** | Phúc | Kiến trúc anchor-based thế hệ trước so với anchor-free của YOLOv8 trên cùng dữ liệu: độ chính xác và độ ổn định khác nhau thế nào? |
| EXP-3 | **RT-DETR** | Nguyên | Họ thuật toán khác hẳn (transformer, không NMS truyền thống) có hơn họ YOLO trên bài toán này không? |

Model hiện tại **YOLOv8n** (đã train 95 epoch, mAP50 0.795) đóng vai **baseline** trong bảng so sánh — không tính là 1 trong 3 thử nghiệm "thuật toán khác".

**Giao thức chung (bắt buộc để so sánh công bằng và bảo vệ được khi vấn đáp):**

- Dùng **dataset gốc** (368 train / 65 val, split hiện tại, `/content/data_helmet.yaml`) — không dùng dữ liệu hard-negative (đó là track fine-tune riêng).
- Cùng `imgsz = 640`, cùng seed 0, **50 epoch** cho cả 3 thử nghiệm (ghi rõ trong báo cáo: các lượt so sánh chạy 50 epoch; model baseline YOLOv8n đã train 95 epoch — khi kết luận phải nêu rõ khác biệt này, không trộn lẫn).
- Mỗi lượt lưu đủ: `args.yaml`, `results.csv`, weights, confusion matrix. Đặt tên thư mục: `exp1_yolov8s`, `exp2_yolov5n`, `exp3_rtdetr`.
- Chỉ số ghi cho từng thử nghiệm: Precision, Recall, mAP50, mAP50-95 (tổng) + Precision/Recall **theo class 3, 5, 6** + tốc độ suy luận (ms/ảnh hoặc FPS, đo trên cùng một mẫu frame, ghi rõ cách đo) + kích thước weights/số tham số nếu lấy được.
- Dataset nhỏ nên mỗi lượt train chỉ ~5–10 phút trên Colab T4 (lượt gốc 95 epoch tốn ~610 giây). Thời gian thật tốn ở khâu chạy eval + ghi chép, không phải train.
- **Không bịa số.** Lượt nào lỗi/chưa chạy thì ghi đúng trạng thái; thà báo cáo 3 thử nghiệm thật với số thật còn hơn bảng đẹp không có nguồn.
- Kết quả đổ vào **notebook so sánh** (Phúc chủ trì gộp, Thứ 6) → bảng tổng hợp dùng cho Phần 3 + Phần 4 của báo cáo.

**Nếu thiếu thời gian:** giảm epoch xuống 30 cho cả 3 (đồng đều) chứ **không bỏ bớt thử nghiệm** — số lượng thử nghiệm là yêu cầu cứng.

---

## 2. Lịch theo ngày (đã cập nhật trạng thái thật)

| Ngày | Khoa (ML) | Phúc (Backend/Data) | Nguyên (Frontend) |
|---|---|---|---|
| **T4 18/11 (tối nay)** | ✅ md5 weights — XONG. ✅ Merge PR #2 cùng Phúc — XONG (main `7e98fa42`). ⏳ Chạy nốt 2 script Phase A trên Colab, gửi output. ⏳ Chốt duyệt tiêu chí nhãn với nhóm (bản DRAFT đã có). ⏳ `git pull` main mới + chạy thử Streamlit thật | ✅ Merge PR #2 — XONG. ⏳ Kiểm tra DB/schema/reset/đường evidence trên main đã merge | ⏳ Chạy Streamlit trên main đã merge, liệt kê bug UI |
| **T5 19/11** | Sáng: đưa debouncing + 62 test từ local lên repo, chạy pytest trên repo thật. **Chạy EXP-1 (YOLOv8s) trên Colab.** Chiều: **cả 3 cùng gán nhãn** hard-negative (2–3 giờ); tối: chốt tập train fine-tune (nếu kịp) | **Chạy EXP-2 (YOLOv5n).** Gán nhãn. Kiểm thử FastAPI (không nối UI). Tối: bắt đầu viết Phần 1–2 báo cáo (mô tả + dữ liệu, dùng kết quả Phase A) | **Chạy EXP-3 (RT-DETR).** Gán nhãn. Sửa bug UI theo danh sách 18/11. **Đưa các notebook huấn luyện từ Colab/Drive vào repo** (yêu cầu Project 3) |
| **T6 20/11** | Sáng: fine-tune trên Colab **nếu đủ điều kiện** (xem cut-line). Chiều: đánh giá model fine-tune trên holdout (file gốc) + val cũ; phán quyết theo track | Gộp **notebook so sánh 4 model** (baseline + 3 thử nghiệm); kiểm thử tích hợp OCR → DB → UI; viết Phần 3–4 báo cáo (các mô hình + chọn mô hình) từ bảng so sánh | Hoàn thiện dashboard/thống kê; viết phần giao diện + khung slide; nếu còn thời gian: nối **một** luồng UI qua `APIClient` (xem mục 3), không kịp thì thôi |
| **T7 21/11** | **Trưa: quyết định model cuối (cut-line)**; tích hợp model được chọn; chạy lại 62 test. Chiều: viết Phần ML còn lại của báo cáo (kết quả, hạn chế — gồm vụ video quay màn hình) | Sáng: `pip freeze` chốt requirements (**xóa `sqlalchemy`** — không dùng ở đâu cả; ghi phiên bản đầy đủ theo template). Chiều: gộp báo cáo, điền **bảng % đóng góp trung thực** | Sáng: sửa bug cuối. Chiều: **quay video demo chính thức + bản dự phòng** (dùng file video gốc); gộp slide |
| **CN 22/11** | **NỘP / BẢO VỆ.** Chỉ sửa bug chặn demo. Không code mới, không train mới. | | |

**Tối T7 21/11, 20:00: FREEZE** — giữ nguyên.

---

## 3. Các điểm lệch với yêu cầu giảng viên — cách xử lý đã chốt

| Điểm | Hiện trạng (đã kiểm chứng) | Xử lý |
|---|---|---|
| Project 1 — API backend | ✅ FastAPI trên main, input mẫu → JSON | Demo trực tiếp tại trang `/docs` của FastAPI khi bảo vệ |
| Project 2 — app dùng API của mình | ⚠️ Streamlit đọc SQLite trực tiếp; `api_client.py` có nhưng **chưa được gọi ở đâu** | Ưu tiên thấp: Thứ 6 nếu còn thời gian, Nguyên + Phúc nối 1 luồng (ví dụ dashboard) qua APIClient. Không kịp → ghi trung thực trong báo cáo là hạn chế + hướng phát triển. **Không** để việc này ăn vào track thử nghiệm |
| Project 3 — notebook xây dựng model | ⚠️ Chưa có file `.ipynb` nào trong cây local/repo; notebook train nằm ở Colab/Drive | Nguyên đưa vào repo trong ngày Thứ 5 (kèm notebook của 3 thử nghiệm khi có) |
| Báo cáo 5 phần | `ml_report.md` đang rỗng | Viết theo đúng khung thầy: 1 Mô tả · 2 Xác định dữ liệu · 3 Các mô hình (đổ bảng thử nghiệm vào đây) · 4 Chọn mô hình · 5 Kết luận + bảng % đóng góp |
| requirements.txt | Chưa ghim phiên bản; có `sqlalchemy` thừa | Phúc xử lý sáng Thứ 7 khi freeze (xem lịch) |
| Deploy | Thầy không yêu cầu | Không làm — giữ nguyên quyết định cũ |

---

## 4. Cut-line (cập nhật)

- **Thử nghiệm thuật toán: không có cut-line loại bỏ** — chỉ được giảm epoch (50 → 30) khi thiếu thời gian.
- **Fine-tune:** giữ nguyên 2 cut-line của v2 — (a) gán nhãn chưa xong tối Thứ 5 → cắt fine-tune, phần ML của báo cáo dùng kết quả Phase A + baseline + bảng thử nghiệm (vẫn đủ chất liệu); (b) **trưa Thứ 7, 21/11**: model fine-tune không giảm dương tính giả ở `5pc`/`1pc` (file gốc) mà không làm tệ `2pc`/`3pc`/`4pc`/`6pc`/`7pc` và val cũ → bỏ, dùng weights hiện tại + debouncing.
- **Chọn thuật toán cuối:** mặc định YOLOv8n trừ khi bảng thử nghiệm cho thấy một thuật toán khác hơn rõ rệt **và** kịp tích hợp trước freeze — lý do chọn phải viết từ số liệu thật của bảng so sánh (Phần 4 báo cáo), chuẩn bị sẵn câu trả lời cho phần vấn đáp "vì sao chọn mô hình này, làm như thế nào".

## 5. Bàn giao phải xong tối Thứ 7, 21/11

Repo sạch + README chạy được từ đầu · pytest pass (ghi số test + commit) · **3 thử nghiệm có đủ artifact (args/results/weights/confusion matrix) + bảng so sánh** · notebook huấn luyện + notebook thử nghiệm đã vào repo · báo cáo 5 phần có số liệu đo thật + % đóng góp · slide · video demo + bản dự phòng · file ghi phiên bản weights đã dùng (tên + md5 `48be4ebb…`).

---

*Lập tối 18/11/2026, thay thế bản v2 cùng ngày. Thay đổi chính: thêm track thử nghiệm thuật toán bắt buộc (mỗi người 1 thử nghiệm), sắp lại ưu tiên (thử nghiệm trước fine-tune), ghi nhận PR #2 + md5 đã xong, thêm mục xử lý các điểm lệch Project 2/Project 3/requirements.*
