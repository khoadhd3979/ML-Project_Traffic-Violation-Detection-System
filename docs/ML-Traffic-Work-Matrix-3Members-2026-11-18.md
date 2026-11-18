# MA TRẬN PHÂN CÔNG VIỆC 3 THÀNH VIÊN — TRAFFIC VIOLATION DETECTION

**Phạm vi:** tối 18/11/2026 → bảo vệ CN 22/11/2026. **Freeze: 20:00 Thứ 7, 21/11.**
**Cơ sở:** Sprint v3 + kế hoạch phản biện của Claude phía model (đã tiếp thu các điểm: rủi ro YOLOv8s, smoke test trước khi train dài, bảo vệ holdout, chạy pytest thật, tách E1/E2, đo FPS cùng điều kiện, giảm tải cho Khoa).
**Nguyên tắc cân bằng:** mỗi người = 1 thử nghiệm thuật toán + 1 mảng sản phẩm chính + phần việc chung (gán nhãn, báo cáo). Khoa giữ vai trò *duyệt chất lượng ML* thay vì tự làm mọi việc ML.

**Tải việc chính theo người (để tự đối chiếu khi thấy lệch):**
- **Khoa (ML Lead):** Phase A · EXP-1 · chủ trì gán nhãn + xử lý ca `REVIEW` · fine-tune (nếu mở) + E2 · chốt model + release.
- **Phúc (Backend/Data):** DB/API · EXP-2 · notebook tổng hợp E1 + bảng so sánh · chủ trì gộp báo cáo (Phần 1–4) · requirements freeze.
- **Nguyên (Frontend):** UI · EXP-3 · đưa notebook vào repo · luồng APIClient · slide + video demo.

---

## TỐI THỨ 4, 18/11 — Ổn định nền tảng + chốt các quyết định đang chặn

| Thời gian | Việc cần làm | Người | Mục tiêu | Tiêu chí hoàn thành | Test lại | Gợi ý thêm |
|---|---|---|---|---|---|---|
| 20:00–20:30 | Họp chốt 4 quyết định: (1) duyệt labeling policy DRAFT; (2) lập danh sách video nào là **file gốc**, video nào là quay màn hình; (3) chỉ định **holdout** = video gốc không được trích frame train; (4) chốt mỗi người dùng Colab riêng hay chung GPU | Cả 3, Khoa chủ trì | Mở khóa mọi việc ngày 19/11 | Có văn bản chốt (ghi vào nhóm chat + file manifest video); labeling policy chuyển trạng thái APPROVED | — | Viết manifest ngay trong tối nay: tên video · nguồn · gốc/quay màn hình · dùng cho train/eval/holdout — sau này E2 tra theo manifest này |
| 20:00–20:30 | Nhắn hỏi giảng viên 2 câu: (1) YOLOv8s (cùng họ YOLOv8, khác kích thước) có được tính là "thuật toán khác" không; (2) Project 2 có bắt buộc UI phải gọi API không | Khoa | Loại rủi ro mất điểm lớn nhất trước khi chạy thử nghiệm | Đã gửi; hạn có câu trả lời: sáng 19/11 | — | Nếu thầy không trả lời kịp: vẫn chạy EXP-1 như kế hoạch (nó có ý nghĩa thực nghiệm riêng) và chuẩn bị sẵn phương án B ở cuối bảng này |
| 20:30–21:30 | `git pull` main `7e98fa42`, chạy Streamlit thật, chạy 1 file video gốc qua luồng đầy đủ | Khoa | Chứng minh hệ thống chạy end-to-end trên main đã merge | App lên; model load đúng weights; có evidence tạo ra; DB có bản ghi vi phạm tương ứng | Chạy lại lần 2 với video khác để loại trừ ăn may | Ghi lại lỗi gặp phải (nếu có) thành danh sách, không sửa lan man trong tối nay trừ lỗi chặn chạy |
| 20:30–21:30 | Smoke test DB/API: schema tự tạo, seed chạy, reset `XACNHAN`, FastAPI nhận input mẫu trả JSON, mở `/docs` | Phúc | Xác nhận phần backend của PR #2 chạy thật, không chỉ merge được | 5 mục đều PASS, ghi kết quả từng mục | Tạo vi phạm thử → kiểm tra API trả đúng bản ghi đó | Kết quả smoke này là tư liệu viết Phần 1 báo cáo — lưu lại dạng bảng PASS/FAIL |
| 20:30–21:30 | Smoke test UI: dashboard đọc dữ liệu, danh sách vi phạm + ảnh evidence hiển thị; lập danh sách bug UI | Nguyên | Có danh sách bug rõ ràng cho ngày 19–20/11 | Bug list có mức độ (chặn demo / không chặn) | Mở lại app lần 2 kiểm tra dữ liệu mới từ lượt chạy của Khoa có hiện không | Bug "chặn demo" mới sửa ngay trong tuần; bug trang trí để sau freeze nếu thiếu giờ |
| 21:30–22:30 | Chạy bổ sung 2 script Phase A: audit dataset (mục [1] dHash, [2] nhóm nguồn, [3] hai người/xe, [4] độ sáng) + raw predictions trên video gốc chẩn đoán | Khoa | Đủ output để đóng Phase A vào tối 19/11 | Có đủ output 4 mục + thư mục `diag_out`/`raw_class56.csv`; gửi toàn bộ cho Claude phía model | Đối chiếu số đếm 368/65 và phân bố class với số liệu đã biết — lệch là dừng và báo ngay | Chạy script nào ra script đó, lưu output dạng file (không chụp màn hình) để còn đối chiếu về sau |
| 21:30–22:30 | Dựng khung notebook tổng hợp E1 (bảng so sánh trống) + `experiment_manifest` (giao thức chung: dataset gốc, split cố định, imgsz 640, seed 0, 50 epoch) | Phúc | Cả 3 thử nghiệm mai chạy cùng một giao thức, số liệu ghép được vào 1 bảng | Có file khung; 3 chủ trì thử nghiệm đã đọc và thống nhất | Phúc tự kiểm: bảng có đủ cột P/R/mAP50/mAP50-95 + P/R class 3/5/6 + latency + size + epoch + checkpoint | Đây là việc "âm thầm cứu cả nhóm": lệch giao thức là bảng E1 vô giá trị khi vấn đáp |
| 21:30–22:30 | Tạo thư mục notebook trong repo, đưa notebook train YOLOv8n từ Colab/Drive vào; chốt variant RT-DETR (đề xuất `rtdetr-l` qua ultralytics) + ghi phiên bản thư viện | Nguyên | Đạt yêu cầu Project 3; EXP-3 mai chạy ngay không mất giờ chọn variant | Notebook train đã vào repo; variant + version được ghi vào manifest của Phúc | Mở notebook kiểm tra còn nguyên cell train + đường dẫn data | Đồng thời gửi cho Claude phía model: notebook train + `data_helmet.yaml` (nó đang cần để soạn script thử nghiệm khớp API thật) |

---

## THỨ 5, 19/11 — Sáng: hạ cánh code + 3 thử nghiệm · Chiều: gán nhãn + đóng Phase A

| Thời gian | Việc cần làm | Người | Mục tiêu | Tiêu chí hoàn thành | Test lại | Gợi ý thêm |
|---|---|---|---|---|---|---|
| 07:30–09:00 | Đưa debouncing + 2 file test từ local lên repo (branch → PR), **chạy pytest thật trên repo** | Khoa | Bằng chứng test chạy trên repo thật, thay cho suy luận từ `.pytest_cache` | Có output pytest thật (ghi số pass/fail + commit); PR được merge vào main | Chạy lại pytest lần 2 sau merge trên main sạch | Chưa có output chạy thật thì **không** ghi "62/62 PASS" vào bất cứ đâu — đây là điểm Claude đã cảnh báo đúng |
| 09:00–09:45 | Smoke test từng thử nghiệm: load model + YAML, chạy thử **1 epoch**, chạy 1 lần validation | Cả 3 (mỗi người thử nghiệm của mình) | Ước lượng thời gian thật, phát hiện lỗi môi trường/API sớm | Cả 3 đều qua 1 epoch + 1 validation; có thời gian/epoch đo thật | Kiểm tra model nhận đủ 7 class, class mapping không đổi | Dựa trên số đo này quyết định giữ 50 epoch hay giảm đồng đều xuống 30 — quyết trước 10:00, không quyết giữa chừng từng model |
| 09:45–13:00 | Chạy full EXP-1 **YOLOv8s** | Khoa | Trả lời: tăng dung lượng n→s có cứu class 3/5/6 không, giá tốc độ/size bao nhiêu | Thư mục `exp1_yolov8s/` đủ: args.yaml, results.csv, weights, confusion matrix, ghi GPU + thời gian thật | Validation lại từ weights đã lưu (không dùng số trong log chạy dở) | Nếu giảng viên trả lời YOLOv8s KHÔNG được tính: xem phương án B cuối bảng — quyết trước giờ chạy full, đừng chạy xong mới đổi |
| 09:45–13:00 | Chạy full EXP-2 **YOLOv5n** | Phúc | So sánh kiến trúc anchor-based (YOLOv5) với anchor-free (YOLOv8) trên cùng dữ liệu | Thư mục `exp2_yolov5n/` đủ artifact như EXP-1 | Tương tự EXP-1 | Ghi rõ cách cài/bản YOLOv5 đã dùng — thầy hỏi lý thuyết sẽ đào vào khác biệt anchor-based vs anchor-free |
| 09:45–13:00 | Chạy full EXP-3 **RT-DETR** | Nguyên | Kiểm tra họ transformer có lợi thế gì trên bài toán này | Thư mục `exp3_rtdetr/` đủ artifact như EXP-1 | Tương tự EXP-1 | RT-DETR nặng nhất trong 3 — nếu smoke test cho thấy quá chậm, báo ngay để cả nhóm quyết giảm epoch đồng đều thay vì để EXP-3 trễ một mình |
| 13:00–16:30 | **Buổi gán nhãn hard-negative chung**: trích frame từ video gốc (theo manifest, **không** lấy từ holdout), gán nhãn toàn bộ box trong frame | Cả 3; Khoa chủ trì chất lượng + xử lý ca `REVIEW` | Có tập hard-negative sạch, đủ đa dạng (mũ tối màu, hai người/xe, che khuất, nhiều xe) | 30–40 frame đã gán đủ box (xe + người + đầu), có manifest ghi video nguồn + frame index; ca mờ → `REVIEW`, không đoán | Kiểm tra chéo: mỗi người soát một phần nhãn của người khác ngay trong buổi | Thà 30 frame chắc còn hơn 40 frame ẩu — chất lượng nhãn quyết định fine-tune có đáng chạy hay không |
| 16:30–18:00 | Điền kết quả 3 thử nghiệm vào bảng E1; đo latency của 4 model (gồm baseline) **trên cùng một máy, cùng tập frame** | Phúc chủ trì; Khoa hỗ trợ đo baseline | Bảng E1 có số thật, đo cùng điều kiện | Bảng E1 đủ 4 dòng × đủ cột chỉ số; ghi rõ máy + cách đo latency | Phúc đối chiếu số trong bảng với results.csv gốc của từng thử nghiệm | Không lấy FPS từ 3 Colab runtime khác nhau xếp chung — đo lại tập trung tại một nơi như Claude đã nhắc |
| 16:30–18:00 | Hoàn tất kiểm tra chéo nhãn; đưa toàn bộ notebook (train + 3 thử nghiệm) vào repo | Nguyên | Nhãn sẵn sàng cho quyết định fine-tune; Project 3 hoàn chỉnh | 100% frame hard-negative đã qua kiểm tra chéo; notebook có trên repo | Nguyên mở repo kiểm tra file hiển thị đúng | Frame nào 2 người soát không thống nhất → Khoa quyết cuối cùng theo labeling policy, ghi lại quyết định |
| 18:00–20:00 | Tổng hợp **báo cáo Phase A hoàn chỉnh** (audit + raw predictions + kết luận giả thuyết) gửi Khoa duyệt đóng | Khoa | Đóng Phase A trong tối 19/11, mở cổng quyết định fine-tune | Báo cáo nêu rõ: điều gì đã xác nhận, điều gì còn là giả thuyết (domain gap), giới hạn của split nếu có near-duplicate | Đối chiếu mọi con số trong báo cáo với file output gốc | Nếu Phase A phát hiện leakage gần-trùng nặng: ghi thẳng vào báo cáo là hạn chế của evaluation — trung thực điểm này còn quý hơn số đẹp |

**Cổng quyết định cuối ngày 19/11 (cả nhóm, Khoa chốt):** Phase A đóng? Nhãn đạt chuẩn? → Mở hay cắt fine-tune ngày 20/11. Không đạt một trong hai → cắt fine-tune ngay tại đây, không chờ tới 20/11.

---

## THỨ 6, 20/11 — Fine-tune (có điều kiện) + E1/E2 + báo cáo + APIClient

| Thời gian | Việc cần làm | Người | Mục tiêu | Tiêu chí hoàn thành | Test lại | Gợi ý thêm |
|---|---|---|---|---|---|---|
| 08:00–11:00 | **Nếu đủ điều kiện:** fine-tune từ `helmet_best.pt` (đã xác minh hash), trộn hard-negative + train gốc, LR 0.001–0.002, weights ra thư mục riêng. **Nếu không đủ điều kiện:** chuyển ngay sang viết phân tích Phase A cho báo cáo | Khoa | Có model ứng viên fine-tune, hoặc đóng nhánh fine-tune dứt khoát | Có weights mới + args/results đầy đủ; hoặc có văn bản chốt cắt fine-tune kèm lý do | Kiểm tra `model.names` của weights mới khớp 7 class cũ | Không train lại vòng 2 dù kết quả chưa ưng — lịch không còn chỗ, cut-line đã quy định |
| 08:00–11:00 | Viết Phần 3 + Phần 4 báo cáo từ bảng E1 (các mô hình + chọn mô hình), gồm đoạn trả lời mẫu "vì sao chọn thuật toán này" | Phúc | Xong phần xương sống của báo cáo trước khi có kết quả fine-tune | Bản thảo Phần 3–4 điền số thật từ E1; lý do chọn model viết từ số liệu, không viết trước kết quả | Khoa đọc kiểm tra lại mọi con số + cách diễn giải | Viết theo 3 tình huống Claude đã phác (baseline thắng / model khác thắng rõ / không chênh lệch rõ) — chọn nhánh đúng với số thật |
| 08:00–11:00 | Hoàn thiện dashboard/thống kê UI; sửa nốt bug "chặn demo" từ danh sách 18/11 | Nguyên | UI sẵn sàng cho demo | Bug chặn demo = 0; dashboard hiển thị đúng dữ liệu DB | Chạy app với DB có dữ liệu vi phạm mẫu, kiểm tra số liệu khớp DB | Chụp sẵn ảnh màn hình các trạng thái chính của app — dùng cho slide và phòng khi demo trực tiếp lỗi |
| 11:00–15:00 | Đánh giá E2 (nếu có fine-tune): chạy model cũ + mới trên val cũ + holdout video gốc, **phán quyết theo track**; đo 2 lớp model-level (raw) và system-level (sau debouncing) | Khoa chủ trì; Phúc phụ phán quyết track (kiểm tra chéo) | Trả lời: fine-tune có giảm FP ở `5pc`/`1pc` mà không làm tệ các video hồi quy không | Bảng E2 đủ: FP/FN trước–sau theo từng video + val; ghi rõ video nào là file gốc | Chạy lại 1 video hồi quy bằng weights cũ để đối chứng số liệu không lệch do môi trường | Video nào chỉ có bản quay màn hình → loại khỏi bảng E2, ghi chú rõ; đây là điểm dễ bị thầy bắt nếu lẫn vào |
| 11:00–15:00 | Kiểm thử tích hợp OCR → DB → UI trên main; viết Phần 1 + Phần 2 báo cáo (mô tả + dữ liệu) | Phúc | Hệ thống tích hợp được kiểm chứng; nửa đầu báo cáo xong | Có bảng kết quả kiểm thử tích hợp; Phần 1–2 hoàn chỉnh | Tạo 1 vi phạm từ video, truy vết tới UI hiển thị đúng biển số OCR | Phần 2 dùng luôn số liệu Phase A (368/65, phân bố class, imbalance 7.7) — không viết lại số khác |
| 13:00–15:00 | **Khung giờ cố định** nối 1 luồng UI qua `APIClient` (đề xuất: dashboard đọc thống kê qua API) — giới hạn đúng 2 giờ, xong hay không cũng dừng | Nguyên + Phúc | Đạt đúng mô tả Project 2 của giảng viên ở ít nhất 1 luồng | Luồng đó chạy qua FastAPI thật; các màn còn lại giữ đọc SQLite và mô tả trung thực trong báo cáo | Tắt FastAPI → luồng đó báo lỗi rõ ràng (chứng minh có gọi API thật); bật lại → chạy | Đây là nâng cấp so với sprint v3 theo phản biện của Claude (rủi ro 7) — nhưng giữ timebox để không ăn vào báo cáo/demo |
| 15:00–19:00 | Dựng khung slide bảo vệ (cấu trúc 5 phần báo cáo + bảng E1 + sơ đồ hệ thống) | Nguyên chủ trì; Khoa + Phúc cung cấp nội dung phần mình | Slide có khung + số liệu thật, chỉ chờ model cuối | Khung slide hoàn chỉnh, các ô chờ model cuối được đánh dấu rõ | Cả nhóm đọc lướt kiểm tra số liệu khớp báo cáo | Làm slide từ số thật đã có (E1, Phase A) trước; phần model cuối để 21/11 điền sau |
| 15:00–19:00 | Rà soát toàn bộ báo cáo nháp: số liệu, tên class (giữ typo `helemt` như weights thật khi trích dẫn), hash weights | Khoa | Báo cáo không còn số mâu thuẫn giữa các phần | Danh sách chỗ sửa đã gửi cho Phúc gộp | Đối chiếu chéo: mAP50 baseline 0.795 chỉ xuất hiện đúng ngữ cảnh (train 95 epoch) | Tên class trong weights là `passenger_with_helemt` — báo cáo trích đúng tên thật, có ghi chú typo, đừng tự "sửa" thành helmet |

---

## THỨ 7, 21/11 — Chốt model trưa · Freeze 20:00

| Thời gian | Việc cần làm | Người | Mục tiêu | Tiêu chí hoàn thành | Test lại | Gợi ý thêm |
|---|---|---|---|---|---|---|
| 08:00–10:00 | Tập hợp E1 + E2 + holdout, soạn đề xuất model cuối theo 6 tiêu chí (hiệu quả tổng · class 3/5/6 · FP/FN theo track · tốc độ/size · tích hợp · tái lập) | Khoa | Có đề xuất chọn model kèm bằng chứng, sẵn sàng họp chốt | Văn bản đề xuất 1 trang: model chọn + lý do bằng số + rủi ro | Tự kiểm: mọi lý do đều trỏ được về một bảng/artifact cụ thể | Nếu model khác thắng rõ nhưng chưa tích hợp: vẫn phải thử tích hợp trong sáng nay, không quyết trên giấy |
| 10:00–12:00 | **Họp chốt model cuối (CUT-LINE trưa)** — cả nhóm quyết theo đề xuất + số liệu | Cả 3 | Một model duy nhất cho bản nộp | Quyết định ghi thành văn bản (model, weights, lý do) | — | Nguyên tắc: chênh lệch không rõ ràng → giữ model đã tích hợp ổn định; không khẳng định model giữ lại là "chính xác nhất" nếu số không nói vậy |
| 12:00–15:00 | Tích hợp model cuối vào app: cập nhật weights/đường dẫn, xác minh hash, chạy lại bộ test, chạy end-to-end 1 video gốc | Khoa | Bản nộp chạy model đã chốt, có bằng chứng | Commit tích hợp; hash weights khớp bản ghi; test pass (output thật); video gốc chạy qua có evidence + DB | Chạy lại pytest + mở UI kiểm tra hiển thị | Ghi file phiên bản weights (tên + md5) vào repo — đây là mục trong checklist bàn giao |
| 12:00–15:00 | `pip freeze` chốt requirements: ghim phiên bản đã kiểm thử, **xóa `sqlalchemy`**, loại package rác của môi trường Colab; dọn README (cách chạy từ đầu, mapping 7 class) | Phúc | Người khác clone về chạy được theo README | requirements sạch + có phiên bản; README chạy được từng bước | Phúc tự làm sạch môi trường thử cài lại theo requirements (hoặc ít nhất rà từng dòng) | Template của thầy nhấn mạnh requirements có phiên bản để tái lập — điểm này dễ ăn điểm quy trình, đừng bỏ |
| 12:00–15:00 | Gộp báo cáo hoàn chỉnh 5 phần + **bảng % đóng góp trung thực** (theo đúng ma trận này: ai làm việc gì ghi việc đó) | Phúc chủ trì; cả 3 xác nhận phần % của mình | Báo cáo nộp được | Đủ 5 phần + % đóng góp; số liệu khớp artifact | Khoa + Nguyên đọc chéo toàn văn lần cuối | % đóng góp lấy theo đầu việc thật trong ma trận này — thầy đã dặn "trung thực", đừng chia đều 33/33/33 cho đẹp |
| 15:00–19:30 | Quay **video demo chính thức + bản dự phòng** (chạy trên file video gốc); hoàn thiện slide | Nguyên | Demo không phụ thuộc vào may rủi ngày bảo vệ | 2 video demo (chính + dự phòng) xem lại đạt: rõ luồng detect → evidence → DB → UI | Phát lại video trên máy khác/máy sẽ dùng bảo vệ | Quay cả cảnh mở FastAPI `/docs` (bằng chứng Project 1) và cảnh dashboard qua APIClient (bằng chứng Project 2) |
| 15:00–19:30 | Hoàn thiện phần ML của báo cáo (kết quả, hạn chế — gồm domain gap, dataset nhỏ, vụ video quay màn hình); hỗ trợ slide phần model | Khoa | Phần ML trung thực, đủ chất liệu vấn đáp | Phần ML xong, hạn chế ghi rõ ràng | Đối chiếu lần cuối với báo cáo Phase A + bảng E1/E2 | Chuẩn bị sẵn 3 câu trả lời vấn đáp: vì sao chọn model này · lỗi mũ tối màu xử lý thế nào · giới hạn lớn nhất của hệ thống |
| **20:00** | **FREEZE** | Cả 3 | Khóa sản phẩm | Sau giờ này chỉ sửa bug chặn demo | — | Đóng gói thư mục nộp + kiểm tra mở được mọi file (báo cáo, slide, video, notebook) |

## CN 22/11 — Nộp / bảo vệ

| Thời gian | Việc cần làm | Người | Mục tiêu | Tiêu chí hoàn thành | Test lại | Gợi ý thêm |
|---|---|---|---|---|---|---|
| Trước giờ bảo vệ | Chuẩn bị môi trường demo: DB đã seed, weights đúng bản chốt, video gốc + video demo dự phòng sẵn sàng, mở sẵn `/docs` | Cả 3 (Nguyên lo máy demo, Khoa lo model/DB, Phúc lo API) | Vào phòng là chạy được ngay | Checklist chuẩn bị tích đủ mục | Chạy thử toàn bộ luồng demo 1 lần ngay tại chỗ | Mang theo file hash weights + bảng E1 in sẵn — thầy hỏi số liệu là có ngay, không phải mở tìm |
| Trong bảo vệ | Chỉ xử lý bug chặn demo nếu phát sinh | Cả 3 | Bảo vệ trọn vẹn | — | — | Không train, không sửa code lớn, không đổi weights dưới bất kỳ áp lực nào |

---

## PHƯƠNG ÁN B (chỉ kích hoạt khi có điều kiện tương ứng)

| Điều kiện | Hành động thay thế | Người quyết |
|---|---|---|
| Giảng viên trả lời YOLOv8s **không** được tính là thuật toán khác | EXP-1 đổi sang **Faster R-CNN (torchvision, ResNet50-FPN)** — chấp nhận chi phí setup (chuyển nhãn YOLO → COCO format, vòng train riêng); EXP-2/EXP-3 giữ nguyên. Nếu setup quá 2 giờ không xong: ghi trạng thái thật + giữ YOLOv8s làm thử nghiệm quy mô, trình bày trung thực khi vấn đáp | Khoa quyết sáng 19/11 trước giờ chạy full |
| Cả nhóm dùng chung 1 GPU Colab | Chạy thử nghiệm **tuần tự** Khoa → Phúc → Nguyên theo thứ tự EXP-1 → 2 → 3; người chờ thì làm gán nhãn/Phase A song song; latency vẫn đo tập trung 1 máy chiều 19/11 | Khoa điều phối sáng 19/11 |
| Smoke test cho thấy không đủ giờ cho 50 epoch | Giảm **đồng đều cả 3** xuống 30 epoch, ghi rõ điều kiện vào báo cáo; tuyệt đối không giảm riêng lẻ từng model | Cả nhóm quyết trước 10:00 ngày 19/11 |
| Một thử nghiệm lỗi không sửa kịp | Ghi trạng thái + nguyên nhân + log vào báo cáo; chạy lại trong khung dự phòng chiều 20/11 nếu còn chỗ; **không** điền số thay thế | Chủ trì thử nghiệm đó báo ngay khi phát hiện |
| Tối 19/11 chưa đóng Phase A hoặc nhãn chưa đạt | Cắt fine-tune ngay; Khoa chuyển sang viết phân tích Phase A + E1 cho báo cáo (đã có sẵn trong lịch 20/11) | Cả nhóm tại cổng quyết định 19/11 |

---

*Ma trận v1, lập tối 18/11/2026 dựa trên Sprint v3 và kế hoạch phản biện của Claude phía model. Nơi ma trận này khác Sprint v3 (khung giờ APIClient cố định thay vì "nếu thừa thời gian", Phase A đóng tối 19/11, thêm bước smoke test 1 epoch và đo latency tập trung), ma trận này là bản áp dụng.*
