# Chính sách gán nhãn — mũ bảo hiểm vs nón vải/nón kết (DRAFT)

**Trạng thái: BẢN DỰ THẢO — chưa được nhóm duyệt.** Chỉ dùng để nhóm chốt trước buổi gán nhãn chiều thứ Năm 19/11/2026. Khi nhóm xác nhận, đổi trạng thái thành APPROVED và đặt file vào `docs/labeling_policy.md` trong repo `khoadhd3979/ML-Project_Traffic-Violation-Detection-System`.
**Ngày soạn:** 18/11/2026 · **Nguồn:** bản dự thảo Muse đưa 18/11/2026 + báo cáo Phase A cùng ngày (spec chỉ chốt mũ tối màu = class 0/3; tiêu chí mũ vs nón vải là điểm mở duy nhất còn lại của Phase A).

## 1. Vì sao phải có file này trước khi gán nhãn

Buổi gán nhãn chiều 19/11 (cả 3 người cùng gán 30–40 ảnh hard-negative từ `5pc`/`1pc`) sẽ tranh cãi lại từng ảnh nếu chưa có quy tắc chung. Quy tắc phải có trên giấy trước khi bất kỳ ai gán nhãn dữ liệu mới. Ảnh train mới phải là **frame đầy đủ, gán nhãn mọi box** — crop chẩn đoán trong `diag_out` chỉ để xem, không dùng để train.

## 2. Ngữ nghĩa class (theo weights đang dùng — KHÔNG đổi tên)

| Class | Ý nghĩa | Box |
|---|---|---|
| 0 `driver_with_helmet` | Tài xế đội mũ bảo hiểm | Box **đầu** |
| 3 `passenger_with_helemt` | Người ngồi sau đội mũ bảo hiểm | Box **đầu** |
| 5 `driver_without_helmet` | Tài xế không đội mũ bảo hiểm | Box **đầu** |
| 6 `passenger_without_helemt` | Người ngồi sau không đội mũ bảo hiểm | Box **đầu** |
| 2 `driver`, 4 `passenger` | Người (không có thông tin nón) | Box **cả người** |

Lưu ý: tên class trong weights có typo (`helemt`) — **không "sửa" tên khi fine-tune**, code sẽ lệch với `model.names`.

## 3. Quy tắc phân biệt

**Là mũ bảo hiểm (class 0/3):** vỏ cứng bao phủ đỉnh đầu + hai bên/sau đầu, dáng cứng tròn, thường có quai đeo. **Màu tối (đen, xanh đậm, xám) vẫn là mũ bảo hiểm** — điểm này spec đã chốt, không gán sang class 5/6 chỉ vì màu tối.

**Không phải mũ bảo hiểm (class 5/6):** nón kết, nón vải, mũ tai bèo — chất liệu mềm, có vành/lưỡi trai, thấy nếp vải.

**Ảnh không đủ rõ hoặc bị che khuất:** đánh dấu `REVIEW`, **không gán đoán**, loại khỏi tập train cho tới khi có người thứ hai phân xử.

## 4. Quy trình gán nhãn buổi 19/11

1. Cả 3 người cùng gán nhãn trong 2–3 giờ; ảnh nào 2/3 người không đồng ý thì đánh `REVIEW`, không ép chốt trong buổi.
2. Tập train fine-tune = ảnh mới đã gán nhãn + **toàn bộ train gốc** (không train chỉ bằng 30–40 ảnh mới — model sẽ quên các class khác). LR 0.001–0.002.
3. Đánh giá model mới trên cả val cũ lẫn holdout mới (video holdout tuyệt đối không đưa vào tập train).

## 5. Cut-line liên quan

Nếu tập train chưa xong tối thứ Năm 19/11 thì **cắt fine-tune ngay** — viết báo cáo từ kết quả Phase A + baseline thay vì để việc train ăn vào ngày viết báo cáo (theo sprint v2, hạn nộp Chủ nhật 22/11/2026, freeze 20h thứ Bảy 21/11).
