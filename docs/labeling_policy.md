# Chính sách gán nhãn — mũ bảo hiểm vs nón vải/nón kết

**Trạng thái: APPROVED — cả 3 thành viên đã xác nhận tối 18/11/2026.** Đây là bản chính thức, đặt tại `docs/labeling_policy.md` trong repo và dùng cho buổi gán nhãn ngày 19/11/2026. Mọi thay đổi sau thời điểm duyệt phải được cả nhóm đồng ý lại.
**Quá trình duyệt:** DRAFT v1 (18/11) → v2 bổ sung 3 điều khoản Addendum → v3 bổ sung theo rà soát của cố vấn model (class 1 `bike`; quan hệ person box/head box theo annotation gốc; công thức class agreement) → APPROVED 18/11/2026.
**Thay đổi so với DRAFT v1:** bổ sung 3 điều khoản của Addendum — (a) không nhìn thấy quai không tự động kết luận không mũ; (b) ảnh không rõ cấu trúc mũ chuyển REVIEW; (c) REVIEW không phải class train, ảnh REVIEW bị loại khỏi training set; và quy định Khoa là người phân xử cuối.

## 1. Vì sao phải có file này trước khi gán nhãn

Buổi gán nhãn chiều 19/11 (cả 3 người cùng gán tối thiểu 30 frame hard-negative từ video gốc) sẽ tranh cãi lại từng ảnh nếu chưa có quy tắc chung. Ảnh train mới phải là **frame gốc đầy đủ (full-frame), gán nhãn mọi box** — crop chẩn đoán trong `diag_out` chỉ để xem, không dùng để train. Frame lấy từ video holdout bị cấm tuyệt đối.

## 2. Ngữ nghĩa class (theo weights đang dùng — KHÔNG đổi tên)

| Class | Ý nghĩa | Box |
|---|---|---|
| 0 `driver_with_helmet` | Tài xế đội mũ bảo hiểm | Box **đầu** |
| 1 `bike` | Xe máy | Box **xe**, theo đúng convention của dataset gốc |
| 3 `passenger_with_helemt` | Người ngồi sau đội mũ bảo hiểm | Box **đầu** |
| 5 `driver_without_helmet` | Tài xế không đội mũ bảo hiểm | Box **đầu** |
| 6 `passenger_without_helemt` | Người ngồi sau không đội mũ bảo hiểm | Box **đầu** |
| 2 `driver`, 4 `passenger` | Người (không có thông tin nón) | Box **cả người** |

Lưu ý: tên class trong weights có typo (`helemt`) — **không "sửa" tên khi fine-tune**, code sẽ lệch với `model.names`.

**Quan hệ person box và head box (bổ sung theo rà soát 18/11):**
- Phải duy trì **đúng convention annotation của dataset train gốc** cho mọi class, kể cả class 1 `bike` — không tự thay đổi cách vẽ box so với dữ liệu hiện tại.
- Không tự ý bỏ class person (2/4) hoặc head-state (0/3/5/6) nếu dataset gốc yêu cầu cả hai loại box cho cùng một người.
- **Trước buổi gán nhãn, cả nhóm phải mở một số ảnh + label gốc ra đối chiếu** để xác nhận cách biểu diễn một người trên xe (một người có đồng thời xuất hiện ở class person và head-state hay không). Chưa xác định được thì đối chiếu annotation gốc — **không tự suy luận**.

## 3. Quy tắc phân biệt

**Là mũ bảo hiểm (class 0/3):** vỏ cứng bao phủ đỉnh đầu + hai bên/sau đầu, dáng cứng tròn, thường có quai đeo. **Màu tối (đen, xanh đậm, xám) vẫn là mũ bảo hiểm** — không gán sang class 5/6 chỉ vì màu tối.

**Không phải mũ bảo hiểm (class 5/6):** nón kết, nón vải, mũ tai bèo — chất liệu mềm, có vành/lưỡi trai, thấy nếp vải.

**Điều khoản bổ sung (Addendum 18/11):**
- **Không nhìn thấy quai đeo KHÔNG tự động có nghĩa là không đội mũ.** Quai có thể bị góc chụp, tóc hoặc độ phân giải che mất. Phán quyết dựa trên cấu trúc vỏ: vỏ cứng rõ ràng → mũ bảo hiểm, kể cả khi không thấy quai.
- **Ảnh không rõ cấu trúc mũ** (mờ, quá nhỏ, che khuất nặng, không phân biệt được vỏ cứng với nón mềm) → đánh dấu `REVIEW`, không gán đoán.
- **`REVIEW` không phải là một class train.** Ảnh REVIEW bị loại khỏi training set cho tới khi được phân xử; ảnh không đủ bằng chứng sau phân xử vẫn bị loại hẳn.

## 4. Phân xử và quy trình buổi gán nhãn 19/11

1. Cả 3 người cùng gán nhãn; ảnh nào có bất đồng thì ghi lại để phân xử, không ép chốt tại chỗ.
2. **Khoa là người phân xử cuối** theo đúng policy này. Kết quả phân xử ghi vào manifest nhãn.
3. Điều kiện nghiệm thu bộ nhãn (cổng fine-tune): tối thiểu **30 frame gốc full-frame** đã gán nhãn toàn bộ box, không trùng holdout; **class agreement tối thiểu 90%** theo cách tính ở mục 4.1 dưới đây; mọi bất đồng phải được phân xử hoặc chuyển REVIEW.

### 4.1 Cách tính class agreement (chốt theo rà soát 18/11)

- Chọn **ngẫu nhiên tối thiểu 20%** số frame đã gán nhãn để một người thứ hai **review độc lập** (không xem nhãn của người gán trước khi review).
- Agreement được tính **trước khi Khoa phân xử** — không dùng kết quả sau phân xử để tính, vì như vậy tỷ lệ không phản ánh chất lượng gán nhãn độc lập.
- Đơn vị tính là **object**: hai người phải cùng xác định đó là một đối tượng tương ứng trong frame thì mới đưa vào mẫu số.
- Công thức: `Class agreement = Số object được thống nhất class / Tổng số object được review × 100%`
- Object bất đồng class → Khoa phân xử theo policy hoặc chuyển REVIEW.
- **Kiểm tra bounding box (vị trí/kích thước khớp hay không) là bước QA riêng**, không tính vào class agreement.
4. Tập train fine-tune = ảnh mới đạt chuẩn + **toàn bộ train gốc** (không train chỉ bằng ảnh mới — model sẽ quên các class khác). LR 0.001–0.002, weights mới lưu thư mục riêng, không ghi đè weights cũ.
5. Đánh giá model mới trên cả val cũ lẫn holdout độc lập (nếu có); không có holdout video gốc độc lập thì E2 chỉ báo cáo trên validation data và **không mở fine-tune nếu không có dữ liệu độc lập đủ điều kiện đánh giá**.

## 5. Cut-line liên quan

Nếu bộ nhãn không đạt điều kiện nghiệm thu ở mục 4.3 tối thứ Năm 19/11 thì **cắt fine-tune ngay** — viết báo cáo từ kết quả Phase A + baseline + bảng E1. Hạn nộp Chủ nhật 22/11/2026, freeze 20:00 thứ Bảy 21/11.

## 6. Xác nhận duyệt

| Thành viên | Xác nhận | Ngày |
|---|---|---|
| Khoa (ML Lead, người phân xử cuối) | ☑ Đã xác nhận | 18/11/2026 |
| Phúc (Backend & Data) | ☑ Đã xác nhận | 18/11/2026 |
| Nguyên (Frontend) | ☑ Đã xác nhận | 18/11/2026 |

*Xác nhận của cả nhóm được Khoa thông báo cho Muse lúc 21:06 ngày 18/11/2026; trạng thái APPROVED được áp ngay sau đó.*
