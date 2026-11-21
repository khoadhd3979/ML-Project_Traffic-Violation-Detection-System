# VIDEO MANIFEST — ML Traffic Violation Detection
Ngày lập: 18/11/2026 | Người lập: Khoa (Muse điền sẵn theo dữ kiện đã xác nhận, Khoa bổ sung link nguồn)

## Tuyên bố chung
Cả 7 video test của nhóm đều là **SCREEN_RECORDING** (quay màn hình từ YouTube/Pinterest).
**Không video nào được dùng làm train / eval / holdout.** Chỉ dùng cho demo và chẩn đoán định tính.
Holdout video gốc độc lập: **UNAVAILABLE** (không tồn tại file gốc ở bất kỳ đâu).

| # | File | Độ phân giải | Loại | Nguồn | Dùng cho |
|---|------|--------------|------|-------|----------|
| 1 | 1pc.mp4 | 2160x3840 | SCREEN_RECORDING | YouTube/Pinterest (Khoa bổ sung link) | Demo + chẩn đoán (đã chạy raw predictions: 78 lượt class 5/6) |
| 2 | 2pc.mp4 | (bổ sung) | SCREEN_RECORDING | (bổ sung link) | Demo |
| 3 | 3pc.mp4 | (bổ sung) | SCREEN_RECORDING | (bổ sung link) | Demo |
| 4 | 4pc.mp4 | (bổ sung) | SCREEN_RECORDING | (bổ sung link) | Demo |
| 5 | 5pc.mp4 | 1920x1080 | SCREEN_RECORDING | (bổ sung link) | Demo + chẩn đoán (281 lượt class 5/6) |
| 6 | 6pc.mp4 | (bổ sung) | SCREEN_RECORDING | (bổ sung link) | Demo |
| 7 | 7pc.mp4 | 1920x1080 | SCREEN_RECORDING | (bổ sung link) | Demo + chẩn đoán (25 lượt class 5/6) |

Ghi chú: một clip nguồn đã biết: https://youtu.be/wfaoFEWisbU (đoạn 14 giây tải bằng yt-dlp trong notebook) — Khoa xác nhận clip này ứng với video nào trong 7 video trên.

## Hệ quả
- E2 (fine-tune before/after): KHÔNG mở — fine-tune gate NO-GO (chốt 18/11/2026).
- Đánh giá model dựa trên: validation split của dataset gốc + bảng E1 (3 thuật toán).
- Mọi số liệu raw predictions trên 3 video trên chỉ là chẩn đoán định tính, KHÔNG phải false-positive count (không có ground truth theo track).
