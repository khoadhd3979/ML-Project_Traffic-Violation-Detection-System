# Quy ước thư mục & đặt tên file bằng chứng

## Cấu trúc thư mục

```
data/evidence/
└── {YYYYMMDD}/                          # 1 thư mục con theo ngày phát hiện
    ├── {session_id}_{violation_id}_{rule_code}.jpg
    └── {session_id}_{violation_id}_{rule_code}.mp4   (nếu có clip)
```

Ví dụ thực tế:

```
data/evidence/20260925/12_305_RED_LIGHT.jpg
data/evidence/20260925/12_305_RED_LIGHT.mp4
data/evidence/20260925/12_306_SPEEDING.jpg
```

## Vì sao đặt tên như vậy

- **`{YYYYMMDD}/`**: tránh 1 thư mục chứa hàng trăm nghìn file, dễ dọn dẹp/archive theo ngày.
- **`{session_id}`**: biết ngay ảnh này sinh ra từ phiên xử lý nào (video/camera nào), phục vụ debug.
- **`{violation_id}`**: khớp trực tiếp với `violations.id` trong DB — không cần tra ngược, đảm bảo duy nhất.
- **`{rule_code}`**: đọc tên file là biết loại vi phạm ngay, không cần mở DB.

## Quy tắc bắt buộc

1. `rule_code` lấy từ `violation_rules.rule_code`, viết hoa, không dấu, không khoảng trắng (vd: `RED_LIGHT`, `SPEEDING`, `WRONG_LANE`).
2. Đường dẫn lưu trong `violations.evidence_image_path` / `evidence_video_clip_path` là đường dẫn **tương đối** tính từ thư mục gốc project (bắt đầu bằng `data/evidence/...`), không lưu đường dẫn tuyệt đối để dễ di chuyển giữa các máy.
3. Ảnh bằng chứng nên là frame gốc đã được vẽ bounding box + biển số (không chỉnh sửa/nén mất chi tiết biển số).
4. Dùng hàm `DatabaseManager.build_evidence_path()` trong `database_manager.py` để sinh đường dẫn thống nhất, tránh mỗi module tự ghép chuỗi khác nhau.
