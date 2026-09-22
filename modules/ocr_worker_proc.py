# Chạy trong process riêng biệt (multiprocessing) để tránh bị GIL của PaddleOCR
# block main thread hiển thị camera.
#
# Phiên bản này phục vụ nhiều xe cùng lúc (tracking đa đối tượng): mỗi item
# trong input_q là một tuple (track_id, plate_crop), và mỗi kết quả trả về
# result_q là tuple (track_id, text) để bên gọi biết kết quả OCR này thuộc
# về xe nào.

import cv2


def run(input_q, result_q):
    from ocr_engine import PlateOCR

    reader = PlateOCR(gpu=False)

    while True:
        item = input_q.get()   # Block cho đến khi có dữ liệu
        if item is None:
            break

        track_id, plate_crop = item

        try:
            text = reader.read_plate(plate_crop)
            if text:
                # KHÔNG dọn dẹp result_q ở đây: mỗi track_id có kết quả riêng,
                # xóa hàng đợi sẽ phá hủy kết quả OCR của các xe khác đang
                # chờ xử lý trong hệ thống tracking đa đối tượng.
                result_q.put((track_id, text))
            # Nếu text rỗng: không put gì → bên gọi giữ nguyên confirm_count
            # của track_id này (tránh reset counter chỉ vì một frame mờ tạm thời)
        except Exception as e:
            print(f"[OCR Process] Lỗi (track_id={track_id}): {e}")
