import cv2
import numpy as np
import os


def process_plate(image, bbox):
    h_img, w_img = image.shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w_img, x_max), min(h_img, y_max)

    plate_crop = image[y_min:y_max, x_min:x_max]

    if plate_crop.size == 0:
        return None

    h_crop, w_crop = plate_crop.shape[:2]
    ratio = 200.0 / float(h_crop)
    plate_crop = cv2.resize(plate_crop, (int(w_crop * ratio), 200), interpolation=cv2.INTER_CUBIC)

    # Chuyển sang hệ màu YUV để chỉ tăng độ sáng (kênh Y) mà không làm hỏng màu (U, V)
    img_yuv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2YUV)

    # Dùng CLAHE để cân bằng sáng (khử bóng râm, chống chói)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    # Chuyển ngược lại về ảnh BGR (ảnh màu chuẩn)
    enhanced_plate = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return enhanced_plate
