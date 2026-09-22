import cv2
import numpy as np
import multiprocessing as mp
import time
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from detection import detect_license_plate
from processing import process_plate
from config import (
    VIDEO_SOURCE, DISPLAY_SIZE, YOLO_INTERVAL, YOLO_INPUT_WIDTH,
    OCR_COOLDOWN, VERIFY_INTERVAL, PLATE_REGEX,
)
import database_manager as db

import threading

# Camera phát hiện vi phạm (đa xe) → ghi vào DB
# Dashboard Streamlit đọc ra → hiển thị cho nhân viên


# ==== VÙNG VI PHẠM (Polygon tĩnh) ====
# Tọa độ theo hệ tọa độ frame GỐC (chưa resize). Chỉnh lại theo góc đặt camera thực tế.
VIOLATION_ZONE_POLYGON = np.array([
    [300, 200],
    [900, 200],
    [1000, 700],
    [200, 700],
], dtype=np.int32)

# Số lần OCR liên tiếp cùng kết quả mới được coi là "xác nhận" và gửi lên DB
CONFIRM_THRESHOLD = 2

# Nếu 1 track_id biến mất khỏi frame quá lâu (giây) → coi như xe đã rời khung hình, xóa state
VEHICLE_TIMEOUT = 2.0

# Kích thước tối đa của queue OCR — cho phép xếp hàng crop của nhiều xe cùng lúc
OCR_QUEUE_MAXSIZE = 10


def is_valid_plate(text: str) -> bool:
    # Kiểm tra biển số có đúng định dạng biển số VN không
    if not text or len(text) < 7:
        return False
    return bool(re.match(PLATE_REGEX, text.strip().upper()))


# CAMERA STREAM (thread riêng, không block main loop)
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self._lock:
                self.grabbed = grabbed
                self.frame = frame
            if not grabbed:
                self.stop()

    def read(self):
        with self._lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()


def _new_vehicle_state(bbox, class_id, now):
    return {
        "bbox": bbox,
        "class_id": class_id,
        "last_text": "",
        "last_shared_plate": "",
        "last_ocr_time": 0.0,
        "last_verify_time": 0.0,
        "confirm_candidate": "",
        "confirm_count": 0,
        "last_seen": now,
        "in_violation_zone": False,
    }


# MAIN LOOP
def run_live_camera(video_source=VIDEO_SOURCE):
    conn = db.init_db()
    conn.close()

    # Hai Queue giao tiếp với OCR. Mỗi item mang theo track_id để main loop
    # biết kết quả OCR thuộc về xe nào khi có nhiều xe trong cùng 1 frame.
    #   input_q : main → OCR   item = (track_id, gray_processed_image)
    #   result_q: OCR → main   item = (track_id, text)
    # LƯU Ý: ocr_worker_proc.py cần được cập nhật để unpack/pack tuple (track_id, ...)
    # thay vì chỉ ảnh/text đơn lẻ như phiên bản cũ.
    input_q  = mp.Queue(maxsize=OCR_QUEUE_MAXSIZE)
    result_q = mp.Queue(maxsize=OCR_QUEUE_MAXSIZE)

    # --- Khởi chạy OCR process (KHÔNG phải thread → không bị GIL) ---
    import ocr_worker_proc
    ocr_proc = mp.Process(
        target=ocr_worker_proc.run,
        args=(input_q, result_q),
        daemon=True,
    )
    ocr_proc.start()
    print("Đang khởi tạo OCR process...")

    print(f"Đang kết nối Camera: {video_source}...")
    cam = CameraStream(video_source).start()
    time.sleep(1.5)   # Chờ camera ổn định + OCR process load model

    if not cam.grabbed:
        print("Lỗi: Không thể mở camera!")
        input_q.put(None)
        ocr_proc.join(timeout=2)
        return

    print("Hệ thống theo dõi vi phạm (đa đối tượng) đã sẵn sàng! Nhấn 'q' để thoát.")

    frame_count = 0
    scale = src_w = src_h = None

    # --- Quản lý vòng đời & trạng thái nhiều xe cùng lúc, khóa theo track_id ---
    active_vehicles = {}

    while True:
        loop_start = time.time()

        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        frame_count += 1

        if scale is None:
            src_h, src_w = frame.shape[:2]
            scale = YOLO_INPUT_WIDTH / src_w

        # --- Vẽ vùng vi phạm lên frame hiển thị ---
        cv2.polylines(frame, [VIOLATION_ZONE_POLYGON], isClosed=True,
                      color=(0, 0, 255), thickness=2)

        # --- Tracking đa đối tượng, chạy ngắt quãng mỗi YOLO_INTERVAL frame ---
        if frame_count % YOLO_INTERVAL == 0:
            small = cv2.resize(frame, (int(src_w * scale), int(src_h * scale)))
            detections = detect_license_plate(small)

            now = time.time()
            seen_ids = set()

            for det in detections:
                sx1, sy1, sx2, sy2 = det["bbox"]
                bbox = [
                    int(sx1 / scale), int(sy1 / scale),
                    int(sx2 / scale), int(sy2 / scale),
                ]
                track_id = det["track_id"]
                class_id = det["class_id"]
                seen_ids.add(track_id)

                if track_id not in active_vehicles:
                    active_vehicles[track_id] = _new_vehicle_state(bbox, class_id, now)
                    print(f"[TRACK] Xe mới xuất hiện — ID {track_id}")
                else:
                    v = active_vehicles[track_id]
                    v["bbox"] = bbox
                    v["class_id"] = class_id
                    v["last_seen"] = now

                    # Đã có kết quả OCR trước đó → cứ mỗi VERIFY_INTERVAL giây verify lại
                    if v["last_text"] and (now - v["last_verify_time"]) >= VERIFY_INTERVAL:
                        v["last_text"] = ""
                        v["last_ocr_time"] = 0.0
                        v["last_verify_time"] = now

            # --- Dọn các track_id đã rời khỏi khung hình quá lâu ---
            stale_ids = [
                tid for tid, v in active_vehicles.items()
                if tid not in seen_ids and (now - v["last_seen"]) > VEHICLE_TIMEOUT
            ]
            for tid in stale_ids:
                del active_vehicles[tid]

        # --- Lấy toàn bộ kết quả OCR đang chờ (non-blocking), gắn theo track_id ---
        try:
            while True:
                result = result_q.get_nowait()
                if not result:
                    continue
                track_id, new_text = result
                v = active_vehicles.get(track_id)
                if v is None or not new_text:
                    continue

                if new_text != v["last_text"]:
                    print(f"[OCR] ID {track_id}: {v['last_text']!r} → {new_text!r}")
                v["last_text"] = new_text

                # --- Voting: tích lũy CONFIRM_THRESHOLD lần liên tiếp cùng biển số ---
                if is_valid_plate(new_text) and new_text != v["last_shared_plate"]:
                    if new_text == v["confirm_candidate"]:
                        v["confirm_count"] += 1
                        print(f"[VOTE] ID {track_id} {new_text!r} — lần {v['confirm_count']}/{CONFIRM_THRESHOLD}")
                    else:
                        v["confirm_candidate"] = new_text
                        v["confirm_count"] = 1
                        print(f"[VOTE] ID {track_id} biển mới {new_text!r} — bắt đầu đếm (1/{CONFIRM_THRESHOLD})")

                    if v["confirm_count"] >= CONFIRM_THRESHOLD:
                        db.update_detected_plate(new_text)
                        print(f"[DETECT] ID {track_id} xác nhận {CONFIRM_THRESHOLD} lần: {new_text} → gửi lên Dashboard")
                        v["last_shared_plate"] = new_text
                        v["confirm_count"] = 0
                else:
                    if new_text != v["last_shared_plate"]:
                        v["confirm_candidate"] = ""
                        v["confirm_count"] = 0
        except Exception:
            pass   # Chưa có kết quả mới → dùng last_text cũ của từng xe

        # --- Vẽ UI & gửi crop cho OCR — lặp qua từng xe đang được theo dõi ---
        now = time.time()
        for track_id, v in active_vehicles.items():
            x1, y1, x2, y2 = map(int, v["bbox"])

            # Xe có nằm trong vùng vi phạm không (dựa vào tâm bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            inside = cv2.pointPolygonTest(VIOLATION_ZONE_POLYGON, (float(cx), float(cy)), False) >= 0
            v["in_violation_zone"] = inside

            box_color_bbox = (0, 0, 255) if inside else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color_bbox, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, max(15, y1 - 45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color_bbox, 2)

            # Gửi crop cho OCR: cooldown riêng theo từng xe, dùng chung 1 queue
            if (now - v["last_ocr_time"]) >= OCR_COOLDOWN and not input_q.full():
                py1, py2 = max(0, y1), min(src_h, y2)
                px1, px2 = max(0, x1), min(src_w, x2)
                plate_crop = frame[py1:py2, px1:px2]

                if plate_crop.size > 0:
                    processed = process_plate(frame, v["bbox"])
                    if processed is not None:
                        try:
                            # Grayscale giúp giảm kích thước truyền tải IPC đi 3 lần
                            gray_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                            input_q.put_nowait((track_id, gray_processed))
                            v["last_ocr_time"] = now
                        except Exception:
                            pass  # Queue đầy (OCR vẫn bận) → bỏ qua

            if v["last_text"]:
                # Màu box: xanh lá nếu hợp lệ, vàng nếu chưa xác nhận
                box_color = (0, 200, 0) if is_valid_plate(v["last_text"]) else (0, 200, 255)
                cv2.rectangle(frame, (x1, max(0, y1 - 40)), (x2, y1), (0, 0, 0), -1)
                cv2.putText(frame, v["last_text"], (x1 + 5, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # --- FPS ---
        elapsed = time.time() - loop_start
        fps = (1.0 / elapsed) if elapsed > 0 else 999
        cv2.putText(frame, f"FPS: {int(fps)} | Xe: {len(active_vehicles)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Traffic Violation Detection",
                   cv2.resize(frame, DISPLAY_SIZE))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Dọn dẹp ---
    input_q.put(None)
    ocr_proc.join(timeout=3)
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.freeze_support()
    run_live_camera()
