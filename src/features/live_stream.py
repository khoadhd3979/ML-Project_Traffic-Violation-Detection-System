import os
import sys
import time
import cv2
import multiprocessing as mp

from src.models.detection import track_vehicles_and_helmets, detect_plate_in_crop
from src.features.processing import process_plate, check_helmet_violation, crop_with_padding
from src.models import ocr_worker_proc
from src.data.database_manager import DatabaseManager


def generate_live_stream(source=0):
    """
    Generator đọc video stream từ camera/video source, chạy YOLO tracking,
    xử lý vi phạm mũ bảo hiểm & OCR biển số song song bằng multiprocessing Queue.

    Yields:
        (frame, info_dict):
            frame (np.ndarray): Ảnh BGR đã được vẽ khung vi phạm, biển số và FPS.
            info_dict (dict): Thông tin trạng thái gồm FPS, số vụ vi phạm, sự kiện mới nhất...
    """
    # Chuẩn hóa nguồn camera / video
    if source is None or str(source).strip() == "":
        source = 0
    elif isinstance(source, str) and source.strip().isdigit():
        source = int(source.strip())

    source_str = str(source).lower().strip()
    if source == 0 or source_str == "0":
        source_type = "webcam"
    elif source_str.startswith(("rtsp://", "http://", "https://")):
        source_type = "rtsp_stream"
    else:
        source_type = "video_file"

    db_manager = DatabaseManager()
    schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "schema.sql"))
    session_id = None
    rule_id = 1
    try:
        if os.path.exists(schema_path):
            db_manager.init_db(schema_path)
        
        rule = db_manager.get_violation_rule_by_code("NO_HELMET")
        if rule:
            rule_id = rule["id"]
        else:
            rule_id = db_manager.create_violation_rule(
                rule_code="NO_HELMET",
                rule_name="Không đội mũ bảo hiểm",
                severity="low"
            )

        session_id = db_manager.create_processing_session(
            session_name="Live Stream Session",
            source_type=source_type,
            source_path=str(source)
        )
    except Exception as e:
        print(f"[LiveStream DB Warning] {e}")

    # Multiprocessing OCR Queue & Worker
    input_q = mp.Queue(maxsize=10)
    result_q = mp.Queue(maxsize=10)

    ocr_proc = mp.Process(
        target=ocr_worker_proc.run,
        args=(input_q, result_q),
        daemon=True,
    )
    ocr_proc.start()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LiveStream Lỗi] Không thể mở nguồn video/camera: '{source}'")
        try:
            input_q.put_nowait(None)
        except Exception:
            pass
        if ocr_proc.is_alive():
            ocr_proc.terminate()
        return

    ocr_sent_time = {}       # track_id -> timestamp
    recorded_violations = {} # track_id -> plate_text
    track_plates = {}        # track_id -> plate_text

    # -------------------------------------------------------------------------
    # Cấu hình Debouncing: Chỉ xác nhận vi phạm sau N frame liên tiếp
    # Giảm thiểu tối đa dương tính giả do lỗi chập chờn / bắt nhầm của model AI
    # -------------------------------------------------------------------------
    VIOLATION_CONFIRM_FRAMES = 3
    track_violation_streak = {} # track_id -> số frame liên tiếp vi phạm

    frame_count = 0
    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            h_frame, w_frame = frame.shape[:2]

            # 1. Detection & Tracking
            detections = track_vehicles_and_helmets(frame)
            violating_vehicles = check_helmet_violation(detections)

            # Cập nhật bộ đếm Debouncing theo track_id
            current_violating_ids = set()
            for v in violating_vehicles:
                tid = v.get("track_id")
                if tid is not None:
                    current_violating_ids.add(tid)
                    track_violation_streak[tid] = track_violation_streak.get(tid, 0) + 1

            # Reset streak cho các xe máy có mặt trong frame nhưng không vi phạm (hoặc đội mũ)
            all_bike_tracks = set(d.get("track_id") for d in detections if d.get("class_id") == 1 and d.get("track_id") is not None)
            for tid in all_bike_tracks:
                if tid not in current_violating_ids and tid not in recorded_violations:
                    track_violation_streak[tid] = 0

            latest_violation_info = None
            now = time.time()

            # Lọc các xe máy đã đạt đủ số frame vi phạm liên tiếp (streak >= VIOLATION_CONFIRM_FRAMES)
            confirmed_violators = [
                v for v in violating_vehicles 
                if v.get("track_id") is not None and track_violation_streak.get(v.get("track_id"), 0) >= VIOLATION_CONFIRM_FRAMES
            ]

            for v in confirmed_violators:
                track_id = v["track_id"]
                vx1, vy1, vx2, vy2 = map(int, v["bbox"])
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w_frame, vx2), min(h_frame, vy2)

                vehicle_crop = frame[vy1:vy2, vx1:vx2]

                # Ghi nhận vi phạm vào DB với biển "UNKNOWN" khi đã qua bộ lọc debounce
                if track_id not in recorded_violations:
                    try:
                        evidence_rel_path = None
                        full_violator_crop = crop_with_padding(frame, v.get("full_violator_bbox"))
                        if full_violator_crop is not None:
                            day_dir = os.path.join("data", "evidence", time.strftime("%Y%m%d"))
                            os.makedirs(day_dir, exist_ok=True)
                            evidence_filename = f"track_{track_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            evidence_rel_path = os.path.join(day_dir, evidence_filename).replace("\\", "/")
                            cv2.imwrite(evidence_rel_path, full_violator_crop)
                        else:
                            print(f"[LiveStream Warning] full_violator_bbox không hợp lệ cho track_id={track_id}: {v.get('full_violator_bbox')}")

                        unknown_vehicle = db_manager.get_vehicle_by_plate("UNKNOWN")
                        if not unknown_vehicle:
                            unknown_vehicle_id = db_manager.create_vehicle(license_plate="UNKNOWN", vehicle_type="motorbike")
                        else:
                            unknown_vehicle_id = unknown_vehicle["id"]

                        v_id = None
                        if session_id is not None:
                            v_id = db_manager.create_violation(
                                vehicle_id=unknown_vehicle_id,
                                rule_id=rule_id,
                                session_id=session_id,
                                evidence_image_path=evidence_rel_path
                            )

                        recorded_violations[track_id] = {
                            "violation_id": v_id,
                            "vehicle_id": unknown_vehicle_id,
                            "plate": "UNKNOWN",
                            "evidence_path": evidence_rel_path
                        }
                        track_plates[track_id] = "UNKNOWN"

                        latest_violation_info = {
                            "track_id": track_id,
                            "plate": "UNKNOWN",
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "violation": "Không đội mũ bảo hiểm",
                            "evidence_path": evidence_rel_path
                        }
                        print(f"[LiveStream Debounced] Đã xác nhận vi phạm sau {VIOLATION_CONFIRM_FRAMES} frames liên tiếp cho track_id={track_id} (Ảnh bằng chứng: {evidence_rel_path})")
                    except Exception as db_err:
                        print(f"[LiveStream DB Error] {db_err}")

                last_sent = ocr_sent_time.get(track_id, 0.0)
                if vehicle_crop.size > 0 and (now - last_sent) >= 1.0:
                    plate_bbox = detect_plate_in_crop(vehicle_crop)
                    if plate_bbox is not None:
                        enhanced_plate = process_plate(vehicle_crop, plate_bbox)
                        if enhanced_plate is not None and not input_q.full():
                            try:
                                input_q.put_nowait((track_id, enhanced_plate))
                                ocr_sent_time[track_id] = now
                            except Exception:
                                pass

            # 2. Check OCR Queue Results non-blocking: Cập nhật lại biển số nếu OCR đọc thành công
            try:
                while True:
                    result = result_q.get_nowait()
                    if not result:
                        continue
                    track_id, text = result
                    if text:
                        plate_text = text.strip().upper()
                        track_plates[track_id] = plate_text
                        
                        rec_info = recorded_violations.get(track_id)
                        if rec_info and rec_info.get("plate") != plate_text:
                            try:
                                vehicle = db_manager.get_vehicle_by_plate(plate_text)
                                if not vehicle:
                                    real_vehicle_id = db_manager.create_vehicle(license_plate=plate_text, vehicle_type="motorbike")
                                else:
                                    real_vehicle_id = vehicle["id"]

                                v_id = rec_info.get("violation_id")
                                if v_id:
                                    db_manager.update_violation_vehicle(violation_id=v_id, vehicle_id=real_vehicle_id)

                                rec_info["vehicle_id"] = real_vehicle_id
                                rec_info["plate"] = plate_text

                                latest_violation_info = {
                                    "track_id": track_id,
                                    "plate": plate_text,
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "violation": "Không đội mũ bảo hiểm"
                                }
                                print(f"[LiveStream OCR Update] Đã cập nhật biển số cho track_id={track_id} -> Biển thật: {plate_text}")
                            except Exception as db_err:
                                print(f"[LiveStream DB Update Error] {db_err}")
            except Exception:
                pass

            # 3. Draw bounding boxes & labels (tự động co giãn theo độ phân giải frame, hỗ trợ video 4K dọc như 1pc.mp4)
            scale_factor = max(1.0, min(w_frame, h_frame) / 720.0)
            box_thickness = max(2, int(scale_factor * 2))
            font_scale = max(0.5, scale_factor * 0.45)
            font_thickness = max(1, int(scale_factor * 1.5))

            # 3.1 Vẽ trực quan các vùng đầu: Có mũ bảo hiểm (Xanh lá) và Không đội mũ (Đỏ)
            for det in detections:
                cls_id = det.get("class_id")
                conf = det.get("conf", 0.0)
                bx1, by1, bx2, by2 = map(int, det["bbox"])
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w_frame, bx2), min(h_frame, by2)

                if cls_id in (0, 3):  # driver_with_helmet, passenger_with_helemt
                    color = (0, 200, 0)
                    text_lbl = f"Co mu {conf:.2f}"
                elif cls_id in (5, 6):  # driver_without_helmet, passenger_without_helemt
                    color = (0, 0, 255)
                    text_lbl = f"Khong mu {conf:.2f}"
                else:
                    continue

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, max(1, box_thickness - 1))
                (tw, th), _ = cv2.getTextSize(text_lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, font_thickness)
                pad_h = int(3 * scale_factor)
                cv2.rectangle(frame, (bx1, max(0, by1 - th - 2 * pad_h)), (bx1 + tw + 2 * pad_h, by1), color, -1)
                cv2.putText(frame, text_lbl, (bx1 + pad_h, max(th + pad_h, by1 - pad_h)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (255, 255, 255), font_thickness)

            # 3.2 Vẽ xe máy:
            # - Xe đã xác nhận vi phạm (streak >= VIOLATION_CONFIRM_FRAMES): Khung ĐỎ đậm
            # - Xe đang nghi vấn (chờ đủ streak): Khung VÀNG cam (ví dụ: Nghi van 1/3)
            for v in violating_vehicles:
                track_id = v["track_id"]
                vx1, vy1, vx2, vy2 = map(int, v["bbox"])
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w_frame, vx2), min(h_frame, vy2)

                streak = track_violation_streak.get(track_id, 0)
                is_confirmed = streak >= VIOLATION_CONFIRM_FRAMES

                if is_confirmed:
                    box_color = (0, 0, 255)  # Đỏ
                    plate_str = track_plates.get(track_id, "KHONG DOI NON")
                    label = f"ID {track_id}: {plate_str}"
                else:
                    box_color = (0, 165, 255)  # Cam/Vàng
                    label = f"ID {track_id}: Nghi van ({streak}/{VIOLATION_CONFIRM_FRAMES})"

                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), box_color, box_thickness)
                (w_txt, h_txt), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                pad = int(5 * scale_factor)
                cv2.rectangle(frame, (vx1, max(0, vy1 - h_txt - 2 * pad)), (vx1 + w_txt + 2 * pad, vy1), box_color, -1)
                cv2.putText(frame, label, (vx1 + pad, max(h_txt + pad, vy1 - pad)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f} | Vi pham: {len(violating_vehicles)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            info = {
                "fps": fps,
                "violating_count": len(violating_vehicles),
                "latest_event": latest_violation_info,
                "recorded_violations": recorded_violations,
                "violating_vehicles": violating_vehicles
            }

            yield frame, info


    finally:
        try:
            input_q.put_nowait(None)
        except Exception:
            pass
        if ocr_proc.is_alive():
            ocr_proc.terminate()
        cap.release()
        if session_id is not None:
            try:
                db_manager.finish_processing_session(session_id)
            except Exception:
                pass