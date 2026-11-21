import os
import sys
import time
import cv2
import multiprocessing as mp

from src.models.detection import track_vehicles_and_helmets, detect_plate_in_crop
from src.features.processing import process_plate, check_helmet_violation, crop_with_padding, _iou
from src.models import ocr_worker_proc
from src.data.database_manager import DatabaseManager

# Số inference cycle LIÊN TIẾP mà cùng một track_id xe phải xuất hiện trong kết quả
# check_helmet_violation() trước khi vi phạm được xác nhận (mới tạo Evidence / DB record / OCR).
# Hiện YOLO chạy trên MỌI frame đọc được (không có frame skipping trong vòng lặp này),
# nên 1 cycle = 1 frame. Hằng số bất biến, không phải state dùng chung.
# Lưu ý: xác nhận theo track_id của XE, không chứng minh cùng một NGƯỜI vi phạm qua các cycle.
VIOLATION_CONFIRM_FRAMES = 3
STAGE2_REJECT_COOLDOWN_FRAMES = 15
MAX_PLATE_ATTEMPTS = 3
ID_SWITCH_IOU_THRESHOLD = 0.45
ID_SWITCH_MAX_AGE_FRAMES = 15


def _draw_box_label(frame, bbox, label, color, thickness=2, fill_label=True):
    """Vẽ khung và nhãn lên frame, tự clamp vào kích thước ảnh."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    if fill_label:
        cv2.rectangle(frame, (x1, max(0, y1 - h_txt - 10)), (x1 + w_txt + 10, y1), color, -1)
        txt_color = (255, 255, 255)
    else:
        txt_color = color
    cv2.putText(frame, label, (x1 + 5, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 2)


def generate_live_stream(source=0):
    """
    Generator đọc video stream từ camera/video source, chạy YOLO tracking,
    xử lý vi phạm mũ bảo hiểm & OCR biển số song song bằng multiprocessing Queue.

    Vi phạm chỉ được xác nhận (Evidence + DB + OCR + latest_event + vẽ khung đỏ) sau khi cùng
    một track_id xe vi phạm trong VIOLATION_CONFIRM_FRAMES inference cycle liên tiếp.

    Yields:
        (frame, info_dict):
            frame (np.ndarray): Ảnh BGR đã được vẽ khung vi phạm, biển số và FPS.
            info_dict (dict): Thông tin trạng thái gồm FPS, số vụ vi phạm, sự kiện mới nhất...
                "violating_vehicles"/"violating_count" chỉ gồm các xe ĐÃ XÁC NHẬN ở cycle hiện tại.
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
    plate_attempts = {}      # track_id -> số lần đã thử detect biển số
    recorded_violations = {} # track_id -> dict thông tin violation
    track_plates = {}        # track_id -> plate_text
    recorded_last_seen = {}  # track_id -> (frame_count, bbox) để chống trùng khi ID switch

    # --- Debouncing state: LOCAL cho từng generator (mỗi camera/video một bộ độc lập) ---
    violation_streaks = {}      # track_id -> số inference cycle liên tiếp có vi phạm (chỉ track chưa ghi nhận)
    pending_evidence = {}       # track_id -> đường dẫn evidence đã lưu nhưng DB chưa ghi được (để retry không lưu trùng)
    stage2_cooldown_until = {}  # track_id -> frame_count được phép kiểm tra lại sau khi Stage 2 bác bỏ
    stage2_rejected_count = 0

    reset_fn = getattr(sys.modules.get("src.models.detection"), "reset_tracker", None)
    if reset_fn is not None:
        reset_fn()

    frame_count = 0
    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            h_frame, w_frame = frame.shape[:2]

            # 1. Detection & Tracking (chạy trên mọi frame => mỗi vòng lặp là một inference cycle mới)
            detections = track_vehicles_and_helmets(frame)
            violating_vehicles = check_helmet_violation(detections)

            latest_violation_info = None
            now = time.time()

            # 1a. Debouncing theo track_id của xe. Bỏ qua track_id=None, mỗi cycle chỉ tính một lần cho mỗi track.
            current_violators = {}
            for v in violating_vehicles:
                tid = v.get("track_id")
                if tid is None or tid in current_violators:
                    continue
                current_violators[tid] = v

            # Track chưa xác nhận mà không vi phạm ở cycle này => reset bộ đếm (không cộng dồn lần cách quãng).
            for tid in list(violation_streaks):
                if tid not in current_violators:
                    del violation_streaks[tid]
                    orphan = pending_evidence.pop(tid, None)
                    if orphan:
                        try:
                            os.remove(orphan)
                        except OSError:
                            pass

            confirmed_vehicles = []
            rejected_this_cycle = []
            verify_crop_fn = getattr(sys.modules.get("src.models.detection"), "verify_violation_in_crop", None)
            for tid, v in current_violators.items():
                if tid not in recorded_violations:
                    if frame_count < stage2_cooldown_until.get(tid, 0):
                        continue
                    violation_streaks[tid] = violation_streaks.get(tid, 0) + 1
                    if violation_streaks[tid] < VIOLATION_CONFIRM_FRAMES:
                        continue
                    if tid not in pending_evidence and verify_crop_fn is not None:
                        crop_check = crop_with_padding(frame, v.get("full_violator_bbox"))
                        if not verify_crop_fn(crop_check):
                            violation_streaks[tid] = 0
                            stage2_cooldown_until[tid] = frame_count + STAGE2_REJECT_COOLDOWN_FRAMES
                            stage2_rejected_count += 1
                            rejected_this_cycle.append(v)
                            print(f"[Stage2 Reject] track_id={tid}: anh crop can canh xac nhan CO MU (bo qua)")
                            continue

                    # Chống lưu trùng cùng một xe khi ByteTrack bị đổi track_id (ID switch)
                    matched_old_tid = None
                    for old_tid, (last_f, old_bbox) in recorded_last_seen.items():
                        if old_tid in current_violators:
                            continue
                        if (frame_count - last_f) <= ID_SWITCH_MAX_AGE_FRAMES and _iou(v["bbox"], old_bbox) >= ID_SWITCH_IOU_THRESHOLD:
                            matched_old_tid = old_tid
                            break
                    if matched_old_tid is not None:
                        recorded_violations[tid] = recorded_violations[matched_old_tid]
                        track_plates[tid] = track_plates.get(matched_old_tid, "UNKNOWN")
                        violation_streaks.pop(tid, None)
                        pending_evidence.pop(tid, None)

                if tid in recorded_violations:
                    recorded_last_seen[tid] = (frame_count, v["bbox"])
                confirmed_vehicles.append(v)

            for v in confirmed_vehicles:
                track_id = v["track_id"]
                vx1, vy1, vx2, vy2 = map(int, v["bbox"])
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w_frame, vx2), min(h_frame, vy2)

                vehicle_crop = frame[vy1:vy2, vx1:vx2]


                # DECOUPLING: Ghi nhận ngay vi phạm vào DB với biển "UNKNOWN" để tránh rớt vi phạm
                # (chỉ sau khi đã xác nhận qua VIOLATION_CONFIRM_FRAMES cycle liên tiếp)
                if track_id not in recorded_violations:
                    try:
                        # Lưu ảnh bằng chứng xe vi phạm vào thư mục data/evidence/{YYYYMMDD}/
                        # Evidence = toàn bộ người vi phạm + toàn bộ xe (full_violator_bbox từ
                        # processing.py), crop từ frame gốc ở cycle xác nhận, có padding nhỏ và đã clamp vào frame.
                        # KHÔNG dùng vehicle_crop (chỉ xe) làm evidence; vehicle_crop vẫn dùng cho OCR biển số.
                        evidence_rel_path = pending_evidence.get(track_id)
                        if evidence_rel_path is None:
                            full_violator_crop = crop_with_padding(frame, v.get("full_violator_bbox"))
                            if full_violator_crop is not None:
                                day_dir = os.path.join("data", "evidence", time.strftime("%Y%m%d"))
                                os.makedirs(day_dir, exist_ok=True)
                                evidence_filename = f"track_{track_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                evidence_rel_path = os.path.join(day_dir, evidence_filename).replace("\\", "/")
                                if cv2.imwrite(evidence_rel_path, full_violator_crop):
                                    # Nhớ lại để nếu bước DB bên dưới lỗi thì lần retry dùng lại đúng file này.
                                    pending_evidence[track_id] = evidence_rel_path
                                else:
                                    print(f"[LiveStream Warning] Không ghi được ảnh evidence: {evidence_rel_path}")
                                    evidence_rel_path = None
                            else:
                                print(f"[LiveStream Warning] full_violator_bbox không hợp lệ cho track_id={track_id}: {v.get('full_violator_bbox')}")

                        if session_id is None:
                            # Không có session thì không tạo được violation => KHÔNG đánh dấu đã ghi nhận.
                            raise RuntimeError("Chưa có processing session, chưa thể tạo violation record")

                        unknown_vehicle = db_manager.get_vehicle_by_plate("UNKNOWN")
                        if not unknown_vehicle:
                            unknown_vehicle_id = db_manager.create_vehicle(license_plate="UNKNOWN", vehicle_type="motorbike")
                        else:
                            unknown_vehicle_id = unknown_vehicle["id"]

                        v_id = db_manager.create_violation(
                            vehicle_id=unknown_vehicle_id,
                            rule_id=rule_id,
                            session_id=session_id,
                            evidence_image_path=evidence_rel_path
                        )
                        if not v_id:
                            raise RuntimeError("create_violation không trả về violation_id")

                        # Chỉ tới đây (record đã tạo thành công) mới đánh dấu track đã ghi nhận.
                        recorded_violations[track_id] = {
                            "violation_id": v_id,
                            "vehicle_id": unknown_vehicle_id,
                            "plate": "UNKNOWN",
                            "evidence_path": evidence_rel_path
                        }
                        track_plates[track_id] = "UNKNOWN"
                        recorded_last_seen[track_id] = (frame_count, v["bbox"])
                        violation_streaks.pop(track_id, None)
                        pending_evidence.pop(track_id, None)

                        latest_violation_info = {
                            "track_id": track_id,
                            "plate": "UNKNOWN",
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "violation": "Không đội mũ bảo hiểm",
                            "evidence_path": evidence_rel_path
                        }
                        print(f"[LiveStream Decoupled] Đã ghi nhận vi phạm cho track_id={track_id} (Ảnh bằng chứng: {evidence_rel_path})")
                    except Exception as db_err:
                        # Không có entry trong recorded_violations => cycle sau sẽ retry (evidence được tái sử dụng).
                        print(f"[LiveStream DB Error] {db_err}")


                # OCR chỉ chạy cho xe đã xác nhận, chưa đọc được biển số và chưa vượt số lần thử tối đa.
                last_sent = ocr_sent_time.get(track_id, 0.0)
                rec_entry = recorded_violations.get(track_id)
                attempts = plate_attempts.get(track_id, 0)
                if (
                    rec_entry is not None
                    and rec_entry.get("plate", "UNKNOWN") == "UNKNOWN"
                    and attempts < MAX_PLATE_ATTEMPTS
                    and vehicle_crop.size > 0
                    and (now - last_sent) >= 1.0
                ):
                    plate_attempts[track_id] = attempts + 1
                    ocr_sent_time[track_id] = now
                    plate_bbox = detect_plate_in_crop(vehicle_crop)
                    if plate_bbox is not None:
                        enhanced_plate = process_plate(vehicle_crop, plate_bbox)
                        if enhanced_plate is not None and not input_q.full():
                            try:
                                input_q.put_nowait((track_id, enhanced_plate))
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

            # 3. Draw bounding boxes & labels (xe dang theo doi, non bao hiem, bac bo Stage 2, va vi pham da xac nhan)
            confirmed_ids = {v["track_id"] for v in confirmed_vehicles}
            rejected_ids = {v["track_id"] for v in rejected_this_cycle}
            bike_count = 0
            helmet_count = 0

            for det in detections:
                cls_id = det.get("class_id")
                bbox = det.get("bbox")
                if not bbox:
                    continue
                if cls_id == 1:
                    bike_count += 1
                    tid = det.get("track_id")
                    if tid not in confirmed_ids and tid not in rejected_ids:
                        lbl = f"Xe {tid}" if tid is not None else "Xe"
                        _draw_box_label(frame, bbox, lbl, (255, 180, 0), thickness=1, fill_label=False)
                elif cls_id in (0, 3):
                    helmet_count += 1
                    conf = det.get("conf")
                    lbl = f"CO MU {conf:.2f}" if conf is not None else "CO MU"
                    _draw_box_label(frame, bbox, lbl, (0, 200, 0), thickness=2, fill_label=False)

            for v in rejected_this_cycle:
                _draw_box_label(frame, v["bbox"], f"ID {v['track_id']}: XAC MINH CO MU", (0, 215, 255), thickness=2)

            for v in confirmed_vehicles:
                track_id = v["track_id"]
                plate_str = track_plates.get(track_id, "KHONG DOI NON")
                _draw_box_label(frame, v["bbox"], f"ID {track_id}: {plate_str}", (0, 0, 255), thickness=2)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            hud_scale = 1.4 if w_frame >= 1500 else 0.7
            hud_thick = 3 if w_frame >= 1500 else 2
            hud_y = 55 if w_frame >= 1500 else 30
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Xe: {bike_count} | Co mu: {helmet_count} | Bac bo S2: {stage2_rejected_count} | Vi pham: {len(confirmed_vehicles)}",
                (15, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                hud_scale,
                (0, 255, 0),
                hud_thick,
            )

            info = {
                "fps": fps,
                "violating_count": len(confirmed_vehicles),
                "latest_event": latest_violation_info,
                "recorded_violations": recorded_violations,
                "violating_vehicles": confirmed_vehicles
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
