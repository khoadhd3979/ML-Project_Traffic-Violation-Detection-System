"""
seed_data.py
Bơm dữ liệu giả vào file traffic.db đã được khởi tạo.
"""
import os
import random
from datetime import datetime, timedelta
from database_manager import DatabaseManager

DB_PATH = "traffic.db"

def seed_data():
    if not os.path.exists(DB_PATH):
        print(f"Lỗi: Không tìm thấy {DB_PATH}! Hãy chạy file init_database.py trước.")
        return

    db = DatabaseManager(DB_PATH)

    # 1) violation_rules
    rules = [
        dict(rule_code="RED_LIGHT", rule_name="Vượt đèn đỏ", severity="high",
             fine_amount_min=4000000, fine_amount_max=6000000,
             legal_reference="Nghị định 100/2019/NĐ-CP, Điều 5",
             description="Xe không dừng lại khi đèn tín hiệu chuyển đỏ."),
        dict(rule_code="SPEEDING", rule_name="Vượt quá tốc độ cho phép", severity="medium",
             fine_amount_min=800000, fine_amount_max=2000000,
             legal_reference="Nghị định 100/2019/NĐ-CP, Điều 5-6",
             description="Tốc độ đo được vượt quá giới hạn cho phép của tuyến đường."),
        dict(rule_code="WRONG_LANE", rule_name="Đi sai làn đường", severity="medium",
             fine_amount_min=400000, fine_amount_max=1000000,
             legal_reference="Nghị định 100/2019/NĐ-CP, Điều 5",
             description="Xe di chuyển không đúng làn đường quy định."),
        dict(rule_code="NO_HELMET", rule_name="Không đội mũ bảo hiểm", severity="low",
             fine_amount_min=200000, fine_amount_max=300000,
             legal_reference="Nghị định 100/2019/NĐ-CP, Điều 6",
             description="Người điều khiển xe máy không đội mũ bảo hiểm."),
        dict(rule_code="WRONG_WAY", rule_name="Đi ngược chiều", severity="high",
             fine_amount_min=3000000, fine_amount_max=5000000,
             legal_reference="Nghị định 100/2019/NĐ-CP, Điều 5",
             description="Xe di chuyển ngược chiều quy định trên tuyến đường một chiều."),
    ]
    rule_ids = {}
    for r in rules:
        rid = db.create_violation_rule(**r)
        rule_ids[r["rule_code"]] = rid

    # 2) processing_sessions
    cameras = [
        ("CAM-001", "Ngã tư Nguyễn Huệ - Lê Lợi"),
        ("CAM-002", "Ngã tư Hàng Xanh"),
        ("CAM-003", "Vòng xoay Điện Biên Phủ"),
    ]
    session_ids = []
    base_time = datetime(2026, 9, 20, 6, 0, 0)
    for i, (cam_id, loc) in enumerate(cameras):
        start = base_time + timedelta(days=i)
        sid = db.create_processing_session(
            session_name=f"Session-{cam_id}-{start.strftime('%Y%m%d')}",
            source_type="video_file",
            source_path=f"videos/{cam_id.lower()}_{start.strftime('%Y%m%d')}.mp4",
            camera_id=cam_id,
            location=loc,
            model_name="YOLOv8n-traffic",
            model_version="v1.2.0",
            config={"conf_threshold": 0.5, "iou_threshold": 0.45},
        )
        db.update_session_stats(sid, total_frames_processed=54000,
                                 total_vehicles_detected=random.randint(200, 500))
        db.finish_processing_session(sid, status="completed")
        session_ids.append((sid, cam_id, loc))

    # 3) vehicles
    plate_prefixes = ["51H", "59A", "60A", "61B", "30F"]
    vehicle_types = ["car", "motorbike", "truck", "bus"]
    colors = ["trắng", "đen", "bạc", "đỏ", "xanh dương"]
    brands = ["Toyota", "Honda", "Yamaha", "Hyundai", "Ford", "Kia", None]

    vehicle_ids = []
    for i in range(30):
        plate = f"{random.choice(plate_prefixes)}-{random.randint(100,999)}.{random.randint(10,99)}"
        vid = db.create_vehicle(
            license_plate=plate,
            vehicle_type=random.choice(vehicle_types),
            color=random.choice(colors),
            brand=random.choice(brands),
            plate_confidence=round(random.uniform(0.75, 0.99), 2),
        )
        vehicle_ids.append(vid)

    # 4) violations
    rule_codes = list(rule_ids.keys())
    statuses = ["pending", "confirmed", "rejected", "processed"]

    for i in range(60):
        session_id, cam_id, loc = random.choice(session_ids)
        vehicle_id = random.choice(vehicle_ids)
        rule_code = random.choice(rule_codes)
        rule_id = rule_ids[rule_code]
        detected_at = base_time + timedelta(
            days=random.randint(0, 2), hours=random.randint(6, 20), minutes=random.randint(0, 59)
        )

        speed_kmh = None
        speed_limit_kmh = None
        if rule_code == "SPEEDING":
            speed_limit_kmh = 50
            speed_kmh = round(speed_limit_kmh + random.uniform(5, 40), 1)

        evidence_path = db.build_evidence_path(
            evidence_root="data/evidence",
            session_id=session_id,
            violation_id=i + 1,
            rule_code=rule_code,
            detected_at=detected_at,
        )

        violation_id = db.create_violation(
            vehicle_id=vehicle_id,
            rule_id=rule_id,
            session_id=session_id,
            camera_id=cam_id,
            location=loc,
            confidence_score=round(random.uniform(0.6, 0.98), 2),
            frame_number=random.randint(1, 54000),
            speed_kmh=speed_kmh,
            speed_limit_kmh=speed_limit_kmh,
            evidence_image_path=evidence_path,
            detected_at=detected_at.isoformat(sep=" ", timespec="seconds"),
        )

        status = random.choice(statuses)
        if status in ("confirmed", "rejected", "processed"):
            db.update_violation_status(violation_id, status=status, reviewed_by="admin_demo")

    print("Đã nạp dữ liệu giả thành công!")
    print(" - violation_rules:", len(rules))
    print(" - processing_sessions:", len(session_ids))
    print(" - vehicles:", len(vehicle_ids))
    print(" - violations: 60\n")
    print("Thống kê vi phạm theo luật:")
    for row in db.get_violation_stats_by_rule():
        print(" ", row)

if __name__ == "__main__":
    seed_data()