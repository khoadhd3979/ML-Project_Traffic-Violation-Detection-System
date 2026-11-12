import os
import sys
import random
from datetime import datetime, timedelta

# Đảm bảo đường dẫn gốc project nằm trong sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.database_manager import DatabaseManager


def seed_data(num_violations: int = 40):
    """Nạp dữ liệu vi phạm giao thông mẫu vào cơ sở dữ liệu SQLite."""
    print("=" * 60)
    print("🌱 Bắt đầu nạp dữ liệu mẫu (Seed Data) cho TVDS...")
    print("=" * 60)

    db = DatabaseManager()

    # 1. Đảm bảo rules mặc định đã tồn tại
    rules = db.get_all_violation_rules()
    if not rules:
        print("[1/4] Khởi tạo bảng danh mục lỗi vi phạm...")
        db.create_violation_rule("NO_HELMET", "Không đội mũ bảo hiểm", "low", 500000)
        db.create_violation_rule("WRONG_LANE", "Sai làn đường", "medium", 500000)
        db.create_violation_rule("WRONG_WAY", "Đi ngược chiều", "high", 1500000)
        db.create_violation_rule("RED_LIGHT", "Vượt đèn đỏ", "critical", 900000)
        rules = db.get_all_violation_rules()

    rule_ids = [r["id"] for r in rules]
    rule_map = {r["id"]: r["rule_code"] for r in rules}

    # 2. Tạo phiên quét mẫu
    print("[2/4] Tạo phiên quét mẫu (Processing Session)...")
    session_id = db.create_processing_session(
        session_name="Demo Live Traffic Session",
        source_type="video_file",
        source_path="data/raw/traffic_demo.mp4"
    )

    # 3. Tạo danh sách xe mẫu
    print("[3/4] Tạo danh sách phương tiện...")
    db.create_vehicle("UNKNOWN", "motorbike", "Xe chưa nhận diện được biển số")

    sample_plates = [
        "59A1-12345", "51G-98765", "29B1-55555", "43C1-88888",
        "60B8-23412", "72A-34567", "65E1-77789", "30K-99999",
        "59V2-66882", "50Y1-11223", "92D1-44556", "36B3-78901"
    ]

    vehicle_ids = []
    for plate in sample_plates:
        vid = db.create_vehicle(plate, "motorbike")
        vehicle_ids.append(vid)

    # 4. Tạo các bản ghi vi phạm rải rác trong 3 ngày gần đây
    print(f"[4/4] Tạo {num_violations} bản ghi vi phạm ngẫu nhiên...")
    now = datetime.now()

    # Đảm bảo có thư mục evidence mẫu
    day_str = now.strftime("%Y%m%d")
    evidence_dir = os.path.join("data", "evidence", day_str)
    os.makedirs(evidence_dir, exist_ok=True)

    with db._get_connection() as conn:
        for i in range(num_violations):
            # Tạo thời gian ngẫu nhiên trong 72 giờ qua
            hours_ago = random.uniform(0.1, 72.0)
            detected_time = now - timedelta(hours=hours_ago)
            time_str = detected_time.strftime("%Y-%m-%d %H:%M:%S")

            rule_id = random.choices(
                rule_ids,
                weights=[0.55, 0.20, 0.15, 0.10],  # Không mũ chiếm đa số
                k=1
            )[0]

            # 20% khả năng biển số chưa đọc được (UNKNOWN)
            if random.random() < 0.20:
                veh_id = db.get_vehicle_by_plate("UNKNOWN")["id"]
            else:
                veh_id = random.choice(vehicle_ids)

            track_id = random.randint(1, 80)
            frame_num = random.randint(100, 5000)
            conf = round(random.uniform(0.70, 0.98), 2)
            rule_code = rule_map.get(rule_id, "VIOLATION")

            # Đường dẫn ảnh bằng chứng giả lập
            evidence_path = f"data/evidence/{day_str}/{session_id}_{i+1}_{rule_code}.jpg"

            query = """
                INSERT INTO violations (
                    session_id, vehicle_id, rule_id, track_id, frame_number,
                    confidence, evidence_image_path, detected_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending');
            """
            conn.execute(
                query,
                (session_id, veh_id, rule_id, track_id, frame_num, conf, evidence_path, time_str)
            )

        conn.commit()

    db.finish_processing_session(session_id, total_frames=5200, status="completed")

    print("=" * 60)
    print("✅ Hoàn tất nạp dữ liệu mẫu!")
    stats = db.get_statistics()
    print(f"📊 Tổng vi phạm trong DB: {stats['total_violations']}")
    print(f"📊 Vi phạm hôm nay: {stats['violations_today']}")
    print(f"📊 Phương tiện vi phạm: {stats['unique_vehicles']}")
    print(f"📊 Biển số đọc được: {stats['plates_identified']}")
    print("=" * 60)


if __name__ == "__main__":
    seed_data()
