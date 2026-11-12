import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DatabaseManager:
    """
    Quản lý kết nối và các thao tác CRUD với cơ sở dữ liệu SQLite cho hệ thống TVDS.
    Hỗ trợ đa luồng an toàn, tối ưu cho luồng xử lý video song song và giao diện web.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = os.path.abspath(db_path)
        else:
            # Đường dẫn mặc định: data/traffic.db từ thư mục gốc
            root_dir = Path(__file__).resolve().parent.parent.parent
            default_path = os.environ.get(
                "TVDS_DB_PATH",
                os.path.join(root_dir, "data", "traffic.db"),
            )
            self.db_path = os.path.abspath(default_path)

        # Đảm bảo thư mục data và data/evidence tồn tại
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        evidence_dir = os.path.join(db_dir, "evidence")
        os.makedirs(evidence_dir, exist_ok=True)

        # Tự động nạp schema nếu file DB chưa tồn tại hoặc rỗng
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            if os.path.exists(schema_path):
                self.init_db(schema_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối SQLite an toàn với row_factory và bật foreign key constraints."""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self, schema_path: Optional[str] = None) -> None:
        """Khởi tạo cấu trúc bảng từ schema.sql."""
        if schema_path is None:
            schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Không tìm thấy file schema tại: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    # --------------------------------------------------------------------------
    # 1. VIOLATION RULES
    # --------------------------------------------------------------------------
    def get_violation_rule_by_code(self, rule_code: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM violation_rules WHERE rule_code = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (rule_code.upper().strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_violation_rule(
        self,
        rule_code: str,
        rule_name: str,
        severity: str = "medium",
        penalty_amount: int = 0,
        description: Optional[str] = None
    ) -> int:
        query = """
            INSERT INTO violation_rules (rule_code, rule_name, severity, penalty_amount, description)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(rule_code) DO UPDATE SET
                rule_name=excluded.rule_name,
                severity=excluded.severity,
                penalty_amount=excluded.penalty_amount;
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (rule_code.upper().strip(), rule_name, severity, penalty_amount, description))
            conn.commit()
            rule = self.get_violation_rule_by_code(rule_code)
            return rule["id"] if rule else cursor.lastrowid

    def get_all_violation_rules(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM violation_rules ORDER BY id ASC;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(r) for r in cursor.fetchall()]

    # --------------------------------------------------------------------------
    # 2. PROCESSING SESSIONS
    # --------------------------------------------------------------------------
    def create_processing_session(
        self,
        session_name: str = "Live Stream Session",
        source_type: str = "webcam",
        source_path: Optional[str] = None
    ) -> int:
        query = """
            INSERT INTO processing_sessions (session_name, source_type, source_path, started_at, status)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'active');
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (session_name, source_type, str(source_path) if source_path else None))
            conn.commit()
            return cursor.lastrowid

    def finish_processing_session(
        self,
        session_id: int,
        total_frames: int = 0,
        status: str = "completed"
    ) -> None:
        query = """
            UPDATE processing_sessions
            SET ended_at = CURRENT_TIMESTAMP, total_frames = ?, status = ?
            WHERE id = ?;
        """
        with self._get_connection() as conn:
            conn.execute(query, (total_frames, status, session_id))
            conn.commit()

    # --------------------------------------------------------------------------
    # 3. VEHICLES
    # --------------------------------------------------------------------------
    def get_vehicle_by_plate(self, license_plate: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM vehicles WHERE license_plate = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (license_plate.upper().strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_vehicle(
        self,
        license_plate: str,
        vehicle_type: str = "motorbike",
        notes: Optional[str] = None
    ) -> int:
        clean_plate = license_plate.upper().strip()
        query = """
            INSERT INTO vehicles (license_plate, vehicle_type, first_detected_at, last_detected_at, notes)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            ON CONFLICT(license_plate) DO UPDATE SET
                last_detected_at = CURRENT_TIMESTAMP;
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (clean_plate, vehicle_type, notes))
            conn.commit()
            veh = self.get_vehicle_by_plate(clean_plate)
            return veh["id"] if veh else cursor.lastrowid

    # --------------------------------------------------------------------------
    # 4. VIOLATIONS
    # --------------------------------------------------------------------------
    def create_violation(
        self,
        vehicle_id: int,
        rule_id: Optional[int] = None,
        session_id: Optional[int] = None,
        track_id: Optional[int] = None,
        frame_number: int = 0,
        confidence: float = 0.0,
        evidence_image_path: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        query = """
            INSERT INTO violations (
                vehicle_id, rule_id, session_id, track_id, frame_number,
                confidence, evidence_image_path, notes, detected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    vehicle_id,
                    rule_id,
                    session_id,
                    track_id,
                    frame_number,
                    confidence,
                    evidence_image_path,
                    notes
                )
            )
            conn.commit()
            return cursor.lastrowid

    def update_violation_vehicle(self, violation_id: int, vehicle_id: int) -> None:
        """Cập nhật liên kết vehicle_id cho vi phạm sau khi OCR nhận diện thành công biển số."""
        query = "UPDATE violations SET vehicle_id = ? WHERE id = ?;"
        with self._get_connection() as conn:
            conn.execute(query, (vehicle_id, violation_id))
            conn.commit()

    def update_violation_status(self, violation_id: int, status: str) -> bool:
        """Cập nhật trạng thái duyệt vi phạm ('pending', 'approved', 'rejected') cho CSGT."""
        if status not in ("pending", "approved", "rejected"):
            return False
        query = "UPDATE violations SET status = ? WHERE id = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (status, violation_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_violations(
        self,
        limit: int = 100,
        offset: int = 0,
        rule_code: Optional[str] = None,
        plate: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Truy vấn danh sách vi phạm từ View v_violation_details kèm bộ lọc."""
        conditions = []
        params = []

        if rule_code and rule_code != "Tất cả loại lỗi":
            conditions.append("(rule_code = ? OR rule_name = ?)")
            params.extend([rule_code, rule_code])

        if plate and plate.strip():
            conditions.append("license_plate LIKE ?")
            params.append(f"%{plate.strip().upper()}%")

        if start_date:
            conditions.append("date(detected_at) >= date(?)")
            params.append(start_date)

        if end_date:
            conditions.append("date(detected_at) <= date(?)")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM v_violation_details
            {where_clause}
            ORDER BY detected_at DESC
            LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def count_violations(
        self,
        rule_code: Optional[str] = None,
        plate: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        conditions = []
        params = []
        if rule_code and rule_code != "Tất cả loại lỗi":
            conditions.append("(rule_code = ? OR rule_name = ?)")
            params.extend([rule_code, rule_code])
        if plate and plate.strip():
            conditions.append("license_plate LIKE ?")
            params.append(f"%{plate.strip().upper()}%")
        if start_date:
            conditions.append("date(detected_at) >= date(?)")
            params.append(start_date)
        if end_date:
            conditions.append("date(detected_at) <= date(?)")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) AS total FROM v_violation_details {where_clause};"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row["total"] if row else 0

    def get_violation_by_id(self, violation_id: int) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM v_violation_details WHERE id = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (violation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --------------------------------------------------------------------------
    # 5. DASHBOARD & STATS
    # --------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """Tính toán các chỉ số KPI cho Dashboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Tổng số vi phạm
            cursor.execute("SELECT COUNT(*) AS c FROM violations;")
            total_violations = cursor.fetchone()["c"]

            # Vi phạm hôm nay
            cursor.execute("SELECT COUNT(*) AS c FROM violations WHERE date(detected_at) = date('now', 'localtime');")
            violations_today = cursor.fetchone()["c"]

            # Số phương tiện vi phạm
            cursor.execute("SELECT COUNT(DISTINCT vehicle_id) AS c FROM violations;")
            unique_vehicles = cursor.fetchone()["c"]

            # Biển số đọc được (khác UNKNOWN)
            cursor.execute("""
                SELECT COUNT(DISTINCT v.vehicle_id) AS c
                FROM violations v
                JOIN vehicles veh ON veh.id = v.vehicle_id
                WHERE UPPER(veh.license_plate) != 'UNKNOWN';
            """)
            plates_identified = cursor.fetchone()["c"]

            # Phân bố theo loại lỗi
            cursor.execute("""
                SELECT 
                    COALESCE(r.rule_name, 'Khác') AS label,
                    COUNT(v.id) AS count
                FROM violations v
                LEFT JOIN violation_rules r ON r.id = v.rule_id
                GROUP BY label
                ORDER BY count DESC;
            """)
            by_rule = [dict(r) for r in cursor.fetchall()]

            # Phân bố theo giờ trong ngày
            cursor.execute("""
                SELECT 
                    CAST(strftime('%H', detected_at) AS INTEGER) AS hour,
                    COUNT(id) AS count
                FROM violations
                GROUP BY hour
                ORDER BY hour ASC;
            """)
            by_hour = [dict(r) for r in cursor.fetchall()]

            return {
                "total_violations": total_violations,
                "violations_today": violations_today,
                "unique_vehicles": unique_vehicles,
                "plates_identified": plates_identified,
                "by_rule": by_rule,
                "by_hour": by_hour,
            }

    # --------------------------------------------------------------------------
    # 6. EVIDENCE PATH BUILDER
    # --------------------------------------------------------------------------
    @staticmethod
    def build_evidence_path(
        session_id: int,
        violation_id: int,
        rule_code: str,
        ext: str = "jpg"
    ) -> str:
        """Sinh đường dẫn chuẩn data/evidence/YYYYMMDD/{session}_{violation}_{rule}.jpg"""
        day_str = datetime.now().strftime("%Y%m%d")
        filename = f"{session_id}_{violation_id}_{rule_code.upper().strip()}.{ext.lstrip('.')}"
        return os.path.join("data", "evidence", day_str, filename).replace("\\", "/")

    # --------------------------------------------------------------------------
    # 7. RESET DATABASE (DEMO/TESTING)
    # --------------------------------------------------------------------------
    def reset_database(self, confirmation: str) -> bool:
        """Xóa dữ liệu kiểm thử (giữ nguyên bảng violation_rules). Yêu cầu nhập 'XACNHAN'."""
        if confirmation != "XACNHAN":
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM violations;")
            cursor.execute("DELETE FROM processing_sessions;")
            cursor.execute("DELETE FROM vehicles WHERE license_plate != 'UNKNOWN';")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('violations', 'processing_sessions');")
            conn.commit()
            return True

    def reset_database_and_evidence(self, confirmation: str = "XACNHAN") -> bool:
        """Xóa sạch CSDL (giữ nguyên rules) và dọn dẹp các tệp ảnh bằng chứng trong data/evidence."""
        if confirmation != "XACNHAN":
            return False
        success = self.reset_database(confirmation)
        evidence_dir = os.path.join("data", "evidence")
        if os.path.exists(evidence_dir):
            for root, dirs, files in os.walk(evidence_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
        return success
