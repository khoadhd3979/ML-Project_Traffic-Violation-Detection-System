"""
database_manager.py
====================
Module quản lý truy cập cơ sở dữ liệu SQLite cho hệ thống phát hiện
vi phạm giao thông.

Sử dụng:
    from database_manager import DatabaseManager

    db = DatabaseManager("traffic.db")
    db.init_db("schema.sql")          # chỉ cần chạy 1 lần khi khởi tạo

    vehicle_id = db.create_vehicle(license_plate="59A-12345", vehicle_type="car")
    session_id = db.create_processing_session(session_name="Cam01 - 2026-09-25",
                                               source_type="video_file",
                                               source_path="videos/cam01.mp4")
    violation_id = db.create_violation(vehicle_id=vehicle_id, rule_id=1,
                                        session_id=session_id,
                                        evidence_image_path="data/evidence/...")
"""

import sqlite3
import os
import json
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any


class DatabaseManager:
    def __init__(self, db_path: str = "traffic.db"):
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Kết nối / khởi tạo
    # ------------------------------------------------------------------
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self, schema_path: str = "schema.sql") -> None:
        """Khởi tạo database từ file schema.sql (idempotent nhờ IF NOT EXISTS)."""
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with self._get_connection() as conn:
            conn.executescript(schema_sql)

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return dict(row) if row is not None else None

    @staticmethod
    def _rows_to_list(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
        return [dict(r) for r in rows]

    # ==================================================================
    # VEHICLES
    # ==================================================================
    def create_vehicle(
        self,
        license_plate: str,
        vehicle_type: Optional[str] = None,
        color: Optional[str] = None,
        brand: Optional[str] = None,
        plate_confidence: Optional[float] = None,
    ) -> int:
        """Tạo mới xe. Nếu biển số đã tồn tại thì trả về id của xe đó (upsert nhẹ)."""
        plate = license_plate.strip().upper()
        existing = self.get_vehicle_by_plate(plate)
        if existing:
            return existing["id"]

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO vehicles (license_plate, vehicle_type, color, brand, plate_confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (plate, vehicle_type, color, brand, plate_confidence),
            )
            return cur.lastrowid

    def get_vehicle(self, vehicle_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM vehicles WHERE id = ?", (vehicle_id,)
            ).fetchone()
            return self._row_to_dict(row)

    def get_vehicle_by_plate(self, license_plate: str) -> Optional[Dict[str, Any]]:
        plate = license_plate.strip().upper()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM vehicles WHERE license_plate = ?", (plate,)
            ).fetchone()
            return self._row_to_dict(row)

    def list_vehicles(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM vehicles ORDER BY last_seen_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return self._rows_to_list(rows)

    def update_vehicle(self, vehicle_id: int, **fields) -> bool:
        if not fields:
            return False
        allowed = {"vehicle_type", "color", "brand", "plate_confidence", "last_seen_at"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [vehicle_id]
        with self._get_connection() as conn:
            cur = conn.execute(
                f"UPDATE vehicles SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                values,
            )
            return cur.rowcount > 0

    def delete_vehicle(self, vehicle_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,))
            return cur.rowcount > 0

    # ==================================================================
    # VIOLATION_RULES
    # ==================================================================
    def create_violation_rule(
        self,
        rule_code: str,
        rule_name: str,
        description: Optional[str] = None,
        severity: str = "medium",
        fine_amount_min: Optional[int] = None,
        fine_amount_max: Optional[int] = None,
        legal_reference: Optional[str] = None,
    ) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO violation_rules
                    (rule_code, rule_name, description, severity,
                     fine_amount_min, fine_amount_max, legal_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rule_code, rule_name, description, severity,
                 fine_amount_min, fine_amount_max, legal_reference),
            )
            return cur.lastrowid

    def get_violation_rule(self, rule_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM violation_rules WHERE id = ?", (rule_id,)
            ).fetchone()
            return self._row_to_dict(row)

    def get_violation_rule_by_code(self, rule_code: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM violation_rules WHERE rule_code = ?", (rule_code,)
            ).fetchone()
            return self._row_to_dict(row)

    def list_violation_rules(self, active_only: bool = True) -> List[Dict[str, Any]]:
        query = "SELECT * FROM violation_rules"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY rule_code"
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return self._rows_to_list(rows)

    def deactivate_violation_rule(self, rule_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE violation_rules SET is_active = 0 WHERE id = ?", (rule_id,)
            )
            return cur.rowcount > 0

    # ==================================================================
    # PROCESSING_SESSIONS
    # ==================================================================
    def create_processing_session(
        self,
        session_name: str,
        source_type: str,
        source_path: Optional[str] = None,
        camera_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> int:
        config_json = json.dumps(config, ensure_ascii=False) if config else None
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO processing_sessions
                    (session_name, source_type, source_path, camera_id,
                     location, model_name, model_version, config_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (session_name, source_type, source_path, camera_id,
                 location, model_name, model_version, config_json),
            )
            return cur.lastrowid

    def get_processing_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM processing_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return self._row_to_dict(row)

    def list_processing_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM processing_sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return self._rows_to_list(rows)

    def update_session_stats(
        self,
        session_id: int,
        total_frames_processed: Optional[int] = None,
        total_vehicles_detected: Optional[int] = None,
        total_violations_detected: Optional[int] = None,
    ) -> bool:
        fields = {
            k: v for k, v in {
                "total_frames_processed": total_frames_processed,
                "total_vehicles_detected": total_vehicles_detected,
                "total_violations_detected": total_violations_detected,
            }.items() if v is not None
        }
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [session_id]
        with self._get_connection() as conn:
            cur = conn.execute(
                f"UPDATE processing_sessions SET {set_clause} WHERE id = ?", values
            )
            return cur.rowcount > 0

    def finish_processing_session(
        self, session_id: int, status: str = "completed", error_message: Optional[str] = None
    ) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE processing_sessions
                SET status = ?, error_message = ?, ended_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, error_message, session_id),
            )
            return cur.rowcount > 0

    def delete_processing_session(self, session_id: int, force: bool = False) -> bool:
        """
        Xóa phiên xử lý. Chỉ cho phép xóa nếu:
          - Trạng thái là 'failed' hoặc 'cancelled'
          - Hoặc hoàn thành nhưng không có vi phạm nào (total_violations_detected == 0)
        Dùng force=True để bỏ qua kiểm tra an toàn.
        """
        with self._get_connection() as conn:
            if not force:
                row = conn.execute(
                    "SELECT status, total_violations_detected FROM processing_sessions WHERE id = ?",
                    (session_id,)
                ).fetchone()
                
                if not row:
                    return False
                    
                status = row["status"]
                total_violations = row["total_violations_detected"]
                
                is_safe = (status in ("failed", "cancelled")) or \
                          (status == "completed" and total_violations == 0)
                if not is_safe:
                    return False

            cur = conn.execute("DELETE FROM processing_sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0
    # ==================================================================
    # VIOLATIONS
    # ==================================================================
    def create_violation(
        self,
        vehicle_id: int,
        rule_id: int,
        session_id: Optional[int] = None,
        camera_id: Optional[str] = None,
        location: Optional[str] = None,
        confidence_score: Optional[float] = None,
        frame_number: Optional[int] = None,
        speed_kmh: Optional[float] = None,
        speed_limit_kmh: Optional[float] = None,
        evidence_image_path: Optional[str] = None,
        evidence_video_clip_path: Optional[str] = None,
        detected_at: Optional[str] = None,
    ) -> int:
        detected_at = detected_at or datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO violations
                    (vehicle_id, rule_id, session_id, detected_at, camera_id, location,
                     confidence_score, frame_number, speed_kmh, speed_limit_kmh,
                     evidence_image_path, evidence_video_clip_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (vehicle_id, rule_id, session_id, detected_at, camera_id, location,
                 confidence_score, frame_number, speed_kmh, speed_limit_kmh,
                 evidence_image_path, evidence_video_clip_path),
            )
            return cur.lastrowid

    def get_violation(self, violation_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM violations WHERE id = ?", (violation_id,)
            ).fetchone()
            return self._row_to_dict(row)

    def get_violation_details(self, violation_id: int) -> Optional[Dict[str, Any]]:
        """Lấy vi phạm kèm thông tin xe / luật / session (dùng view v_violation_details)."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM v_violation_details WHERE violation_id = ?", (violation_id,)
            ).fetchone()
            return self._row_to_dict(row)

    def list_violations(
        self,
        vehicle_id: Optional[int] = None,
        rule_id: Optional[int] = None,
        session_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM violations WHERE 1=1"
        params: List[Any] = []
        if vehicle_id is not None:
            query += " AND vehicle_id = ?"
            params.append(vehicle_id)
        if rule_id is not None:
            query += " AND rule_id = ?"
            params.append(rule_id)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY detected_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return self._rows_to_list(rows)

    def update_violation_status(
        self,
        violation_id: int,
        status: str,
        reviewed_by: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        assert status in ("pending", "confirmed", "rejected", "processed")
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE violations
                SET status = ?, reviewed_by = ?, notes = COALESCE(?, notes),
                    reviewed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, reviewed_by, notes, violation_id),
            )
            return cur.rowcount > 0

    def delete_violation(self, violation_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
            return cur.rowcount > 0

    # ==================================================================
    # THỐNG KÊ / TIỆN ÍCH
    # ==================================================================
    def get_violation_stats_by_rule(self) -> List[Dict[str, Any]]:
        query = """
            SELECT r.rule_code, r.rule_name, COUNT(v.id) AS total
            FROM violation_rules r
            LEFT JOIN violations v ON v.rule_id = r.id
            GROUP BY r.id
            ORDER BY total DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return self._rows_to_list(rows)

    def build_evidence_path(
        self, evidence_root: str, session_id: int, violation_id: int,
        rule_code: str, detected_at: Optional[datetime] = None, ext: str = "jpg",
    ) -> str:
        """
        Sinh đường dẫn ảnh bằng chứng theo quy tắc đặt tên thống nhất:

            data/evidence/{YYYYMMDD}/{session_id}_{violation_id}_{rule_code}.{ext}

        Xem chi tiết quy ước trong evidence_naming_convention.md
        """
        dt = detected_at or datetime.now()
        day_folder = dt.strftime("%Y%m%d")
        filename = f"{session_id}_{violation_id}_{rule_code}.{ext}"
        return os.path.join(evidence_root, day_folder, filename)


if __name__ == "__main__":
    # Demo / smoke test nhanh khi chạy trực tiếp file này
    db = DatabaseManager("traffic_demo.db")
    db.init_db(os.path.join(os.path.dirname(__file__), "schema.sql"))
    v_id = db.create_vehicle("51H-99999", vehicle_type="car", color="trắng")
    print("Vehicle:", db.get_vehicle(v_id))
