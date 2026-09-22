# import sqlite3
# import datetime
# import os
# import logging
# import threading

# # CẤU HÌNH — đọc từ config.py (nguồn chân lý duy nhất)
# try:
#     from config import DB_PATH
# except ImportError:
#     # Fallback khi chạy database_manager.py trực tiếp ngoài thư mục modules/
#     DB_PATH = os.environ.get("VIOLATION_DB_PATH", "violations.db")

# # Định dạng timestamp thống nhất toàn hệ thống (ISO-8601 microseconds)
# DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(asctime)s] %(levelname)s %(message)s",
#     datefmt="%H:%M:%S",
# )
# log = logging.getLogger("violation_db")


# # KHỞI TẠO DATABASE
# def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
#     conn = sqlite3.connect(db_path, check_same_thread=False)
#     conn.row_factory = sqlite3.Row          # truy cập cột bằng tên
#     conn.execute("PRAGMA journal_mode=WAL") # ghi an toàn khi nhiều reader
#     conn.execute("PRAGMA foreign_keys=ON")

#     conn.executescript("""
#         -- Bảng ghi nhận vĩnh viễn các sự kiện vi phạm không gian/thời gian
#         CREATE TABLE IF NOT EXISTS traffic_violations (
#             violation_id        INTEGER PRIMARY KEY AUTOINCREMENT,
#             track_id            INTEGER NOT NULL,
#             plate_number        TEXT    NOT NULL,
#             violation_time      TEXT    NOT NULL,   -- ISO-8601 microseconds
#             violation_type      TEXT    NOT NULL,   -- vd: "Sai làn", "Không đội mũ"
#             image_evidence_path TEXT
#         );

#         -- Index tăng tốc truy vấn tra cứu theo biển số / track_id
#         CREATE INDEX IF NOT EXISTS idx_plate_number
#             ON traffic_violations (plate_number);

#         CREATE INDEX IF NOT EXISTS idx_track_id
#             ON traffic_violations (track_id);
#     """)

#     conn.commit()
#     log.info("Database khởi tạo thành công: %s", os.path.abspath(db_path))
#     return conn


# # TIỆN ÍCH NỘI BỘ
# class SQLiteConnectionWrapper:
#     """
#     Bọc sqlite3.Connection để biến hàm close() thành no-op (bỏ qua việc đóng),
#     giúp giữ kết nối dài hạn và tái sử dụng an toàn mà không cần sửa code ở nơi gọi.
#     """
#     def __init__(self, conn):
#         self._conn = conn

#     def __getattr__(self, name):
#         return getattr(self._conn, name)

#     def __enter__(self):
#         return self._conn.__enter__()

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         return self._conn.__exit__(exc_type, exc_val, exc_tb)

#     def close(self):
#         # Không làm gì cả để giữ kết nối dùng chung luôn mở
#         pass


# _local_db = threading.local()


# def _get_connection(db_path: str = DB_PATH) -> SQLiteConnectionWrapper:
#     """
#     Tái sử dụng kết nối trong từng luồng (Thread-local connection)
#     và trả về đối tượng bọc để chống đóng kết nối sớm.
#     """
#     if not hasattr(_local_db, "conn") or _local_db.conn is None:
#         try:
#             # Mở kết nối SQLite dài hạn cho Thread này
#             conn = sqlite3.connect(db_path, check_same_thread=False)
#             conn.row_factory = sqlite3.Row
#             conn.execute("PRAGMA journal_mode=WAL")
#             conn.execute("PRAGMA foreign_keys=ON")
#             _local_db.conn = conn
#         except sqlite3.Error as e:
#             log.error(f"[SQLite Connection] Không thể kết nối DB: {e}")
#             raise e

#     # Bọc kết nối lại để các hàm bên ngoài khi gọi conn.close() không thực sự đóng kết nối
#     return SQLiteConnectionWrapper(_local_db.conn)


# # API CÔNG KHAI
# def log_violation(
#     track_id: int,
#     plate_number: str,
#     violation_type: str,
#     image_evidence_path: str = None,
#     db_path: str = DB_PATH,
# ) -> dict:
#     """
#     Điểm tích hợp chính — gọi từ luồng xử lý camera khi phát hiện một
#     sự kiện vi phạm giao thông đã được xác nhận (biển số đã OCR xong).

#     Tham số
#     -------
#     track_id             : ID theo dõi của phương tiện trong hệ thống tracking đa đối tượng
#     plate_number         : biển số xe (đã qua hậu xử lý RegEx từ OCR)
#     violation_type       : loại vi phạm, vd "Sai làn", "Không đội mũ", "Vượt đèn đỏ"
#     image_evidence_path  : đường dẫn ảnh bằng chứng đã lưu trên đĩa (tuỳ chọn)

#     Trả về
#     ------
#     dict với các key:
#         status         : "LOGGED" | "ERROR"
#         violation_id   : mã bản ghi vi phạm (None nếu lỗi)
#         track_id       : ID theo dõi
#         plate_number   : biển số
#         violation_type : loại vi phạm
#         violation_time : thời điểm ghi nhận (ISO-8601)
#         message        : mô tả kết quả (tiếng Việt)

#     Lưu ý
#     -----
#     Đây là bản ghi VĨNH VIỄN — không có khái niệm check-in/check-out,
#     mỗi lần gọi tạo ra một dòng mới, không cập nhật hay ghi đè bản ghi cũ.
#     """
#     plate_number = (plate_number or "").strip().upper()

#     if not plate_number:
#         return _error_result(track_id, "", violation_type, "Biển số rỗng — bỏ qua.")

#     if not violation_type:
#         return _error_result(track_id, plate_number, "", "Loại vi phạm rỗng — bỏ qua.")

#     conn = _get_connection(db_path)
#     try:
#         now     = datetime.datetime.now()
#         now_str = now.strftime(DT_FORMAT)

#         cursor = conn.execute(
#             """INSERT INTO traffic_violations
#                    (track_id, plate_number, violation_time, violation_type, image_evidence_path)
#                VALUES (?, ?, ?, ?, ?)""",
#             (track_id, plate_number, now_str, violation_type, image_evidence_path),
#         )
#         conn.commit()
#         violation_id = cursor.lastrowid

#         log.info(
#             "VI PHẠM GHI NHẬN | track_id=%s | %s | %s | vi_phạm=%s | bằng_chứng=%s",
#             track_id, plate_number, now_str, violation_type, image_evidence_path,
#         )

#         return {
#             "status":         "LOGGED",
#             "violation_id":   violation_id,
#             "track_id":       track_id,
#             "plate_number":   plate_number,
#             "violation_type": violation_type,
#             "violation_time": now_str,
#             "message":        f"Đã ghi nhận vi phạm #{violation_id} cho biển số {plate_number}.",
#         }

#     except sqlite3.Error as exc:
#         conn.rollback()
#         log.error("DB error khi ghi vi phạm %s (track_id=%s): %s", plate_number, track_id, exc)
#         return _error_result(track_id, plate_number, violation_type, f"Lỗi cơ sở dữ liệu: {exc}")

#     finally:
#         conn.close()


# # CÁC HÀM TRUY VẤN (dùng cho Dashboard / báo cáo)
# def get_recent_violations(limit: int = 20, db_path: str = DB_PATH) -> list:
#     """Lấy N bản ghi vi phạm gần nhất."""
#     conn = _get_connection(db_path)
#     try:
#         rows = conn.execute(
#             """SELECT violation_id, track_id, plate_number, violation_time,
#                       violation_type, image_evidence_path
#                FROM traffic_violations
#                ORDER BY violation_id DESC
#                LIMIT ?""",
#             (limit,),
#         ).fetchall()
#         return [dict(row) for row in rows]
#     finally:
#         conn.close()


# def get_violations_by_plate(plate_number: str, db_path: str = DB_PATH) -> list:
#     """Lấy toàn bộ lịch sử vi phạm của một biển số cụ thể."""
#     plate_number = plate_number.strip().upper()
#     conn = _get_connection(db_path)
#     try:
#         rows = conn.execute(
#             """SELECT violation_id, track_id, plate_number, violation_time,
#                       violation_type, image_evidence_path
#                FROM traffic_violations
#                WHERE plate_number = ?
#                ORDER BY violation_time DESC""",
#             (plate_number,),
#         ).fetchall()
#         return [dict(row) for row in rows]
#     finally:
#         conn.close()


# def get_violation_counts_by_type(db_path: str = DB_PATH) -> list:
#     """Thống kê số lượt vi phạm theo từng loại — dùng cho Dashboard."""
#     conn = _get_connection(db_path)
#     try:
#         rows = conn.execute(
#             """SELECT violation_type, COUNT(*) AS total
#                FROM traffic_violations
#                GROUP BY violation_type
#                ORDER BY total DESC"""
#         ).fetchall()
#         return [dict(row) for row in rows]
#     finally:
#         conn.close()


# # TIỆN ÍCH NỘI BỘ
# def _error_result(track_id, plate_number: str, violation_type: str, message: str) -> dict:
#     return {
#         "status":         "ERROR",
#         "violation_id":   None,
#         "track_id":       track_id,
#         "plate_number":   plate_number,
#         "violation_type": violation_type,
#         "violation_time": datetime.datetime.now().strftime(DT_FORMAT),
#         "message":        message,
#     }


# # CHẠY TRỰC TIẾP — kiểm tra nhanh khi debug
# if __name__ == "__main__":
#     import json

#     print("=" * 60)
#     print("Khởi tạo database …")
#     conn = init_db()
#     conn.close()

#     print("\n--- Kiểm tra ghi nhận vi phạm ---")
#     r = log_violation(
#         track_id=101,
#         plate_number="51G-12345",
#         violation_type="Sai làn",
#         image_evidence_path="/evidence/track_101_20260918.jpg",
#     )
#     print(json.dumps(r, ensure_ascii=False, indent=2))

#     print("\n--- Kiểm tra ghi nhận vi phạm khác cho cùng track_id/biển số ---")
#     r = log_violation(
#         track_id=101,
#         plate_number="51G-12345",
#         violation_type="Không đội mũ",
#     )
#     print(json.dumps(r, ensure_ascii=False, indent=2))

#     print("\n--- 5 bản ghi gần nhất ---")
#     for rec in get_recent_violations(5):
#         print(rec)

#     print("\n--- Thống kê theo loại vi phạm ---")
#     print(json.dumps(get_violation_counts_by_type(), ensure_ascii=False, indent=2))
import sqlite3
import datetime
import os
import logging
import threading

# CẤU HÌNH
DB_PATH = "parking.db"
DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("db_manager")

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # 1. BẢNG HỆ THỐNG GIAO THÔNG (MỚI)[cite: 5]
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS traffic_violations (
            violation_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id            INTEGER NOT NULL,
            plate_number        TEXT    NOT NULL,
            violation_time      TEXT    NOT NULL,
            violation_type      TEXT    NOT NULL,
            image_evidence_path TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_plate_number ON traffic_violations (plate_number);
        CREATE INDEX IF NOT EXISTS idx_track_id ON traffic_violations (track_id);
        
        CREATE TABLE IF NOT EXISTS temp_plate (
            id          INTEGER PRIMARY KEY,
            plate_text  TEXT    NOT NULL,
            detected_at TEXT    NOT NULL
        );
    """)

    # 2. BẢNG BÃI ĐỖ XE (CŨ)[cite: 4] - Đã sửa lỗi NULL cột status
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS parking_logs (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            time_in TEXT NOT NULL,
            time_out TEXT,
            status TEXT NOT NULL DEFAULT 'IN', 
            fee REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY,
            max_capacity INTEGER DEFAULT 100
        );
    """)
    
    # Khởi tạo cấu hình mặc định nếu chưa có
    if not conn.execute("SELECT 1 FROM system_config WHERE id = 1").fetchone():
        conn.execute("INSERT INTO system_config (id, max_capacity) VALUES (1, 100)")
        
    conn.commit()
    log.info("Database hợp nhất khởi tạo thành công: %s", os.path.abspath(db_path))
    return conn

class SQLiteConnectionWrapper:
    def __init__(self, conn): self._conn = conn
    def __getattr__(self, name): return getattr(self._conn, name)
    def close(self): pass

_local_db = threading.local()

def _get_connection(db_path: str = DB_PATH) -> SQLiteConnectionWrapper:
    if not hasattr(_local_db, "conn") or _local_db.conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local_db.conn = conn
    return SQLiteConnectionWrapper(_local_db.conn)

# ==========================================
# PHẦN 1: CÁC HÀM QUẢN LÝ BÃI ĐỖ XE (HỖ TRỢ UI CŨ)[cite: 4]
# ==========================================

def get_status(db_path: str = DB_PATH) -> dict:
    conn = _get_connection(db_path)
    cap = conn.execute("SELECT max_capacity FROM system_config WHERE id = 1").fetchone()[0]
    vehicles = conn.execute("SELECT * FROM parking_logs WHERE status = 'IN'").fetchall()
    return {
        "max_capacity": cap,
        "occupancy": len(vehicles),
        "vehicles_inside": [dict(v) for v in vehicles]
    }

def get_recent_logs(limit: int = 20, db_path: str = DB_PATH) -> list:
    conn = _get_connection(db_path)
    rows = conn.execute("SELECT * FROM parking_logs ORDER BY time_in DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]

def get_revenue_today(db_path: str = DB_PATH) -> dict:
    conn = _get_connection(db_path)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    rows = conn.execute("SELECT fee FROM parking_logs WHERE status = 'OUT' AND time_out LIKE ?", (f"{today_str}%",)).fetchall()
    return {"revenue": sum(r["fee"] for r in rows), "checkouts": len(rows)}

def set_max_capacity(new_capacity: int, db_path: str = DB_PATH) -> bool:
    conn = _get_connection(db_path)
    occ = conn.execute("SELECT COUNT(*) FROM parking_logs WHERE status = 'IN'").fetchone()[0]
    if new_capacity < occ: return False
    conn.execute("UPDATE system_config SET max_capacity = ? WHERE id = 1", (new_capacity,))
    conn.commit()
    return True

def process_vehicle(plate_number: str, db_path: str = DB_PATH) -> dict:
    conn = _get_connection(db_path)
    now_str = datetime.datetime.now().isoformat()
    
    # Kiểm tra xe đang ở trong bãi
    in_db = conn.execute("SELECT * FROM parking_logs WHERE plate_number = ? AND status = 'IN'", (plate_number,)).fetchone()
    
    if in_db: # Thực hiện Check-out
        fee = 50000 # Phí giả định
        conn.execute("UPDATE parking_logs SET status = 'OUT', time_out = ?, fee = ? WHERE ticket_id = ?", (now_str, fee, in_db["ticket_id"]))
        conn.commit()
        return {"status": "CHECK-OUT", "ticket_id": in_db["ticket_id"], "fee": fee, "hours": 2, "minutes": 30, "time_in": in_db["time_in"]}
    else: # Thực hiện Check-in
        cap = conn.execute("SELECT max_capacity FROM system_config WHERE id = 1").fetchone()[0]
        occ = conn.execute("SELECT COUNT(*) FROM parking_logs WHERE status = 'IN'").fetchone()[0]
        if occ >= cap: return {"status": "FULL", "message": "Bãi xe đã đầy chỗ!"}
        
        cursor = conn.execute("INSERT INTO parking_logs (plate_number, time_in, status) VALUES (?, ?, 'IN')", (plate_number, now_str))
        conn.commit()
        return {"status": "CHECK-IN", "ticket_id": cursor.lastrowid}

# ==========================================
# PHẦN 2: CÁC HÀM GIAO THÔNG & CAMERA (MỚI)[cite: 5]
# ==========================================

def log_violation(track_id: int, plate_number: str, violation_type: str, image_evidence_path: str = None, db_path: str = DB_PATH) -> dict:
    plate_number = (plate_number or "").strip().upper()
    if not plate_number or not violation_type: return {"status": "ERROR"}

    conn = _get_connection(db_path)
    now_str = datetime.datetime.now().strftime(DT_FORMAT)
    cursor = conn.execute(
        "INSERT INTO traffic_violations (track_id, plate_number, violation_time, violation_type, image_evidence_path) VALUES (?, ?, ?, ?, ?)",
        (track_id, plate_number, now_str, violation_type, image_evidence_path),
    )
    conn.commit()
    return {"status": "LOGGED", "violation_id": cursor.lastrowid}

def update_detected_plate(plate_text: str, db_path: str = DB_PATH) -> None:
    conn = _get_connection(db_path)
    conn.execute("DELETE FROM temp_plate")
    conn.execute("INSERT INTO temp_plate (id, plate_text, detected_at) VALUES (1, ?, ?)", (plate_text, datetime.datetime.now().strftime(DT_FORMAT)))
    conn.commit()

def get_detected_plate(db_path: str = DB_PATH):
    conn = _get_connection(db_path)
    row = conn.execute("SELECT plate_text FROM temp_plate WHERE id = 1").fetchone()
    return row["plate_text"] if row else None

def clear_detected_plate(db_path: str = DB_PATH) -> None:
    conn = _get_connection(db_path)
    conn.execute("DELETE FROM temp_plate WHERE id = 1")
    conn.commit()