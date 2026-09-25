"""
init_database.py
Khởi tạo cấu trúc cơ sở dữ liệu trống từ file schema.sql.
Lưu ý: Chạy file này sẽ xóa database cũ (nếu có) để tạo lại từ đầu.
"""
import os
from database_manager import DatabaseManager

DB_PATH = "traffic.db"
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def init_database():
    # Xóa file cũ nếu đã tồn tại để tạo mới hoàn toàn
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Đã xóa file database cũ: {DB_PATH}")
    
    db = DatabaseManager(DB_PATH)
    db.init_db(SCHEMA_PATH)
    print(f"Đã khởi tạo thành công database trống tại {DB_PATH} dựa trên {SCHEMA_PATH}.")

if __name__ == "__main__":
    init_database()