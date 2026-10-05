import re

with open('tests/integration/live_test.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix sys.path
content = re.sub(r'sys\.path\.insert\(0, os\.path\.dirname\(__file__\)\)', 
                 r'sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))', content)

# Fix imports
content = re.sub(r'from detection import detect_license_plate', r'from src.models.detection import detect_license_plate', content)
content = re.sub(r'from processing import process_plate', r'from src.features.processing import process_plate', content)
content = re.sub(r'from ML\.src\.utils\.config import', r'from src.utils.config import', content)
content = re.sub(r'import database_manager as db', r'from src.data.database_manager import DatabaseManager\ndb_manager = DatabaseManager()', content)

# Fix init_db
content = re.sub(r'conn = db\.init_db\(\)\n    conn\.close\(\)', r'db_manager.init_db("src/data/schema.sql")\n    session_id = db_manager.create_processing_session(session_name="Live Session", source_type="camera", source_path=str(video_source))', content)

# Fix update_detected_plate
replacement = '''
                        # Thay th? db.update_detected_plate(new_text) cu b?ng DatabaseManager m?i
                        vehicle = db_manager.get_vehicle_by_plate(new_text)
                        if not vehicle:
                            vehicle_id = db_manager.create_vehicle(license_plate=new_text, vehicle_type="unknown")
                        else:
                            vehicle_id = vehicle["id"]
                        
                        if v["in_violation_zone"]:
                            db_manager.create_violation(vehicle_id=vehicle_id, rule_id=1, session_id=session_id)
'''
content = re.sub(r'db\.update_detected_plate\(new_text\)', replacement.strip(), content)

# Fix ocr_worker_proc import
content = re.sub(r'import ocr_worker_proc', r'from src.models import ocr_worker_proc', content)

with open('tests/integration/live_test.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("live_test.py updated successfully.")
