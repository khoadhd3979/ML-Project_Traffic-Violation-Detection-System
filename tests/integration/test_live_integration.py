"""Chạy generate_live_stream THẬT (file live_stream.py) với video tổng hợp.
Giả lập (mock): src.models.detection (YOLO), OCR worker, DatabaseManager. processing.py là bản thật."""
import importlib.util, sys, types, os, glob, time, tempfile, multiprocessing as mp
import numpy as np, cv2, pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BIKE = [250, 250, 390, 400]
PERSON = [270, 120, 380, 330]          # nhô lên trên box xe (góc chéo/đầu cao)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class FakeDB:
    calls = []
    def __init__(s): s._vid = 0
    def init_db(s, p): pass
    def get_violation_rule_by_code(s, c): return {"id": 1}
    def create_violation_rule(s, **k): return 1
    def create_processing_session(s, **k): return 1
    def get_vehicle_by_plate(s, p): return None
    def create_vehicle(s, **k):
        s._vid += 1; FakeDB.calls.append(("create_vehicle", k)); return s._vid
    def create_violation(s, **k): FakeDB.calls.append(("create_violation", k)); return 100
    def update_violation_vehicle(s, **k): FakeDB.calls.append(("update_violation_vehicle", k))
    def finish_processing_session(s, sid): pass

def fake_ocr_run(input_q, result_q):
    while True:
        item = input_q.get()
        if item is None: break
        tid, img = item
        result_q.put((tid, "59a1-12345"))

def run_pipeline(proc_path, live_path, tmp):
    FakeDB.calls = []
    proc = load(proc_path, "proc_under_test")
    det = types.ModuleType("src.models.detection")
    det.track_vehicles_and_helmets = lambda f: [
        {"bbox": [float(x) for x in BIKE], "track_id": 11, "class_id": 1, "conf": .9},
        {"bbox": [float(x) for x in PERSON], "track_id": 12, "class_id": 5, "conf": .8}]
    det.detect_plate_in_crop = lambda crop: [10.0, 10.0, 60.0, 40.0]
    ocr = types.ModuleType("src.models.ocr_worker_proc"); ocr.run = fake_ocr_run
    dbm = types.ModuleType("src.data.database_manager"); dbm.DatabaseManager = FakeDB
    pkgs = {}
    for n in ("src","src.models","src.features","src.data"):
        pkgs[n] = types.ModuleType(n); pkgs[n].__path__ = []
    pkgs["src.models"].detection = det; pkgs["src.models"].ocr_worker_proc = ocr
    pkgs["src.features"].processing = proc
    sys.modules.update({**pkgs, "src.models.detection": det, "src.models.ocr_worker_proc": ocr,
                        "src.features.processing": proc, "src.data.database_manager": dbm})
    live = load(live_path, "live_under_test")

    video = os.path.join(tmp, "v.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for i in range(40):
        f = np.full((480, 640, 3), 90, np.uint8); cv2.rectangle(f, (BIKE[0],BIKE[1]), (BIKE[2],BIKE[3]), (200,50,50), -1)
        cv2.rectangle(f, (PERSON[0],PERSON[1]), (PERSON[2],PERSON[3]), (50,200,50), 2); vw.write(f)
    vw.release()

    cwd = os.getcwd(); os.chdir(tmp)
    try:
        infos = []
        for frame, info in live.generate_live_stream(video):
            infos.append(info); time.sleep(0.03)
    finally:
        os.chdir(cwd)
    return infos

def evidence_files(tmp): return sorted(glob.glob(os.path.join(tmp, "data", "evidence", "*", "*.jpg")))

def test_new_pipeline_saves_full_person_and_bike_evidence_then_updates_plate():
    with tempfile.TemporaryDirectory() as tmp:
        infos = run_pipeline(f"{HERE}/new/processing.py", f"{HERE}/new/live_stream.py", tmp)
        files = evidence_files(tmp)
        assert len(files) == 1                                   # chống ghi trùng vẫn hoạt động
        img = cv2.imread(files[0]); h, w = img.shape[:2]
        # union = [250,120,390,400]; padding 5% -> x:[243,397] y:[106,414]
        assert (w, h) == (154, 308)
        assert w >= 390-250 and h >= 400-120                    # chứa đủ xe và người
        cv = [c for c in FakeDB.calls if c[0] == "create_violation"]
        assert len(cv) == 1 and cv[0][1]["evidence_image_path"].replace("\\", "/").endswith(os.path.basename(files[0]))
        assert any(c[0] == "update_violation_vehicle" for c in FakeDB.calls)   # OCR cập nhật sau
        assert any(i["recorded_violations"].get(11, {}).get("plate") == "59A1-12345" for i in infos)
        assert set(infos[0]["violating_vehicles"][0]) == {"track_id", "bbox", "full_violator_bbox"}

def test_original_pipeline_saves_vehicle_only_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        run_pipeline(f"{HERE}/orig/processing.py", f"{HERE}/orig/live_stream.py", tmp)
        img = cv2.imread(evidence_files(tmp)[0]); h, w = img.shape[:2]
        assert (w, h) == (140, 150)          # chỉ xe: thiếu phần đầu/thân người nhô lên (y 120..250)
