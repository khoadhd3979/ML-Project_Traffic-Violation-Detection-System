"""Chạy generate_live_stream() THẬT (src/features/live_stream.py) cùng check_helmet_violation() THẬT
(src/features/processing.py).

Giả lập (mock): YOLO (src.models.detection), OCR worker, DatabaseManager, VideoCapture và multiprocessing.
- Detection trả về theo KỊCH BẢN từng inference cycle (không cần video thật, không sleep).
- mp.Queue/Process được thay bằng bản đồng bộ trong process để kết quả OCR xác định (không race).
- Mọi thay đổi sys.modules đều qua monkeypatch nên được hoàn trả sau mỗi test.
Mỗi frame mã hoá stream-id ở kênh B và chỉ số cycle ở kênh G, để kiểm tra evidence lấy từ đúng frame.
"""
import glob
import importlib.util
import os
import queue
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.features.processing  # noqa: E402,F401  (nạp processing THẬT trước khi patch sys.modules)

LIVE_PATH = ROOT / "src" / "features" / "live_stream.py"
PLATE = "59a1-12345"
PLATE_NORM = "59A1-12345"
INFO_KEYS = {"fps", "violating_count", "latest_event", "recorded_violations", "violating_vehicles"}


# ----------------------------------------------------------------------------- fakes
class FakeDB:
    """Mỗi test có subclass riêng (xem Env) nên calls/lỗi không rò rỉ giữa các test."""
    calls = None
    fail = None
    attempts = None

    def __init__(s):
        s._vid = 0
    def init_db(s, p): pass
    def get_violation_rule_by_code(s, c): return {"id": 1}
    def create_violation_rule(s, **k): return 1
    def create_processing_session(s, **k):
        if s.fail.get("session"):
            raise RuntimeError("session fail")
        return 1
    def get_vehicle_by_plate(s, p):
        for c, k in s.calls:
            if c == "create_vehicle" and k["license_plate"] == p:
                return {"id": k["_id"]}
        return None
    def create_vehicle(s, **k):
        vid = 1 + sum(1 for c, _ in s.calls if c == "create_vehicle")
        s.calls.append(("create_vehicle", dict(k, _id=vid)))
        return vid
    def create_violation(s, **k):
        s.attempts.append(k)
        if s.fail.get("violation", 0) > 0:
            s.fail["violation"] -= 1
            raise RuntimeError("db down")
        s.calls.append(("create_violation", k))
        return 100 + sum(1 for c, _ in s.calls if c == "create_violation")
    def update_violation_vehicle(s, **k): s.calls.append(("update_violation_vehicle", k))
    def finish_processing_session(s, sid): pass


class FakeCap:
    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0
    def isOpened(self): return self.i < len(self.frames)
    def read(self):
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i].copy()
        self.i += 1
        return True, f
    def release(self): pass


class CV2Proxy:
    """Giống cv2 thật, chỉ VideoCapture trả về FakeCap theo `source`."""
    def __init__(self, env): self._env = env
    def VideoCapture(self, source): return self._env.caps[source]
    def __getattr__(self, name): return getattr(cv2, name)


class InQ(queue.Queue):
    """Queue vào của OCR: ghi log và (tuỳ chế độ) trả kết quả ngay hoặc chờ flush."""
    def __init__(self, env, result_q, maxsize):
        super().__init__(maxsize)
        self.env, self.result_q, self.pending = env, result_q, []
    def put_nowait(self, item):
        if item is None:
            return
        self.env.ocr_log.append((self.env.det_calls, item[0]))
        if self.env.ocr_mode == "auto":
            self.result_q.put((item[0], PLATE))
        else:
            self.pending.append(item[0])


class FakeProc:
    def __init__(self, target=None, args=(), daemon=None): self.alive = False
    def start(self): self.alive = True
    def is_alive(self): return self.alive
    def terminate(self): self.alive = False


class FakeMP:
    def __init__(self, env):
        self.env, self._n, self._res = env, 0, None
    def Queue(self, maxsize=0):
        self._n += 1
        if self._n % 2 == 1:                       # live_stream tạo input_q rồi result_q liền nhau
            self._res = queue.Queue()
            q = InQ(self.env, self._res, maxsize)
            self.env.input_queues.append(q)
            return q
        return self._res
    Process = FakeProc


def stream_id_of(frame):
    return (int(frame[0, 0, 0]) - 100) // 50


def make_frames(n, sid):
    frames = []
    for i in range(n):
        f = np.zeros((480, 640, 3), np.uint8)
        f[..., 0] = 100 + 50 * sid
        f[..., 1] = 30 + 10 * i
        frames.append(f)
    return frames


def make_dets(cycle):
    """cycle: list[(track_id, violating)]; phần tử i đặt ở slot i. Xe: x=[20+300i, +140], y=[250,400].
    Người vi phạm (class 5) nhô lên trên xe; người đội nón (class 0) thì không vi phạm."""
    dets = []
    for i, (tid, violating) in enumerate(cycle):
        bx = 20 + 300 * i
        dets.append({"bbox": [float(bx), 250.0, float(bx + 140), 400.0], "track_id": tid, "class_id": 1, "conf": .9})
        dets.append({"bbox": [float(bx + 20), 120.0, float(bx + 130), 330.0], "track_id": 900 + i,
                     "class_id": 5 if violating else 0, "conf": .8})
    return dets


class Run:
    def __init__(self):
        self.infos, self.recorded, self.creates, self.events, self.counts, self.frames = [], [], [], [], [], []


class Env:
    def __init__(self, tmp_path, monkeypatch, confirm=None):
        monkeypatch.delenv("VIOLATION_CONFIRM_FRAMES", raising=False)   # mặc định không phụ thuộc môi trường máy chạy test
        if confirm is not None:
            monkeypatch.setenv("VIOLATION_CONFIRM_FRAMES", confirm)
        self.ocr_mode = "manual"          # "manual": chờ flush_ocr(); "auto": trả biển số ngay
        self.ocr_log, self.input_queues = [], []
        self.det_calls = 0
        self.scripts, self.caps = {}, {}
        self.db = type("DB", (FakeDB,), {"calls": [], "attempts": [], "fail": {}})

        det = types.ModuleType("src.models.detection")
        det.track_vehicles_and_helmets = self._detect
        det.detect_plate_in_crop = lambda crop: [10.0, 10.0, 60.0, 40.0]
        ocr = types.ModuleType("src.models.ocr_worker_proc")
        ocr.run = lambda input_q, result_q: None
        dbm = types.ModuleType("src.data.database_manager")
        dbm.DatabaseManager = self.db
        models = types.ModuleType("src.models"); models.__path__ = []
        models.detection, models.ocr_worker_proc = det, ocr
        data = types.ModuleType("src.data"); data.__path__ = []
        for name, mod in {"src.models": models, "src.models.detection": det, "src.models.ocr_worker_proc": ocr,
                          "src.data": data, "src.data.database_manager": dbm}.items():
            monkeypatch.setitem(sys.modules, name, mod)

        monkeypatch.chdir(tmp_path)
        self.root = tmp_path
        spec = importlib.util.spec_from_file_location(f"live_under_test_{id(self)}", LIVE_PATH)
        self.live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.live)          # nạp file THẬT, bind vào các fake ở trên
        self.live.mp = FakeMP(self)
        self.live.cv2 = CV2Proxy(self)

    def _detect(self, frame):
        self.det_calls += 1
        return make_dets(next(self.scripts[stream_id_of(frame)]))

    def stream(self, cycles, sid=0):
        self.scripts[sid] = iter(cycles)
        src = f"cam{sid}"
        self.caps[src] = FakeCap(make_frames(len(cycles), sid))
        return self.live.generate_live_stream(source=src)

    def flush_ocr(self):
        for q in self.input_queues:
            for tid in q.pending:
                q.result_q.put((tid, PLATE))
            q.pending.clear()

    def snapshot(self, run, frame, info):
        run.infos.append(info)
        run.frames.append(frame)
        run.recorded.append({k: dict(v) for k, v in info["recorded_violations"].items()})
        run.creates.append(sum(1 for c, _ in self.db.calls if c == "create_violation"))
        run.events.append(info["latest_event"])
        run.counts.append(info["violating_count"])

    def run(self, cycles, on_cycle=None, sid=0):
        r = Run()
        for i, (frame, info) in enumerate(self.stream(cycles, sid)):
            self.snapshot(r, frame, info)
            if on_cycle:
                on_cycle(i)
        return r

    def evidence(self):
        return sorted(glob.glob(str(self.root / "data" / "evidence" / "*" / "*.jpg")))

    def calls(self, name):
        return [k for c, k in self.db.calls if c == name]


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


V = [(11, True)]       # xe 11 vi phạm
N = [(11, False)]      # xe 11 có mặt, đội nón
GONE = []              # không có detection


# ----------------------------------------------------------------------------- pipeline hiện có
def test_confirmed_violation_saves_full_person_and_bike_evidence_then_updates_plate(env):
    env.ocr_mode = "auto"
    r = env.run([V] * 6)
    files = env.evidence()
    assert len(files) == 1                                    # chống ghi trùng
    rel = os.path.relpath(files[0], env.root).replace("\\", "/")
    assert rel.startswith("data/evidence/")
    img = cv2.imread(files[0]); h, w = img.shape[:2]
    # union xe [20,250,140+20] + người [40,120,150,330] = [20,120,160,400]; padding 5% -> x:[13,167] y:[106,414]
    assert (w, h) == (154, 308)
    assert w >= 140 and h >= 280                              # chứa đủ xe và người
    assert abs(int(img[:, :, 1].mean()) - (30 + 10 * 2)) <= 4  # lấy từ frame cycle thứ 3 (xác nhận)
    cv = env.calls("create_violation")
    assert len(cv) == 1 and cv[0]["evidence_image_path"].replace("\\", "/") == rel
    assert env.calls("update_violation_vehicle")              # OCR cập nhật sau khi record tồn tại
    assert r.recorded[-1][11]["plate"] == PLATE_NORM
    first = next(i for i in r.infos if i["violating_vehicles"])
    assert set(first["violating_vehicles"][0]) == {"track_id", "bbox", "full_violator_bbox"}
    assert all(set(i) == INFO_KEYS for i in r.infos)


# ----------------------------------------------------------------------------- debouncing
def test_confirm_constant_is_three(env):
    assert env.live.VIOLATION_CONFIRM_FRAMES == 3


def test_confirm_frames_env_override_to_five(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch, confirm="5")
    assert e.live.VIOLATION_CONFIRM_FRAMES == 5
    r = e.run([V] * 6)
    assert r.creates == [0, 0, 0, 0, 1, 1]                    # chỉ xác nhận ở cycle thứ 5
    assert r.counts == [0, 0, 0, 0, 1, 1]
    assert e.ocr_log == [(5, 11)]
    assert len(e.evidence()) == 1


def test_confirm_frames_env_five_gap_resets(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch, confirm="5")
    e.run([V, V, V, V, GONE, V, V, V, V])                      # 4 + 4 không liên tiếp => không xác nhận
    assert e.db.calls == [] and e.evidence() == []


def test_confirm_frames_env_one_confirms_immediately(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch, confirm=" 1 ")              # khoảng trắng được chấp nhận
    r = e.run([V, V])
    assert e.live.VIOLATION_CONFIRM_FRAMES == 1 and r.creates == [1, 1]


@pytest.mark.parametrize("bad", ["", "   ", "abc", "0", "-2", "2.5", "3 frames"])
def test_confirm_frames_env_invalid_falls_back_to_three(tmp_path, monkeypatch, bad):
    e = Env(tmp_path, monkeypatch, confirm=bad)
    assert e.live.VIOLATION_CONFIRM_FRAMES == 3
    r = e.run([V, V, V])
    assert r.creates == [0, 0, 1]


def test_db01_single_cycle_not_recorded(env):
    r = env.run([V, N, N, N])
    assert env.db.calls == [] and env.evidence() == [] and env.ocr_log == []
    assert r.recorded[-1] == {} and r.counts == [0, 0, 0, 0] and all(e is None for e in r.events)


def test_db02_two_consecutive_cycles_not_recorded(env):
    r = env.run([V, V, GONE, GONE])
    assert env.db.calls == [] and env.evidence() == [] and env.ocr_log == []
    assert r.counts == [0, 0, 0, 0]


def test_db03_three_consecutive_cycles_record_exactly_once(env):
    r = env.run([V, V, V])
    assert r.creates == [0, 0, 1]                              # không side effect ở 2 cycle đầu
    assert [len(env.evidence())] == [1]
    assert env.ocr_log == [(3, 11)]                            # OCR chỉ sau xác nhận, ở inference thứ 3
    assert r.counts == [0, 0, 1]
    ev = r.events[2]
    assert ev["track_id"] == 11 and ev["plate"] == "UNKNOWN" and ev["evidence_path"]
    assert r.recorded[2][11]["plate"] == "UNKNOWN" and r.recorded[2][11]["violation_id"]
    assert env.calls("create_violation")[0]["evidence_image_path"] == ev["evidence_path"]
    unknown = [k for k in env.calls("create_vehicle") if k["license_plate"] == "UNKNOWN"]
    assert len(unknown) == 1


def test_confirmed_vehicle_drawn_only_after_confirmation(env):
    r = env.run([V, V, V])
    def has_red(frame):
        return (frame[300, 17:24] == np.array([0, 0, 255])).all(axis=1).any()
    assert [has_red(f) for f in r.frames] == [False, False, True]


@pytest.mark.parametrize("script", [
    [V, GONE, V],
    [V, N, V],
    [V, V, GONE, V, V],
    [V, V, N, V, V],
])
def test_db04_gap_resets_counter(env, script):
    env.run(script)
    assert env.db.calls == [] and env.evidence() == []


def test_db04_counter_restarts_from_zero_after_gap(env):
    r = env.run([V, V, GONE, V, V, V])
    assert r.creates == [0, 0, 0, 0, 0, 1]                     # xác nhận ở cycle thứ 3 SAU khi reset


def test_db05_independent_counters_for_two_vehicles(env):
    two = [(11, True), (12, True)]
    r = env.run([V, two, two, two])
    assert r.creates == [0, 0, 1, 2]                           # 11 xác nhận ở cycle 3, 12 ở cycle 4
    assert sorted(r.recorded[-1]) == [11, 12]
    assert len(env.evidence()) == 2


def test_db05_flickering_vehicle_never_confirms_while_other_does(env):
    r = env.run([[(11, True), (12, True)], [(11, True), (12, False)], [(11, True), (12, True)],
                 [(11, True), (12, False)], [(11, True), (12, True)]])
    assert sorted(r.recorded[-1]) == [11]
    assert len(env.calls("create_violation")) == 1


def test_db06_track_id_none_never_recorded(env):
    r = env.run([[(None, True)]] * 6)
    assert env.db.calls == [] and env.evidence() == [] and env.ocr_log == []
    assert r.recorded[-1] == {} and r.counts == [0] * 6


def test_db06_none_track_does_not_block_valid_track(env):
    r = env.run([[(None, True), (11, True)]] * 4)
    assert sorted(r.recorded[-1]) == [11]
    assert len(env.calls("create_violation")) == 1 and None not in r.recorded[-1]


def test_db07_confirmed_track_not_duplicated(env):
    r = env.run([V] * 3 + [V] * 8 + [GONE] * 2 + [V] * 4)
    assert len(env.calls("create_violation")) == 1 and len(env.evidence()) == 1
    assert sum(1 for e in r.events if e) == 1                  # latest_event chỉ phát một lần (không có OCR)
    assert len(env.ocr_log) <= 1 or all(t == 11 for _, t in env.ocr_log)


def test_db08_generators_do_not_share_candidate_state(env):
    ga = env.stream([V, V, V, V], sid=0)
    gb = env.stream([V, V, GONE, GONE], sid=1)                 # cùng track_id 11 ở hai camera
    ra, rb = Run(), Run()
    for _ in range(4):
        env.snapshot(ra, *next(ga))
        env.snapshot(rb, *next(gb))
        if len(ra.infos) == 2:
            assert env.calls("create_violation") == []         # 2+2 cycle gộp lại sẽ >= 3 nếu state bị chia sẻ
    assert len(env.calls("create_violation")) == 1
    assert sorted(ra.recorded[-1]) == [11] and rb.recorded[-1] == {}
    assert ra.infos[0]["recorded_violations"] is not rb.infos[0]["recorded_violations"]


def test_db09_ocr_result_after_violation_updates_plate(env):
    def on_cycle(i):
        if i == 3:
            env.flush_ocr()                                    # OCR trả kết quả SAU khi record đã ghi
    r = env.run([V] * 6, on_cycle)
    names = [c for c, _ in env.db.calls]
    assert names.index("create_violation") < names.index("update_violation_vehicle")
    assert r.recorded[3][11]["plate"] == "UNKNOWN"             # trước khi OCR về vẫn UNKNOWN
    assert r.recorded[4][11]["plate"] == PLATE_NORM
    upd = env.calls("update_violation_vehicle")
    assert len(upd) == 1 and upd[0]["violation_id"] == r.recorded[4][11]["violation_id"]
    real = [k for k in env.calls("create_vehicle") if k["license_plate"] == PLATE_NORM]
    assert len(real) == 1 and upd[0]["vehicle_id"] == real[0]["_id"]
    assert r.events[4]["plate"] == PLATE_NORM
    assert len(env.calls("create_violation")) == 1


def test_ocr_result_before_confirmation_cannot_exist(env):
    env.ocr_mode = "auto"
    env.run([V, V, GONE, GONE])
    assert env.ocr_log == [] and env.calls("update_violation_vehicle") == []


# ----------------------------------------------------------------------------- lỗi DB & ngữ nghĩa inference
def test_db_failure_not_marked_recorded_and_retried_without_duplicate_evidence(env):
    env.db.fail["violation"] = 1
    r = env.run([V] * 6)
    assert len(env.db.attempts) == 2                           # lần 1 lỗi, lần 2 (cycle 4) thành công
    assert r.recorded[2] == {} and r.creates[2] == 0 and r.events[2] is None
    assert 11 in r.recorded[3] and r.creates[3] == 1
    assert env.ocr_log and env.ocr_log[0][0] == 4              # OCR chỉ sau khi record tồn tại
    assert len(env.calls("create_violation")) == 1
    files = env.evidence()
    assert len(files) == 1                                     # retry dùng lại file, không lưu trùng
    assert env.calls("create_violation")[0]["evidence_image_path"].endswith(os.path.basename(files[0]))


def test_db_failure_then_track_lost_removes_orphan_evidence(env):
    env.db.fail["violation"] = 99
    env.run([V, V, V, GONE])
    assert env.calls("create_violation") == [] and env.evidence() == []


def test_no_processing_session_means_nothing_marked_recorded(env):
    env.db.fail["session"] = True
    r = env.run([V] * 5)
    assert r.recorded[-1] == {} and env.calls("create_violation") == [] and env.ocr_log == []
    assert len(env.evidence()) <= 1


def test_inference_runs_once_per_frame_no_stale_reuse(env):
    env.run([V] * 5)
    assert env.det_calls == 5
