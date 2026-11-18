"""Regression tests cho src.features.processing và src.models.detection (bản hiện tại trong repo).

- processing.py được import trực tiếp.
- detection.py được nạp từ file thật nhưng với `ultralytics` giả (module thật nạp weights .pt ngay khi import);
  nạp bằng spec_from_file_location nên KHÔNG để lại module giả trong sys.modules.
Expected values dựa trên implementation hiện tại, không điều chỉnh để test pass.
"""
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import processing as P  # noqa: E402


def D(cls, bbox, conf=0.8, tid=None):
    return {"class_id": cls, "bbox": (list(bbox) if bbox is not None else None), "conf": conf, "track_id": tid}


def ids(res):
    return sorted(r["track_id"] for r in res)


BIKE = [100, 200, 200, 300]
RIDER = [115, 120, 185, 260]


# ------------------------------------------------------------------ matching
def test_helmet_suppression_dark_helmet_misread_as_no_helmet():
    # "có nón" mạnh hơn trùng bbox "không nón" -> bỏ "không nón"
    assert P.check_helmet_violation([D(1, BIKE, .9, 1), D(5, RIDER, .5), D(0, RIDER, .7)]) == []
    # "không nón" mạnh hơn -> vẫn vi phạm
    assert ids(P.check_helmet_violation([D(1, BIKE, .9, 1), D(5, RIDER, .7), D(0, RIDER, .5)])) == [1]


def test_helmet_suppression_passenger_class_6_vs_3():
    assert P.check_helmet_violation([D(1, BIKE, .9, 1), D(6, RIDER, .5), D(3, RIDER, .7)]) == []
    assert ids(P.check_helmet_violation([D(1, BIKE, .9, 1), D(6, RIDER, .7), D(3, RIDER, .5)])) == [1]


def test_helmet_suppression_is_not_class_specific_current_behaviour():
    # Đặc tả hành vi HIỆN TẠI: HELMET_CLASSES=(0,3) áp cho cả NO_HELMET_CLASSES=(5,6),
    # nên class 3 (passenger_with_helmet) cũng loại được class 5 (driver_without_helmet).
    assert P.check_helmet_violation([D(1, BIKE, .9, 1), D(5, RIDER, .5), D(3, RIDER, .7)]) == []


def test_suppression_needs_high_iou():
    far_helmet = [160, 120, 230, 260]   # IoU thấp với RIDER
    assert ids(P.check_helmet_violation([D(1, BIKE, .9, 1), D(5, RIDER, .5), D(0, far_helmet, .9)])) == [1]


def test_no_helmet_driver_and_passenger():
    assert ids(P.check_helmet_violation([D(1, BIKE, .9, 1), D(5, RIDER, .7)])) == [1]
    dets = [D(1, BIKE, .9, 7), D(0, [115, 120, 150, 260], .8), D(6, [150, 130, 190, 260], .7)]
    assert ids(P.check_helmet_violation(dets)) == [7]


def test_adjacent_bikes_exclusive_and_ambiguous():
    A = [100, 200, 200, 300]
    B = [190, 200, 290, 300]
    clear = [185, 150, 225, 260]           # rõ ràng gần B hơn
    assert ids(P.check_helmet_violation([D(1, A, .9, 1), D(1, B, .9, 2), D(5, clear, .7)])) == [2]
    midway = [155, 150, 235, 260]          # tâm x=195: cách đều hai xe
    assert P.check_helmet_violation([D(1, A, .9, 1), D(1, B, .9, 2), D(5, midway, .7)]) == []


def test_dense_traffic_three_bikes_one_violator():
    bikes = [[100, 200, 200, 300], [210, 200, 310, 300], [320, 200, 420, 300]]
    dets = []
    for i, b in enumerate(bikes):
        dets.append(D(1, b, .9, i + 1))
        r = [b[0] + 15, 120, b[2] - 15, 260]
        dets.append(D(0 if i != 1 else 5, r, .8 if i != 1 else .7))
    assert ids(P.check_helmet_violation(dets)) == [2]


def test_duplicate_bike_detection_not_treated_as_rival():
    dets = [D(1, BIKE, .9, 1), D(1, [101, 201, 201, 301], .8, 2), D(5, RIDER, .7)]
    assert len(P.check_helmet_violation(dets)) == 1


def test_pedestrian_beside_bike_depends_on_overlap_switch(monkeypatch):
    ped = [170, 200, 260, 320]             # tâm ngoài vùng xe mở rộng, overlap 41%
    dets = [D(1, BIKE, .9, 1), D(5, ped, .7)]
    assert ids(P.check_helmet_violation(dets)) == [1]      # theo yêu cầu (HOẶC) đường overlap-only đang mở
    monkeypatch.setattr(P, "ALLOW_OVERLAP_ONLY_MATCH", False)
    assert P.check_helmet_violation(dets) == []


def test_tiny_and_far_persons_rejected():
    assert P.check_helmet_violation([D(1, BIKE, .9, 1), D(5, [140, 180, 146, 190], .7)]) == []
    assert P.check_helmet_violation([D(1, BIKE, .9, 1), D(5, [600, 100, 660, 260], .9)]) == []


@pytest.mark.parametrize("dets", [
    None, [], [D(1, BIKE, .9, 1)], [D(5, RIDER, .9)], [D(None, RIDER), {"bbox": BIKE}, {}, "x", None],
    [D(1, [], .9, 1), D(5, RIDER)], [D(1, [10, 10, 10, 10], .9, 1), D(5, RIDER)], [D(1, BIKE, .9, 1), D(5, [0, 0, 0, 0])],
    [D(1, BIKE, .9, 1), D(5, [float("nan")] * 4)], [D(1, BIKE, .9, 1), D(5, None)],
    [D(1, [200, 300, 100, 200], .9, 1), D(5, RIDER)],
])
def test_degenerate_inputs(dets):
    assert P.check_helmet_violation(dets) == []


def test_bike_without_track_id_not_reported_not_stolen():
    A = [100, 200, 200, 300]
    B = [190, 200, 290, 300]
    dets = [D(1, A, .9, 1), D(1, B, .9, None), D(5, [185, 150, 225, 260], .7)]
    assert P.check_helmet_violation(dets) == []


# ------------------------------------------------------------------ full_violator_bbox
def test_full_violator_bbox_formula_uses_original_bike_box():
    person = [90, 100, 210, 260]
    r = P.check_helmet_violation([D(1, BIKE, .9, 5), D(5, person, .7)])
    assert r[0]["bbox"] == BIKE
    assert r[0]["full_violator_bbox"] == [90, 100, 210, 300]
    assert set(r[0]) == {"track_id", "bbox", "full_violator_bbox"}


def test_angled_view_person_sticks_out_above_and_beside():
    person = [120, 120, 230, 240]
    r = P.check_helmet_violation([D(1, BIKE, .9, 5), D(5, person, .7)])
    assert ids(r) == [5]
    fb = r[0]["full_violator_bbox"]
    assert fb == [100, 120, 230, 300]
    assert fb[0] <= BIKE[0] and fb[1] <= BIKE[1] and fb[2] >= BIKE[2] and fb[3] >= BIKE[3]
    assert fb[0] <= person[0] and fb[1] <= person[1] and fb[2] >= person[2] and fb[3] >= person[3]


def test_two_violators_on_one_bike_union_both():
    r = P.check_helmet_violation([D(1, BIKE, .9, 7), D(5, [100, 110, 150, 260], .7), D(6, [150, 100, 215, 270], .7)])
    assert r[0]["full_violator_bbox"] == [100, 100, 215, 300]


# ------------------------------------------------------------------ crop_with_padding
def coord_frame(h=400, w=600):
    """Frame int32: kênh 0 = x, kênh 1 = y => đọc lại chính xác vị trí crop trong frame."""
    f = np.zeros((h, w, 3), np.int32)
    f[..., 0] = np.arange(w)[None, :]
    f[..., 1] = np.arange(h)[:, None]
    return f


def test_crop_contains_full_bbox_with_padding():
    c = P.crop_with_padding(coord_frame(), [90, 100, 210, 300])
    h, w = c.shape[:2]
    x0, y0 = int(c[0, 0, 0]), int(c[0, 0, 1])
    # padding 5%: 6px ngang, 10px dọc
    assert (x0, y0, w, h) == (84, 90, 132, 220)
    assert x0 <= 90 and y0 <= 100 and x0 + w >= 210 and y0 + h >= 300


def test_crop_clamped_to_frame_edges():
    f = coord_frame()
    c = P.crop_with_padding(f, [-30, -20, 50, 60])
    assert (int(c[0, 0, 0]), int(c[0, 0, 1])) == (0, 0)
    assert c.shape[:2] == (64, 54)
    c = P.crop_with_padding(f, [550, 350, 700, 500])
    x0, y0 = int(c[0, 0, 0]), int(c[0, 0, 1])
    assert (x0, y0) == (542, 342)
    assert x0 + c.shape[1] == 600 and y0 + c.shape[0] == 400


@pytest.mark.parametrize("bb", [None, [], [10, 10, 10, 10], [50, 50, 10, 10], [float("nan"), 0, 5, 5],
                                [700, 500, 800, 600], [-50, -50, -10, -10]])
def test_crop_invalid_returns_none(bb):
    assert P.crop_with_padding(coord_frame(), bb) is None


def test_crop_none_or_empty_frame():
    assert P.crop_with_padding(None, [0, 0, 5, 5]) is None
    assert P.crop_with_padding(np.zeros((0, 0, 3), np.uint8), [0, 0, 5, 5]) is None


def test_process_plate():
    img = np.zeros((300, 400, 3), np.uint8)
    assert P.process_plate(img, [10, 10, 110, 60]).shape[0] == 200
    assert P.process_plate(img, [500, 500, 600, 600]) is None


# ------------------------------------------------------------------ detection.py (ultralytics giả)
class T:
    def __init__(s, a): s.a = np.asarray(a)
    def cpu(s): return s
    def numpy(s): return s.a


class Boxes:
    def __init__(s, rows, ids=True):
        s.rows = rows
        s.xyxy = T([r[0] for r in rows]); s.cls = T([r[1] for r in rows]); s.conf = T([r[2] for r in rows])
        s.id = T(list(range(1, len(rows) + 1))) if ids else None
    def __len__(s): return len(s.rows)


class FakeYOLO:
    last_kwargs = None
    names = {i: str(i) for i in range(7)}
    next_boxes = None
    def __init__(s, *a, **k): pass
    def track(s, image, **kw):
        FakeYOLO.last_kwargs = kw
        return [types.SimpleNamespace(boxes=FakeYOLO.next_boxes)]


@pytest.fixture
def det(monkeypatch):
    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    path = ROOT / "src" / "models" / "detection.py"
    spec = importlib.util.spec_from_file_location("detection_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_det(mod, rows, ids_=True):
    FakeYOLO.next_boxes = Boxes(rows, ids_)
    return mod.track_vehicles_and_helmets(np.zeros((10, 10, 3), np.uint8))


def test_class_thresholds_bike_035_nohelmet_045_others_040(det):
    bb = [0, 0, 10, 10]
    rows = [(bb, 1, .36), (bb, 1, .34), (bb, 5, .46), (bb, 5, .44), (bb, 6, .44), (bb, 6, .46),
            (bb, 0, .41), (bb, 0, .39), (bb, 3, .41), (bb, 3, .39), (bb, 2, .39), (bb, 4, .41)]
    got = [(d["class_id"], round(d["conf"], 2)) for d in run_det(det, rows)]
    assert got == [(1, .36), (5, .46), (6, .46), (0, .41), (3, .41), (4, .41)]
    assert FakeYOLO.last_kwargs["conf"] == 0.25 and FakeYOLO.last_kwargs["persist"] is True
    assert FakeYOLO.last_kwargs["tracker"] == "bytetrack.yaml"


def test_threshold_comparison_is_strict_current_behaviour(det):
    bb = [0, 0, 10, 10]
    got = run_det(det, [(bb, 1, .35), (bb, 5, .45), (bb, 6, .45), (bb, 0, .40)])
    assert got == []        # implementation dùng conf > ngưỡng (không phải >=)


def test_detection_no_ids_and_empty(det):
    assert run_det(det, [([0, 0, 10, 10], 1, .9)], ids_=False)[0]["track_id"] is None
    FakeYOLO.next_boxes = None
    assert det.track_vehicles_and_helmets(None) == []
    FakeYOLO.next_boxes = Boxes([])
    assert det.track_vehicles_and_helmets(None) == []
