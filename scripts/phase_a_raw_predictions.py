# Phase A – raw YOLO prediction trên video chẩn đoán (Colab, cần GPU/CPU + ultralytics).
# Không tracking, không lọc ngưỡng của pipeline, không gán nhãn tự động: chỉ đo và xuất crop để BẠN review bằng mắt.
import cv2, csv, numpy as np
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = "/content/helmet_best.pt"                       # bản đang dùng trong repo (so md5 với zip, xem báo cáo)
VIDEOS = ["/content/5pc.mp4", "/content/1pc.mp4", "/content/7pc.mp4"]
STEP, MIN_GAP, TOPK, CONF_FLOOR, REVIEW_CONF = 2, 15, 25, 0.05, 0.45
OUT = Path("/content/diag_out"); OUT.mkdir(exist_ok=True)

m = YOLO(WEIGHTS); print("names:", m.names)
assert [m.names[i] for i in range(7)][:1] == ["driver_with_helmet"], "class mapping lạ"

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

for v in VIDEOS:
    if not Path(v).exists(): print("THIẾU:", v); continue
    cap = cv2.VideoCapture(v); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows, last_saved, i = [], {}, -1
    while True:
        ok, f = cap.read()
        if not ok: break
        i += 1
        if i % STEP: continue
        b = m.predict(f, conf=CONF_FLOOR, imgsz=640, verbose=False)[0].boxes      # raw: không tracker
        xy, c, s = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        for j in range(len(c)):
            if c[j] not in (5, 6) or s[j] < REVIEW_CONF: continue
            rival = max([(s[k], c[k]) for k in range(len(c)) if c[k] in (0, 3) and iou(xy[j], xy[k]) >= 0.5], default=(0.0, -1))
            hh = (xy[j][3] - xy[j][1]) / H
            rows.append(dict(frame=i, cls=int(c[j]), conf=float(s[j]), rival_helmet_conf=float(rival[0]), rival_cls=int(rival[1]),
                             head_h_frac=float(hh), head_h_px_at640=float(hh * 640 * H / max(H, W)), box=[round(float(x), 1) for x in xy[j]]))
    cap.release()
    if not rows: print(f"\n{Path(v).name}: không có class 5/6 >= {REVIEW_CONF}"); continue
    sc = np.array([r["conf"] for r in rows]); rv = np.array([r["rival_helmet_conf"] for r in rows])
    print(f"\n== {Path(v).name} ({W}x{H}) | class5/6 >= {REVIEW_CONF}: {len(rows)} lượt (mỗi {STEP} frame)")
    print(f"   conf p10/50/90: {np.percentile(sc, [10, 50, 90]).round(2).tolist()} | >=0.6: {np.mean(sc >= .6):.0%} | >=0.8: {np.mean(sc >= .8):.0%}")
    print(f"   có box 'có nón' (0/3) IoU>=0.5: {np.mean(rv > 0):.0%} | khi có, conf nón median {np.median(rv[rv > 0]).round(2) if (rv > 0).any() else None}")
    print(f"   chiều cao bbox đầu: median {np.median([r['head_h_frac'] for r in rows]):.3f} ảnh | ~{np.median([r['head_h_px_at640'] for r in rows]):.0f}px ở 640 (so với train ở script dataset)")
    d = OUT / Path(v).stem; d.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(v); saved = 0
    for r in sorted(rows, key=lambda r: -r["conf"]):
        if saved >= TOPK: break
        if any(abs(r["frame"] - fr) < MIN_GAP for fr in last_saved.values()): continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"]); ok, f = cap.read()
        if not ok: continue
        x1, y1, x2, y2 = r["box"]; p = int(0.6 * max(x2 - x1, y2 - y1))                # crop rộng hơn đầu để thấy cả người
        crop = f[max(0, int(y1) - p):int(y2) + p, max(0, int(x1) - p):int(x2) + p]
        cv2.imwrite(str(d / f"f{r['frame']:05d}_c{r['cls']}_{r['conf']:.2f}_rival{r['rival_helmet_conf']:.2f}.jpg"), crop)
        last_saved[saved] = r["frame"]; saved += 1
    cap.release()
    with open(d / "raw_class56.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"   đã xuất {saved} crop + CSV vào {d}")
