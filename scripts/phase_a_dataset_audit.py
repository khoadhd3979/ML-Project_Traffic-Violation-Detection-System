# Phase A – audit bổ sung dataset (chạy trên Colab). Chỉ đọc, không ghi đè dữ liệu.
import yaml, re, cv2, collections, numpy as np
from pathlib import Path

YAML = Path("/content/data_helmet.yaml")          # sửa nếu cần
cfg = yaml.safe_load(YAML.read_text())
names = cfg["names"]; names = dict(enumerate(names)) if isinstance(names, list) else names
root = Path(cfg.get("path", YAML.parent))
HEAD = (0, 3, 5, 6); BIKE = 1
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def dirs(key):
    v = cfg.get(key); v = [v] if isinstance(v, str) else (v or [])
    return [Path(p) if Path(p).is_absolute() else root / p for p in v]

def lbl_path(im):
    s = str(im).replace("/images/", "/labels/")
    return Path(s).with_suffix(".txt")

def read_labels(lp):
    out = []
    if lp.exists():
        for l in lp.read_text().splitlines():
            p = l.split()
            if len(p) >= 5:
                try: out.append((int(float(p[0])), *map(float, p[1:5])))   # cls, xc, yc, w, h (chuẩn hoá)
                except ValueError: pass
    return out

def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)

def ham(a, b): return bin(a ^ b).count("1")

data = {}
for split in ("train", "val"):
    items = []
    for d in dirs(split):
        for im in sorted(d.rglob("*")):
            if im.suffix.lower() in EXT:
                img = cv2.imread(str(im))
                if img is None: continue
                items.append(dict(path=im, stem=im.stem, hw=img.shape[:2], h=dhash(img),
                                  labels=read_labels(lbl_path(im)), img=img if split == "train" else None))
    data[split] = items
    print(f"{split}: {len(items)} ảnh | kích thước (h,w) median:",
          np.median([i['hw'] for i in items], axis=0).tolist() if items else None)

# 1) Near-duplicate (dHash) – train↔val và trong train
tr, va = data["train"], data["val"]
cross = [(a["stem"], b["stem"], ham(a["h"], b["h"])) for a in tr for b in va if ham(a["h"], b["h"]) <= 6]
print(f"\n[1] Cặp train↔val gần trùng (Hamming<=6): {len(cross)}  | ảnh val bị ảnh hưởng: {len({c[1] for c in cross})}/{len(va)}")
for c in cross[:15]: print("   ", c)
seen, clusters = set(), 0
for i, a in enumerate(tr):
    if i in seen: continue
    grp = [j for j, b in enumerate(tr) if ham(a["h"], b["h"]) <= 4]
    seen.update(grp); clusters += 1
print(f"    Trong train: {len(tr)} ảnh ≈ {clusters} cụm gần-trùng (Hamming<=4)")

# 2) Nguồn theo tên file
def src(stem):
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", stem); s = re.sub(r"_(jpg|png|jpeg)$", "", s)
    return re.sub(r"[\s_\-]*\(?\d+\)?$", "", s)
cs = {s: collections.Counter(src(i["stem"]) for i in data[s]) for s in data}
shared = set(cs["train"]) & set(cs["val"])
print(f"\n[2] Nhóm tên file: train {len(cs['train'])} | val {len(cs['val'])} | dùng chung {len(shared)}")
print("    top train:", cs["train"].most_common(8)); print("    top val:", cs["val"].most_common(8))

# 3) Cấu hình người/xe: gán đầu (0/3/5/6) vào xe nếu tâm nằm trong bbox xe mở rộng (xấp xỉ)
def config(labels):
    bikes = [l for l in labels if l[0] == BIKE]; heads = [l for l in labels if l[0] in HEAD]
    res = []
    for _, bx, by, bw, bh in bikes:
        x1, x2, y1, y2 = bx - bw * .6, bx + bw * .6, by - bh * .85, by + bh * .55
        hs = sorted(names[h[0]].replace("_helemt", "").replace("_helmet", "") for h in heads if x1 <= h[1] <= x2 and y1 <= h[2] <= y2)
        res.append(tuple(hs))
    return res
for split in ("train", "val"):
    cnt = collections.Counter(c for i in data[split] for c in config(i["labels"]))
    print(f"\n[3] Cấu hình đầu trên mỗi xe – {split} (xấp xỉ, tổng {sum(cnt.values())} xe):")
    for k, v in cnt.most_common(): print(f"    {v:4d}  {k}")

# 4) Kích thước bbox đầu (tỉ lệ so với ảnh) và độ sáng vùng đầu theo class (CHỈ chẩn đoán, không dùng để gán nhãn)
print("\n[4] Head bbox & độ sáng (train):")
for c in HEAD:
    ws, hs, vs = [], [], []
    for i in tr:
        H, W = i["hw"]
        for cl, xc, yc, w, h in i["labels"]:
            if cl != c: continue
            ws.append(w); hs.append(h)
            x1, x2 = int((xc - w * .3) * W), int((xc + w * .3) * W); y1, y2 = int((yc - h * .3) * H), int((yc + h * .3) * H)
            crop = i["img"][max(0, y1):y2, max(0, x1):x2]
            if crop.size: vs.append(float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2])))
    if ws:
        q = np.percentile(vs, [10, 50, 90]).round(0).tolist() if vs else None
        print(f"    {c} {names[c]:28s} n={len(ws):3d} | w_med={np.median(ws):.3f} h_med={np.median(hs):.3f} | V(p10/50/90)={q} | V<70: {np.mean(np.array(vs) < 70):.0%}")
