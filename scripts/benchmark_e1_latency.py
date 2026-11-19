"""Benchmark latency/FPS và kích thước cho 4 model E1 trên cùng 1 máy, cùng tập frame.

Cách dùng ví dụ:
    .\\.venv\\Scripts\\python.exe scripts/benchmark_e1_latency.py ^
        --frames-dir data/train_raw_frames ^
        --yolov8n models/weights_helmet/helmet_best.pt ^
        --yolov8s models/exp1_yolov8s/best.pt ^
        --yolov5n models/exp2_yolov5n/best.pt ^
        --rtdetr models/exp3_rtdetr/best.pt
"""
import argparse
import glob
import os
import platform
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_frames(frames_dir: str, limit: int) -> List[np.ndarray]:
    paths = sorted(
        p for p in glob.glob(os.path.join(frames_dir, "**", "*"), recursive=True)
        if os.path.splitext(p)[1].lower() in IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {frames_dir}")

    frames: List[np.ndarray] = []
    for p in paths[:limit]:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
    if not frames:
        raise RuntimeError(f"Không đọc được ảnh hợp lệ nào từ: {frames_dir}")
    return frames


def make_ultralytics_runner(weights_path: str, imgsz: int, device: str, is_rtdetr: bool = False) -> Tuple[Callable[[np.ndarray], None], Optional[int]]:
    from ultralytics import RTDETR, YOLO

    model = RTDETR(weights_path) if is_rtdetr else YOLO(weights_path)
    params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else None

    def run_one(frame: np.ndarray) -> None:
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)

    return run_one, params


def make_yolov5_runner(
    weights_path: str,
    imgsz: int,
    device: str,
    yolov5_repo: Optional[str] = None,
) -> Tuple[Callable[[np.ndarray], None], Optional[int]]:
    if yolov5_repo and os.path.isdir(yolov5_repo):
        model = torch.hub.load(yolov5_repo, "custom", path=weights_path, source="local", trust_repo=True)
    else:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, trust_repo=True)

    if device != "cpu" and torch.cuda.is_available():
        model.to(f"cuda:{device}" if device.isdigit() else device)
    else:
        model.to("cpu")

    params = sum(p.numel() for p in model.parameters())

    def run_one(frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model(rgb, size=imgsz)

    return run_one, params


def benchmark_model(
    label: str,
    weights_path: str,
    runner_fn: Callable[[np.ndarray], None],
    params: Optional[int],
    frames: List[np.ndarray],
    warmup: int,
) -> dict:
    for i in range(warmup):
        runner_fn(frames[i % len(frames)])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for frame in frames:
        runner_fn(frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_frame = (elapsed / len(frames)) * 1000.0
    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    return {
        "label": label,
        "weights": weights_path,
        "size_mb": round(size_mb, 2),
        "params_m": round(params / 1e6, 2) if params else None,
        "ms": round(ms_per_frame, 2),
        "fps": round(fps, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo latency/FPS chung cho 4 mô hình TVDS E1")
    parser.add_argument("--frames-dir", required=True, help="Thư mục chứa tập frame mẫu dùng chung")
    parser.add_argument("--limit", type=int, default=50, help="Số frame dùng để đo (mặc định: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="Số lượt warmup trước khi bấm giờ")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định: 640)")
    parser.add_argument("--device", default="cpu", help="'cpu' hoặc '0' (GPU)")
    parser.add_argument("--yolov8n", help="Đường dẫn weights baseline YOLOv8n (.pt)")
    parser.add_argument("--yolov8s", help="Đường dẫn weights EXP-1 YOLOv8s (.pt)")
    parser.add_argument("--yolov5n", help="Đường dẫn weights EXP-2 YOLOv5n (.pt)")
    parser.add_argument("--rtdetr", help="Đường dẫn weights EXP-3 RT-DETR-L (.pt)")
    parser.add_argument("--yolov5-repo", help="Đường dẫn repo ultralytics/yolov5 local nếu chạy offline")
    args = parser.parse_args()

    frames = load_frames(args.frames_dir, args.limit)
    print(f"Environment: {platform.platform()} | CPU: {platform.processor()} | CUDA: {torch.cuda.is_available()} | Device: {args.device}")
    print(f"Frames: {len(frames)} | imgsz: {args.imgsz} | warmup: {args.warmup}\n")

    specs = [
        ("Baseline YOLOv8n", args.yolov8n, "ultralytics"),
        ("EXP-1 YOLOv8s", args.yolov8s, "ultralytics"),
        ("EXP-2 YOLOv5n", args.yolov5n, "yolov5"),
        ("EXP-3 RT-DETR-L", args.rtdetr, "ultralytics"),
    ]

    results = []
    for label, path, kind in specs:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"[SKIP] Không tìm thấy file weights cho {label}: {path}")
            continue
        if kind == "yolov5":
            runner, params = make_yolov5_runner(path, args.imgsz, args.device, args.yolov5_repo)
        else:
            runner, params = make_ultralytics_runner(path, args.imgsz, args.device, is_rtdetr=(label == "EXP-3 RT-DETR-L"))
        results.append(benchmark_model(label, path, runner, params, frames, args.warmup))

    print("| Mô hình | Size (MB) | Params (M) | Latency (ms/frame) | FPS |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        p_str = f"{r['params_m']:.2f}" if r["params_m"] is not None else "—"
        print(f"| {r['label']} | {r['size_mb']:.2f} | {p_str} | {r['ms']:.2f} | {r['fps']:.2f} |")


if __name__ == "__main__":
    main()
