import os
import cv2

def extract_frames_from_video(video_path, output_dir, step=4, prefix="frame", max_dim=1920):
    if not os.path.exists(video_path):
        print(f"[Warning] Khong tim thay video tai: {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[*] Dang xu ly: {video_path} ({total_frames} frames, {fps:.1f} FPS)...")

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            # Thu phong nhe neu anh qua lon (nhu 4K) de upload Roboflow nhanh hon
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            filename = f"{prefix}_{count:04d}.jpg"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"[+] Da trich xuat {saved} anh tu {video_path} vao: {output_dir}")
    return saved

def main():
    output_dir = os.path.join("data", "train_raw_frames")
    os.makedirs(output_dir, exist_ok=True)

    print("=== BAT DAU TRICH XUAT FRAME CHO FINE-TUNE YOLO ===")
    # Trich thua (step=15..20) de tranh cac frame gan trung lap lam overfit model.
    # Luu y: KHONG trich frame tu video duoc chon lam Holdout danh gia!
    s1 = extract_frames_from_video("data/raw/1pc.mp4", output_dir, step=15, prefix="1pc", max_dim=1920)
    s2 = extract_frames_from_video("data/raw/5pc.mp4", output_dir, step=15, prefix="5pc", max_dim=1920)

    print(f"\n[HOAN TAT] Tong cong da luu {s1 + s2} anh tai thu muc: {output_dir}")
    print("[Huong dan tiep theo] Hay zip thu muc nay hoac keo tha truc tiep vao Roboflow de gan nhan!")

if __name__ == "__main__":
    main()
