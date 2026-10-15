import sys
import os
import cv2

# 1. Thêm đường dẫn root dự án vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.live_stream import generate_live_stream


def main():
    # Nhận nguồn video qua tham số CLI (mặc định 0 - webcam)
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"[Camera Test] Đã kết nối tới nguồn: {source}. Nhấn 'q' trên cửa sổ cv2 để dừng.")

    try:
        for frame, info in generate_live_stream(source=source):
            cv2.imshow("Helmet Violation Detection - Live Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Info] Người dùng yêu cầu dừng script.")
                break
    except KeyboardInterrupt:
        print("[Info] Đã dừng script từ bàn phím.")
    finally:
        cv2.destroyAllWindows()
        print("[Cleaned] Kết thúc Live Test thành công.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
