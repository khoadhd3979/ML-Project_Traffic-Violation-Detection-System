import os
import tempfile
import cv2
import streamlit as st

from app.components.violation_table import VIOLATION_NAMES
from src.features.live_stream import generate_live_stream


def _save_upload(uploaded_file) -> str:
    """Lưu video tạm đúng một lần cho mỗi file (Streamlit rerun sau mỗi thao tác)."""
    key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_video_key") != key:
        old = st.session_state.get("_video_path")
        if old and os.path.exists(old):
            os.remove(old)
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["_video_key"] = key
        st.session_state["_video_path"] = f.name
    return st.session_state["_video_path"]


def _run_video(path, selected, settings, pipeline, frame_slot, info_slot, progress):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        extra = {}
        if pipeline is not None:
            frame, extra = pipeline(frame, selected, settings)
        if idx % 2 == 0:  # giảm tải render
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            info_slot.markdown(f"Khung hình **{idx}/{total}**, đối tượng **{extra.get('objects', 0)}**, "
                               f"vi phạm **{extra.get('violations', 0)}**")
            progress.progress(min(idx / total, 1.0), text=f"Đang xử lý {idx}/{total}")
    cap.release()
    progress.progress(1.0, text="Hoàn tất")


def render_live_monitor(df, settings, pipeline=None):
    """pipeline(frame, selected_codes, settings) -> (annotated_frame, {"objects": n, "violations": m}).
    Để None thì chỉ phát lại video gốc."""
    left, right = st.columns([2.3, 1], gap="medium")

    with right:
        source = st.radio("Nguồn", ["Tệp video", "Camera trực tiếp"], horizontal=True,
                          label_visibility="collapsed")
        uploaded = None
        stream_url = "0"
        if source == "Tệp video":
            uploaded = st.file_uploader("Video đầu vào (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        else:
            stream_url = st.text_input("Địa chỉ luồng / Webcam ID", value="0", help="Nhập 0 cho webcam máy tính hoặc URL RTSP/HTTP")

        selected = st.multiselect("Loại vi phạm cần phát hiện", list(VIOLATION_NAMES),
                                  default=list(VIOLATION_NAMES)[:3], format_func=VIOLATION_NAMES.get)
        info_slot = st.empty()
        info_slot.markdown("Khung hình **0**, đối tượng **0**, vi phạm **0**")
        st.markdown("**Sự kiện gần nhất**")
        recent = (df.sort_values("Time", ascending=False).head(5)[["Time", "Plate", "Violation"]]
                    .rename(columns={"Time": "Giờ", "Plate": "Biển số", "Violation": "Lỗi"}))
        st.dataframe(recent, hide_index=True, height=200, width="stretch",
                     column_config={"Giờ": st.column_config.DatetimeColumn(format="HH:mm:ss")})

    with left:
        frame_slot = st.empty()
        progress = st.progress(0, text="Sẵn sàng")

        if source == "Tệp video":
            run = st.button("Chạy nhận diện", type="primary", disabled=uploaded is None or not selected)
            path = _save_upload(uploaded) if uploaded is not None else None
            if run:
                _run_video(path, selected, settings, pipeline, frame_slot, info_slot, progress)
            elif path:
                frame_slot.video(path)
            else:
                frame_slot.info("Chọn tệp video ở cột bên phải để bắt đầu.")
        else:
            is_streaming = st.session_state.get("_live_stream_active", False)
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if not is_streaming:
                    if st.button("▶️ Bật Camera Trực Tiếp", type="primary"):
                        st.session_state["_live_stream_active"] = True
                        st.rerun()
                else:
                    if st.button("⏹️ Dừng Camera"):
                        st.session_state["_live_stream_active"] = False
                        st.rerun()

            if is_streaming:
                url_clean = stream_url.strip()
                if url_clean.isdigit():
                    cam_source = int(url_clean)
                elif url_clean == "":
                    cam_source = 0
                else:
                    cam_source = url_clean

                progress.progress(1.0, text="Đang phát trực tiếp...")
                has_frames = False
                for frame, info in generate_live_stream(source=cam_source):
                    has_frames = True
                    if not st.session_state.get("_live_stream_active", False):
                        break
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(rgb_frame, use_container_width=True)
                    info_slot.markdown(f"**FPS**: {info['fps']:.1f} | **Vi phạm**: {info['violating_count']}")

                if not has_frames and st.session_state.get("_live_stream_active", False):
                    st.session_state["_live_stream_active"] = False
                    st.error(f"❌ Không thể mở nguồn camera: '{stream_url}'. Vui lòng nhập '0' cho Webcam máy tính.")
            else:
                frame_slot.info("Nhấn 'Bật Camera Trực Tiếp' để xem luồng video thời gian thực.")