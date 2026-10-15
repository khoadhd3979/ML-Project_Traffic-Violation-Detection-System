import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import tempfile
from datetime import date, datetime, timedelta

# Thêm thư mục root của dự án vào sys.path để tránh ModuleNotFoundError (Monorepo architecture)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.features.live_stream import generate_live_stream

# from app.components.chatbot_ui import render_chatbot  # Tạm tắt Chatbot UI theo kế hoạch
from app.components.violation_table import violation_label

try:
    import src.data.database_manager as db
    DB_IMPORT_ERROR = None
except Exception as e:
    db, DB_IMPORT_ERROR = None, f"{type(e).__name__}: {e}"

try:
    from src.utils.config import PLATE_REGEX, VIDEO_SOURCE, YOLO_INPUT_WIDTH, YOLO_INTERVAL
except Exception:
    PLATE_REGEX = r"^\d{2}[A-Z]\d?[-–]?\d{3,5}\.?\d{0,2}$"
    VIDEO_SOURCE, YOLO_INPUT_WIDTH, YOLO_INTERVAL = 0, 640, 2

st.set_page_config(page_title="Hệ thống phát hiện vi phạm giao thông", layout="wide")

# ------------------------------------------------------------------ CẤU HÌNH & CSS
COLUMNS = ["ID", "Track ID", "Time", "Plate", "Code", "Violation", "Evidence"]
MONITORED = ["WRONG_LANE", "RED_LIGHT", "NO_HELMET"]
IMPLEMENTED = {"WRONG_LANE"}

EVIDENCE_DIR = os.path.join("data", "evidence")
ZONE_REL = np.array([[0.23, 0.28], [0.70, 0.28], [0.78, 0.97], [0.16, 0.97]])

CONFIRM = 2
ZONE_HITS = 4
OCR_EVERY = 3
WAIT_PLATE = 60
LOST_AFTER = 40

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, button, input, textarea { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
[data-testid="stToolbar"], #MainMenu, footer { display: none; }
header[data-testid="stHeader"] { background: transparent; height: 0.5rem; }
.block-container { padding: 1rem 1.5rem 3rem; max-width: 100%; }
h1 { text-align: center; font-size: 1.8rem; font-weight: 700; margin: 0rem 0 1rem; color: #1e293b; }
h2, h3 { font-size: 1.1rem; font-weight: 600; margin: .1rem 0 .4rem; }

/* Custom khung lịch sử bên trái */
.history-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
}

/* Các ô trạng thái và thông tin cột bên phải */
.info-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}
.info-title {
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 4px;
    font-weight: 500;
}
.info-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
}

/* Thẻ trạng thái lỗi */
.status-box-red {
    background-color: #fef2f2;
    border: 2px solid #ef4444;
    border-radius: 8px;
    padding: 12px;
    color: #991b1b;
    font-weight: bold;
    text-align: center;
    margin-bottom: 12px;
}
.status-box-green {
    background-color: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 8px;
    padding: 12px;
    color: #166534;
    font-weight: bold;
    text-align: center;
    margin-bottom: 12px;
}

/* Khung đen màn hình Live */
.black-screen {
    width: 100%;
    height: 400px;
    background-color: #000000;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #888;
    font-size: 1.1rem;
    border: 1px solid #333;
}

/* Tùy chỉnh nút Quét và Thống kê nhỏ gọn */
.st-key-btn_scan button {
    width: auto !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
    margin-top: 28px;
}
.st-key-btn_stats button {
    width: auto !important;
    padding-left: 15px !important;
    padding-right: 15px !important;
}

/* Bong bóng Chatbot dạng tin nhắn Zalo/Messenger */
.st-key-chatbot [data-testid="stPopover"] { position: fixed; right: 1.6rem; bottom: 1.4rem; z-index: 1000; width: auto; }
.st-key-chatbot [data-testid="stPopoverButton"] { width: 56px; height: 56px; border-radius: 50%; background: #0b5c78; border: none; box-shadow: 0 4px 14px rgba(11,92,120,.4); justify-content: center; }
.st-key-chatbot [data-testid="stPopoverButton"] p { color: #fff; font-weight: 600; margin: 0; }
.st-key-chatbot [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] { display: none; }

/* Khung chat cố định không scroll trang ngoài */
[data-testid="stPopoverBody"] {
    width: 380px !important;
    max-width: 92vw !important;
    height: 520px !important;
    max-height: 80vh !important;
    overflow: hidden !important;
    border-radius: 12px !important;
}

/* Khung tin nhắn bo góc kiểu Messenger */
[data-testid="stChatMessage"] {
    border-radius: 18px !important;
    padding: 8px 14px !important;
    margin-bottom: 8px !important;
    max-width: 85% !important;
    width: fit-content !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #0084ff !important;
    color: white !important;
    margin-left: auto !important;
    border-bottom-right-radius: 4px !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: #e4e6eb !important;
    color: black !important;
    margin-right: auto !important;
    border-bottom-left-radius: 4px !important;
}
</style>
"""

# ------------------------------------------------------------------ DỮ LIỆU TỪ DATABASE
def load_data():
    empty_df = pd.DataFrame(columns=COLUMNS)
    if db is None:
        return empty_df, "Lỗi nạp database", DB_IMPORT_ERROR
    
    try:
        db_mgr = db.DatabaseManager()
        query = """
            SELECT 
                v.id AS ID,
                v.frame_number AS "Track ID",
                v.detected_at AS Time,
                veh.license_plate AS Plate,
                r.rule_code AS Code,
                COALESCE(r.rule_name, v.notes, 'Vi phạm giao thông') AS Violation,
                v.evidence_image_path AS Evidence
            FROM violations v
            JOIN vehicles veh ON veh.id = v.vehicle_id
            LEFT JOIN violation_rules r ON r.id = v.rule_id
            ORDER BY v.detected_at DESC
        """
        with db_mgr._get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return empty_df, "Dữ liệu trống", None
            
        df["Time"] = pd.to_datetime(df["Time"])
        return df.reindex(columns=COLUMNS), "Cơ sở dữ liệu", None
    except Exception as e:
        return empty_df, "Lỗi truy vấn CSĐL", str(e)

def _fmt_time(t):
    if pd.isna(t) or t is None:
        return ""
    if isinstance(t, str):
        try:
            t = pd.to_datetime(t)
        except Exception:
            return t
    return t.strftime("%d/%m/%Y %H:%M:%S")

# ------------------------------------------------------------------ PIPELINE PROCESSING
def is_valid_plate(text):
    return bool(text) and len(text) >= 7 and bool(re.match(PLATE_REGEX, text.strip().upper()))

def fmt_plate(text):
    if not text or not text.isalnum() or len(text) < 7:
        return text
    tail = 5 if len(text) >= 8 else 4
    return f"{text[:-tail]}-{text[-tail:]}"
# Logic phát hiện vi phạm và tracking đã được hợp nhất trong src.features.live_stream.generate_live_stream



# ------------------------------------------------------------------ MODAL LỊCH SỬ FULL MÀN HÌNH
@st.dialog("Bảng lịch sử vi phạm chi tiết", width="large")
def show_full_history_dialog(df):
    st.subheader("Lịch sử quét xe vi phạm")
    
    f1, f2, f3 = st.columns([2, 2, 2])
    q = f1.text_input("Tìm biển số xe", placeholder="Ví dụ: 51G...")
    types = f2.multiselect("Lọc theo lỗi", sorted(df["Violation"].dropna().unique()))
    days = f3.date_input("Lọc theo ngày", value=(), format="DD/MM/YYYY")

    view = df.copy()
    if q.strip():
        view = view[view["Plate"].fillna("").str.contains(q.strip(), case=False, regex=False)]
    if types:
        view = view[view["Violation"].isin(types)]
    if len(days) >= 1:
        view = view[(view["Time"] >= pd.Timestamp(days[0])) & (view["Time"] < pd.Timestamp(days[-1]) + pd.Timedelta(days=1))]
    view = view.sort_values("Time", ascending=False).reset_index(drop=True)

    st.markdown(f"**Tổng số bản ghi:** {len(view)}")
    
    shown = view[["ID", "Time", "Plate", "Violation", "Track ID", "Evidence"]].copy()
    shown = shown.rename(columns={
        "Time": "Thời gian quét",
        "Plate": "Biển số xe",
        "Violation": "Lỗi vi phạm",
        "Evidence": "Ảnh bằng chứng"
    })
    
    st.dataframe(
        shown,
        hide_index=True,
        use_container_width=True,
        height=320,
        column_config={
            "Thời gian quét": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm:ss"),
            "Ảnh bằng chứng": st.column_config.ImageColumn(label="Ảnh bằng chứng", help="Ảnh crop xe vi phạm")
        }
    )

    with st.expander("🔍 Xem chi tiết ảnh bằng chứng vi phạm"):
        valid_evidences = view[view["Evidence"].notna() & (view["Evidence"] != "")]
        if not valid_evidences.empty:
            options = {
                f"ID {row['ID']} - Biển: {row['Plate']} ({_fmt_time(row['Time'])})": row["Evidence"]
                for _, row in valid_evidences.iterrows()
            }

            selected_label = st.selectbox("Chọn bản ghi vi phạm để xem ảnh:", list(options.keys()))
            selected_path = options[selected_label]
            if os.path.exists(selected_path):
                st.image(selected_path, caption=f"Ảnh bằng chứng: {selected_label}", use_container_width=True)
            else:
                st.warning(f"File ảnh bằng chứng không tồn tại hoặc đã bị di chuyển: '{selected_path}'")
        else:
            st.info("Chưa có ảnh bằng chứng nào được lưu trong hệ thống.")
    

# ------------------------------------------------------------------ MODAL THỐNG KÊ FULL MÀN HÌNH
@st.dialog("Thống kê chi tiết vi phạm giao thông", width="large")
def show_statistics_dialog(df):
    st.subheader("Báo cáo và Thống kê vi phạm")
    
    total_violations = len(df)
    today_violations = len(df[df["Time"].dt.date == date.today()]) if not df.empty else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số vi phạm", total_violations)
    m2.metric("Vi phạm trong ngày", today_violations)
    m3.metric("Số biển số phân biệt", df["Plate"].nunique() if not df.empty else 0)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Tỷ lệ các lỗi vi phạm")
        if not df.empty and "Violation" in df.columns:
            counts = df["Violation"].value_counts().reset_index()
            counts.columns = ["Lỗi vi phạm", "Số lượng"]
            fig_pie = px.pie(counts, names="Lỗi vi phạm", values="Số lượng", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu.")

    with c2:
        st.markdown("##### Số lượt vi phạm theo thời gian")
        if not df.empty and "Time" in df.columns:
            df_time = df.set_index("Time").resample("D").size().reset_index(name="Số lượt")
            fig_line = px.line(df_time, x="Time", y="Số lượt", markers=True,
                               labels={"Time": "Ngày", "Số lượt": "Số vụ vi phạm"})
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu.")

# ------------------------------------------------------------------ GIAO DIỆN CHÍNH
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<h1>HỆ THỐNG PHÁT HIỆN VI PHẠM GIAO THÔNG</h1>", unsafe_allow_html=True)

    for k, v in {"conf": 0.5, "ocr": True, "use_mock": False, "scan_events": []}.items():
        st.session_state.setdefault(k, v)

# Thanh bên Quản trị
    with st.sidebar.expander("⚙️ Cài đặt Quản trị (Dành cho Demo)"):
        confirm_text = st.text_input("Gõ chữ 'XACNHAN' để xác nhận reset hệ thống:")
        is_disabled = confirm_text.strip() != "XACNHAN"
        if st.button("🗑️ Khôi phục hệ thống (Reset)", disabled=is_disabled):
            if db is not None:
                db_mgr = db.DatabaseManager()
                db_mgr.reset_database_and_evidence()
            st.session_state["scan_events"] = []
            st.success("✅ Đã khôi phục hệ thống và xóa toàn bộ dữ liệu thành công!")
            st.rerun()
            
    df, source, error = load_data()

    # Layout 3 cột [Lịch sử, Màn hình, Trạng thái & Thống kê]
    col_left, col_center, col_right = st.columns([1.1, 2.3, 1.2], gap="large")

    # ==========================================
    # CỘT TRÁI: LỊCH SỬ NHỎ + NÚT CHI TIẾT
    # ==========================================
    with col_left:
        c_title, c_btn = st.columns([3, 1.5])
        c_title.subheader("Lịch sử quét")
        if c_btn.button("Chi tiết", help="Xem đầy đủ lịch sử"):
            show_full_history_dialog(df)

        st.caption("Các xe vi phạm gần đây:")
        
        # Kết hợp dữ liệu mới vừa quét từ session_state và dữ liệu thực tế từ DB
        events = st.session_state.get("scan_events", [])
        event_rows = []
        for ev in reversed(events):
            event_rows.append({
                "Plate": ev.get("plate", "UNKNOWN"),
                "Violation": violation_label(ev.get("code", "WRONG_LANE")),
                "Time": ev.get("time", datetime.now())
            })
        
        if not df.empty:
            for _, row in df.sort_values("Time", ascending=False).iterrows():
                event_rows.append({
                    "Plate": row["Plate"],
                    "Violation": row["Violation"],
                    "Time": row["Time"]
                })

        if event_rows:
            for item in event_rows[:8]:
                with st.container():
                    st.markdown(f"""
                    <div class="history-card">
                        <div style="font-weight: bold; font-size: 1rem; color: #0b5c78;">{item['Plate']}</div>
                        <div style="font-size: 0.88rem; color: #dc2626; font-weight: 500;">{item['Violation']}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">{_fmt_time(item['Time'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Chưa có dữ liệu lịch sử.")

        st.divider()
        with st.expander("⚙️ Cài đặt Quản trị (Dành cho Demo)"):
            confirm_text = st.text_input("Gõ chữ 'XACNHAN' để xác nhận reset hệ thống:")
            is_disabled = confirm_text.strip() != "XACNHAN"
            if st.button("🗑️ Khôi phục hệ thống (Reset)", disabled=is_disabled):
                if db is not None:
                    db_mgr = db.DatabaseManager()
                    db_mgr.reset_database_and_evidence()
                st.session_state["scan_events"] = []
                st.success("✅ Đã khôi phục hệ thống và xóa toàn bộ dữ liệu thành công!")
                st.rerun()

    # ==========================================
    # CỘT GIỮA: MÀN HÌNH CAMERA & NGUỒN PHÁT
    # ==========================================
    with col_center:
        st.subheader("Màn hình giám sát trực tiếp")
        frame_slot = st.empty()
        progress_slot = st.empty()

        source_mode = st.radio("Nguồn video", ["Live Camera", "Upload Video"], horizontal=True, label_visibility="collapsed")
        
        uploaded_file = None
        btn_start = False

        if source_mode == "Upload Video":
            c_up, c_btn_scan = st.columns([4, 1])
            with c_up:
                uploaded_file = st.file_uploader("Tải video lên (MP4, AVI)", type=["mp4", "avi", "mov"], label_visibility="collapsed")
            with c_btn_scan:
                with st.container(key="btn_scan"):
                    btn_start = st.button("Quét", type="primary")
        else:
            c_src, c_cam_btn = st.columns([3, 1.5])
            with c_src:
                cam_source_input = st.text_input("Nguồn Camera", value="0", help="0 là Webcam, hoặc nhập đường dẫn video/URL RTSP", label_visibility="collapsed")
            with c_cam_btn:
                is_live_active = st.session_state.get("live_streaming_active", False)
                if not is_live_active:
                    if st.button("▶️ Bật Camera", type="primary"):
                        st.session_state["live_streaming_active"] = True
                        st.rerun()
                else:
                    if st.button("⏹️ Dừng Camera"):
                        st.session_state["live_streaming_active"] = False
                        st.rerun()

        # Quét video khi bấm nút Upload
        if source_mode == "Upload Video" and btn_start and uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(uploaded_file.getbuffer())
                video_path = f.name

            st.session_state["scan_events"] = []

            # Đếm tổng số frame để hiển thị thanh tiến trình
            cap_info = cv2.VideoCapture(video_path)
            total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_info.isOpened() else 0
            cap_info.release()

            processed_frames = 0
            try:
                for frame, info in generate_live_stream(source=video_path):
                    processed_frames += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(rgb_frame, use_container_width=True)

                    if total_frames > 0:
                        progress_slot.progress(min(processed_frames / total_frames, 1.0))

                    if info.get("latest_event"):
                        ev = info["latest_event"]
                        st.session_state.setdefault("scan_events", []).append({
                            "track_id": ev["track_id"],
                            "plate": ev["plate"],
                            "code": "NO_HELMET",
                            "time": datetime.now(),
                            "image": None
                        })
            finally:
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass

            if total_frames > 0:
                progress_slot.progress(1.0)
            st.success("✅ Đã xử lý xong video!")
            st.rerun()


        # Phát trực tiếp Live Camera khi được kích hoạt
        elif source_mode == "Live Camera" and st.session_state.get("live_streaming_active", False):
            cam_src_clean = cam_source_input.strip()
            if cam_src_clean.isdigit():
                cam_source = int(cam_src_clean)
            elif cam_src_clean == "":
                cam_source = 0
            else:
                cam_source = cam_src_clean

            has_frames = False
            for frame, info in generate_live_stream(source=cam_source):
                has_frames = True
                if not st.session_state.get("live_streaming_active", False):
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(rgb_frame, use_container_width=True)

                if info.get("latest_event"):
                    ev = info["latest_event"]
                    st.session_state.setdefault("scan_events", []).append({
                        "track_id": ev["track_id"],
                        "plate": ev["plate"],
                        "code": "NO_HELMET",
                        "time": datetime.now(),
                        "image": None
                    })

            if not has_frames and st.session_state.get("live_streaming_active", False):
                st.session_state["live_streaming_active"] = False
                st.error(f"❌ Không thể kết nối tới nguồn camera: '{cam_source_input}'. Vui lòng nhập '0' cho Webcam máy tính hoặc kiểm tra lại đường dẫn video/RTSP.")
        elif source_mode == "Live Camera":
            frame_slot.markdown('<div class="black-screen">LIVE CAMERA (Nhấn "Bật Camera" để kết nối luồng)</div>', unsafe_allow_html=True)

    # ==========================================
    # CỘT PHẢI: TRẠNG THÁI & NÚT THỐNG KÊ
    # ==========================================
    with col_right:
        # Nút Thống kê nhỏ vừa chữ
        with st.container(key="btn_stats"):
            if st.button("Thống kê"):
                show_statistics_dialog(df)

        st.divider()

        # Hiển thị dữ liệu vi phạm mới nhất từ scan_events hoặc từ cơ sở dữ liệu thực tế
        events = st.session_state.get("scan_events", [])
        if events:
            last_ev = events[-1]
            has_violation = True
            plate_val = last_ev['plate']
            violation_val = violation_label(last_ev.get('code', 'WRONG_LANE'))
            time_val = _fmt_time(last_ev['time'])
        else:
            # Lấy bản ghi vi phạm mới nhất trong cơ sở dữ liệu thực tế
            sample = df.iloc[0] if not df.empty else None
            has_violation = True if sample is not None else False
            plate_val = sample["Plate"] if sample is not None else "---"
            violation_val = sample["Violation"] if sample is not None else "Không có lỗi"
            time_val = _fmt_time(sample["Time"]) if sample is not None else "---"

        # 1. Ô THÔNG BÁO CÓ LỖI HAY KHÔNG
        if has_violation:
            st.markdown('<div class="status-box-red">PHÁT HIỆN VI PHẠM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box-green">BÌNH THƯỜNG</div>', unsafe_allow_html=True)

        # 2. Ô BIỂN SỐ QUÉT ĐƯỢC
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Biển số xe quét được</div>
            <div class="info-value">{plate_val}</div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Ô CHỨA LỖI VI PHẠM
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Lỗi vi phạm</div>
            <div class="info-value" style="color: {'#dc2626' if has_violation else '#1e293b'};">{violation_val}</div>
        </div>
        """, unsafe_allow_html=True)

        # 4. Ô CHỨA NGÀY GIỜ QUÉT
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Thời gian quét</div>
            <div class="info-value">{time_val}</div>
        </div>
        """, unsafe_allow_html=True)

    # Chatbot cố định khung hình (Tạm tắt theo kế hoạch phát triển UI/API)
    # with st.container(key="chatbot"):
    #     render_chatbot(df)

if __name__ == "__main__":
    main()