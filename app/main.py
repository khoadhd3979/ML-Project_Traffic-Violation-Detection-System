import os
import random
import re
import sqlite3
import tempfile
from datetime import date, datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.chatbot_ui import render_chatbot
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
[data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], #MainMenu, footer { display: none; }
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

# ------------------------------------------------------------------ DỮ LIỆU MẪU ĐỂ XEM TRƯỚC UI
def _mock_initial_data():
    now = datetime.now()
    return [
        {"ID": 105, "Track ID": 12, "Plate": "51G-12345", "Time": now - timedelta(minutes=5), "Code": "WRONG_LANE", "Violation": "Đi sai làn đường", "Evidence": None},
        {"ID": 104, "Track ID": 8, "Plate": "29A-99988", "Time": now - timedelta(minutes=18), "Code": "RED_LIGHT", "Violation": "Vượt đèn đỏ", "Evidence": None},
        {"ID": 103, "Track ID": 22, "Plate": "59P1-45678", "Time": now - timedelta(minutes=42), "Code": "NO_HELMET", "Violation": "Không đội mũ bảo hiểm", "Evidence": None},
        {"ID": 102, "Track ID": 15, "Plate": "30E-55566", "Time": now - timedelta(hours=2), "Code": "WRONG_LANE", "Violation": "Đi sai làn đường", "Evidence": None},
        {"ID": 101, "Track ID": 3, "Plate": "59C2-33344", "Time": now - timedelta(hours=3), "Code": "RED_LIGHT", "Violation": "Vượt đèn đỏ", "Evidence": None},
    ]

def load_data():
    df = pd.DataFrame(_mock_initial_data())
    df["Time"] = pd.to_datetime(df["Time"])
    return df.reindex(columns=COLUMNS), "Dữ liệu mẫu", None

def _fmt_time(t):
    return "" if pd.isna(t) else t.strftime("%d/%m/%Y %H:%M:%S")

# ------------------------------------------------------------------ PIPELINE PROCESSING
def is_valid_plate(text):
    return bool(text) and len(text) >= 7 and bool(re.match(PLATE_REGEX, text.strip().upper()))

def fmt_plate(text):
    if not text or not text.isalnum() or len(text) < 7:
        return text
    tail = 5 if len(text) >= 8 else 4
    return f"{text[:-tail]}-{text[-tail:]}"

@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_models():
    from src.models import detection
    from src.models.ocr_engine import PlateOCR
    from src.features.processing import process_plate
    return {"detection": detection, "process": process_plate, "ocr": PlateOCR()}

def _reset_tracker(model):
    try:
        for tr in getattr(getattr(model, "predictor", None), "trackers", None) or []:
            tr.reset()
    except Exception:
        pass

class ViolationPipeline:
    def __init__(self, models, rules, use_ocr):
        self.m, self.use_ocr = models, use_ocr
        self.rules = set(rules) & IMPLEMENTED
        self.idx, self.zone, self.tracks, self.events = 0, None, {}, []
        _reset_tracker(models["detection"].model)

    def _new_track(self):
        return {"plate": "", "cand": "", "count": 0, "checks": 0, "hits": 0, "last_text": "", "last": 0,
                "bbox": (0, 0, 0, 0), "flagged": False, "emitted": False, "flag_idx": 0, "code": None,
                "snap": None, "snap_due": False}

    def _read_plate(self, frame, t):
        crop = self.m["process"](frame, t["bbox"])
        if crop is None: return
        text = self.m["ocr"].read_plate(crop)
        if not text: return
        t["last_text"] = text
        if is_valid_plate(text):
            if text == t["cand"]: t["count"] += 1
            else: t["cand"], t["count"] = text, 1
            if t["count"] >= CONFIRM: t["plate"] = fmt_plate(text)

    def _detect(self, frame, w, h):
        scale = YOLO_INPUT_WIDTH / w
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        for d in self.m["detection"].detect_license_plate(small):
            x1, y1, x2, y2 = (int(v / scale) for v in d["bbox"])
            t = self.tracks.setdefault(d["track_id"], self._new_track())
            t["bbox"], t["last"] = (x1, y1, x2, y2), self.idx
            inside = cv2.pointPolygonTest(self.zone, (float((x1 + x2) // 2), float((y1 + y2) // 2)), False) >= 0
            t["hits"] = t["hits"] + 1 if inside else 0
            if self.use_ocr and not t["plate"] and t["checks"] % OCR_EVERY == 0:
                self._read_plate(frame, t)
            t["checks"] += 1
            if "WRONG_LANE" in self.rules and not t["flagged"] and t["hits"] >= ZONE_HITS:
                t.update(flagged=True, flag_idx=self.idx, snap_due=True, code="WRONG_LANE")

    def _emit(self, tid, t):
        t["emitted"] = True
        plate = t["plate"] or (fmt_plate(t["last_text"]) if is_valid_plate(t["last_text"]) else "UNKNOWN")
        ev = {"track_id": tid, "plate": plate, "code": t["code"], "time": datetime.now(), "image": t["snap"]}
        self.events.append(ev)
        return ev

    def process(self, frame):
        self.idx += 1
        h, w = frame.shape[:2]
        if self.zone is None:
            self.zone = (ZONE_REL * np.array([w, h])).astype(np.int32)
        if self.idx % YOLO_INTERVAL == 0:
            self._detect(frame, w, h)

        vis = frame.copy()
        cv2.polylines(vis, [self.zone], True, (0, 0, 255), 2)
        for tid, t in self.tracks.items():
            if self.idx - t["last"] > YOLO_INTERVAL * 2: continue
            x1, y1, x2, y2 = t["bbox"]
            color = (0, 0, 255) if t["flagged"] else (0, 200, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid} {t['plate'] or t['last_text']}".strip()
            cv2.putText(vis, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        for t in self.tracks.values():
            if t["snap_due"]:
                t["snap"], t["snap_due"] = vis.copy(), False

        new = []
        for tid, t in list(self.tracks.items()):
            lost = self.idx - t["last"] > LOST_AFTER
            if t["flagged"] and not t["emitted"]:
                if t["plate"] or not self.use_ocr or lost or self.idx - t["flag_idx"] >= WAIT_PLATE:
                    new.append(self._emit(tid, t))
            if lost: del self.tracks[tid]
        return vis, {"new": new}

    def finish(self):
        return [self._emit(tid, t) for tid, t in self.tracks.items() if t["flagged"] and not t["emitted"]]

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
    
    shown = view[["ID", "Time", "Plate", "Violation", "Track ID"]].rename(columns={
        "Time": "Thời gian quét", "Plate": "Biển số xe", "Violation": "Lỗi vi phạm"})
    
    st.dataframe(shown, hide_index=True, use_container_width=True, height=400,
                 column_config={"Thời gian quét": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm:ss")})

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
        if not df.empty:
            recent_df = df.sort_values("Time", ascending=False).head(8)
            for _, row in recent_df.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="history-card">
                        <div style="font-weight: bold; font-size: 1rem; color: #0b5c78;">{row['Plate']}</div>
                        <div style="font-size: 0.88rem; color: #dc2626; font-weight: 500;">{row['Violation']}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">{_fmt_time(row['Time'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Chưa có dữ liệu lịch sử.")

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
            frame_slot.markdown('<div class="black-screen">LIVE CAMERA (Chưa kết nối luồng)</div>', unsafe_allow_html=True)

        # Quét video khi bấm nút
        if btn_start and uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(uploaded_file.getbuffer())
                video_path = f.name
            cap = cv2.VideoCapture(video_path)

            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                models = load_models()
                pipe = ViolationPipeline(models, MONITORED, st.session_state["ocr"])

                idx = 0
                st.session_state["scan_events"] = []
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok: break
                    idx += 1
                    vis, extra = pipe.process(frame)
                    
                    if extra["new"]:
                        for ev in extra["new"]:
                            st.session_state["scan_events"].append(ev)

                    if idx % 2 == 0:
                        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                        frame_slot.image(rgb, use_container_width=True)
                        if total > 0:
                            progress_slot.progress(min(idx / total, 1.0))
                cap.release()

    # ==========================================
    # CỘT PHẢI: TRẠNG THÁI & NÚT THỐNG KÊ
    # ==========================================
    with col_right:
        # Nút Thống kê nhỏ vừa chữ
        with st.container(key="btn_stats"):
            if st.button("Thống kê"):
                show_statistics_dialog(df)

        st.divider()

        # Hiển thị dữ liệu vi phạm mẫu hoặc dữ liệu vừa quét
        events = st.session_state.get("scan_events", [])
        if events:
            last_ev = events[-1]
            has_violation = True
            plate_val = last_ev['plate']
            violation_val = violation_label(last_ev['code'])
            time_val = last_ev['time'].strftime("%d/%m/%Y %H:%M:%S")
        else:
            # Lấy bản ghi đầu tiên trong dữ liệu mẫu để hiển thị UI
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

    # Chatbot cố định khung hình
    with st.container(key="chatbot"):
        render_chatbot(df)

if __name__ == "__main__":
    main()