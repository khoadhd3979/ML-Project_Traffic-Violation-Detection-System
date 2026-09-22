import streamlit as st
import datetime
import sys
import os
import cv2
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# PATH SETUP — đảm bảo import được modules/
MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

import modules.database_manager as db
from modules.config import HISTORY_DISPLAY_LIMIT

from modules.detection import detect_license_plate
from modules.processing import process_plate
from modules.ocr_engine import PlateOCR


# CẤU HÌNH TRANG
st.set_page_config(
    page_title="Hệ thống Nhận diện Biển số xe",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS TỐI ƯU GIAO DIỆN
st.markdown("""
<style>
.block-container { padding-top: 0.6rem !important; padding-bottom: 0.6rem !important; max-width: 100% !important; }
div[data-testid="stVerticalBlock"] > div { padding-bottom: 0rem !important; }
h1 { margin-top: 2rem !important; margin-bottom: 0.5rem !important; padding-top: 0 !important; text-align: center; }
h2, h3 { margin-top: 0 !important; margin-bottom: 0.2rem !important; padding-top: 0 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { padding: 4px 10px; font-weight: bold; }
hr { margin: 0.4rem 0 !important; }
div[data-testid="stFormSubmitButton"] > button, .stButton > button { margin-top: 0rem !important; }
/* Giới hạn chiều ngang dialog ~3/5 màn hình */
div[data-testid="stModal"] { width: 60vw !important; max-width: 60vw !important; }
[class*="st-emotion-cache"] [role="dialog"] { max-width: 60vw !important; width: 60vw !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db_initialized():
    """Khởi tạo DB một lần duy nhất suốt vòng đời app."""
    conn = db.init_db()
    conn.close()
    return True

get_db_initialized()

@st.cache_resource
def get_ocr_engine():
    """Khởi tạo mô hình OCR một lần duy nhất."""
    return PlateOCR()

ocr_engine = get_ocr_engine()

# SESSION STATE TRẠNG THÁI XE & THÔNG BÁO (ĐỒNG BỘ CỘT PHẢI)
if "last_action"       not in st.session_state: st.session_state.last_action        = None
if "last_action_plate" not in st.session_state: st.session_state.last_action_plate  = ""
if "last_action_fee"   not in st.session_state: st.session_state.last_action_fee    = ""
if "last_action_slot"  not in st.session_state: st.session_state.last_action_slot   = ""
if "last_scanned"      not in st.session_state: st.session_state.last_scanned       = None
if "last_out_info"      not in st.session_state: st.session_state.last_out_info       = {}
if "action_timestamp"   not in st.session_state: st.session_state.action_timestamp    = None

# KEY ĐỘNG ĐỂ KHỞI ĐỘNG LẠI COMPONENT
if "input_version"      not in st.session_state: st.session_state.input_version       = 0
if "camera_version"     not in st.session_state: st.session_state.camera_version      = 0
if "need_right_refresh" not in st.session_state: st.session_state.need_right_refresh  = False

# CƠ CHẾ ƯU TIÊN NGUỒN DỮ LIỆU (Data Source Routing)
# "auto"    = Camera ngoài chạy ngầm được phép cập nhật dữ liệu biển số
# "browser" = Nhân viên đang quét bằng webcam trình duyệt → khóa luồng camera ngoài
if "active_source"      not in st.session_state: st.session_state.active_source       = "auto"

# SESSION STATE BỘ LỌC LỊCH SỬ NÂNG CAO
if "filter_all_days"     not in st.session_state: st.session_state.filter_all_days     = False
if "filter_start_time"   not in st.session_state: st.session_state.filter_start_time   = datetime.time(0, 0)
if "filter_end_time"     not in st.session_state: st.session_state.filter_end_time     = datetime.time(23, 59)
if "filter_date"         not in st.session_state: st.session_state.filter_date         = datetime.date.today()
if "filter_applied"      not in st.session_state: st.session_state.filter_applied      = False


# TỰ ĐỘNG KIỂM TRA CAMERA NGOÀI CHẠY NGẦM (REFRESH 3 GIÂY — giảm tải CPU cho mô hình AI)
@st.fragment(run_every=1)
def auto_check_camera_ngam():
    # Nếu nhân viên đang quét bằng webcam trình duyệt → không được ghi đè dữ liệu
    if st.session_state.active_source == "browser":
        return
    
    camera_detected = db.get_detected_plate()
    if camera_detected:
        detected_clean = camera_detected.strip().upper()
        # Nếu phát hiện biển số mới từ camera ngoài và bối cảnh hiện tại chưa bị khóa dữ liệu cũ
        if not st.session_state.last_scanned:
            st.session_state.last_scanned = detected_clean
            
            # ĐỒNG BỘ HIỆU ỨNG REFRESH: Đưa trạng thái cột phải về "Đang chờ..." giống nhập thủ công
            st.session_state.last_action       = None
            st.session_state.last_action_plate = ""
            st.session_state.last_action_fee   = ""
            st.session_state.last_action_slot  = ""
            st.session_state.last_out_info     = {}
            
            st.rerun()

# Chạy ngầm quét dữ liệu camera ngoài mỗi 3s
auto_check_camera_ngam()


# ĐỌC DỮ LIỆU THỰC TỪ DATABASE ĐỂ HIỂN THỊ LÊN UI LÀM MỚI
status_data  = db.get_status()
recent_logs  = db.get_recent_logs(limit=HISTORY_DISPLAY_LIMIT)
revenue_data = db.get_revenue_today()

capacity    = status_data["max_capacity"]
occupancy   = status_data["occupancy"]
vehicles_in = status_data["vehicles_inside"]


def fmt_time(raw: str) -> str:
    try:
        return raw[11:19]
    except Exception:
        return raw or "--:--:--"

def fmt_date(raw: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(raw)
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return raw or "--/--/----"


st.markdown("<h1>Hệ thống bãi đỗ xe</h1>", unsafe_allow_html=True)

# BỐ CỤC CHÍNH (3 cột)
col_left, col_center, col_right = st.columns([1.2, 1.6, 1.2], gap="large")


# CỘT TRÁI – XE TRONG BÃI & LỊCH SỬ HỆ THỐNG
with col_left:
    tab_current, tab_hist = st.tabs([f"Xe trong bãi ({occupancy})", "Lịch sử hệ thống"])

    # --- TAB 1: XE ĐANG TRONG BÃI ---
    with tab_current:
        search_in = st.text_input("Tìm xe:", placeholder="Nhập biển số...", key="search_in", label_visibility="collapsed")
        filtered = [
            v for v in vehicles_in
            if search_in.upper() in v["plate_number"].upper()
        ] if search_in else vehicles_in

        with st.container(height=490, border=True):
            if not filtered:
                st.caption("Không tìm thấy xe." if search_in else "Bãi đang trống.")
            else:
                for v in filtered:
                    st.info(
                        f"**{v['plate_number']}**\n\n"
                        f"Vào: {fmt_date(v['time_in'])} {fmt_time(v['time_in'])} | Vé: **#{v['ticket_id']}**"
                    )

    # --- TAB 2: LỊCH SỬ ---
    with tab_hist:
        search_hist = st.text_input("Tìm lịch sử:", placeholder="Nhập biển số...", key="search_hist", label_visibility="collapsed")
        
        with st.popover("Tùy chọn lọc", use_container_width=True):
            st.session_state.filter_all_days = st.checkbox("Xem toàn bộ các ngày", value=st.session_state.filter_all_days)
            if not st.session_state.filter_all_days:
                st.session_state.filter_date = st.date_input("Chọn ngày:", st.session_state.filter_date)
            
            t1, t2 = st.columns(2)
            with t1:
                st.session_state.filter_start_time = st.time_input("Từ:", st.session_state.filter_start_time)
            with t2:
                st.session_state.filter_end_time = st.time_input("Đến:", st.session_state.filter_end_time)
            st.session_state.filter_applied = True

        hist_filtered = []
        today_date = datetime.date.today()

        for h in recent_logs:
            if search_hist and search_hist.upper() not in h["plate_number"].upper():
                continue
            try:
                log_time_str = h.get("time_out") if (h.get("status") == "OUT" and h.get("time_out")) else h.get("time_in")
                log_datetime = datetime.datetime.fromisoformat(log_time_str)
                log_date = log_datetime.date()
                log_time = log_datetime.time()
                
                if st.session_state.filter_applied:
                    if not st.session_state.filter_all_days and log_date != st.session_state.filter_date:
                        continue
                else:
                    if log_date != today_date:
                        continue
                if not (st.session_state.filter_start_time <= log_time <= st.session_state.filter_end_time):
                    continue
                hist_filtered.append(h)
            except Exception:
                hist_filtered.append(h)

        with st.container(height=410, border=True):
            if not hist_filtered:
                st.caption("Không có dữ liệu lịch sử phù hợp.")
            else:
                for h in hist_filtered:
                    time_out_disp = fmt_time(h["time_out"]) if h.get("time_out") else "--:--:--"
                    date_out_disp = fmt_date(h["time_out"]) if h.get("time_out") else "--/--/----"
                    fee_disp      = f"{h['fee']:,.0f}đ" if h.get("status") == "OUT" else "Đang trong bãi"

                    if h.get("status") == "OUT":
                        st.success(
                            f"**{h['plate_number']}** (Vé #{h['ticket_id']})\n\n"
                            f"Vào: {fmt_date(h['time_in'])} {fmt_time(h['time_in'])}\n\n"
                            f"Ra:  {date_out_disp} {time_out_disp}\n\n"
                            f"Phí: **{fee_disp}**"
                        )
                    else:
                        st.info(
                            f"**{h['plate_number']}** (Vé #{h['ticket_id']})\n\n"
                            f"Vào: {fmt_date(h['time_in'])} {fmt_time(h['time_in'])}\n\n"
                            f"Phí: **{fee_disp}**"
                        )

        # NÚT XUẤT LỊSH SỬ (dựa trên dữ liệu đã lọc)
        if hist_filtered:
            import csv, io
            def build_csv(records):
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Biển số", "Vé #", "Trạng thái", "Ngày vào", "Giờ vào", "Ngày ra", "Giờ ra", "Phí (đ)"])
                for h in records:
                    time_out_disp = fmt_time(h["time_out"]) if h.get("time_out") else ""
                    date_out_disp = fmt_date(h["time_out"]) if h.get("time_out") else ""
                    fee_val       = h["fee"] if h.get("status") == "OUT" else 0
                    writer.writerow([
                        h["plate_number"],
                        h["ticket_id"],
                        h.get("status", ""),
                        fmt_date(h["time_in"]),
                        fmt_time(h["time_in"]),
                        date_out_disp,
                        time_out_disp,
                        fee_val,
                    ])
                return output.getvalue().encode("utf-8-sig")

            export_filename = f"lich_su_{datetime.date.today().strftime('%Y%m%d')}.csv"
            st.download_button(
                label=f"⬇Xuất lịch sử ({len(hist_filtered)} bản ghi)",
                data=build_csv(hist_filtered),
                file_name=export_filename,
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇Xuất lịch sử", disabled=True, use_container_width=True, help="Không có dữ liệu để xuất")

# CỘT GIỮA – CAMERA & XỬ LÝ TRANSACTIONS (XÓA CHỮ OCR, RESET ẢNH)
with col_center:
    st.markdown("<p style='font-weight:bold; margin:0;'>Màn hình nhận diện</p>", unsafe_allow_html=True)
    
    img_buffer = st.camera_input(
        "Quét biển số xe", 
        label_visibility="collapsed",
        key=f"cam_input_v_{st.session_state.camera_version}"
    )

    # Khi chụp ảnh qua camera trình duyệt → OCR một lần
    if img_buffer:
        # Tạo ID duy nhất cho bức ảnh dựa trên tên và kích thước để tránh chạy lại AI vô ích khi Rerun trang
        current_img_id = f"{img_buffer.name}_{img_buffer.size}"
        # Đọc ảnh từ buffer của Streamlit
        image = Image.open(img_buffer)
        # Chuyển sang mảng numpy format BGR của OpenCV
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Kiểm xem ảnh này đã được quét AI chưa
        if st.session_state.get("processed_img_id") != current_img_id:
            bbox = detect_license_plate(frame)
            cleaned_text = None
            if bbox is not None:
                processed_crop = process_plate(frame, bbox)
                if processed_crop is not None:
                    text = ocr_engine.read_plate(processed_crop)
                    if text:
                        cleaned_text = text.strip().upper()
            
            # Lưu kết quả vào session state để dùng lại
            st.session_state.processed_img_id = current_img_id
            st.session_state.last_scanned_bbox = bbox
            st.session_state.last_scanned_text = cleaned_text
            
            # Cập nhật kết quả quét mới nhất vào session state
            st.session_state.last_scanned = cleaned_text
            
            # KHÓA LUỒNG CAMERA NGOÀI: Nhân viên đang quét bằng webcam trình duyệt
            st.session_state.active_source = "browser"
            
            # KHI BẤM CHỤP ẢNH MỚI: Cũng đưa cột bên phải về trạng thái "Đang chờ..."
            st.session_state.last_action       = None
            st.session_state.last_action_plate = ""
            st.session_state.last_action_fee   = ""
            st.session_state.last_action_slot  = ""
            st.session_state.last_out_info     = {}
            st.session_state.need_right_refresh = True  # Báo hiệu cột phải cần refresh riêng
        else:
            # Tái sử dụng kết quả đã nhận diện trước đó
            bbox = st.session_state.get("last_scanned_bbox")
            cleaned_text = st.session_state.get("last_scanned_text")

        # Vẽ UI dựa trên kết quả nhận diện đã được caching
        if bbox is not None:
            if not cleaned_text:
                st.warning("Tìm thấy biển nhưng không đọc được chữ!")

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if cleaned_text:
                cv2.rectangle(frame, (x1, max(0, y1 - 35)), (x2, y1), (0, 0, 0), -1)
                cv2.putText(frame, cleaned_text, (x1 + 5, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            st.warning("Không tìm thấy biển số trong ảnh!")

        # st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)  # Ẩn ảnh YOLO bbox
    else:
        # Nếu trước đó ĐÃ có ảnh (processed_img_id có giá trị) nhưng bây giờ img_buffer lại trống (None)
        # Chứng tỏ người dùng vừa bấm "Clear photo" để xóa ảnh
        if st.session_state.get("processed_img_id") is not None:
            st.session_state.processed_img_id = None
            st.session_state.last_scanned_bbox = None
            st.session_state.last_scanned_text = None
            st.session_state.last_scanned = None
            # MỞ KHÓA LUỒNG CAMERA NGOÀI: Nhân viên đã xóa ảnh, cho phép camera ngoài hoạt động lại
            st.session_state.active_source = "auto"

    # Đưa thông báo OCR trực quan và nút bấm rõ ràng
    if st.session_state.last_scanned:
        st.success(f"OCR thành công: **{st.session_state.last_scanned}** — Hãy kiểm tra lại và bấm XÁC NHẬN.")

    st.markdown("<p style='font-weight:bold; margin-top:0.4rem; margin-bottom:0;'>Nhập thủ công</p>", unsafe_allow_html=True)
    
    # Ô nhập thủ công lấy trực tiếp từ session_state, tự động dọn sạch khi submit thành công
    user_typed_plate = st.text_input(
        "Nhập thủ công:",
        value=st.session_state.last_scanned or "",
        placeholder="VD: 29A-12345",
        label_visibility="collapsed",
        key=f"plate_input_v_{st.session_state.input_version}"
    )

    now = datetime.datetime.now()
    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

    # KHU VỰC ĐIỀU KHIỂN NẰM NGANG (XÁC NHẬN & HỦY) — Đã loại bỏ @st.fragment lỗi tại đây
    col_btn_cancel, col_btn_confirm = st.columns([1, 1])

    with col_btn_cancel:
        cancel_clicked = st.button("HỦY", use_container_width=True, type="secondary")

    with col_btn_confirm:
        confirm_clicked = st.button("XÁC NHẬN", use_container_width=True, type="primary")

    if cancel_clicked:
        st.session_state.camera_version += 1  # rebuild camera_input → xóa ảnh
        st.session_state.processed_img_id = None
        st.session_state.last_scanned_bbox = None
        st.session_state.last_scanned_text = None
        st.session_state.last_scanned = None
        st.session_state.input_version += 1
        st.session_state.active_source = "auto"
        db.clear_detected_plate()
        st.rerun()

    # XỬ LÝ SỰ KIỆN NÚT XÁC NHẬN — Bây giờ đã bắt được tín hiệu hoàn hảo
    if confirm_clicked:
        final_plate = ""
        if user_typed_plate.strip():
            final_plate = user_typed_plate.strip().upper()
        elif st.session_state.last_scanned:
            final_plate = st.session_state.last_scanned.strip().upper()

        if not final_plate:
            st.warning("Vui lòng nhập hoặc quét biển số trước khi xác nhận!")
        else:
            result = db.process_vehicle(final_plate)
            s = result.get("status")
            
            if s == "CHECK-IN":
                st.session_state.last_action       = "in"
                st.session_state.last_action_plate = final_plate
                st.session_state.last_action_slot  = f"Vé #{result.get('ticket_id')}"
                st.session_state.last_action_fee   = ""
                st.session_state.last_out_info     = {}
                st.session_state.action_timestamp  = time.time()  # Ghi nhận thời điểm xác nhận để auto-clear sau 5s
                
                # Làm mới toàn bộ UI: Xóa chữ, Xóa ảnh chụp, Xóa hàng đợi camera ngoài
                st.session_state.last_scanned      = None 
                st.session_state.input_version    += 1     
                st.session_state.camera_version   += 1
                st.session_state.active_source     = "auto"  # MỞ KHÓA: Cho phép camera ngoài hoạt động lại
                db.clear_detected_plate()                  
                st.rerun()
                
            elif s == "CHECK-OUT":
                time_in_raw = result.get("time_in")
                try:
                    dt_in = datetime.datetime.fromisoformat(time_in_raw)
                    date_in_str = dt_in.strftime("%d/%m/%Y")
                    time_in_str = dt_in.strftime("%H:%M:%S")
                except Exception:
                    date_in_str = fmt_date(time_in_raw)
                    time_in_str = fmt_time(time_in_raw)

                record = {
                    "plate":     final_plate,
                    "ticket_id": result.get("ticket_id"),
                    "fee":       f"{result.get('fee', 0):,.0f}đ",
                    "hours":     result.get("hours", 0),
                    "minutes":   result.get("minutes", 0),
                    "time_out":  now.strftime("%H:%M:%S"),
                    "date_out":  now.strftime("%d/%m/%Y"),
                    "time_in":   time_in_str,
                    "date_in":   date_in_str,
                }
                st.session_state.last_action       = "out"
                st.session_state.last_action_plate = final_plate
                st.session_state.last_action_fee   = f"{result.get('fee', 0):,.0f}đ"
                st.session_state.last_action_slot  = ""
                st.session_state.action_timestamp  = time.time()  # Ghi nhận thời điểm xác nhận để auto-clear sau 5s
                st.session_state.last_out_info     = record
                
                # Làm mới toàn bộ UI: Xóa chữ, Xóa ảnh chụp, Xóa hàng đợi camera ngoài
                st.session_state.last_scanned      = None 
                st.session_state.input_version    += 1     
                st.session_state.camera_version   += 1
                st.session_state.active_source     = "auto"  # MỞ KHÓA: Cho phép camera ngoài hoạt động lại
                db.clear_detected_plate()                  
                st.rerun()
            elif s == "FULL":
                st.error(result.get("message", "Bãi xe đã đầy chỗ!"))
            else:
                st.error(result.get("message", "Không tìm thấy xe hoặc xảy ra lỗi hệ thống!"))


# HÀM TẠO BIỂU ĐỒ CỘT PLOTLY (Bắt đầu từ 0, trục Y linh hoạt)
def make_bar_chart(labels, values, color, y_tick_vals, y_tick_texts, title_y="Doanh thu (đ)", scrollable=False):
    """Tạo biểu đồ cột Plotly với trục Y tùy chỉnh, bắt đầu từ 0."""
    bar_width = 0.5
    if scrollable and len(labels) > 10:
        chart_width = max(900, len(labels) * 38)
    else:
        chart_width = None  # auto

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=color,
        marker_line_color="rgba(0,0,0,0.15)",
        marker_line_width=1,
        width=bar_width,
        hovertemplate="%{x}<br><b>%{y:,.0f} đ</b><extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, max(y_tick_vals) * 1.08],
            tickvals=y_tick_vals,
            ticktext=y_tick_texts,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        xaxis=dict(
            tickangle=-45 if len(labels) > 10 else 0,
        ),
        height=320,
        width=chart_width,
        bargap=0.25,
    )
    return fig


# CỬA SỔ POP-UP THỐNG KÊ — NGANG RỘNG, 3 LỰA CHỌN
@st.dialog("THỐNG KÊ DOANH THU", width="small")
def show_revenue_stats_modal():
    # Ép dialog về ~60% chiều ngang màn hình bằng JS
    st.markdown("""
    <script>
    (function() {
        function resizeDialog() {
            var dialogs = window.parent.document.querySelectorAll('[role="dialog"]');
            dialogs.forEach(function(d) {
                d.style.maxWidth = '60vw';
                d.style.width = '60vw';
            });
        }
        resizeDialog();
        setTimeout(resizeDialog, 100);
        setTimeout(resizeDialog, 300);
    })();
    </script>
    <style>
    [role="dialog"] { max-width: 60vw !important; width: 60vw !important; }
    </style>
    """, unsafe_allow_html=True)
    all_logs = db.get_recent_logs(limit=3000)
    today    = datetime.date.today()

    tab_day, tab_month = st.tabs(["Trong ngày", "Trong tháng"])

    # ─── TAB 1: NGÀY — 24 cột theo giờ, đơn vị 0 / 25k / 50k / 100k ───
    with tab_day:
        target_date = st.date_input("Chọn ngày:", today, key="stat_day_pick")

        hourly_data   = {i: 0.0 for i in range(24)}
        total_day_rev = 0.0

        for h in all_logs:
            if h.get("status") == "OUT" and h.get("time_out"):
                try:
                    log_dt = datetime.datetime.fromisoformat(h["time_out"])
                    if log_dt.date() == target_date:
                        hourly_data[log_dt.hour] += float(h.get("fee", 0))
                        total_day_rev             += float(h.get("fee", 0))
                except:
                    pass

        st.metric(f"Tổng doanh thu ngày {target_date.strftime('%d/%m/%Y')}", f"{total_day_rev:,.0f} đ")

        labels     = [f"{i:02d}:00" for i in range(24)]
        values     = [hourly_data[i] for i in range(24)]
        max_val    = max(values) if max(values) > 0 else 100000
        # Trục Y: bước nhảy 25k hoặc tự động scale nếu doanh thu lớn hơn
        step       = 500000
        top        = max(100000, ((int(max_val) // step) + 2) * step)
        y_ticks    = list(range(0, top + 1, step))
        y_labels   = [f"{v//1000}k" if v > 0 else "0" for v in y_ticks]

        fig = make_bar_chart(labels, values, "#2ecc71", y_ticks, y_labels)
        st.plotly_chart(fig, use_container_width=True)

    # ─── TAB 2: TUẦN — 7 cột theo ngày, đơn vị 0 / 100k / 200k / 500k ───
    # with tab_week:
    #     # Chọn tuần: mặc định tuần hiện tại (Thứ 2 → Chủ nhật)
    #     week_start_default = today - datetime.timedelta(days=today.weekday())
    #     week_start = st.date_input("Chọn ngày bắt đầu tuần (Thứ 2):", week_start_default, key="stat_week_pick")
    #     # Căn về thứ Hai gần nhất
    #     week_start = week_start - datetime.timedelta(days=week_start.weekday())
    #     week_end   = week_start + datetime.timedelta(days=6)

    #     week_days     = [week_start + datetime.timedelta(days=i) for i in range(7)]
    #     daily_data_w  = {d: 0.0 for d in week_days}
    #     total_week_rev = 0.0
    #     day_vi        = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

    #     for h in all_logs:
    #         if h.get("status") == "OUT" and h.get("time_out"):
    #             try:
    #                 log_dt = datetime.datetime.fromisoformat(h["time_out"])
    #                 d      = log_dt.date()
    #                 if d in daily_data_w:
    #                     daily_data_w[d]  += float(h.get("fee", 0))
    #                     total_week_rev   += float(h.get("fee", 0))
    #             except:
    #                 pass

    #     st.metric(
    #         f"Tổng doanh thu tuần ({week_start.strftime('%d/%m')} – {week_end.strftime('%d/%m/%Y')})",
    #         f"{total_week_rev:,.0f} đ"
    #     )

    #     labels  = [f"{day_vi[i]}\n{week_days[i].strftime('%d/%m')}" for i in range(7)]
    #     values  = [daily_data_w[d] for d in week_days]
    #     max_val = max(values) if max(values) > 0 else 200000
    #     step    = 100000
    #     top     = max(200000, ((int(max_val) // step) + 2) * step)
    #     y_ticks = list(range(0, top + 1, step))
    #     y_labels= [f"{v//1000}k" if v > 0 else "0" for v in y_ticks]

    #     fig = make_bar_chart(labels, values, "#f39c12", y_ticks, y_labels)
    #     st.plotly_chart(fig, use_container_width=True)

    # ─── TAB 3: THÁNG — 28-31 cột, scroll ngang, đơn vị 0 / 100k / 200k / 500k ───
    with tab_month:
        c_y, c_m = st.columns(2)
        with c_y:
            sel_year  = st.number_input("Năm:", min_value=2020, max_value=2030, value=today.year, step=1, key="stat_yr")
        with c_m:
            sel_month = st.number_input("Tháng:", min_value=1, max_value=12, value=today.month, step=1, key="stat_mo")

        import calendar
        days_in_month = calendar.monthrange(int(sel_year), int(sel_month))[1]
        daily_data_m  = {i: 0.0 for i in range(1, days_in_month + 1)}
        total_month_rev = 0.0

        for h in all_logs:
            if h.get("status") == "OUT" and h.get("time_out"):
                try:
                    log_dt = datetime.datetime.fromisoformat(h["time_out"])
                    if log_dt.year == int(sel_year) and log_dt.month == int(sel_month):
                        daily_data_m[log_dt.day] += float(h.get("fee", 0))
                        total_month_rev           += float(h.get("fee", 0))
                except:
                   pass

        st.metric(f"Tổng doanh thu Tháng {sel_month}/{sel_year}", f"{total_month_rev:,.0f} đ")

        labels  = [f"{i}/{sel_month}" for i in range(1, days_in_month + 1)]
        values  = [daily_data_m[i] for i in range(1, days_in_month + 1)]
        max_val = max(values) if max(values) > 0 else 200000
        step    = 1000000
        top     = max(200000, ((int(max_val) // step) + 2) * step)
        y_ticks = list(range(0, top + 1, step))
        y_labels= [f"{v//1000}k" if v > 0 else "0" for v in y_ticks]

        # Scroll ngang: dùng HTML container bao quanh plotly
        fig = make_bar_chart(labels, values, "#3498db", y_ticks, y_labels, scrollable=True)
        st.markdown(
            "<div style='overflow-x:auto; width:100%;'>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)


# CỘT PHẢI – TRẠNG THÁI CHỖ TRỐNG & ĐỐI CHIẾU THÔNG TIN CHI TIẾT
@st.fragment(run_every=3)
def render_col_right():
    # NẾU CÓ BIỂN SỐ MỚI ĐANG CHỜ XỬ LÝ (last_scanned tồn tại):
    # Hệ thống sẽ giữ nguyên màn hình, KHÔNG chạy tự động clear thông tin cũ nữa.
    if st.session_state.last_scanned:
        st.session_state.action_timestamp = None
    
    # TỰ ĐỘNG RESET SAU 5 GIÂY: Nếu không có biển mới và đã quá 5s kể từ lúc bấm XÁC NHẬN
    elif st.session_state.action_timestamp and (time.time() - st.session_state.action_timestamp >= 3):
        st.session_state.last_action       = None
        st.session_state.last_action_plate = ""
        st.session_state.last_action_fee   = ""
        st.session_state.last_action_slot  = ""
        st.session_state.last_out_info     = {}
        st.session_state.action_timestamp  = None
        try:
            st.rerun(scope="fragment")
        except Exception:
            st.rerun()

    # Đọc lại data mới nhất từ DB
    _status_data  = db.get_status()
    _revenue_data = db.get_revenue_today()
    _capacity     = _status_data["max_capacity"]
    _occupancy    = _status_data["occupancy"]
    _vehicles_in  = _status_data["vehicles_inside"]

    free_slots = _capacity - _occupancy
    st.markdown(f"Chỗ còn trống: **{free_slots} / {_capacity}**")
    st.progress(_occupancy / _capacity if _capacity > 0 else 0)

    rev       = _revenue_data.get("revenue", 0)
    checkouts = _revenue_data.get("checkouts", 0)
    
    # Chia làm 2 cột nhỏ: cột trái chứa chữ doanh thu, cột phải chứa nút hình vuông biểu tượng đồ thị
    col_rev_text, col_rev_btn = st.columns([3.2, 0.8], vertical_alignment="center")
    
    with col_rev_text:
        st.caption(f"Doanh thu hôm nay: **{rev:,.0f}đ** ({checkouts} lượt ra)")
        
    with col_rev_btn:
        if st.button("+", use_container_width=True, help="Thống kê doanh thu"):
            show_revenue_stats_modal()

    st.divider()
    st.subheader("Thông tin chi tiết")

    action = st.session_state.last_action
    if action == "in":    st.success("XE VÀO THÀNH CÔNG")
    elif action == "out": st.error("XE RA THÀNH CÔNG")
    else:                 st.info("Đang chờ quét xe...")

    st.write("**Biển số xe vừa thao tác:**")
    st.code(st.session_state.last_action_plate or "-- --- --", language="text")


    # KHU VỰC PHÍ GỬI (ĐỒNG BỘ NẰM DƯỚI BIỂN SỐ VÀ LÀM NỔI BẬT)
    p_fee = "0đ"
    p_date_in = p_time_in = p_date_out = p_time_out = "--"

    if action == "in" and st.session_state.last_action_plate:
        found = next(
            (v for v in _vehicles_in if v["plate_number"].upper() == st.session_state.last_action_plate.upper()),
            None,
        )
        if found:
            p_date_in = fmt_date(found["time_in"])
            p_time_in = fmt_time(found["time_in"])
            p_fee     = "Xe đang trong bãi"
        st.info(f"**Phí gửi:** {p_fee}")

    elif action == "out":
        info = st.session_state.last_out_info
        if info:
            p_date_in  = info.get("date_in", "--")
            p_time_in  = info.get("time_in", "--")
            p_date_out = info.get("date_out", "--")
            p_time_out = info.get("time_out", "--")
            p_fee      = info.get("fee", "0đ")
            p_duration = f"{info.get('hours', 0)}h {info.get('minutes', 0)}m"
        st.error(f"**Phí gửi:** {p_fee}")
    else:
        st.warning("**Phí gửi:** --đ")
        
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    st.write("**Đối chiếu thời gian:**")
    st.write(f"Ngày vào:     {p_date_in}")
    st.write(f"Giờ vào:      {p_time_in}")
    st.write(f"Ngày ra:      {p_date_out}")
    st.write(f"Giờ ra:       {p_time_out}")
    if action == "out" and info:
        st.write(f"Tổng thời gian đỗ: {p_duration}")

    st.divider()

    with st.popover("Cài đặt hệ thống", use_container_width=True):
        st.markdown("**Cấu hình thông số bãi xe**")
        new_capacity = st.number_input(
            "Sức chứa tối đa (chỗ):",
            min_value=1, max_value=500, value=_capacity, step=1,
        )
        if st.button("Lưu sức chứa", use_container_width=True):
            ok = db.set_max_capacity(new_capacity)
            if ok:
                st.toast(f"Đã lưu sức chứa: {new_capacity} chỗ!")
                st.rerun(scope="fragment")
            else:
                st.error(f"Không thể giảm xuống {new_capacity} — hiện có {_occupancy} xe trong bãi!")

with col_right:
    render_col_right()