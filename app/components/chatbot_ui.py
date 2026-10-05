import re

import pandas as pd
import streamlit as st

from app.components.violation_table import VIOLATION_NAMES

GREETING = "Xin chào. Bạn có thể hỏi theo biển số, theo tên lỗi hoặc hỏi tổng số vi phạm."
SUGGESTIONS = ["Tổng số vi phạm là bao nhiêu?", "Đi ngược chiều được phát hiện thế nào?"]
HOW_DETECTED = {  # mô tả cách hệ thống phát hiện, không phải nguồn pháp luật
    "NO_HELMET": "Hệ thống phát hiện xe máy, rồi kiểm tra người lái có đội mũ bảo hiểm qua nhiều khung hình liên tiếp.",
    "WRONG_WAY": "Hệ thống theo dõi quỹ đạo của từng phương tiện và so hướng di chuyển với hướng được phép.",
    "RED_LIGHT": "Hệ thống kiểm tra phương tiện có vượt vạch dừng trong lúc đèn đỏ hay không.",
    "WRONG_LANE": "Hệ thống so vị trí phương tiện với vùng làn đường đã cấu hình cho camera.",
}


def _norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _fmt(t):
    return "" if pd.isna(t) else t.strftime("%d/%m/%Y %H:%M:%S")


def local_answer(q, df):
    """Tra cứu trực tiếp từ dữ liệu, không tự suy đoán mức phạt."""
    nq, low = _norm(q), q.lower()

    for plate in df["Plate"].dropna().unique():
        n = _norm(plate)
        if n and n != "UNKNOWN" and n in nq:
            sub = df[df["Plate"] == plate].sort_values("Time", ascending=False)
            lines = [f"- {_fmt(r.Time)}: {r.Violation}" for r in sub.head(5).itertuples()]
            return f"Biển số **{plate}** có {len(sub)} vi phạm:\n\n" + "\n".join(lines)

    for code, name in VIOLATION_NAMES.items():
        if name.lower() in low or code.lower() in low or code.lower().replace("_", " ") in low:
            return (f"**{name}**\n\n{HOW_DETECTED[code]}\n\nMức phạt và căn cứ pháp lý chưa có trong dữ liệu, "
                    "cần đối chiếu văn bản chính thức.")

    if any(k in low for k in ("tổng", "bao nhiêu", "thống kê")):
        counts = df["Violation"].value_counts()
        if counts.empty:
            return "Hiện chưa có vi phạm nào trong cơ sở dữ liệu."
        return f"Hiện có **{len(df)}** vi phạm:\n\n" + "\n".join(f"- {k}: {v}" for k, v in counts.items())

    return "Không tìm thấy thông tin phù hợp. Hãy thử hỏi theo biển số hoặc tên lỗi vi phạm."


def _ask(q, df, answer_fn):
    """Thêm câu hỏi và câu trả lời vào lịch sử chat. Chạy trong callback nên tin nhắn mới hiện ngay ở lần vẽ kế tiếp."""
    q = (q or "").strip()
    if not q:
        return
    st.session_state.messages.append({"role": "user", "content": q})
    try:
        reply = answer_fn(q) if answer_fn else local_answer(q, df)
    except Exception as e:
        reply = f"Không tra cứu được lúc này: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})


def _submit(df, answer_fn):
    _ask(st.session_state.get("chat_q", ""), df, answer_fn)


def _clear():
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]


def render_chatbot(df, answer_fn=None):
    """Bong bóng chat góc dưới phải (vị trí do CSS trong app.py).
    Bố cục từ trên xuống: tiêu đề + nút xóa, khung hội thoại (gợi ý nằm trong khung), ô nhập ở dưới cùng.
    answer_fn(question) -> str, để None thì tra cứu từ df."""
    if "messages" not in st.session_state:
        _clear()

    with st.popover("AI"):
        head, clear = st.columns([5, 1], vertical_alignment="center")
        head.markdown("**Trợ lý tra cứu vi phạm**")
        clear.button(":material/delete:", key="chat_clear", on_click=_clear, help="Xóa cuộc trò chuyện", width="stretch")

        box = st.container(height=340, border=True, key="chat_box")
        with box:
            for m in st.session_state.messages:
                st.chat_message(m["role"]).markdown(m["content"])
            if len(st.session_state.messages) == 1:  # gợi ý nằm chung trong khung trả lời
                st.caption("Gợi ý cho bạn")
                for i, s in enumerate(SUGGESTIONS):
                    st.button(s, key=f"sg{i}", on_click=_ask, args=(s, df, answer_fn), width="stretch")

        with st.form("chat_form", clear_on_submit=True, border=False):
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.text_input("Câu hỏi", key="chat_q", placeholder="Nhập câu hỏi...", label_visibility="collapsed")
            c2.form_submit_button(":material/send:", on_click=_submit, args=(df, answer_fn), width="stretch")