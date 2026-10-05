import os

import pandas as pd
import streamlit as st

VIOLATION_NAMES = {
    "NO_HELMET": "Không đội mũ bảo hiểm",
    "WRONG_WAY": "Đi ngược chiều",
    "RED_LIGHT": "Vượt đèn đỏ",
    "WRONG_LANE": "Sai làn đường",
}


def violation_label(code):
    """Mã lỗi (NO_HELMET...) thành tên tiếng Việt; giá trị lạ giữ nguyên."""
    return VIOLATION_NAMES.get(str(code).upper(), str(code))


def _detail(r):
    st.subheader(f"Sự kiện #{r['ID']}")
    ev = r["Evidence"]
    if isinstance(ev, str) and os.path.exists(ev):
        st.image(ev, width=300)
    else:
        st.info("Chưa có ảnh bằng chứng.")
    t = "" if pd.isna(r["Time"]) else r["Time"].strftime("%d/%m/%Y %H:%M:%S")
    st.markdown(f"Lỗi: **{r['Violation']}**  \nBiển số: **{r['Plate']}**  \n"
                f"Track ID: {r['Track ID']}  \nThời gian: {t}")


def render_violation_table(df):
    f1, f2, f3 = st.columns([2, 1.4, 1], gap="small")
    q = f1.text_input("Tìm", placeholder="Tìm theo biển số, ví dụ 51G", label_visibility="collapsed")
    rule = f2.selectbox("Loại lỗi", ["Tất cả loại lỗi"] + sorted(df["Violation"].dropna().unique()),
                        label_visibility="collapsed")

    view = df.copy()
    if q:
        view = view[view["Plate"].fillna("").str.contains(q, case=False, regex=False)]
    if rule != "Tất cả loại lỗi":
        view = view[view["Violation"] == rule]
    view = view.sort_values("Time", ascending=False).reset_index(drop=True)

    f3.download_button("Xuất CSV", view.drop(columns=["Evidence", "Code"]).to_csv(index=False).encode("utf-8-sig"),
                       "violations.csv", "text/csv", width="stretch")

    if view.empty:
        st.warning("Chưa có sự kiện nào. Nếu chỉ muốn xem giao diện, bật dữ liệu mẫu trong mục Cài đặt.")
        return

    table, panel = st.columns([2.2, 1], gap="medium")
    with table:
        shown = view[["ID", "Time", "Plate", "Violation", "Track ID"]].rename(columns={
            "Time": "Thời gian", "Plate": "Biển số", "Violation": "Lỗi vi phạm"})
        event = st.dataframe(shown, hide_index=True, height=400, width="stretch",
                             on_select="rerun", selection_mode="single-row", key="violation_table",
                             column_config={"Thời gian": st.column_config.DatetimeColumn(format="DD/MM HH:mm:ss")})
    rows = event.selection.rows
    with panel:
        _detail(view.iloc[rows[0] if rows else 0])