import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ACCENT = "#0b5c78"


def _kpi(label, value, tone=""):
    st.markdown(f"<div class='kpi {tone}'><span class='kpi-label'>{label}</span>"
                f"<span class='kpi-value'>{value:,}</span></div>", unsafe_allow_html=True)


def render_kpis(df):
    """Dải 4 số liệu gọn, tính từ dữ liệu thật trong bảng traffic_violations."""
    known = df["Plate"].notna() & (df["Plate"].astype(str).str.upper() != "UNKNOWN")
    today = pd.Timestamp.now().normalize()
    c = st.columns(4, gap="small")
    with c[0]:
        _kpi("Tổng vi phạm", len(df), "kpi-alert")
    with c[1]:
        _kpi("Vi phạm hôm nay", int((df["Time"] >= today).sum()), "kpi-warn")
    with c[2]:
        _kpi("Phương tiện vi phạm", int(df["Track ID"].nunique()))
    with c[3]:
        _kpi("Biển số đọc được", int(known.sum()))


def _style(fig):
    fig.update_layout(height=220, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans, sans-serif", size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e6ebf0")
    return fig


def render_charts(df):
    if df.empty:
        st.info("Chưa có dữ liệu để vẽ biểu đồ.")
        return
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown("**Vi phạm theo loại**")
        counts = df["Violation"].value_counts()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                               marker_color=ACCENT, text=counts.values, textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(_style(fig), width="stretch")
    with b:
        st.markdown("**Vi phạm theo giờ trong ngày**")
        hours = df["Time"].dt.hour.value_counts().reindex(range(24), fill_value=0)
        fig = go.Figure(go.Bar(x=hours.index, y=hours.values, marker_color=ACCENT))
        fig.update_xaxes(dtick=2)
        st.plotly_chart(_style(fig), width="stretch")