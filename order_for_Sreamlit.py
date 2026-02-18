from __future__ import annotations

from pathlib import Path
import csv
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Config
# ----------------------------
DATA_FILE = Path("Orders.csv")


# ----------------------------
# Helpers
# ----------------------------
def clean_str_series(s: pd.Series) -> pd.Series:
    s2 = (
        s.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
    )
    s2 = s2.where(~s2.str.lower().isin(["nan", "none", "null", ""]), other=pd.NA)
    return s2


def sniff_delimiter(path: Path, fallback: str = ",") -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "|", "\t", ";"])
        return dialect.delimiter
    except Exception:
        if sample.count("|") > sample.count(",") and sample.count("|") > 5:
            return "|"
        return fallback


@st.cache_data(show_spinner=False)
def load_and_clean_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")

    delim = sniff_delimiter(path)
    df = pd.read_csv(path, sep=delim, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names (accept variants)
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["studentid", "student_id", "student id"]:
            col_map[c] = "StudentID"
        elif lc in ["programname", "program_name", "program name"]:
            col_map[c] = "ProgramName"
        elif lc in ["ordertypedescription", "order type description", "ordertypedesc", "order_type_description"]:
            col_map[c] = "OrderTypeDescription"
        elif lc in ["orderdate", "order_date", "order date"]:
            col_map[c] = "OrderDate"

    df = df.rename(columns=col_map)

    required = ["StudentID", "ProgramName", "OrderTypeDescription", "OrderDate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df["StudentID"] = clean_str_series(df["StudentID"])
    df["ProgramName"] = clean_str_series(df["ProgramName"])
    df["OrderTypeDescription"] = clean_str_series(df["OrderTypeDescription"])
    df["OrderDateRaw"] = clean_str_series(df["OrderDate"])

    # Parse date (try YYYY-MM-DD first, then fallback)
    df["OrderDate"] = pd.to_datetime(df["OrderDateRaw"], errors="coerce", format="%Y-%m-%d")
    bad = df["OrderDate"].isna() & df["OrderDateRaw"].notna()
    if bad.any():
        df.loc[bad, "OrderDate"] = pd.to_datetime(df.loc[bad, "OrderDateRaw"], errors="coerce")

    df["IsBadDate"] = df["OrderDate"].isna()

    # Keep valid key fields
    df = df.dropna(subset=["StudentID", "ProgramName", "OrderTypeDescription"]).copy()
    return df


def to_period_start(s: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return s.dt.floor("D")
    if freq == "W":
        return s.dt.to_period("W-MON").dt.start_time
    if freq == "M":
        return s.dt.to_period("M").dt.start_time
    raise ValueError("freq must be one of D/W/M")


def build_kpis(df_all: pd.DataFrame, df_valid: pd.DataFrame) -> dict:
    return {
        "total_rows": int(len(df_all)),
        "valid_dated_rows": int(len(df_valid)),
        "bad_date_rows": int(df_all["OrderDate"].isna().sum()),
        "unique_students": int(df_valid["StudentID"].nunique(dropna=True)),
        "unique_programs": int(df_valid["ProgramName"].nunique(dropna=True)),
        "unique_types": int(df_valid["OrderTypeDescription"].nunique(dropna=True)),
        "date_min": df_valid["OrderDate"].min().strftime("%Y-%m-%d") if len(df_valid) else "",
        "date_max": df_valid["OrderDate"].max().strftime("%Y-%m-%d") if len(df_valid) else "",
    }


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Orders Dashboard", layout="wide")
st.title("Orders Dashboard")

if not DATA_FILE.exists():
    st.error(f"File not found: {DATA_FILE.resolve()}")
    st.stop()

df = load_and_clean_orders(DATA_FILE)
df_valid = df.dropna(subset=["OrderDate"]).copy()

k = build_kpis(df, df_valid)
updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

st.caption(
    f"✔ Data Updated: {updated} • Source: {DATA_FILE.name} • "
    f"Valid date range: {k['date_min']} → {k['date_max']}"
)

# KPIs (removed Bad / Missing Dates)
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Rows", f"{k['total_rows']:,}")
c2.metric("Valid Dated Rows", f"{k['valid_dated_rows']:,}")
c3.metric("Unique Students", f"{k['unique_students']:,}")
c4.metric("Programs", f"{k['unique_programs']:,}")
c5.metric("Order Types", f"{k['unique_types']:,}")


types = sorted(df_valid["OrderTypeDescription"].dropna().unique().tolist())

with st.sidebar:
    st.header("Controls")
    type_choice = st.selectbox("Type", ["All"] + types, index=0)
    freq_choice = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)

    # ✅ show Top Students ONLY if not "All"
    show_top_students = st.checkbox("Show Top Students", value=True)

    # slider only useful when we show it
    top_n = st.slider("Top N Students", 5, 50, 15, 1) if show_top_students else 15

freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
freq = freq_map[freq_choice]

# Filter by type
if type_choice == "All":
    dff = df_valid
else:
    dff = df_valid[df_valid["OrderTypeDescription"] == type_choice].copy()

# ----------------------------
# Row 1: Orders Over Time + Orders by Type
# ----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Orders Over Time")

    dff_ts = dff.copy()
    dff_ts["Bucket"] = to_period_start(dff_ts["OrderDate"], freq)
    s = dff_ts.groupby("Bucket").size().sort_index()

    if len(s):
        full_idx = pd.date_range(
            s.index.min(),
            s.index.max(),
            freq=("D" if freq == "D" else ("W-MON" if freq == "W" else "MS"))
        )
        s = s.reindex(full_idx, fill_value=0)

    df_ts = pd.DataFrame({"Date": s.index, "Orders": s.values})
    fig_ts = px.line(df_ts, x="Date", y="Orders", markers=True, title="")
    fig_ts.update_layout(xaxis_title="", yaxis_title="Orders", height=360)
    st.plotly_chart(fig_ts, use_container_width=True)

with right:
    st.subheader("Orders by Type (Overall)")
    bt = df_valid.groupby("OrderTypeDescription").size().sort_values(ascending=False)
    df_bt = bt.reset_index()
    df_bt.columns = ["OrderType", "Count"]

    fig_bt = px.bar(df_bt.iloc[::-1], x="Count", y="OrderType", orientation="h", title="")
    fig_bt.update_layout(xaxis_title="Count", yaxis_title="", height=360)
    st.plotly_chart(fig_bt, use_container_width=True)

# ----------------------------
# Row 2: Top students (optional)
#  ✅ Removed Calendar entirely
#  ✅ Hide Top Students when Type = All (your request)
# ----------------------------
if show_top_students and type_choice != "All":
    st.subheader(f"Top Students by Orders ({type_choice})")

    top = dff.groupby("StudentID").size().sort_values(ascending=False).head(int(top_n))
    df_top = top.reset_index()
    df_top.columns = ["StudentID", "Count"]

    fig_top = px.bar(df_top.iloc[::-1], x="Count", y="StudentID", orientation="h", title="")
    fig_top.update_layout(
        xaxis_title="Orders",
        yaxis_title="",
        height=max(360, 26 * len(df_top) + 180)
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ----------------------------
# Pies: one pie per type, slices=program
# ✅ Bigger pies + better legend
# ----------------------------
st.subheader("Order Type Distribution by Program")
st.caption("One pie per **OrderTypeDescription**. Slices are **ProgramName**.")

pie_size = st.slider("Pie Size", 420, 800, 560, 20)

cols = st.columns(2)
col_idx = 0

for t in types:
    dtt = df_valid[df_valid["OrderTypeDescription"] == t]
    counts = dtt.groupby("ProgramName").size().sort_values(ascending=False)

    df_pie = counts.reset_index()
    df_pie.columns = ["ProgramName", "Count"]

    fig_pie = px.pie(
        df_pie,
        names="ProgramName",
        values="Count",
        hole=0.55,
        title=t,
    )

    fig_pie.update_traces(
        sort=False,
        direction="clockwise",
        rotation=0,  # start at 12:00
        texttemplate="%{value} (%{percent})",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )

    fig_pie.update_layout(
        height=pie_size,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="top", y=-0.15),
        margin=dict(l=10, r=10, t=60, b=80),
    )

    with cols[col_idx]:
        st.plotly_chart(fig_pie, use_container_width=True)

    col_idx = 1 - col_idx
