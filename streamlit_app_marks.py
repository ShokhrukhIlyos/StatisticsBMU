from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


DATA_FILE = Path("StudentModuleMarksProgramEnrollment.csv")


# ----------------------------
# Helpers
# ----------------------------
def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def _clean_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            s = (
                df[col]
                .astype("string")
                .str.replace("\u00a0", " ", regex=False)
                .str.strip()
            )
            s = s.where(~s.str.lower().isin(["nan", "none", "null", ""]), other=pd.NA)
            df[col] = s
    return df


def detect_and_read_file(path: Path) -> pd.DataFrame:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8000]
    pipe_like = ("|" in sample) and (sample.count("|") > sample.count(","))

    if pipe_like:
        df = pd.read_csv(path, sep=r"\s*\|\s*", engine="python")
    else:
        df = pd.read_csv(path)

    df = _standardize_cols(df)

    # Drop empty/unnamed columns
    bad_cols = []
    for c in df.columns:
        c2 = str(c).strip()
        if c2 == "" or c2.lower().startswith("unnamed:"):
            bad_cols.append(c)
    if bad_cols:
        df = df.drop(columns=bad_cols, errors="ignore")

    return df


def load_and_clean(path: Path) -> pd.DataFrame:
    df = detect_and_read_file(path)

    required = {"StudentID", "ProgramName", "ModuleCode", "ModuleName", "TotalMark", "TotalGrade"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")

    df = _clean_text(df, ["StudentID", "ProgramName", "ModuleCode", "ModuleName", "TotalGrade"])

    df["TotalMark"] = pd.to_numeric(df["TotalMark"], errors="coerce")

    # remove NS
    df = df.dropna(subset=["TotalMark"]).copy()

    df["ModuleLabel"] = (
        df["ModuleCode"].astype("string").fillna("").str.strip()
        + " — "
        + df["ModuleName"].astype("string").fillna("").str.strip()
    )
    df.loc[df["ModuleLabel"].str.strip().isin(["", "—", " — "]), "ModuleLabel"] = pd.NA

    bins = [0, 25, 50, 75, 100.0000001]
    labels = ["0-25", "25-50", "50-75", "75-100"]
    df["ScoreRangeBand"] = pd.cut(df["TotalMark"], bins=bins, labels=labels, include_lowest=True)

    # GradePie
    m = df["TotalMark"]
    grade = pd.Series(pd.NA, index=df.index, dtype="string")
    grade.loc[m >= 80] = "A*"

    g = df["TotalGrade"].astype("string").str.strip().str.upper()
    use_grade = g.notna() & (g != "")
    grade.loc[grade.isna() & use_grade] = g.loc[grade.isna() & use_grade]

    need_fallback = grade.isna()
    mm = m.loc[need_fallback]
    grade.loc[need_fallback & (mm >= 70)] = "A"
    grade.loc[need_fallback & (mm >= 60) & (mm < 70)] = "B"
    grade.loc[need_fallback & (mm >= 50) & (mm < 60)] = "C"
    grade.loc[need_fallback & (mm >= 40) & (mm < 50)] = "D"
    grade.loc[need_fallback & (mm >= 30) & (mm < 40)] = "E"
    grade.loc[need_fallback & (mm < 30)] = "F"

    df["GradePie"] = grade
    return df


def program_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ProgramName", dropna=False)["TotalMark"]
    out = g.agg(
        Min="min",
        Mean="mean",
        Median="median",
        Max="max",
        StdDev="std",
        Count="count",
    ).reset_index()

    out["Mid"] = (out["Min"] + out["Max"]) / 2.0

    out["PerformanceTier"] = pd.cut(
        out["Mean"],
        bins=[-np.inf, 50, 60, 70, np.inf],
        labels=["Below Average", "Average", "Good", "Excellent"]
    )

    for c in ["Min", "Mean", "Median", "Max", "Mid", "StdDev"]:
        out[c] = out[c].round(2)

    return out


def module_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["ModuleCode", "ModuleName"], dropna=False)["TotalMark"]
    out = g.agg(
        Min="min",
        Mean="mean",
        Median="median",
        Max="max",
        StdDev="std",
        Count="count",
    ).reset_index()

    out["Mid"] = (out["Min"] + out["Max"]) / 2.0
    out["ModuleLabel"] = (
        out["ModuleCode"].astype("string").fillna("").str.strip()
        + " — "
        + out["ModuleName"].astype("string").fillna("").str.strip()
    )

    for c in ["Min", "Mean", "Median", "Max", "Mid", "StdDev"]:
        out[c] = out[c].round(2)

    return out


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Academic Program Performance Dashboard", layout="wide")
st.title("Academic Program Performance Dashboard")

if not DATA_FILE.exists():
    st.error(f"File not found: {DATA_FILE.resolve()}")
    st.stop()

df_raw = load_and_clean(DATA_FILE)
df_prog = program_metrics(df_raw)

data_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.caption(f"✔ Data Updated: {data_updated} • Data Source: StudentID | ProgramName | ModuleCode | ModuleName | TotalMark | TotalGrade")

program_options = sorted(df_prog["ProgramName"].dropna().unique().tolist())
score_band_options = ["0-25", "25-50", "50-75", "75-100"]
tier_options = ["Below Average", "Average", "Good", "Excellent"]

with st.sidebar:
    st.header("Filters")
    f_program = st.multiselect("Program Name", program_options, default=program_options)
    f_scoreband = st.multiselect("Score Range", score_band_options, default=score_band_options)
    f_tier = st.multiselect("Performance Tier", tier_options, default=tier_options)
    top_n = st.slider("Top N", 5, 20, 10, 1)
    sort_order = st.selectbox("Sort Order (Mean)", ["desc", "asc"], index=0)

# Apply filters
dff_raw = df_raw.copy()
if f_program:
    dff_raw = dff_raw[dff_raw["ProgramName"].isin(f_program)]
if f_scoreband:
    dff_raw = dff_raw[dff_raw["ScoreRangeBand"].isin(f_scoreband)]

dff = df_prog.copy()
if f_program:
    dff = dff[dff["ProgramName"].isin(f_program)]
if f_tier:
    dff = dff[dff["PerformanceTier"].isin(f_tier)]

dff = dff.sort_values("Mean", ascending=(sort_order == "asc"), na_position="last")

# KPIs
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
col1.metric("Total Programs", int(dff["ProgramName"].nunique()))
col2.metric("Average (Mean) Score", f"{dff['Mean'].mean():.2f}" if len(dff) else "—")
col3.metric("Highest Max Score", f"{dff['Max'].max():.2f}" if len(dff) else "—")
col4.metric("Average StdDev", f"{dff['StdDev'].mean():.2f}" if len(dff) else "—")
col5.metric("Records Used", f"{len(dff_raw):,}")

# ----------------------------
# Charts (UPDATED AS REQUESTED)
# ----------------------------

# ✅ Standard Deviation by Program (LINE graph)
st.subheader("Standard Deviation by Program (Trend Line)")

std_trend = (
    dff.dropna(subset=["StdDev"])
       .sort_values("StdDev", ascending=False)  # keep the "downward" feel
)

fig_std_line = px.line(
    std_trend,
    x="ProgramName",
    y="StdDev",
    markers=True,
    title=""
)
fig_std_line.update_layout(
    xaxis_title="Program Name",
    yaxis_title="Standard Deviation",
    xaxis_tickangle=-30
)
st.plotly_chart(fig_std_line, use_container_width=True)

# ✅ Same row:
# Left = Top Programs by Mean
# Right = Score Distribution Trend by Program (Median) as BAR chart (like old StdDev histogram)
cA, cB = st.columns(2)

with cA:
    st.subheader("Top Programs by Mean Score (Average)")
    top_mean = dff.dropna(subset=["Mean"]).head(int(top_n))
    fig_top_mean = px.bar(
        top_mean.iloc[::-1],
        x="Mean",
        y="ProgramName",
        orientation="h",
        title=""
    )
    st.plotly_chart(fig_top_mean, use_container_width=True)

with cB:
    st.subheader("Score Distribution Trend by Program (Median)")

    median_bar = (
        dff.dropna(subset=["Median"])
           .sort_values("Median", ascending=False)
           .head(int(top_n))
    )

    fig_median_bar = px.bar(
        median_bar,
        x="Median",
        y="ProgramName",
        orientation="h",
        title=""
    )
    fig_median_bar.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_median_bar, use_container_width=True)

# ❌ Removed: Top Programs by Maximum Score


# ----------------------------
# Module Statistics
# ----------------------------
st.subheader("Module Statistics")

colA, colB = st.columns([2, 1])

with colA:
    module_metric = st.selectbox(
        "Module Metric",
        ["Mean", "Median", "Min", "Max", "Mid", "StdDev"],
        index=0,
        key="module_metric"
    )

with colB:
    module_top_n = st.selectbox(
        "Top N Modules",
        [5, 10, 15, 20, 25, 50],
        index=1,
        key="module_top_n"
    )

dff_mod = module_metrics(dff_raw).dropna(subset=[module_metric])

top_modules = (
    dff_mod.sort_values(module_metric, ascending=False)
           .head(int(module_top_n))
)

dynamic_height = max(450, 40 * len(top_modules))

fig_module = px.bar(
    top_modules,
    x=module_metric,
    y="ModuleLabel",
    orientation="h",
    title=""
)
fig_module.update_layout(
    yaxis=dict(autorange="reversed"),
    height=dynamic_height,
    margin=dict(l=10, r=10, t=30, b=10),
)

st.plotly_chart(fig_module, use_container_width=True)

# ----------------------------
# Donuts
# ----------------------------
st.subheader("Grade Distribution by Program")

grade_order = ["A*", "A", "B", "C", "D", "E", "F"]
grade_colors = {
    "A*": "#2ca02c", "A": "#98df8a", "B": "#1f77b4", "C": "#9467bd",
    "D": "#ff7f0e", "E": "#bcbd22", "F": "#d62728",
}

programs_to_draw = sorted(dff_raw["ProgramName"].dropna().unique().tolist())

for i in range(0, len(programs_to_draw), 2):
    row_cols = st.columns(2)

    for j in range(2):
        idx = i + j
        if idx >= len(programs_to_draw):
            break

        prog = programs_to_draw[idx]
        prog_df = dff_raw[dff_raw["ProgramName"] == prog]

        counts = prog_df["GradePie"].value_counts().reindex(grade_order, fill_value=0)
        total = int(counts.sum()) if int(counts.sum()) > 0 else 1

        grade_counts = counts.reset_index()
        grade_counts.columns = ["Grade", "Count"]

        grade_counts["Grade"] = pd.Categorical(
            grade_counts["Grade"], categories=grade_order, ordered=True
        )
        grade_counts = grade_counts.sort_values("Grade")

        grade_counts["Percent"] = (grade_counts["Count"] / total * 100).round(1)
        grade_counts["LegendLabel"] = (
            grade_counts["Grade"].astype(str)
            + "  "
            + grade_counts["Count"].astype(int).astype(str)
            + " ("
            + grade_counts["Percent"].astype(str)
            + "%)"
        )

        fig = px.pie(
            grade_counts,
            names="LegendLabel",
            values="Count",
            hole=0.55,
            color="Grade",
            color_discrete_map=grade_colors,
        )

        fig.update_traces(
            sort=False,
            direction="clockwise",
            rotation=0,
            textinfo="none",
            texttemplate="%{value} (%{percent})",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )

        fig.update_layout(
            showlegend=True,
            legend_title_text="",
            margin=dict(l=0, r=0, t=30, b=0),
        )

        row_cols[j].markdown(f"**{prog}**")
        row_cols[j].plotly_chart(fig, use_container_width=True)
