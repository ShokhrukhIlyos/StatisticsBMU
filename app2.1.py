from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

import plotly.express as px

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False)

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


# ----------------------------
# Data
# ----------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    df = detect_and_read_file(path)

    required = {"StudentID", "ProgramName", "ModuleCode", "ModuleName", "TotalMark", "TotalGrade"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = _clean_text(df, ["StudentID", "ProgramName", "ModuleCode", "ModuleName", "TotalGrade"])

    # TotalMark
    df["TotalMark"] = pd.to_numeric(df["TotalMark"], errors="coerce")

    # REMOVE ALL NS: drop rows where TotalMark is missing
    df = df.dropna(subset=["TotalMark"]).copy()

    # Module label
    df["ModuleLabel"] = (
        df["ModuleCode"].astype("string").fillna("").str.strip()
        + " — "
        + df["ModuleName"].astype("string").fillna("").str.strip()
    )
    df.loc[df["ModuleLabel"].str.strip().isin(["", "—", " — "]), "ModuleLabel"] = pd.NA

    # Score bands (no NS exists now)
    bins = [0, 25, 50, 75, 100.0000001]
    labels = ["0-25", "25-50", "50-75", "75-100"]
    df["ScoreRangeBand"] = pd.cut(df["TotalMark"], bins=bins, labels=labels, include_lowest=True)

    # GradePie (no NS)
    m = df["TotalMark"]
    grade = pd.Series(pd.NA, index=df.index, dtype="string")

    # A* based on mark>=80
    grade.loc[m >= 80] = "A*"

    # use TotalGrade if present
    g = df["TotalGrade"].astype("string").str.strip().str.upper()
    use_grade = g.notna() & (g != "")
    grade.loc[grade.isna() & use_grade] = g.loc[grade.isna() & use_grade]

    # fallback from mark where TotalGrade missing
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
        Mean="mean",       # average
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


def kpi_card(title: str, value: str, icon: str = "✅"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"color": "#6b7280", "fontSize": "12px"}),
            html.Div([html.Span(icon, style={"marginRight": "6px"}), value],
                     style={"fontSize": "22px", "fontWeight": "700"})
        ]),
        className="shadow-sm",
        style={"borderRadius": "14px"}
    )


# ----------------------------
# App start
# ----------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"File not found: {DATA_FILE.resolve()}")

df_raw = load_and_clean(DATA_FILE)
df_prog = program_metrics(df_raw)

data_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Academic Program Performance Dashboard"

program_options = sorted([p for p in df_prog["ProgramName"].dropna().unique().tolist()])
score_band_options = ["0-25", "25-50", "50-75", "75-100"]
tier_options = ["Below Average", "Average", "Good", "Excellent"]
metric_options = ["Min", "Median", "Mean", "Max", "Mid", "StdDev"]

sidebar = dbc.Card(
    dbc.CardBody([
        html.H5("Filters", style={"fontWeight": "700"}),

        html.Div("Program Name", className="mt-2", style={"fontSize": "12px", "color": "#6b7280"}),
        dcc.Dropdown(
            id="f_program",
            options=[{"label": p, "value": p} for p in program_options],
            value=program_options,
            multi=True
        ),

        html.Div("Score Range", className="mt-3", style={"fontSize": "12px", "color": "#6b7280"}),
        dcc.Dropdown(
            id="f_scoreband",
            options=[{"label": s, "value": s} for s in score_band_options],
            value=score_band_options,
            multi=True
        ),

        html.Div("Performance Tier", className="mt-3", style={"fontSize": "12px", "color": "#6b7280"}),
        dcc.Dropdown(
            id="f_tier",
            options=[{"label": t, "value": t} for t in tier_options],
            value=tier_options,
            multi=True
        ),

        html.Div("Metric Type", className="mt-3", style={"fontSize": "12px", "color": "#6b7280"}),
        dcc.Dropdown(
            id="f_metric",
            options=[{"label": m, "value": m} for m in metric_options],
            value=["Min", "Median", "Mean", "Max"],
            multi=True
        ),

        html.Hr(),
        html.Div("Top N", style={"fontSize": "12px", "color": "#6b7280"}),
        dcc.Slider(id="top_n", min=5, max=20, step=1, value=10,
                   marks={5: "5", 10: "10", 15: "15", 20: "20"}),

        html.Div("Sort Order (Mean)", className="mt-3", style={"fontSize": "12px", "color": "#6b7280"}),
        dcc.Dropdown(
            id="sort_order",
            options=[{"label": "Mean Descending", "value": "desc"}, {"label": "Mean Ascending", "value": "asc"}],
            value="desc",
            clearable=False
        ),
    ]),
    className="shadow-sm",
    style={"borderRadius": "14px"}
)

header = dbc.Card(
    dbc.CardBody([
        html.H3("Academic Program Performance Dashboard", style={"fontWeight": "800", "marginBottom": "6px"}),
        html.Div([
            html.Span("✔ Data Updated: ", style={"fontWeight": "600"}),
            html.Span(data_updated),
            html.Span("   •   "),
            html.Span("🗄 Data Source: StudentID | ProgramName | ModuleCode | ModuleName | TotalMark | TotalGrade"),
        ], style={"color": "#6b7280", "fontSize": "13px"})
    ]),
    className="shadow-sm",
    style={"borderRadius": "14px"}
)

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([dbc.Col(header, width=12)], className="mt-3"),

    dbc.Row([
        dbc.Col(sidebar, width=3, className="mt-3"),

        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(id="kpi1"), width=12, lg=2),
                dbc.Col(html.Div(id="kpi2"), width=12, lg=2),
                dbc.Col(html.Div(id="kpi3"), width=12, lg=2),
                dbc.Col(html.Div(id="kpi4"), width=12, lg=2),
                dbc.Col(html.Div(id="kpi5"), width=12, lg=4),
            ], className="mt-3 g-2"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Score Distribution Trend by Program", style={"fontWeight": "700"}),
                    dcc.Graph(id="g_trend", config={"displayModeBar": False})
                ]), className="shadow-sm", style={"borderRadius": "14px"}), width=12),
            ], className="mt-3"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Top Programs by Mean Score (Average)", style={"fontWeight": "700"}),
                    dcc.Graph(id="g_top_mean", config={"displayModeBar": False})
                ]), className="shadow-sm", style={"borderRadius": "14px"}), width=6),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Standard Deviation by Program", style={"fontWeight": "700"}),
                    dcc.Graph(id="g_stddev", config={"displayModeBar": False})
                ]), className="shadow-sm", style={"borderRadius": "14px"}), width=6),
            ], className="mt-3"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Top Programs by Maximum Score", style={"fontWeight": "700"}),
                    dcc.Graph(id="g_top_max", config={"displayModeBar": False})
                ]), className="shadow-sm", style={"borderRadius": "14px"}), width=12),
            ], className="mt-3"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                        children=[
                            html.H5("Module Statistics", style={"fontWeight": "700", "marginBottom": "0px"}),
                            dcc.Dropdown(
                                id="module_metric",
                                options=[
                                    {"label": "Mean", "value": "Mean"},
                                    {"label": "Median", "value": "Median"},
                                    {"label": "Min", "value": "Min"},
                                    {"label": "Max", "value": "Max"},
                                    {"label": "Mid (Min+Max)/2", "value": "Mid"},
                                    {"label": "StdDev", "value": "StdDev"},
                                ],
                                value="Mean",
                                clearable=False,
                                style={"width": "220px"}
                            )
                        ]
                    ),
                    html.Div("Top modules ranked by selected metric (based on current filters).",
                             style={"color": "#6b7280", "fontSize": "12px", "marginTop": "6px"}),
                    dcc.Graph(id="g_module_stats", config={"displayModeBar": False}),
                ]), className="shadow-sm", style={"borderRadius": "14px"}), width=12),
            ], className="mt-3"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Grade Distribution by Program", style={"fontWeight": "700"}),
                    html.Div("Donut charts + grade legend with % on the right (no NS).",
                             style={"color": "#6b7280", "fontSize": "12px"}),
                    html.Div(id="grade_pie_grid")
                ]), className="shadow-sm", style={"borderRadius": "14px"}), width=12),
            ], className="mt-3"),

        ], width=9)
    ], className="g-3"),
])


@app.callback(
    Output("kpi1", "children"),
    Output("kpi2", "children"),
    Output("kpi3", "children"),
    Output("kpi4", "children"),
    Output("kpi5", "children"),
    Output("g_trend", "figure"),
    Output("g_top_mean", "figure"),
    Output("g_stddev", "figure"),
    Output("g_top_max", "figure"),
    Output("g_module_stats", "figure"),
    Output("grade_pie_grid", "children"),
    Input("f_program", "value"),
    Input("f_scoreband", "value"),
    Input("f_tier", "value"),
    Input("f_metric", "value"),
    Input("top_n", "value"),
    Input("sort_order", "value"),
    Input("module_metric", "value"),
)
def update_dashboard(programs, scorebands, tiers, metrics, top_n, sort_order, module_metric):
    programs = programs or []
    scorebands = scorebands or []
    tiers = tiers or []
    metrics = metrics or []

    # Filter raw rows (no NS exists)
    dff_raw = df_raw.copy()
    if programs:
        dff_raw = dff_raw[dff_raw["ProgramName"].isin(programs)]
    if scorebands:
        dff_raw = dff_raw[dff_raw["ScoreRangeBand"].isin(scorebands)]

    # Program metrics filtered
    dff = df_prog.copy()
    if programs:
        dff = dff[dff["ProgramName"].isin(programs)]
    if tiers:
        dff = dff[dff["PerformanceTier"].isin(tiers)]

    dff = dff.sort_values("Mean", ascending=(sort_order == "asc"), na_position="last")

    # KPIs
    total_programs = int(dff["ProgramName"].nunique())
    avg_mean = float(dff["Mean"].mean()) if len(dff) else None
    highest_max = float(dff["Max"].max()) if len(dff) else None
    avg_std = float(dff["StdDev"].mean()) if len(dff) else None
    records_used = int(len(dff_raw))

    k1 = kpi_card("Total Programs", f"{total_programs}", "📌")
    k2 = kpi_card("Average (Mean) Score", f"{avg_mean:.2f}" if avg_mean is not None else "—", "📈")
    k3 = kpi_card("Highest Max Score", f"{highest_max:.2f}" if highest_max is not None else "—", "🏆")
    k4 = kpi_card("Average StdDev", f"{avg_std:.2f}" if avg_std is not None else "—", "📊")
    k5 = kpi_card("Records Used", f"{records_used:,}", "🧾")

    # Trend metric (defaults to Mean)
    show_metric = "Median" if ("Median" in metrics) else (metrics[0] if metrics else "Median")
    fig_trend = px.bar(dff, x="ProgramName", y=show_metric, title="")
    fig_trend.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Program Name", yaxis_title=f"{show_metric} Mark")
    fig_trend.update_traces(hovertemplate="<b>%{x}</b><br>" + f"{show_metric}: " + "%{y:.2f}<extra></extra>")

    # Top mean
    top_mean = dff.dropna(subset=["Mean"]).head(int(top_n))
    fig_top_mean = px.bar(top_mean.iloc[::-1], x="Mean", y="ProgramName", orientation="h", title="")
    fig_top_mean.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Mean (Average) Mark", yaxis_title="")
    fig_top_mean.update_traces(hovertemplate="<b>%{y}</b><br>Mean: %{x:.2f}<extra></extra>")

    # StdDev chart
    std_df = dff.dropna(subset=["StdDev"]).head(int(top_n))
    fig_std = px.bar(std_df.iloc[::-1], x="StdDev", y="ProgramName", orientation="h", title="")
    fig_std.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Standard Deviation", yaxis_title="")
    fig_std.update_traces(hovertemplate="<b>%{y}</b><br>StdDev: %{x:.2f}<extra></extra>")

    # Top max
    top_max = dff.sort_values("Max", ascending=False, na_position="last").head(int(top_n)).dropna(subset=["Max"])
    fig_top_max = px.bar(top_max.iloc[::-1], x="Max", y="ProgramName", orientation="h", title="")
    fig_top_max.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Max Mark", yaxis_title="")
    fig_top_max.update_traces(hovertemplate="<b>%{y}</b><br>Max: %{x:.2f}<extra></extra>")

    # Module stats
    dff_mod = module_metrics(dff_raw).dropna(subset=[module_metric])
    top_modules = dff_mod.sort_values(module_metric, ascending=False).head(15)

    fig_module = px.bar(
        top_modules.iloc[::-1],
        x=module_metric,
        y="ModuleLabel",
        orientation="h",
        title=""
    )
    fig_module.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title=f"{module_metric}",
                             yaxis_title="")
    fig_module.update_traces(
        customdata=top_modules.iloc[::-1][["Count", "Min", "Mean", "Median", "Max", "Mid", "StdDev"]].to_numpy(),
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{module_metric}: %{{x:.2f}}<br>"
            "Count: %{customdata[0]}<br>"
            "Min: %{customdata[1]:.2f}<br>"
            "Mean: %{customdata[2]:.2f}<br>"
            "Median: %{customdata[3]:.2f}<br>"
            "Max: %{customdata[4]:.2f}<br>"
            "Mid: %{customdata[5]:.2f}<br>"
            "StdDev: %{customdata[6]:.2f}<br>"
            "<extra></extra>"
        )
    )

    # -------------------------
    # Grade donut grid (no NS)  <-- IMPORTANT: INSIDE function
    # -------------------------
    grade_order = ["A*", "A", "B", "C", "D", "E", "F"]
    grade_colors = {
        "A*": "#2ca02c", "A": "#98df8a", "B": "#1f77b4", "C": "#9467bd",
        "D": "#ff7f0e", "E": "#bcbd22", "F": "#d62728",
    }

    programs_to_draw = sorted([p for p in dff_raw["ProgramName"].dropna().unique().tolist()])
    pie_cards = []

    for prog in programs_to_draw:
        prog_df = dff_raw[dff_raw["ProgramName"] == prog]
        counts = prog_df["GradePie"].value_counts().reindex(grade_order, fill_value=0)
        total = int(counts.sum())

        grade_counts = counts.reset_index()
        grade_counts.columns = ["Grade", "Count"]

        # Force order for plotly
        grade_counts["Grade"] = pd.Categorical(
            grade_counts["Grade"],
            categories=grade_order,
            ordered=True
        )
        grade_counts = grade_counts.sort_values("Grade")

        fig = px.pie(
            grade_counts,
            names="Grade",
            values="Count",
            hole=0.55,
            color="Grade",
            color_discrete_map=grade_colors,
            category_orders={"Grade": grade_order},
        )

        fig.update_traces(
            sort=False,
            direction="clockwise",
            rotation=0,  # start at 12:00
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)

        legend_rows = []
        for g in grade_order:
            c = int(counts[g])
            if c == 0:
                continue
            pct = (c / total * 100) if total else 0.0
            legend_rows.append(
                html.Div(
                    style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                           "gap": "10px", "padding": "2px 0"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Span("●", style={"color": grade_colors[g], "fontSize": "14px"}),
                                html.Span(g, style={"fontWeight": "700", "fontSize": "12px"}),
                            ],
                        ),
                        html.Span(f"{c}  ({pct:.1f}%)", style={"fontSize": "12px", "color": "#374151"}),
                    ],
                )
            )

        pie_cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(prog, style={"fontWeight": "700", "fontSize": "13px", "marginBottom": "10px"}),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1.2fr 1fr", "gap": "10px",
                                   "alignItems": "center"},
                            children=[
                                dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "240px"}),
                                html.Div(style={"borderLeft": "1px solid #e5e7eb", "paddingLeft": "10px"},
                                         children=legend_rows)
                            ]
                        )
                    ]),
                    className="shadow-sm",
                    style={"borderRadius": "14px"}
                ),
                width=6
            )
        )

    pie_grid = dbc.Row(pie_cards, className="g-2")

    return (
        k1, k2, k3, k4, k5,
        fig_trend, fig_top_mean, fig_std, fig_top_max,
        fig_module,
        pie_grid
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

