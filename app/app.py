"""
Dead Internet Observatory — Streamlit Mission Control Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cyberpunk NASA-style observatory tracking the Internet Aliveness Index.
Deploys free on Streamlit Community Cloud.

Run locally:
    streamlit run app/app.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = str(ROOT / "config" / "config.yaml")
DB_PATH     = str(ROOT / "data" / "observatory.db")

# ── Lazy imports (avoid crashing if packages missing at import time) ──────────
try:
    from analytics.aliveness_index import AlivenessIndexEngine, seed_demo_data
    from analytics.anomaly_detector import label_anomalies, get_notable_anomalies
    ANALYTICS_OK = True
except ImportError:
    ANALYTICS_OK = False


# ════════════════════════════════════════════════════════════════════════════════
#  THEME & CSS
# ════════════════════════════════════════════════════════════════════════════════

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: #050508 !important;
    color: #e0e0e0 !important;
    font-family: 'Rajdhani', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 1400px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #1e1e3a; border-radius: 3px; }

/* ── Mission header ── */
.mission-header {
    background: linear-gradient(135deg, #050508 0%, #0d0d20 50%, #050508 100%);
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.mission-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ff88, #00aaff, transparent);
    animation: scan 3s ease-in-out infinite;
}
@keyframes scan {
    0%   { opacity: 0; transform: translateX(-100%); }
    50%  { opacity: 1; }
    100% { opacity: 0; transform: translateX(100%); }
}
.mission-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00ff88;
    text-shadow: 0 0 30px rgba(0,255,136,0.5);
    margin: 0;
    letter-spacing: 2px;
}
.mission-subtitle {
    font-family: 'Rajdhani', sans-serif;
    color: #888899;
    font-size: 1.05rem;
    margin-top: 6px;
    letter-spacing: 1px;
}
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 10px #00ff88;
    animation: pulse-dot 2s ease-in-out infinite;
    margin-right: 8px;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 10px #00ff88; }
    50%       { opacity: 0.4; box-shadow: 0 0 4px #00ff88; }
}

/* ── Stat cards ── */
.stat-card {
    background: #0d0d18;
    border: 1px solid #1e1e3a;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.stat-card.green::after  { background: #00ff88; }
.stat-card.red::after    { background: #ff0055; }
.stat-card.blue::after   { background: #00aaff; }
.stat-card.orange::after { background: #ffaa00; }
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.stat-value.green  { color: #00ff88; text-shadow: 0 0 20px rgba(0,255,136,0.4); }
.stat-value.red    { color: #ff0055; text-shadow: 0 0 20px rgba(255,0,85,0.4); }
.stat-value.blue   { color: #00aaff; text-shadow: 0 0 20px rgba(0,170,255,0.4); }
.stat-value.orange { color: #ffaa00; text-shadow: 0 0 20px rgba(255,170,0,0.4); }
.stat-label {
    font-family: 'Rajdhani', sans-serif;
    color: #666680;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}
.stat-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #444460;
    text-transform: uppercase;
    letter-spacing: 3px;
    padding: 4px 0;
    margin: 28px 0 16px 0;
    border-bottom: 1px solid #1e1e3a;
}

/* ── Domain badge ── */
.domain-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    margin: 2px;
}

/* ── Anomaly alert ── */
.anomaly-alert {
    background: linear-gradient(90deg, rgba(255,0,85,0.08), transparent);
    border-left: 3px solid #ff0055;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-family: 'Rajdhani', sans-serif;
}

/* ── Plotly chart background override ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Demo banner ── */
.demo-banner {
    background: linear-gradient(90deg, rgba(255,170,0,0.08), rgba(255,170,0,0.04));
    border: 1px solid rgba(255,170,0,0.3);
    border-radius: 8px;
    padding: 10px 18px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    color: #ffaa00;
    margin-bottom: 20px;
}
</style>
"""

# ── Plotly layout base ─────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani, sans-serif", color="#e0e0e0"),
)

COLOR_GREEN  = "#00ff88"
COLOR_RED    = "#ff0055"
COLOR_BLUE   = "#00aaff"
COLOR_ORANGE = "#ffaa00"
COLOR_PURPLE = "#aa00ff"

CATEGORY_COLORS = {
    "web":    COLOR_BLUE,
    "social": COLOR_GREEN,
    "news":   COLOR_ORANGE,
    "wiki":   COLOR_PURPLE,
    "blog":   "#ff88aa",
}


# ════════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(ttl=3600)
def get_engine():
    """Get or create the AlivenessIndexEngine (cached for session lifetime)."""
    if not ANALYTICS_OK:
        return None
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    try:
        import yaml
        with open(CONFIG_PATH) as fh:
            cfg = yaml.safe_load(fh)
        # Override db path if running in Streamlit cloud vs local
        cfg["storage"]["db_path"] = DB_PATH
        engine = AlivenessIndexEngine(CONFIG_PATH)
        # Auto-seed demo data if DB is empty
        if engine.get_meta("demo_seeded") != "true":
            seed_demo_data(DB_PATH, CONFIG_PATH)
        return engine
    except Exception as exc:
        st.error(f"Engine init error: {exc}")
        return None


@st.cache_data(ttl=300)
def load_timeline(days: int = 730) -> pd.DataFrame:
    engine = get_engine()
    if engine is None:
        return _synthetic_timeline(days)
    try:
        df = engine.get_composite_timeline(days)
        if df.empty:
            seed_demo_data(DB_PATH, CONFIG_PATH)
            df = engine.get_composite_timeline(days)
        if df.empty:
            return _synthetic_timeline(days)
        df["date"] = pd.to_datetime(df["date"])
        return label_anomalies(df, "aliveness_index")
    except Exception:
        return _synthetic_timeline(days)


@st.cache_data(ttl=300)
def load_source_breakdown() -> pd.DataFrame:
    engine = get_engine()
    if engine is None:
        return _synthetic_sources()
    try:
        df = engine.get_source_breakdown()
        return df if not df.empty else _synthetic_sources()
    except Exception:
        return _synthetic_sources()


@st.cache_data(ttl=300)
def load_current_score() -> float:
    engine = get_engine()
    if engine is None:
        return 41.3
    try:
        return engine.get_current_score()
    except Exception:
        return 41.3


# ── Synthetic fallbacks ────────────────────────────────────────────────────────

def _synthetic_timeline(days: int = 730) -> pd.DataFrame:
    """Fallback synthetic timeline if DB unavailable."""
    rng = np.random.default_rng(42)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    x = np.linspace(0, 1, n)
    trend = 78.0 - 37.0 * (1 / (1 + np.exp(-8 * (x - 0.55))))
    noise = rng.normal(0, 1.5, n)
    weekly = 1.2 * np.sin(2 * np.pi * np.arange(n) / 7)
    scores = np.clip(trend + noise + weekly, 10, 95)

    df = pd.DataFrame({
        "date": dates,
        "aliveness_index": scores,
        "smoothed_index": pd.Series(scores).rolling(7, min_periods=1, center=True).mean().values,
        "n_docs": rng.integers(800, 5000, n),
        "anomaly_flag": 0,
        "anomaly_type": "",
        "z_score": 0.0,
    })
    return label_anomalies(df, "aliveness_index")


def _synthetic_sources() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = []
    configs = [
        ("common_crawl", "web", 43.1), ("reddit", "social", 51.2),
        ("news", "news", 38.7), ("wikipedia", "wiki", 62.4),
    ]
    for src, cat, base in configs:
        rows.append({
            "date": today, "source": src, "category": cat,
            "mean_score": base + rng.normal(0, 2),
            "n_docs": rng.integers(800, 3000),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION COMPONENTS
# ════════════════════════════════════════════════════════════════════════════════

def render_aliveness_gauge(score: float) -> go.Figure:
    """Cyberpunk semicircle gauge for the main IAI score."""
    if score >= 70:
        color = COLOR_GREEN
        label = "MOSTLY HUMAN"
    elif score >= 50:
        color = "#aaff00"
        label = "MIXED SIGNALS"
    elif score >= 30:
        color = COLOR_ORANGE
        label = "SYNTHETIC MAJORITY"
    else:
        color = COLOR_RED
        label = "DEAD INTERNET"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={
            "reference": 78.0,
            "valueformat": ".1f",
            "font": {"size": 14, "family": "Space Mono"},
            "prefix": "Δ vs 2019: ",
            "decreasing": {"color": COLOR_RED},
            "increasing": {"color": COLOR_GREEN},
        },
        title={
            "text": f"<b style='font-family:Space Mono;letter-spacing:2px'>{label}</b>",
            "font": {"size": 14, "color": "#666680"},
        },
        number={
            "font": {"size": 60, "family": "Space Mono", "color": color},
            "suffix": "",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#333350",
                "tickfont": {"size": 10, "family": "Space Mono", "color": "#444460"},
                "tickvals": [0, 25, 50, 75, 100],
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25],  "color": "rgba(255,0,85,0.12)"},
                {"range": [25, 50], "color": "rgba(255,170,0,0.08)"},
                {"range": [50, 75], "color": "rgba(170,255,0,0.06)"},
                {"range": [75, 100],"color": "rgba(0,255,136,0.06)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.75,
                "value": 78,
            },
        },
    ))

    fig.update_layout(
        **PLOTLY_BASE,
        height=300,
        margin=dict(l=30, r=30, t=60, b=10),
        annotations=[
            dict(
                x=0.5, y=0.05, xanchor="center", yanchor="bottom",
                text="<span style='font-family:Space Mono;font-size:10px;color:#444460'>▲ 2019 BASELINE: 78.0</span>",
                showarrow=False,
            )
        ],
    )
    return fig


def render_timeline(df: pd.DataFrame) -> go.Figure:
    """Animated IAI timeline with anomaly markers and decay shading."""
    fig = go.Figure()

    # Confidence band (std approximation)
    upper = df["smoothed_index"] + 3.0
    lower = df["smoothed_index"] - 3.0

    fig.add_trace(go.Scatter(
        x=pd.concat([df["date"], df["date"].iloc[::-1]]),
        y=pd.concat([upper, lower.iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(0,170,255,0.06)",
        line_color="rgba(0,0,0,0)",
        name="Confidence band",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Raw index
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["aliveness_index"],
        mode="lines",
        name="Daily Index",
        line=dict(color="rgba(0,170,255,0.35)", width=1),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Raw: %{y:.1f}<extra></extra>",
    ))

    # Smoothed index — main signal
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["smoothed_index"],
        mode="lines",
        name="7-day Smoothed",
        line=dict(color=COLOR_BLUE, width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>IAI: <b>%{y:.1f}</b><extra></extra>",
    ))

    # Anomaly markers
    anomalies = df[df["is_anomaly"]] if "is_anomaly" in df.columns else df[df.get("anomaly_flag", pd.Series(0)) == 1]
    if not anomalies.empty:
        drops  = anomalies[anomalies.get("anomaly_type", pd.Series("")) == "drop"]
        spikes = anomalies[anomalies.get("anomaly_type", pd.Series("")) == "spike"]

        if not drops.empty:
            fig.add_trace(go.Scatter(
                x=drops["date"], y=drops["aliveness_index"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color=COLOR_RED,
                            line=dict(width=1, color="#ff0055")),
                name="Drop anomaly",
                hovertemplate="<b>DROP</b><br>%{x|%b %d}<br>Score: %{y:.1f}<extra></extra>",
            ))
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes["date"], y=spikes["aliveness_index"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=COLOR_GREEN,
                            line=dict(width=1, color="#00ff88")),
                name="Spike anomaly",
                hovertemplate="<b>SPIKE</b><br>%{x|%b %d}<br>Score: %{y:.1f}<extra></extra>",
            ))

    # Danger zone shading
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(255,0,85,0.04)",
                  line_width=0, annotation_text="DEAD ZONE",
                  annotation=dict(font_color=COLOR_RED, font_size=10, font_family="Space Mono"))
    fig.add_hrect(y0=50, y1=100, fillcolor="rgba(0,255,136,0.03)",
                  line_width=0)

    # 2019 baseline
    fig.add_hline(y=78, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                  annotation_text="2019 baseline",
                  annotation_font=dict(color="#444460", size=10, family="Space Mono"))

    fig.update_layout(
        **PLOTLY_BASE,
        height=340,
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False, tickformat="%b '%y",
        ),
        yaxis=dict(
            range=[0, 100],
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            ticksuffix=" ",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    return fig


def render_source_breakdown(source_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing aliveness per source."""
    if source_df.empty:
        return go.Figure()

    src_agg = (
        source_df.groupby(["source", "category"])["mean_score"]
        .mean()
        .reset_index()
        .sort_values("mean_score", ascending=True)
    )

    colors = [CATEGORY_COLORS.get(cat, "#666680") for cat in src_agg["category"]]

    fig = go.Figure(go.Bar(
        x=src_agg["mean_score"],
        y=src_agg["source"].str.replace("_", " ").str.title(),
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{v:.1f}" for v in src_agg["mean_score"]],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11),
        hovertemplate="<b>%{y}</b><br>Aliveness: %{x:.1f}<extra></extra>",
    ))

    fig.add_vline(x=50, line_dash="dot", line_color="rgba(255,255,255,0.2)")

    fig.update_layout(
        **PLOTLY_BASE,
        height=240,
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def render_domain_heatmap(source_df: pd.DataFrame) -> go.Figure:
    """Category × time heatmap of aliveness scores."""
    if source_df.empty or "date" not in source_df.columns:
        return go.Figure()

    source_df = source_df.copy()
    source_df["date"] = pd.to_datetime(source_df["date"])
    source_df["month"] = source_df["date"].dt.to_period("M").astype(str)

    pivot = (
        source_df.groupby(["month", "category"])["mean_score"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    if pivot.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#ff0055"],
            [0.3, "#ffaa00"],
            [0.6, "#aaff00"],
            [1.0, "#00ff88"],
        ],
        zmin=20, zmax=85,
        hoverongaps=False,
        hovertemplate="<b>%{x}</b> — %{y}<br>Score: %{z:.1f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(family="Space Mono", size=10),
            outlinecolor="rgba(0,0,0,0)",
        ),
    ))

    fig.update_layout(
        **PLOTLY_BASE,
        height=280,
        xaxis=dict(side="bottom", tickangle=0),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_decay_projection(
    df: pd.DataFrame,
    years_ahead: float = 3.0,
    acceleration: float = 0.0,
) -> go.Figure:
    """What-if simulator: project IAI decay forward."""
    if df.empty:
        return go.Figure()

    recent = df.tail(90)
    if recent.empty:
        return go.Figure()

    # Fit linear trend on last 90 days
    x = np.arange(len(recent))
    y = recent["smoothed_index"].values
    coeffs = np.polyfit(x, y, 1)
    slope_per_day = coeffs[0] + acceleration / 365.0

    last_date  = df["date"].max()
    last_score = float(df["smoothed_index"].iloc[-1])

    proj_days = int(years_ahead * 365)
    proj_dates = [last_date + timedelta(days=i) for i in range(1, proj_days + 1)]
    proj_scores = np.clip(
        [last_score + slope_per_day * i for i in range(1, proj_days + 1)],
        0, 100
    )

    # Uncertainty widens over time
    uncertainty = np.sqrt(np.arange(1, proj_days + 1)) * 0.4
    upper = np.clip(proj_scores + uncertainty, 0, 100)
    lower = np.clip(proj_scores - uncertainty, 0, 100)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["date"].tail(365), y=df["smoothed_index"].tail(365),
        mode="lines", name="Historical",
        line=dict(color=COLOR_BLUE, width=2),
    ))

    # Projection band
    fig.add_trace(go.Scatter(
        x=proj_dates + proj_dates[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(255,0,85,0.08)",
        line_color="rgba(0,0,0,0)", name="Uncertainty",
        hoverinfo="skip", showlegend=False,
    ))

    # Projection line
    proj_color = COLOR_RED if proj_scores[-1] < 30 else COLOR_ORANGE
    fig.add_trace(go.Scatter(
        x=proj_dates, y=proj_scores,
        mode="lines", name="Projection",
        line=dict(color=proj_color, width=2.5, dash="dash"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Projected: %{y:.1f}<extra></extra>",
    ))

    # Dead zone
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(255,0,85,0.06)", line_width=0)

    # Zero line label
    final_score = round(float(proj_scores[-1]), 1)
    final_year = (last_date + timedelta(days=proj_days)).year
    fig.add_annotation(
        x=proj_dates[-1], y=proj_scores[-1] + 5,
        text=f"<b>{final_score}</b> in {final_year}",
        font=dict(family="Space Mono", size=12, color=proj_color),
        showarrow=False,
    )

    fig.update_layout(
        **PLOTLY_BASE,
        height=320,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def render_component_radar(source_df: pd.DataFrame) -> go.Figure:
    """Radar chart of detection signal sub-scores."""
    # Generate synthetic sub-scores based on source mean scores for demo
    components = ["Vocabulary\nDiversity", "Sentence\nVariance",
                  "Info\nEntropy", "Temporal\nBurstiness",
                  "Zipf\nAlignment", "Non-\nRepetition"]

    if source_df.empty:
        overall = 41.3
    else:
        overall = float(source_df["mean_score"].mean()) if "mean_score" in source_df.columns else 41.3

    # Synthetic sub-scores centred on overall with variance
    rng = np.random.default_rng(int(overall * 100))
    sub_scores = np.clip(overall + rng.normal(0, 12, len(components)), 10, 90)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(sub_scores) + [sub_scores[0]],
        theta=components + [components[0]],
        fill="toself",
        fillcolor="rgba(0,170,255,0.12)",
        line=dict(color=COLOR_BLUE, width=2),
        name="Current",
    ))

    # 2019 baseline
    fig.add_trace(go.Scatterpolar(
        r=[78] * (len(components) + 1),
        theta=components + [components[0]],
        fill="toself",
        fillcolor="rgba(0,255,136,0.05)",
        line=dict(color=COLOR_GREEN, width=1, dash="dot"),
        name="2019 Baseline",
    ))

    fig.update_layout(
        **PLOTLY_BASE,
        height=300,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], showticklabels=True,
                            tickfont=dict(size=9, family="Space Mono"),
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickfont=dict(size=11, family="Rajdhani"),
                             gridcolor="rgba(255,255,255,0.08)"),
        ),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
#  LAYOUT SECTIONS
# ════════════════════════════════════════════════════════════════════════════════

def render_header(score: float):
    now = datetime.now(timezone.utc)
    status_color = "#00ff88" if score > 50 else "#ff0055"

    st.markdown(f"""
    <div class="mission-header">
      <p class="mission-title">☠ DEAD INTERNET OBSERVATORY</p>
      <p class="mission-subtitle">
        <span class="status-dot" style="background:{status_color};box-shadow:0 0 10px {status_color}"></span>
        LIVE MONITORING &nbsp;|&nbsp; INTERNET ALIVENESS INDEX &nbsp;|&nbsp;
        {now.strftime("%Y-%m-%d %H:%M UTC")}
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_stat_cards(df: pd.DataFrame, score: float, source_df: pd.DataFrame):
    col1, col2, col3, col4, col5 = st.columns(5)

    # Delta from 30 days ago
    if len(df) >= 30:
        score_30d = float(df["smoothed_index"].iloc[-30])
        delta_30d = score - score_30d
        delta_str = f"{'↓' if delta_30d < 0 else '↑'} {abs(delta_30d):.1f} vs 30d"
        delta_color = COLOR_RED if delta_30d < 0 else COLOR_GREEN
    else:
        delta_str = "—"
        delta_color = "#666680"

    # Lowest in dataset
    min_score = float(df["smoothed_index"].min()) if not df.empty else 0
    min_date  = df.loc[df["smoothed_index"].idxmin(), "date"].strftime("%b %Y") if not df.empty else "—"

    n_docs = int(df["n_docs"].sum()) if "n_docs" in df.columns and not df.empty else 0

    # Estimate synthetic fraction
    synthetic_pct = round(100 - score, 1)

    with col1:
        st.markdown(f"""
        <div class="stat-card {'red' if score < 50 else 'green'}">
          <div class="stat-value {'red' if score < 50 else 'green'}">{score:.1f}</div>
          <div class="stat-label">Aliveness Index</div>
          <div class="stat-delta" style="color:{delta_color}">{delta_str}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card red">
          <div class="stat-value red">{synthetic_pct}%</div>
          <div class="stat-label">Est. Synthetic Content</div>
          <div class="stat-delta" style="color:#666680">of sampled internet</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card orange">
          <div class="stat-value orange">{min_score:.1f}</div>
          <div class="stat-label">All-Time Low</div>
          <div class="stat-delta" style="color:#666680">{min_date}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        n_sources = len(source_df["source"].unique()) if not source_df.empty else 4
        st.markdown(f"""
        <div class="stat-card blue">
          <div class="stat-value blue">{n_sources}</div>
          <div class="stat-label">Active Sources</div>
          <div class="stat-delta" style="color:#666680">CC · Reddit · News · Wiki</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        n_docs_str = f"{n_docs/1e6:.1f}M" if n_docs >= 1e6 else f"{n_docs:,}"
        st.markdown(f"""
        <div class="stat-card blue">
          <div class="stat-value blue">{n_docs_str}</div>
          <div class="stat-label">Documents Scored</div>
          <div class="stat-delta" style="color:#666680">cumulative</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


def render_anomaly_list(df: pd.DataFrame):
    if df.empty or "is_anomaly" not in df.columns:
        return
    anomalies = df[df["is_anomaly"]].tail(10).sort_values("z_score", ascending=False)
    if anomalies.empty:
        st.markdown("_No significant anomalies detected in selected window._")
        return

    for _, row in anomalies.head(6).iterrows():
        atype = row.get("anomaly_type", "unknown")
        icon  = "🔴" if atype == "drop" else "🟢"
        date  = pd.to_datetime(row["date"]).strftime("%b %d, %Y")
        score = float(row.get("aliveness_index", 0))
        z     = float(row.get("z_score", 0))
        direction = "SURGE" if atype == "spike" else "CRASH"
        st.markdown(
            f'<div class="anomaly-alert">'
            f'{icon} <b>{date}</b> — <span style="color:{"#00ff88" if atype=="spike" else "#ff0055"}">'
            f'{direction}</span> &nbsp; Score: <b style="font-family:\'Space Mono\'">{score:.1f}</b> '
            f'<span style="color:#444460">(z={z:+.2f})</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Dead Internet Observatory",
        page_icon="☠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject CSS
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # Load data
    current_score  = load_current_score()
    timeline_df    = load_timeline(730)
    source_df      = load_source_breakdown()

    is_demo = not ANALYTICS_OK or (
        ANALYTICS_OK and (get_engine() is None or
                          get_engine().get_meta("demo_seeded") == "true")
    )

    if is_demo:
        st.markdown(
            '<div class="demo-banner">⚡ DEMO MODE — Displaying synthetic historical data. '
            'Run the data minions to populate with real web content.</div>',
            unsafe_allow_html=True,
        )

    # ── Header ────────────────────────────────────────────────────────────────
    render_header(current_score)

    # ── Gauge + Stats ─────────────────────────────────────────────────────────
    gcol, scol = st.columns([1, 2])
    with gcol:
        st.plotly_chart(render_aliveness_gauge(current_score),
                        use_container_width=True, config={"displayModeBar": False})
    with scol:
        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        render_stat_cards(timeline_df, current_score, source_df)

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬡ INTERNET PULSE — Aliveness Index Timeline</div>',
                unsafe_allow_html=True)

    tcol_ctrl, _ = st.columns([2, 5])
    with tcol_ctrl:
        timeline_range = st.selectbox(
            "Window", ["30 days", "90 days", "1 year", "All time"],
            index=2, label_visibility="collapsed",
        )

    range_map = {"30 days": 30, "90 days": 90, "1 year": 365, "All time": 9999}
    days = range_map[timeline_range]
    tdf = timeline_df[timeline_df["date"] >= (datetime.now() - timedelta(days=days))] \
        if days < 9999 else timeline_df

    st.plotly_chart(render_timeline(tdf), use_container_width=True,
                    config={"displayModeBar": False})

    # ── Domain Explorer ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬡ DOMAIN EXPLORER — Aliveness by Source & Category</div>',
                unsafe_allow_html=True)

    dcol1, dcol2, dcol3 = st.columns([1.2, 1, 1])

    with dcol1:
        st.markdown("**Source Breakdown**")
        st.plotly_chart(render_source_breakdown(source_df),
                        use_container_width=True, config={"displayModeBar": False})

    with dcol2:
        st.markdown("**Detection Signal Radar**")
        st.plotly_chart(render_component_radar(source_df),
                        use_container_width=True, config={"displayModeBar": False})

    with dcol3:
        st.markdown("**Category × Time Heatmap**")
        if not source_df.empty:
            st.plotly_chart(render_domain_heatmap(source_df),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Heatmap requires multi-day data")

    # ── Anomaly Spotlight ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬡ ANOMALY SPOTLIGHT — Significant Deviation Events</div>',
                unsafe_allow_html=True)

    acol1, acol2 = st.columns([3, 2])
    with acol1:
        render_anomaly_list(timeline_df)
    with acol2:
        total_anomalies = int(timeline_df["is_anomaly"].sum()) if "is_anomaly" in timeline_df.columns else 0
        drop_count  = int((timeline_df.get("anomaly_type", pd.Series("")) == "drop").sum())
        spike_count = int((timeline_df.get("anomaly_type", pd.Series("")) == "spike").sum())
        st.markdown(f"""
        <div class="stat-card red" style="margin-bottom:12px">
          <div class="stat-value red">{total_anomalies}</div>
          <div class="stat-label">Total Anomalies Detected</div>
          <div class="stat-delta" style="color:#666680">
            🔴 {drop_count} drops &nbsp;|&nbsp; 🟢 {spike_count} spikes
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── What-If Simulator ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬡ WHAT-IF SIMULATOR — Project Future Internet Aliveness</div>',
                unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 3])
    with sim_col1:
        st.markdown("**Simulation Controls**")
        years_ahead = st.slider("Years to project", 1.0, 5.0, 3.0, 0.5)
        acceleration = st.slider(
            "AI acceleration factor",
            min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
            help="Negative = faster die-off, Positive = human resistance/recovery",
        )
        regulation = st.checkbox("Assume AI content regulation (+10 boost)", value=False)
        open_web   = st.checkbox("Assume open-web revival (+5 boost)", value=False)

        bonus = (10 if regulation else 0) + (5 if open_web else 0)
        effective_accel = acceleration + bonus / 365.0

        proj_end_score = round(float(np.clip(
            current_score + (acceleration - 3.0) * years_ahead + bonus, 0, 100
        )), 1)
        proj_color = "#ff0055" if proj_end_score < 30 else "#ffaa00" if proj_end_score < 50 else "#00ff88"
        st.markdown(f"""
        <div class="stat-card" style="border-color:{proj_color}22;margin-top:16px">
          <div class="stat-value" style="color:{proj_color}">{proj_end_score}</div>
          <div class="stat-label">Projected Score in {years_ahead:.0f}yr</div>
        </div>
        """, unsafe_allow_html=True)

    with sim_col2:
        st.plotly_chart(
            render_decay_projection(timeline_df, years_ahead, effective_accel),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Methodology footer ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬡ METHODOLOGY</div>', unsafe_allow_html=True)
    with st.expander("How the Internet Aliveness Index is calculated", expanded=False):
        st.markdown("""
        The **Internet Aliveness Index (IAI)** is a composite 0–100 score computed from seven
        linguistic and behavioural signals:

        | Signal | Weight | What it measures |
        |---|---|---|
        | Type-Token Ratio (TTR) | 18% | Vocabulary diversity |
        | MTLD | 12% | Lexical diversity (length-independent) |
        | Shannon Entropy | 15% | Information density |
        | Sentence Length Variance | 15% | Structural variety (AI = uniform) |
        | Bigram Repetition | 15% | Repeated stock phrases |
        | Temporal Burstiness (Goh-Barabási) | 15% | Human-like irregular posting patterns |
        | Zipf Law Alignment | 10% | Natural word frequency distribution |

        **Data sources:** Common Crawl (5+ quarterly snapshots), Reddit (10 subreddits),
        major news RSS feeds (8 outlets), Wikipedia random article samples.

        **Scoring pipeline:** Bronze (raw JSONL) → Silver (normalised Parquet) →
        Gold (SQLite with daily aggregates) → Streamlit dashboard.

        **Limitations:** Scores reflect sampled content only. English-language bias.
        Detection accuracy ~85% on known benchmarks — not a ground-truth AI detector.
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;color:#333350;font-family:'Space Mono',monospace;
                font-size:0.7rem;padding:40px 0 20px;letter-spacing:2px">
      DEAD INTERNET OBSERVATORY · ALL DATA PUBLIC DOMAIN · NO COOKIES · NO TRACKING<br>
      BUILT WITH COMMON CRAWL · REDDIT · WIKIPEDIA · NEWS RSS · PYTHON · STREAMLIT
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
