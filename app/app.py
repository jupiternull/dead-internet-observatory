"""
Dead Internet Observatory — Streamlit Dashboard
Research-grade interface tracking the Internet Aliveness Index.

Aesthetic: observatory meets scholar's study — parchment, ink, brass, and starlight.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
CONFIG_PATH = str(ROOT / "config" / "config.yaml")

try:
    from analytics.anomaly_detector import label_anomalies
    ANALYTICS_OK = True
except ImportError:
    def label_anomalies(df, col):  # noqa: E302
        return df
    ANALYTICS_OK = False


_DEFAULT_SUPABASE_URL = "https://qwuabrmpudlqngfcxezh.supabase.co"
_DEFAULT_SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF3dWFicm1wdWRscW5nZmN4ZXpoIiwicm9sZSI6"
    "ImFub24iLCJpYXQiOjE3Nzc5NDYxMDIsImV4cCI6MjA5MzUyMjEwMn0"
    ".eTcAx9mcyAdOf4iBymIvpwK-E-Ayg-FKwphoDtzr6Ss"
)
_SUPABASE_URL = None
_SUPABASE_KEY = None
_SUPABASE_SESSION = None

CACHE_TTL_SECONDS = 21600


def _secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
        supabase = st.secrets.get("supabase", {})
        value = supabase.get(name) if hasattr(supabase, "get") else None
        return str(value) if value else None
    except Exception:
        return None


def _supabase_config() -> tuple[str, str]:
    global _SUPABASE_URL, _SUPABASE_KEY
    if _SUPABASE_URL and _SUPABASE_KEY:
        return _SUPABASE_URL, _SUPABASE_KEY

    _SUPABASE_URL = (
        _secret("SUPABASE_URL")
        or _secret("url")
        or os.getenv("SUPABASE_URL")
        or _DEFAULT_SUPABASE_URL
    )
    _SUPABASE_KEY = (
        _secret("SUPABASE_ANON_KEY")
        or _secret("SUPABASE_KEY")
        or _secret("anon_key")
        or _secret("key")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_KEY")
        or _DEFAULT_SUPABASE_KEY
    )
    return _SUPABASE_URL, _SUPABASE_KEY


def _supabase_session() -> requests.Session:
    global _SUPABASE_SESSION
    if _SUPABASE_SESSION:
        return _SUPABASE_SESSION

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _SUPABASE_SESSION = session
    return session


def _sb_get(table: str, params: dict = None) -> list:
    url, key = _supabase_config()
    r = _supabase_session().get(
        f"{url.rstrip('/')}/rest/v1/{table}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        params=params or {},
        timeout=(5, 20),
    )
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE & THEME
# ══════════════════════════════════════════════════════════════════════════════

P = {
    "bg":          "#F2EDE4",   # parchment
    "bg_dark":     "#E8E0D3",   # slightly darker parchment
    "card":        "#FAF7F2",   # near-white paper
    "border":      "#D4C9B8",   # warm border
    "border_soft": "#E4DDD0",   # very soft border

    "ink":         "#1C1812",   # near-black ink
    "ink_mid":     "#4A3F32",   # medium brown ink
    "ink_light":   "#8C7B68",   # faded ink

    "navy":        "#1E3A5F",   # observatory navy
    "navy_light":  "#2E5490",   # lighter navy
    "burgundy":    "#6B1F1F",   # study leather
    "forest":      "#1B4332",   # wizard's grove
    "gold":        "#9A7B2F",   # antique brass
    "gold_light":  "#C4A24D",   # polished brass
    "purple":      "#3D2B5E",   # mystic dusk
    "rust":        "#7A3B1E",   # aged copper

    "good":        "#1B4332",   # forest green for high scores
    "warn":        "#7A5C00",   # amber for mid scores
    "bad":         "#6B1F1F",   # burgundy for low scores
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Georgia, 'Times New Roman', serif", color=P["ink_mid"]),
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body {{
    background-color: {P["bg"]} !important;
    color: {P["ink"]} !important;
    font-family: 'Inter', sans-serif;
}}
/* Streamlit root containers (works across versions) */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
section[data-testid="stSidebar"],
div[data-testid="stVerticalBlock"] {{
    background-color: {P["bg"]} !important;
}}
/* Streamlit widget backgrounds */
[data-testid="stForm"],
[data-testid="stExpander"],
div.stSelectbox > div > div {{
    background-color: {P["card"]} !important;
    border-color: {P["border"]} !important;
}}
/* Text color propagation */
.stApp, .stApp p, .stApp span, .stApp label,
.stMarkdown p, .stMarkdown span {{
    color: {P["ink"]} !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container,
[data-testid="stMainBlockContainer"] {{
    padding-top: 2rem !important;
    max-width: 1280px;
}}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {P["bg_dark"]}; }}
::-webkit-scrollbar-thumb {{ background: {P["border"]}; border-radius: 3px; }}

/* ── Masthead ── */
.masthead {{
    border-bottom: 2px solid {P["ink"]};
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}}
.masthead-eyebrow {{
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {P["ink_light"]};
    margin-bottom: 0.3rem;
}}
.masthead-title {{
    font-family: 'Crimson Pro', serif;
    font-size: 2.6rem;
    font-weight: 300;
    color: {P["ink"]};
    letter-spacing: 0.03em;
    line-height: 1;
    margin: 0;
}}
.masthead-subtitle {{
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 1rem;
    color: {P["ink_light"]};
    margin-top: 0.3rem;
}}
.masthead-meta {{
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: {P["ink_light"]};
    letter-spacing: 0.05em;
    margin-top: 0.8rem;
    display: flex;
    gap: 1.5rem;
    align-items: center;
}}
.live-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {P["forest"]};
    margin-right: 5px;
    animation: blink 2.5s ease-in-out infinite;
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ── Section labels ── */
.section-rule {{
    border: none;
    border-top: 1px solid {P["border"]};
    margin: 2rem 0 1rem 0;
}}
.section-label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {P["ink_light"]};
    margin-bottom: 1rem;
}}

/* ── Stat cards ── */
.stat-grid {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
.stat-card {{
    background: {P["card"]};
    border: 1px solid {P["border_soft"]};
    border-radius: 4px;
    padding: 1rem 1.2rem;
    flex: 1;
    position: relative;
}}
.stat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 4px 4px 0 0;
}}
.stat-card.navy::before  {{ background: {P["navy"]}; }}
.stat-card.forest::before {{ background: {P["forest"]}; }}
.stat-card.burgundy::before {{ background: {P["burgundy"]}; }}
.stat-card.gold::before {{ background: {P["gold"]}; }}
.stat-card.purple::before {{ background: {P["purple"]}; }}
.stat-number {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    line-height: 1;
    color: {P["ink"]};
}}
.stat-number.navy  {{ color: {P["navy"]}; }}
.stat-number.forest {{ color: {P["forest"]}; }}
.stat-number.burgundy {{ color: {P["burgundy"]}; }}
.stat-number.gold {{ color: {P["gold"]}; }}
.stat-number.purple {{ color: {P["purple"]}; }}
.stat-label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {P["ink_light"]};
    margin-top: 0.35rem;
}}
.stat-note {{
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 0.8rem;
    color: {P["ink_light"]};
    margin-top: 0.2rem;
}}

/* ── Finding callout ── */
.finding {{
    background: {P["card"]};
    border: 1px solid {P["border_soft"]};
    border-left: 3px solid {P["navy"]};
    border-radius: 0 4px 4px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-family: 'Crimson Pro', serif;
    font-size: 0.95rem;
    color: {P["ink_mid"]};
}}
.finding .date {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {P["ink_light"]};
    margin-bottom: 0.2rem;
}}
.finding.drop {{ border-left-color: {P["burgundy"]}; }}
.finding.spike {{ border-left-color: {P["forest"]}; }}

/* ── Source table ── */
.source-row {{
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid {P["border_soft"]};
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
}}
.source-name {{
    width: 120px;
    font-weight: 500;
    color: {P["ink"]};
    text-transform: capitalize;
}}
.source-bar-wrap {{
    flex: 1;
    background: {P["bg_dark"]};
    border-radius: 2px;
    height: 6px;
    margin: 0 0.75rem;
    overflow: hidden;
}}
.source-bar {{ height: 100%; border-radius: 2px; }}
.source-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: {P["ink_mid"]};
    width: 40px;
    text-align: right;
}}

/* ── Health bar ── */
.health-bar-section {{ margin: 0.25rem 0 1rem 0; }}
.hbar-row {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid {P["border_soft"]};
}}
.hbar-label {{
    width: 130px;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {P["ink_light"]};
    flex-shrink: 0;
}}
.hbar-segments {{
    display: flex;
    flex: 1;
    gap: 2px;
    height: 16px;
}}
.hbar-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    white-space: nowrap;
    flex-shrink: 0;
}}

/* ── Methodology box ── */
.method-box {{
    font-family: 'Crimson Pro', serif;
    font-size: 0.95rem;
    color: {P["ink_mid"]};
    line-height: 1.65;
}}

/* ── Score pill ── */
.score-pill {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin-left: 6px;
}}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_timeline(days: int = 3650) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    data = _sb_get("composite_index", {
        "date": f"gte.{cutoff}",
        "select": "date,aliveness_index,smoothed_index,n_docs,anomaly_flag,anomaly_reason",
        "order": "date.asc",
    })
    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        # Drop days with too few docs — CC backfill years that haven't
        # accumulated enough data yet (1-5 docs) produce meaningless scores
        # and create noise between the solid 2013-2014 and 2025-2026 clusters
        df = df[df["n_docs"] >= 20].reset_index(drop=True)
        return label_anomalies(df, "aliveness_index")
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sources() -> pd.DataFrame:
    return pd.DataFrame(_sb_get("latest_source_scores", {
        "select": "source,mean_score,n_docs,date",
    }))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_score() -> float:
    data = _sb_get("composite_index", {"select": "smoothed_index", "order": "date.desc", "limit": "1"})
    if not data:
        raise RuntimeError("composite_index returned no current score")
    return float(data[0]["smoothed_index"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_total_docs() -> int:
    data = _sb_get("meta", {"key": "eq.total_scored_count", "select": "value"})
    if not data:
        raise RuntimeError("meta.total_scored_count returned no value")
    return int(data[0]["value"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_platform_trends() -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()
    data = _sb_get("daily_index", {
        "source": "in.(reddit,hackernews,bluesky,youtube,fourchan,steam)",
        "date": f"gte.{cutoff}",
        "select": "date,source,aliveness_index",
        "order": "date.asc",
    })
    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df




# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_gauge(score: float) -> go.Figure:
    if score >= 65:
        color, label = P["forest"],   "Predominantly Human"
    elif score >= 50:
        color, label = P["gold"],     "Mixed — Significant Synthetic Presence"
    elif score >= 35:
        color, label = P["rust"],     "Synthetic Majority Detected"
    else:
        color, label = P["burgundy"], "Critical — Internet Largely Synthetic"

    ink_light = P["ink_light"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title=dict(
            text=f"<span style='font-family:Crimson Pro,serif;font-size:13px;color:{ink_light};font-style:italic'>{label}</span>",
            font=dict(size=13),
        ),
        number=dict(
            font=dict(size=64, family="JetBrains Mono, monospace", color=color),
            suffix="",
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=P["border"],
                tickfont=dict(size=9, family="Inter, sans-serif", color=P["ink_light"]),
                tickvals=[0, 25, 50, 75, 100],
            ),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                {"range": [0, 25],   "color": "rgba(107,31,31,0.07)"},
                {"range": [25, 50],  "color": "rgba(122,59,30,0.05)"},
                {"range": [50, 75],  "color": "rgba(154,123,47,0.04)"},
                {"range": [75, 100], "color": "rgba(27,67,50,0.05)"},
            ],
            threshold=dict(
                line=dict(color=P["ink_light"], width=1.5),
                thickness=0.7,
                value=68,
            ),
        ),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        annotations=[dict(
            x=0.5, y=0.08, xanchor="center",
            text=f"<span style='font-family:Inter,sans-serif;font-size:9px;color:{ink_light};letter-spacing:0.1em'>ESTIMATED 2019 BASELINE  ▲  68.0</span>",
            showarrow=False,
        )],
    )
    return fig


def chart_platform_trends(trends_df: pd.DataFrame) -> go.Figure:
    platform_colors = {
        "reddit":      P["gold"],
        "hackernews":  P["rust"],
        "bluesky":     "#0085FF",
        "youtube":     "#CC0000",
        "fourchan":    "#6B8E23",
        "steam":       "#1B2838",
    }

    fig = go.Figure()
    for source, grp in trends_df.groupby("source"):
        grp = grp.sort_values("date").copy()
        grp["smoothed"] = grp["aliveness_index"].rolling(7, min_periods=1).mean()
        color = platform_colors.get(source, P["ink_light"])
        fig.add_trace(go.Scatter(
            x=grp["date"],
            y=grp["smoothed"],
            mode="lines",
            name=source.replace("_", " ").title(),
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{source.replace('_',' ').title()}</b><br>"
                "%{x|%d %b %Y}<br>"
                "Aliveness: <b>%{y:.1f}</b><extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_BASE,
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor=P["border_soft"],
            zeroline=False,
            tickformat="%b '%y",
            tickfont=dict(size=10, family="Inter"),
        ),
        yaxis=dict(
            title=dict(
                text="Aliveness Score",
                font=dict(size=10, family="Inter", color=P["ink_light"]),
            ),
            range=[0, 100],
            showgrid=True,
            gridcolor=P["border_soft"],
            zeroline=False,
            tickfont=dict(size=10, family="Inter"),
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=10, family="Inter"),
            bgcolor="rgba(242,237,228,0.85)",
            bordercolor=P["border_soft"],
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def render_overall_healthbar(score: float) -> str:
    if score >= 65:
        color, status = P["forest"],   "Predominantly Human"
    elif score >= 50:
        color, status = P["gold"],     "Mixed — Significant Synthetic Presence"
    elif score >= 35:
        color, status = P["rust"],     "Synthetic Majority Detected"
    else:
        color, status = P["burgundy"], "Critical — Internet Largely Synthetic"

    n_segs = 20
    filled = int(score / 100 * n_segs)
    frac   = (score / 100 * n_segs) - filled
    segments = ""
    for i in range(n_segs):
        br_left  = "4px" if i == 0 else "2px"
        br_right = "4px" if i == n_segs - 1 else "2px"
        if i < filled:
            seg_style = f"background:{color};opacity:1;"
        elif i == filled and frac > 0:
            opacity = round(0.12 + frac * 0.88, 3)
            seg_style = f"background:{color};opacity:{opacity};"
        else:
            seg_style = f"background:{color};opacity:0.12;"
        segments += (
            f'<div style="flex:1;height:28px;border-radius:{br_left} {br_right} '
            f'{br_right} {br_left};{seg_style};transition:opacity 0.3s"></div>'
        )

    synth = round(100 - score, 1)
    return f"""
<div style="margin:1.5rem 0 1rem">
  <div style="display:flex;align-items:baseline;gap:0.75rem;margin-bottom:0.5rem">
    <span style="font-family:JetBrains Mono,monospace;font-size:2.8rem;font-weight:600;color:{color};line-height:1">{score:.1f}</span>
    <span style="font-family:Inter,sans-serif;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:{P['ink_light']}">{status}</span>
  </div>
  <div style="display:flex;gap:3px">{segments}</div>
  <div style="display:flex;justify-content:space-between;margin-top:0.35rem;font-family:Inter,sans-serif;font-size:0.65rem;color:{P['ink_light']};letter-spacing:0.08em">
    <span>0 — Dead Internet</span>
    <span style="color:{P['ink_light']}">▲ 68.0 est. 2019 baseline</span>
    <span>100 — Fully Human</span>
  </div>
  <div style="margin-top:0.4rem;font-family:Inter,sans-serif;font-size:0.7rem;color:{P['ink_light']}">{synth}% of sampled content estimated synthetic</div>
</div>
"""


def render_masthead(live: bool = True):
    now = datetime.now(timezone.utc)
    status = "Live" if live else "Data unavailable"
    dot_style = "live-dot" if live else ""
    st.markdown(f"""
    <div class="masthead">
      <div class="masthead-eyebrow">Observational Research  ·  Internet Linguistics  ·  Open Data</div>
      <div class="masthead-title">Dead Internet Observatory</div>
      <div class="masthead-subtitle">Tracking the synthetic displacement of human-authored content on the public web</div>
      <div class="masthead-meta">
        <span><span class="{dot_style}"></span>{status}</span>
        <span>Internet Aliveness Index</span>
        <span>{now.strftime("%-d %B %Y, %H:%M UTC")}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats(df: pd.DataFrame, score: float, src_df: pd.DataFrame, total_docs: Optional[int] = None):
    delta_30 = ""
    delta_color = P["ink_light"]
    if len(df) >= 2:
        past = df[df["date"] <= df["date"].max() - timedelta(days=30)]
        if not past.empty:
            d = score - float(past["smoothed_index"].iloc[-1])
            delta_30 = f"{'↑' if d > 0 else '↓'} {abs(d):.1f} vs 30 days prior"
            delta_color = P["forest"] if d > 0 else P["burgundy"]

    n_docs_str = "Unavailable"
    if total_docs is not None:
        n_docs_str = f"{total_docs/1e6:.2f}M" if total_docs >= 1e6 else f"{total_docs:,}"
    synth = round(100 - score, 1)
    min_score = f"{float(df['smoothed_index'].min()):.1f}" if not df.empty else "Unavailable"

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card navy">
        <div class="stat-number navy">{score:.1f}</div>
        <div class="stat-label">Aliveness Index</div>
        <div class="stat-note" style="color:{delta_color}">{delta_30 or "Current composite score"}</div>
      </div>
      <div class="stat-card burgundy">
        <div class="stat-number burgundy">{synth}%</div>
        <div class="stat-label">Est. Synthetic Content</div>
        <div class="stat-note">of sampled web corpus</div>
      </div>
      <div class="stat-card gold">
        <div class="stat-number gold">{min_score}</div>
        <div class="stat-label">Observed Low</div>
        <div class="stat-note">lowest recorded index value</div>
      </div>
      <div class="stat-card purple">
        <div class="stat-number purple">{n_docs_str}</div>
        <div class="stat-label">Documents Scored</div>
        <div class="stat-note">cumulative corpus</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_source_rows(src_df: pd.DataFrame):
    if src_df.empty:
        return
    agg = src_df.groupby("source")["mean_score"].mean().sort_values(ascending=False)
    bar_colors = {
        "news":         P["navy"],
        "wayback":      P["forest"],
        "wikipedia":    P["purple"],
        "reddit":       P["gold"],
        "hackernews":   P["rust"],
        "common_crawl": P["burgundy"],
        "bluesky":      "#0085FF",
        "fourchan":     "#6B8E23",
        "steam":        "#1B2838",
        "youtube":      "#CC0000",
        "linkedin":     "#0A66C2",
        "twitter":      "#1DA1F2",
        "stackoverflow": "#F48024",
        "mastodon":      "#6364FF",
        "substack":      "#FF6719",
        "github":        "#6E40C9",
    }
    rows_html = ""
    for src, score in agg.items():
        pct = score
        color = bar_colors.get(src, P["ink_light"])
        rows_html += f"""
        <div class="source-row">
          <div class="source-name">{src.replace("_"," ").title()}</div>
          <div class="source-bar-wrap">
            <div class="source-bar" style="width:{pct}%;background:{color}"></div>
          </div>
          <div class="source-score">{score:.1f}</div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)


_HBAR_COLORS = [
    "#7A0000", "#A83200", "#CC5500", "#D97D00", "#C4A200",
    "#8DAA00", "#4A9A30", "#1E8855", "#008866", "#00A090",
]


def render_health_bar(label: str, score: float) -> str:
    clamped  = max(0.0, min(100.0, score))
    filled   = int(clamped / 10)          # fully-lit segments
    frac     = (clamped % 10) / 10        # 0.0–1.0 partial fill for next segment
    segments = ""
    for i, color in enumerate(_HBAR_COLORS):
        br_left  = "3px" if i == 0 else "1px"
        br_right = "3px" if i == 9 else "1px"
        if i < filled:
            seg_style = f"background:{color};opacity:1;"
        elif i == filled and frac > 0:
            # partial segment: fade between dim and full using the fractional value
            opacity = round(0.12 + frac * 0.88, 3)
            seg_style = f"background:{color};opacity:{opacity};"
        else:
            seg_style = f"background:{color};opacity:0.12;"
        segments += (
            f'<div style="flex:1;height:16px;border-radius:{br_left} {br_right} '
            f'{br_right} {br_left};{seg_style}"></div>'
        )

    if score >= 65:
        score_color = P["forest"]
    elif score >= 45:
        score_color = P["gold"]
    else:
        score_color = P["burgundy"]

    synth = round(100 - score, 1)
    score_text = f'{score:.1f}&nbsp;&nbsp;·&nbsp;&nbsp;{synth}% synthetic'

    return (
        f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0;'
        f'border-bottom:1px solid {P["border_soft"]}">'
        f'<div style="width:130px;font-family:Inter,sans-serif;font-size:0.68rem;'
        f'font-weight:600;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{P["ink_light"]};flex-shrink:0">{label}</div>'
        f'<div style="display:flex;flex:1;gap:2px;height:16px">{segments}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
        f'white-space:nowrap;flex-shrink:0;color:{score_color}">{score_text}</div>'
        f'</div>'
    )


def render_platform_health_bars(src_df: pd.DataFrame) -> str:
    if src_df.empty:
        return ""
    agg = src_df.groupby("source")["mean_score"].mean().sort_values(ascending=False)
    bars = "".join(
        render_health_bar(src.replace("_", " ").title(), float(score))
        for src, score in agg.items()
        if float(score) > 0
    )
    return f'<div class="health-bar-section">{bars}</div>'


def report_load_failure(label: str, exc: Exception, *, critical: bool = False):
    message = (
        f"{label} could not be loaded from Supabase after bounded retries. "
        f"Live data for that section is unavailable. {type(exc).__name__}: {exc}"
    )
    if critical:
        st.error(message)
    else:
        st.warning(message)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Dead Internet Observatory",
        page_icon="☠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    score = None
    tl_df = pd.DataFrame()
    src_df = pd.DataFrame()
    total_docs = None

    try:
        score = load_score()
    except Exception as exc:
        report_load_failure("Current Internet Aliveness Index", exc, critical=True)

    try:
        tl_df = load_timeline()
    except Exception as exc:
        report_load_failure("Historical timeline", exc)

    try:
        src_df = load_sources()
    except Exception as exc:
        report_load_failure("Source aliveness scores", exc)

    try:
        total_docs = load_total_docs()
    except Exception as exc:
        report_load_failure("Document count", exc)

    # ── Masthead ──────────────────────────────────────────────────────────────
    render_masthead(live=score is not None)

    # ── Lede ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <p style="font-family:'Crimson Pro',serif;font-style:italic;font-size:1.05rem;
              color:{P['ink_mid']};max-width:640px;margin:0.25rem 0 1.5rem;line-height:1.65">
      A growing share of what you read online was not written by a person.
      This index deploys bots across the internet to continuously measure the shift.
    </p>
    """, unsafe_allow_html=True)

    if score is None:
        st.error(
            "Current score unavailable. Refusing to render a zero placeholder as live observatory data."
        )
    else:
        # ── Overall Healthbar ─────────────────────────────────────────────────
        st.markdown(render_overall_healthbar(score), unsafe_allow_html=True)

        # ── Stats ─────────────────────────────────────────────────────────────
        render_stats(tl_df, score, src_df, total_docs)

    # ── Platform Health Bars ─────────────────────────────────────────────────
    st.markdown('<hr class="section-rule"><div class="section-label">Platform Aliveness Health</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-family:'Crimson Pro',serif;font-style:italic;font-size:0.9rem;
              color:{P['ink_light']};margin:0.1rem 0 0.75rem;line-height:1.6">
      The decline is not uniform. Some platforms retain stronger human signal than others.
      The bars below show each source's current aliveness score relative to the 0–100 index.
    </p>
    """, unsafe_allow_html=True)
    platform_bars = render_platform_health_bars(src_df)
    if platform_bars:
        st.markdown(platform_bars, unsafe_allow_html=True)
    else:
        st.warning("Source score data is unavailable; platform health bars are hidden.")

    # ── Methodology ───────────────────────────────────────────────────────────
    st.markdown('<hr class="section-rule"><div class="section-label">Methodology</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <p style="font-family:'Crimson Pro',serif;font-style:italic;font-size:0.9rem;
              color:{P['ink_light']};margin:0.1rem 0 0.75rem;line-height:1.6">
      How the Internet Aliveness Index is computed.
    </p>
    <div class="method-box">
    The <b>Internet Aliveness Index (IAI)</b> is a composite 0–100 score derived from seven
    statistical signals computed on every harvested document. No external AI models are called;
    all detection is performed with classical NLP and information theory.<br><br>

    <b>Signal weights:</b><br>
    Type-Token Ratio (18%) · Shannon Entropy (15%) · Sentence Length Variance (15%) ·
    Bigram Repetition (15%) · Temporal Burstiness (Goh-Barabási, 15%) ·
    MTLD Lexical Diversity (12%) · Zipf Law Alignment (10%)<br><br>

    <b>Sources:</b> Common Crawl WET extracts · Reddit public JSON API · RSS news feeds ·
    Wikipedia API · Hacker News via Algolia · Internet Archive Wayback Machine ·
    Bluesky public firehose · 4chan public API · Steam review API ·
    YouTube public data · LinkedIn public posts · Twitter/X public data<br><br>

    <b>Data architecture:</b> Bronze (raw JSONL via GitHub Artifacts) → Silver (normalised Parquet) →
    Gold (scored Parquet + SQLite index). All source code and data at
    <a href="https://github.com/jupiternull/dead-internet-observatory" style="color:{P['navy']}">
    github.com/jupiternull/dead-internet-observatory</a>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 0 1.5rem;border-top:1px solid {P['border_soft']};
                margin-top:2rem;font-family:Inter,sans-serif;font-size:0.65rem;
                color:{P['ink_light']};letter-spacing:0.12em;text-transform:uppercase">
      Dead Internet Observatory &nbsp;·&nbsp; MIT License &nbsp;·&nbsp;
      All source data public domain &nbsp;·&nbsp; No cookies &nbsp;·&nbsp; No tracking
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
