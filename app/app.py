"""
Dead Internet Observatory — Streamlit Dashboard
Research-grade interface tracking the Internet Aliveness Index.

Aesthetic: dark terminal observatory — slate, cyan signal, coral warning.
"""

import hashlib
import sqlite3
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


CACHE_TTL_SECONDS = 300
DATASET_REPO = "jupiternull/dead-internet-observatory"
DATASET_API_URL = f"https://huggingface.co/api/datasets/{DATASET_REPO}"


def _http_session() -> requests.Session:
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
    return session


def _dataset_revision() -> str:
    response = _http_session().get(DATASET_API_URL, timeout=(5, 20))
    response.raise_for_status()
    return response.json()["sha"]


@st.cache_resource(ttl=CACHE_TTL_SECONDS)
def _database_path(revision: str) -> str:
    url = (
        f"https://huggingface.co/datasets/{DATASET_REPO}"
        f"/resolve/{revision}/observatory.db"
    )
    response = _http_session().get(url, timeout=(5, 60))
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()[:16]
    path = Path("/tmp") / f"dead-internet-observatory-{revision[:12]}-{digest}.db"
    if not path.exists():
        path.write_bytes(response.content)
    return str(path)


def _query(sql: str, params: tuple = ()) -> list:
    path = _database_path(_dataset_revision())
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE & THEME
# ══════════════════════════════════════════════════════════════════════════════

P = {
    "bg":          "#111111",
    "bg_dark":     "#1e2228",
    "card":        "#111111",
    "card_alt":    "#282c34",
    "border":      "#355a66",
    "border_soft": "#2a4650",

    "ink":         "#9cdef2",
    "ink_mid":     "#b9e6f4",
    "ink_light":   "#6b8a94",

    "navy":        "#9cdef2",
    "navy_light":  "#b9e6f4",
    "burgundy":    "#e06c75",
    "forest":      "#50fa7b",
    "gold":        "#f0ad4e",
    "gold_light":  "#ffd28a",
    "purple":      "#b48ead",
    "rust":        "#f0989e",

    "good":        "#50fa7b",
    "warn":        "#f0ad4e",
    "bad":         "#e06c75",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Fira Code', 'JetBrains Mono', monospace", color=P["ink_mid"]),
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body {{
    background-color: {P["bg"]} !important;
    color: {P["ink"]} !important;
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
}}
html {{
    color-scheme: dark;
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
.stApp {{
    background:
        radial-gradient(1100px 520px at 82% -10%, rgba(224,108,117,0.08), transparent 60%),
        radial-gradient(900px 520px at 0% 0%, rgba(53,90,102,0.18), transparent 55%),
        radial-gradient(circle, rgba(156,222,242,0.055) 1px, transparent 1.4px),
        {P["bg"]} !important;
    background-size: cover, cover, 24px 24px, auto !important;
}}
/* Streamlit widget backgrounds */
[data-testid="stForm"],
[data-testid="stExpander"],
div.stSelectbox > div > div,
[data-testid="stSelectbox"] > div,
[data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background-color: {P["card"]} !important;
    border-color: {P["border"]} !important;
    color: {P["ink"]} !important;
}}
/* Text color propagation */
.stApp, .stApp p, .stApp label,
.stMarkdown p, .stMarkdown li {{
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
::-webkit-scrollbar-thumb {{ background: {P["navy"]}; border-radius: 3px; }}
a, a:visited {{
    color: {P["navy_light"]} !important;
    text-decoration-color: rgba(156, 222, 242, 0.45) !important;
}}
[data-testid="stAlert"] {{
    background: rgba(17, 17, 17, 0.92) !important;
    border: 1px solid {P["border"]} !important;
    color: {P["ink"]} !important;
}}

/* ── Masthead ── */
.masthead {{
    background: {P["card"]};
    border: 1px solid {P["border"]};
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 2rem;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
}}
.masthead-eyebrow {{
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {P["burgundy"]};
    margin-bottom: 0.3rem;
    text-shadow: 0 2px 20px rgba(0, 0, 0, 0.45);
}}
.masthead-title {{
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 2.75rem;
    font-weight: 700;
    color: {P["ink"]};
    letter-spacing: -0.01em;
    line-height: 1;
    margin: 0;
    text-shadow: 0 2px 20px rgba(0, 0, 0, 0.45);
}}
.masthead-subtitle {{
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-style: normal;
    font-size: 1rem;
    color: {P["ink_light"]};
    margin-top: 0.3rem;
}}
.masthead-meta {{
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
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
    box-shadow: 0 0 8px {P["forest"]};
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
    box-shadow: 0 -1px 0 rgba(156, 222, 242, 0.08);
}}
.section-label {{
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {P["burgundy"]};
    margin-bottom: 1rem;
}}

/* ── Stat cards ── */
.stat-grid {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
.stat-card {{
    background: {P["card"]};
    border: 1px solid {P["border"]};
    border-radius: 8px;
    padding: 1.15rem 1.25rem;
    flex: 1;
    position: relative;
    box-shadow: none;
}}
.stat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 8px 8px 0 0;
}}
.stat-card.navy::before  {{ background: {P["navy"]}; }}
.stat-card.burgundy::before {{ background: {P["burgundy"]}; }}
.stat-card.gold::before {{ background: {P["burgundy"]}; }}
.stat-card.purple::before {{ background: {P["burgundy"]}; }}
.stat-number {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.05rem;
    font-weight: 500;
    line-height: 1;
    color: {P["ink"]};
    text-shadow: 0 2px 20px rgba(0, 0, 0, 0.45);
}}
.stat-number.navy  {{ color: {P["navy"]}; }}
.stat-number.burgundy {{ color: {P["burgundy"]}; }}
.stat-number.gold {{ color: {P["navy"]}; }}
.stat-number.purple {{ color: {P["navy"]}; }}
.stat-label {{
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {P["ink_light"]};
    margin-top: 0.35rem;
}}
.stat-note {{
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-style: normal;
    font-size: 0.8rem;
    color: {P["ink_light"]};
    margin-top: 0.2rem;
}}

/* ── Finding callout ── */
.finding {{
    background: {P["card"]};
    border: 1px solid {P["border_soft"]};
    border-left: 3px solid {P["navy"]};
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
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
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
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
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
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
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: {P["ink_mid"]};
    line-height: 1.65;
    background: {P["card"]};
    border: 1px solid {P["border"]};
    padding: 1rem 1.1rem;
    border-radius: 8px;
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

.terminal-panel {{
    background: {P["card"]};
    border: 1px solid {P["border"]};
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    margin: 0.75rem 0 1.35rem;
}}

.term-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1.4rem -1.6rem 1.35rem;
    padding: 0.45rem 0.7rem 0.45rem 0.95rem;
    background: {P["bg_dark"]};
    border-bottom: 1px solid {P["border"]};
    border-radius: 8px 8px 0 0;
    color: {P["ink_light"]};
    font-size: 0.72rem;
}}

.term-controls {{
    display: flex;
    gap: 0.65rem;
    color: {P["ink_light"]};
}}

.term-prompt {{
    color: {P["forest"]};
}}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_timeline(days: int = 3650) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    data = _query(
        """SELECT date, aliveness_index, smoothed_index, n_docs,
                  anomaly_flag, anomaly_reason
           FROM composite_index
           WHERE date >= ?
           ORDER BY date ASC""",
        (cutoff,),
    )
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
    return pd.DataFrame(_query(
        """SELECT source, date,
                  CASE WHEN SUM(n_docs) > 0
                       THEN SUM(mean_score * n_docs) / SUM(n_docs)
                       ELSE AVG(mean_score)
                  END AS mean_score,
                  SUM(n_docs) AS n_docs
           FROM daily_index
           WHERE mean_score IS NOT NULL
             AND date = (
                 SELECT MAX(latest.date)
                 FROM daily_index latest
                 WHERE latest.source = daily_index.source
                   AND latest.mean_score IS NOT NULL
             )
           GROUP BY source, date
           ORDER BY source"""
    ))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_score() -> float:
    data = _query(
        "SELECT smoothed_index FROM composite_index ORDER BY date DESC LIMIT 1"
    )
    if not data:
        raise RuntimeError("composite_index returned no current score")
    return float(data[0]["smoothed_index"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_total_docs() -> int:
    data = _query("SELECT value FROM meta WHERE key = 'total_scored_count'")
    if not data:
        raise RuntimeError("meta.total_scored_count returned no value")
    return int(data[0]["value"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_platform_trends() -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()
    data = _query(
        """SELECT date, source, aliveness_index
           FROM daily_index
           WHERE source IN ('reddit', 'hackernews', 'bluesky', 'youtube', 'fourchan', 'steam')
             AND date >= ?
           ORDER BY date ASC""",
        (cutoff,),
    )
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
            text=f"<span style='font-family:Fira Code,JetBrains Mono,monospace;font-size:13px;color:{ink_light}'>{label}</span>",
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
                tickfont=dict(size=9, family="Fira Code, JetBrains Mono, monospace", color=P["ink_light"]),
                tickvals=[0, 25, 50, 75, 100],
            ),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                {"range": [0, 25],   "color": "rgba(224,108,117,0.15)"},
                {"range": [25, 50],  "color": "rgba(240,152,158,0.12)"},
                {"range": [50, 75],  "color": "rgba(240,173,78,0.12)"},
                {"range": [75, 100], "color": "rgba(80,250,123,0.12)"},
            ],
            threshold=dict(
                line=dict(color=P["navy_light"], width=1.5),
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
            text=f"<span style='font-family:Fira Code,JetBrains Mono,monospace;font-size:9px;color:{ink_light};letter-spacing:0.1em'>ESTIMATED 2019 BASELINE  ▲  68.0</span>",
            showarrow=False,
        )],
    )
    return fig


def chart_platform_trends(trends_df: pd.DataFrame) -> go.Figure:
    platform_colors = {
        "reddit":      P["gold"],
        "hackernews":  P["rust"],
        "bluesky":     P["navy_light"],
        "youtube":     P["burgundy"],
        "fourchan":    P["forest"],
        "steam":       "#5fb6cc",
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
            tickfont=dict(size=10, family="Fira Code", color=P["ink_light"]),
        ),
        yaxis=dict(
            title=dict(
                text="Aliveness Score",
                font=dict(size=10, family="Fira Code", color=P["ink_light"]),
            ),
            range=[0, 100],
            showgrid=True,
            gridcolor=P["border_soft"],
            zeroline=False,
            tickfont=dict(size=10, family="Fira Code", color=P["ink_light"]),
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=10, family="Fira Code", color=P["ink_mid"]),
            bgcolor="rgba(17,17,17,0.88)",
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
<div class="terminal-panel">
  <div style="display:flex;align-items:baseline;gap:0.75rem;margin-bottom:0.5rem">
    <span style="font-family:JetBrains Mono,monospace;font-size:2.8rem;font-weight:600;color:{color};line-height:1">{score:.1f}</span>
    <span style="font-family:Fira Code,JetBrains Mono,monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:{P['ink_light']}">{status}</span>
  </div>
  <div style="display:flex;gap:3px">{segments}</div>
  <div style="display:flex;justify-content:space-between;margin-top:0.35rem;font-family:Fira Code,JetBrains Mono,monospace;font-size:0.65rem;color:{P['ink_light']};letter-spacing:0.08em">
    <span>0 — Dead Internet</span>
    <span style="color:{P['ink_light']}">▲ 68.0 est. 2019 baseline</span>
    <span>100 — Fully Human</span>
  </div>
  <div style="margin-top:0.4rem;font-family:Fira Code,JetBrains Mono,monospace;font-size:0.7rem;color:{P['ink_light']}">{synth}% of sampled content estimated synthetic</div>
</div>
"""


def render_masthead(live: bool = True):
    now = datetime.now(timezone.utc)
    status = "Live" if live else "Data unavailable"
    dot_style = "live-dot" if live else ""
    st.markdown(f"""
    <div class="masthead">
      <div class="term-bar">
        <span>user@observatory: ~</span>
        <span class="term-controls">− ×</span>
      </div>
      <div class="masthead-eyebrow">Observational Research  ·  Internet Linguistics  ·  Open Data</div>
      <div class="masthead-title"><span class="term-prompt">&gt;</span> Dead Internet Observatory</div>
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
        "bluesky":      P["navy_light"],
        "fourchan":     P["forest"],
        "steam":        "#5fb6cc",
        "youtube":      P["burgundy"],
        "linkedin":     P["navy"],
        "twitter":      P["navy_light"],
        "stackoverflow": P["rust"],
        "mastodon":      P["purple"],
        "substack":      P["gold_light"],
        "github":        P["purple"],
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
    "#5a252b", "#8f3f47", "#e06c75", "#f0989e", "#f0ad4e",
    "#d4c56a", "#8fdc72", "#50fa7b", "#5fb6cc", "#9cdef2",
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
        f'<div style="width:130px;font-family:Fira Code,JetBrains Mono,monospace;font-size:0.68rem;'
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
    return f'<div class="terminal-panel health-bar-section">{bars}</div>'


def report_load_failure(label: str, exc: Exception, *, critical: bool = False):
    message = (
        f"{label} could not be loaded from the published dataset snapshot. "
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
    <p style="font-family:'Fira Code','JetBrains Mono',monospace;font-style:normal;font-size:1.05rem;
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
    <p style="font-family:'Fira Code','JetBrains Mono',monospace;font-style:normal;font-size:0.9rem;
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
    <p style="font-family:'Fira Code','JetBrains Mono',monospace;font-style:normal;font-size:0.9rem;
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
                margin-top:2rem;font-family:Fira Code,JetBrains Mono,monospace;font-size:0.65rem;
                color:{P['ink_light']};letter-spacing:0.12em;text-transform:uppercase">
      Dead Internet Observatory &nbsp;·&nbsp; MIT License &nbsp;·&nbsp;
      All source data public domain &nbsp;·&nbsp; No cookies &nbsp;·&nbsp; No tracking
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
