"""
Dead Internet Observatory — Streamlit Dashboard
Research-grade interface tracking the Internet Aliveness Index.

An experimental public research instrument for sampled web text.
"""

import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from huggingface_hub import hf_hub_download
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

REQUIRED_DATABASE_SCHEMA = {
    "composite_index": {
        "date", "aliveness_index", "smoothed_index", "n_docs",
        "anomaly_flag", "anomaly_reason",
    },
    "daily_index": {"date", "source", "mean_score", "aliveness_index", "n_docs"},
    "meta": {"key", "value"},
}


@st.cache_resource()
def _database_path_state() -> dict:
    return {"last_valid_path": None, "lock": threading.Lock()}


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
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="observatory.db",
        revision=revision,
    )
    _validate_database(path)
    state = _database_path_state()
    with state["lock"]:
        state["last_valid_path"] = path
    return path


def _validate_database(path: str) -> None:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        result = conn.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise sqlite3.DatabaseError("database quick_check failed")
        for table, required_columns in REQUIRED_DATABASE_SCHEMA.items():
            columns = {
                row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
            }
            if not required_columns.issubset(columns):
                raise sqlite3.DatabaseError(
                    f"database is missing required schema for {table}"
                )


def _resolve_database_path() -> str:
    try:
        return _database_path(_dataset_revision())
    except Exception:
        state = _database_path_state()
        with state["lock"]:
            fallback = state["last_valid_path"]
        if fallback is None:
            raise
        _validate_database(fallback)
        return fallback


def _query(path: str, sql: str, params: tuple = ()) -> list:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


P = {
    "bg": "#f7f5ef", "ink": "#202e3d", "muted": "#56616c",
    "blue": "#24649a", "rust": "#9d4a36", "rule": "#d8d8cf",
}

PLOTLY_BASE = dict(
    template="plotly_white",
    paper_bgcolor=P["bg"], plot_bgcolor=P["bg"],
    font=dict(family="Arial, sans-serif", size=13, color=P["ink"]),
    hoverlabel=dict(bgcolor="#ffffff", font_size=13),
)

CSS = """
<style>
html { color-scheme: light; }
.stApp { background: #f7f5ef; color: #202e3d; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1180px; padding-top: 3rem; padding-bottom: 3rem;
}
.stApp h1, .stApp h2, .stApp h3 {
    font-family: Georgia, 'Times New Roman', serif; color: #202e3d;
    font-weight: 400; letter-spacing: -.025em;
}
.stApp h1 { font-size: clamp(2.5rem, 5vw, 4.5rem); line-height: 1.08; padding-top: .6rem; }
.stApp h2 { font-size: 2rem; }
.stApp p, .stApp li { line-height: 1.65; }
.stApp a { color: #24649a; text-underline-offset: .2em; }
.stApp a:focus-visible, .stApp button:focus-visible,
.stApp input:focus-visible, .stApp [tabindex]:focus-visible {
    outline: 3px solid #24649a !important; outline-offset: 3px;
}
.kicker { font-size: .75rem; font-weight: 600; letter-spacing: .14em;
    text-transform: uppercase; color: #56616c; border-top: 2px solid #202e3d; padding-top: 1rem; }
.deck { font-size: 1.2rem; max-width: 760px; color: #56616c; margin: .4rem 0 1.6rem; }
.measure { border-top: 1px solid #d8d8cf; padding: 1.3rem 0 1rem; }
.measure-label { font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.measure-value { font-family: Georgia, 'Times New Roman', serif; font-size: 5rem;
    color: #24649a; line-height: 1.15; font-variant-numeric: lining-nums tabular-nums; }
.measure-value small { font-family: Arial, sans-serif; font-size: 1.1rem; color: #56616c; }
.measure-note { color: #56616c; font-size: .9rem; margin-top: .45rem; }
.section { border-top: 1px solid #d8d8cf; margin-top: 2rem; padding-top: 1rem; }
.scope { border-left: 3px solid #9d4a36; padding-left: 1rem; margin: 1rem 0; max-width: 780px; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] { color: #56616c; }
[data-testid="stExpander"] { border-color: #d8d8cf; border-radius: 3px; }
@media (max-width: 640px) {
    .block-container, [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }
    .measure-value { font-size: 4rem; }
    .deck { font-size: 1.05rem; }
}
</style>
"""


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_timeline(path: str, days: int = None) -> pd.DataFrame:
    cutoff = ((datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
              if days is not None else "0001-01-01")
    data = _query(
        path,
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
        # Keep the existing minimum sample threshold for displayed observations.
        df = df[df["n_docs"] >= 20].reset_index(drop=True)
        return label_anomalies(df, "aliveness_index")
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sources(path: str) -> pd.DataFrame:
    return pd.DataFrame(_query(
        path,
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
def load_score(path: str) -> float:
    data = _query(
        path,
        "SELECT smoothed_index FROM composite_index ORDER BY date DESC LIMIT 1"
    )
    if not data:
        raise RuntimeError("composite_index returned no current score")
    return float(data[0]["smoothed_index"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_total_docs(path: str) -> int:
    data = _query(path, "SELECT value FROM meta WHERE key = 'total_scored_count'")
    if not data:
        raise RuntimeError("meta.total_scored_count returned no value")
    return int(data[0]["value"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_platform_trends(path: str) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()
    data = _query(
        path,
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


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_coverage(path: str) -> dict:
    rows = _query(path, """SELECT MIN(date) AS first_date, MAX(date) AS last_date,
                                   COUNT(DISTINCT source) AS source_count
                            FROM daily_index""")
    latest = _query(path, """SELECT date, n_docs FROM composite_index
                              ORDER BY date DESC LIMIT 1""")
    return {**rows[0], "latest": latest[0] if latest else None}


def chart_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # Separate runs so unobserved dates are never drawn as a continuous trend.
    runs = df.sort_values("date").copy()
    runs["run"] = runs["date"].diff().dt.days.gt(1).cumsum()
    for col, name, color, width in (
        ("aliveness_index", "Daily composite", P["muted"], 1),
        ("smoothed_index", "Published smoothed index", P["blue"], 2.5),
    ):
        for i, (_, group) in enumerate(runs.groupby("run")):
            fig.add_trace(go.Scatter(
                x=group["date"], y=group[col], mode="lines+markers",
                name=name, legendgroup=col, showlegend=i == 0,
                line=dict(color=color, width=width),
                marker=dict(size=3 if col == "aliveness_index" else 4),
                customdata=group["n_docs"], connectgaps=False,
                hovertemplate="%{x|%d %b %Y}<br>" + name +
                    ": %{y:.1f}<br>Documents: %{customdata:,.0f}<extra></extra>",
            ))
    flags = df[df["anomaly_flag"].fillna(0).astype(bool)]
    if not flags.empty:
        fig.add_trace(go.Scatter(
            x=flags["date"], y=flags["aliveness_index"], mode="markers",
            name="Published deviation flag", marker=dict(color=P["rust"], size=8, symbol="circle-open"),
            hovertemplate="%{x|%d %b %Y}<br>Flagged daily score: %{y:.1f}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_BASE, height=440, margin=dict(l=20, r=20, t=85, b=30),
        xaxis=dict(title="Observation date", showgrid=False, zeroline=False,
                   tickfont=dict(color=P["muted"])),
        yaxis=dict(title="Internet Aliveness Index · 0–100", range=[0, 100],
                   dtick=20, gridcolor=P["rule"], zeroline=False,
                   tickfont=dict(color=P["muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        hovermode="closest",
    )
    return fig


def chart_sources(df: pd.DataFrame) -> go.Figure:
    ordered = df.sort_values("mean_score").copy()
    fig = go.Figure(go.Scatter(
        x=ordered["mean_score"], y=ordered["source"].str.replace("_", " ").str.title(),
        mode="markers", marker=dict(color=P["blue"], size=10),
        customdata=ordered[["date", "n_docs"]],
        hovertemplate="%{y}<br>Mean score: %{x:.1f}<br>Observation: %{customdata[0]}"
                      "<br>Documents: %{customdata[1]:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=max(280, 32 * len(ordered) + 100),
        margin=dict(l=10, r=20, t=10, b=20),
        xaxis=dict(title="Mean document score · 0–100", range=[0, 100],
                   dtick=20, gridcolor=P["rule"], zeroline=False,
                   tickfont=dict(color=P["muted"])),
        yaxis=dict(showgrid=False, automargin=True, tickfont=dict(color=P["muted"])),
        showlegend=False,
    )
    return fig


def report_load_failure(label: str, exc: Exception, *, critical: bool = False):
    message = (
        f"{label} could not be loaded from the published dataset snapshot. "
        f"Data for that section is unavailable. {type(exc).__name__}: {exc}"
    )
    if critical:
        st.error(message)
    else:
        st.warning(message)


def section(title: str):
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.subheader(title)


def main():
    st.set_page_config(page_title="Dead Internet Observatory", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="kicker">Public web research / Open data</div>', unsafe_allow_html=True)
    st.title("Dead Internet Observatory")
    st.markdown('<p class="deck">An experimental record of linguistic and behavioural '
                'signals in sampled public web text. Explore the index, its coverage, '
                'and the assumptions behind it.</p>', unsafe_allow_html=True)

    score, total_docs, coverage = None, None, {}
    tl_df, src_df = pd.DataFrame(), pd.DataFrame()
    try:
        database_path = _resolve_database_path()
    except Exception as exc:
        report_load_failure("Published dataset snapshot", exc, critical=True)
        database_path = None

    if database_path is not None:
        for label, loader, key, critical in (
            ("Current Internet Aliveness Index", load_score, "score", True),
            ("Historical timeline", load_timeline, "timeline", False),
            ("Source scores", load_sources, "sources", False),
            ("Document count", load_total_docs, "count", False),
            ("Coverage metadata", load_coverage, "coverage", False),
        ):
            try:
                value = loader(database_path)
                if key == "score":
                    if not pd.notna(value) or not 0 <= value <= 100:
                        raise ValueError("current score is not a finite value within 0–100")
                    score = value
                elif key == "timeline":
                    tl_df = value
                elif key == "sources":
                    src_df = value
                elif key == "count":
                    total_docs = value
                else:
                    coverage = value
            except Exception as exc:
                report_load_failure(label, exc, critical=critical)

    left, right = st.columns([1, 2], gap="large")
    with left:
        value = f'{score:.1f}<small> / 100</small>' if score is not None else '<small>Unavailable</small>'
        latest = coverage.get("latest") or {}
        date = latest.get("date")
        st.markdown(
            '<div class="measure"><div class="measure-label">Internet Aliveness Index</div>'
            f'<div class="measure-value">{value}</div>'
            '<div class="measure-note">Latest published smoothed estimate</div></div>',
            unsafe_allow_html=True,
        )
        if date:
            st.caption(f"Observation date: {date}")
        st.caption("Higher scores indicate stronger human-like signals under the scoring rules.")
    with right:
        st.markdown('<div class="measure"></div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Documents scored · cumulative", f"{total_docs:,}" if total_docs is not None else "Unavailable")
        b.metric("Sources represented · all dates", coverage.get("source_count", "Unavailable"))
        if coverage.get("first_date") and coverage.get("last_date"):
            st.caption(f"Corpus dates: {coverage['first_date']} to {coverage['last_date']}")
        if latest.get("n_docs") is not None:
            st.caption(f"Latest composite sample: {latest['n_docs']:,} documents")
        st.markdown('<p class="scope">This score is an observational measure, not a calibrated '
                    'probability of AI authorship. It does not estimate the percentage of '
                    'the internet that is synthetic.</p>', unsafe_allow_html=True)
    st.caption("Provenance: published Hugging Face SQLite snapshot. Refresh checks are cached for five minutes; "
               "a previously validated snapshot may be used when the upstream service is unavailable. "
               "Observation dates describe the corpus, not the download time.")

    section("The historical record")
    st.write("Daily source-weighted composites and the smoothed series published by the pipeline. "
             "Changes can reflect shifts in the sample as well as changes in its text.")
    if tl_df.empty:
        st.info("No historical observations with at least 20 documents are available in the displayed period.")
    else:
        periods = {"Full available record": None, "Last 12 months": 365, "Last 90 days": 90}
        period = st.radio("Display period", list(periods), horizontal=True)
        shown = tl_df
        if periods[period] is not None:
            shown = tl_df[tl_df["date"] >= tl_df["date"].max() - timedelta(days=periods[period])]
        st.plotly_chart(chart_timeline(shown), width="stretch", config={"displaylogo": False})
        st.caption(f"{len(shown):,} observed dates · "
                   f"{shown['date'].min():%d %b %Y}–{shown['date'].max():%d %b %Y}. "
                   "Display excludes days with fewer than 20 documents. "
                   "Lines break across missing dates. Smoothing uses a centred rolling window of observations; "
                   "it may span gaps. Rust markers identify stored statistical deviation flags, not attributed events.")
        with st.expander("Inspect historical observations"):
            st.dataframe(shown[["date", "aliveness_index", "smoothed_index", "n_docs", "anomaly_flag"]]
                         .rename(columns={"date": "Date", "aliveness_index": "Daily index",
                                          "smoothed_index": "Smoothed index", "n_docs": "Documents",
                                          "anomaly_flag": "Deviation flag"}), hide_index=True, width="stretch")
            st.download_button("Download displayed series (CSV)",
                               shown[["date", "aliveness_index", "smoothed_index", "n_docs", "anomaly_flag"]]
                               .to_csv(index=False), "observatory-series.csv", "text/csv")

    section("Across sources")
    st.write("Latest available daily mean document score for each source, weighted by document count "
             "across categories. Observation dates differ; these are not simultaneous platform rankings.")
    if src_df.empty:
        st.info("Source observations are unavailable in this snapshot.")
    else:
        st.plotly_chart(chart_sources(src_df), width="stretch", config={"displaylogo": False})
        with st.expander("Inspect source observations"):
            st.dataframe(src_df[["source", "date", "mean_score", "n_docs"]]
                         .rename(columns={"source": "Source", "date": "Latest observation",
                                          "mean_score": "Mean document score", "n_docs": "Documents"}),
                         hide_index=True, width="stretch")
        st.caption("Source: daily_index in the same SQLite snapshot. Sample size and genre affect comparability. "
                   "A source represented here may have historical data without current collection.")

    section("Methodology & interpretation")
    method, limits = st.columns(2, gap="large")
    with method:
        st.markdown("### From text to an index")
        st.write("The pipeline normalises collected text, scores each document, aggregates daily source means, "
                 "and combines available sources using configured source weights. The published composite "
                 "uses a centred rolling mean; near the series edges the window is partial.")
        st.write("Seven classical signals form the default score. These are heuristic features rather than "
                 "independent tests of authorship. Temporal burstiness receives a neutral value when "
                 "timestamps are absent; the current dataframe scorer does not pass timestamp sequences.")
        st.dataframe(pd.DataFrame({"Classical signal": ["Type-token ratio", "Shannon entropy",
                     "Sentence length variance", "Bigram repetition", "Temporal burstiness",
                     "MTLD lexical diversity", "Zipf alignment"],
                     "Default weight": ["18%", "15%", "15%", "15%", "15%", "12%", "10%"]}),
                     hide_index=True, width="stretch")
        st.write("Optional local DistilGPT-2 perplexity scoring is enabled by ENABLE_PERPLEXITY=1. "
                 "It receives 15% of the score and scales the classical weights proportionally. "
                 "Common Crawl runs disable it to limit runtime. The aggregate snapshot does not identify "
                 "which scoring mode produced every observation.")
    with limits:
        st.markdown("### Scope & limitations")
        st.markdown("- **A collected sample, not a census.** Coverage depends on collector access, "
                    "API limits, search terms, archives, and processing caps.\n"
                    "- **Language is not authorship.** Genre, length, editing, and repeated phrasing can "
                    "change these signals in human and generated text alike. No calibrated error rates "
                    "or uncertainty intervals are supplied by this dashboard.\n"
                    "- **Historical samples differ.** Archive and crawl dates mix with recent platform "
                    "observations. Changing source composition and scoring modes limit comparisons over time.\n"
                    "- **Access gaps matter.** Twitter/X and Substack collectors are blocked; their absence "
                    "does not imply an absence of activity. Bluesky collection uses topic searches, not a full firehose.")
        st.write("Consult the collection and scoring code before interpreting a shift as a change "
                 "in synthetic content. Statistical flags identify unusual scores and do not explain their cause.")

    section("Data & code")
    st.markdown(
        "[Published dataset & SQLite snapshot](https://huggingface.co/datasets/jupiternull/dead-internet-observatory)"
        " · [Source repository](https://github.com/jupiternull/dead-internet-observatory)"
        " · [Scoring implementation](https://github.com/jupiternull/dead-internet-observatory/tree/master/detection)"
        " · [Index aggregation](https://github.com/jupiternull/dead-internet-observatory/blob/master/analytics/aliveness_index.py)"
    )
    st.caption("Source code is MIT licensed. Collected web content remains subject to its original terms; "
               "public availability does not make it public domain. Published text may include personal information. "
               "See the repository README for dataset removal and security reporting procedures.")


if __name__ == "__main__":
    main()
