"""
Index & Analytics Agent — Internet Aliveness Index (IAI) computation.

The IAI is a composite 0–100 score measuring how much of the sampled
internet appears authentically human versus synthetic/automated.

100 = completely human
 50 = roughly half synthetic
  0 = dead internet (all bots/AI)

The index is:
  1. Computed per document from the detection pipeline
  2. Aggregated per source/category per day
  3. Smoothed with a configurable rolling window
  4. Stored in SQLite for the Streamlit app to query

Historical reconstruction uses multiple crawl dates to show the
decay from ~78 (pre-ChatGPT baseline) to current levels.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ── Database schema ───────────────────────────────────────────────────────────

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS daily_index (
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    n_docs INTEGER,
    mean_score REAL,
    median_score REAL,
    std_score REAL,
    pct_below_50 REAL,
    bot_fraction REAL,
    aliveness_index REAL,
    PRIMARY KEY (date, source, category)
);

CREATE TABLE IF NOT EXISTS composite_index (
    date TEXT PRIMARY KEY,
    aliveness_index REAL,
    smoothed_index REAL,
    n_docs INTEGER,
    anomaly_flag INTEGER DEFAULT 0,
    anomaly_reason TEXT
);

CREATE TABLE IF NOT EXISTS domain_scores (
    date TEXT NOT NULL,
    domain TEXT NOT NULL,
    category TEXT,
    mean_score REAL,
    n_docs INTEGER,
    PRIMARY KEY (date, domain)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class AlivenessIndexEngine:

    DEFAULT_SOURCE_WEIGHTS = {
        "common_crawl":  0.30,
        "reddit":        0.15,
        "news":          0.10,
        "wikipedia":     0.10,
        "wayback":       0.08,
        "hackernews":    0.06,
        "bluesky":       0.05,
        "youtube":       0.05,
        "fourchan":      0.04,
        "steam":         0.03,
        "mastodon":      0.03,
        "stackoverflow": 0.03,
        "linkedin":      0.02,
        "github":        0.02,
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)

        self.db_path = self.config["storage"].get("db_path", "./data/observatory.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        cfg = self.config["analytics"]
        self.smooth_window: int = cfg.get("index_smoothing_window", 7)
        self.anomaly_z: float = cfg.get("anomaly_zscore_threshold", 2.5)

        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(CREATE_TABLES)
            conn.execute(
                "INSERT OR IGNORE INTO meta VALUES ('created_at', ?)",
                (datetime.now(timezone.utc).isoformat(),)
            )

    # ── Index computation ─────────────────────────────────────────────────────

    def ingest_scored_df(self, df: pd.DataFrame, replace_affected_dates: bool = False):
        """
        Take a scored silver DataFrame (output of ai_content_detector.score_dataframe)
        and compute daily index values, inserting/replacing rows in SQLite.
        """
        if df.empty or "aliveness_score" not in df.columns:
            print("[INDEX] Nothing to ingest — empty or missing aliveness_score")
            return

        df = df.copy()

        # Coerce created_dt to date
        if "created_dt" in df.columns:
            df["date"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True).dt.date
        else:
            df["date"] = datetime.now(timezone.utc).date()

        df["date"] = df["date"].fillna(datetime.now(timezone.utc).date())

        affected_dates = sorted({str(date) for date in df["date"].dropna().unique()})

        with self._conn() as conn:
            if replace_affected_dates and affected_dates:
                placeholders = ",".join("?" for _ in affected_dates)
                conn.execute(f"DELETE FROM daily_index WHERE date IN ({placeholders})", affected_dates)
                conn.execute(f"DELETE FROM domain_scores WHERE date IN ({placeholders})", affected_dates)
                conn.execute(f"DELETE FROM composite_index WHERE date IN ({placeholders})", affected_dates)

            # Per-source-category daily aggregates
            grp = df.groupby(["date", "source", "category"])
            for (date, source, category), group in grp:
                scores = group["aliveness_score"].dropna()
                if scores.empty:
                    continue
                conn.execute(
                    """INSERT INTO daily_index VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(date, source, category) DO UPDATE SET
                           n_docs          = excluded.n_docs,
                           mean_score      = excluded.mean_score,
                           median_score    = excluded.median_score,
                           std_score       = excluded.std_score,
                           pct_below_50    = excluded.pct_below_50,
                           bot_fraction    = excluded.bot_fraction,
                           aliveness_index = excluded.aliveness_index""",
                    (
                        str(date), source, category,
                        len(group),
                        round(float(scores.mean()), 3),
                        round(float(scores.median()), 3),
                        round(float(scores.std()), 3),
                        round(float((scores < 50).mean() * 100), 2),
                        0.0,   # bot_fraction placeholder
                        round(float(scores.mean()), 3),   # = aliveness_index
                    )
                )

            # Domain-level scores
            if "domain" in df.columns:
                dgrp = df.groupby(["date", "domain"])
                for (date, domain), group in dgrp:
                    scores = group["aliveness_score"].dropna()
                    if scores.empty:
                        continue
                    cat = group["category"].iloc[0] if "category" in group.columns else ""
                    conn.execute(
                        """INSERT INTO domain_scores VALUES (?,?,?,?,?)
                           ON CONFLICT(date, domain) DO UPDATE SET
                               category   = excluded.category,
                               mean_score = excluded.mean_score,
                               n_docs     = excluded.n_docs""",
                        (str(date), domain, cat,
                         round(float(scores.mean()), 3), len(group))
                    )

        # Recompute composite index
        self._recompute_composite()
        print(f"[INDEX] Ingested {len(df):,} docs, updated composite index")

    def ingest_scored_delta(self, df: pd.DataFrame):
        if df.empty or "aliveness_score" not in df.columns:
            print("[INDEX] Nothing to ingest — empty or missing aliveness_score")
            return

        df = df.copy()
        if "created_dt" in df.columns:
            df["date"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True).dt.date
        else:
            df["date"] = datetime.now(timezone.utc).date()
        df["date"] = df["date"].fillna(datetime.now(timezone.utc).date())

        with self._conn() as conn:
            for (date, source, category), group in df.groupby(["date", "source", "category"]):
                scores = group["aliveness_score"].dropna()
                if scores.empty:
                    continue
                values = (
                    str(date), source, category, len(group),
                    round(float(scores.mean()), 3),
                    round(float(scores.median()), 3),
                    round(float(scores.std()), 3),
                    round(float((scores < 50).mean() * 100), 2),
                    0.0,
                    round(float(scores.mean()), 3),
                )
                conn.execute(
                    """INSERT INTO daily_index VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(date, source, category) DO UPDATE SET
                           mean_score = (
                               daily_index.mean_score * daily_index.n_docs
                               + excluded.mean_score * excluded.n_docs
                           ) / (daily_index.n_docs + excluded.n_docs),
                           median_score = excluded.median_score,
                           std_score = excluded.std_score,
                           pct_below_50 = (
                               daily_index.pct_below_50 * daily_index.n_docs
                               + excluded.pct_below_50 * excluded.n_docs
                           ) / (daily_index.n_docs + excluded.n_docs),
                           bot_fraction = (
                               daily_index.bot_fraction * daily_index.n_docs
                               + excluded.bot_fraction * excluded.n_docs
                           ) / (daily_index.n_docs + excluded.n_docs),
                           aliveness_index = (
                               daily_index.aliveness_index * daily_index.n_docs
                               + excluded.aliveness_index * excluded.n_docs
                           ) / (daily_index.n_docs + excluded.n_docs),
                           n_docs = daily_index.n_docs + excluded.n_docs""",
                    values,
                )

            if "domain" in df.columns:
                for (date, domain), group in df.groupby(["date", "domain"]):
                    scores = group["aliveness_score"].dropna()
                    if scores.empty:
                        continue
                    category = group["category"].iloc[0] if "category" in group.columns else ""
                    conn.execute(
                        """INSERT INTO domain_scores VALUES (?,?,?,?,?)
                           ON CONFLICT(date, domain) DO UPDATE SET
                               mean_score = (
                                   domain_scores.mean_score * domain_scores.n_docs
                                   + excluded.mean_score * excluded.n_docs
                               ) / (domain_scores.n_docs + excluded.n_docs),
                               n_docs = domain_scores.n_docs + excluded.n_docs,
                               category = excluded.category""",
                        (str(date), domain, category, round(float(scores.mean()), 3), len(group)),
                    )

        self._recompute_composite()
        print(f"[INDEX] Merged {len(df):,} docs into the composite index")

    def _recompute_composite(self):
        """Recompute the composite daily index from daily_index table."""
        with self._conn() as conn:
            rows = conn.execute("SELECT date, source, mean_score, n_docs FROM daily_index").fetchall()

        if not rows:
            with self._conn() as conn:
                conn.execute("DELETE FROM composite_index")
            return

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])

        def weighted_mean(group: pd.DataFrame) -> float:
            total_weight = 0.0
            weighted_sum = 0.0
            for _, row in group.iterrows():
                w = self.DEFAULT_SOURCE_WEIGHTS.get(row["source"], 0.1)
                weighted_sum += row["mean_score"] * w
                total_weight += w
            return weighted_sum / total_weight if total_weight > 0 else 50.0

        n_docs_series = df.groupby("date")["n_docs"].sum()

        composite = (
            df.groupby("date")
            .apply(weighted_mean)
            .reset_index()
            .rename(columns={0: "aliveness_index"})
            .sort_values("date")
        )

        composite["n_docs"] = composite["date"].map(n_docs_series).fillna(0).apply(lambda x: int(float(x)))

        # Drop days with too few docs — single-doc outliers distort the composite
        composite = composite[composite["n_docs"] >= 10].reset_index(drop=True)

        # Rolling smoothing
        composite["smoothed_index"] = (
            composite["aliveness_index"]
            .rolling(window=self.smooth_window, min_periods=1, center=True)
            .mean()
            .round(3)
        )

        # Anomaly detection via rolling z-score
        rolling_mean = composite["aliveness_index"].rolling(30, min_periods=7).mean()
        rolling_std  = composite["aliveness_index"].rolling(30, min_periods=7).std()
        z_scores = (composite["aliveness_index"] - rolling_mean) / (rolling_std + 1e-9)
        composite["anomaly_flag"] = (z_scores.abs() > self.anomaly_z).astype(int)
        composite["anomaly_reason"] = np.where(
            z_scores > self.anomaly_z, "spike",
            np.where(z_scores < -self.anomaly_z, "drop", "")
        )

        with self._conn() as conn:
            conn.execute("DELETE FROM composite_index")
            for _, row in composite.iterrows():
                conn.execute(
                    """INSERT INTO composite_index VALUES (?,?,?,?,?,?)
                       ON CONFLICT(date) DO UPDATE SET
                           aliveness_index = excluded.aliveness_index,
                           smoothed_index  = excluded.smoothed_index,
                           n_docs          = excluded.n_docs,
                           anomaly_flag    = excluded.anomaly_flag,
                           anomaly_reason  = excluded.anomaly_reason""",
                    (
                        str(row["date"].date()),
                        round(float(row["aliveness_index"]), 3),
                        round(float(row["smoothed_index"]), 3),
                        int(row["n_docs"]) if not pd.isna(row["n_docs"]) else 0,
                        int(row["anomaly_flag"]),
                        row["anomaly_reason"],
                    )
                )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_composite_timeline(self, days: int = 365) -> pd.DataFrame:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM composite_index WHERE date >= ? ORDER BY date",
                (str(cutoff),)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_source_breakdown(self, date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM daily_index"
        params: tuple = ()
        if date:
            query += " WHERE date = ?"
            params = (date,)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_top_domains(self, n: int = 50, date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM domain_scores"
        params: tuple = ()
        if date:
            query += " WHERE date = ?"
            params = (date,)
        query += " ORDER BY n_docs DESC LIMIT ?"
        params = params + (n,)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_current_score(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT smoothed_index FROM composite_index ORDER BY date DESC LIMIT 1"
            ).fetchone()
        return float(row["smoothed_index"]) if row else self.config["app"]["current_aliveness"]

    def get_total_docs(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(n_docs), 0) FROM daily_index"
            ).fetchone()
        return int(row[0])

    def get_meta(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str):
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))


# ── Demo data seeder (for Streamlit demo mode) ────────────────────────────────

def seed_demo_data(db_path: str = "./data/observatory.db", config_path: str = "config/config.yaml"):
    """
    Generate two years of synthetic daily IAI data for demo mode.
    Simulates a realistic decline curve from ~78 in Jan 2024 to ~41 in May 2026.
    """
    print("[DEMO] Seeding synthetic historical index data …")
    engine = AlivenessIndexEngine(config_path)

    rng = np.random.default_rng(42)

    # Build date range
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end   = datetime(2026, 5, 3, tzinfo=timezone.utc)
    n_days = (end - start).days + 1
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # Sigmoid-ish decay from 78 → 41 with noise
    x = np.linspace(0, 1, n_days)
    trend = 78.0 - 37.0 * (1 / (1 + np.exp(-8 * (x - 0.55))))
    noise = rng.normal(0, 1.5, n_days)
    # Periodic weekly cycles (humans post less on weekends → slightly higher on weekdays)
    weekly = 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    base_scores = trend + noise + weekly

    # Add a few notable events (anomaly spikes / drops)
    events = {
        180: ("GPT-4o release surge", -8),
        320: ("Llama 3 flood", -6),
        450: ("Reddit API lockout (recovery)", +5),
        550: ("Election bot storm", -10),
        700: ("News AI disclaimer push", +3),
    }

    sources = ["common_crawl", "reddit", "news", "wikipedia"]
    categories = ["web", "social", "news", "wiki"]
    source_offsets = {"common_crawl": -2, "reddit": 0, "news": +2, "wikipedia": +4}

    with engine._conn() as conn:
        for i, date in enumerate(dates):
            # Apply event effects
            event_delta = 0
            for day_offset, (reason, delta) in events.items():
                if abs(i - day_offset) < 7:
                    event_delta += delta * max(0, 1 - abs(i - day_offset) / 7)

            composite = float(np.clip(base_scores[i] + event_delta, 10, 95))
            smoothed = composite  # will be overwritten by _recompute_composite

            date_str = date.strftime("%Y-%m-%d")

            # Per-source rows
            for src, cat in zip(sources, categories):
                src_score = float(np.clip(
                    composite + source_offsets[src] + rng.normal(0, 2),
                    10, 95
                ))
                conn.execute(
                    "INSERT OR REPLACE INTO daily_index VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (date_str, src, cat,
                     int(rng.integers(500, 3000)),   # n_docs — must be plain int
                     round(src_score, 3),
                     round(src_score - float(rng.uniform(0, 3)), 3),
                     round(float(rng.uniform(8, 20)), 3),
                     round(float(rng.uniform(20, 60)), 2),
                     round(float(rng.uniform(0.05, 0.4)), 4),
                     round(src_score, 3),
                     )
                )

        print(f"[DEMO] Inserted {n_days * len(sources):,} source-daily rows")

    # Recompute composite (applies smoothing + anomaly detection properly)
    engine._recompute_composite()

    # Mark as demo seeded
    engine.set_meta("demo_seeded", "true")
    engine.set_meta("last_seeded_at", datetime.now(timezone.utc).isoformat())
    print("[DEMO] ✓ Synthetic data ready")


if __name__ == "__main__":
    import sys
    if "--seed-demo" in sys.argv:
        seed_demo_data()
    else:
        print("Run with --seed-demo to populate the database with synthetic data")
