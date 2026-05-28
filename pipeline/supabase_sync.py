"""
Supabase sync — lightweight doc registry and index persistence.

Only stores doc_ids (for dedup) and running count (for dashboard).
Full document text is never written to Supabase — storage stays minimal.

Tables used:
  doc_registry (doc_id TEXT PRIMARY KEY, scored_at TIMESTAMPTZ)
  meta         (key TEXT PRIMARY KEY, value TEXT)
  daily_index  — aggregated source scores
  composite_index — composite IAI per day

Run scripts/migrate_supabase_slim.py once to migrate from the old
full-documents schema to this lightweight schema.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger("pipeline.supabase_sync")

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_OK = True
except ImportError:
    PSYCOPG2_OK = False

BATCH_SIZE = 500
SOURCE_WEIGHTS = {
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


def _conn() -> Optional["psycopg2.connection"]:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return None
    if not PSYCOPG2_OK:
        logger.warning("[SUPABASE] psycopg2 not installed — skipping sync")
        return None
    try:
        return psycopg2.connect(url, connect_timeout=15)
    except Exception as exc:
        logger.warning(f"[SUPABASE] Connection failed: {exc}")
        return None


# ── Doc registry (dedup) ──────────────────────────────────────────────────────

def sync_doc_ids(scored_df: pd.DataFrame):
    """Write only doc_ids to doc_registry. No text, no features."""
    conn = _conn()
    if conn is None:
        return None

    now = datetime.now(timezone.utc).isoformat()
    ids = scored_df["doc_id"].dropna().unique().tolist()
    if not ids:
        return 0

    try:
        with conn:
            cur = conn.cursor()
            inserted = psycopg2.extras.execute_values(cur, """
                INSERT INTO doc_registry (doc_id, scored_at)
                VALUES %s
                ON CONFLICT (doc_id) DO NOTHING
                RETURNING doc_id
            """, [(doc_id, now) for doc_id in ids], page_size=BATCH_SIZE, fetch=True)
        inserted_ids = {row[0] for row in inserted or []}
        logger.info(f"[SUPABASE] ✓ {len(inserted_ids):,} new doc_ids registered")
        return inserted_ids
    except Exception as exc:
        logger.warning(f"[SUPABASE] sync_doc_ids error: {exc}")
        return None
    finally:
        conn.close()


def _increment_doc_count(n: int):
    """Increment the persistent running total in meta table."""
    conn = _conn()
    if conn is None:
        return
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO meta (key, value) VALUES ('total_scored_count', %s)
                ON CONFLICT (key) DO UPDATE
                    SET value = (CAST(meta.value AS BIGINT) + %s)::TEXT
            """, (str(n), n))
    except Exception as exc:
        logger.warning(f"[SUPABASE] _increment_doc_count error: {exc}")
    finally:
        conn.close()


def _refresh_doc_count():
    """Refresh the persistent total from the authoritative registry table."""
    conn = _conn()
    if conn is None:
        return None
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM doc_registry")
            count = int(cur.fetchone()[0])
            cur.execute("""
                INSERT INTO meta (key, value) VALUES ('total_scored_count', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (str(count),))
            return count
    except Exception as exc:
        logger.warning(f"[SUPABASE] _refresh_doc_count error: {exc}")
        return None
    finally:
        conn.close()


# ── Index tables ──────────────────────────────────────────────────────────────

def sync_index(engine):
    conn = _conn()
    if conn is None:
        return

    try:
        daily_df     = engine.get_source_breakdown()
        composite_df = engine.get_composite_timeline(days=99999)

        with conn:
            cur = conn.cursor()

            if not daily_df.empty:
                daily_cols = [
                    "date", "source", "category", "n_docs", "mean_score",
                    "median_score", "std_score", "pct_below_50",
                    "bot_fraction", "aliveness_index",
                ]
                for col in daily_cols:
                    if col not in daily_df.columns:
                        daily_df[col] = None
                rows = [tuple(r) for r in daily_df[daily_cols].itertuples(index=False, name=None)]
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO daily_index
                        (date, source, category, n_docs, mean_score, median_score,
                         std_score, pct_below_50, bot_fraction, aliveness_index)
                    VALUES %s
                    ON CONFLICT (date, source, category) DO UPDATE SET
                        mean_score      = EXCLUDED.mean_score,
                        median_score    = EXCLUDED.median_score,
                        n_docs          = EXCLUDED.n_docs,
                        std_score       = EXCLUDED.std_score,
                        pct_below_50    = EXCLUDED.pct_below_50,
                        bot_fraction    = EXCLUDED.bot_fraction,
                        aliveness_index = EXCLUDED.aliveness_index
                """, rows, page_size=BATCH_SIZE)
                logger.info(f"[SUPABASE] ✓ {len(rows):,} daily_index rows upserted")

            if not composite_df.empty:
                comp_cols = [
                    "date", "aliveness_index", "smoothed_index", "n_docs",
                    "anomaly_flag", "anomaly_reason",
                ]
                for col in comp_cols:
                    if col not in composite_df.columns:
                        composite_df[col] = None
                rows = [tuple(r) for r in composite_df[comp_cols].itertuples(index=False, name=None)]
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO composite_index
                        (date, aliveness_index, smoothed_index, n_docs,
                         anomaly_flag, anomaly_reason)
                    VALUES %s
                    ON CONFLICT (date) DO UPDATE SET
                        aliveness_index = EXCLUDED.aliveness_index,
                        smoothed_index  = EXCLUDED.smoothed_index,
                        n_docs          = EXCLUDED.n_docs,
                        anomaly_flag    = EXCLUDED.anomaly_flag,
                        anomaly_reason  = EXCLUDED.anomaly_reason
                """, rows, page_size=BATCH_SIZE)
                logger.info(f"[SUPABASE] ✓ {len(rows):,} composite_index rows upserted")

    except Exception as exc:
        logger.warning(f"[SUPABASE] sync_index error: {exc}")
    finally:
        conn.close()


def _daily_delta(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty or "aliveness_score" not in scored_df.columns:
        return pd.DataFrame()

    df = scored_df.copy()
    if "created_dt" in df.columns:
        df["date"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True).dt.date
    else:
        df["date"] = datetime.now(timezone.utc).date()
    df["date"] = df["date"].fillna(datetime.now(timezone.utc).date())

    rows = []
    for (date, source, category), group in df.groupby(["date", "source", "category"]):
        scores = group["aliveness_score"].dropna()
        if scores.empty:
            continue
        rows.append({
            "date": str(date),
            "source": source,
            "category": category,
            "n_docs": int(len(group)),
            "mean_score": round(float(scores.mean()), 3),
            "median_score": round(float(scores.median()), 3),
            "std_score": round(float(scores.std()), 3),
            "pct_below_50": round(float((scores < 50).mean() * 100), 2),
            "bot_fraction": 0.0,
            "aliveness_index": round(float(scores.mean()), 3),
        })
    return pd.DataFrame(rows)


def _recompute_composite_from_supabase(cur):
    weight_cases = " ".join(
        f"WHEN source = '{source}' THEN {weight}"
        for source, weight in SOURCE_WEIGHTS.items()
    )
    cur.execute(f"""
        WITH daily AS (
            SELECT
                date,
                SUM(mean_score * source_weight) / NULLIF(SUM(source_weight), 0) AS aliveness_index,
                SUM(n_docs)::INTEGER AS n_docs
            FROM (
                SELECT
                    date,
                    source,
                    mean_score,
                    COALESCE(n_docs, 0) AS n_docs,
                    CASE {weight_cases} ELSE 0.1 END AS source_weight
                FROM daily_index
                WHERE mean_score IS NOT NULL
            ) rows
            GROUP BY date
            HAVING SUM(n_docs) >= 10
        ),
        smoothed AS (
            SELECT
                date,
                aliveness_index,
                AVG(aliveness_index) OVER (
                    ORDER BY date
                    ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
                ) AS smoothed_index,
                n_docs
            FROM daily
        )
        INSERT INTO composite_index
            (date, aliveness_index, smoothed_index, n_docs, anomaly_flag, anomaly_reason)
        SELECT
            date,
            ROUND(aliveness_index::numeric, 3)::REAL,
            ROUND(smoothed_index::numeric, 3)::REAL,
            n_docs,
            0,
            ''
        FROM smoothed
        ON CONFLICT (date) DO UPDATE SET
            aliveness_index = EXCLUDED.aliveness_index,
            smoothed_index  = EXCLUDED.smoothed_index,
            n_docs          = EXCLUDED.n_docs,
            anomaly_flag    = EXCLUDED.anomaly_flag,
            anomaly_reason  = EXCLUDED.anomaly_reason
    """)


def sync_index_delta(scored_df: pd.DataFrame):
    daily_df = _daily_delta(scored_df)
    if daily_df.empty:
        return

    conn = _conn()
    if conn is None:
        return

    daily_cols = [
        "date", "source", "category", "n_docs", "mean_score",
        "median_score", "std_score", "pct_below_50",
        "bot_fraction", "aliveness_index",
    ]

    try:
        with conn:
            cur = conn.cursor()
            rows = [tuple(r) for r in daily_df[daily_cols].itertuples(index=False, name=None)]
            psycopg2.extras.execute_values(cur, """
                INSERT INTO daily_index
                    (date, source, category, n_docs, mean_score, median_score,
                     std_score, pct_below_50, bot_fraction, aliveness_index)
                VALUES %s
                ON CONFLICT (date, source, category) DO UPDATE SET
                    n_docs = COALESCE(daily_index.n_docs, 0) + EXCLUDED.n_docs,
                    mean_score = (
                        (daily_index.mean_score * COALESCE(daily_index.n_docs, 0))
                        + (EXCLUDED.mean_score * EXCLUDED.n_docs)
                    ) / NULLIF(COALESCE(daily_index.n_docs, 0) + EXCLUDED.n_docs, 0),
                    median_score = EXCLUDED.median_score,
                    std_score = EXCLUDED.std_score,
                    pct_below_50 = (
                        (daily_index.pct_below_50 * COALESCE(daily_index.n_docs, 0))
                        + (EXCLUDED.pct_below_50 * EXCLUDED.n_docs)
                    ) / NULLIF(COALESCE(daily_index.n_docs, 0) + EXCLUDED.n_docs, 0),
                    bot_fraction = (
                        (daily_index.bot_fraction * COALESCE(daily_index.n_docs, 0))
                        + (EXCLUDED.bot_fraction * EXCLUDED.n_docs)
                    ) / NULLIF(COALESCE(daily_index.n_docs, 0) + EXCLUDED.n_docs, 0),
                    aliveness_index = (
                        (daily_index.aliveness_index * COALESCE(daily_index.n_docs, 0))
                        + (EXCLUDED.aliveness_index * EXCLUDED.n_docs)
                    ) / NULLIF(COALESCE(daily_index.n_docs, 0) + EXCLUDED.n_docs, 0)
            """, rows, page_size=BATCH_SIZE)
            _recompute_composite_from_supabase(cur)
        logger.info(f"[SUPABASE] ✓ {len(rows):,} daily_index delta rows merged")
    except Exception as exc:
        logger.warning(f"[SUPABASE] sync_index_delta error: {exc}")
    finally:
        conn.close()


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_scored_doc_ids() -> set:
    """Return all doc_ids already scored. Used by pipeline for dedup."""
    conn = _conn()
    if conn is None:
        return set()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT doc_id FROM doc_registry")
            return {row[0] for row in cur.fetchall()}
    except Exception as exc:
        logger.warning(f"[SUPABASE] get_scored_doc_ids failed: {exc}")
        return set()
    finally:
        conn.close()


def get_existing_doc_ids(candidate_ids: Iterable[str]) -> set:
    """Return scored doc_ids matching the supplied candidates only."""
    ids = [doc_id for doc_id in dict.fromkeys(candidate_ids) if doc_id]
    if not ids:
        return set()

    conn = _conn()
    if conn is None:
        return set()

    existing = set()
    try:
        with conn:
            cur = conn.cursor()
            for i in range(0, len(ids), BATCH_SIZE):
                chunk = ids[i:i + BATCH_SIZE]
                cur.execute(
                    "SELECT doc_id FROM doc_registry WHERE doc_id = ANY(%s)",
                    (chunk,),
                )
                existing.update(row[0] for row in cur.fetchall())
        return existing
    except Exception as exc:
        logger.warning(f"[SUPABASE] get_existing_doc_ids failed: {exc}")
        return set()
    finally:
        conn.close()


def get_total_doc_count() -> int:
    """Return the all-time scored document count from the meta counter."""
    conn = _conn()
    if conn is None:
        return 0
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM meta WHERE key = 'total_scored_count'")
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception as exc:
        logger.warning(f"[SUPABASE] get_total_doc_count failed: {exc}")
        return 0
    finally:
        conn.close()


# ── Convenience ───────────────────────────────────────────────────────────────

def sync_all(scored_df: pd.DataFrame, engine):
    if not os.environ.get("DATABASE_URL"):
        logger.info("[SUPABASE] DATABASE_URL not set — skipping sync")
        return
    logger.info("[SUPABASE] Syncing to Supabase …")
    inserted_ids = sync_doc_ids(scored_df)
    if inserted_ids is None:
        refreshed_count = _refresh_doc_count()
        if refreshed_count is not None:
            logger.info(f"[SUPABASE] ✓ total_scored_count refreshed to {refreshed_count:,}")
    else:
        _increment_doc_count(len(inserted_ids))
        refreshed_count = _refresh_doc_count()
        if refreshed_count is not None:
            logger.info(f"[SUPABASE] ✓ total_scored_count refreshed to {refreshed_count:,}")
        if inserted_ids:
            sync_index_delta(scored_df[scored_df["doc_id"].isin(inserted_ids)].copy())
