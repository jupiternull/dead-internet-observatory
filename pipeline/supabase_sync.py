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
from typing import Optional

import pandas as pd

logger = logging.getLogger("pipeline.supabase_sync")

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_OK = True
except ImportError:
    PSYCOPG2_OK = False

BATCH_SIZE = 500


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
        return

    now = datetime.now(timezone.utc).isoformat()
    ids = scored_df["doc_id"].dropna().unique().tolist()
    if not ids:
        return

    try:
        with conn:
            cur = conn.cursor()
            psycopg2.extras.execute_values(cur, """
                INSERT INTO doc_registry (doc_id, scored_at)
                VALUES %s
                ON CONFLICT (doc_id) DO NOTHING
            """, [(doc_id, now) for doc_id in ids], page_size=BATCH_SIZE)
        logger.info(f"[SUPABASE] ✓ {len(ids):,} doc_ids registered")
    except Exception as exc:
        logger.warning(f"[SUPABASE] sync_doc_ids error: {exc}")
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
    sync_doc_ids(scored_df)
    _increment_doc_count(len(scored_df))
    sync_index(engine)
