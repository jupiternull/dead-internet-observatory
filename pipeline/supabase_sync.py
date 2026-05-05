"""
Supabase sync — persists scored documents and index data to Postgres.

Runs automatically at the end of the gold pipeline step.
Gracefully no-ops if DATABASE_URL is not set, so local runs are unaffected.

Set DATABASE_URL in your environment or GitHub Actions secrets:
  postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
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
TEXT_MAX   = 10_000   # chars — keeps storage manageable on free tier


def _conn() -> Optional["psycopg2.connection"]:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return None
    if not PSYCOPG2_OK:
        logger.warning("[SUPABASE] psycopg2 not installed — skipping sync")
        return None
    try:
        c = psycopg2.connect(url, connect_timeout=15)
        return c
    except Exception as exc:
        logger.warning(f"[SUPABASE] Connection failed: {exc}")
        return None


# ── Documents ─────────────────────────────────────────────────────────────────

def sync_documents(scored_df: pd.DataFrame):
    conn = _conn()
    if conn is None:
        return

    now = datetime.now(timezone.utc).isoformat()
    needed = [
        "doc_id", "source", "category", "domain", "url", "title",
        "text", "text_length", "author", "created_dt",
        "crawl_partition", "ingested_at", "content_hash", "aliveness_score",
    ]

    df = scored_df.copy()
    for col in needed:
        if col not in df.columns:
            df[col] = None

    df = df[needed].copy()
    df["text"] = df["text"].apply(
        lambda t: t[:TEXT_MAX] if isinstance(t, str) else t
    )
    df["created_dt"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True)
    df["created_dt"] = df["created_dt"].apply(
        lambda t: t.isoformat() if pd.notna(t) else None
    )

    total = 0
    try:
        with conn:
            cur = conn.cursor()
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                rows = [tuple(r) + (now,) for r in batch.itertuples(index=False, name=None)]
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO documents
                        (doc_id, source, category, domain, url, title, text,
                         text_length, author, created_dt, crawl_partition,
                         ingested_at, content_hash, aliveness_score, scored_at)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE SET
                        aliveness_score = EXCLUDED.aliveness_score,
                        scored_at       = EXCLUDED.scored_at
                """, rows, page_size=BATCH_SIZE)
                total += len(batch)
        logger.info(f"[SUPABASE] ✓ {total:,} documents upserted")
    except Exception as exc:
        logger.warning(f"[SUPABASE] Document sync error: {exc}")
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
                        anomaly_reason  = EXCLUDED.anomaly_reason,
                        updated_at      = NOW()
                """, rows, page_size=BATCH_SIZE)
                logger.info(f"[SUPABASE] ✓ {len(rows):,} composite_index rows upserted")

    except Exception as exc:
        logger.warning(f"[SUPABASE] Index sync error: {exc}")
    finally:
        conn.close()


# ── Convenience: run both ─────────────────────────────────────────────────────

def sync_all(scored_df: pd.DataFrame, engine):
    if not os.environ.get("DATABASE_URL"):
        logger.info("[SUPABASE] DATABASE_URL not set — skipping sync (SQLite only)")
        return
    logger.info("[SUPABASE] Syncing to Postgres …")
    sync_documents(scored_df)
    sync_index(engine)
