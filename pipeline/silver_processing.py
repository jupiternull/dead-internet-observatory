"""
Silver → Gold processing pipeline.

Incremental: only scores documents not already present in the gold layer.
New docs are appended to scored.parquet; the durable registry and SQLite index
are updated with each run.
"""

import os
from pathlib import Path

import pandas as pd
import yaml

from detection.ai_content_detector import score_dataframe, corpus_summary
from analytics.aliveness_index import AlivenessIndexEngine
from pipeline.supabase_sync import get_scored_doc_ids

MAX_NEW_DOCS = 6_000
INDEX_TABLES = (
    "daily_index",
    "composite_index",
    "domain_scores",
    "meta",
)


class SilverToGoldPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)
        data_root = Path(self.config["storage"]["local_path"])
        self.silver_root = data_root / self.config["storage"]["silver_path"]
        self.gold_root   = data_root / self.config["storage"]["gold_path"]
        self.gold_root.mkdir(parents=True, exist_ok=True)
        self.engine = AlivenessIndexEngine(config_path)

    def _bootstrap_index(self):
        if not os.environ.get("DATABASE_URL"):
            return

        try:
            import psycopg2
        except ImportError:
            print("[GOLD] psycopg2 unavailable; skipping Supabase index bootstrap")
            return

        with self.engine._conn() as local:
            existing = local.execute("SELECT COUNT(*) FROM composite_index").fetchone()[0]
        if existing > 1:
            return

        print("[GOLD] Bootstrapping aggregate history from Supabase")
        remote = psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=15)
        try:
            with remote.cursor() as cur, self.engine._conn() as local:
                for table in INDEX_TABLES:
                    columns = [
                        row[1]
                        for row in local.execute(f"PRAGMA table_info({table})").fetchall()
                    ]
                    cur.execute(f"SELECT {','.join(columns)} FROM {table}")
                    rows = cur.fetchall()
                    if not rows:
                        continue
                    placeholders = ",".join("?" for _ in columns)
                    local.execute(f"DELETE FROM {table}")
                    local.executemany(
                        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                        rows,
                    )
                    print(f"[GOLD] Imported {len(rows):,} {table} rows")
        finally:
            remote.close()

    def run(self, source_file: str = "combined.parquet"):
        self._bootstrap_index()
        silver_path = self.silver_root / source_file
        if not silver_path.exists():
            print(f"[GOLD] Silver file not found: {silver_path}")
            return

        print(f"[GOLD] Loading {silver_path} …")
        silver_df = pd.read_parquet(silver_path)
        print(f"[GOLD] {len(silver_df):,} docs in silver")

        # ── Load already-scored doc_ids ───────────────────────────────────────
        gold_path = self.gold_root / "scored.parquet"
        registry_path = self.gold_root / "doc_registry.parquet"
        local_existing_ids = set()
        if gold_path.exists():
            local_existing_ids = set(pd.read_parquet(gold_path, columns=["doc_id"])["doc_id"].dropna())

        registry_ids = set()
        if registry_path.exists():
            registry_ids = set(
                pd.read_parquet(registry_path, columns=["doc_id"])["doc_id"].dropna()
            )
        elif os.environ.get("DATABASE_URL"):
            print("[GOLD] Bootstrapping durable registry from Supabase")
            registry_ids = get_scored_doc_ids()
            if not registry_ids:
                raise RuntimeError("Supabase registry bootstrap returned no document IDs")
        registry_ids.update(local_existing_ids)
        if registry_ids:
            pd.DataFrame({"doc_id": sorted(registry_ids)}).to_parquet(
                registry_path, index=False, engine="pyarrow"
            )
            with self.engine._conn() as conn:
                conn.execute(
                    """INSERT INTO meta (key, value) VALUES ('total_scored_count', ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                    (str(len(registry_ids)),),
                )
            print(f"[GOLD] Persisted {len(registry_ids):,} registry IDs")

        candidate_ids = silver_df["doc_id"].dropna().unique().tolist()
        existing_ids = local_existing_ids | registry_ids

        if existing_ids:
            print(
                f"[GOLD] {len(existing_ids):,} docs already scored "
                f"({len(local_existing_ids):,} scored, {len(registry_ids):,} registry)"
            )
        else:
            print("[GOLD] No prior scored docs found — scoring full silver batch")

        new_df = silver_df[~silver_df["doc_id"].isin(existing_ids)].copy()
        new_df = new_df.drop_duplicates(subset=["doc_id"], keep="last")
        print(f"[GOLD] {len(new_df):,} new docs to score")

        if new_df.empty:
            print("[GOLD] Nothing new — index already up to date")
            return

        # Sort newest-first, then cap to stay within GH Actions timeout
        if "created_dt" in new_df.columns:
            new_df = new_df.sort_values("created_dt", ascending=False)

        if len(new_df) > MAX_NEW_DOCS:
            print(f"[GOLD] Capping to {MAX_NEW_DOCS:,} docs (was {len(new_df):,})")
            new_df = new_df.head(MAX_NEW_DOCS)

        # ── Score, with perplexity disabled for common_crawl (too slow) ───────
        orig_perplexity = os.environ.get("ENABLE_PERPLEXITY", "")
        if "source" in new_df.columns:
            cc_mask = new_df["source"] == "common_crawl"
        else:
            cc_mask = pd.Series(False, index=new_df.index)

        cc_only = new_df[cc_mask].copy()
        non_cc  = new_df[~cc_mask].copy()

        scored_parts = []
        if not cc_only.empty:
            print(f"[GOLD] Scoring {len(cc_only):,} CC docs (perplexity disabled)")
            os.environ["ENABLE_PERPLEXITY"] = ""
            scored_parts.append(score_dataframe(cc_only))
            os.environ["ENABLE_PERPLEXITY"] = orig_perplexity

        if not non_cc.empty:
            print(f"[GOLD] Scoring {len(non_cc):,} non-CC docs")
            scored_parts.append(score_dataframe(non_cc))

        scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()

        if scored.empty:
            print("[GOLD] No scored output — nothing to persist")
            return

        summary = corpus_summary(scored)
        print(f"[GOLD] Summary: {summary}")

        # ── Append to gold parquet ────────────────────────────────────────────
        if gold_path.exists():
            combined = pd.concat(
                [pd.read_parquet(gold_path), scored],
                ignore_index=True,
            )
        else:
            combined = scored

        combined = combined.drop_duplicates(subset=["doc_id"], keep="last")
        combined.to_parquet(gold_path, index=False, engine="pyarrow")
        print(f"[GOLD] ✓ scored.parquet now has {len(combined):,} docs")

        registry_ids.update(scored["doc_id"].dropna().unique())
        pd.DataFrame({"doc_id": sorted(registry_ids)}).to_parquet(
            registry_path, index=False, engine="pyarrow"
        )
        print(f"[GOLD] ✓ doc_registry.parquet now has {len(registry_ids):,} docs")

        self.engine.ingest_scored_delta(scored)
        with self.engine._conn() as conn:
            conn.execute(
                """INSERT INTO meta (key, value) VALUES ('total_scored_count', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (str(len(registry_ids)),),
            )
        print("[GOLD] ✓ SQLite index updated")


if __name__ == "__main__":
    SilverToGoldPipeline().run()
