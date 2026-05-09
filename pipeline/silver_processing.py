"""
Silver → Gold processing pipeline.

Incremental: only scores documents not already present in the gold layer.
New docs are appended to scored.parquet; the SQLite index is updated each run.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import yaml

from detection.ai_content_detector import score_dataframe, corpus_summary
from analytics.aliveness_index import AlivenessIndexEngine


class SilverToGoldPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)
        data_root = Path(self.config["storage"]["local_path"])
        self.silver_root = data_root / self.config["storage"]["silver_path"]
        self.gold_root   = data_root / self.config["storage"]["gold_path"]
        self.gold_root.mkdir(parents=True, exist_ok=True)
        self.engine = AlivenessIndexEngine(config_path)

    def run(self, source_file: str = "combined.parquet", max_new_docs: int = 3000):
        silver_path = self.silver_root / source_file
        if not silver_path.exists():
            print(f"[GOLD] Silver file not found: {silver_path}")
            return

        print(f"[GOLD] Loading {silver_path} …")
        silver_df = pd.read_parquet(silver_path)
        print(f"[GOLD] {len(silver_df):,} docs in silver")

        # ── Load already-scored doc_ids from SQLite (committed to repo) ─────────
        # Gold parquet is not committed between runs, so use observatory.db as
        # the persistent source of truth for which doc_ids have been scored.
        db_path = self.config["storage"].get("db_path", "./data/observatory.db")
        existing_ids: set = set()
        if Path(db_path).exists():
            try:
                with sqlite3.connect(db_path) as con:
                    rows = con.execute("SELECT doc_id FROM scored_docs").fetchall()
                existing_ids = {r[0] for r in rows}
                print(f"[GOLD] {len(existing_ids):,} docs already scored (from SQLite) — skipping")
            except Exception as exc:
                print(f"[GOLD] Could not read scored_docs from SQLite ({exc}) — will score all")
        else:
            print("[GOLD] observatory.db not found — will score all docs")

        new_df = silver_df[~silver_df["doc_id"].isin(existing_ids)].copy()
        total_new = len(new_df)
        print(f"[GOLD] {total_new:,} new docs to score")

        # Cap per-run to keep the Gold step within the pipeline timeout.
        # Sort newest-first so fresh content is always prioritised.
        if total_new > max_new_docs:
            sort_col = "collected_at" if "collected_at" in new_df.columns else None
            if sort_col:
                new_df = new_df.sort_values(sort_col, ascending=False)
            new_df = new_df.head(max_new_docs)
            print(f"[GOLD] Capped to {max_new_docs:,} docs this run — {total_new - max_new_docs:,} deferred to next run")

        if new_df.empty:
            print("[GOLD] Nothing new — index already up to date")
            return

        # ── Score only the new docs ───────────────────────────────────────────
        scored = score_dataframe(new_df)
        summary = corpus_summary(scored)
        print(f"[GOLD] Summary: {summary}")

        # ── Append to gold parquet ────────────────────────────────────────────
        gold_path = self.gold_root / "scored.parquet"
        if gold_path.exists():
            combined = pd.concat(
                [pd.read_parquet(gold_path), scored],
                ignore_index=True,
            )
        else:
            combined = scored

        combined.to_parquet(gold_path, index=False, engine="pyarrow")
        print(f"[GOLD] ✓ scored.parquet now has {len(combined):,} docs")

        # ── Update SQLite index with this run's delta only ────────────────────
        self.engine.ingest_scored_df(scored)
        print("[GOLD] ✓ SQLite index updated")


if __name__ == "__main__":
    SilverToGoldPipeline().run()
