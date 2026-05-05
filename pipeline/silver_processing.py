"""
Silver → Gold processing pipeline.

Incremental: only scores documents not already present in the gold layer.
New docs are appended to scored.parquet; the SQLite index and Supabase are
updated with each run's delta.
"""

from pathlib import Path

import pandas as pd
import yaml

from detection.ai_content_detector import score_dataframe, corpus_summary
from analytics.aliveness_index import AlivenessIndexEngine
from pipeline.supabase_sync import sync_all


class SilverToGoldPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)
        data_root = Path(self.config["storage"]["local_path"])
        self.silver_root = data_root / self.config["storage"]["silver_path"]
        self.gold_root   = data_root / self.config["storage"]["gold_path"]
        self.gold_root.mkdir(parents=True, exist_ok=True)
        self.engine = AlivenessIndexEngine(config_path)

    def run(self, source_file: str = "combined.parquet"):
        silver_path = self.silver_root / source_file
        if not silver_path.exists():
            print(f"[GOLD] Silver file not found: {silver_path}")
            return

        print(f"[GOLD] Loading {silver_path} …")
        silver_df = pd.read_parquet(silver_path)
        print(f"[GOLD] {len(silver_df):,} docs in silver")

        # ── Load already-scored doc_ids ───────────────────────────────────────
        gold_path = self.gold_root / "scored.parquet"
        if gold_path.exists():
            existing_ids = set(pd.read_parquet(gold_path, columns=["doc_id"])["doc_id"])
            print(f"[GOLD] {len(existing_ids):,} docs already scored")
        else:
            existing_ids = set()

        new_df = silver_df[~silver_df["doc_id"].isin(existing_ids)].copy()
        print(f"[GOLD] {len(new_df):,} new docs to score")

        if new_df.empty:
            print("[GOLD] Nothing new — index already up to date")
            return

        # ── Score only the new docs ───────────────────────────────────────────
        scored = score_dataframe(new_df)
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

        combined.to_parquet(gold_path, index=False, engine="pyarrow")
        print(f"[GOLD] ✓ scored.parquet now has {len(combined):,} docs")

        # ── Update SQLite index with this run's delta only ────────────────────
        self.engine.ingest_scored_df(scored)
        print("[GOLD] ✓ SQLite index updated")

        # ── Sync delta to Supabase ────────────────────────────────────────────
        sync_all(scored, self.engine)


if __name__ == "__main__":
    SilverToGoldPipeline().run()
