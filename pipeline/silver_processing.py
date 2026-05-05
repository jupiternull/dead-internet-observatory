"""
Silver → Gold processing pipeline.

Reads the combined silver Parquet, runs the detection engine on
every document, and writes scored Parquet + updates the SQLite
gold layer via AlivenessIndexEngine.
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
        df = pd.read_parquet(silver_path)
        print(f"[GOLD] {len(df):,} documents loaded")

        # Score
        scored = score_dataframe(df)
        summary = corpus_summary(scored)
        print(f"[GOLD] Summary: {summary}")

        # Save scored Parquet to gold
        out = self.gold_root / "scored.parquet"
        scored.to_parquet(out, index=False, engine="pyarrow")
        print(f"[GOLD] ✓ Scored Parquet → {out}")

        # Update SQLite index
        self.engine.ingest_scored_df(scored)
        print("[GOLD] ✓ SQLite index updated")

        # Sync to Supabase (no-op if DATABASE_URL not set)
        sync_all(scored, self.engine)


if __name__ == "__main__":
    SilverToGoldPipeline().run()
