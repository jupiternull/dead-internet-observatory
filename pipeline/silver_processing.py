"""
Silver → Gold processing pipeline.

Incremental: only scores documents not already in data/scored_ids.json.
That file is preserved between GitHub Actions runs via actions/cache.
On cache eviction the full silver batch is rescored once, then the cache
rebuilds normally.

A hard cap (max_new_docs) keeps the Gold step within the 120-min GH Actions
timeout regardless of how many new docs accumulate between runs.
Perplexity scoring (slow on CPU) is skipped for common_crawl docs.
"""

import os
from pathlib import Path

import pandas as pd
import yaml

from detection.ai_content_detector import score_dataframe, corpus_summary
from analytics.aliveness_index import AlivenessIndexEngine
from pipeline.dedup_persistence import load_scored_ids, append_scored_ids

MAX_NEW_DOCS = 3_000


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
        existing_ids = load_scored_ids()
        if existing_ids:
            print(f"[GOLD] {len(existing_ids):,} docs already scored (cache) — skipping")
        else:
            print("[GOLD] No prior scored ids found — scoring fresh batch")

        new_df = silver_df[~silver_df["doc_id"].isin(existing_ids)].copy()
        print(f"[GOLD] {len(new_df):,} new docs to score")

        if new_df.empty:
            print("[GOLD] Nothing new — index already up to date")
            return

        # Sort newest-first so we prioritise recent content when capping
        if "created_dt" in new_df.columns:
            new_df = new_df.sort_values("created_dt", ascending=False)

        # Cap per run to stay within GH Actions timeout
        if len(new_df) > MAX_NEW_DOCS:
            print(f"[GOLD] Capping to {MAX_NEW_DOCS:,} docs (was {len(new_df):,})")
            new_df = new_df.head(MAX_NEW_DOCS)

        # ── Disable perplexity for common_crawl (too slow on CPU) ────────────
        orig_perplexity = os.environ.get("ENABLE_PERPLEXITY", "")
        if "source" in new_df.columns:
            cc_mask = new_df["source"] == "common_crawl"
        else:
            cc_mask = pd.Series(False, index=new_df.index)
        non_cc  = new_df[~cc_mask].copy()
        cc_only = new_df[cc_mask].copy()

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

        # ── Persist scored ids so next run skips them ─────────────────────────
        append_scored_ids(set(scored["doc_id"].dropna()))
        print("[GOLD] ✓ scored_ids.json updated")

        # ── Update SQLite index with this run's delta ─────────────────────────
        self.engine.ingest_scored_df(scored)
        print("[GOLD] ✓ SQLite index updated")


if __name__ == "__main__":
    SilverToGoldPipeline().run()
