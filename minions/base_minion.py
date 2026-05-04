"""
BaseMinion — abstract foundation for all data-collection bots.

Every minion inherits from here and gets: config loading, structured logging,
bronze-layer save helpers, content hashing, and throttling.
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml


class BaseMinion(ABC):

    def __init__(self, config_path: str = "config/config.yaml", minion_name: str = "base"):
        self.minion_name = minion_name
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        self.data_root = Path(self.config["storage"]["local_path"])
        self.bronze_root = self.data_root / self.config["storage"]["bronze_path"]
        self.bronze_root.mkdir(parents=True, exist_ok=True)

        self.stats: Dict[str, int] = {
            "fetched": 0,
            "processed": 0,
            "errors": 0,
            "skipped": 0,
        }

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> Dict:
        with open(path) as fh:
            return yaml.safe_load(fh)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"minion.{self.minion_name}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                f"[%(asctime)s] \033[96m[{self.minion_name.upper():15s}]\033[0m %(levelname)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(handler)
        return logger

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def content_hash(text: str) -> str:
        """Short SHA-256 fingerprint for dedup."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    def throttle(self, seconds: float = 1.0):
        """Polite delay between requests."""
        time.sleep(seconds)

    # ── Storage ───────────────────────────────────────────────────────────────

    def _bronze_dir(self, source: str, partition: Optional[str] = None) -> Path:
        partition = partition or datetime.now(timezone.utc).strftime("%Y/%m/%d")
        out = self.bronze_root / source / partition
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _stamp(self, record: Dict, source: str) -> Dict:
        record["_ingested_at"] = datetime.now(timezone.utc).isoformat()
        record["_source"] = source
        record["_minion"] = self.minion_name
        return record

    def save_bronze(
        self,
        records: List[Dict],
        source: str,
        partition: Optional[str] = None,
    ) -> Path:
        """Persist a list of records as JSONL in the bronze layer."""
        out_dir = self._bronze_dir(source, partition)
        ts = datetime.now(timezone.utc).strftime("%H%M%S_%f")
        out_file = out_dir / f"{self.minion_name}_{ts}.jsonl"

        with open(out_file, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(self._stamp(rec, source), ensure_ascii=False) + "\n")

        self.logger.info(f"  ✓ Saved {len(records):,} records → {out_file.relative_to(self.data_root)}")
        return out_file

    def save_bronze_stream(
        self,
        records: Iterator[Dict],
        source: str,
        partition: Optional[str] = None,
    ):
        """Stream-save records without buffering the full list in memory."""
        out_dir = self._bronze_dir(source, partition)
        ts = datetime.now(timezone.utc).strftime("%H%M%S_%f")
        out_file = out_dir / f"{self.minion_name}_{ts}.jsonl"

        count = 0
        with open(out_file, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(self._stamp(rec, source), ensure_ascii=False) + "\n")
                count += 1

        self.logger.info(f"  ✓ Streamed {count:,} records → {out_file.relative_to(self.data_root)}")
        return out_file, count

    # ── Stats ─────────────────────────────────────────────────────────────────

    def report_stats(self):
        self.logger.info(
            f"Final stats — "
            f"fetched={self.stats['fetched']:,}  "
            f"processed={self.stats['processed']:,}  "
            f"errors={self.stats['errors']:,}  "
            f"skipped={self.stats['skipped']:,}"
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def run(self):
        """Each minion implements its own crawl/harvest logic here."""
        ...
