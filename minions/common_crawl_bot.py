"""
Common Crawl Minion — downloads and processes WET (plain-text) files.

Common Crawl releases quarterly snapshots of the entire public web.
WET files are pre-extracted plaintext — no HTML parsing needed.
We sample a configurable number of WET segments per crawl date,
stream each file, parse WARC records, filter to English, and save
to the bronze layer as JSONL.

Free quota: CC data is 100% free via S3 (no egress from us-east-1)
or via HTTPS. We use HTTPS for simplicity.
"""

import gzip
import io
import random
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse

import requests

from .base_minion import BaseMinion


class CommonCrawlMinion(BaseMinion):

    CC_BASE = "https://data.commoncrawl.org"
    ENGLISH_STOPWORDS = frozenset({
        "the", "is", "are", "was", "were", "and", "or", "but", "in", "on",
        "at", "to", "for", "of", "a", "an", "that", "this", "with", "from",
        "by", "as", "it", "its", "not", "be", "have", "had", "has", "do",
        "did", "will", "would", "can", "could", "should", "may", "might",
    })

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path, "common_crawl")
        cfg = self.config["sources"]["common_crawl"]
        self.crawl_dates: List[str] = cfg["crawl_dates"]
        self.max_segments: int = cfg["max_segments_per_date"]
        self.wet_per_segment: int = cfg.get("wet_files_per_segment", 5)
        self.max_per_wet: int = cfg.get("max_records_per_wet", 2000)
        self.min_text: int = self.config["detection"]["min_text_length"]

    # ── WET path discovery ────────────────────────────────────────────────────

    def get_wet_paths(self, crawl_id: str) -> List[str]:
        """Fetch the master WET paths index for a crawl snapshot."""
        url = f"{self.CC_BASE}/crawl-data/CC-MAIN-{crawl_id}/wet.paths.gz"
        self.logger.info(f"Fetching WET index for CC-MAIN-{crawl_id} …")
        try:
            resp = requests.get(url, timeout=90, stream=True)
            resp.raise_for_status()
            content = gzip.decompress(resp.content).decode("utf-8")
            paths = [ln.strip() for ln in content.splitlines() if ln.strip()]
            self.logger.info(f"  Found {len(paths):,} WET segments")
            return paths
        except Exception as exc:
            self.logger.error(f"  Could not fetch WET index: {exc}")
            return []

    # ── WET streaming ─────────────────────────────────────────────────────────

    def stream_wet_records(self, wet_path: str) -> Iterator[Dict]:
        """Stream and yield parsed records from a single gzipped WET file."""
        url = f"{self.CC_BASE}/{wet_path}"
        self.logger.info(f"  Streaming {wet_path.split('/')[-1]} …")

        try:
            resp = requests.get(url, timeout=180, stream=True)
            resp.raise_for_status()

            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                buf.write(chunk)
            buf.seek(0)

            with gzip.GzipFile(fileobj=buf) as gz:
                raw = gz.read().decode("utf-8", errors="replace")

            count = 0
            for block in raw.split("WARC/1.0\r\n"):
                if count >= self.max_per_wet:
                    break
                record = self._parse_wet_block(block)
                if record:
                    yield record
                    count += 1

            self.logger.info(f"    → Parsed {count:,} records from segment")

        except Exception as exc:
            self.logger.error(f"  Error streaming {wet_path}: {exc}")
            self.stats["errors"] += 1

    def _parse_wet_block(self, block: str) -> Optional[Dict]:
        """Parse one WARC/WET block into a structured record."""
        if "WARC-Type: conversion" not in block:
            return None

        # Split headers from body at first blank line
        header_section, _, body = block.partition("\r\n\r\n")
        headers: Dict[str, str] = {}
        for line in header_section.splitlines():
            if ": " in line:
                k, _, v = line.partition(": ")
                headers[k.strip()] = v.strip()

        url = headers.get("WARC-Target-URI", "")
        date = headers.get("WARC-Date", "")
        body = body.strip()

        if not url or len(body) < self.min_text:
            return None

        domain = self._extract_domain(url)
        if not domain or not self._is_likely_english(body):
            return None

        # Classify domain category
        category = self._categorize_domain(domain, url)

        return {
            "url": url,
            "domain": domain,
            "category": category,
            "date": date,
            "text": body[: self.config["detection"]["max_text_length_for_features"]],
            "text_length": len(body),
            "content_hash": self.content_hash(body),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        try:
            return urlparse(url).netloc.lower().lstrip("www.")
        except Exception:
            return None

    def _is_likely_english(self, text: str) -> bool:
        sample_words = set(text[:600].lower().split())
        return len(sample_words & self.ENGLISH_STOPWORDS) >= 4

    @staticmethod
    def _categorize_domain(domain: str, url: str) -> str:
        news_tlds = {"reuters.com", "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com",
                     "theguardian.com", "washingtonpost.com", "apnews.com", "npr.org"}
        social = {"reddit.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
                  "tiktok.com", "linkedin.com", "tumblr.com", "mastodon.social"}
        wiki = {"wikipedia.org", "wikimedia.org"}
        blog = {"medium.com", "substack.com", "wordpress.com", "blogspot.com"}

        if any(domain.endswith(n) for n in news_tlds):
            return "news"
        if any(domain.endswith(s) for s in social):
            return "social"
        if any(domain.endswith(w) for w in wiki):
            return "wiki"
        if any(domain.endswith(b) for b in blog):
            return "blog"
        return "web"

    # ── Entry point ───────────────────────────────────────────────────────────

    def _latest_crawl_id(self) -> Optional[str]:
        """Fetch the most recent CC crawl ID from the collinfo index."""
        try:
            resp = requests.get(
                "https://index.commoncrawl.org/collinfo.json", timeout=30
            )
            resp.raise_for_status()
            # collinfo returns list sorted newest-first; each entry has 'id' like 'CC-MAIN-2025-13'
            entries = resp.json()
            latest = entries[0]["id"].replace("CC-MAIN-", "")
            self.logger.info(f"Latest CC crawl: {latest}")
            return latest
        except Exception as exc:
            self.logger.warning(f"Could not fetch latest crawl ID: {exc}")
            return None

    def run(self, dry_run: bool = False):
        if not self.config["sources"]["common_crawl"].get("enabled", True):
            self.logger.info("Common Crawl minion disabled — exiting")
            return

        cfg = self.config["sources"]["common_crawl"]
        backfill_done = cfg.get("backfill_done", False)

        if backfill_done:
            # Ongoing mode: only process the latest crawl snapshot
            latest = self._latest_crawl_id()
            if not latest:
                self.logger.error("Could not determine latest crawl — aborting")
                return
            crawl_ids = [latest]
            self.logger.info(f"🤖 Common Crawl Minion (current mode) | crawl={latest}")
        else:
            # Backfill mode: process all configured historical dates
            crawl_ids = self.crawl_dates
            self.logger.info(
                f"🤖 Common Crawl Minion (backfill mode) | "
                f"dates={crawl_ids} | max_segments={self.max_segments}"
            )

        for crawl_id in crawl_ids:
            self.logger.info(f"\n── Crawl: CC-MAIN-{crawl_id} ──────────────────")

            all_paths = self.get_wet_paths(crawl_id)
            if not all_paths:
                continue

            sampled = random.sample(all_paths, min(self.max_segments, len(all_paths)))

            for seg_idx, wet_path in enumerate(sampled, 1):
                self.logger.info(f"  Segment {seg_idx}/{len(sampled)}")
                batch: List[Dict] = []

                for record in self.stream_wet_records(wet_path):
                    batch.append(record)
                    self.stats["fetched"] += 1

                    if len(batch) >= 500:
                        if not dry_run:
                            partition = crawl_id.replace("-", "/")
                            self.save_bronze(batch, "common_crawl", partition)
                        self.stats["processed"] += len(batch)
                        batch = []

                if batch and not dry_run:
                    partition = crawl_id.replace("-", "/")
                    self.save_bronze(batch, "common_crawl", partition)
                    self.stats["processed"] += len(batch)

                self.throttle(3.0)   # respectful delay between large files

        self.report_stats()


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    CommonCrawlMinion().run(dry_run=dry)
