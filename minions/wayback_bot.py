"""
Wayback Machine Minion — longitudinal text comparison via Internet Archive.

This is the observatory's time-machine: it fetches the same set of
"sentinel URLs" (high-traffic, representative pages) at multiple historical
timestamps and archives the extracted text.

Comparing feature scores across time (2019 → 2021 → 2023 → 2025) is the
strongest direct evidence of linguistic drift — you're watching the same
page change, not sampling different pages.

APIs used (both free, no auth):
  CDX API   — find available snapshots for a URL in a date range
  Wayback   — fetch the actual archived HTML at a given timestamp
"""

import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .base_minion import BaseMinion


# ── Sentinel URLs ─────────────────────────────────────────────────────────────
# Mix of news, social, tech, reference, and forum sources.
# Choose pages that (a) existed pre-2019 and (b) update frequently.

DEFAULT_SENTINEL_URLS = [
    # News front pages
    "https://www.reuters.com/",
    "https://www.bbc.com/news",
    "https://www.theguardian.com/international",
    "https://apnews.com/",
    # Tech / HN
    "https://news.ycombinator.com/",
    "https://techcrunch.com/",
    # Reddit (text-only versions are more stable in Wayback)
    "https://old.reddit.com/r/worldnews/",
    "https://old.reddit.com/r/technology/",
    "https://old.reddit.com/r/science/",
    # Reference
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    # Forums
    "https://forums.metafilter.com/",
    "https://lobste.rs/",
    # Blogs / commentary
    "https://marginalrevolution.com/",
    "https://slatestarcodex.com/",
]

# Historical snapshot years to sample — bracket the ChatGPT inflection point
SNAPSHOT_YEARS = [2019, 2021, 2023, 2024, 2025]


class WaybackMinion(BaseMinion):

    CDX_API     = "https://web.archive.org/cdx/search/cdx"
    WAYBACK_BASE = "https://web.archive.org/web"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; DeadInternetObservatory/1.0; "
            "+https://github.com/dead-internet-observatory)"
        )
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path, "wayback")
        cfg = self.config["sources"].get("wayback", {})
        self.sentinel_urls: List[str] = cfg.get("sentinel_urls", DEFAULT_SENTINEL_URLS)
        self.snapshot_years: List[int] = cfg.get("snapshot_years", SNAPSHOT_YEARS)
        self.snapshots_per_year: int = cfg.get("snapshots_per_year", 2)
        self.min_text: int = self.config["detection"]["min_text_length"]

    # ── CDX snapshot discovery ────────────────────────────────────────────────

    def find_snapshots(self, url: str, year: int, limit: int = 3) -> List[str]:
        """
        Return up to `limit` CDX timestamps for `url` in the given year.
        Format: YYYYMMDDHHmmss
        """
        try:
            resp = requests.get(
                self.CDX_API,
                headers=self.HEADERS,
                params={
                    "url": url,
                    "output": "json",
                    "fl": "timestamp,statuscode",
                    "filter": "statuscode:200",
                    "from": f"{year}0601",   # mid-year — avoid Jan/Dec extremes
                    "to": f"{year}1130",
                    "limit": limit,
                    "fastLatest": "true",
                    "collapse": "timestamp:8",   # one per day
                },
                timeout=30,
            )
            resp.raise_for_status()
            rows = resp.json()
            # First row is header ["timestamp","statuscode"]
            if len(rows) <= 1:
                return []
            return [row[0] for row in rows[1:] if row[1] == "200"]
        except Exception as exc:
            self.logger.debug(f"CDX lookup failed for {url} ({year}): {exc}")
            return []

    # ── Snapshot fetching ─────────────────────────────────────────────────────

    def fetch_snapshot(self, url: str, timestamp: str) -> Optional[Dict]:
        """
        Fetch a specific Wayback snapshot and extract article text.
        Returns None if the page is unavailable or too short.
        """
        wayback_url = f"{self.WAYBACK_BASE}/{timestamp}/{url}"

        try:
            resp = requests.get(
                wayback_url,
                headers=self.HEADERS,
                timeout=40,
                allow_redirects=True,
            )
            # Wayback occasionally returns 404 or redirects to error pages
            if resp.status_code != 200:
                return None
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct and "text" not in ct:
                return None
        except requests.Timeout:
            self.logger.debug(f"Timeout: {wayback_url}")
            return None
        except Exception as exc:
            self.logger.debug(f"Fetch error {wayback_url}: {exc}")
            return None

        text = self._extract_text(resp.content, url)
        if not text or len(text) < self.min_text:
            return None

        # Parse the snapshot date from the timestamp (YYYYMMDDHHmmss)
        try:
            snap_dt = datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
            snap_date = snap_dt.isoformat()
            snap_year = snap_dt.year
        except ValueError:
            snap_date = None
            snap_year = int(timestamp[:4]) if len(timestamp) >= 4 else None

        domain = urlparse(url).netloc.lstrip("www.")

        return {
            "sentinel_url": url,
            "wayback_url": wayback_url,
            "timestamp": timestamp,
            "snapshot_date": snap_date,
            "snapshot_year": snap_year,
            "domain": domain,
            "category": self._categorise(domain),
            "text": text[: self.config["detection"]["max_text_length_for_features"]],
            "text_length": len(text),
            "content_hash": self.content_hash(text),
        }

    # ── Text extraction ───────────────────────────────────────────────────────

    BODY_SELECTORS = [
        "article", "main", '[role="main"]',
        ".article-body", ".story-body", ".entry-content",
        ".post-content", "#content", "#main-content",
    ]

    def _extract_text(self, html_bytes: bytes, source_url: str) -> str:
        try:
            soup = BeautifulSoup(html_bytes, "lxml")
        except Exception:
            return ""

        # Strip Wayback toolbar and noise
        for sel in ["#wm-ipp-base", "#wm-ipp", ".wb-autocomplete-suggestion",
                    "script", "style", "nav", "footer", "header",
                    "aside", "iframe", "noscript", ".ad", ".advertisement"]:
            for tag in soup.select(sel):
                tag.decompose()

        # Try semantic containers
        for selector in self.BODY_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(separator="\n", strip=True)
                if len(text) >= self.min_text:
                    return text

        # Paragraph fallback
        paras = [p.get_text(strip=True) for p in soup.find_all("p")
                 if len(p.get_text(strip=True)) > 40]
        return "\n".join(paras)

    @staticmethod
    def _categorise(domain: str) -> str:
        news = {"reuters.com", "bbc.com", "bbc.co.uk", "theguardian.com",
                "apnews.com", "techcrunch.com"}
        social = {"reddit.com", "old.reddit.com", "metafilter.com",
                  "news.ycombinator.com", "lobste.rs"}
        wiki = {"wikipedia.org"}
        blog = {"marginalrevolution.com", "slatestarcodex.com", "substack.com"}
        if any(domain.endswith(n) for n in news):   return "news"
        if any(domain.endswith(s) for s in social): return "social"
        if any(domain.endswith(w) for w in wiki):   return "wiki"
        if any(domain.endswith(b) for b in blog):   return "blog"
        return "web"

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        if not self.config["sources"].get("wayback", {}).get("enabled", True):
            self.logger.info("Wayback minion disabled — exiting")
            return

        self.logger.info(
            f"🤖 Wayback Minion starting | "
            f"sentinels={len(self.sentinel_urls)} | years={self.snapshot_years}"
        )

        # Partition by current run date — but records carry their own snapshot_year
        run_partition = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        batch: List[Dict] = []

        for url in self.sentinel_urls:
            domain = urlparse(url).netloc.lstrip("www.")
            self.logger.info(f"  Sentinel: {domain}")

            for year in self.snapshot_years:
                timestamps = self.find_snapshots(url, year, self.snapshots_per_year)

                if not timestamps:
                    self.logger.debug(f"    No snapshots found for {domain} in {year}")
                    self.stats["skipped"] += 1
                    continue

                for ts in timestamps:
                    self.stats["fetched"] += 1
                    record = self.fetch_snapshot(url, ts)

                    if record:
                        batch.append(record)
                        self.stats["processed"] += 1
                        self.logger.info(
                            f"    ✓ {domain} {year} — {len(record['text'])} chars"
                        )
                    else:
                        self.stats["skipped"] += 1

                    # Wayback asks for polite crawling — 1 req/sec
                    self.throttle(1.5)

                if len(batch) >= 50:
                    self.save_bronze(batch, "wayback", run_partition)
                    batch = []

            # Extra pause between sentinel URLs
            self.throttle(2.0)

        if batch:
            self.save_bronze(batch, "wayback", run_partition)

        self.report_stats()


if __name__ == "__main__":
    WaybackMinion().run()
