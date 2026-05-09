"""
Substack Minion — harvests posts from curated Substack publications via RSS.

Parses each publication's feed at https://{slug}.substack.com/feed, strips
HTML from the post body, and saves records to the bronze layer.
"""

import hashlib
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

from minions.base_minion import BaseMinion


SUBSTACKS = [
    "platformer",
    "theovershoot",
    "noahpinion",
    "astralcodexten",
    "stratechery",
    "thebulwark",
    "slowboring",
    "simonwillison",
    "thezvi",
    "interconnects",
    "aisnakeoil",
    "importai",
    "chartr",
    "notboring",
    "worksinprogress",
    "Construction-Physics",
    "secondbreakfast",
    "grimoireofcode",
    "thelogic",
    "apricitas",
]

TEXT_TRUNCATE = 5000
MIN_TEXT_LEN = 100


class SubstackBot(BaseMinion):

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0 "
            "DeadInternetObservatory/1.0"
        )
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="substack")
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _post_id(url: str) -> str:
        """SHA-1 of the URL used as a stable post identifier."""
        return hashlib.sha1(url.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip all HTML tags and return plain text."""
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n", strip=True)

    @staticmethod
    def _parse_date(entry) -> str:
        """Return an ISO8601 date string from a feedparser entry."""
        if entry.get("published_parsed"):
            try:
                return datetime(
                    *entry.published_parsed[:6], tzinfo=timezone.utc
                ).isoformat()
            except Exception:
                pass
        return entry.get("published", "")

    def _get_raw_text(self, entry) -> str:
        """Prefer full content over summary; fall back gracefully."""
        content_list = entry.get("content", [])
        if content_list:
            return content_list[0].get("value", "")
        return entry.get("summary", "") or ""

    # ── Feed parsing ──────────────────────────────────────────────────────────

    def _parse_publication(self, slug: str) -> List[Dict]:
        feed_url = f"https://{slug}.substack.com/feed"
        self.logger.info(f"  Fetching {feed_url}")

        try:
            # feedparser accepts a URL but also respects a pre-fetched response;
            # pass the URL directly and let feedparser handle HTTP.
            feed = feedparser.parse(
                feed_url,
                request_headers=self.HEADERS,
            )
        except Exception as exc:
            self.logger.warning(f"  [{slug}] Feed parse exception: {exc}")
            self.stats["errors"] += 1
            return []

        status = getattr(feed, "status", 200)
        if status in (403, 404):
            self.logger.warning(f"  [{slug}] HTTP {status} — skipping")
            self.stats["skipped"] += 1
            return []
        if status not in range(200, 400):
            self.logger.warning(f"  [{slug}] HTTP {status} — skipping")
            self.stats["errors"] += 1
            return []

        entries = feed.get("entries", [])
        self.logger.info(f"  [{slug}] {len(entries)} entries found")
        self.stats["fetched"] += len(entries)

        records: List[Dict] = []
        for entry in entries:
            url = entry.get("link", "").strip()
            if not url:
                self.stats["skipped"] += 1
                continue

            raw_html = self._get_raw_text(entry)
            text = self._strip_html(raw_html).strip()

            if len(text) < MIN_TEXT_LEN:
                self.stats["skipped"] += 1
                continue

            title = entry.get("title", "").strip()
            author = (
                entry.get("author", "")
                or entry.get("author_detail", {}).get("name", "")
            ).strip()
            published_at = self._parse_date(entry)
            word_count = len(text.split())

            records.append({
                "post_id": self._post_id(url),
                "publication": slug,
                "title": title,
                "text": text[:TEXT_TRUNCATE],
                "author": author,
                "published_at": published_at,
                "url": url,
                "word_count": word_count,
                "category": "blog",
            })
            self.stats["processed"] += 1

        return records

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting — {len(SUBSTACKS)} publications")
        all_records: List[Dict] = []

        for slug in SUBSTACKS:
            records = self._parse_publication(slug)
            all_records.extend(records)
            self.throttle(1.0)

        if all_records:
            self.save_bronze(all_records, source="substack")
            self.logger.info(f"Done — {len(all_records)} posts saved")
        else:
            self.logger.warning("Done — no records collected")

        self.report_stats()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = SubstackBot(config_path=config_path)
    bot.run()
