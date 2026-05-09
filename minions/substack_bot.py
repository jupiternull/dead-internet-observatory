"""
Substack Minion — harvests posts from curated Substack publications.

Uses the undocumented but public JSON API at /{slug}.substack.com/api/v1/posts
instead of RSS feeds, which Substack blocks from server IP ranges (403).
"""

import hashlib
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

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

    # ── JSON API ──────────────────────────────────────────────────────────────

    def _fetch_publication(self, slug: str) -> List[Dict]:
        api_url = f"https://{slug}.substack.com/api/v1/posts?limit=25&offset=0"
        self.logger.info(f"  Fetching {slug}")
        try:
            resp = self.session.get(api_url, timeout=20)
            if resp.status_code in (403, 404):
                self.logger.warning(f"  [{slug}] HTTP {resp.status_code} — skipping")
                self.stats["skipped"] += 1
                return []
            resp.raise_for_status()
            posts = resp.json()
        except Exception as exc:
            self.logger.warning(f"  [{slug}] Failed: {exc}")
            self.stats["errors"] += 1
            return []

        self.logger.info(f"  [{slug}] {len(posts)} posts")
        self.stats["fetched"] += len(posts)

        records: List[Dict] = []
        for post in posts:
            url = post.get("canonical_url") or post.get("url", "")
            if not url:
                self.stats["skipped"] += 1
                continue

            raw_html = post.get("body_html") or post.get("subtitle") or ""
            text = self._strip_html(raw_html).strip() if raw_html else ""
            if not text:
                text = (post.get("subtitle") or "").strip()
            if len(text) < MIN_TEXT_LEN:
                self.stats["skipped"] += 1
                continue

            author = ""
            if post.get("publishedBylines"):
                author = post["publishedBylines"][0].get("name", "")

            records.append({
                "post_id":      self._post_id(url),
                "publication":  slug,
                "title":        (post.get("title") or "").strip(),
                "text":         text[:TEXT_TRUNCATE],
                "author":       author,
                "published_at": post.get("post_date") or post.get("updated_at", ""),
                "url":          url,
                "word_count":   len(text.split()),
                "category":     "blog",
            })
            self.stats["processed"] += 1

        return records

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting — {len(SUBSTACKS)} publications")
        all_records: List[Dict] = []

        for slug in SUBSTACKS:
            records = self._fetch_publication(slug)
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
