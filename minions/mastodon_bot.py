"""
Mastodon Minion — harvests public toots from multiple Fediverse instances.

Uses only the public REST API — no authentication required.
Two endpoints per instance:
  - /api/v1/timelines/public?limit=40&local=true  — local public timeline
  - /api/v1/timelines/tag/{tag}?limit=40          — per-hashtag timeline

Posts are deduplicated by status_id across the full run.
Non-English posts and very short posts (< 20 chars) are skipped.
"""

import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup

from minions.base_minion import BaseMinion


INSTANCES = [
    "mastodon.social",
    "fosstodon.org",
    "infosec.exchange",
    "hachyderm.io",
    "sigmoid.social",
]

TAGS = [
    "technology",
    "ai",
    "news",
    "science",
    "programming",
]

SOURCE = "mastodon"
CATEGORY = "social"
HEADERS = {
    "User-Agent": "DeadInternetObservatory/1.0 (research; github.com/dead-internet-observatory)"
}
MIN_TEXT_LEN = 20


def _strip_html(html: str) -> str:
    """Strip HTML tags from Mastodon content field, returning clean plain text."""
    if not html:
        return ""
    try:
        return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)
    except Exception:
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()


class MastodonBot(BaseMinion):

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="mastodon")
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.seen_ids: Set[str] = set()

    # ── Fetch helpers ─────────────────────────────────────────────────────────

    def _get_timeline(self, instance: str, url: str) -> List[Dict]:
        """Fetch a single timeline page and return raw status items."""
        try:
            resp = self.session.get(url, timeout=20)
            if resp.status_code == 404:
                self.logger.debug(f"  404 for {url} — skipping")
                return []
            if resp.status_code == 403:
                self.logger.debug(f"  403 for {instance} — instance may require auth")
                return []
            resp.raise_for_status()
            return resp.json() or []
        except requests.exceptions.Timeout:
            self.logger.warning(f"  Timeout fetching {url}")
            self.stats["errors"] += 1
            return []
        except Exception as exc:
            self.logger.warning(f"  Error fetching {url}: {exc}")
            self.stats["errors"] += 1
            return []

    def _fetch_public_timeline(self, instance: str) -> List[Dict]:
        url = f"https://{instance}/api/v1/timelines/public?limit=40&local=true"
        self.logger.debug(f"  Public timeline: {instance}")
        return self._get_timeline(instance, url)

    def _fetch_tag_timeline(self, instance: str, tag: str) -> List[Dict]:
        url = f"https://{instance}/api/v1/timelines/tag/{tag}?limit=40"
        self.logger.debug(f"  Tag timeline: {instance} #{tag}")
        return self._get_timeline(instance, url)

    # ── Record builder ────────────────────────────────────────────────────────

    def _parse_status(self, status: Dict, instance: str) -> Optional[Dict]:
        """
        Parse a raw Mastodon status object into a bronze record.
        Returns None if the post should be filtered out.
        """
        # Skip boosts (reblogs)
        if status.get("reblog") is not None:
            self.stats["skipped"] += 1
            return None

        status_id = str(status.get("id", ""))
        if not status_id:
            return None

        # Dedup
        if status_id in self.seen_ids:
            self.stats["skipped"] += 1
            return None

        # Language filter
        lang = (status.get("language") or "").lower()
        if lang and lang != "en":
            self.stats["skipped"] += 1
            return None

        # Strip HTML content
        text = _strip_html(status.get("content", ""))
        if len(text) < MIN_TEXT_LEN:
            self.stats["skipped"] += 1
            return None

        self.seen_ids.add(status_id)

        account = status.get("account") or {}
        acct = account.get("acct", "")
        # Normalise: if the acct has no @instance suffix, append it
        if acct and "@" not in acct:
            acct = f"{acct}@{instance}"

        tag_names = [t.get("name", "") for t in (status.get("tags") or []) if t.get("name")]

        return {
            "status_id":        status_id,
            "text":             text,
            "author_acct":      acct,
            "instance":         instance,
            "created_at":       status.get("created_at", ""),
            "language":         lang or "en",
            "url":              status.get("url") or status.get("uri", ""),
            "replies_count":    status.get("replies_count", 0),
            "reblogs_count":    status.get("reblogs_count", 0),
            "favourites_count": status.get("favourites_count", 0),
            "tags":             tag_names,
            "category":         CATEGORY,
        }

    # ── Main run ──────────────────────────────────────────────────────────────

    def _harvest_instance(self, instance: str) -> List[Dict]:
        """Collect records from the public timeline and all tag timelines for one instance."""
        records: List[Dict] = []

        # Public local timeline
        statuses = self._fetch_public_timeline(instance)
        self.stats["fetched"] += len(statuses)
        for status in statuses:
            rec = self._parse_status(status, instance)
            if rec:
                records.append(rec)
                self.stats["processed"] += 1
        self.throttle(0.5)

        # Per-tag timelines
        for tag in TAGS:
            statuses = self._fetch_tag_timeline(instance, tag)
            self.stats["fetched"] += len(statuses)
            for status in statuses:
                rec = self._parse_status(status, instance)
                if rec:
                    records.append(rec)
                    self.stats["processed"] += 1
            self.throttle(0.3)

        return records

    def run(self):
        self.logger.info(
            f"Mastodon Minion starting | "
            f"instances={len(INSTANCES)} | tags={len(TAGS)}"
        )

        for instance in INSTANCES:
            self.logger.info(f"  Harvesting {instance} …")
            try:
                records = self._harvest_instance(instance)
            except Exception as exc:
                self.logger.error(f"  Unhandled error for {instance}: {exc}")
                self.stats["errors"] += 1
                records = []

            if records:
                self.save_bronze(records, source=SOURCE)
                self.logger.info(
                    f"  {instance}: {len(records)} records saved "
                    f"(total seen: {len(self.seen_ids)})"
                )
            else:
                self.logger.info(f"  {instance}: no records collected")

        self.logger.info(
            f"Done — {len(self.seen_ids)} unique statuses collected across all instances"
        )
        self.report_stats()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = MastodonBot(config_path=config_path)
    bot.run()
