"""
Wikipedia Minion — tracks article text quality and edit-pattern signals.

Two modes:
  1. recent_changes — fetch all edits in the last N hours, compute
     aggregate bot-ratio, revert-ratio, and editor diversity metrics.
  2. article_sample — fetch random articles' plaintext + revision history
     for linguistic analysis.

Both datasets feed the detection pipeline as longitudinal signals.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Iterator, List, Optional

import requests

from .base_minion import BaseMinion


class WikipediaMinion(BaseMinion):

    API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {
        "User-Agent": (
            "DeadInternetObservatory/1.0 (academic research; "
            "contact via github.com/dead-internet-observatory)"
        )
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path, "wikipedia")
        cfg = self.config["sources"]["wikipedia"]
        self.sample_size: int = cfg.get("sample_articles", 500)
        self.recent_hours: int = cfg.get("recent_changes_hours", 24)

    # ── API wrapper ───────────────────────────────────────────────────────────

    def _api(self, params: Dict) -> Optional[Dict]:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        try:
            resp = requests.get(self.API, headers=self.HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.error(f"Wikipedia API error: {exc}")
            self.stats["errors"] += 1
            return None

    # ── Recent-changes harvesting ─────────────────────────────────────────────

    def fetch_recent_changes(self) -> List[Dict]:
        """Fetch all article edits in the last self.recent_hours hours."""
        start = (datetime.now(timezone.utc) - timedelta(hours=self.recent_hours))
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        changes: List[Dict] = []
        rccontinue = None

        while len(changes) < 10_000:
            params: Dict = {
                "action": "query",
                "list": "recentchanges",
                "rcstart": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rcend": start_str,
                "rcprop": "title|ids|sizes|flags|user|timestamp|comment|tags",
                "rctype": "edit|new",
                "rcnamespace": "0",
                "rclimit": "500",
            }
            if rccontinue:
                params["rccontinue"] = rccontinue

            data = self._api(params)
            if not data:
                break

            batch = data.get("query", {}).get("recentchanges", [])
            changes.extend(batch)
            self.logger.info(f"  … {len(changes):,} changes collected")

            cont = data.get("continue", {})
            rccontinue = cont.get("rccontinue")
            if not rccontinue:
                break
            self.throttle(0.3)

        self.logger.info(f"Total recent changes: {len(changes):,}")
        return changes

    def summarize_edit_patterns(self, changes: List[Dict]) -> Dict:
        """Compute aggregate edit-pattern health metrics."""
        if not changes:
            return {}

        total = len(changes)

        # Bot detection: 'bot' flag OR tag containing 'mw-'
        bot_count = sum(
            1 for c in changes
            if c.get("bot") or any("mw-" in t for t in c.get("tags", []))
        )

        # Revert detection via comment heuristics
        revert_count = sum(
            1 for c in changes
            if any(
                kw in (c.get("comment") or "").lower()
                for kw in ("revert", "rv ", "reverted", "undid")
            )
        )

        # Edit size distribution
        sizes = [abs(c.get("newlen", 0) - c.get("oldlen", 0)) for c in changes]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0

        # Unique human editors
        human_editors = {
            c["user"] for c in changes
            if not c.get("anon") and not c.get("bot") and c.get("user")
        }
        anon_count = sum(1 for c in changes if c.get("anon"))

        # Temporal clustering: edits per hour-of-day
        hour_dist: Dict[int, int] = {}
        for c in changes:
            ts = c.get("timestamp", "")
            if ts:
                try:
                    h = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").hour
                    hour_dist[h] = hour_dist.get(h, 0) + 1
                except ValueError:
                    pass

        return {
            "window_hours": self.recent_hours,
            "total_edits": total,
            "bot_edit_ratio": round(bot_count / total, 4) if total else 0,
            "revert_ratio": round(revert_count / total, 4) if total else 0,
            "avg_edit_size_bytes": round(avg_size, 1),
            "unique_human_editors": len(human_editors),
            "anon_editor_ratio": round(anon_count / total, 4) if total else 0,
            "hourly_distribution": hour_dist,
        }

    # ── Article text sampling ─────────────────────────────────────────────────

    def iter_random_titles(self, count: int) -> Iterator[str]:
        """Yield random article titles from Wikipedia's random list API."""
        fetched = 0
        while fetched < count:
            n = min(10, count - fetched)
            data = self._api({
                "action": "query",
                "list": "random",
                "rnnamespace": "0",
                "rnlimit": str(n),
            })
            if not data:
                break
            for page in data.get("query", {}).get("random", []):
                yield page["title"]
                fetched += 1
            self.throttle(0.4)

    def fetch_article(self, title: str) -> Optional[Dict]:
        """Fetch plaintext + revision metadata for one article."""
        data = self._api({
            "action": "query",
            "prop": "extracts|info|revisions",
            "titles": title,
            "explaintext": True,
            "exsectionformat": "plain",
            "exlimit": "1",
            "inprop": "url",
            "rvprop": "ids|timestamp|user|size|tags",
            "rvlimit": "5",
        })
        if not data:
            return None

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if page.get("missing"):
            return None

        text = page.get("extract", "").strip()
        if len(text) < self.config["detection"]["min_text_length"]:
            return None

        revs = page.get("revisions", [])
        return {
            "title": page.get("title", ""),
            "pageid": page.get("pageid", 0),
            "url": page.get("canonicalurl", ""),
            "text": text[: self.config["detection"]["max_text_length_for_features"]],
            "text_length": len(text),
            "category": "wiki",
            "domain": "en.wikipedia.org",
            "revisions": revs,
            "last_editor": revs[0].get("user", "") if revs else "",
            "last_edit_ts": revs[0].get("timestamp", "") if revs else "",
            "content_hash": self.content_hash(text),
        }

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        if not self.config["sources"]["wikipedia"].get("enabled", True):
            self.logger.info("Wikipedia minion disabled — exiting")
            return

        self.logger.info("🤖 Wikipedia Minion starting")
        partition = datetime.now(timezone.utc).strftime("%Y/%m/%d")

        # ── Phase 1: recent changes ──────────────────────────────────────────
        self.logger.info(f"Phase 1 — fetching recent changes ({self.recent_hours}h window)")
        changes = self.fetch_recent_changes()
        patterns = self.summarize_edit_patterns(changes)

        if changes:
            change_records = [{"type": "edit_event", **c} for c in changes]
            self.save_bronze(change_records, "wikipedia/changes", partition)
            self.stats["processed"] += len(change_records)

        self.save_bronze(
            [{"type": "edit_patterns", **patterns}],
            "wikipedia/patterns",
            partition,
        )

        # ── Phase 2: random article samples ─────────────────────────────────
        self.logger.info(f"Phase 2 — sampling {self.sample_size} random articles")
        article_batch: List[Dict] = []

        for title in self.iter_random_titles(self.sample_size):
            article = self.fetch_article(title)
            if article:
                article["type"] = "article_sample"
                article_batch.append(article)
                self.stats["fetched"] += 1
            else:
                self.stats["skipped"] += 1

            if len(article_batch) >= 100:
                self.save_bronze(article_batch, "wikipedia/articles", partition)
                self.stats["processed"] += len(article_batch)
                article_batch = []

            self.throttle(0.3)

        if article_batch:
            self.save_bronze(article_batch, "wikipedia/articles", partition)
            self.stats["processed"] += len(article_batch)

        self.report_stats()


if __name__ == "__main__":
    WikipediaMinion().run()
