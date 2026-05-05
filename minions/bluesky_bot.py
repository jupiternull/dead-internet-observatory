"""
Bluesky Minion — scrapes public AT Protocol posts via the AppView API.
No authentication required.
"""

import sys
from datetime import datetime, timezone

import requests

from minions.base_minion import BaseMinion


class BlueskyBot(BaseMinion):

    BASE_URL = "https://api.bsky.app/xrpc"

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="bluesky")
        cfg = self.config.get("sources", {}).get("bluesky", {})
        self.max_posts: int = cfg.get("max_posts", 1000)
        self.search_terms: list = cfg.get(
            "search_terms", ["technology", "news", "science", "culture"]
        )
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        })

    # ── Fetch helpers ──────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _extract_post(self, post: dict) -> dict | None:
        record = post.get("record", {})
        text = record.get("text", "").strip()
        if not text:
            return None
        langs = record.get("langs", [])
        return {
            "uri":           post.get("uri", ""),
            "cid":           post.get("cid", ""),
            "author_did":    post.get("author", {}).get("did", ""),
            "author_handle": post.get("author", {}).get("handle", ""),
            "text":          text,
            "created_at":    record.get("createdAt", ""),
            "like_count":    post.get("likeCount", 0),
            "repost_count":  post.get("repostCount", 0),
            "reply_count":   post.get("replyCount", 0),
            "langs":         ",".join(langs) if isinstance(langs, list) else str(langs),
            "category":      "social",
        }

    def _search_posts(self, term: str, limit: int = 100) -> list[dict]:
        try:
            data = self._get(
                "app.bsky.feed.searchPosts",
                {"q": term, "limit": min(limit, 100)},
            )
            posts = data.get("posts", [])
            records = [self._extract_post(p) for p in posts]
            return [r for r in records if r]
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"Search '{term}' failed: {exc}")
            self.stats["errors"] += 1
            return []

    def _fetch_whats_hot(self, limit: int = 100) -> list[dict]:
        try:
            data = self._get(
                "app.bsky.feed.getFeed",
                {
                    "feed": "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.whats-hot",
                    "limit": min(limit, 100),
                },
            )
            posts = [item.get("post", {}) for item in data.get("feed", [])]
            records = [self._extract_post(p) for p in posts]
            return [r for r in records if r]
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"What's Hot fetch failed: {exc}")
            self.stats["errors"] += 1
            return []

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting — target {self.max_posts} posts")
        seen_uris: set[str] = set()
        all_records: list[dict] = []

        for term in self.search_terms:
            if len(all_records) >= self.max_posts:
                break
            remaining = self.max_posts - len(all_records)
            results = self._search_posts(term, min(remaining, 100))
            new_count = 0
            for rec in results:
                if rec["uri"] not in seen_uris:
                    seen_uris.add(rec["uri"])
                    all_records.append(rec)
                    new_count += 1
            self.logger.info(f"  Term '{term}': {new_count} new posts")
            self.stats["fetched"] += len(results)
            self.stats["skipped"] += len(results) - new_count
            self.throttle(0.5)

        if all_records:
            self.save_bronze(all_records, source="bluesky")
            self.stats["processed"] = len(all_records)

        self.logger.info(f"Done — {len(all_records)} unique posts saved")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = BlueskyBot(config_path=config_path)
    bot.run()
    bot.report_stats()
