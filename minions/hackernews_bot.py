"""
Hacker News Minion — harvests stories and comments via Algolia HN Search API.

Two complementary APIs, both free and unauthenticated:
  - Algolia HN API  (bulk search, date-range queries, fast)
  - Official HN Firebase API  (item-level details, comment trees)

HN is one of the highest-signal corpora for this project: the comments
are long-form, opinionated, and written by humans who strongly police
AI-generated content — making it a useful upper bound on aliveness.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Iterator, List, Optional

import requests

from .base_minion import BaseMinion


class HackerNewsMinion(BaseMinion):

    ALGOLIA_BASE  = "https://hn.algolia.com/api/v1"
    FIREBASE_BASE = "https://hacker-news.firebaseio.com/v0"
    HEADERS = {"User-Agent": "DeadInternetObservatory/1.0 (research; github.com/dead-internet-observatory)"}

    # Story types to harvest
    TAGS = ["story", "ask_hn", "show_hn", "poll"]

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path, "hackernews")
        cfg = self.config["sources"].get("hackernews", {})
        self.max_stories: int = cfg.get("max_stories", 1000)
        self.include_comments: bool = cfg.get("include_comments", True)
        self.max_comments_per_story: int = cfg.get("max_comments_per_story", 30)
        self.lookback_days: int = cfg.get("lookback_days", 7)

    # ── Algolia bulk fetcher ───────────────────────────────────────────────────

    def iter_recent_stories(self) -> Iterator[Dict]:
        """
        Yield stories from the last `lookback_days` days via Algolia.
        Algolia returns up to 1000 results per query; we paginate.
        """
        cutoff_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).timestamp()
        )
        fetched = 0
        page = 0

        self.logger.info(f"  Fetching stories (last {self.lookback_days} days) via Algolia …")

        while fetched < self.max_stories:
            try:
                resp = requests.get(
                    f"{self.ALGOLIA_BASE}/search_by_date",
                    headers=self.HEADERS,
                    params={
                        "tags": "story",
                        "numericFilters": f"created_at_i>{cutoff_ts}",
                        "hitsPerPage": 100,
                        "page": page,
                        "attributesToRetrieve": (
                            "objectID,title,url,points,author,created_at,"
                            "num_comments,story_text,_tags"
                        ),
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                self.logger.error(f"  Algolia error (page {page}): {exc}")
                self.stats["errors"] += 1
                break

            hits = data.get("hits", [])
            if not hits:
                break

            for hit in hits:
                story = self._parse_algolia_hit(hit)
                if story:
                    yield story
                    fetched += 1
                    self.stats["fetched"] += 1
                if fetched >= self.max_stories:
                    break

            # Check if more pages exist
            total_pages = data.get("nbPages", 1)
            page += 1
            if page >= total_pages:
                break

            self.throttle(0.5)

        self.logger.info(f"  Algolia: {fetched} stories fetched")

    def _parse_algolia_hit(self, hit: Dict) -> Optional[Dict]:
        title = (hit.get("title") or "").strip()
        body  = (hit.get("story_text") or "").strip()

        # Strip HTML tags from story_text (Algolia sometimes returns raw HTML)
        if body and "<" in body:
            try:
                from bs4 import BeautifulSoup
                body = BeautifulSoup(body, "lxml").get_text(separator=" ", strip=True)
            except ImportError:
                import re
                body = re.sub(r"<[^>]+>", " ", body).strip()

        text = f"{title}\n\n{body}".strip() if body else title
        if not text:
            return None

        created_ts = hit.get("created_at_i", 0)
        return {
            "id": hit.get("objectID", ""),
            "title": title,
            "text": text[:3000],
            "text_length": len(text),
            "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID','')}",
            "hn_url": f"https://news.ycombinator.com/item?id={hit.get('objectID','')}",
            "author": hit.get("author", ""),
            "points": hit.get("points", 0),
            "num_comments": hit.get("num_comments", 0),
            "created_utc": created_ts,
            "created_dt": (
                datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat()
                if created_ts else None
            ),
            "tags": hit.get("_tags", []),
            "domain": self._url_domain(hit.get("url", "")),
            "category": "social",
            "content_hash": self.content_hash(text),
            "comments": [],
        }

    # ── Firebase comment fetcher ───────────────────────────────────────────────

    def fetch_comments(self, story_id: str, kids: List[int]) -> List[Dict]:
        """Fetch top-level comments for a story via the Firebase API."""
        if not kids:
            return []

        comments: List[Dict] = []
        for kid_id in kids[: self.max_comments_per_story]:
            comment = self._fetch_item(kid_id)
            if comment:
                comments.append(comment)
            self.throttle(0.2)

        return comments

    def _fetch_item(self, item_id: int) -> Optional[Dict]:
        try:
            resp = requests.get(
                f"{self.FIREBASE_BASE}/item/{item_id}.json",
                headers=self.HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            item = resp.json()
        except Exception as exc:
            self.logger.debug(f"  Firebase item {item_id} error: {exc}")
            return None

        if not item or item.get("deleted") or item.get("dead"):
            return None

        text = (item.get("text") or "").strip()
        if not text or len(text) < 15:
            return None

        # Strip HTML
        if "<" in text:
            try:
                from bs4 import BeautifulSoup
                text = BeautifulSoup(text, "lxml").get_text(separator=" ", strip=True)
            except ImportError:
                import re
                text = re.sub(r"<[^>]+>", " ", text).strip()

        created = item.get("time", 0)
        return {
            "id": str(item.get("id", "")),
            "text": text[:2000],
            "author": item.get("by", ""),
            "score": item.get("score", 0),
            "created_utc": created,
            "created_dt": (
                datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                if created else None
            ),
            "content_hash": self.content_hash(text),
        }

    def _fetch_story_kids(self, story_id: str) -> List[int]:
        """Get the list of direct child comment IDs for a story."""
        try:
            resp = requests.get(
                f"{self.FIREBASE_BASE}/item/{story_id}.json",
                headers=self.HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            item = resp.json()
            return (item or {}).get("kids", [])
        except Exception:
            return []

    # ── Algolia comment search (bulk alternative to Firebase) ─────────────────

    def fetch_comments_algolia(self, story_id: str) -> List[Dict]:
        """
        Fetch comments for a story via Algolia — faster than Firebase
        for high-comment-count stories.
        """
        try:
            resp = requests.get(
                f"{self.ALGOLIA_BASE}/search",
                headers=self.HEADERS,
                params={
                    "tags": f"comment,story_{story_id}",
                    "hitsPerPage": self.max_comments_per_story,
                    "attributesToRetrieve": "objectID,comment_text,author,created_at_i,points",
                },
                timeout=20,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            self.logger.debug(f"  Algolia comment fetch error for {story_id}: {exc}")
            return []

        comments = []
        for hit in hits:
            text = (hit.get("comment_text") or "").strip()
            if not text or len(text) < 15:
                continue
            if "<" in text:
                try:
                    from bs4 import BeautifulSoup
                    text = BeautifulSoup(text, "lxml").get_text(separator=" ", strip=True)
                except ImportError:
                    import re
                    text = re.sub(r"<[^>]+>", " ", text).strip()

            created = hit.get("created_at_i", 0)
            comments.append({
                "id": hit.get("objectID", ""),
                "text": text[:2000],
                "author": hit.get("author", ""),
                "score": hit.get("points", 0),
                "created_utc": created,
                "created_dt": (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created else None
                ),
                "content_hash": self.content_hash(text),
            })

        return comments

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _url_domain(url: str) -> str:
        if not url:
            return "news.ycombinator.com"
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lstrip("www.") or "news.ycombinator.com"
        except Exception:
            return ""

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        if not self.config["sources"].get("hackernews", {}).get("enabled", True):
            self.logger.info("HackerNews minion disabled — exiting")
            return

        self.logger.info(
            f"🤖 HackerNews Minion starting | "
            f"max_stories={self.max_stories} | lookback={self.lookback_days}d"
        )

        partition = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        batch: List[Dict] = []

        for story in self.iter_recent_stories():
            if self.include_comments and story["num_comments"] > 0:
                story["comments"] = self.fetch_comments_algolia(story["id"])
                self.throttle(0.5)

            batch.append(story)
            self.stats["processed"] += 1

            if len(batch) >= 200:
                self.save_bronze(batch, "hackernews", partition)
                batch = []

        if batch:
            self.save_bronze(batch, "hackernews", partition)

        self.report_stats()


if __name__ == "__main__":
    HackerNewsMinion().run()
