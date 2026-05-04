"""
Reddit Minion — harvests posts and comments via Reddit's public JSON API.

No OAuth required: appending .json to any Reddit URL returns raw data.
We iterate through /new.json with pagination to collect recent content.
Optionally fetches top-level comment trees per post.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

import requests

from .base_minion import BaseMinion


class RedditMinion(BaseMinion):

    REDDIT_BASE = "https://www.reddit.com"
    HEADERS = {
        "User-Agent": "DeadInternetObservatory/1.0 (academic research; github.com/dead-internet-observatory)"
    }
    DELETED = frozenset({"[deleted]", "[removed]", ""})

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path, "reddit")
        cfg = self.config["sources"]["reddit"]
        self.subreddits: List[str] = cfg["subreddits"]
        self.posts_per_sub: int = cfg["posts_per_sub"]
        self.include_comments: bool = cfg.get("include_comments", True)
        self.max_comments: int = cfg.get("max_comments_per_post", 50)

    # ── API helpers ───────────────────────────────────────────────────────────

    def _get(self, url: str, params: Dict) -> Optional[Dict]:
        params["raw_json"] = 1
        try:
            resp = requests.get(url, headers=self.HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                self.logger.warning(f"Rate-limited — sleeping {retry_after}s")
                time.sleep(retry_after)
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            self.logger.error(f"HTTP error: {exc}")
        except Exception as exc:
            self.logger.error(f"Request error: {exc}")
            self.stats["errors"] += 1
        return None

    # ── Post harvesting ───────────────────────────────────────────────────────

    def iter_subreddit_posts(self, sub: str) -> Iterator[Dict]:
        """Paginate through /r/{sub}/new, yielding structured post records."""
        fetched = 0
        after: Optional[str] = None
        self.logger.info(f"  Harvesting r/{sub} …")

        while fetched < self.posts_per_sub:
            batch = min(100, self.posts_per_sub - fetched)
            params: Dict = {"limit": batch}
            if after:
                params["after"] = after

            data = self._get(f"{self.REDDIT_BASE}/r/{sub}/new.json", params)
            if not data or "data" not in data:
                break

            children = data["data"].get("children", [])
            if not children:
                break

            for child in children:
                if child.get("kind") != "t3":
                    continue
                post = self._parse_post(child["data"], sub)
                if post:
                    yield post
                    fetched += 1
                    self.stats["fetched"] += 1

            after = data["data"].get("after")
            if not after:
                break

            self.throttle(2.0)

    def _parse_post(self, pd: Dict, sub: str) -> Optional[Dict]:
        selftext = pd.get("selftext", "").strip()
        title = pd.get("title", "").strip()
        if selftext in self.DELETED:
            selftext = ""

        text = f"{title}\n\n{selftext}".strip() if selftext else title
        if not text:
            return None

        created_utc = pd.get("created_utc", 0)
        created_dt = (
            datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
            if created_utc else None
        )

        record: Dict = {
            "id": pd.get("id", ""),
            "title": title,
            "text": text[:3000],
            "text_length": len(text),
            "url": pd.get("url", ""),
            "permalink": f"https://reddit.com{pd.get('permalink', '')}",
            "author": pd.get("author", "[unknown]"),
            "subreddit": sub,
            "score": pd.get("score", 0),
            "upvote_ratio": pd.get("upvote_ratio", 0.0),
            "num_comments": pd.get("num_comments", 0),
            "created_utc": created_utc,
            "created_dt": created_dt,
            "flair": pd.get("link_flair_text") or "",
            "is_self": pd.get("is_self", False),
            "domain": pd.get("domain", ""),
            "gilded": pd.get("gilded", 0),
            "content_hash": self.content_hash(text),
            "comments": [],
        }

        if self.include_comments and pd.get("num_comments", 0) > 0:
            record["comments"] = self._fetch_comments(sub, pd["id"])
            self.throttle(1.0)

        return record

    # ── Comment harvesting ────────────────────────────────────────────────────

    def _fetch_comments(self, sub: str, post_id: str) -> List[Dict]:
        url = f"{self.REDDIT_BASE}/r/{sub}/comments/{post_id}.json"
        data = self._get(url, {"limit": 100, "depth": 3})
        if not data or len(data) < 2:
            return []
        comments: List[Dict] = []
        self._extract_comments(data[1]["data"].get("children", []), comments, 0)
        return comments[: self.max_comments]

    def _extract_comments(self, children: List[Dict], out: List[Dict], depth: int):
        if depth > 4:
            return
        for child in children:
            if child.get("kind") != "t1":
                continue
            cd = child.get("data", {})
            body = cd.get("body", "").strip()
            if body and body not in self.DELETED and len(body) > 10:
                created = cd.get("created_utc", 0)
                out.append({
                    "id": cd.get("id", ""),
                    "text": body[:2000],
                    "author": cd.get("author", "[unknown]"),
                    "score": cd.get("score", 0),
                    "created_utc": created,
                    "created_dt": (
                        datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                        if created else None
                    ),
                    "depth": depth,
                    "controversiality": cd.get("controversiality", 0),
                    "content_hash": self.content_hash(body),
                })
            replies = cd.get("replies", {})
            if isinstance(replies, dict):
                self._extract_comments(
                    replies.get("data", {}).get("children", []),
                    out, depth + 1,
                )

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        if not self.config["sources"]["reddit"].get("enabled", True):
            self.logger.info("Reddit minion disabled — exiting")
            return

        self.logger.info(
            f"🤖 Reddit Minion starting | subs={len(self.subreddits)} | "
            f"posts_per_sub={self.posts_per_sub}"
        )

        partition = datetime.now(timezone.utc).strftime("%Y/%m/%d")

        for sub in self.subreddits:
            batch: List[Dict] = []
            for post in self.iter_subreddit_posts(sub):
                batch.append(post)
                if len(batch) >= 200:
                    self.save_bronze(batch, "reddit", partition)
                    self.stats["processed"] += len(batch)
                    batch = []
            if batch:
                self.save_bronze(batch, "reddit", partition)
                self.stats["processed"] += len(batch)

        self.report_stats()


if __name__ == "__main__":
    RedditMinion().run()
