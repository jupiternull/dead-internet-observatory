"""
4chan Minion — scrapes public board content via the 4chan JSON API.
No authentication required.
"""

import html
import random
import re
import sys

import requests

from minions.base_minion import BaseMinion


def clean_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


class FourchanBot(BaseMinion):

    CDN = "https://a.4cdn.org"

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="fourchan")
        cfg = self.config.get("sources", {}).get("fourchan", {})
        self.boards: list = cfg.get("boards", ["g", "sci", "tv", "lit", "biz", "int"])
        self.threads_per_board: int = cfg.get("threads_per_board", 15)
        self.posts_per_thread: int = cfg.get("posts_per_thread", 50)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DeadInternetObservatory/1.0 (research)",
        })

    # ── Fetch helpers ──────────────────────────────────────────────────────────

    def _get_json(self, url: str) -> dict | list | None:
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"Request failed {url}: {exc}")
            self.stats["errors"] += 1
            return None

    def _catalog_threads(self, board: str) -> list[dict]:
        data = self._get_json(f"{self.CDN}/{board}/catalog.json")
        if not data:
            return []
        threads = []
        for page in data:
            for t in page.get("threads", []):
                threads.append({
                    "thread_no": t["no"],
                    "subject":   t.get("sub", ""),
                    "op_text":   clean_html(t.get("com", "")),
                    "replies":   t.get("replies", 0),
                })
        return threads

    def _fetch_thread(self, board: str, thread_no: int) -> list[dict]:
        data = self._get_json(f"{self.CDN}/{board}/thread/{thread_no}.json")
        if not data:
            return []
        posts = data.get("posts", [])
        records = []
        for post in posts[: self.posts_per_thread]:
            text = clean_html(post.get("com", ""))
            if not text:
                continue
            records.append({
                "board":        board,
                "thread_no":    thread_no,
                "post_no":      post["no"],
                "is_op":        post["no"] == thread_no,
                "subject":      clean_html(post.get("sub", "")),
                "text":         text,
                "timestamp":    post.get("time", 0),
                "country":      post.get("country", ""),
                "filename":     post.get("filename", "") + post.get("ext", "")
                                if "filename" in post else "",
                "replies_count": post.get("replies", 0),
                "category":     "forum",
            })
        return records

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting — boards={self.boards}")
        for board in self.boards:
            self.logger.info(f"  Board /{board}/")
            threads = self._catalog_threads(board)
            if not threads:
                self.logger.warning(f"  No threads for /{board}/")
                continue

            sample_size = min(self.threads_per_board, len(threads))
            sampled = random.sample(threads, sample_size)
            self.stats["fetched"] += len(threads)
            self.throttle(0.5)

            board_records: list[dict] = []
            for thread in sampled:
                posts = self._fetch_thread(board, thread["thread_no"])
                board_records.extend(posts)
                self.stats["processed"] += len(posts)
                self.throttle(0.5)

            if board_records:
                self.save_bronze(board_records, source="fourchan",
                                 partition=None)
                self.logger.info(f"  /{board}/: {len(board_records)} posts saved")

        self.logger.info("Done")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = FourchanBot(config_path=config_path)
    bot.run()
    bot.report_stats()
