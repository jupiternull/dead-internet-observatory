"""
YouTube Minion — scrapes public video comments via YouTube Data API v3.
Requires YOUTUBE_API_KEY environment variable.
"""

import os
import sys

import requests

from minions.base_minion import BaseMinion


DEFAULT_TOPICS = [
    "technology",
    "artificial intelligence",
    "world news",
    "gaming",
    "science",
]


class YoutubeBot(BaseMinion):

    API_BASE = "https://www.googleapis.com/youtube/v3"

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="youtube")
        self.api_key = os.environ.get("YOUTUBE_API_KEY", "")
        self.disabled = not bool(self.api_key)
        if self.disabled:
            self.logger.warning("YOUTUBE_API_KEY not set — YouTube minion will produce no data")

        cfg = self.config.get("sources", {}).get("youtube", {})
        self.search_topics: list  = cfg.get("search_topics", DEFAULT_TOPICS)
        self.videos_per_topic: int = cfg.get("videos_per_topic", 10)
        self.comments_per_video: int = cfg.get("comments_per_video", 100)
        self.session = requests.Session()

    # ── Fetch helpers ──────────────────────────────────────────────────────────

    def _search_videos(self, topic: str) -> list[dict]:
        try:
            resp = self.session.get(f"{self.API_BASE}/search", params={
                "part":              "snippet",
                "q":                 topic,
                "type":              "video",
                "maxResults":        self.videos_per_topic,
                "order":             "date",
                "relevanceLanguage": "en",
                "key":               self.api_key,
            }, timeout=30)
            if resp.status_code in (403, 400):
                self.logger.warning(f"  Search '{topic}': HTTP {resp.status_code}")
                self.stats["errors"] += 1
                return []
            resp.raise_for_status()
            items = resp.json().get("items", [])
            return [
                {
                    "video_id":      item["id"]["videoId"],
                    "video_title":   item["snippet"]["title"],
                    "channel_id":    item["snippet"]["channelId"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at":  item["snippet"]["publishedAt"],
                }
                for item in items
                if item.get("id", {}).get("videoId")
            ]
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"  Search '{topic}' failed: {exc}")
            self.stats["errors"] += 1
            return []

    def _fetch_comments(self, video: dict) -> list[dict]:
        try:
            resp = self.session.get(f"{self.API_BASE}/commentThreads", params={
                "part":       "snippet",
                "videoId":    video["video_id"],
                "maxResults": min(self.comments_per_video, 100),
                "order":      "relevance",
                "textFormat": "plainText",
                "key":        self.api_key,
            }, timeout=30)
            if resp.status_code in (403, 400):
                self.stats["skipped"] += 1
                return []
            resp.raise_for_status()
            items = resp.json().get("items", [])
            records = []
            for item in items:
                tl = item.get("snippet", {}).get("topLevelComment", {})
                snip = tl.get("snippet", {})
                text = snip.get("textOriginal", "").strip()
                if not text:
                    continue
                records.append({
                    "video_id":        video["video_id"],
                    "video_title":     video["video_title"],
                    "channel_id":      video["channel_id"],
                    "channel_title":   video["channel_title"],
                    "comment_id":      tl.get("id", ""),
                    "comment_text":    text,
                    "author_channel_id": snip.get("authorChannelId", {}).get("value", ""),
                    "like_count":      snip.get("likeCount", 0),
                    "published_at":    snip.get("publishedAt", ""),
                    "reply_count":     item.get("snippet", {}).get("totalReplyCount", 0),
                    "category":        "social",
                })
            return records
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"  Comments for {video['video_id']} failed: {exc}")
            self.stats["errors"] += 1
            return []

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        if self.disabled:
            self.logger.warning("Skipping — no API key")
            return

        self.logger.info(f"Starting — {len(self.search_topics)} topics")
        all_records: list[dict] = []

        for topic in self.search_topics:
            videos = self._search_videos(topic)
            self.logger.info(f"  Topic '{topic}': {len(videos)} videos")
            self.stats["fetched"] += len(videos)
            self.throttle(0.3)

            for video in videos:
                comments = self._fetch_comments(video)
                all_records.extend(comments)
                self.stats["processed"] += len(comments)
                self.throttle(0.3)

        if all_records:
            self.save_bronze(all_records, source="youtube")

        self.logger.info(f"Done — {len(all_records)} comments saved")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = YoutubeBot(config_path=config_path)
    bot.run()
    bot.report_stats()
