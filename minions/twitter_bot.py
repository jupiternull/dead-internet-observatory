"""
Twitter/X Minion — scrapes public posts via official API v2 (if bearer token set)
or Nitter RSS as fallback.
"""

import hashlib
import os
import sys

import feedparser
import requests
from bs4 import BeautifulSoup

from minions.base_minion import BaseMinion


NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
]

DEFAULT_TERMS = [
    "technology",
    "artificial intelligence",
    "breaking news",
    "science",
    "world news",
]


class TwitterBot(BaseMinion):

    API_V2 = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="twitter")
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        self.use_official_api = bool(self.bearer_token)
        cfg = self.config.get("sources", {}).get("twitter", {})
        self.search_terms: list = cfg.get("search_terms", DEFAULT_TERMS)
        self.max_tweets: int = cfg.get("max_tweets", 500)
        self.session = requests.Session()

    # ── Official API ──────────────────────────────────────────────────────────

    def _official_search(self, term: str, max_results: int = 100) -> list[dict]:
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        try:
            resp = self.session.get(self.API_V2, headers=headers, params={
                "query":        f"{term} lang:en -is:retweet",
                "max_results":  min(max_results, 100),
                "tweet.fields": "created_at,author_id,public_metrics,lang,text,source",
            }, timeout=30)
            if resp.status_code == 429:
                self.logger.warning("  Official API: rate limited — stopping")
                return []
            if resp.status_code == 403:
                self.logger.warning(f"  Official API: 403 for '{term}'")
                self.stats["errors"] += 1
                return []
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"  Official API request failed: {exc}")
            self.stats["errors"] += 1
            return []

        records = []
        for tweet in data:
            metrics = tweet.get("public_metrics", {})
            records.append({
                "tweet_id":      tweet["id"],
                "text":          tweet.get("text", ""),
                "author_id":     tweet.get("author_id", ""),
                "created_at":    tweet.get("created_at", ""),
                "like_count":    metrics.get("like_count", 0),
                "retweet_count": metrics.get("retweet_count", 0),
                "reply_count":   metrics.get("reply_count", 0),
                "lang":          tweet.get("lang", ""),
                "source_app":    tweet.get("source", ""),
                "is_nitter":     False,
                "search_term":   term,
                "category":      "social",
            })
        return records

    # ── Nitter fallback ───────────────────────────────────────────────────────

    def _find_nitter(self) -> str | None:
        for instance in NITTER_INSTANCES:
            try:
                resp = self.session.get(instance, timeout=10)
                if resp.status_code == 200:
                    return instance
            except requests.exceptions.RequestException:
                continue
        return None

    def _nitter_search(self, instance: str, term: str) -> list[dict]:
        feed_url = f"{instance}/search/rss?q={requests.utils.quote(term)}&f=tweets"
        try:
            parsed = feedparser.parse(feed_url)
            if not parsed.entries:
                return []
        except Exception as exc:
            self.logger.warning(f"  Nitter feed failed {feed_url}: {exc}")
            self.stats["errors"] += 1
            return []

        records = []
        for entry in parsed.entries:
            raw_text = entry.get("summary", "") or entry.get("title", "")
            text = BeautifulSoup(raw_text, "html.parser").get_text(" ", strip=True)
            if not text:
                continue
            uid = hashlib.sha1(
                (entry.get("link", "") + text[:40]).encode()
            ).hexdigest()[:16]
            records.append({
                "tweet_id":      uid,
                "text":          text,
                "author_id":     entry.get("author", ""),
                "created_at":    entry.get("published", ""),
                "like_count":    0,
                "retweet_count": 0,
                "reply_count":   0,
                "lang":          "en",
                "source_app":    "nitter",
                "is_nitter":     True,
                "search_term":   term,
                "category":      "social",
            })
        return records

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        all_records: list[dict] = []

        if self.use_official_api:
            self.logger.info("Using official Twitter API v2")
            for term in self.search_terms:
                if len(all_records) >= self.max_tweets:
                    break
                remaining = self.max_tweets - len(all_records)
                results = self._official_search(term, min(remaining, 100))
                self.logger.info(f"  '{term}': {len(results)} tweets")
                all_records.extend(results)
                self.stats["fetched"] += len(results)
                self.throttle(0.5)
        else:
            self.logger.info("No bearer token — trying Nitter RSS fallback")
            instance = self._find_nitter()
            if not instance:
                self.logger.warning("All Nitter instances unavailable — no Twitter data this run")
                return
            self.logger.info(f"  Using Nitter instance: {instance}")
            for term in self.search_terms:
                if len(all_records) >= self.max_tweets:
                    break
                results = self._nitter_search(instance, term)
                self.logger.info(f"  '{term}': {len(results)} tweets (nitter)")
                all_records.extend(results)
                self.stats["fetched"] += len(results)
                self.throttle(0.5)

        self.stats["processed"] = len(all_records)
        if all_records:
            self.save_bronze(all_records, source="twitter")

        self.logger.info(f"Done — {len(all_records)} tweets saved")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = TwitterBot(config_path=config_path)
    bot.run()
    bot.report_stats()
