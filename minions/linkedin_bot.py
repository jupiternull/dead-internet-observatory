"""
LinkedIn Minion — scrapes public LinkedIn Pulse articles via RSS and Google News.
No authentication required.
"""

import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

from minions.base_minion import BaseMinion


PULSE_FEEDS = [
    "https://news.google.com/rss/search?q=site:linkedin.com/pulse&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=linkedin+pulse+professional&hl=en-US&gl=US&ceid=US:en",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DeadInternetObservatory/1.0; research)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class LinkedinBot(BaseMinion):

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="linkedin")
        cfg = self.config.get("sources", {}).get("linkedin", {})
        self.max_articles: int = cfg.get("max_articles", 100)
        self.request_timeout: int = cfg.get("request_timeout", 20)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── Fetch helpers ──────────────────────────────────────────────────────────

    def _collect_urls(self) -> list[tuple[str, str]]:
        """Return list of (url, feed_name) from RSS feeds."""
        urls: list[tuple[str, str]] = []
        for feed_url in PULSE_FEEDS:
            try:
                parsed = feedparser.parse(feed_url)
                for entry in parsed.entries:
                    link = entry.get("link", "")
                    if link and "linkedin.com" in link:
                        urls.append((link, feed_url))
            except Exception as exc:
                self.logger.warning(f"  Feed parse failed {feed_url}: {exc}")
                self.stats["errors"] += 1
        return urls

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for selector in ["article", ".reader-article-content", ".prose",
                         '[class*="article"]']:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) >= 100:
                    return text
        meta = soup.find("meta", property="og:description")
        if meta:
            return meta.get("content", "").strip()
        return ""

    def _fetch_article(self, url: str, feed_name: str) -> dict | None:
        try:
            resp = self.session.get(url, timeout=self.request_timeout, allow_redirects=True)
            if resp.status_code in (403, 429, 999):
                self.logger.warning(f"  Blocked ({resp.status_code}): {url}")
                self.stats["skipped"] += 1
                return None
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"  Request failed {url}: {exc}")
            self.stats["errors"] += 1
            return None

        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")

        text = self._extract_text(soup)
        if not text or len(text) < 100:
            self.stats["skipped"] += 1
            return None

        title_tag = soup.find("meta", property="og:title") or soup.find("h1")
        title = ""
        if title_tag:
            title = title_tag.get("content", "") or title_tag.get_text(strip=True)

        return {
            "url":         url,
            "title":       title,
            "text":        text[:5000],
            "published_at": "",
            "source_feed": feed_name,
            "scraped_at":  datetime.now(timezone.utc).isoformat(),
            "category":    "professional",
        }

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("Starting — collecting LinkedIn Pulse article URLs")
        url_pairs = self._collect_urls()
        self.logger.info(f"  Found {len(url_pairs)} candidate URLs")
        self.stats["fetched"] = len(url_pairs)

        seen: set[str] = set()
        records: list[dict] = []

        for url, feed_name in url_pairs:
            if len(records) >= self.max_articles:
                break
            if url in seen:
                self.stats["skipped"] += 1
                continue
            seen.add(url)

            article = self._fetch_article(url, feed_name)
            if article:
                records.append(article)
                self.stats["processed"] += 1
                self.logger.info(f"  Saved: {article['title'][:60]}")
            self.throttle(2.0)

        if records:
            self.save_bronze(records, source="linkedin")

        self.logger.info(f"Done — {len(records)} articles saved")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = LinkedinBot(config_path=config_path)
    bot.run()
    bot.report_stats()
