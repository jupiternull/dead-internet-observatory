"""
News Crawler Minion — harvests articles from RSS/Atom feeds.

Parses each configured feed, fetches the linked article URL,
extracts the article body with BeautifulSoup, and saves to bronze.
The extraction pipeline tries semantic tags first (article, main,
[itemprop="articleBody"]) then falls back to paragraph aggregation.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

from .base_minion import BaseMinion


class NewsCrawlerMinion(BaseMinion):

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0 "
            "DeadInternetObservatory/1.0"
        )
    }

    # Selectors tried in order for article body extraction
    BODY_SELECTORS = [
        "article",
        '[itemprop="articleBody"]',
        ".article-body",
        ".story-body",
        ".article__body",
        ".post-content",
        ".entry-content",
        "main",
        "#main-content",
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path, "news_crawler")
        cfg = self.config["sources"]["news"]
        self.feeds: List[str] = cfg["feeds"]
        self.max_articles: int = cfg["max_articles_per_feed"]
        self.timeout: int = cfg.get("request_timeout", 20)
        self.min_text: int = self.config["detection"]["min_text_length"]

    # ── Feed parsing ──────────────────────────────────────────────────────────

    def parse_feed(self, feed_url: str) -> List[Dict]:
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.get("entries", [])
            self.logger.info(f"  Feed {feed_url.split('/')[2]}: {len(entries)} entries")
            return entries[: self.max_articles]
        except Exception as exc:
            self.logger.error(f"Feed parse error {feed_url}: {exc}")
            self.stats["errors"] += 1
            return []

    # ── Article extraction ────────────────────────────────────────────────────

    def extract_article(self, url: str) -> Optional[Dict]:
        try:
            resp = requests.get(
                url, headers=self.HEADERS, timeout=self.timeout,
                allow_redirects=True, stream=False,
            )
            resp.raise_for_status()
            # Bail if not HTML
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct and "text" not in ct:
                return None

        except requests.Timeout:
            self.logger.warning(f"  Timeout: {url}")
            self.stats["errors"] += 1
            return None
        except Exception as exc:
            self.logger.error(f"  Fetch error {url}: {exc}")
            self.stats["errors"] += 1
            return None

        soup = BeautifulSoup(resp.content, "lxml")

        # Strip noise
        for noise in soup(["script", "style", "nav", "footer", "header",
                           "aside", "figure", "form", "noscript", "iframe",
                           ".ad", ".advertisement", ".sidebar", ".related"]):
            noise.decompose()

        # Body extraction
        body_text = ""
        for selector in self.BODY_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                body_text = elem.get_text(separator="\n", strip=True)
                if len(body_text) >= self.min_text:
                    break

        # Paragraph fallback
        if len(body_text) < self.min_text:
            paras = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
            body_text = "\n".join(paras)

        if len(body_text) < self.min_text:
            self.stats["skipped"] += 1
            return None

        title = self._extract_title(soup)
        pub_date = self._extract_pub_date(soup)
        domain = urlparse(url).netloc.lower().lstrip("www.")

        return {
            "url": url,
            "domain": domain,
            "title": title[:300],
            "text": body_text[: self.config["detection"]["max_text_length_for_features"]],
            "text_length": len(body_text),
            "pub_date": pub_date,
            "content_hash": self.content_hash(body_text),
        }

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        # og:title is usually cleaner than <title>
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
        tag = soup.find("h1") or soup.find("title")
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _extract_pub_date(soup: BeautifulSoup) -> Optional[str]:
        candidates = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"property": "og:article:published_time"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"name": "date"}),
            ("meta", {"name": "publish-date"}),
            ("time", {"itemprop": "datePublished"}),
        ]
        for tag, attrs in candidates:
            elem = soup.find(tag, attrs)
            if elem:
                return elem.get("content") or elem.get("datetime") or None
        return None

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        if not self.config["sources"]["news"].get("enabled", True):
            self.logger.info("News crawler minion disabled — exiting")
            return

        self.logger.info(f"🤖 News Crawler Minion starting | feeds={len(self.feeds)}")
        partition = datetime.now(timezone.utc).strftime("%Y/%m/%d")

        for feed_url in self.feeds:
            entries = self.parse_feed(feed_url)
            batch: List[Dict] = []

            for entry in entries:
                url = entry.get("link", "").strip()
                if not url:
                    continue

                self.stats["fetched"] += 1
                article = self.extract_article(url)
                if not article:
                    continue

                # Merge feed-level metadata
                article.update({
                    "feed_url": feed_url,
                    "feed_published": entry.get("published", ""),
                    "feed_summary": (entry.get("summary") or "")[:500],
                    "feed_tags": [t.get("term", "") for t in entry.get("tags", [])],
                    "category": "news",
                })

                batch.append(article)
                self.stats["processed"] += 1
                self.throttle(0.5)

            if batch:
                self.save_bronze(batch, "news", partition)

            self.throttle(1.0)

        self.report_stats()


if __name__ == "__main__":
    NewsCrawlerMinion().run()
