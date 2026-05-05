"""
Bronze → Silver ingestion pipeline.

Reads raw JSONL files from the bronze layer, normalises schema,
deduplicates by content_hash, and writes a clean Parquet table
per source per date to the silver layer.

Can run as plain pandas (local/Colab) or swap in PySpark
by uncommenting the marked sections for Databricks/large scale.
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd
import yaml


# ── Unified schema for silver layer ─────────────────────────────────────────

SILVER_COLUMNS = [
    "doc_id",           # sha256 of (source + url/id + text[:64])
    "source",           # common_crawl | reddit | news | wikipedia
    "category",         # web | social | news | wiki | blog
    "domain",
    "url",
    "title",
    "text",
    "text_length",
    "author",
    "created_dt",       # ISO8601 string (nullable)
    "crawl_partition",  # YYYY/MM or YYYY/WW for CC
    "ingested_at",
    "content_hash",
]


def _make_doc_id(source: str, identifier: str, text_prefix: str) -> str:
    raw = f"{source}:{identifier}:{text_prefix[:64]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Record normalisers per source ─────────────────────────────────────────────

def _normalise_common_crawl(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    url = raw.get("url", "")
    return {
        "doc_id": _make_doc_id("common_crawl", url, text),
        "source": "common_crawl",
        "category": raw.get("category", "web"),
        "domain": raw.get("domain", ""),
        "url": url,
        "title": "",
        "text": text,
        "text_length": len(text),
        "author": "",
        "created_dt": raw.get("date"),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("content_hash", ""),
    }


def _normalise_reddit(raw: Dict, partition: str) -> List[Dict]:
    """Returns post + flattened comments as separate documents."""
    docs: List[Dict] = []

    text = (raw.get("text") or raw.get("title") or "").strip()
    if text:
        docs.append({
            "doc_id": _make_doc_id("reddit", raw.get("id", ""), text),
            "source": "reddit",
            "category": "social",
            "domain": "reddit.com",
            "url": raw.get("permalink", ""),
            "title": raw.get("title", ""),
            "text": text,
            "text_length": len(text),
            "author": raw.get("author", ""),
            "created_dt": raw.get("created_dt"),
            "crawl_partition": partition,
            "ingested_at": raw.get("_ingested_at", ""),
            "content_hash": raw.get("content_hash", ""),
        })

    for comment in raw.get("comments", []):
        ct = (comment.get("text") or "").strip()
        if ct and len(ct) > 15:
            docs.append({
                "doc_id": _make_doc_id("reddit_comment", comment.get("id", ""), ct),
                "source": "reddit",
                "category": "social",
                "domain": "reddit.com",
                "url": raw.get("permalink", ""),
                "title": "",
                "text": ct,
                "text_length": len(ct),
                "author": comment.get("author", ""),
                "created_dt": comment.get("created_dt"),
                "crawl_partition": partition,
                "ingested_at": raw.get("_ingested_at", ""),
                "content_hash": comment.get("content_hash", ""),
            })

    return docs


def _normalise_news(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    url = raw.get("url", "")
    return {
        "doc_id": _make_doc_id("news", url, text),
        "source": "news",
        "category": "news",
        "domain": raw.get("domain", ""),
        "url": url,
        "title": raw.get("title", ""),
        "text": text,
        "text_length": len(text),
        "author": "",
        "created_dt": raw.get("pub_date") or raw.get("feed_published"),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("content_hash", ""),
    }


def _normalise_wikipedia(raw: Dict, partition: str) -> Optional[Dict]:
    if raw.get("type") != "article_sample":
        return None
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    return {
        "doc_id": _make_doc_id("wikipedia", str(raw.get("pageid", "")), text),
        "source": "wikipedia",
        "category": "wiki",
        "domain": "en.wikipedia.org",
        "url": raw.get("url", ""),
        "title": raw.get("title", ""),
        "text": text,
        "text_length": len(text),
        "author": raw.get("last_editor", ""),
        "created_dt": raw.get("last_edit_ts"),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("content_hash", ""),
    }


def _normalise_hackernews(raw: Dict, partition: str) -> List[Dict]:
    """Returns story + flattened top-level comments as separate docs."""
    docs: List[Dict] = []

    text = (raw.get("text") or raw.get("title") or "").strip()
    if text:
        docs.append({
            "doc_id": _make_doc_id("hackernews", raw.get("id", ""), text),
            "source": "hackernews",
            "category": "social",
            "domain": raw.get("domain", "news.ycombinator.com"),
            "url": raw.get("hn_url", raw.get("url", "")),
            "title": raw.get("title", ""),
            "text": text,
            "text_length": len(text),
            "author": raw.get("author", ""),
            "created_dt": raw.get("created_dt"),
            "crawl_partition": partition,
            "ingested_at": raw.get("_ingested_at", ""),
            "content_hash": raw.get("content_hash", ""),
        })

    for comment in raw.get("comments", []):
        ct = (comment.get("text") or "").strip()
        if ct and len(ct) > 15:
            docs.append({
                "doc_id": _make_doc_id("hn_comment", comment.get("id", ""), ct),
                "source": "hackernews",
                "category": "social",
                "domain": "news.ycombinator.com",
                "url": raw.get("hn_url", ""),
                "title": "",
                "text": ct,
                "text_length": len(ct),
                "author": comment.get("author", ""),
                "created_dt": comment.get("created_dt"),
                "crawl_partition": partition,
                "ingested_at": raw.get("_ingested_at", ""),
                "content_hash": comment.get("content_hash", ""),
            })

    return docs


def _normalise_wayback(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    # Use snapshot_year in the doc_id so same URL across years = distinct docs
    identifier = f"{raw.get('sentinel_url','')}:{raw.get('snapshot_year','')}"
    return {
        "doc_id": _make_doc_id("wayback", identifier, text),
        "source": "wayback",
        "category": raw.get("category", "web"),
        "domain": raw.get("domain", ""),
        "url": raw.get("wayback_url", raw.get("sentinel_url", "")),
        "title": raw.get("sentinel_url", ""),
        "text": text,
        "text_length": len(text),
        "author": "",
        "created_dt": raw.get("snapshot_date"),
        "crawl_partition": str(raw.get("snapshot_year", partition)),
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("content_hash", ""),
    }


def _normalise_bluesky(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    return {
        "doc_id": _make_doc_id("bluesky", raw.get("uri", ""), text),
        "source": "bluesky",
        "category": "social",
        "domain": "bsky.app",
        "url": f"https://bsky.app/profile/{raw.get('author_handle','')}/post/{raw.get('uri','').split('/')[-1]}",
        "title": "",
        "text": text,
        "text_length": len(text),
        "author": raw.get("author_handle", ""),
        "created_dt": raw.get("created_at"),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("cid", ""),
    }


def _normalise_fourchan(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    board = raw.get("board", "")
    post_no = raw.get("post_no", "")
    return {
        "doc_id": _make_doc_id("fourchan", f"{board}/{post_no}", text),
        "source": "fourchan",
        "category": "forum",
        "domain": "4chan.org",
        "url": f"https://boards.4chan.org/{board}/thread/{raw.get('thread_no','')}",
        "title": raw.get("subject", ""),
        "text": text,
        "text_length": len(text),
        "author": raw.get("country", ""),
        "created_dt": str(raw.get("timestamp", "")),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": "",
    }


def _normalise_steam(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    return {
        "doc_id": _make_doc_id("steam", raw.get("review_id", ""), text),
        "source": "steam",
        "category": "social",
        "domain": "store.steampowered.com",
        "url": f"https://store.steampowered.com/app/{raw.get('app_id','')}",
        "title": raw.get("game_name", ""),
        "text": text,
        "text_length": len(text),
        "author": raw.get("author_steam_id", ""),
        "created_dt": str(raw.get("timestamp_created", "")),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("review_id", ""),
    }


def _normalise_youtube(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("comment_text") or "").strip()
    if not text:
        return None
    return {
        "doc_id": _make_doc_id("youtube", raw.get("comment_id", ""), text),
        "source": "youtube",
        "category": "social",
        "domain": "youtube.com",
        "url": f"https://www.youtube.com/watch?v={raw.get('video_id','')}",
        "title": raw.get("video_title", ""),
        "text": text,
        "text_length": len(text),
        "author": raw.get("author_channel_id", ""),
        "created_dt": raw.get("published_at", ""),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("comment_id", ""),
    }


def _normalise_linkedin(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    url = raw.get("url", "")
    return {
        "doc_id": _make_doc_id("linkedin", url, text),
        "source": "linkedin",
        "category": "professional",
        "domain": "linkedin.com",
        "url": url,
        "title": raw.get("title", ""),
        "text": text,
        "text_length": len(text),
        "author": "",
        "created_dt": raw.get("published_at", ""),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": "",
    }


def _normalise_twitter(raw: Dict, partition: str) -> Optional[Dict]:
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    return {
        "doc_id": _make_doc_id("twitter", raw.get("tweet_id", ""), text),
        "source": "twitter",
        "category": "social",
        "domain": "x.com",
        "url": "",
        "title": "",
        "text": text,
        "text_length": len(text),
        "author": raw.get("author_id", ""),
        "created_dt": raw.get("created_at", ""),
        "crawl_partition": partition,
        "ingested_at": raw.get("_ingested_at", ""),
        "content_hash": raw.get("tweet_id", ""),
    }


NORMALISERS = {
    "common_crawl": _normalise_common_crawl,
    "reddit":        _normalise_reddit,
    "news":          _normalise_news,
    "wikipedia":     _normalise_wikipedia,
    "hackernews":    _normalise_hackernews,
    "wayback":       _normalise_wayback,
    "bluesky":       _normalise_bluesky,
    "fourchan":      _normalise_fourchan,
    "steam":         _normalise_steam,
    "youtube":       _normalise_youtube,
    "linkedin":      _normalise_linkedin,
    "twitter":       _normalise_twitter,
}


# ── Bronze reader ─────────────────────────────────────────────────────────────

def iter_bronze_records(bronze_root: Path, source: str) -> Iterator[tuple]:
    """Yield (raw_record, partition_string) from all JSONL files for a source."""
    source_dir = bronze_root / source
    if not source_dir.exists():
        return
    for jsonl_file in sorted(source_dir.rglob("*.jsonl")):
        # Derive partition from relative path
        rel_parts = jsonl_file.parent.relative_to(source_dir).parts
        partition = "/".join(rel_parts) if rel_parts else "unknown"
        with open(jsonl_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line), partition
                    except json.JSONDecodeError:
                        continue


# ── Silver writer ─────────────────────────────────────────────────────────────

class BronzeToSilverPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)
        data_root = Path(self.config["storage"]["local_path"])
        self.bronze_root = data_root / self.config["storage"]["bronze_path"]
        self.silver_root = data_root / self.config["storage"]["silver_path"]
        self.silver_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config["storage"].get("db_path", "./data/observatory.db")

    def process_source(self, source: str) -> pd.DataFrame:
        """Normalise and deduplicate all bronze records for one source."""
        print(f"\n[PIPELINE] Processing source: {source}")
        normaliser = NORMALISERS.get(source.split("/")[0])
        if not normaliser:
            print(f"  No normaliser for {source}")
            return pd.DataFrame()

        docs: List[Dict] = []
        seen_hashes: set = set()

        for raw, partition in iter_bronze_records(self.bronze_root, source):
            result = normaliser(raw, partition)
            if result is None:
                continue
            if isinstance(result, list):
                results = result
            else:
                results = [result]
            for doc in results:
                h = doc.get("content_hash") or doc["doc_id"]
                if h and h not in seen_hashes:
                    seen_hashes.add(h)
                    docs.append(doc)

        if not docs:
            print(f"  No records found for {source}")
            return pd.DataFrame()

        df = pd.DataFrame(docs, columns=SILVER_COLUMNS)
        df["created_dt"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True)

        out = self.silver_root / f"{source.replace('/', '_')}.parquet"
        df.to_parquet(out, index=False, engine="pyarrow")
        print(f"  ✓ {len(df):,} records → {out}")
        return df

    def run_all(self) -> pd.DataFrame:
        """Process all sources and combine into a single silver DataFrame."""
        sources = [
            "common_crawl", "reddit", "news", "wikipedia/articles",
            "hackernews", "wayback",
            "bluesky", "fourchan", "steam", "youtube", "linkedin", "twitter",
        ]
        frames: List[pd.DataFrame] = []

        for src in sources:
            df = self.process_source(src)
            if not df.empty:
                frames.append(df)

        if not frames:
            print("[PIPELINE] No data to combine")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(subset=["doc_id"], inplace=True)

        out = self.silver_root / "combined.parquet"
        combined.to_parquet(out, index=False, engine="pyarrow")
        print(f"\n[PIPELINE] Combined silver: {len(combined):,} unique records → {out}")
        return combined


if __name__ == "__main__":
    BronzeToSilverPipeline().run_all()
