"""
Stack Overflow Minion — harvests questions and top answers via the StackExchange API v2.3.

Free tier: ~300 requests/day per IP (unauthenticated).
With STACKOVERFLOW_API_KEY: 10,000 requests/day.

Strategy per run:
  - 10 tags × 1 page of 20 questions each  → 10 question-list requests
  - Up to 2 answers per question (batch fetch per question) → up to 200 answer requests
  Total: ≤ 210 requests/run well inside both quotas.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from minions.base_minion import BaseMinion


TAGS = [
    "python",
    "javascript",
    "machine-learning",
    "artificial-intelligence",
    "chatgpt",
    "llm",
    "web-development",
    "sql",
    "react",
    "docker",
]

SOURCE = "stackoverflow"
CATEGORY = "qa"
API_BASE = "https://api.stackexchange.com/2.3"
HEADERS = {
    "User-Agent": "DeadInternetObservatory/1.0 (research; github.com/dead-internet-observatory)"
}


def _strip_html(html: str) -> str:
    """Strip HTML tags from a string, returning plain text."""
    if not html:
        return ""
    try:
        return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)
    except Exception:
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()


def _iso(unix_ts: Optional[int]) -> str:
    if not unix_ts:
        return ""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


class StackOverflowBot(BaseMinion):

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="stackoverflow")
        self.api_key: Optional[str] = os.environ.get("STACKOVERFLOW_API_KEY") or None
        self.delay = 0.1 if self.api_key else 0.5
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        if self.api_key:
            self.logger.info("Running with API key (10K quota/day)")
        else:
            self.logger.info("Running without API key (300 quota/day) — throttle=0.5s")

    # ── StackExchange API helpers ─────────────────────────────────────────────

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """GET from the StackExchange API, handling quota and errors."""
        if self.api_key:
            params["key"] = self.api_key
        try:
            resp = self.session.get(
                f"{API_BASE}/{endpoint}",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # Log quota remaining so CI logs stay informative
            quota = data.get("quota_remaining")
            if quota is not None and quota < 50:
                self.logger.warning(f"  Quota remaining: {quota}")
            return data
        except requests.HTTPError as exc:
            self.logger.error(f"  HTTP error [{endpoint}]: {exc}")
            self.stats["errors"] += 1
            return None
        except Exception as exc:
            self.logger.error(f"  Request error [{endpoint}]: {exc}")
            self.stats["errors"] += 1
            return None

    def _fetch_questions(self, tag: str, pagesize: int = 20) -> List[Dict]:
        """Return raw question items for a given tag."""
        data = self._get("questions", {
            "order":    "desc",
            "sort":     "activity",
            "site":     "stackoverflow",
            "tagged":   tag,
            "filter":   "withbody",
            "pagesize": pagesize,
        })
        if not data:
            return []
        self.stats["fetched"] += len(data.get("items", []))
        return data.get("items", [])

    def _fetch_answers(self, question_id: int, max_answers: int = 2) -> List[Dict]:
        """Return the top-voted answers for a question (up to max_answers)."""
        data = self._get(f"questions/{question_id}/answers", {
            "order":    "desc",
            "sort":     "votes",
            "site":     "stackoverflow",
            "filter":   "withbody",
            "pagesize": max_answers,
        })
        if not data:
            return []
        return data.get("items", [])

    # ── Record builders ───────────────────────────────────────────────────────

    def _build_question_record(self, item: Dict, tag: str) -> Dict:
        text = _strip_html(item.get("body", ""))
        return {
            "question_id":      item.get("question_id"),
            "answer_id":        None,
            "record_type":      "question",
            "title":            item.get("title", ""),
            "text":             text,
            "tags":             item.get("tags", []),
            "score":            item.get("score", 0),
            "is_accepted":      False,
            "view_count":       item.get("view_count", 0),
            "answer_count":     item.get("answer_count", 0),
            "owner_reputation": (item.get("owner") or {}).get("reputation", 0),
            "created_at":       _iso(item.get("creation_date")),
            "url":              item.get("link", ""),
            "category":         CATEGORY,
            "search_tag":       tag,
        }

    def _build_answer_record(self, answer: Dict, question: Dict) -> Dict:
        text = _strip_html(answer.get("body", ""))
        return {
            "question_id":      question.get("question_id"),
            "answer_id":        answer.get("answer_id"),
            "record_type":      "answer",
            "title":            question.get("title", ""),
            "text":             text,
            "tags":             question.get("tags", []),
            "score":            answer.get("score", 0),
            "is_accepted":      answer.get("is_accepted", False),
            "view_count":       question.get("view_count", 0),
            "answer_count":     question.get("answer_count", 0),
            "owner_reputation": (answer.get("owner") or {}).get("reputation", 0),
            "created_at":       _iso(answer.get("creation_date")),
            "url":              answer.get("link", ""),
            "category":         CATEGORY,
            "search_tag":       question.get("search_tag", ""),
        }

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info(
            f"Stack Overflow Minion starting | "
            f"tags={len(TAGS)} | pagesize=20 | max_answers_per_q=2"
        )

        records: List[Dict] = []
        seen_question_ids = set()

        for tag in TAGS:
            self.logger.info(f"  Fetching tag: {tag}")
            questions = self._fetch_questions(tag, pagesize=20)
            self.logger.info(f"    {len(questions)} questions returned")
            self.throttle(self.delay)

            for item in questions:
                qid = item.get("question_id")
                if not qid or qid in seen_question_ids:
                    self.stats["skipped"] += 1
                    continue
                seen_question_ids.add(qid)

                # Build and collect question record
                q_record = self._build_question_record(item, tag)
                records.append(q_record)
                self.stats["processed"] += 1

                # Fetch top 2 answers
                if item.get("answer_count", 0) > 0:
                    answers = self._fetch_answers(qid, max_answers=2)
                    for ans in answers:
                        a_record = self._build_answer_record(ans, q_record)
                        records.append(a_record)
                        self.stats["processed"] += 1
                    self.throttle(self.delay)

            # Flush batch after each tag to keep memory low
            if records:
                self.save_bronze(records, source=SOURCE)
                records = []

        # Flush any remainder
        if records:
            self.save_bronze(records, source=SOURCE)

        self.logger.info(
            f"Done — {len(seen_question_ids)} unique questions processed"
        )
        self.report_stats()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = StackOverflowBot(config_path=config_path)
    bot.run()
