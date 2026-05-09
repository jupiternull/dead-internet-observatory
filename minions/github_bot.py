"""
GitHub Minion — harvests READMEs and open issues from active/trending repos.

Uses the GitHub REST API (search + readme + issues endpoints). Authenticates
via GITHUB_TOKEN env var when available (5 000 req/hr); falls back to
unauthenticated (60 req/hr) without it.
"""

import base64
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import requests

from minions.base_minion import BaseMinion


README_TRUNCATE = 8000
ISSUE_TRUNCATE = 3000
MIN_README_LEN = 200

SEARCH_QUERIES = [
    "stars:>500 pushed:>2024-01-01 topic:machine-learning",
    "stars:>500 pushed:>2024-01-01 topic:web",
    "stars:>500 pushed:>2024-01-01",
]


class GitHubBot(BaseMinion):

    API_BASE = "https://api.github.com"

    HEADERS = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DeadInternetObservatory/1.0",
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="github")
        self.token = os.environ.get("GITHUB_TOKEN", "")
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
            self.logger.info("Using authenticated GitHub API (5000 req/hr)")
        else:
            self.logger.warning("GITHUB_TOKEN not set — unauthenticated (60 req/hr)")

    # ── API helpers ───────────────────────────────────────────────────────────

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Perform a GET request; return parsed JSON or None on error."""
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None  # silently skip missing resources
            if resp.status_code in (403, 429):
                self.logger.warning(f"  Rate-limited or forbidden: {url}")
                self.stats["errors"] += 1
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"  Request failed {url}: {exc}")
            self.stats["errors"] += 1
            return None

    # ── Search ────────────────────────────────────────────────────────────────

    def _search_repos(self, query: str) -> List[Dict]:
        """Return up to 30 repo metadata dicts from a search query."""
        self.logger.info(f"  Search: {query!r}")
        data = self._get(
            f"{self.API_BASE}/search/repositories",
            params={"q": query, "sort": "updated", "per_page": 30},
        )
        self.throttle(0.5)

        if not data:
            return []

        items = data.get("items", [])
        self.logger.info(f"    -> {len(items)} repos")
        return items

    # ── README ────────────────────────────────────────────────────────────────

    def _fetch_readme(self, full_name: str) -> Optional[str]:
        """Fetch and base64-decode the default-branch README. Returns None if absent."""
        data = self._get(f"{self.API_BASE}/repos/{full_name}/readme")
        self.throttle(0.5)

        if not data:
            return None

        encoded = data.get("content", "")
        if not encoded:
            return None

        try:
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
        except Exception as exc:
            self.logger.warning(f"  README decode error for {full_name}: {exc}")
            return None

    # ── Issues ────────────────────────────────────────────────────────────────

    def _fetch_issues(self, full_name: str) -> List[Dict]:
        """Fetch up to 5 most recently updated open issues."""
        data = self._get(
            f"{self.API_BASE}/repos/{full_name}/issues",
            params={"state": "open", "per_page": 5, "sort": "updated"},
        )
        self.throttle(0.5)

        if not data or not isinstance(data, list):
            return []

        return data

    # ── Record builders ───────────────────────────────────────────────────────

    def _build_readme_record(self, repo: Dict, readme_text: str) -> Dict:
        return {
            "repo_full_name": repo["full_name"],
            "record_type": "readme",
            "title": repo.get("name", ""),
            "text": readme_text[:README_TRUNCATE],
            "stars": repo.get("stargazers_count", 0),
            "language": repo.get("language") or "",
            "description": repo.get("description") or "",
            "url": repo.get("html_url", ""),
            "pushed_at": repo.get("pushed_at", ""),
            "category": "code_doc",
        }

    def _build_issue_record(self, repo: Dict, issue: Dict) -> Optional[Dict]:
        body = (issue.get("body") or "").strip()
        if not body:
            return None

        labels = [lbl.get("name", "") for lbl in issue.get("labels", [])]

        return {
            "repo_full_name": repo["full_name"],
            "issue_number": issue.get("number", 0),
            "record_type": "issue",
            "title": issue.get("title", "").strip(),
            "text": body[:ISSUE_TRUNCATE],
            "author": (issue.get("user") or {}).get("login", ""),
            "url": issue.get("html_url", ""),
            "created_at": issue.get("created_at", ""),
            "state": issue.get("state", ""),
            "labels": labels,
            "category": "code_doc",
        }

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting — {len(SEARCH_QUERIES)} search queries")

        seen_repos: Set[str] = set()
        all_records: List[Dict] = []

        for query in SEARCH_QUERIES:
            repos = self._search_repos(query)
            self.stats["fetched"] += len(repos)

            for repo in repos:
                full_name = repo.get("full_name", "")
                if not full_name or full_name in seen_repos:
                    self.stats["skipped"] += 1
                    continue
                seen_repos.add(full_name)

                self.logger.info(f"  Processing {full_name}")

                # README
                readme_text = self._fetch_readme(full_name)
                if readme_text and len(readme_text) >= MIN_README_LEN:
                    all_records.append(self._build_readme_record(repo, readme_text))
                    self.stats["processed"] += 1
                else:
                    self.stats["skipped"] += 1

                # Issues
                issues = self._fetch_issues(full_name)
                for issue in issues:
                    record = self._build_issue_record(repo, issue)
                    if record:
                        all_records.append(record)
                        self.stats["processed"] += 1
                    else:
                        self.stats["skipped"] += 1

        if all_records:
            self.save_bronze(all_records, source="github")
            self.logger.info(f"Done — {len(all_records)} records saved ({len(seen_repos)} repos)")
        else:
            self.logger.warning("Done — no records collected")

        self.report_stats()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = GitHubBot(config_path=config_path)
    bot.run()
