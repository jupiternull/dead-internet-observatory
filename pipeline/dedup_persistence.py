"""
Scored-doc dedup persistence via a local JSON file.

In GitHub Actions the file lives at data/scored_ids.json and is preserved
between pipeline runs using actions/cache (keyed to the repo + branch).
On cache eviction the set is empty, causing a one-time full rescore of all
silver docs — the next successful run then rebuilds the cache.
"""

import json
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "scored_ids.json"


def _path() -> Path:
    return _DEFAULT_PATH


def load_scored_ids() -> set:
    p = _path()
    if not p.exists():
        return set()
    try:
        with p.open() as fh:
            return set(json.load(fh))
    except Exception:
        return set()


def append_scored_ids(new_ids: set) -> None:
    if not new_ids:
        return
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_scored_ids()
    merged = existing | new_ids
    with p.open("w") as fh:
        json.dump(list(merged), fh)
