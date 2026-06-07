"""
Push Silver and Gold Parquet datasets to Hugging Face.

Called by the pipeline GitHub Actions workflow after each index update.
Requires HF_TOKEN env var (write-access token from huggingface.co/settings/tokens).

Usage:
    python scripts/push_to_hf.py --repo jupiternull/dead-internet-observatory
"""

import argparse
import os
import sys
import time
from pathlib import Path


def is_rate_limit(exc):
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def retry_hf(label, func, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if is_rate_limit(exc):
                if attempt == max_attempts:
                    raise
                delay = 60 * attempt
                print(f"[HF] Rate limited while {label}; retrying in {delay}s")
                time.sleep(delay)
                continue
            raise


def push(repo_id: str, data_root: Path, token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed — run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    parquet_files = sorted(data_root.glob("silver/**/*.parquet"))
    parquet_files.extend(sorted(data_root.glob("gold/**/*.parquet")))
    db_path = data_root / "observatory.db"
    uploaded = len(parquet_files) + int(db_path.exists())
    if not uploaded:
        print("[HF] Error: no Parquet datasets or observatory.db found")
        sys.exit(1)

    print(f"[HF] Uploading {uploaded} dataset files in one commit …")
    try:
        retry_hf(
            "uploading dataset snapshot",
            lambda: api.upload_folder(
                folder_path=str(data_root),
                path_in_repo="",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                allow_patterns=["*.parquet", "observatory.db"],
                commit_message="update observatory dataset snapshot",
            ),
        )
    except Exception as exc:
        if not is_rate_limit(exc):
            print(f"[HF] Dataset upload failed: {exc}")
            sys.exit(1)
        print("[HF] Dataset sync deferred due to rate limiting.")
        print("[HF] Pipeline output is complete; Hugging Face sync can catch up on the next run.")
        return

    print(f"[HF] ✓ Done — {uploaded} files pushed to {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="jupiternull/dead-internet-observatory",
                        help="HuggingFace dataset repo ID (owner/name)")
    parser.add_argument("--data-root", default="./data",
                        help="Path to the local data/ directory")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    push(args.repo, Path(args.data_root), token)
