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

    # Create dataset repo if it doesn't exist
    try:
        retry_hf(
            "creating dataset repo",
            lambda: api.create_repo(repo_id=repo_id, repo_type="dataset",
                                    private=False, exist_ok=True, token=token),
        )
        print(f"[HF] Repo ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as exc:
        if is_rate_limit(exc):
            print(f"[HF] Rate limited creating repo; assuming it already exists and continuing: {exc}")
        else:
            print(f"[HF] Repo creation error (may already exist): {exc}")

    # Upload Parquet layers in two commits instead of one API call per file.
    uploaded = 0
    failed = []
    rate_limited = []
    for layer in ["silver", "gold"]:
        layer_dir = data_root / layer
        if not layer_dir.exists():
            print(f"[HF] Skipping {layer} — directory not found")
            continue

        parquet_files = sorted(layer_dir.rglob("*.parquet"))
        if not parquet_files:
            print(f"[HF] Skipping {layer} — no Parquet files found")
            continue

        print(f"[HF] Uploading {layer}/ layer ({len(parquet_files)} parquet files) …")
        try:
            retry_hf(
                f"uploading {layer}/",
                lambda: api.upload_folder(
                    folder_path=str(layer_dir),
                    path_in_repo=layer,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    allow_patterns="*.parquet",
                    commit_message=f"update {layer} parquet datasets",
                ),
            )
            uploaded += len(parquet_files)
        except Exception as exc:
            print(f"[HF] Upload failed for {layer}/: {exc}")
            if is_rate_limit(exc):
                rate_limited.append(layer)
            else:
                failed.append(layer)

    # Also upload the SQLite index as a convenience snapshot
    db_path = data_root / "observatory.db"
    if db_path.exists():
        print("[HF] Uploading observatory.db snapshot …")
        try:
            retry_hf(
                "uploading observatory.db",
                lambda: api.upload_file(
                    path_or_fileobj=str(db_path),
                    path_in_repo="observatory.db",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message="update observatory.db",
                ),
            )
            uploaded += 1
        except Exception as exc:
            print(f"[HF] DB upload failed: {exc}")
            if is_rate_limit(exc):
                rate_limited.append("observatory.db")
            else:
                failed.append("observatory.db")

    if failed:
        print(f"[HF] Error: {len(failed)} required upload(s) failed: {', '.join(failed)}")
        sys.exit(1)

    if rate_limited:
        print(f"[HF] Deferred due to rate limit: {', '.join(rate_limited)}")
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
