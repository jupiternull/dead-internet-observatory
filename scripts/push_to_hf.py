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

import requests


STATE_FILES = (
    "gold/doc_registry.parquet",
    "gold/scored.parquet",
    "observatory.db",
)
RESTORE_RETRY_STATUSES = {429, 500, 502, 503, 504}


def is_rate_limit(exc):
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def is_transient_hf_error(exc):
    response = getattr(exc, "response", None)
    if response is not None and response.status_code in RESTORE_RETRY_STATUSES:
        return True
    return is_rate_limit(exc)


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


def restore(repo_id: str, data_root: Path):
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    restored = 0

    for relative_path in STATE_FILES:
        destination = data_root / relative_path
        url = f"{base_url}/{relative_path}"
        response = None

        for attempt in range(1, 4):
            try:
                response = requests.get(url, headers=headers, timeout=120)
                if response.status_code == 404:
                    print(f"[HF] No prior {relative_path}")
                    break
                if response.status_code in RESTORE_RETRY_STATUSES and attempt < 3:
                    delay = 30 * attempt
                    print(
                        f"[HF] Restore {relative_path} returned HTTP "
                        f"{response.status_code}; retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(response.content)
                restored += 1
                print(f"[HF] Restored {relative_path}")
                break
            except requests.RequestException as exc:
                if attempt == 3:
                    if is_transient_hf_error(exc):
                        if destination.exists():
                            print(
                                f"[HF] Restore deferred for {relative_path}: {exc}; "
                                "using cached local state"
                            )
                        else:
                            print(
                                f"[HF] Restore deferred for {relative_path}: {exc}; "
                                "state file unavailable"
                            )
                        break
                    print(f"[HF] Restore failed for {relative_path}: {exc}")
                    raise
                delay = 30 * attempt
                print(f"[HF] Restore {relative_path} failed: {exc}; retrying in {delay}s")
                time.sleep(delay)

        if response is None or response.status_code == 404:
            continue

    print(f"[HF] Restored {restored}/{len(STATE_FILES)} state files")


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
    parser.add_argument("--restore", action="store_true",
                        help="Restore persistent pipeline state from Hugging Face")
    parser.add_argument("--repo", default="jupiternull/dead-internet-observatory",
                        help="HuggingFace dataset repo ID (owner/name)")
    parser.add_argument("--data-root", default="./data",
                        help="Path to the local data/ directory")
    args = parser.parse_args()

    if args.restore:
        restore(args.repo, Path(args.data_root))
        sys.exit(0)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    push(args.repo, Path(args.data_root), token)
