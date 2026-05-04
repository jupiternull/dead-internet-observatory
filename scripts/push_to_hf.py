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
from pathlib import Path


def push(repo_id: str, data_root: Path, token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed — run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Create dataset repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        private=False, exist_ok=True, token=token)
        print(f"[HF] Repo ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as exc:
        print(f"[HF] Repo creation error (may already exist): {exc}")

    # Upload all Parquet files from silver and gold layers
    uploaded = 0
    for layer in ["silver", "gold"]:
        layer_dir = data_root / layer
        if not layer_dir.exists():
            print(f"[HF] Skipping {layer} — directory not found")
            continue

        for parquet_file in sorted(layer_dir.rglob("*.parquet")):
            rel_path = parquet_file.relative_to(data_root)
            print(f"[HF] Uploading {rel_path} …")
            try:
                api.upload_file(
                    path_or_fileobj=str(parquet_file),
                    path_in_repo=str(rel_path),
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"update {rel_path}",
                )
                uploaded += 1
            except Exception as exc:
                print(f"[HF] Upload failed for {rel_path}: {exc}")

    # Also upload the SQLite index as a convenience snapshot
    db_path = data_root / "observatory.db"
    if db_path.exists():
        print("[HF] Uploading observatory.db snapshot …")
        try:
            api.upload_file(
                path_or_fileobj=str(db_path),
                path_in_repo="observatory.db",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="update observatory.db",
            )
            uploaded += 1
        except Exception as exc:
            print(f"[HF] DB upload failed: {exc}")

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
