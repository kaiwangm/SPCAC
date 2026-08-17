#!/usr/bin/env python3
"""
Upload SPCAC datasets to Hugging Face Dataset Hub (kaiwangm/SPCAC-datasets).

Usage:
    python utils/upload_datasets.py            # preview
    python utils/upload_datasets.py --run      # execute upload

Auth (either):
    huggingface-cli login          # CLI login (recommended)
    export HF_TOKEN=hf_...         # environment variable
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"
DATASET_REPO = "kaiwangm/SPCAC-datasets"

# Enable Xet high-performance mode (multi-threaded chunked transfer)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def main():
    dry_run = "--dry-run" in sys.argv
    run_real = "--run" in sys.argv

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    try:
        whoami = api.whoami()
        print(f"Authenticated as: {whoami['name']}")
    except Exception as e:
        print(f"Auth failed: {e}")
        print("Login first: huggingface-cli login  or  export HF_TOKEN=hf_...")
        sys.exit(1)

    try:
        info = api.repo_info(DATASET_REPO, repo_type="dataset")
        visibility = "private" if info.private else "public"
        print(f"Repo exists: {DATASET_REPO} (visibility: {visibility})")
    except Exception:
        create_repo(DATASET_REPO, private=False, repo_type="dataset", exist_ok=True)
        print(f"Repo created: {DATASET_REPO} (public)")

    # Collect upload items
    items = []

    # hdf5 files
    for hdf5 in sorted(DATA_DIR.glob("*.hdf5")):
        items.append({
            "type": "file",
            "path": hdf5,
            "repo_path": hdf5.name,
            "label": hdf5.name,
        })

    # ply subdirectories (uploaded together via allow_patterns)
    for subdir in sorted(DATA_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith(".") or subdir.name == "__pycache__":
            continue
        items.append({
            "type": "folder",
            "path": subdir,
            "repo_path": subdir.name,
            "label": f"{subdir.name}/",
        })

    if not items:
        print("No dataset files found to upload")
        return

    # Calculate sizes
    for item in items:
        p = item["path"]
        if item["type"] == "file":
            item["size_mb"] = p.stat().st_size / (1024 * 1024)
        else:
            # test split only (train split is not uploaded)
            item["size_mb"] = sum(f.stat().st_size for f in p.rglob("test/*.ply")) / (1024 * 1024)

    total_mb = sum(item["size_mb"] for item in items)

    print(f"\nFound {len(items)} items to upload\n")

    if dry_run or not run_real:
        print("=== PREVIEW (ply: test split only) ===\n")
        for item in items:
            tag = " (test)" if item["type"] == "folder" else ""
            print(f"  {item['label']:<30s} {item['size_mb']:>8.1f} MB{tag}")
        print(f"\nTotal: {total_mb:.1f} MB")
        print("\nRun with --run to upload")
        return

    print(f"Uploading {len(items)} items to {DATASET_REPO} (total {total_mb:.1f} MB)\n")

    success = 0
    failed = 0

    # Single Xet multi-threaded upload for everything (hdf5 + ply test splits),
    # showing one progress bar instead of separate ones per file type
    try:
        print("  Uploading all items...", end=" ", flush=True)
        api.upload_folder(
            folder_path=str(DATA_DIR),
            repo_id=DATASET_REPO,
            repo_type="dataset",
            allow_patterns=["*.hdf5", "**/test/*.ply"],
        )
        print("OK")
        success = len(items)
    except Exception as e:
        print(f"FAILED: {e}")
        failed = len(items)

    print(f"\nDone: {success} success, {failed} failed")
    print(f"Repo: https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()
