#!/usr/bin/env python3
"""
Upload SPCAC checkpoints to Hugging Face Hub (eb_las.pth only).

Uploads only the kept model classes:
    factorized_prior, mean_scale_hyperprior, grouping, elpcac_l, elpcac

Usage:
    uv run python utils/upload_checkpoints.py            # preview
    uv run python utils/upload_checkpoints.py --run      # execute upload
    uv run python utils/upload_checkpoints.py --dry-run  # preview

Auth (either):
    huggingface-cli login          # CLI login (recommended)
    export HF_TOKEN=hf_...         # environment variable
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_REPO = os.environ.get(
    "SPCAC_HF_CHECKPOINT_REPO", "kaiwangm/SPCAC-checkpoints"
)

# Kept model class names (checkpoint directory names under each dataset)
KEPT_MODELS = (
    "factorized_prior",
    "mean_scale_hyperprior",
    "grouping",
    "elpcac_l",
    "elpcac",
)

# Enable Xet high-performance mode (multi-threaded chunked transfer)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def collect_quality_dirs():
    """Group las files by (dataset, model, quality) for kept models only.

    Returns list of (local_dir, repo_path, file_count, total_bytes).
    """
    groups = defaultdict(list)

    for f in sorted(CHECKPOINTS_DIR.rglob("eb_las.pth")):
        # Skip HF / local caches
        if ".cache" in f.parts:
            continue

        rel = f.relative_to(CHECKPOINTS_DIR)
        parts = rel.parts  # e.g. ('coco3d', 'elpcac', 'quality_0', 'eb_las.pth')
        if len(parts) < 4:
            continue

        dataset, model, quality = parts[0], parts[1], parts[2]
        if model not in KEPT_MODELS:
            continue
        if not quality.startswith("quality_"):
            continue

        groups[(dataset, model, quality)].append(f)

    dirs = []
    for (dataset, model, quality), files in sorted(groups.items()):
        local_dir = CHECKPOINTS_DIR / dataset / model / quality
        repo_path = f"{dataset}/{model}/{quality}"
        size_bytes = sum(f.stat().st_size for f in files)
        dirs.append((local_dir, repo_path, len(files), size_bytes))

    return dirs


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
        info = api.repo_info(CHECKPOINT_REPO, repo_type="model")
        visibility = "private" if info.private else "public"
        print(f"Repo exists: {CHECKPOINT_REPO} (visibility: {visibility})")
    except Exception:
        create_repo(CHECKPOINT_REPO, private=False, repo_type="model", exist_ok=True)
        print(f"Repo created: {CHECKPOINT_REPO} (public)")

    dirs = collect_quality_dirs()
    total_files = sum(d[2] for d in dirs)
    total_mb = sum(d[3] for d in dirs) / (1024 * 1024)

    print(f"\nKept models: {', '.join(KEPT_MODELS)}")
    print(f"Found {len(dirs)} quality dirs ({total_files} files, {total_mb:.1f} MB)\n")

    if dry_run or not run_real:
        print("=== PREVIEW ===\n")
        current_ds = None
        for local_dir, repo_path, count, size in dirs:
            ds = repo_path.split("/", 1)[0]
            if ds != current_ds:
                current_ds = ds
                print(f"  [{ds}]")
            print(f"    {repo_path:<50s} {count:>2} files  {size/1024/1024:>6.1f} MB")
        print(f"\nTotal: {total_mb:.1f} MB")
        print("\nRun with --run to upload")
        return

    allow_patterns = [
        f"**/{model}/quality_*/eb_las.pth" for model in KEPT_MODELS
    ]

    print(f"Uploading {total_mb:.1f} MB with Xet multi-threaded upload\n")

    try:
        api.upload_folder(
            folder_path=str(CHECKPOINTS_DIR),
            repo_id=CHECKPOINT_REPO,
            repo_type="model",
            allow_patterns=allow_patterns,
            ignore_patterns=["**/.cache/**"],
            commit_message=(
                "Upload checkpoints: factorized_prior, mean_scale_hyperprior, "
                "grouping, elpcac_l, elpcac"
            ),
        )
        print("\nUpload complete")
        success = total_files
    except Exception as e:
        print(f"\nFAILED: {e}")
        success = 0

    print(f"\nDone: {success} success, {total_files - success} failed")
    print(f"Repo: https://huggingface.co/{CHECKPOINT_REPO}")


if __name__ == "__main__":
    main()
