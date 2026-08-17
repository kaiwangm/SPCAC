#!/usr/bin/env python3
"""
Create hdf5 files for j8ivfbv2 and owlii datasets for fast loading.

The hdf5 schema matches sensat_urban.hdf5:
    dataset/{split}/{idx}/points  (N, 3) float64
    dataset/{split}/{idx}/colors  (N, 3) float64

ply files are left untouched (they are also uploaded to HF).

Usage:
    python utils/create_hdf5.py            # preview
    python utils/create_hdf5.py --run      # create hdf5 files
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"

# dataset name -> local directory
DATASETS = {
    "j8ivfbv2": DATA_DIR / "j8ivfbv2",
    "owlii": DATA_DIR / "owlii",
}


def collect_ply_files(src_dir):
    """Return {split: sorted list of ply paths}."""
    splits = {}
    for subdir in sorted(src_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for split in ("train", "test"):
            split_dir = subdir / split
            if not split_dir.is_dir():
                continue
            files = sorted(split_dir.glob("*.ply"))
            if files:
                splits.setdefault(split, []).extend(files)
    return splits


def create_hdf5(dataset_name, src_dir):
    hdf5_path = DATA_DIR / f"{dataset_name}.hdf5"
    if hdf5_path.exists():
        print(f"SKIP: {hdf5_path.name} already exists")
        return 0

    splits = collect_ply_files(src_dir)
    total = sum(len(v) for v in splits.values())
    if total == 0:
        print(f"SKIP: no ply files found for {dataset_name}")
        return 0

    print(f"Creating {hdf5_path.name} ({total} ply files)...")
    with h5py.File(hdf5_path, "w") as f:
        f.require_group("dataset")
        for split, files in sorted(splits.items()):
            split_group = f["dataset"].create_group(split)
            for idx, ply_path in enumerate(files):
                pc = o3d.io.read_point_cloud(str(ply_path))
                points = np.asarray(pc.points)
                colors = np.asarray(pc.colors)
                if colors.size == 0:
                    colors = np.zeros_like(points)
                sample = split_group.create_group(str(idx))
                sample.create_dataset("points", data=points)
                sample.create_dataset("colors", data=colors)
                print(f"  [{split}/{idx}] {ply_path.name}  {points.shape}")
    size_mb = hdf5_path.stat().st_size / (1024 * 1024)
    print(f"Done: {hdf5_path.name} ({size_mb:.1f} MB)\n")
    return total


def main():
    run_real = "--run" in sys.argv

    for dataset_name, src_dir in sorted(DATASETS.items()):
        splits = collect_ply_files(src_dir)
        total = sum(len(v) for v in splits.values())
        size_mb = sum(f.stat().st_size for v in splits.values() for f in v) / (1024 * 1024)
        print(f"{dataset_name}: {total} ply files ({size_mb:.1f} MB) -> {dataset_name}.hdf5")

    if not run_real:
        print("\nRun with --run to create the hdf5 files")
        return

    print()
    for dataset_name, src_dir in sorted(DATASETS.items()):
        create_hdf5(dataset_name, src_dir)


if __name__ == "__main__":
    main()
