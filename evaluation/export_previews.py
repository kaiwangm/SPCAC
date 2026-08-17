"""Export original / reconstructed point clouds for the Space viewer (full resolution).

Writes ``result/{dataset}/{name}.npz`` next to ``metrics.json``. Does not copy
metrics elsewhere.

Geometry is kept (attribute compression). Original and reconstructions share
the same xyz sort so the viewer compares the same points.

Usage:
    uv run python -m evaluation.export_previews
    uv run python -m evaluation.export_previews --dataset j8ivfbv2-longdress10
    uv run python -m evaluation.export_previews --models elpcac --qualities 3
    uv run python -m evaluation.export_previews --skip_infer   # original only
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.evaluation import get_model
from evaluation.rd_collect import DEFAULT_DATASETS, DEFAULT_MODELS, DEFAULT_QUALITIES

MAX_POINTS = 0  # 0 = keep every point


def _sort_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    return xyz[order], rgb[order]


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
    idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = _as_uint8_rgb(np.asarray(rgb).reshape(-1, 3))
    xyz, rgb = _sort_xyz_rgb(xyz, rgb)
    n = xyz.shape[0]
    if max_points > 0 and n > max_points:
        if idx is None:
            idx = np.sort(rng.choice(n, max_points, replace=False).astype(np.int32))
        xyz, rgb = xyz[idx], rgb[idx]
    else:
        idx = np.arange(0, dtype=np.int32)
    return xyz, rgb, idx


def save_npz(path: Path, xyz: np.ndarray, rgb: np.ndarray, n_full: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "xyz": np.asarray(xyz, dtype=np.float32),
        "rgb": _as_uint8_rgb(rgb),
        "n_full": np.int32(n_full),
    }
    if extra:
        for key, value in extra.items():
            payload[key] = value
    np.savez_compressed(path, **payload)
    print(f"  wrote {path}  ({xyz.shape[0]} pts, full={n_full})", flush=True)


def load_test_dataset(dataset: str):
    from dataset import load_dataset

    ds = load_dataset(dataset, mode="test")
    if len(ds) == 0:
        raise RuntimeError(f"dataset {dataset} has no test frames")
    return ds


def load_frame(ds, frame: int):
    index = min(max(int(frame), 0), len(ds) - 1)
    points, colors = ds[index]
    points = np.asarray(points).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    return points, colors, index


def load_first_frame(dataset: str, frame: int):
    ds = load_test_dataset(dataset)
    points, colors, index = load_frame(ds, frame)
    return ds, points, colors, index


def preview_out_path(
    preview_dir: Path,
    dataset: str,
    frame: int,
    model: str | None = None,
    quality: int | None = None,
) -> Path:
    name = "original.npz" if model is None else f"{model}_q{quality}.npz"
    if int(frame) == 0:
        return preview_dir / dataset / name
    return preview_dir / dataset / f"s{int(frame)}" / name


def train_dataset_of(result_dir: Path, dataset: str, fallback: str) -> str:
    path = result_dir / dataset / "metrics.json"
    if not path.is_file():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    return str(data.get("train_dataset") or fallback)


def write_samples_json(preview_dir: Path, dataset: str, frames: list[int], n_total: int) -> None:
    path = preview_dir / dataset / "samples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_total": int(n_total),
        "indices": [int(i) for i in frames],
        "split": "test",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {path}", flush=True)


def reconstruct_with_model(model, criterion, points: np.ndarray, colors: np.ndarray):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = torch.from_numpy(np.asarray(points)).int().to(device)
    cols = np.asarray(colors, dtype=np.float32)
    if cols.size and float(np.nanmax(cols)) > 1.0:
        cols = cols / 255.0
    cols = torch.from_numpy(cols).float().to(device)
    if pts.ndim == 2:
        pts = pts.unsqueeze(0)
        cols = cols.unsqueeze(0)

    with torch.no_grad():
        out_enc = model.compress(pts, cols)
        out_dec = model.decompress(out_enc)
        out_net = {**out_enc, **out_dec}
        metrics = criterion.evaluate(out_net)

    xyz = out_net["x_hat"].C[:, 1:].detach().cpu().numpy().astype(np.float32)
    rgb = torch.clip(out_net["x_hat"].F, 0.0, 1.0).detach().cpu().numpy()
    extra = {
        "bpp": np.float32(float(metrics["bpp"])),
        "psnr_yuv": np.float32(float(metrics.get("psnr_yuv", 0.0))),
        "psnr_y": np.float32(float(metrics.get("psnr_y", 0.0))),
    }
    return xyz, rgb, extra


def reconstruct_frame(opt_base, model_name: str, quality: int, points: np.ndarray, colors: np.ndarray, category: str, attributes: str):
    import torch

    opt = SimpleNamespace(**vars(opt_base))
    opt.model = model_name
    opt.quality = quality
    opt.category = category
    opt.attributes = attributes
    opt.metric = "mse"

    model, criterion = get_model(opt)
    try:
        return reconstruct_with_model(model, criterion, points, colors)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def try_load_reconstructed_ply(
    dataset: str,
    model: str,
    quality: int,
    frame: int,
    expected_n: int | None = None,
):
    path = Path("reconstructed") / dataset / model / f"quality_{quality}" / f"{frame}_q{quality}.ply"
    if not path.is_file():
        return None
    try:
        import open3d as o3d
    except ImportError:
        return None
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    if xyz.size == 0:
        return None
    if expected_n is not None and int(xyz.shape[0]) != int(expected_n):
        print(
            f"[warn] skip {path} ({xyz.shape[0]} pts, expected {expected_n})",
            flush=True,
        )
        return None
    return xyz, rgb


def write_manifest(
    preview_dir: Path,
    datasets: list[str],
    models: list[str],
    qualities: list[int],
    frames: list[int],
    max_points: int,
) -> None:
    items = []
    for dataset in datasets:
        for frame in frames:
            orig = preview_out_path(preview_dir, dataset, frame)
            if orig.is_file():
                items.append({"dataset": dataset, "kind": "original", "frame": int(frame)})
            for model in models:
                for quality in qualities:
                    path = preview_out_path(preview_dir, dataset, frame, model, quality)
                    if path.is_file():
                        items.append({
                            "dataset": dataset,
                            "kind": "reconstructed",
                            "model": model,
                            "quality": quality,
                            "frame": int(frame),
                        })
    manifest = {
        "frames": [int(f) for f in frames],
        "max_points": max_points,
        "datasets": datasets,
        "models": models,
        "qualities": qualities,
        "items": items,
    }
    path = preview_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def resolve_frames(args: argparse.Namespace) -> list[int]:
    if args.frames:
        return [int(f) for f in args.frames]
    start = int(args.frame)
    if args.frame_end is not None:
        end = int(args.frame_end)
        if end < start:
            raise SystemExit(f"--frame_end {end} < --frame {start}")
        return list(range(start, end + 1))
    return [start]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export downsampled previews for the SPCAC viewer")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--qualities", nargs="+", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train_dataset", type=str, default="coco3d")
    parser.add_argument("--epoch", type=str, default="las")
    parser.add_argument("--frame", type=int, default=0, help="Test-split frame index (or start if --frame_end is set)")
    parser.add_argument("--frame_end", type=int, default=None, help="Inclusive end frame index")
    parser.add_argument("--frames", nargs="+", type=int, default=None, help="Explicit frame indices")
    parser.add_argument(
        "--max_points",
        type=int,
        default=MAX_POINTS,
        help="Max points to store (0 = full cloud, no downsample)",
    )
    parser.add_argument("--out_dir", type=str, default="result")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip_infer",
        action="store_true",
        help="Only export original clouds (and reconstructed PLYs if present)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing npz files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    datasets = list(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    qualities = list(args.qualities) if args.qualities else list(DEFAULT_QUALITIES)
    frames = resolve_frames(args)
    if args.model:
        models = [args.model]
    if args.dataset:
        datasets = [args.dataset]

    random.seed(args.seed)
    np.random.seed(args.seed)

    preview_dir = Path(args.out_dir)
    result_dir = Path(args.result_dir)

    print(
        f"Export previews: {len(datasets)} datasets × {len(models)} models × "
        f"{len(qualities)} qualities × {len(frames)} frames {frames[0]}–{frames[-1]}, "
        f"max_points={args.max_points}",
        flush=True,
    )

    for dataset in datasets:
        print(f"\n=== {dataset} ===", flush=True)
        train_dataset = train_dataset_of(result_dir, dataset, args.train_dataset)
        opt_base = SimpleNamespace(
            epoch=args.epoch,
            train_dataset=train_dataset,
            seed=args.seed,
            verbose=False,
        )
        print(f"train_dataset={train_dataset}", flush=True)
        try:
            ds = load_test_dataset(dataset)
        except Exception as exc:
            print(f"[fail] load {dataset}: {exc}", flush=True)
            continue

        frames_use = [f for f in frames if 0 <= f < len(ds)]
        if not frames_use:
            print(f"[fail] {dataset}: no frames in range (n_test={len(ds)})", flush=True)
            continue
        write_samples_json(preview_dir, dataset, frames_use, len(ds))

        category = getattr(ds, "category", "dense")
        attributes = getattr(ds, "attributes", "RGB")
        packed: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for frame in frames_use:
            rng = np.random.default_rng(args.seed + int(frame))
            try:
                points, colors, index = load_frame(ds, frame)
            except Exception as exc:
                print(f"[fail] load {dataset} frame {frame}: {exc}", flush=True)
                continue
            orig_path = preview_out_path(preview_dir, dataset, index)
            if orig_path.is_file() and not args.force:
                with np.load(orig_path) as data:
                    idx = np.asarray(data["idx"]) if "idx" in data.files else None
                if idx is None:
                    _, _, idx = downsample(points, colors, args.max_points, rng)
                print(f"[skip] {orig_path}", flush=True)
            else:
                xyz0, rgb0, idx = downsample(points, colors, args.max_points, rng)
                extra={"frame": np.int32(index)}
                if idx.size:
                    extra["idx"] = idx
                save_npz(
                    orig_path,
                    xyz0,
                    rgb0,
                    n_full=points.shape[0],
                    extra=extra,
                )
            packed[index] = (points, colors, idx)

        if args.skip_infer:
            for frame, (points, colors, idx) in packed.items():
                rng = np.random.default_rng(args.seed + int(frame))
                for model in models:
                    for quality in qualities:
                        tag = f"{model} | {dataset} | q{quality} | #{frame}"
                        out_path = preview_out_path(preview_dir, dataset, frame, model, quality)
                        if out_path.is_file() and not args.force:
                            print(f"[skip] {tag}", flush=True)
                            continue
                        loaded = try_load_reconstructed_ply(
                            dataset, model, quality, frame, expected_n=points.shape[0]
                        )
                        if loaded is None:
                            print(f"[skip] {tag} — no reconstructed PLY and --skip_infer", flush=True)
                            continue
                        print(f"[ply ] {tag}", flush=True)
                        xyz, rgb = loaded
                        xyz_ds, rgb_ds, _ = downsample(
                            xyz, rgb, args.max_points, rng,
                            idx=idx if xyz.shape[0] == points.shape[0] else None,
                        )
                        save_npz(out_path, xyz_ds, rgb_ds, n_full=xyz.shape[0], extra={})
            continue

        import torch

        for model in models:
            for quality in qualities:
                missing = []
                for frame, (points, colors, idx) in packed.items():
                    out_path = preview_out_path(preview_dir, dataset, frame, model, quality)
                    if out_path.is_file() and not args.force:
                        print(f"[skip] {model} | {dataset} | q{quality} | #{frame}", flush=True)
                        continue
                    missing.append(frame)
                if not missing:
                    continue

                opt = SimpleNamespace(**vars(opt_base))
                opt.model = model
                opt.quality = quality
                opt.category = category
                opt.attributes = attributes
                opt.metric = "mse"
                print(f"[load] {model} | {dataset} | q{quality}  ({len(missing)} frames)", flush=True)
                try:
                    net, criterion = get_model(opt)
                except Exception as exc:
                    print(f"[fail] load {model} q{quality}: {exc}", flush=True)
                    continue
                try:
                    for frame in missing:
                        points, colors, idx = packed[frame]
                        tag = f"{model} | {dataset} | q{quality} | #{frame}"
                        out_path = preview_out_path(preview_dir, dataset, frame, model, quality)
                        rng = np.random.default_rng(args.seed + int(frame))
                        xyz = rgb = extra = None
                        loaded = try_load_reconstructed_ply(
                            dataset, model, quality, frame, expected_n=points.shape[0]
                        )
                        if loaded is not None:
                            print(f"[ply ] {tag}", flush=True)
                            xyz, rgb = loaded
                            extra = {}
                        else:
                            print(f"[run ] {tag}", flush=True)
                            try:
                                xyz, rgb, extra = reconstruct_with_model(net, criterion, points, colors)
                            except Exception as exc:
                                print(f"[fail] {tag}: {exc}", flush=True)
                                if torch.cuda.is_available():
                                    try:
                                        torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                continue
                        xyz_ds, rgb_ds, _ = downsample(
                            xyz, rgb, args.max_points, rng,
                            idx=idx if xyz.shape[0] == points.shape[0] else None,
                        )
                        save_npz(out_path, xyz_ds, rgb_ds, n_full=xyz.shape[0], extra=extra)
                finally:
                    del net
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

    write_manifest(preview_dir, datasets, models, qualities, frames, args.max_points)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
