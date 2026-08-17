"""Collect rate-distortion points for RD curve plotting.

Runs evaluation over model × dataset × quality, saves structured JSON under
``result/{dataset}/metrics.json``. Existing (model, quality) points
are skipped unless ``--force`` is set, so collection can be resumed incrementally.

Usage examples:
    # All default models × datasets × qualities
    uv run python -m evaluation.rd_collect

    # One model on one dataset
    uv run python -m evaluation.rd_collect --model elpcac --dataset j8ivfbv2-longdress10

    # Specific qualities only
    uv run python -m evaluation.rd_collect --model grouping --qualities 0 1 2

    # Overwrite existing points
    uv run python -m evaluation.rd_collect --model elpcac --force
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.evaluation import evaluation

DEFAULT_MODELS = [
    'baseline_factorized',
    'baseline_mean',
    'grouping',
    'elpcac_l',
    'elpcac',
]

DEFAULT_DATASETS = [
    'j8ivfbv2-longdress10',
    'j8ivfbv2-loot10',
    'j8ivfbv2-redandblack10',
    'j8ivfbv2-soldier10',
]

DEFAULT_QUALITIES = [0, 1, 2, 3, 4, 5]

MODEL_CLASS = {
    'baseline_factorized': 'factorized_prior',
    'baseline_mean': 'mean_scale_hyperprior',
    'grouping': 'grouping',
    'elpcac_l': 'elpcac_l',
    'elpcac': 'elpcac',
}


def result_path(out_dir: Path, dataset: str) -> Path:
    """JSON path: ``{out_dir}/{dataset}/metrics.json``."""
    return out_dir / dataset / 'metrics.json'


def load_result(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data['updated_at'] = datetime.now(timezone.utc).isoformat()
    fd, tmp_name = tempfile.mkstemp(
        suffix='.json', dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def has_point(curves: dict, model: str, quality: int) -> bool:
    for point in curves.get(model, []):
        if int(point.get('quality', -1)) == quality:
            return True
    return False


def upsert_point(curves: dict, model: str, point: dict) -> None:
    quality = int(point['quality'])
    points = curves.setdefault(model, [])
    for i, existing in enumerate(points):
        if int(existing.get('quality', -1)) == quality:
            points[i] = point
            points.sort(key=lambda p: int(p['quality']))
            return
    points.append(point)
    points.sort(key=lambda p: int(p['quality']))


def checkpoint_exists(train_dataset: str, model: str, quality: int, epoch: str) -> bool:
    model_class = MODEL_CLASS.get(model, model)
    local = Path('checkpoints') / train_dataset / model_class / f'quality_{quality}' / f'eb_{epoch}.pth'
    if local.is_file():
        return True
    # Allow HF Hub fallback during evaluation itself
    return bool(os.environ.get('SPCAC_HF_CHECKPOINT_REPO'))


def run_one(opt_base: SimpleNamespace, model: str, dataset: str, quality: int) -> dict:
    opt = SimpleNamespace(**vars(opt_base))
    opt.model = model
    opt.dataset = dataset
    opt.quality = quality
    opt.skip_reconstruct = True
    mean = evaluation(opt)
    return {
        'quality': quality,
        'bpp': float(mean['bpp']),
        'psnr_yuv': float(mean['psnr_yuv']),
        'psnr_y': float(mean['psnr_y']),
        'psnr': float(mean['psnr']),
        'ospcqm': float(mean['ospcqm']),
        'enc_time': float(mean['enc_time']),
        'dec_time': float(mean['dec_time']),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect RD points incrementally for curve plotting')
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='Model profiles (default: all five)')
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Eval dataset profiles (default: 8iVFBv2 sequences)')
    parser.add_argument(
        '--qualities', nargs='+', type=int, default=None,
        help='Quality indices (default: 0-5)')
    parser.add_argument('--train_dataset', type=str, default='coco3d')
    parser.add_argument('--epoch', type=str, default='las')
    parser.add_argument('--seed', type=int, default=777777)
    parser.add_argument(
        '--out_dir', type=str, default='result',
        help='Directory for per-dataset JSON results')
    parser.add_argument(
        '--force', action='store_true',
        help='Re-run and overwrite existing (model, quality) points')
    parser.add_argument('--verbose', action='store_true', default=False)
    # Convenience aliases for single-item runs
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    datasets = list(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    qualities = list(args.qualities) if args.qualities else list(DEFAULT_QUALITIES)

    if args.model is not None:
        models = [args.model]
    if args.dataset is not None:
        datasets = [args.dataset]

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    opt_base = SimpleNamespace(
        epoch=args.epoch,
        train_dataset=args.train_dataset,
        seed=args.seed,
        verbose=args.verbose,
        skip_reconstruct=True,
    )

    total = len(models) * len(datasets) * len(qualities)
    done = skipped = failed = 0
    print(
        f'Collect RD: {len(models)} models × {len(datasets)} datasets × '
        f'{len(qualities)} qualities = {total} jobs '
        f'(force={args.force}, out={out_dir})'
    )

    for dataset in datasets:
        path = result_path(out_dir, dataset)
        data = load_result(path)
        if not data:
            data = {
                'dataset': dataset,
                'train_dataset': args.train_dataset,
                'epoch': args.epoch,
                'curves': {},
            }
        else:
            data.setdefault('curves', {})
            data['dataset'] = dataset
            data['train_dataset'] = args.train_dataset
            data['epoch'] = args.epoch

        curves = data['curves']
        dirty = False

        for model in models:
            for quality in qualities:
                tag = f'{model} | {dataset} | q{quality}'
                if not args.force and has_point(curves, model, quality):
                    print(f'[skip] {tag}')
                    skipped += 1
                    continue

                if not checkpoint_exists(
                        args.train_dataset, model, quality, args.epoch):
                    print(
                        f'[miss] {tag} — checkpoint not found and '
                        f'SPCAC_HF_CHECKPOINT_REPO unset'
                    )
                    failed += 1
                    continue

                print(f'[run ] {tag}')
                try:
                    point = run_one(opt_base, model, dataset, quality)
                except Exception as exc:
                    print(f'[fail] {tag}: {exc}')
                    failed += 1
                    continue

                upsert_point(curves, model, point)
                dirty = True
                done += 1
                print(
                    f'       bpp={point["bpp"]:.4f}, '
                    f'psnr_yuv={point["psnr_yuv"]:.2f}, '
                    f'psnr_y={point["psnr_y"]:.2f}'
                )

                # Persist after each successful point for crash safety
                save_result(path, data)
                dirty = False

        if dirty:
            save_result(path, data)

        print(f'Saved: {path}')

    print(
        f'Done. ran={done}, skipped={skipped}, failed={failed}, '
        f'total={total}'
    )


if __name__ == '__main__':
    main()
