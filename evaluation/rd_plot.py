"""Plot RD curves from JSON collected by ``evaluation.rd_collect``.

For each ``result/{dataset}/metrics.json``, writes ``psnr_yuv.png``
into the same dataset folder (no quality labels on points).

Usage examples:
    # Plot everything under the default result directory
    uv run python -m evaluation.rd_plot

    # One dataset
    uv run python -m evaluation.rd_plot --dataset j8ivfbv2-longdress10

    # Custom input / output
    uv run python -m evaluation.rd_plot --in_dir result --out_dir result/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

MODEL_DISPLAY = {
    'baseline_factorized': 'factorized prior',
    'baseline_mean': 'mean-scale hyperprior',
    'grouping': 'grouping',
    'elpcac_l': 'elpcac-l',
    'elpcac': 'elpcac',
}

# Stable color / marker cycle for known models; extras fall back to C-cycle
MODEL_STYLE = {
    'baseline_factorized': {'color': '#1f77b4', 'marker': 'o'},
    'baseline_mean': {'color': '#ff7f0e', 'marker': 's'},
    'grouping': {'color': '#2ca02c', 'marker': '^'},
    'elpcac_l': {'color': '#d62728', 'marker': 'D'},
    'elpcac': {'color': '#9467bd', 'marker': 'v'},
}

METRICS = [
    ('psnr_yuv', 'PSNR-YUV (dB)', 'psnr_yuv'),
]


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model).lower()


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def discover_jsons(in_dir: Path, dataset: str | None) -> list[Path]:
    """Find ``{in_dir}/{dataset}/metrics.json`` (or legacy ``{dataset}.json``)."""
    if not in_dir.exists():
        return []

    paths: list[Path] = []
    if dataset:
        for candidate in (
            in_dir / dataset / 'metrics.json',
            in_dir / f'{dataset}.json',
        ):
            if candidate.is_file():
                paths.append(candidate)
                break
        return paths

    # Prefer per-dataset folders named after the eval dataset
    for child in sorted(in_dir.iterdir()):
        if not child.is_dir():
            continue
        metrics = child / 'metrics.json'
        if metrics.is_file():
            paths.append(metrics)

    # Legacy flat / train_dataset-nested JSON
    if not paths:
        paths.extend(sorted(in_dir.glob('*.json')))
        paths.extend(sorted(in_dir.glob('*/*.json')))

    seen = set()
    unique = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def plot_dataset(data: dict, out_dir: Path, dpi: int) -> list[Path]:
    dataset = data.get('dataset', 'unknown')
    curves = data.get('curves', {})
    if not curves:
        print(f'[warn] no curves in dataset={dataset}, skip')
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Prefer a fixed model order when available
    preferred = list(MODEL_DISPLAY.keys())
    models = [m for m in preferred if m in curves]
    models += [m for m in sorted(curves.keys()) if m not in models]

    for metric_key, ylabel, suffix in METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plotted = 0

        for idx, model in enumerate(models):
            points = sorted(curves[model], key=lambda p: int(p['quality']))
            xs, ys = [], []
            for p in points:
                bpp = p.get('bpp')
                val = p.get(metric_key)
                if bpp is None or val is None:
                    continue
                # Ignore placeholder zeros for unused metrics
                if metric_key == 'psnr_yuv' and float(val) == 0.0:
                    continue
                xs.append(float(bpp))
                ys.append(float(val))

            if len(xs) < 1:
                continue

            style = MODEL_STYLE.get(model, {
                'color': f'C{idx % 10}',
                'marker': 'o',
            })
            ax.plot(
                xs, ys,
                label=display_name(model),
                color=style['color'],
                marker=style['marker'],
                linewidth=1.8,
                markersize=6,
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_title(f'RD Comparison — {dataset}')
        ax.set_xlabel('Rate (bpp)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()

        out_path = out_dir / f'{suffix}.png'
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        saved.append(out_path)
        print(f'Saved: {out_path}')

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot RD curves from collected JSON results')
    parser.add_argument(
        '--in_dir', type=str, default='result',
        help='Directory produced by evaluation.rd_collect')
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help='Unused when writing into each dataset folder; kept for CLI compat')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument(
        '--json', type=str, nargs='+', default=None,
        help='Explicit JSON file(s) to plot')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)

    if args.json:
        paths = [Path(p) for p in args.json]
    else:
        paths = discover_jsons(in_dir, args.dataset)

    if not paths:
        print(f'No JSON found under {in_dir} (dataset={args.dataset})')
        return

    total = 0
    for path in paths:
        print(f'Plot: {path}')
        data = load_json(path)
        dataset = data.get('dataset') or path.parent.name
        # Write figures into the test-dataset folder next to metrics.json
        if path.name == 'metrics.json':
            plot_root = path.parent
        else:
            plot_root = in_dir / dataset
        saved = plot_dataset(data, plot_root, dpi=args.dpi)
        total += len(saved)

    print(f'Done. wrote {total} figure(s) under {in_dir}/{{dataset}}/')


if __name__ == '__main__':
    main()
