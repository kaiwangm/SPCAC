"""SPCAC evaluation viewer: pick a dataset, click an RD point, compare point clouds.

Reads ``metrics.json`` (rate–distortion) and optional downsampled previews
written by ``evaluation.export_previews``. Does not import MinkowskiEngine.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PREVIEW_DIR = Path(__file__).resolve().parent
REPO_DIR = (
    PREVIEW_DIR.parent
    if (PREVIEW_DIR.parent / "evaluation").is_dir()
    else PREVIEW_DIR
)

MODEL_DISPLAY = {
    "baseline_factorized": "factorized prior",
    "baseline_mean": "mean-scale hyperprior",
    "grouping": "grouping",
    "elpcac_l": "elpcac-l",
    "elpcac": "elpcac",
}

MODEL_STYLE = {
    "baseline_factorized": {"color": "#1f77b4", "marker": "circle"},
    "baseline_mean": {"color": "#ff7f0e", "marker": "square"},
    "grouping": {"color": "#2ca02c", "marker": "triangle-up"},
    "elpcac_l": {"color": "#d62728", "marker": "diamond"},
    "elpcac": {"color": "#9467bd", "marker": "triangle-down"},
}

PREFERRED_MODELS = list(MODEL_DISPLAY.keys())
DEFAULT_MODEL = "elpcac"
DEFAULT_QUALITY = 5
DEFAULT_SAMPLE = 0
ORIGINAL = "__original__"
MAX_PLOT_POINTS = 200_000  # Plotly/WebGL cannot render the full 0.8–2.4M clouds
DEFAULT_LANG = "en"
CLOUD_HEIGHT = 760
MULTI_SAMPLE_DATASETS: set[str] = set()
MULTI_SAMPLE_MAX = 1
PREVIEW_DATASET_PREFIXES = ("j8ivfbv2", "owlii")

UI = {
    "zh": {
        "header": (
            "<h1>SPCAC 基于稀疏张量的点云属性压缩 · 结果可视化</h1>\n\n"
            "左右预览可分别选择**原始点云**或任意**模型 / 质量**，方便两两对比。"
            "点击下方 RD 曲线上的点会加载到右侧窗口。"
        ),
        "no_metrics": (
            "未找到 `metrics.json`。请先运行 `evaluation.rd_collect`，"
            "结果应位于 `result/{dataset}/metrics.json`。"
        ),
        "dataset": "数据集",
        "sample": "样本序号",
        "rd_title": "RD 对比 — 点击曲线上的点加载点云",
        "rd_label": "率失真曲线",
        "x_title": "码率 (bpp)",
        "y_title": "PSNR-YUV (dB)",
        "quality": "质量",
        "models": "模型",
        "gallery": "缩略图（点击加载到右侧）",
        "original": "原始点云",
        "reconstructed": "重建点云",
        "original_caption": "原始",
        "left_view": "左侧",
        "right_view": "右侧",
        "pane_original": "原始",
        "vs": "对比",
        "pick_dataset": "请选择数据集。",
        "click_hint": "**数据集：** `{dataset}` — 点击上方 RD 曲线上的点以查看对应重建点云。",
        "no_result": "**数据集：** `{dataset}` — 没有 `{model}` q{quality} 的结果。",
        "selected": (
            "**数据集：** `{dataset}`  ·  **模型：** `{model}`  ·  "
            "**质量：** q{quality}  ·  **bpp：** {bpp:.4f}  ·  "
            "**PSNR-YUV：** {psnr_yuv:.2f} dB  ·  **PSNR-Y：** {psnr_y:.2f} dB"
        ),
        "preview_note": "\n\n点云预览尚未导出。在仓库根目录运行：\n```bash\nuv run python -m evaluation.export_previews --dataset {dataset}\n```",
        "missing_orig": "未找到点云预览。请先运行 `uv run python -m evaluation.export_previews`",
        "missing_rec": "未找到 {model} q{quality} 的重建预览",
        "missing_rec_sample": "未导出 {model} q{quality} 样本 #{sample} 的重建预览（当前仅样本 0 有重建）",
        "rec_click_title": "重建 — 点击 RD 曲线上的点",
        "rec_click_empty": "点击 RD 曲线上的点以加载该模型 / 质量的重建点云",
        "rec_title": "{model}  ·  q{quality}  ·  {bpp:.3f} bpp  ·  PSNR-YUV {psnr_yuv:.2f}  ·  PSNR-Y {psnr_y:.2f} dB",
    },
    "en": {
        "header": (
            "<h1>SPCAC Sparse Tensor-based Point Cloud Attribute Compression · Results</h1>\n\n"
            "Each pane can show the **original** cloud or any **model / quality**, "
            "so you can compare any pair. Clicking an RD point loads it into the right pane."
        ),
        "no_metrics": (
            "No `metrics.json` found. Run `evaluation.rd_collect` first; "
            "results belong at `result/{dataset}/metrics.json`."
        ),
        "dataset": "Dataset",
        "sample": "Sample",
        "rd_title": "RD comparison — click a point to load the cloud",
        "rd_label": "Rate–distortion curve",
        "x_title": "Rate (bpp)",
        "y_title": "PSNR-YUV (dB)",
        "quality": "Quality",
        "models": "Model",
        "gallery": "Thumbnails (click to load the right pane)",
        "original": "Original",
        "reconstructed": "Reconstructed",
        "original_caption": "original",
        "left_view": "Left",
        "right_view": "Right",
        "pane_original": "Original",
        "vs": "vs",
        "pick_dataset": "Please select a dataset.",
        "click_hint": "**Dataset:** `{dataset}` — click a point on the RD curve to view the reconstruction.",
        "no_result": "**Dataset:** `{dataset}` — no result for `{model}` q{quality}.",
        "selected": (
            "**Dataset:** `{dataset}`  ·  **Model:** `{model}`  ·  "
            "**Quality:** q{quality}  ·  **bpp:** {bpp:.4f}  ·  "
            "**PSNR-YUV:** {psnr_yuv:.2f} dB  ·  **PSNR-Y:** {psnr_y:.2f} dB"
        ),
        "preview_note": "\n\nPoint-cloud previews are not exported yet. From the repo root run:\n```bash\nuv run python -m evaluation.export_previews --dataset {dataset}\n```",
        "missing_orig": "No point-cloud preview. Run `uv run python -m evaluation.export_previews` first.",
        "missing_rec": "No reconstruction preview for {model} q{quality}",
        "missing_rec_sample": "No reconstruction preview for {model} q{quality} sample #{sample} (only sample 0 is exported)",
        "rec_click_title": "Reconstructed — click a point on the RD curve",
        "rec_click_empty": "Click a point on the RD curve to load that model / quality",
        "rec_title": "{model}  ·  q{quality}  ·  {bpp:.3f} bpp  ·  PSNR-YUV {psnr_yuv:.2f}  ·  PSNR-Y {psnr_y:.2f} dB",
    },
}


def ui(lang: str) -> dict:
    return UI.get(lang, UI[DEFAULT_LANG])


APP_CSS = """
h1, .prose h1 { font-size: 1.7rem; font-weight: 650; letter-spacing: 0; }
h1 sub, h1 sup, .prose h1 sub, .prose h1 sup { font-size: 1em !important; vertical-align: baseline; }
#orig-cloud, #rec-cloud { min-height: 780px; }
#left-quality, #right-quality { transition: opacity 0.15s ease; }
#left-quality:has(input:disabled), #right-quality:has(input:disabled) { opacity: 0.4; pointer-events: none; }
"""

CAMERA_SYNC_JS = """
<script>
(function () {
  function plotOf(id) {
    const root = document.getElementById(id);
    if (!root) return null;
    return root.querySelector(".js-plotly-plot");
  }
  function bind() {
    const a = plotOf("orig-cloud");
    const b = plotOf("rec-cloud");
    if (!a || !b || typeof Plotly === "undefined") return;
    if (a._spcacBound === b && b._spcacBound === a) return;
    a._spcacBound = b;
    b._spcacBound = a;
    let lock = false;
    const copy = (src, dst) => {
      if (lock || !src.layout || !src.layout.scene || !src.layout.scene.camera) return;
      lock = true;
      const cam = JSON.parse(JSON.stringify(src.layout.scene.camera));
      Plotly.relayout(dst, {"scene.camera": cam})
        .then(function () { lock = false; })
        .catch(function () { lock = false; });
    };
    a.on("plotly_relayouting", function () { copy(a, b); });
    b.on("plotly_relayouting", function () { copy(b, a); });
    a.on("plotly_relayout", function () { copy(a, b); });
    b.on("plotly_relayout", function () { copy(b, a); });
  }
  function start() {
    bind();
    if (document.body) {
      new MutationObserver(bind).observe(document.body, { childList: true, subtree: true });
    }
    setInterval(bind, 1000);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
</script>
"""


def _first_existing(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_dir():
            return path
    return None


def metrics_root() -> Path:
    env = os.environ.get("SPCAC_RESULT_DIR")
    if env:
        return Path(env)
    return REPO_DIR / "result"


def preview_root() -> Path:
    env = os.environ.get("SPCAC_PREVIEW_DIR")
    if env:
        return Path(env)
    repo_result = REPO_DIR / "result"
    legacy = REPO_DIR / "previews"
    found = _first_existing(repo_result, legacy)
    return found or repo_result


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def list_datasets() -> list[str]:
    root = metrics_root()
    names = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if (child / "metrics.json").is_file():
                names.append(child.name)
    return [n for n in names if n.startswith(PREVIEW_DATASET_PREFIXES)]


def metrics_path(dataset: str) -> Path:
    return metrics_root() / dataset / "metrics.json"


@lru_cache(maxsize=32)
def load_metrics(dataset: str) -> dict:
    path = metrics_path(dataset)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ordered_models(curves: dict) -> list[str]:
    models = [m for m in PREFERRED_MODELS if m in curves]
    models += [m for m in sorted(curves.keys()) if m not in models]
    return models


def lookup_point(curves: dict, model: str, quality: int) -> dict | None:
    for point in curves.get(model, []):
        if int(point.get("quality", -1)) == quality:
            return point
    return None


def available_qualities(curves: dict, model: str | None = None) -> list[int]:
    qualities = set()
    models = [model] if model else curves.keys()
    for name in models:
        for point in curves.get(name, []):
            qualities.add(int(point["quality"]))
    return sorted(qualities)


def default_selection(dataset: str) -> tuple[str | None, int | None]:
    if not dataset:
        return None, None
    curves = load_metrics(dataset).get("curves", {})
    if lookup_point(curves, DEFAULT_MODEL, DEFAULT_QUALITY):
        return DEFAULT_MODEL, DEFAULT_QUALITY
    models = ordered_models(curves)
    if not models:
        return None, None
    model = models[-1]
    qualities = available_qualities(curves, model)
    quality = DEFAULT_QUALITY if DEFAULT_QUALITY in qualities else (qualities[0] if qualities else None)
    return model, quality


def hdf5_path(dataset: str) -> Path | None:
    path = REPO_DIR / "datasets" / f"{dataset}.hdf5"
    return path if path.is_file() else None


@lru_cache(maxsize=8)
def hdf5_n_test(dataset: str) -> int | None:
    path = hdf5_path(dataset)
    if path is None:
        return None
    try:
        import h5py
    except ImportError:
        return None
    with h5py.File(path, "r") as f:
        split = f["dataset"]["test"]
        return int(len(split))


def list_sample_indices(dataset: str) -> list[int]:
    if not dataset:
        return [DEFAULT_SAMPLE]
    if dataset in MULTI_SAMPLE_DATASETS:
        meta = preview_root() / dataset / "samples.json"
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            if "indices" in data:
                return [int(i) for i in data["indices"]]
        n = hdf5_n_test(dataset)
        if n:
            return list(range(min(n, MULTI_SAMPLE_MAX)))
        return list(range(MULTI_SAMPLE_MAX))
    root = preview_root() / dataset
    indices = set()
    if (root / "original.npz").is_file() or (root / "s0" / "original.npz").is_file():
        indices.add(0)
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("s") and child.name[1:].isdigit():
                if (child / "original.npz").is_file():
                    indices.add(int(child.name[1:]))
    return sorted(indices) or [DEFAULT_SAMPLE]


def is_multi_sample(dataset: str) -> bool:
    return len(list_sample_indices(dataset)) > 1


def coerce_sample(dataset: str, sample: int | str | None) -> int:
    indices = list_sample_indices(dataset)
    try:
        value = int(sample)
    except (TypeError, ValueError):
        value = DEFAULT_SAMPLE
    return value if value in indices else (indices[0] if indices else DEFAULT_SAMPLE)


def sample_tag(dataset: str, sample: int) -> str:
    if not is_multi_sample(dataset):
        return ""
    return f"#{int(sample)}"


def preview_npz(
    dataset: str,
    model: str | None = None,
    quality: int | None = None,
    sample: int = DEFAULT_SAMPLE,
) -> Path:
    root = preview_root() / dataset
    name = "original.npz" if model is None else f"{model}_q{quality}.npz"
    sample = int(sample or 0)
    nested = root / f"s{sample}" / name
    if nested.is_file():
        return nested
    if sample == 0:
        return root / name
    return nested


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb
    if rgb.size and float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _cap_points(xyz: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if MAX_PLOT_POINTS > 0 and xyz.shape[0] > MAX_PLOT_POINTS:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(xyz.shape[0], MAX_PLOT_POINTS, replace=False))
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


@lru_cache(maxsize=8)
def _load_cloud_cached(path_str: str, mtime: float) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(path_str)
    with np.load(path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = _as_uint8_rgb(np.asarray(data["rgb"]))
    if xyz.ndim != 2 or xyz.shape[0] == 0:
        return None
    return _cap_points(xyz, rgb)


def load_cloud(path_str: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    return _load_cloud_cached(path_str, path.stat().st_mtime)


@lru_cache(maxsize=16)
def _load_hdf5_original_cached(dataset: str, sample: int, mtime: float) -> tuple[np.ndarray, np.ndarray] | None:
    path = hdf5_path(dataset)
    if path is None:
        return None
    try:
        import h5py
    except ImportError:
        return None
    with h5py.File(path, "r") as f:
        split = f["dataset"]["test"]
        key = str(int(sample))
        if key not in split:
            return None
        xyz = np.asarray(split[key]["points"][:], dtype=np.float32).reshape(-1, 3)
        rgb = _as_uint8_rgb(np.asarray(split[key]["colors"][:]).reshape(-1, 3))
    if xyz.shape[0] == 0:
        return None
    return _cap_points(xyz, rgb)


def load_hdf5_original(dataset: str, sample: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = hdf5_path(dataset)
    if path is None:
        return None
    return _load_hdf5_original_cached(dataset, int(sample), path.stat().st_mtime)


def load_preview_cloud(
    dataset: str,
    model: str | None = None,
    quality: int | None = None,
    sample: int = DEFAULT_SAMPLE,
) -> tuple[np.ndarray, np.ndarray] | None:
    loaded = load_cloud(str(preview_npz(dataset, model, quality, sample)))
    if loaded is not None:
        return loaded
    if model is None:
        return load_hdf5_original(dataset, sample)
    return None


def has_preview(
    dataset: str,
    model: str | None = None,
    quality: int | None = None,
    sample: int = DEFAULT_SAMPLE,
) -> bool:
    if preview_npz(dataset, model, quality, sample).is_file():
        return True
    if model is None:
        n = hdf5_n_test(dataset)
        return n is not None and 0 <= int(sample) < n
    return False


def scene_layout(dataset: str) -> dict:
    if dataset.startswith("j8ivfbv2") or dataset.startswith("owlii"):
        # Tall human figures (Y-up). Default Plotly eye (~2.2) clips head/feet.
        camera = dict(up=dict(x=0, y=1, z=0), eye=dict(x=2.7, y=0.75, z=2.55))
    else:
        camera = dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.5, y=-1.45, z=1.2))
    axis = dict(visible=False, showbackground=False)
    return dict(
        aspectmode="data",
        xaxis=axis,
        yaxis=axis,
        zaxis=axis,
        camera=camera,
    )


def empty_fig(message: str, *, is_3d: bool = False, height: int | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#666"),
    )
    layout = dict(
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height if height is not None else (CLOUD_HEIGHT if is_3d else 420),
    )
    if is_3d:
        layout["scene"] = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    fig.update_layout(**layout)
    return fig


def cloud_fig(
    dataset: str,
    title: str,
    lang: str,
    model: str | None = None,
    quality: int | None = None,
    sample: int = DEFAULT_SAMPLE,
) -> go.Figure:
    t = ui(lang)
    try:
        loaded = load_preview_cloud(dataset, model, quality, sample)
    except Exception as exc:
        return empty_fig(f"{type(exc).__name__}: {exc}", is_3d=True)
    if loaded is None:
        if model is None:
            hint = t["missing_orig"]
        elif is_multi_sample(dataset) and int(sample) != 0:
            hint = t["missing_rec_sample"].format(
                model=display_name(model), quality=quality, sample=sample,
            )
        else:
            hint = t["missing_rec"].format(model=display_name(model), quality=quality)
        return empty_fig(hint, is_3d=True)

    xyz, rgb = loaded
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    n = int(xyz.shape[0])
    marker_size = 1.6 if n < 80000 else 1.15
    rgb_u8 = np.ascontiguousarray(rgb, dtype=np.uint8)
    colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_u8]
    fig = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=colors, opacity=1.0),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=0, r=0, t=48, b=0),
        scene=scene_layout(dataset),
        height=CLOUD_HEIGHT,
        uirevision=f"{dataset}-s{sample}-cloud-sync",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_scenes(dragmode="orbit")
    return fig


def xy_thumb(xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    from io import BytesIO

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    if xyz.shape[0] > 40000:
        rng = np.random.default_rng(0)
        pick = rng.choice(xyz.shape[0], 40000, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    fig, ax = plt.subplots(figsize=(2.2, 2.8), dpi=72)
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        c=rgb.astype(np.float32) / 255.0,
        s=0.18,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def rd_table(dataset: str) -> tuple[pd.DataFrame, list[dict]]:
    columns = ["model", "model_id", "quality", "bpp", "psnr_yuv", "psnr_y"]
    if not dataset:
        return pd.DataFrame(columns=columns), []
    curves = load_metrics(dataset).get("curves", {})
    records: list[dict] = []
    for model in ordered_models(curves):
        for point in sorted(curves[model], key=lambda p: int(p["quality"])):
            bpp = point.get("bpp")
            psnr = point.get("psnr_yuv")
            if bpp is None or psnr is None or float(psnr) == 0.0:
                continue
            records.append({
                "model": display_name(model),
                "model_id": model,
                "quality": int(point["quality"]),
                "bpp": float(bpp),
                "psnr_yuv": float(psnr),
                "psnr_y": float(point.get("psnr_y") or 0.0),
            })
    return pd.DataFrame(records, columns=columns), records


def info_markdown(
    dataset: str,
    left_src: str | None,
    left_q: int | None,
    right_src: str | None,
    right_q: int | None,
    lang: str,
    sample: int = DEFAULT_SAMPLE,
) -> str:
    t = ui(lang)
    if not dataset:
        return t["pick_dataset"]
    left = pane_caption(dataset, left_src, left_q, lang, sample)
    right = pane_caption(dataset, right_src, right_q, lang, sample)
    tag = sample_tag(dataset, sample)
    sample_line = f"**{t['sample']}:** {tag}  \n" if tag else ""
    return f"{sample_line}**{t['left_view']}:** {left}  \n**{t['right_view']}:** {right}"


def pane_caption(
    dataset: str, source: str | None, quality: int | None, lang: str, sample: int = DEFAULT_SAMPLE,
) -> str:
    t = ui(lang)
    tag = sample_tag(dataset, sample)
    suffix = f" · {tag}" if tag else ""
    if not source or source == ORIGINAL:
        return t["pane_original"] + suffix
    curves = load_metrics(dataset).get("curves", {}) if dataset else {}
    point = lookup_point(curves, source, quality) if quality is not None else None
    if point is None:
        return t["no_result"].format(dataset=dataset, model=display_name(source), quality=quality)
    return (
        f"`{display_name(source)}` · q{quality}{suffix} · "
        f"{float(point['bpp']):.4f} bpp · "
        f"{float(point['psnr_yuv']):.2f} dB PSNR-YUV · "
        f"{float(point.get('psnr_y') or 0.0):.2f} dB PSNR-Y"
    )


def source_update(dataset: str, quality: int | None, selected: str | None, lang: str, label: str):
    t = ui(lang)
    choices = [(t["pane_original"], ORIGINAL)]
    if dataset:
        curves = load_metrics(dataset).get("curves", {})
        for model in ordered_models(curves):
            point = lookup_point(curves, model, quality) if quality is not None else None
            if point is not None:
                name = f"{display_name(model)}  ·  {float(point['bpp']):.3f} bpp"
            else:
                name = display_name(model)
            choices.append((name, model))
    value = selected if any(item == selected for _, item in choices) else ORIGINAL
    return gr.update(choices=choices, value=value, label=label)


def quality_update(
    dataset: str,
    selected_quality: int | None,
    lang: str,
    label: str,
    *,
    interactive: bool = True,
):
    t = ui(lang)
    if not dataset:
        return gr.update(choices=[], value=None, label=label, interactive=False)
    curves = load_metrics(dataset).get("curves", {})
    qualities = available_qualities(curves)
    choices = [(f"q{q}", q) for q in qualities]
    value = selected_quality if selected_quality in qualities else (
        qualities[0] if qualities else None
    )
    return gr.update(choices=choices, value=value, label=label, interactive=interactive)


def sample_update(dataset: str, sample: int | None, lang: str):
    t = ui(lang)
    indices = list_sample_indices(dataset)
    multi = len(indices) > 1
    value = coerce_sample(dataset, sample)
    last = indices[-1] if indices else 0
    label = t["sample"] if not multi else f"{t['sample']}  (0–{last})"
    return gr.update(
        choices=[(str(i), i) for i in indices],
        value=value,
        label=label,
        visible=multi,
        interactive=multi,
    )


def gallery_items(
    dataset: str, quality: int | None, lang: str, sample: int = DEFAULT_SAMPLE,
) -> list[tuple[np.ndarray, str]]:
    t = ui(lang)
    items: list[tuple[np.ndarray, str]] = []
    if not dataset:
        return items
    try:
        orig = load_preview_cloud(dataset, sample=sample)
        if orig is not None:
            items.append((xy_thumb(*orig), t["original_caption"]))
        if quality is None:
            return items
        curves = load_metrics(dataset).get("curves", {})
        for model in ordered_models(curves):
            point = lookup_point(curves, model, quality)
            loaded = load_preview_cloud(dataset, model, quality, sample)
            if loaded is None or point is None:
                continue
            caption = f"{display_name(model)}  ·  {float(point['bpp']):.3f} bpp"
            items.append((xy_thumb(*loaded), caption))
    except Exception as exc:
        print(f"[gallery] {dataset}: {type(exc).__name__}: {exc}", flush=True)
    return items


def pane_metrics(dataset: str, source: str | None, quality: int | None) -> dict | None:
    if not dataset or not source or source == ORIGINAL or quality is None:
        return None
    return lookup_point(load_metrics(dataset).get("curves", {}), source, quality)


def pane_figure(
    dataset: str, source: str | None, quality: int | None, lang: str, sample: int = DEFAULT_SAMPLE,
) -> go.Figure:
    t = ui(lang)
    tag = sample_tag(dataset, sample)
    prefix = f"{tag}  ·  " if tag else ""
    if not dataset:
        return empty_fig(t["pick_dataset"], is_3d=True)
    if not source or source == ORIGINAL:
        return cloud_fig(dataset, f"{prefix}{t['pane_original']}", lang, sample=sample)
    point = pane_metrics(dataset, source, quality)
    bpp = float(point["bpp"]) if point else float("nan")
    psnr_yuv = float(point["psnr_yuv"]) if point else float("nan")
    psnr_y = float(point.get("psnr_y") or float("nan")) if point else float("nan")
    title = prefix + t["rec_title"].format(
        model=display_name(source),
        quality=quality,
        bpp=bpp,
        psnr_yuv=psnr_yuv,
        psnr_y=psnr_y,
    )
    return cloud_fig(dataset, title, lang, source, quality, sample)


def pane_label(
    dataset: str, source: str | None, quality: int | None, lang: str, side: str, sample: int = DEFAULT_SAMPLE,
) -> str:
    t = ui(lang)
    tag = sample_tag(dataset, sample)
    extra = f" · {tag}" if tag else ""
    if not source or source == ORIGINAL:
        return f"{t[side]} · {t['pane_original']}{extra}"
    point = pane_metrics(dataset, source, quality)
    if point is None:
        return f"{t[side]} · {display_name(source)} q{quality}{extra}"
    return (
        f"{t[side]} · {display_name(source)} q{quality}{extra} · "
        f"{float(point['bpp']):.3f} bpp · "
        f"YUV {float(point['psnr_yuv']):.2f} · "
        f"Y {float(point.get('psnr_y') or 0.0):.2f} dB"
    )


def render(
    dataset: str,
    left_src: str | None,
    left_q: int | None,
    right_src: str | None,
    right_q: int | None,
    lang: str = DEFAULT_LANG,
    sample: int | None = DEFAULT_SAMPLE,
):
    t = ui(lang)
    left_src = left_src or ORIGINAL
    right_src = right_src or ORIGINAL
    sample = coerce_sample(dataset, sample)
    if left_src == ORIGINAL:
        left_q = DEFAULT_QUALITY
    if right_src == ORIGINAL:
        right_q = DEFAULT_QUALITY
    header = t["header"]
    dataset_upd = gr.update(label=t["dataset"])
    sample_upd = sample_update(dataset, sample, lang)
    empty_df = pd.DataFrame(columns=["model", "model_id", "quality", "bpp", "psnr_yuv", "psnr_y"])

    if not dataset:
        empty3d = empty_fig(t["pick_dataset"], is_3d=True)
        return (
            header,
            dataset_upd,
            sample_upd,
            gr.update(value=empty_df, title=t["rd_title"], x_title=t["x_title"], y_title=t["y_title"], label=t["rd_label"]),
            [],
            t["pick_dataset"],
            gr.update(value=empty3d, label=pane_label(dataset, left_src, left_q, lang, "left_view", sample)),
            gr.update(value=empty3d, label=pane_label(dataset, right_src, right_q, lang, "right_view", sample)),
            gr.update(value=[], label=t["gallery"]),
            source_update(dataset, left_q, left_src, lang, t["left_view"]),
            quality_update(dataset, left_q, lang, t["quality"], interactive=left_src != ORIGINAL),
            source_update(dataset, right_q, right_src, lang, t["right_view"]),
            quality_update(dataset, right_q, lang, t["quality"], interactive=right_src != ORIGINAL),
            left_src,
            left_q,
            right_src,
            right_q,
            sample,
        )

    df, records = rd_table(dataset)
    return (
        header,
        dataset_upd,
        sample_upd,
        gr.update(value=df, title=t["rd_title"], x_title=t["x_title"], y_title=t["y_title"], label=t["rd_label"]),
        records,
        info_markdown(dataset, left_src, left_q, right_src, right_q, lang, sample),
        gr.update(value=pane_figure(dataset, left_src, left_q, lang, sample), label=pane_label(dataset, left_src, left_q, lang, "left_view", sample)),
        gr.update(value=pane_figure(dataset, right_src, right_q, lang, sample), label=pane_label(dataset, right_src, right_q, lang, "right_view", sample)),
        gr.update(value=gallery_items(dataset, right_q, lang, sample), label=t["gallery"]),
        source_update(dataset, left_q, left_src, lang, t["left_view"]),
        quality_update(dataset, left_q, lang, t["quality"], interactive=left_src != ORIGINAL),
        source_update(dataset, right_q, right_src, lang, t["right_view"]),
        quality_update(dataset, right_q, lang, t["quality"], interactive=right_src != ORIGINAL),
        left_src,
        left_q,
        right_src,
        right_q,
        sample,
    )


def default_panes(dataset: str) -> tuple[str, int | None, str, int | None]:
    model, quality = default_selection(dataset)
    if lookup_point(load_metrics(dataset).get("curves", {}), DEFAULT_MODEL, DEFAULT_QUALITY):
        model, quality = DEFAULT_MODEL, DEFAULT_QUALITY
    return ORIGINAL, quality, (model or ORIGINAL), quality


def parse_rd_select(evt: gr.SelectData, records: list) -> dict | None:
    if not records:
        return None
    idx = evt.index
    if isinstance(idx, (list, tuple)) and idx:
        idx = idx[0]
    if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < len(records):
        return records[int(idx)]

    val = evt.value
    if isinstance(val, dict):
        model_id = val.get("model_id")
        quality = val.get("quality")
        model_name = val.get("model") or val.get("color")
        if model_id is not None and quality is not None:
            for rec in records:
                if rec["model_id"] == model_id and rec["quality"] == int(quality):
                    return rec
        if model_name is not None and quality is not None:
            for rec in records:
                if rec["model"] == model_name and rec["quality"] == int(quality):
                    return rec
        x = val.get("bpp", val.get("x"))
        y = val.get("psnr_yuv", val.get("y"))
        if x is not None and y is not None:
            return _nearest_record(records, float(x), float(y))
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        try:
            return _nearest_record(records, float(val[0]), float(val[1]))
        except (TypeError, ValueError):
            pass
    return None


def _nearest_record(records: list[dict], x: float, y: float) -> dict | None:
    best = None
    best_d = 1e18
    for rec in records:
        d = abs(rec["bpp"] - x) + 0.05 * abs(rec["psnr_yuv"] - y)
        if d < best_d:
            best_d = d
            best = rec
    return best


def on_dataset_change(dataset: str, lang: str):
    left_src, left_q, right_src, right_q = default_panes(dataset)
    return render(dataset, left_src, left_q, right_src, right_q, lang, DEFAULT_SAMPLE)


def on_lang_change(
    dataset: str, left_src: str, left_q: int, right_src: str, right_q: int, sample: int, lang: str,
):
    return render(dataset, left_src, left_q, right_src, right_q, lang or DEFAULT_LANG, sample)


def on_rd_click(
    dataset: str, records: list, left_src: str, left_q: int, sample: int, lang: str, evt: gr.SelectData,
):
    rec = parse_rd_select(evt, records or [])
    if rec is None:
        _, _, right_src, right_q = default_panes(dataset)
        return render(dataset, left_src, left_q, right_src, right_q, lang, sample)
    return render(dataset, left_src, left_q, rec["model_id"], rec["quality"], lang, sample)


N_OUTPUTS = 18


def _noop_outputs():
    return tuple(gr.update() for _ in range(N_OUTPUTS))


def on_sample_change(
    dataset: str, left_src: str, left_q: int, right_src: str, right_q: int, sample: int, current: int, lang: str,
):
    sample = coerce_sample(dataset, sample)
    try:
        current = int(current)
    except (TypeError, ValueError):
        current = DEFAULT_SAMPLE
    if sample == current:
        return _noop_outputs()
    return render(dataset, left_src, left_q, right_src, right_q, lang, sample)


def on_left_source(
    dataset: str, left_src: str, left_q: int, right_src: str, right_q: int, current: str, sample: int, lang: str,
):
    if not left_src or left_src == current:
        return _noop_outputs()
    return render(dataset, left_src, left_q, right_src, right_q, lang, sample)


def on_left_quality(
    dataset: str, left_src: str, left_q: int, right_src: str, right_q: int, current: int, sample: int, lang: str,
):
    if left_q is None or left_q == current:
        return _noop_outputs()
    return render(dataset, left_src, left_q, right_src, right_q, lang, sample)


def on_right_source(
    dataset: str, left_src: str, left_q: int, right_src: str, right_q: int, current: str, sample: int, lang: str,
):
    if not right_src or right_src == current:
        return _noop_outputs()
    return render(dataset, left_src, left_q, right_src, right_q, lang, sample)


def on_right_quality(
    dataset: str, left_src: str, left_q: int, right_src: str, right_q: int, current: int, sample: int, lang: str,
):
    if right_q is None or right_q == current:
        return _noop_outputs()
    return render(dataset, left_src, left_q, right_src, right_q, lang, sample)


def on_gallery_select(
    dataset: str, left_src: str, left_q: int, right_src: str, right_q: int, sample: int, lang: str, evt: gr.SelectData,
):
    curves = load_metrics(dataset).get("curves", {}) if dataset else {}
    models = ordered_models(curves)
    index = evt.index
    if isinstance(index, (list, tuple)):
        index = index[0]
    index = int(index) if index is not None else 0
    orig_offset = 1 if has_preview(dataset, sample=sample) else 0
    if index < orig_offset:
        return render(dataset, left_src, left_q, ORIGINAL, right_q, lang, sample)
    available = [
        name for name in models
        if lookup_point(curves, name, right_q) and has_preview(dataset, name, right_q, sample)
    ]
    model_index = index - orig_offset
    if right_q is None or model_index >= len(available):
        return render(dataset, left_src, left_q, right_src, right_q, lang, sample)
    return render(dataset, left_src, left_q, available[model_index], right_q, lang, sample)


def build_demo() -> gr.Blocks:
    datasets = list_datasets()
    initial = datasets[0] if datasets else None
    init_left, init_left_q, init_right, init_right_q = default_panes(initial) if initial else (
        ORIGINAL, DEFAULT_QUALITY, DEFAULT_MODEL, DEFAULT_QUALITY,
    )
    init_sample = DEFAULT_SAMPLE
    t0 = ui(DEFAULT_LANG)
    init_indices = list_sample_indices(initial) if initial else [DEFAULT_SAMPLE]

    with gr.Blocks(title="SPCAC Evaluation") as demo:
        header = gr.Markdown(t0["header"], latex_delimiters=[], sanitize_html=False)
        if not datasets:
            gr.Markdown(t0["no_metrics"], latex_delimiters=[])

        lookup_state = gr.State([])
        left_src_state = gr.State(init_left)
        left_q_state = gr.State(init_left_q)
        right_src_state = gr.State(init_right)
        right_q_state = gr.State(init_right_q)
        sample_state = gr.State(init_sample)

        with gr.Row():
            dataset = gr.Dropdown(
                choices=datasets,
                value=initial,
                label=t0["dataset"],
                interactive=True,
                scale=3,
            )
            sample = gr.Dropdown(
                choices=[(str(i), i) for i in init_indices],
                value=init_sample,
                label=t0["sample"],
                interactive=True,
                visible=len(init_indices) > 1,
                scale=1,
            )

        with gr.Row():
            with gr.Column():
                left_source = gr.Dropdown(label=t0["left_view"], interactive=True)
                left_quality = gr.Radio(label=t0["quality"], interactive=True, elem_id="left-quality")
                left_plot = gr.Plot(label=t0["left_view"], elem_id="orig-cloud")
            with gr.Column():
                right_source = gr.Dropdown(label=t0["right_view"], interactive=True)
                right_quality = gr.Radio(label=t0["quality"], interactive=True, elem_id="right-quality")
                right_plot = gr.Plot(label=t0["right_view"], elem_id="rec-cloud")

        info = gr.Markdown()
        gallery = gr.Gallery(
            label=t0["gallery"],
            columns=6,
            height=200,
            allow_preview=False,
            object_fit="contain",
        )
        rd_plot = gr.LinePlot(
            x="bpp",
            y="psnr_yuv",
            color="model",
            color_map={display_name(k): v["color"] for k, v in MODEL_STYLE.items()},
            title=t0["rd_title"],
            x_title=t0["x_title"],
            y_title=t0["y_title"],
            tooltip=["model", "quality", "bpp", "psnr_yuv"],
            height=420,
            label=t0["rd_label"],
        )

        lang = gr.Radio(
            choices=[("English", "en"), ("中文", "zh")],
            value=DEFAULT_LANG,
            label="Language / 语言",
            interactive=True,
        )

        outputs = [
            header,
            dataset,
            sample,
            rd_plot,
            lookup_state,
            info,
            left_plot,
            right_plot,
            gallery,
            left_source,
            left_quality,
            right_source,
            right_quality,
            left_src_state,
            left_q_state,
            right_src_state,
            right_q_state,
            sample_state,
        ]

        demo.load(
            fn=lambda: render(initial, init_left, init_left_q, init_right, init_right_q, DEFAULT_LANG, init_sample),
            outputs=outputs,
        )
        lang.change(
            fn=on_lang_change,
            inputs=[dataset, left_src_state, left_q_state, right_src_state, right_q_state, sample_state, lang],
            outputs=outputs,
        )
        dataset.change(
            fn=on_dataset_change,
            inputs=[dataset, lang],
            outputs=outputs,
        )
        sample.change(
            fn=on_sample_change,
            inputs=[dataset, left_src_state, left_q_state, right_src_state, right_q_state, sample, sample_state, lang],
            outputs=outputs,
        )
        rd_plot.select(
            fn=on_rd_click,
            inputs=[dataset, lookup_state, left_src_state, left_q_state, sample_state, lang],
            outputs=outputs,
        )
        left_source.change(
            fn=on_left_source,
            inputs=[dataset, left_source, left_q_state, right_src_state, right_q_state, left_src_state, sample_state, lang],
            outputs=outputs,
        )
        left_quality.change(
            fn=on_left_quality,
            inputs=[dataset, left_src_state, left_quality, right_src_state, right_q_state, left_q_state, sample_state, lang],
            outputs=outputs,
        )
        right_source.change(
            fn=on_right_source,
            inputs=[dataset, left_src_state, left_q_state, right_source, right_q_state, right_src_state, sample_state, lang],
            outputs=outputs,
        )
        right_quality.change(
            fn=on_right_quality,
            inputs=[dataset, left_src_state, left_q_state, right_src_state, right_quality, right_q_state, sample_state, lang],
            outputs=outputs,
        )
        gallery.select(
            fn=on_gallery_select,
            inputs=[dataset, left_src_state, left_q_state, right_src_state, right_q_state, sample_state, lang],
            outputs=outputs,
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=APP_CSS,
        head=CAMERA_SYNC_JS,
        show_error=True,
        ssr_mode=False,
    )
