# SPCAC: Sparse Tensor-based Point Cloud Attribute Compression

**English** | [中文](README-zh.md)

[![Hugging Face Models](https://img.shields.io/badge/Models-HuggingFace-gold.svg)](https://huggingface.co/kaiwangm/SPCAC-checkpoints)
[![Hugging Face Datasets](https://img.shields.io/badge/Datasets-HuggingFace-gold.svg)](https://huggingface.co/datasets/kaiwangm/SPCAC-datasets)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A baseline model repository for point cloud attribute compression: learning-based coding of 3D point-cloud color attributes with sparse convolution and entropy modeling.
>
> Datasets were collected and models trained from March to December 2023.
>
> The repository was reorganized and released publicly in August 2026, with checkpoints and datasets published on Hugging Face.

## Overview

This repository is a **baseline suite for point cloud attribute compression**. Given known geometry, it compresses color attributes (RGB) under a rate–distortion objective (configurable λ, quality levels q0–q6). It uses **MinkowskiEngine** sparse convolutions and **CompressAI** entropy models, provides 5 models, and covers the full pipeline: training → CDF update → entropy-coded compress/decompress.

**Factorized prior.** Models latent channels independently.

**Hyperprior.** Uses side information to predict Gaussian means and scales, capturing spatial dependencies.

**Grouped encoding / decoding.** Points are split into three groups by coordinate parity and coded progressively, with already-decoded groups as spatial context for later groups.

**Geometry guidance.** Extracts context from occupancy geometry to aid attribute coding.

**Local hyperprior.** Encodes spatial side information from the latent.

**Global hyperprior.** Encodes globally aggregated side information from the latent.

**Cross-attention.** Fuses the global hyperprior with local context to produce entropy parameters.

**Residual blocks and self-attention.** Used in the encoder / decoder.

## Hugging Face Resources

| Resource | Link | Contents |
|----------|------|----------|
| Model checkpoints | [🤗 SPCAC-checkpoints](https://huggingface.co/kaiwangm/SPCAC-checkpoints) | Final checkpoints (`eb_las.pth`) for model × dataset × quality (q0–q6) |
| Datasets | [🤗 SPCAC-datasets](https://huggingface.co/datasets/kaiwangm/SPCAC-datasets) | HDF5 datasets and PLY test splits |

```
SPCAC-checkpoints/
└── {dataset}/{model}/quality_{q}/eb_las.pth   # e.g. coco3d/elpcac/quality_0/eb_las.pth

SPCAC-datasets/
├── {dataset}.hdf5                            # HDF5 preprocessed data
└── {dataset}/test/*.ply                      # PLY test split
```

Both repositories are **public**. Configure repo IDs (a token is only needed for gated repos):

```bash
export SPCAC_HF_CHECKPOINT_REPO="kaiwangm/SPCAC-checkpoints"
export SPCAC_HF_DATASET_REPO="kaiwangm/SPCAC-datasets"
export HF_TOKEN="hf_..."   # optional
```

When local files are missing, evaluation downloads checkpoints automatically, and dataset loading falls back to the dataset repo.

## Installation

This repo was set up on an **NVIDIA GeForce RTX 5080** (Blackwell, compute capability 12.0 / `sm_120`). The lowest CUDA Toolkit that natively supports this GPU is **12.8**, so this machine uses **CUDA 13.0**. Installation was completed in the environment below; other distros, kernels, GCC, or CUDA versions have **not** been tested.

| Item | Version |
|------|---------|
| GPU | NVIDIA GeForce RTX 5080 |
| OS | Ubuntu 26.04 LTS (WSL2) |
| Kernel | `6.18.33.2-microsoft-standard-WSL2` |
| GCC | 15.2.0 |
| CUDA | 13.0 |

Official **MinkowskiEngine** and **pytorch3d** currently fail to install on CUDA 12.8 and 13 ([MinkowskiEngine #614](https://github.com/NVIDIA/MinkowskiEngine/issues/614), [#620](https://github.com/NVIDIA/MinkowskiEngine/issues/620), [#621](https://github.com/NVIDIA/MinkowskiEngine/issues/621), [#632](https://github.com/NVIDIA/MinkowskiEngine/issues/632); [pytorch3d #1962](https://github.com/facebookresearch/pytorch3d/issues/1962), [#1970](https://github.com/facebookresearch/pytorch3d/issues/1970), [#2011](https://github.com/facebookresearch/pytorch3d/issues/2011), [#2016](https://github.com/facebookresearch/pytorch3d/issues/2016)). This repo therefore uses an unofficial fork and a local source tree, declared in [`pyproject.toml`](pyproject.toml) under `[tool.uv.sources]`. `uv sync` will **not** install the official PyPI packages:

```toml
# MinkowskiEngine — built from source on the CUDA 13 fork (build isolation + extra-build-dependencies, no patches needed)
minkowskiengine = { git = "https://github.com/AzharSindhi/MinkowskiEngineCuda13", branch = "cuda13-installation" }

# pytorch3d — built from the local clone (pulsar module excluded)
# pulsar renderer fails to link under CUDA 13.0; the project only needs pytorch3d.ops (knn)
pytorch3d = { path = ".local/pytorch3d" }
```

- **MinkowskiEngine**: CUDA 13 fork [`AzharSindhi/MinkowskiEngineCuda13`](https://github.com/AzharSindhi/MinkowskiEngineCuda13) (`cuda13-installation`).
- **pytorch3d**: local clone at `.local/pytorch3d`, patched to compile only the `knn` ops this project needs (`pytorch3d.ops.knn_points` / `knn_gather`). Rendering, rasterization, and pulsar are skipped.

Install these two libraries for **your** CUDA / compiler stack; do not copy this repo’s build verbatim. Prepare the local pytorch3d tree (or change the `pytorch3d` source) before `uv sync`. Patches and flags are recorded in [`script/setup_env.sh`](script/setup_env.sh) for reference — **do not run it as a one-shot installer**.

```bash
git clone https://github.com/kaiwangm/SPCAC.git
cd SPCAC
uv sync
```

> **Evaluation-only tools**: `utils/bin/` (MPEG G-PCC `tmc13` and `pc_error_d`) is not bundled. For `mpeg` evaluation mode, build these from MPEG sources and place them under `utils/bin/`.

## Model Zoo

Each entry below has a YAML profile in `configs/model/`. All models use MinkowskiEngine sparse convolutions (stride-2 downsample / transpose upsample). Latent widths (`N` / `M` / `HyM`) are set per quality level in each YAML.

| Family | Config | Model class | Encoder / Decoder | Entropy coding |
|--------|--------|-------------|-------------------|----------------|
| **Basic Priors** | `baseline_factorized` | `factorized_prior` | Plain sparse conv stacks | `EntropyBottleneck` on latent `y` |
| | `baseline_mean` | `mean_scale_hyperprior` | Plain sparse conv stacks | Hyperprior `z` → Gaussian scales/means for `y` |
| **Grouping** | `grouping` | `grouping` | Plain sparse conv stacks | Local hyperprior + 3-pass autoregressive group context |
| **Advanced** | `elpcac_l` | `elpcac_l` | Sparse conv stacks without residual blocks or local self-attention | Same as `elpcac` |
| | `elpcac` | `elpcac` | Residual blocks + local self-attention | Geometry context + global/local hyperpriors; 3-pass cross-attention fusion |

## Training

```bash
# Single quality
uv run python -m train.train --model elpcac --quality 0

# Override dataset (default: coco3d from configs/trainer/default.yaml)
uv run python -m train.train --model grouping --quality 3 --dataset coco3d
```

Checkpoints are saved to:

```
checkpoints/{dataset}/{model_class}/quality_{q}/eb_las.pth
```

Supported `--model` values: `baseline_factorized`, `baseline_mean`, `grouping`, `elpcac_l`, `elpcac`.

## Evaluation

```bash
# Single run
uv run python -m evaluation.evaluation \
  --model elpcac --quality 3 \
  --train_dataset coco3d --dataset j8ivfbv2-longdress10 --verbose

# Batch: all models × 8iVFBv2 sequences (or pass model / dataset / quality)
bash script/evaluation_all.sh
bash script/evaluation_all.sh elpcac
bash script/evaluation_all.sh elpcac j8ivfbv2-longdress10 3
```

Logs are written under `result/{dataset}/`. Missing checkpoints are fetched from Hugging Face when `SPCAC_HF_CHECKPOINT_REPO` is set.

## Rate-Distortion Curves

Collection and plotting are separate scripts and can be run independently. Results are stored as per-dataset JSON; existing `(model, quality)` points are skipped (incremental).

```bash
# Collect: default 5 models × 8iVFBv2 sequences × q0–q5
bash script/rd_collect.sh
bash script/rd_collect.sh elpcac                          # one model
bash script/rd_collect.sh elpcac j8ivfbv2-longdress10     # model + dataset
bash script/rd_collect.sh elpcac j8ivfbv2-longdress10 3   # + quality
FORCE=1 bash script/rd_collect.sh elpcac                 # overwrite existing points

# Plot: read JSON, one PSNR-YUV / PSNR-Y comparison figure per dataset
bash script/rd_plot.sh
bash script/rd_plot.sh j8ivfbv2-longdress10
```

Python equivalents:

```bash
uv run python -m evaluation.rd_collect --model elpcac --dataset j8ivfbv2-longdress10
uv run python -m evaluation.rd_plot --dataset j8ivfbv2-longdress10
```

Outputs:

```
result/{dataset}/metrics.json
result/{dataset}/psnr_yuv.png
result/{dataset}/*.npz          # optional 3D previews (gitignored)
```

## Datasets

**coco3d** is our in-house synthetic set and is used for training only: 2D images from COCO are randomly pasted onto generated 3D point clouds to synthesize color attributes. **8iVFBv2** (`datasets/j8ivfbv2`) and **owlii** (`datasets/owlii`) are test-only and are not used for training. **sensat_urban** and **scannet** are used for both training and testing.

| Dataset | Split | Category | Description |
|---------|-------|----------|-------------|
| coco3d | train | dense | In-house synthetic: COCO 2D images randomly pasted onto generated 3D point clouds (default training set) |
| 8iVFBv2 | test | dense | Voxelized full bodies (longdress, loot, redandblack, soldier); not used for training |
| owlii | test | dense | Dynamic sequences (basketball_player, dancer, exercise, model); not used for training |
| sensat_urban | train / test | dense | Large-scale urban scenes |
| scannet | train / test | dense | Indoor RGB-D scans |

Place data under `datasets/{dataset}/` (PLY). Dataset YAMLs live in `configs/dataset/`.

## Configuration

```
configs/
├── trainer/default.yaml   # Optimizer, LR, epochs, wandb, default dataset
├── model/{name}.yaml      # Model class, channels, per-quality λ / widths
└── dataset/{name}.yaml    # Dataset path, depth, attributes
```

Example model config (`configs/model/baseline_factorized.yaml`):

```yaml
model: factorized_prior
category: dense
channels: 3
num_layers: 1
levels:
  q0: { paraments: [64, 16],   lambda: 0.0006 }
  q1: { paraments: [128, 32],  lambda: 0.0018 }
  q2: { paraments: [128, 48],  lambda: 0.0063 }
  q3: { paraments: [128, 64],  lambda: 0.0130 }
  q4: { paraments: [192, 64],  lambda: 0.0530 }
  q5: { paraments: [192, 64],  lambda: 0.1800 }
  q6: { paraments: [192, 80],  lambda: 0.5800 }
```

Loss: $\mathcal{L} = R + \lambda \cdot D$ (larger λ → higher quality, higher bitrate).

## Result viewer

Local Gradio app (English / 中文). It lists **8iVFBv2** and **owlii** test sequences only; coco3d is not shown. Each pane can show the original cloud or any model / quality. **Click a point on the RD curve** to load that reconstruction on the right (bpp labeled). Switch models at the same quality from the dropdown, quality radio, or thumbnails.

```bash
# 1. Export a full-resolution preview of the first test frame (GPU; original-only: SKIP_INFER=1)
bash script/export_previews.sh
bash script/export_previews.sh j8ivfbv2-longdress10
SKIP_INFER=1 bash script/export_previews.sh j8ivfbv2-longdress10

# 2. Launch http://127.0.0.1:7860
bash script/launch_demo.sh
```

Without previews the RD plot and bpp labels still work; the 3D pane tells you to run `export_previews`. Metrics and optional `.npz` previews are stored under `result/`. Push `preview/` (and keep `result/` available, or set `SPCAC_RESULT_DIR` / `SPCAC_PREVIEW_DIR`) to a Hugging Face Space for a public page.
