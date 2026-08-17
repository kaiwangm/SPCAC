[English](README.md) | **中文**

# SPCAC: 基于稀疏张量的点云属性压缩

[![Hugging Face Models](https://img.shields.io/badge/Models-HuggingFace-gold.svg)](https://huggingface.co/kaiwangm/SPCAC-checkpoints)
[![Hugging Face Datasets](https://img.shields.io/badge/Datasets-HuggingFace-gold.svg)](https://huggingface.co/datasets/kaiwangm/SPCAC-datasets)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 点云属性压缩的基线模型仓库：基于稀疏卷积与熵建模，对 3D 点云颜色属性做学习式压缩。
>
> 数据集收集与模型训练完成于 2023 年 3 月至 12 月。
>
> 仓库于 2026 年 8 月重新整理并公开，检查点与数据集同步发布到 Hugging Face。

## 简介

本仓库是 **点云属性压缩** 的基线模型集合，在已知几何上对颜色属性（RGB）做率失真优化压缩（可配置 λ，质量等级 q0–q6）。基于 **MinkowskiEngine** 稀疏卷积与 **CompressAI** 熵模型，提供 5 个模型，覆盖训练 → CDF 更新 → 熵编码压缩/解压。

**因子分解先验。** 独立建模潜在各通道。

**超先验。** 用侧信息预测高斯均值与尺度，捕获空间依赖。

**分组编解码。** 按坐标奇偶将点分成三组并渐进编码，已解码组作为后续组的空间上下文。

**几何引导。** 由占用几何提取上下文，辅助属性编码。

**局部超先验。** 对潜在特征做空间侧信息编码。

**全局超先验。** 对潜在做全局聚合侧信息。

**交叉注意力。** 融合全局超先验与局部上下文，得到熵参数。

**残差块与自注意力。** 用于编码器 / 解码器。

## Hugging Face 资源

| 资源 | 链接 | 内容 |
|------|------|------|
| 模型检查点 | [🤗 SPCAC-checkpoints](https://huggingface.co/kaiwangm/SPCAC-checkpoints) | 最终检查点（`eb_las.pth`），覆盖 模型 × 数据集 × 质量 (q0–q6) |
| 数据集 | [🤗 SPCAC-datasets](https://huggingface.co/datasets/kaiwangm/SPCAC-datasets) | HDF5 数据集与 PLY 测试集 |

```
SPCAC-checkpoints/
└── {dataset}/{model}/quality_{q}/eb_las.pth   # 例如 coco3d/elpcac/quality_0/eb_las.pth

SPCAC-datasets/
├── {dataset}.hdf5                            # HDF5 预处理数据
└── {dataset}/test/*.ply                      # PLY 测试集
```

两个仓库均为**公开**仓库（仅访问受限仓库时需要令牌）：

```bash
export SPCAC_HF_CHECKPOINT_REPO="kaiwangm/SPCAC-checkpoints"
export SPCAC_HF_DATASET_REPO="kaiwangm/SPCAC-datasets"
export HF_TOKEN="hf_..."   # 可选
```

本地文件缺失时，评估会自动下载检查点，数据集加载会回退到数据集仓库。

## 安装

本仓库在 **NVIDIA GeForce RTX 5080**（Blackwell，compute capability 12.0 / `sm_120`）上完成安装。该卡原生支持的最低 CUDA Toolkit 为 **12.8**，因此本机选用 **CUDA 13.0**。已验证环境如下；**其他发行版 / 内核 / GCC / CUDA 组合未尝试过**。

| 项目 | 版本 |
|------|------|
| 显卡 | NVIDIA GeForce RTX 5080 |
| 发行版 | Ubuntu 26.04 LTS（WSL2） |
| 内核 | `6.18.33.2-microsoft-standard-WSL2` |
| GCC | 15.2.0 |
| CUDA | 13.0 |

官方 **MinkowskiEngine** 与 **pytorch3d** 目前在 CUDA 12.8 与 13 上均无法直接安装（[MinkowskiEngine #614](https://github.com/NVIDIA/MinkowskiEngine/issues/614)、[#620](https://github.com/NVIDIA/MinkowskiEngine/issues/620)、[#621](https://github.com/NVIDIA/MinkowskiEngine/issues/621)、[#632](https://github.com/NVIDIA/MinkowskiEngine/issues/632)；[pytorch3d #1962](https://github.com/facebookresearch/pytorch3d/issues/1962)、[#1970](https://github.com/facebookresearch/pytorch3d/issues/1970)、[#2011](https://github.com/facebookresearch/pytorch3d/issues/2011)、[#2016](https://github.com/facebookresearch/pytorch3d/issues/2016)）。本仓库因此改为非官方仓库与本地源码构建，对应 [`pyproject.toml`](pyproject.toml) 的 `[tool.uv.sources]`。`uv sync` **不会**安装官方 PyPI 包：

```toml
# MinkowskiEngine — built from source on the CUDA 13 fork (build isolation + extra-build-dependencies, no patches needed)
minkowskiengine = { git = "https://github.com/AzharSindhi/MinkowskiEngineCuda13", branch = "cuda13-installation" }

# pytorch3d — built from the local clone (pulsar module excluded)
# pulsar renderer fails to link under CUDA 13.0; the project only needs pytorch3d.ops (knn)
pytorch3d = { path = ".local/pytorch3d" }
```

- **MinkowskiEngine**：CUDA 13 适配分支 [`AzharSindhi/MinkowskiEngineCuda13`](https://github.com/AzharSindhi/MinkowskiEngineCuda13)（`cuda13-installation`）。
- **pytorch3d**：本地目录 `.local/pytorch3d`，打补丁后只编译本项目所需的 `knn`（`pytorch3d.ops.knn_points` / `knn_gather`），不安装渲染、光栅化与 pulsar。

请按**自己的** CUDA / 编译器环境安装这两个库，不要照搬本仓库的构建。`uv sync` 前需先准备好本地 pytorch3d 源码（或自行修改 source）。补丁与编译参数见 [`script/setup_env.sh`](script/setup_env.sh)，仅供参考；**不推荐直接执行该脚本作为一键安装**。

```bash
git clone https://github.com/kaiwangm/SPCAC.git
cd SPCAC
uv sync
```

> **仅评估用工具**：`utils/bin/`（MPEG G-PCC `tmc13` 与 `pc_error_d`）不随仓库发布。如需 `mpeg` 评估模式，请自行从 MPEG 源码构建并放入 `utils/bin/`。

## 模型库

下列每个配置对应 `configs/model/` 下的一份 YAML。所有模型均基于 MinkowskiEngine 稀疏卷积（stride-2 下采样 / 转置上采样）。各质量等级的潜在通道数（`N` / `M` / `HyM`）在对应 YAML 中设定。

| 族系 | 配置 | 模型类 | 编码器 / 解码器 | 熵编码 |
|------|------|--------|-----------------|--------|
| **基础先验** | `baseline_factorized` | `factorized_prior` | 普通稀疏卷积堆叠 | 对潜在特征 `y` 使用 `EntropyBottleneck` |
| | `baseline_mean` | `mean_scale_hyperprior` | 普通稀疏卷积堆叠 | 超先验 `z` 预测高斯尺度/均值，再编码 `y` |
| **分组编码** | `grouping` | `grouping` | 普通稀疏卷积堆叠 | 局部超先验 + 3-pass 自回归分组上下文 |
| **高级** | `elpcac_l` | `elpcac_l` | 稀疏卷积堆叠，不含残差块与局部自注意力 | 与 `elpcac` 相同 |
| | `elpcac` | `elpcac` | 残差块 + 局部自注意力 | 几何上下文 + 全局/局部超先验；3-pass 交叉注意力融合 |

## 训练

```bash
# 单个质量等级
uv run python -m train.train --model elpcac --quality 0

# 覆盖数据集（默认取自 configs/trainer/default.yaml 的 coco3d）
uv run python -m train.train --model grouping --quality 3 --dataset coco3d
```

检查点保存路径：

```
checkpoints/{dataset}/{model_class}/quality_{q}/eb_las.pth
```

可用 `--model`：`baseline_factorized`、`baseline_mean`、`grouping`、`elpcac_l`、`elpcac`。

## 评估

```bash
# 单次评估
uv run python -m evaluation.evaluation \
  --model elpcac --quality 3 \
  --train_dataset coco3d --dataset j8ivfbv2-longdress10 --verbose

# 批量：全部模型 × 8iVFBv2 序列（也可指定 model / dataset / quality）
bash script/evaluation_all.sh
bash script/evaluation_all.sh elpcac
bash script/evaluation_all.sh elpcac j8ivfbv2-longdress10 3
```

日志写入 `result/{dataset}/`。若设置了 `SPCAC_HF_CHECKPOINT_REPO`，缺失的检查点会从 Hugging Face 自动下载。

## 率失真曲线

收集与绘图拆成两个脚本，可分开、分次运行。收集结果按数据集写入 JSON，已有 `(model, quality)` 点会跳过（增量更新）。

```bash
# 收集：默认 5 模型 × 8iVFBv2 序列 × q0–q5
bash script/rd_collect.sh
bash script/rd_collect.sh elpcac                          # 指定模型
bash script/rd_collect.sh elpcac j8ivfbv2-longdress10     # 模型 + 数据集
bash script/rd_collect.sh elpcac j8ivfbv2-longdress10 3   # 再限定质量
FORCE=1 bash script/rd_collect.sh elpcac                 # 覆盖已有点

# 绘图：读 JSON，每个数据集输出 PSNR-YUV / PSNR-Y 对比图
bash script/rd_plot.sh
bash script/rd_plot.sh j8ivfbv2-longdress10
```

等价 Python 入口：

```bash
uv run python -m evaluation.rd_collect --model elpcac --dataset j8ivfbv2-longdress10
uv run python -m evaluation.rd_plot --dataset j8ivfbv2-longdress10
```

输出路径：

```
result/{dataset}/metrics.json
result/{dataset}/psnr_yuv.png
result/{dataset}/*.npz          # 可选 3D 预览（不入库）
```

## 数据集

**coco3d** 由我们自行合成，仅用于训练：将 COCO 中的 2D 图像随机贴到生成的 3D 点云上，合成颜色属性。**8iVFBv2**（`datasets/j8ivfbv2`）与 **owlii**（`datasets/owlii`）仅用于测试，不参与训练。**sensat_urban** 与 **scannet** 同时用于训练和测试。

| 数据集 | 用途 | 类别 | 描述 |
|--------|------|------|------|
| coco3d | 训练 | dense | 自行合成：将 COCO 2D 图像随机贴到生成的 3D 点云上（默认训练集） |
| 8iVFBv2 | 测试 | dense | 体素化全身动态点云（longdress、loot、redandblack、soldier），不用于训练 |
| owlii | 测试 | dense | 动态序列（basketball_player、dancer、exercise、model），不用于训练 |
| sensat_urban | 训练 / 测试 | dense | 大规模城市场景 |
| scannet | 训练 / 测试 | dense | 室内 RGB-D 扫描 |

数据放在 `datasets/{dataset}/`（PLY）。数据集 YAML 位于 `configs/dataset/`。

## 配置

```
configs/
├── trainer/default.yaml   # 优化器、学习率、轮数、wandb、默认数据集
├── model/{name}.yaml      # 模型类、通道数、各质量 λ / 宽度
└── dataset/{name}.yaml    # 数据集路径、深度、属性
```

示例（`configs/model/baseline_factorized.yaml`）：

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

损失：$\mathcal{L} = R + \lambda \cdot D$（λ 越大，质量越高、码率越高）。

## 结果展示页

本地 Gradio 页面（中 / 英）。只列出 **8iVFBv2** 与 **owlii** 测试序列，不展示 coco3d。左右栏可各自显示原始点云或任意模型 / 质量。**点击 RD 曲线上的点** 会把该重建加载到右侧，并标注 bpp。同质量下可从下拉框、质量单选或缩略图切换模型。

```bash
# 1. 导出第一帧完整点云预览（需 GPU；只导出原图可 SKIP_INFER=1）
bash script/export_previews.sh
bash script/export_previews.sh j8ivfbv2-longdress10
SKIP_INFER=1 bash script/export_previews.sh j8ivfbv2-longdress10

# 2. 启动页面（http://127.0.0.1:7860）
bash script/launch_demo.sh
```

未导出预览时仍可看 RD 曲线与 bpp；点云区域会提示先跑 `export_previews`。指标与可选 `.npz` 预览都在 `result/`。将 `preview/` 推到 Hugging Face Space 时需同时提供 `result/`，或设置 `SPCAC_RESULT_DIR` / `SPCAC_PREVIEW_DIR`。
