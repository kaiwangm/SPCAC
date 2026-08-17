#!/usr/bin/env bash
# ============================================================================
# SPCAC System Environment Setup Script
# Purpose: Install system-level dependencies only (CUDA Toolkit, compilers,
#          system libraries, uv). Python packages are managed by uv via
#          pyproject.toml — run `uv sync` to install them.
#
# Target OS: Ubuntu 26.04 LTS (WSL2)
# CUDA:      13.0 (compatible with MinkowskiEngineCuda13 fork)
# Python:    3.10 (numpy 1.x, compatible with MinkowskiEngine)
# Date:      2026-07-30
# ============================================================================
set -euo pipefail

# ---------- Color output ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------- Configuration ----------
CUDA_VER="13-0"
CUDA_HOME="/usr/local/cuda-13.0"
CUDA_REPO="wsl-ubuntu"  # NVIDIA apt repository (WSL2 only)

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================================================================
# [1/4] Basic build tools & system libraries (apt)
# ================================================================
info "========== [1/4] Installing basic build tools & system libraries =========="

if ! command -v apt-get &>/dev/null; then
    error "This script only supports the apt package manager (Debian/Ubuntu)."
fi

info "Installing build-essential, git, openblas, opengl, and other system libraries..."
sudo apt-get install -y \
    build-essential \
    git wget curl \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libgl1 libgl1-mesa-dri \
    libx11-6 \
    libglib2.0-0

info "Basic system libraries installed."

# ================================================================
# [2/4] CUDA Toolkit 13.0 (nvcc + development headers for CUDA extension compilation)
# ================================================================
info "========== [2/4] Installing CUDA Toolkit =========="

NEED_CUDA=true
if [[ -d "$CUDA_HOME" ]] && [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
    INSTALLED_VER=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "")
    if [[ "$INSTALLED_VER" == 13.0* ]]; then
        info "CUDA Toolkit $INSTALLED_VER is already installed, skipping."
        NEED_CUDA=false
    fi
fi

if [[ "$NEED_CUDA" == true ]]; then
    info "Installing CUDA Toolkit $CUDA_VER (build components only, skipping nsight/documentation)..."

    # keyring may already be installed (from CUDA 12.6 setup), ensure it exists
    if ! dpkg -l cuda-keyring &>/dev/null; then
        KEYRING_DEB="/tmp/cuda-keyring.deb"
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
        wget -q "$KEYRING_URL" -O "$KEYRING_DEB" || error "Failed to download CUDA keyring"
        sudo dpkg -i "$KEYRING_DEB"
        rm -f "$KEYRING_DEB"
    fi
    sudo apt-get update -qq

    # Install minimal CUDA components (skip nsight to avoid libtinfo5 dependency, not available on Ubuntu 26.04)
    sudo apt-get install -y \
        "cuda-nvcc-${CUDA_VER}" \
        "cuda-cudart-${CUDA_VER}" \
        "cuda-cudart-dev-${CUDA_VER}" \
        "cuda-libraries-dev-${CUDA_VER}" \
        "cuda-nvtx-${CUDA_VER}" \
        "cuda-cccl-${CUDA_VER}"

    info "CUDA Toolkit installed."
fi

# Verify nvcc
export PATH="${CUDA_HOME}/bin:${PATH}"
info "nvcc: $(nvcc --version 2>/dev/null | grep 'release' || echo 'unknown')"

# Patch CUDA math_functions.h: add noexcept(true) to rsqrt/rsqrtf
# Resolves noexcept declaration conflict between Ubuntu 26.04 glibc 2.41 and CUDA
# (system-level fix, not a source patch)
# sinpi/cospi are handled automatically by CUDA 13's __NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER macro
MATH_H="${CUDA_HOME}/include/crt/math_functions.h"
if [[ -f "$MATH_H" ]] && ! grep -q 'rsqrt(double x) noexcept' "$MATH_H"; then
    info "Patching CUDA math_functions.h (adding noexcept(true) to rsqrt/rsqrtf)..."
    sudo sed -i 's/rsqrt(double x);/rsqrt(double x) noexcept(true);/' "$MATH_H"
    sudo sed -i 's/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/' "$MATH_H"
    info "CUDA math_functions.h patched."
fi

# CUDA 13.0 places thrust/cub/cuda headers under the cccl/ subdirectory;
# create symlinks so #include <thrust/...>, #include <cub/...>, #include <cuda/...> resolve correctly
CCCL_DIR="${CUDA_HOME}/include/cccl"
if [[ -d "$CCCL_DIR" ]]; then
    for lib in thrust cub cuda; do
        if [[ ! -e "${CUDA_HOME}/include/$lib" ]]; then
            info "Creating symlink: include/$lib -> cccl/$lib"
            sudo ln -s "cccl/$lib" "${CUDA_HOME}/include/$lib"
        fi
    done
    info "CUDA cccl symlinks created."
fi


# ================================================================
# [3/4] Install uv
# ================================================================
info "========== [3/4] Installing uv =========="

if command -v uv &>/dev/null; then
    info "uv is already installed: $(uv --version)"
else
    info "Installing uv via the Astral official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    info "uv installed: $(uv --version)"
fi

# ================================================================
# [4/4] Set build environment variables + sync Python dependencies
# ================================================================
info "========== [4/4] Setting build environment and syncing Python dependencies =========="

# Detect openblas header path (installed by apt, usually under /usr/include/x86_64-linux-gnu)
BLAS_INCLUDE_DIRS=""
for p in "/usr/include/x86_64-linux-gnu" "/usr/include"; do
    if [[ -f "$p/cblas.h" ]] || [[ -f "$p/openblas/cblas.h" ]]; then
        BLAS_INCLUDE_DIRS="$p"
        break
    fi
done
if [[ -z "$BLAS_INCLUDE_DIRS" ]]; then
    BLAS_INCLUDE_DIRS="/usr/include/x86_64-linux-gnu"
    warn "cblas.h not found, using default path: $BLAS_INCLUDE_DIRS"
fi

# Export environment variables required for compilation (read by MinkowskiEngine / pytorch3d builds)
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="Ampere;Turing;Ada;Hopper"
# CUB_HOME points to CUDA include directory (PyTorch cpp_extension needs to find cub headers)
export CUB_HOME="${CUDA_HOME}/include"
export BLAS=openblas
export BLAS_INCLUDE_DIRS
# CUDA 13.0 supports GCC 15 (Ubuntu 26.04 default)
# Parallel build threads
export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24

info "Build environment variables:"
info "  CUDA_HOME=$CUDA_HOME"
info "  BLAS=$BLAS"
info "  BLAS_INCLUDE_DIRS=$BLAS_INCLUDE_DIRS"
info "  FORCE_CUDA=$FORCE_CUDA"
info "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
info "  CUB_HOME=$CUB_HOME"
info "  GCC=$(gcc -dumpversion)"
info "  MAX_JOBS=$MAX_JOBS"

# ---------- Run uv sync ----------
info "Running uv sync (installing all Python dependencies per pyproject.toml) ..."
info "Note: MinkowskiEngine and pytorch3d need to be built from source, this may take 10–30 minutes."

# Limit uv concurrent installs to 1 to avoid OOM when building minkowskiengine and pytorch3d simultaneously
export UV_CONCURRENT_INSTALLS=1

# Let setuptools use Python stdlib distutils (which includes msvccompiler),
# instead of setuptools' vendored distutils (which has removed msvccompiler).
# MinkowskiEngine fork's numpy.distutils depends on this module.
export SETUPTOOLS_USE_DISTUTILS=stdlib

cd "$PROJECT_ROOT"
rm -f uv.lock  # Python version and CUDA version changed, remove old lock file for re-resolution

# ---------- Prepare pytorch3d local source (trimmed to knn module only) ----------
# The project only needs pytorch3d.ops (knn_points / knn_gather), not rendering/rasterization modules.
# Download source then patch: setup.py builds only knn + ext.cpp, ext.cpp registers only knn functions.
P3D_LOCAL_DIR="$PROJECT_ROOT/.local/pytorch3d"
P3D_SETUP="$P3D_LOCAL_DIR/setup.py"
P3D_EXT="$P3D_LOCAL_DIR/pytorch3d/csrc/ext.cpp"

# Detect old patch (pulsar exclusion only), clean and re-download to apply new patch if found
if [[ -f "$P3D_SETUP" ]] && grep -q "exclude.*pulsar\|排除 pulsar" "$P3D_SETUP" 2>/dev/null; then
    info "Old patch detected, cleaning and re-downloading to apply new patch..."
    rm -rf "$P3D_LOCAL_DIR"
fi

if [[ ! -f "$P3D_SETUP" ]]; then
    info "Downloading pytorch3d (stable) source tarball to $P3D_LOCAL_DIR ..."
    mkdir -p "$PROJECT_ROOT/.local"
    # Use codeload tarball instead of git clone: single HTTP request, faster through proxy
    P3D_TARBALL="/tmp/pytorch3d-stable.tar.gz"
    wget -q --show-progress -O "$P3D_TARBALL" \
        https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/refs/tags/stable \
        || error "Failed to download pytorch3d source"
    tar -xzf "$P3D_TARBALL" -C "$PROJECT_ROOT/.local"
    # Extracted directory is named pytorch3d-stable-<hash>, rename to pytorch3d
    mv "$PROJECT_ROOT/.local/pytorch3d-stable"* "$P3D_LOCAL_DIR" 2>/dev/null || true
    rm -f "$P3D_TARBALL"
    info "pytorch3d source downloaded."
else
    info "pytorch3d local source already exists, skipping download."
fi

# Patch 1: ext.cpp — replace with minimal version (only knn registration, remove #include and registration for the other 18 modules)
if ! grep -q "minimal ext" "$P3D_EXT" 2>/dev/null; then
    info "Patching pytorch3d ext.cpp: trimming to knn module only..."
    cat > "$P3D_EXT" <<'EXTCPP'
// Minimal ext.cpp: only keep the knn module (project only needs pytorch3d.ops.knn)
#include <torch/extension.h>
#include "knn/knn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
  m.def("knn_check_version", &KnnCheckVersion);
#endif
  m.def("knn_points_idx", &KNearestNeighborIdx);
  m.def("knn_points_backward", &KNearestNeighborBackward);
}
EXTCPP
    info "pytorch3d ext.cpp patched."
fi

# Patch 2: setup.py — whitelist approach, only compile knn + ext.cpp (exclude pulsar/rasterize/blending and 18 other modules)
if ! grep -q '"knn" in s or "ext.cpp" in s' "$P3D_SETUP" 2>/dev/null; then
    info "Patching pytorch3d setup.py: whitelisting only knn module..."
    python3 - "$P3D_SETUP" <<'PYPATCH'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()
old = '    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)\n    extension = CppExtension'
new = (
    '    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)\n'
    '    # Only keep knn + ext.cpp (project only needs pytorch3d.ops.knn, exclude the other 18 modules)\n'
    '    sources = [s for s in sources if "knn" in s or "ext.cpp" in s]\n'
    '    source_cuda = [s for s in source_cuda if "knn" in s]\n'
    '    extension = CppExtension'
)
if old not in content:
    print("ERROR: patch target line not found, setup.py may have been updated", file=sys.stderr)
    sys.exit(1)
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
PYPATCH
    info "pytorch3d setup.py patched."
fi

# Patch 3: renderer/points/__init__.py — wrap pulsar import in try/except
# pulsar's Python code references _C.MAX_UINT and other excluded constants, needs graceful degradation
P3D_POINTS_INIT="$P3D_LOCAL_DIR/pytorch3d/renderer/points/__init__.py"
if ! grep -q "pulsar C extension not available" "$P3D_POINTS_INIT" 2>/dev/null; then
    info "Patching pytorch3d renderer/points/__init__.py: graceful pulsar import degradation..."
    python3 - "$P3D_POINTS_INIT" <<'PYPATCH2'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()
old = (
    "# Pulsar not enabled on amd.\n"
    "if not torch.version.hip:\n"
    "    from .pulsar.unified import PulsarPointsRenderer"
)
new = (
    "# Pulsar not enabled on amd.\n"
    "if not torch.version.hip:\n"
    "    try:\n"
    "        from .pulsar.unified import PulsarPointsRenderer\n"
    "    except Exception:\n"
    "        pass  # pulsar C extension not available\n"
)
if old not in content:
    print("ERROR: patch target line not found in renderer/points/__init__.py", file=sys.stderr)
    sys.exit(1)
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
PYPATCH2
    info "pytorch3d renderer/points/__init__.py patched."
fi

# Patch 4: renderer/__init__.py — wrap pulsar import in try/except
# renderer/__init__.py also directly does `from .points import PulsarPointsRenderer`, needs graceful degradation
P3D_RENDERER_INIT="$P3D_LOCAL_DIR/pytorch3d/renderer/__init__.py"
if ! grep -q "pulsar C extension not available" "$P3D_RENDERER_INIT" 2>/dev/null; then
    info "Patching pytorch3d renderer/__init__.py: graceful pulsar import degradation..."
    python3 - "$P3D_RENDERER_INIT" <<'PYPATCH3'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()
old = (
    "# Pulsar is not enabled on amd.\n"
    "if not torch.version.hip:\n"
    "    from .points import PulsarPointsRenderer"
)
new = (
    "# Pulsar is not enabled on amd.\n"
    "if not torch.version.hip:\n"
    "    try:\n"
    "        from .points import PulsarPointsRenderer\n"
    "    except ImportError:\n"
    "        pass  # pulsar C extension not available"
)
if old not in content:
    print("ERROR: patch target line not found in renderer/__init__.py", file=sys.stderr)
    sys.exit(1)
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
PYPATCH3
    info "pytorch3d renderer/__init__.py patched."
fi

# Clean up old pytorch3d installed in .venv (ensure uv sync rebuilds with patched version)
rm -rf "$PROJECT_ROOT/.venv/lib/python3.10/site-packages/pytorch3d"* 2>/dev/null || true
rm -rf "$PROJECT_ROOT/.venv/lib/python3.10/site-packages/pytorch3d-"* 2>/dev/null || true
# Clean uv cache for old wheels (path source packages cache wheels, must force rebuild)
uv cache clean pytorch3d 2>/dev/null || true

uv sync -v

# ---------- Verify ----------
info "========== Environment verification =========="
uv run python - <<'PYEOF'
import sys
print(f"Python:  {sys.version.split()[0]}")

checks = {
    "torch":         "PyTorch",
    "torchvision":   "torchvision",
    "open3d":        "Open3D",
    "compressai":    "CompressAI",
    "kornia":        "Kornia",
    "wandb":         "WandB",
    "rich":          "Rich",
    "plyfile":       "plyfile",
    "h5py":          "h5py",
    "scipy":         "SciPy",
    "pandas":        "Pandas",
    "matplotlib":    "Matplotlib",
    "sklearn":       "scikit-learn",
    "yaml":          "PyYAML",
    "trimesh":       "trimesh",
    "PIL":           "Pillow",
    "openpyxl":      "openpyxl",
    "bson":          "bson",
    "bd_metric":     "bd_metric",
}

all_ok = True
for mod, name in checks.items():
    try:
        __import__(mod)
        ver = getattr(sys.modules[mod], "__version__", "OK")
        print(f"  ✓ {name:20s} {ver}")
    except ImportError as e:
        print(f"  ✗ {name:20s} not installed ({e})")
        all_ok = False

optional = {
    "MinkowskiEngine": "MinkowskiEngine",
    "pytorch3d":       "pytorch3d",
}
for mod, name in optional.items():
    try:
        __import__(mod)
        print(f"  ✓ {name:20s} (optional) OK")
    except ImportError:
        print(f"  △ {name:20s} (optional) not installed")
        all_ok = False

import torch
print(f"\nPyTorch CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if all_ok:
    print("\n🎉 All core dependencies installed successfully!")
else:
    print("\n⚠️  Some dependencies were not installed, see log above.")
PYEOF

info "========== Installation complete =========="
info ""
info "  Training:"
info "    uv run python -m train.train --model=xxx --dataset=xxx --quality=xxx"
info ""
info "  Common uv commands:"
info "    uv sync          # Install/sync dependencies"
info "    uv add xxx       # Add a new dependency"
info "    uv remove xxx    # Remove a dependency"
info "    uv lock          # Update lock file"
info "    uv pip list      # List installed packages"
