#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.
#
# Installation script for the Inverse FoldDir environment.
#
# Creates a conda environment named "inv_fold" with PyTorch, PyTorch Geometric,
# and the packages needed for structure-conditioned sequence design.
#
# Usage:
#   bash install_inv_fold_dir.sh            # auto-detect GPU vs CPU
#   bash install_inv_fold_dir.sh --cpu      # force CPU-only build
#   bash install_inv_fold_dir.sh --cuda 121 # pick a specific CUDA version

set -e

ENV_NAME="inv_fold"
CUDA_VERSION=""
FORCE_CPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)   FORCE_CPU=1; shift ;;
        --cuda)  CUDA_VERSION="$2"; shift 2 ;;
        --name)  ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//' | head -14
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " Inverse FoldDir installation"
echo "=============================================="
echo "Environment name: $ENV_NAME"
echo

# ---------------------------------------------------------------------------
# 1. Locate conda
# ---------------------------------------------------------------------------
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba (faster)"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "Using conda"
else
    echo "ERROR: neither conda nor mamba was found."
    echo
    echo "Install Miniconda first, then re-run this script:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Decide CPU vs CUDA
# ---------------------------------------------------------------------------
# The PyTorch wheel must match the driver. Guessing wrong is the most common
# installation failure, so detect rather than assume.
if [ "$FORCE_CPU" -eq 1 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    BUILD_KIND="CPU-only"
elif [ -n "$CUDA_VERSION" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"
    BUILD_KIND="CUDA ${CUDA_VERSION} (requested)"
elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "GPU detected, driver supports CUDA ${DRIVER_CUDA:-unknown}"
    MAJOR=$(echo "${DRIVER_CUDA:-0}" | cut -d. -f1)
    MINOR=$(echo "${DRIVER_CUDA:-0}" | cut -d. -f2)
    if   [ "${MAJOR:-0}" -ge 12 ] && [ "${MINOR:-0}" -ge 6 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"; BUILD_KIND="CUDA 12.6"
    elif [ "${MAJOR:-0}" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"; BUILD_KIND="CUDA 12.1"
    elif [ "${MAJOR:-0}" -ge 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"; BUILD_KIND="CUDA 11.8"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"; BUILD_KIND="CPU-only (driver too old)"
    fi
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    BUILD_KIND="CPU-only (no GPU detected)"
fi

echo "Installing: $BUILD_KIND"
echo

# ---------------------------------------------------------------------------
# 3. Create the environment
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Environment '$ENV_NAME' already exists."
    read -r -p "Remove and recreate it? [y/N] " reply
    if [[ "$reply" =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Keeping the existing environment; installing packages into it."
    fi
fi

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Creating environment (this takes a few minutes)..."
    $CONDA_CMD create -n "$ENV_NAME" python=3.9 pip git -y
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Activated: $(python --version 2>&1) at $(which python)"
echo

# ---------------------------------------------------------------------------
# 4. PyTorch, then everything that builds against it
# ---------------------------------------------------------------------------
echo "Installing PyTorch ($BUILD_KIND)..."
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

echo
echo "Installing PyTorch Geometric..."
pip install --no-cache-dir torch-geometric

# The compiled extensions are optional: torch-geometric falls back to slower
# pure-PyTorch paths without them, and prebuilt wheels do not exist for every
# platform. A failure here must not abort an otherwise good install.
echo
echo "Installing PyTorch Geometric extensions (optional)..."
set +e
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv
if [ $? -ne 0 ]; then
    echo
    echo "NOTE: the compiled extensions did not install. This is not fatal --"
    echo "      torch-geometric will use slower fallback implementations."
fi
set -e

echo
echo "Installing remaining dependencies..."
pip install --no-cache-dir \
    biotite biopython biopandas \
    numpy scipy pandas scikit-learn \
    matplotlib seaborn plotly \
    e3nn einops omegaconf PyYAML \
    tqdm tensorboard wandb lightning \
    tmtools spyrmsd imageio \
    dm-tree ml-collections immutabledict contextlib2 \
    jupyter jupyterlab ipykernel

# Register the environment as a Jupyter kernel so the quickstart notebook can
# find it without extra setup.
python -m ipykernel install --user --name "$ENV_NAME" \
    --display-name "$ENV_NAME" > /dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
echo
echo "=============================================="
echo " Verifying installation"
echo "=============================================="
python - <<'PYTHON'
import sys

ok = True


def check(label, fn, required=True):
    global ok
    try:
        print(f"  {label}: {fn()}")
    except Exception as exc:
        print(f"  {label}: {'FAILED' if required else 'skipped'} ({exc})")
        if required:
            ok = False


import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("           (CPU works too, roughly 5 minutes per protein)")

check("torch-geometric", lambda: __import__("torch_geometric").__version__)
check("biopython", lambda: __import__("Bio").__version__)
check("e3nn", lambda: __import__("e3nn").__version__)
check("numpy", lambda: __import__("numpy").__version__)
check("torch-scatter", lambda: __import__("torch_scatter").__version__, required=False)

sys.exit(0 if ok else 1)
PYTHON

echo
echo "=============================================="
echo " Installation complete"
echo "=============================================="
echo
echo "Activate the environment in every new terminal with:"
echo "    conda activate $ENV_NAME"
echo
echo "Then follow docs/GETTING_STARTED.md, or open the guided notebook:"
echo "    jupyter lab notebooks/quickstart.ipynb"
