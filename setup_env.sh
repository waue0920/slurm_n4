#!/bin/bash

# --- N4 Benchmark Toolkit: Unified Setup Script ---
# This script handles:
# 1. System package dependencies
# 2. Miniconda installation (if not present)
# 3. Dedicated Conda environment (n4_bench) creation
# 4. Python dependencies installation

set -e

CONDA_PATH="$HOME/miniconda3"
ENV_NAME="n4_bench"
PYTHON_VERSION="3.10"

echo "[1/4] Checking System Dependencies..."
sudo apt update
sudo apt install -y python3-dev libibverbs-dev wget curl git

echo "[2/4] Checking Miniconda..."
if ! command -v conda &> /dev/null; then
    if [ ! -d "$CONDA_PATH" ]; then
        echo "Miniconda not found. Downloading and installing..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p "$CONDA_PATH"
        rm miniconda.sh
    fi
    export PATH="$CONDA_PATH/bin:$PATH"
    # Initialize conda for the current shell session
    eval "$("$CONDA_PATH/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
else
    echo "Conda is already installed."
fi

echo "[3/4] Creating Conda Environment: $ENV_NAME..."
# Accept Terms of Service for non-interactive mode
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# Activate environment
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[4/4] Installing Python Dependencies..."
pip install --upgrade pip
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install other requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

echo ""
echo "===================================================="
echo " Setup Complete!"
echo " To activate the environment manually, use:"
echo "   conda activate $ENV_NAME"
echo "===================================================="
