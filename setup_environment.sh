#!/bin/bash
# Kalenjin ASR Environment Setup Script
# This script creates a conda environment with all required dependencies

set -e

echo "=========================================="
echo "Kalenjin ASR Environment Setup"
echo "=========================================="

# Environment name
ENV_NAME="kalenjin_asr"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch (CPU version - change if GPU needed)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install core dependencies
echo "Installing core dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional audio dependencies
echo "Installing system audio dependencies..."
# Note: These may require system-level installation
# Ubuntu/Debian: sudo apt-get install ffmpeg libsndfile1
# macOS: brew install ffmpeg libsndfile

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import tqdm
print('✓ All core dependencies imported successfully')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ Pandas version: {pd.__version__}')
print(f'✓ Librosa version: {librosa.__version__}')
print(f'✓ Datasets version: {datasets.__version__}')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo "=========================================="
