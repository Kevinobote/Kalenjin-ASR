# Kalenjin ASR Environment Setup Guide

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd "/home/obote/Documents/SE/AOB/2026/Kalenjin ASR"
chmod +x setup_environment.sh
./setup_environment.sh
```

### Option 2: Manual Setup

#### Step 1: Create Conda Environment
```bash
conda create -n kalenjin_asr python=3.10 -y
conda activate kalenjin_asr
```

#### Step 2: Install PyTorch
```bash
# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# For GPU (CUDA 11.8)
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

#### Step 3: Install Core Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Install System Dependencies (if needed)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 sox
```

**macOS:**
```bash
brew install ffmpeg libsndfile sox
```

#### Step 5: Verify Installation
```bash
python -c "import librosa, soundfile, datasets; print('✓ Success')"
```

## Troubleshooting

### Issue: webrtcvad installation fails
```bash
# Install build tools first
sudo apt-get install python3-dev build-essential
pip install webrtcvad
```

### Issue: pyannote.audio installation fails
```bash
# Install from source
pip install pyannote.audio --no-deps
pip install torch torchaudio asteroid-filterbanks pyannote.core pyannote.database pyannote.metrics
```

### Issue: matplotlib backend errors
```bash
# Set backend in your notebook
import matplotlib
matplotlib.use('Agg')
```

## Jupyter Notebook Setup

### Register kernel
```bash
conda activate kalenjin_asr
python -m ipykernel install --user --name=kalenjin_asr --display-name="Kalenjin ASR"
```

### Launch Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

## Minimal Installation (Core Only)

If you encounter issues with optional dependencies, install core only:

```bash
pip install numpy pandas scipy librosa soundfile datasets matplotlib seaborn tqdm scikit-learn
```

## Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export KALENJIN_ASR_ROOT="/home/obote/Documents/SE/AOB/2026/Kalenjin ASR"
export PYTHONPATH="${KALENJIN_ASR_ROOT}:${PYTHONPATH}"
```

## Verification Script

Run this in Python to verify all dependencies:

```python
import sys

dependencies = {
    'numpy': 'np',
    'pandas': 'pd',
    'scipy': 'scipy',
    'librosa': 'librosa',
    'soundfile': 'sf',
    'datasets': 'datasets',
    'matplotlib': 'plt',
    'seaborn': 'sns',
    'plotly': 'px',
    'sklearn': 'sklearn',
    'tqdm': 'tqdm',
}

print("Checking dependencies...")
failed = []

for package, alias in dependencies.items():
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - MISSING")
        failed.append(package)

if failed:
    print(f"\nMissing packages: {', '.join(failed)}")
    print(f"Install with: pip install {' '.join(failed)}")
else:
    print("\n✓ All dependencies installed successfully!")
```

## Current Environment Check

Your current environment appears to be: `audio_ml`

To use the existing environment:
```bash
conda activate audio_ml
pip install -r requirements.txt
```
