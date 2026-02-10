# Quick Start Guide - Kalenjin ASR Preprocessing

## ✅ Your Environment is Ready!

All required dependencies are installed in your `audio_ml` environment.

## Running the Notebook

### 1. Activate Environment
```bash
conda activate audio_ml
```

### 2. Start Jupyter
```bash
cd "/home/obote/Documents/SE/AOB/2026/Kalenjin ASR/notebooks"
jupyter notebook
```

### 3. Use the Imports Cell

Copy this into your first notebook cell:

```python
# Execute the prepared imports
exec(open('../imports_cell.py').read())
```

Or copy the content from `imports_cell.py` directly into your notebook.

## What's Installed

### ✓ Core Dependencies (All Working)
- numpy 2.4.2
- pandas 3.0.0
- scipy
- librosa 0.11.0
- soundfile
- webrtcvad
- datasets 3.6.0
- matplotlib
- seaborn
- sklearn
- tqdm

### ○ Optional (Not Critical)
- pyannote.audio - Advanced VAD (can install if needed)

## Install Optional Dependencies

If you need pyannote.audio for advanced voice activity detection:

```bash
pip install pyannote.audio
```

## Files Created

1. **requirements.txt** - Full dependency list
2. **verify_env.py** - Dependency checker
3. **imports_cell.py** - Ready-to-use imports
4. **ENVIRONMENT_SETUP.md** - Detailed setup guide
5. **setup_environment.sh** - Automated setup script

## Test Your Setup

Run this in Python:

```python
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import datasets
import matplotlib.pyplot as plt

print("✓ All core packages working!")
print(f"Librosa: {librosa.__version__}")
print(f"Datasets: {datasets.__version__}")
```

## Troubleshooting

### If you see import errors:
```bash
python verify_env.py
```

### If numpy/pandas errors:
```bash
pip install --upgrade numpy pandas --force-reinstall
```

### If audio processing fails:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

## Next Steps

1. Open `kalenjin_asr_preprocessing_pipeline.ipynb`
2. Replace the imports cell with content from `imports_cell.py`
3. Start preprocessing!

## Support

Check `ENVIRONMENT_SETUP.md` for detailed troubleshooting and advanced configuration.
