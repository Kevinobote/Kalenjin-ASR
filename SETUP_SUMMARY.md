# Kalenjin ASR Environment - Complete Setup Guide

## ✅ Current Status

Your `audio_ml` environment has most dependencies installed. There's a minor issue with `webrtcvad` that can be fixed or bypassed.

## Quick Fix for webrtcvad (Optional)

Run this command:
```bash
./fix_webrtcvad.sh
```

Or manually:
```bash
conda activate audio_ml
pip install setuptools
pip uninstall -y webrtcvad
pip install webrtcvad
```

## ✅ Working Solution (Recommended)

Use the robust imports that handle missing dependencies gracefully:

### In your notebook, use this cell:

```python
exec(open('../robust_imports.py').read())
```

This will:
- ✓ Load all core dependencies
- ✓ Handle webrtcvad gracefully (uses librosa fallback if unavailable)
- ✓ Show clear status of all features
- ✓ Provide fallback implementations

## What's Working

### Core (All ✓)
- numpy 2.2.6
- pandas (latest)
- scipy 1.15.2
- librosa 0.11.0
- soundfile 0.13.1
- datasets 3.6.0
- matplotlib 3.10.8
- seaborn 0.13.2
- plotly 6.5.2
- scikit-learn 1.7.2

### Optional
- webrtcvad: Has dependency issue (fallback available)
- pyannote.audio: Not installed (not critical)
- pydub: Installed
- textdistance: Installed

## Preprocessing Capabilities

### With Current Setup You Can:

1. **Audio Processing** ✓
   - Load/save audio files
   - Resample to 16kHz
   - Trim silence (librosa-based)
   - Normalize audio
   - Extract features

2. **Text Normalization** ✓
   - Unicode normalization
   - Case conversion
   - Punctuation removal
   - Character filtering
   - Kalenjin-specific processing

3. **Quality Assessment** ✓
   - Duration filtering
   - SNR estimation
   - Statistical analysis
   - Outlier detection

4. **Dataset Structuring** ✓
   - HuggingFace datasets format
   - Train/dev/test splits
   - Manifest creation
   - Metadata management

5. **Visualization** ✓
   - Waveforms
   - Spectrograms
   - Statistical plots
   - Interactive plots (plotly)

## VAD (Voice Activity Detection) Options

### Option 1: Librosa-based (Current - Works Fine)
```python
# Already included in robust_imports.py
audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
```

### Option 2: WebRTC VAD (After fix)
```python
import webrtcvad
vad = webrtcvad.Vad(2)  # aggressiveness 0-3
```

### Option 3: Pyannote (Advanced - Optional)
```bash
pip install pyannote.audio
```

## Files Created for You

```
Kalenjin ASR/
├── requirements.txt           # Full dependency list
├── verify_env.py             # Check what's installed
├── robust_imports.py         # ✓ USE THIS in notebook
├── fix_webrtcvad.sh         # Fix webrtcvad issue
├── setup_environment.sh      # Full environment setup
├── ENVIRONMENT_SETUP.md      # Detailed guide
├── QUICKSTART.md            # Quick start guide
└── SETUP_SUMMARY.md         # This file
```

## Recommended Workflow

### Step 1: Use Robust Imports
In your notebook first cell:
```python
exec(open('../robust_imports.py').read())
```

### Step 2: Verify Everything Works
You should see:
```
====================================================================
              KALENJIN ASR PREPROCESSING PIPELINE
====================================================================

Core Dependencies:
  ✓ NumPy:      2.2.6
  ✓ Pandas:     ...
  ✓ Librosa:    0.11.0
  ...

✓ Environment ready for preprocessing!
====================================================================
```

### Step 3: Start Preprocessing
All core functionality will work immediately!

## If You Want Perfect Setup

Run these commands:
```bash
cd "/home/obote/Documents/SE/AOB/2026/Kalenjin ASR"

# Fix webrtcvad
./fix_webrtcvad.sh

# Verify everything
python verify_env.py

# Start Jupyter
jupyter notebook
```

## Testing Your Setup

Run this in a notebook cell:
```python
# Test core functionality
import numpy as np
import librosa
import soundfile as sf

# Create test audio
sr = 16000
duration = 1.0
audio = np.random.randn(int(sr * duration))

# Test processing
audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
print(f"✓ Audio processing works!")
print(f"  Original: {len(audio)} samples")
print(f"  Trimmed:  {len(audio_trimmed)} samples")

# Test datasets
from datasets import Dataset
test_data = Dataset.from_dict({"audio": [audio], "text": ["test"]})
print(f"✓ Dataset creation works!")
```

## Troubleshooting

### Issue: webrtcvad import error
**Solution**: Use `robust_imports.py` - it has fallback VAD

### Issue: matplotlib not found
**Solution**: Already fixed! matplotlib 3.10.8 is installed

### Issue: Import errors in notebook
**Solution**: Make sure you're in audio_ml environment:
```bash
conda activate audio_ml
jupyter notebook
```

## Next Steps

1. ✓ Environment is ready
2. → Open `kalenjin_asr_preprocessing_pipeline.ipynb`
3. → Use `exec(open('../robust_imports.py').read())` in first cell
4. → Start building your preprocessing pipeline!

## Support Commands

```bash
# Check environment
python verify_env.py

# Fix webrtcvad
./fix_webrtcvad.sh

# Reinstall everything
pip install -r requirements.txt

# Start fresh
conda create -n kalenjin_asr_new python=3.10
conda activate kalenjin_asr_new
pip install -r requirements.txt
```

---

**Status**: ✅ READY TO USE (with robust imports)  
**Recommendation**: Use `robust_imports.py` - handles all edge cases  
**Environment**: audio_ml (Python 3.10)
