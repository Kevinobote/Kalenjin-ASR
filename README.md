# Kalenjin ASR: Low-Resource Speech Recognition

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Mozilla Common Voice](https://img.shields.io/badge/Dataset-Common%20Voice%2024.0-orange.svg)](https://commonvoice.mozilla.org/)

A PhD-level automatic speech recognition (ASR) system for Kalenjin, a low-resource Nilotic language spoken in Kenya. This project implements state-of-the-art preprocessing pipelines and fine-tunes Wav2Vec2-XLS-R-300M for Kalenjin speech recognition.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This project addresses the challenge of building ASR systems for low-resource languages by:

1. **Advanced Preprocessing**: Comprehensive audio and text normalization pipeline
2. **Quality Assessment**: Multi-dimensional quality metrics (SNR, spectral features, VAD)
3. **Linguistic Normalization**: Kalenjin-specific text processing with CTC optimization
4. **Transfer Learning**: Fine-tuning Wav2Vec2-XLS-R-300M pre-trained model
5. **Reproducibility**: Modular, well-documented codebase with validation

### Key Statistics

- **Dataset**: Mozilla Common Voice v24.0 (Kalenjin)
- **Training Samples**: 11,057
- **Validation Samples**: 6,409
- **Test Samples**: 5,675
- **Vocabulary Size**: 32 tokens (26 letters + 6 special tokens)
- **Audio Quality**: Mean SNR ~46 dB
- **Total Duration**: ~15 hours

## ✨ Features

### Preprocessing Pipeline

- ✅ **Audio Processing**
  - Resampling to 16kHz
  - Voice Activity Detection (VAD)
  - Silence trimming
  - Normalization
  - Quality metrics (SNR, RMS, spectral features)

- ✅ **Text Normalization**
  - Lowercase conversion
  - Apostrophe normalization
  - Special character removal
  - CTC delimiter insertion (`|`)
  - Vocabulary optimization

- ✅ **Quality Control**
  - Duration filtering (0.5-20s)
  - SNR thresholding
  - Outlier detection
  - Statistical validation

### Model Training

- ✅ **Wav2Vec2-XLS-R-300M** fine-tuning
- ✅ **CTC Loss** for sequence alignment
- ✅ **SpecAugment** for data augmentation
- ✅ **Mixed Precision** (FP16) training
- ✅ **WER/CER** evaluation metrics

## 📁 Project Structure

```
Kalenjin ASR/
├── cv-corpus-24.0-2025-12-05-kln/     # Raw Common Voice dataset
│   └── cv-corpus-24.0-2025-12-05/
│       └── kln/
│           ├── clips/                  # Audio files (.mp3)
│           ├── train.tsv              # Training metadata
│           ├── dev.tsv                # Validation metadata
│           └── test.tsv               # Test metadata
│
├── processed_data/                     # Preprocessed dataset
│   ├── kalenjin_asr_dataset/          # HuggingFace Dataset format
│   │   ├── train/                     # Training split
│   │   ├── validation/                # Validation split
│   │   ├── test/                      # Test split
│   │   └── dataset_dict.json          # Dataset metadata
│   └── vocab.json                     # Vocabulary mapping
│
├── notebooks/                          # Jupyter notebooks
│   ├── kalenjin_asr_preprocessing_pipeline.ipynb  # Main preprocessing
│   ├── 01_data_verification.ipynb     # Data exploration & validation
│   ├── 02_model_training.ipynb        # Model training
│   └── imports_cell.py                # Reusable imports
│
├── models/                             # Trained models (created during training)
│   └── wav2vec2-kalenjin/
│       ├── processor/                 # Tokenizer & feature extractor
│       ├── final/                     # Final trained model
│       └── test_results.json          # Evaluation results
│
├── results/                            # Quality assessment results
│   └── quality_assessment/
│       ├── quality_report.txt
│       ├── quality_summary.json
│       └── text_quality_results.json
│
├── scripts/                            # Utility scripts
│   ├── quality_assessment.py
│   └── batch_quality_assessment.py
│
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
└── ENVIRONMENT_SETUP.md               # Detailed setup guide
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 50GB+ disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/kalenjin-asr.git
cd kalenjin-asr
```

### Step 2: Create Virtual Environment

```bash
conda create -n audio_ml python=3.10
conda activate audio_ml
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# PyTorch (CPU version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch (GPU version - CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Audio decoding
pip install torchcodec
```

### Step 4: Verify Installation

```bash
python verify_env.py
```

## 📊 Dataset

### Download Common Voice

1. Visit [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
2. Download **Kalenjin (kln)** dataset v24.0
3. Extract to project root:

```bash
tar -xzf cv-corpus-24.0-2025-12-05-kln.tar.gz
```

### Dataset Structure

```
cv-corpus-24.0-2025-12-05-kln/
└── cv-corpus-24.0-2025-12-05/
    └── kln/
        ├── clips/              # 23,141 audio files
        ├── train.tsv          # Training metadata
        ├── dev.tsv            # Validation metadata
        └── test.tsv           # Test metadata
```

## 💻 Usage

### 1. Data Preprocessing

Open and run `notebooks/kalenjin_asr_preprocessing_pipeline.ipynb`:

```python
# The notebook will:
# 1. Load raw Common Voice data
# 2. Apply audio preprocessing (resampling, VAD, normalization)
# 3. Apply text normalization (lowercase, CTC delimiters)
# 4. Calculate quality metrics
# 5. Filter low-quality samples
# 6. Split into train/val/test
# 7. Save to processed_data/kalenjin_asr_dataset/
```

**Output**: Preprocessed dataset in `processed_data/kalenjin_asr_dataset/`

### 2. Data Verification

Open and run `notebooks/01_data_verification.ipynb`:

```python
# The notebook will:
# 1. Load preprocessed dataset
# 2. Display audio samples with waveforms/spectrograms
# 3. Verify text transcriptions
# 4. Show vocabulary mapping
# 5. Display dataset statistics
```

### 3. Model Training

Open and run `notebooks/02_model_training.ipynb`:

```python
# The notebook will:
# 1. Load preprocessed dataset
# 2. Create tokenizer from vocabulary
# 3. Load Wav2Vec2-XLS-R-300M
# 4. Fine-tune with CTC loss
# 5. Evaluate on validation set
# 6. Test on held-out test set
# 7. Save trained model
```

**Output**: Trained model in `models/wav2vec2-kalenjin/final/`

### 4. Inference

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

# Load model
model = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2-kalenjin/final")
processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-kalenjin/final")

# Load audio
audio, sr = librosa.load("path/to/audio.mp3", sr=16000)

# Transcribe
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print(f"Transcription: {transcription}")
```

## 🏗️ Model Architecture

### Wav2Vec2-XLS-R-300M

- **Base Model**: `facebook/wav2vec2-xls-r-300m`
- **Pre-training**: 436K hours of multilingual speech
- **Parameters**: 300M (317M with CTC head)
- **Architecture**: 
  - CNN feature encoder (7 layers)
  - Transformer encoder (24 layers, 1024 hidden size)
  - CTC head (32 output classes)

### Training Configuration

```python
{
    "model": "facebook/wav2vec2-xls-r-300m",
    "vocab_size": 32,
    "batch_size": 8,
    "gradient_accumulation": 2,
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "epochs": 30,
    "fp16": True,
    "optimizer": "AdamW",
    "scheduler": "linear",
    "eval_strategy": "steps",
    "eval_steps": 500
}
```

### Vocabulary

```json
{
  "[PAD]": 0,
  "[UNK]": 1,
  "[CTC]": 2,
  "|": 3,
  " ": 4,
  "'": 5,
  "a": 6, "b": 7, "c": 8, "d": 9, "e": 10,
  "f": 11, "g": 12, "h": 13, "i": 14, "j": 15,
  "k": 16, "l": 17, "m": 18, "n": 19, "o": 20,
  "p": 21, "q": 22, "r": 23, "s": 24, "t": 25,
  "u": 26, "v": 27, "w": 28, "x": 29, "y": 30, "z": 31
}
```

## 📈 Results

### Dataset Statistics

| Split | Samples | Duration | Mean SNR | Mean Duration |
|-------|---------|----------|----------|---------------|
| Train | 11,057 | ~8.5h | 45.9 dB | 2.77s |
| Val | 6,409 | ~4.9h | 46.1 dB | 2.75s |
| Test | 5,675 | ~4.3h | 46.0 dB | 2.73s |
| **Total** | **23,141** | **~17.7h** | **46.0 dB** | **2.76s** |

### Model Performance

*Results will be updated after training*

| Metric | Value |
|--------|-------|
| WER (Word Error Rate) | TBD |
| CER (Character Error Rate) | TBD |
| Training Time | TBD |
| GPU Memory | TBD |

## 🔬 Technical Details

### Audio Preprocessing

1. **Resampling**: 16kHz (Nyquist-Shannon theorem compliance)
2. **VAD**: Librosa-based energy thresholding (top_db=20)
3. **Normalization**: Peak normalization to [-1, 1]
4. **Quality Metrics**:
   - SNR (Signal-to-Noise Ratio)
   - RMS (Root Mean Square)
   - Spectral centroid
   - Zero-crossing rate

### Text Normalization

1. **Lowercase**: All text converted to lowercase
2. **Apostrophe Normalization**: `'`, `'`, `` ` `` → `'`
3. **Character Filtering**: Keep only `[a-z']` and space
4. **CTC Delimiter**: Space → `|` for word boundaries
5. **Whitespace Collapse**: Multiple spaces → single space

### Quality Filtering

- **Duration**: 0.5s ≤ duration ≤ 20s
- **SNR**: SNR ≥ 10 dB
- **Text Length**: 1 ≤ characters ≤ 200
- **Outlier Removal**: Z-score based (|z| < 3)

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{kalenjin-asr-2025,
  author = {Your Name},
  title = {Kalenjin ASR: Low-Resource Speech Recognition for Kalenjin Language},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/kalenjin-asr}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mozilla Common Voice**: For providing the Kalenjin speech dataset
- **Facebook AI**: For Wav2Vec2-XLS-R pre-trained models
- **HuggingFace**: For transformers library and model hosting
- **Kalenjin Community**: For contributing voice recordings

## 📧 Contact

For questions or collaborations:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🔗 Related Work

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [XLS-R Paper](https://arxiv.org/abs/2111.09296)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [Low-Resource ASR Survey](https://arxiv.org/abs/2103.00993)

---

**Status**: 🚧 Active Development | Last Updated: January 2025
