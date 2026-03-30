# Kalenjin ASR: Low-Resource Speech Recognition

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Mozilla Common Voice](https://img.shields.io/badge/Dataset-Common%20Voice%2024.0-orange.svg)](https://commonvoice.mozilla.org/)
[![Model: HuggingFace](https://img.shields.io/badge/Model-RareElf%2Fkalenjin--asr-yellow.svg)](https://huggingface.co/RareElf/kalenjin-asr)

A PhD-level automatic speech recognition (ASR) system for Kalenjin, a low-resource Southern Nilotic language spoken by 5–6 million people in Kenya's Rift Valley. This project implements a two-stage fine-tuning strategy for Wav2Vec2-XLS-R-300M with KenLM language model integration.

## 📊 Results

| System | WER | CER | Improvement |
|--------|-----|-----|-------------|
| Baseline CPU (10% data) | 100.00% | 100.00% | — |
| GPU Baseline (100%, frozen) | 72.11% | 44.94% | 27.89 pp |
| Stage 2 Refined (unfrozen) | 69.13% | 20.11% | 55.3% CER rel. |
| **Stage 2 + KenLM** | **61.75%** | **20.11%** | **7.38 pp (10.7%)** |

### Comparison with PazaBench (Microsoft Research Africa)

| Model | CER | WER |
|-------|-----|-----|
| Omnilingual | 0.40 | 0.81 |
| Paza | 0.54 | 0.86 |
| Facebook MMS | 0.59 | 1.03 |
| OpenAI Whisper | 0.88 | 1.00 |
| **This work (fine-tuned)** | **0.20** | **0.62** |

Our fine-tuned system achieves CER **50% lower** than the best zero-shot model and WER **23.5% lower** than Omnilingual.

## 🎯 Overview

This project addresses the challenge of building ASR systems for low-resource languages through:

1. **Modular Preprocessing Pipeline**: Audio quality assessment (SNR, spectral features, VAD), Kalenjin-specific text normalization preserving the *ng'* velar nasal marker
2. **Two-Stage Fine-Tuning**: Stage 1 (frozen feature encoder, 30 epochs) → Stage 2 (unfrozen, 15 epochs) achieving 55.3% relative CER reduction
3. **Language Model Integration**: 5-gram KenLM with beam search decoding (α=0.6, β=1.0) providing 10.7% relative WER improvement
4. **Comprehensive EDA**: 19 figures analyzing audio quality, linguistic patterns, and corpus characteristics
5. **Reproducibility**: Full codebase, HuggingFace model, and documented methodology

### Key Statistics

| Metric | Value |
|--------|-------|
| Dataset | Mozilla Common Voice v24.0 (Kalenjin) |
| Total Utterances | 23,141 (after quality filtering) |
| Total Duration | 20.22 hours |
| Training Samples | 11,057 (8.71h) |
| Validation Samples | 6,409 (5.38h) |
| Test Samples | 5,675 (6.13h) |
| Vocabulary Size | 32 tokens |
| Mean SNR | 51.7 dB |
| Model | Wav2Vec2-XLS-R-300M (315.5M params) |
| Final WER | 61.75% |
| Final CER | 20.11% |

## 🤗 Model

The trained model is available on HuggingFace:

```python
from transformers import AutoProcessor, AutoModelForCTC

processor = AutoProcessor.from_pretrained("RareElf/kalenjin-asr")
model = AutoModelForCTC.from_pretrained("RareElf/kalenjin-asr")
```

Or use the pipeline:

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="RareElf/kalenjin-asr")
result = pipe("path/to/audio.mp3")
print(result["text"])
```

## 📁 Project Structure

```
Kalenjin-ASR/
├── Low_Resource_ASR_for_Kalenjin/   # IEEE Conference Paper
│   ├── conference_101719.tex        # Main LaTeX source
│   ├── conference_101719.pdf        # Compiled paper (14 pages)
│   ├── figures/                     # 19 generated figures
│   ├── IEEEtran.cls               # IEEE template class
│   └── TODO.txt                    # Completion checklist
│
├── journal/                         # Dissertation-style introduction
│   └── introduction/
│       ├── introduction.tex
│       └── references.bib
│
├── notebooks/                       # Jupyter notebooks
│   ├── kalenjin_asr_preprocessing_pipeline.ipynb  # Main preprocessing
│   ├── kalenjin_asr_advanced_eda.ipynb            # Advanced EDA
│   ├── kalenjin_cv_eda.ipynb                      # Common Voice EDA
│   ├── 01_data_verification.ipynb                 # Data verification
│   ├── 02_model_training.ipynb                    # Model training
│   ├── 03_qualitative_evaluation.ipynb            # Qualitative eval
│   └── utils/                                     # Helper scripts
│
├── processed_data/                  # Processing results
│   ├── train_results.json          # Training split metrics
│   ├── dev_results.json            # Validation split metrics
│   ├── test_results.json           # Test split metrics
│   ├── vocab.json                  # 32-token vocabulary
│   └── qualitative_results.json    # Model predictions (50 samples)
│
├── results/                         # Quality assessment
│   └── quality_assessment/
│
├── scripts/                         # Utility scripts
│   ├── quality_assessment.py
│   ├── batch_quality_assessment.py
│   └── helpers/
│
├── models/                          # Training logs (from Modal Cloud)
│   └── training_logs/
│
├── data/                            # Raw Common Voice data (not in git)
│   └── kln/
│       ├── clips/                   # Audio files (.mp3)
│       ├── train.tsv
│       ├── dev.tsv
│       └── test.tsv
│
├── drafts/                          # Archived earlier drafts
├── METHODOLOGY.pdf                  # Methodology chapter
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/Kevinobote/Kalenjin-ASR.git
cd Kalenjin-ASR

# Create conda environment
conda create -n audio_ml python=3.10
conda activate audio_ml

# Install dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets librosa soundfile jiwer pyctcdecode kenlm
pip install ipykernel
python -m ipykernel install --user --name audio_ml --display-name "Python (audio_ml)"
```

## 💻 Usage

### Quick Inference

```python
from transformers import AutoProcessor, AutoModelForCTC
import librosa, torch

processor = AutoProcessor.from_pretrained("RareElf/kalenjin-asr")
model = AutoModelForCTC.from_pretrained("RareElf/kalenjin-asr")

audio, sr = librosa.load("path/to/audio.mp3", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]
print(transcription)
```

### Qualitative Evaluation

Run `notebooks/03_qualitative_evaluation.ipynb` with the `audio_ml` kernel. This loads the model from HuggingFace, builds a KenLM language model from training data, and evaluates on test samples with both greedy and beam search decoding.

## 🏗️ Model Architecture

- **Base Model**: `facebook/wav2vec2-xls-r-300m` (436K hours, 128 languages)
- **Feature Encoder**: 7 CNN layers (frozen in Stage 1, unfrozen in Stage 2)
- **Transformer Encoder**: 24 layers, 1024 hidden, 16 attention heads
- **CTC Head**: Linear projection to 32-token vocabulary
- **Total Parameters**: 315.5M
- **Language Model**: 5-gram KenLM (17,814 unigrams, Modified Kneser-Ney smoothing)

### Training Configuration

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Feature Encoder | Frozen | Unfrozen |
| Learning Rate | 1×10⁻⁴ | 5×10⁻⁵ |
| Epochs | 30 | 15 |
| Batch Size | 8×4 = 32 | 8×4 = 32 |
| Hardware | NVIDIA A100 (40GB) | NVIDIA A100 (40GB) |
| Training Time | ~8 hours | ~4 hours |

## 📝 Paper

The IEEE conference paper is in `Low_Resource_ASR_for_Kalenjin/`:

**Title**: *Low-Resource ASR for Kalenjin: Two-Stage Fine-Tuning of Wav2Vec2-XLS-R with KenLM Integration*

- 14 pages, 19 figures, 10 tables, 19 equations, 42 references
- IEEE conference format (`IEEEtran.cls`)
- Compile: `cd Low_Resource_ASR_for_Kalenjin && pdflatex conference_101719.tex`

## 📚 Citation

```bibtex
@inproceedings{obote2025kalenjin,
  author    = {Obote, Kevin Omondi and Muli, Annah Mumbua and Mora, Vivian Nyamari and Njiiri, Joyce},
  title     = {Low-Resource {ASR} for {K}alenjin: Two-Stage Fine-Tuning of {W}av2{V}ec2-{XLS}-{R} with {K}en{LM} Integration},
  year      = {2025},
  note      = {Model available at \url{https://huggingface.co/RareElf/kalenjin-asr}}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Mozilla Common Voice**: Kalenjin speech dataset and community contributors
- **Facebook AI**: Wav2Vec2-XLS-R pre-trained models
- **HuggingFace**: Transformers library and model hosting
- **Modal Cloud**: GPU compute infrastructure (NVIDIA A100)
- **Strathmore University** and **Machakos University**: Institutional support
- **Microsoft Research Africa**: PazaBench benchmark for evaluation context
