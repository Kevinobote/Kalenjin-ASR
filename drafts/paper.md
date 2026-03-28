# Low-Resource Automatic Speech Recognition for Kalenjin: A Preprocessing and Fine-Tuning Framework Using Wav2Vec2-XLS-R

---

## Abstract

This paper presents a comprehensive framework for building an automatic speech recognition (ASR) system for Kalenjin, a low-resource Nilotic language spoken by approximately 5 million people in Kenya's Rift Valley region. We develop a modular preprocessing pipeline and fine-tune the Wav2Vec2-XLS-R-300M self-supervised model on the Mozilla Common Voice v24.0 Kalenjin corpus. Our preprocessing pipeline incorporates audio quality assessment (SNR estimation, spectral analysis, voice activity detection), Kalenjin-specific text normalization, and multi-stage quality filtering, yielding 23,141 validated samples totaling approximately 20.2 hours of speech. We describe the complete methodology from raw corpus processing through model training with CTC loss, providing a reproducible blueprint for low-resource ASR development.

**Keywords**: automatic speech recognition, low-resource languages, Kalenjin, Wav2Vec2, transfer learning, CTC, speech preprocessing

---

## 1. Introduction

Automatic speech recognition has achieved remarkable performance for high-resource languages such as English, Mandarin, and Spanish, driven by the availability of thousands of hours of transcribed speech data and large-scale pre-trained models. However, the vast majority of the world's approximately 7,000 languages remain underserved by current ASR technology. Kalenjin, a Southern Nilotic language spoken by roughly 5 million people across Kenya's Rift Valley, is one such language. Despite its significant speaker population, Kalenjin lacks the large annotated speech corpora that modern ASR systems typically require.

The emergence of self-supervised pre-trained speech models, particularly Wav2Vec2 (Baevski et al., 2020) and its multilingual variant XLS-R (Babu et al., 2022), has opened new avenues for low-resource ASR. These models learn universal speech representations from hundreds of thousands of hours of unlabeled multilingual audio, enabling effective fine-tuning with limited labeled data. However, the success of fine-tuning critically depends on the quality of the target-language data and the preprocessing pipeline applied to it.

This work makes the following contributions:

1. **A modular preprocessing pipeline** for Kalenjin speech data that integrates audio processing (resampling, VAD-based silence trimming, normalization), Kalenjin-specific text normalization, and multi-dimensional quality assessment.
2. **A quality assessment framework** incorporating signal-to-noise ratio (SNR) estimation, spectral feature analysis, clipping detection, and text quality validation with configurable thresholds.
3. **A fine-tuning configuration** for Wav2Vec2-XLS-R-300M on the processed Kalenjin corpus using Connectionist Temporal Classification (CTC) loss, with detailed training hyperparameters and evaluation methodology.
4. **A reproducible, open-source codebase** structured for extensibility to other low-resource African languages.

---

## 2. Related Work

### 2.1 Self-Supervised Speech Representations

Wav2Vec2 (Baevski et al., 2020) introduced a framework for learning speech representations through contrastive self-supervised learning over masked latent speech representations. The model consists of a CNN-based feature encoder that processes raw audio waveforms, followed by a Transformer encoder that captures contextual relationships. XLS-R (Babu et al., 2022) extended this approach to 128 languages using 436,000 hours of multilingual speech data, demonstrating strong cross-lingual transfer capabilities.

### 2.2 Low-Resource ASR

Prior work on low-resource ASR has explored data augmentation (Park et al., 2019), multilingual pre-training (Conneau et al., 2021), and transfer learning strategies. SpecAugment (Park et al., 2019) has become a standard augmentation technique, applying time and frequency masking to spectrograms during training. For African languages specifically, several efforts have leveraged Common Voice data with pre-trained models, though Kalenjin has received limited attention.

### 2.3 Mozilla Common Voice

The Mozilla Common Voice project provides crowd-sourced speech datasets for over 100 languages. Version 24.0 includes a Kalenjin corpus contributed by native speakers, providing the foundation for this work.

---

## 3. Dataset

### 3.1 Source Corpus

We use the Mozilla Common Voice v24.0 Kalenjin (kln) corpus, which contains crowd-sourced read speech recordings in MP3 format. The corpus is pre-divided into training, development (validation), and test splits based on speaker-disjoint partitioning performed by the Common Voice platform.

**Table 1: Raw corpus statistics**

| Split       | Total Samples | Valid After Processing | Rejection Rate |
|-------------|---------------|----------------------|----------------|
| Train       | 11,065        | 11,057               | 0.07%          |
| Development | 6,412         | 6,409                | 0.05%          |
| Test        | 5,685         | 5,675                | 0.18%          |
| **Total**   | **23,162**    | **23,141**           | **0.09%**      |

### 3.2 Corpus Characteristics

The processed corpus exhibits the following characteristics:

**Table 2: Audio statistics by split**

| Split       | Samples | Total Duration | Mean Duration | Std Duration | Mean SNR  | Std SNR  |
|-------------|---------|----------------|---------------|--------------|-----------|----------|
| Train       | 11,057  | 8.71 h         | 2.84 s        | 1.30 s       | 50.7 dB   | 14.9 dB  |
| Development | 6,409   | 5.38 h         | 3.02 s        | 1.42 s       | 48.9 dB   | 13.3 dB  |
| Test        | 5,675   | 6.13 h         | 3.89 s        | 2.17 s       | 55.5 dB   | 12.1 dB  |
| **Total**   | **23,141** | **20.22 h** | **3.15 s**    | —            | **51.7 dB** | —      |

The high mean SNR values (48.9–55.5 dB) indicate generally clean recordings, consistent with the controlled recording environments typical of Common Voice contributions.

---

## 4. Methodology

### 4.1 System Overview

Our system follows a three-stage pipeline: (1) data preprocessing and quality assessment, (2) dataset structuring in HuggingFace format, and (3) model fine-tuning with CTC loss. Figure 1 illustrates the overall architecture.

```
Raw Common Voice Data (MP3 + TSV)
        │
        ▼
┌─────────────────────────────┐
│   Audio Processing Module   │
│  • MP3 → WAV conversion     │
│  • Resampling to 16 kHz     │
│  • VAD & silence trimming   │
│  • Peak normalization       │
│  • Quality metric extraction│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Text Normalization Module  │
│  • Lowercase conversion     │
│  • Apostrophe normalization │
│  • Orthographic mapping     │
│  • Character filtering      │
│  • CTC delimiter insertion  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Quality Assessment Module  │
│  • SNR thresholding (≥10dB) │
│  • Duration filtering       │
│  • Spectral validation      │
│  • Text quality checks      │
│  • Outlier removal          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Dataset Export (HuggingFace│
│  DatasetDict format)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Wav2Vec2-XLS-R-300M        │
│  Fine-tuning with CTC Loss  │
└─────────────────────────────┘
```

### 4.2 Audio Processing

#### 4.2.1 Resampling

All audio files are resampled to 16 kHz mono, the standard input sampling rate for Wav2Vec2 models. This satisfies the Nyquist-Shannon sampling theorem for speech signals, which contain relevant information up to approximately 8 kHz. We use librosa for robust format handling and resampling.

#### 4.2.2 Voice Activity Detection and Silence Trimming

We apply voice activity detection (VAD) using WebRTC VAD (mode 3, most aggressive) with a frame duration of 30 ms. Additionally, librosa-based energy thresholding (top_db=20) is used to trim leading and trailing silence. The silence threshold is set at an RMS energy of 0.01. This step is critical for removing non-speech segments that would otherwise degrade CTC alignment during training.

The effectiveness of silence trimming is quantified by the reduction ratio metric. For example, the training set shows a mean original duration that is substantially longer than the trimmed duration, with speech ratios averaging approximately 53%, indicating that nearly half of the original audio consisted of silence or non-speech segments.

#### 4.2.3 Audio Normalization

Peak normalization is applied to scale all audio signals to the range [-1, 1], ensuring consistent amplitude levels across the corpus. This prevents the model from learning amplitude-dependent features rather than linguistic content.

#### 4.2.4 Quality Metrics

For each audio sample, we compute the following quality metrics:

- **Signal-to-Noise Ratio (SNR)**: Estimated using noise floor estimation from the quietest 10% of the signal. Samples with SNR < 10 dB are rejected.
- **RMS Energy**: Root mean square energy as a measure of overall signal level.
- **Spectral Centroid**: The "center of mass" of the spectrum, indicating the brightness of the sound. Valid range: 200–8000 Hz.
- **Zero-Crossing Rate (ZCR)**: Rate at which the signal changes sign, used to detect corrupted audio. Maximum threshold: 0.3.
- **Spectral Rolloff**: The frequency below which 85% of the spectral energy is concentrated.
- **Dynamic Range**: The ratio of peak amplitude to RMS energy in dB. Minimum threshold: 6 dB.
- **Clipping Ratio**: Proportion of samples exceeding 99% of maximum amplitude. Maximum threshold: 1%.

### 4.3 Text Normalization

Text normalization for Kalenjin requires language-specific considerations due to the orthographic conventions of the language.

#### 4.3.1 Normalization Steps

1. **Lowercase conversion**: All text is converted to lowercase to reduce vocabulary size.
2. **Apostrophe normalization**: Various Unicode apostrophe characters (', ', `) are standardized to a single ASCII apostrophe ('). This is particularly important for Kalenjin, where the apostrophe is phonemically significant (e.g., in the common morpheme *ng'*).
3. **Orthographic mapping**: Kalenjin-specific rules are applied, including standardization of consonant clusters (e.g., *ch* → *c*) and simplification of aspirated consonants (e.g., *kh* → *k*), while preserving the linguistically important *ng'* cluster.
4. **Character filtering**: Only characters in the set `[a-z']` and space are retained. All other characters (digits, punctuation, diacritics) are removed.
5. **CTC delimiter insertion**: Word-boundary spaces are replaced with the pipe character (`|`) to serve as explicit word delimiters for CTC decoding.
6. **Whitespace normalization**: Multiple consecutive spaces are collapsed to a single space, and leading/trailing whitespace is removed.

#### 4.3.2 Vocabulary

The resulting vocabulary consists of 32 tokens:

- 26 lowercase Latin letters (a–z)
- 3 special tokens: `[PAD]` (padding, index 0), `[UNK]` (unknown, index 1), `[CTC]` (CTC blank, index 2)
- 3 delimiter/punctuation tokens: `|` (word boundary, index 3), ` ` (space, index 4), `'` (apostrophe, index 5)

This compact vocabulary is well-suited for CTC-based ASR, where character-level modeling avoids the need for a large word-level vocabulary and enables open-vocabulary recognition.

### 4.4 Quality Assessment and Filtering

Quality assessment operates at both the audio and text levels, implemented as a configurable pipeline with the following thresholds:

**Table 3: Quality filtering thresholds**

| Criterion                | Threshold          | Rationale                              |
|--------------------------|--------------------|----------------------------------------|
| Minimum duration         | 0.5 s              | Remove breath sounds, clicks           |
| Maximum duration         | 20.0 s             | Memory constraints, alignment issues   |
| Minimum SNR              | 10.0 dB            | Ensure intelligible speech             |
| Maximum ZCR              | 0.3                | Detect corrupted/noisy audio           |
| Spectral centroid range  | 200–8000 Hz        | Validate speech frequency content      |
| Maximum clipping ratio   | 1%                 | Reject clipped recordings              |
| Minimum dynamic range    | 6.0 dB             | Ensure sufficient signal variation     |
| Minimum word count       | 1                   | Reject empty transcriptions            |
| Maximum word count       | 50                  | Remove anomalously long entries        |
| Maximum char repetition  | 3                   | Detect transcription errors            |

The overall rejection rate of 0.09% (21 out of 23,162 samples) indicates high corpus quality, which is expected given that Common Voice contributions undergo community validation.

### 4.5 Dataset Structuring

The processed data is structured into a HuggingFace DatasetDict with three splits (train, validation, test), each containing:
- **audio**: Raw waveform arrays at 16 kHz
- **text**: Normalized transcriptions
- **audio_metrics**: Per-sample quality metrics (SNR, spectral features, duration, etc.)
- **text_metrics**: Per-sample text statistics (word count, character count, etc.)

This format enables direct integration with the HuggingFace Transformers training pipeline.

### 4.6 Model Architecture

We fine-tune Wav2Vec2-XLS-R-300M (Babu et al., 2022), a self-supervised model pre-trained on 436,000 hours of speech data spanning 128 languages. The architecture consists of:

- **CNN Feature Encoder** (7 convolutional layers): Processes raw 16 kHz audio waveforms into latent speech representations at a 20 ms frame rate. This component is frozen during fine-tuning to preserve the learned acoustic features.
- **Transformer Encoder** (24 layers, 1024 hidden dimension, 16 attention heads): Captures long-range contextual dependencies in the latent representations. This component is fine-tuned.
- **CTC Head** (linear projection): Maps the 1024-dimensional Transformer outputs to the 32-class vocabulary for character-level prediction.

Total parameters: ~317M (300M base + CTC head). The feature encoder is frozen during fine-tuning, reducing the number of trainable parameters.

### 4.7 Training Configuration

**Table 4: Training hyperparameters**

| Hyperparameter                | Value              |
|-------------------------------|--------------------|
| Base model                    | facebook/wav2vec2-xls-r-300m |
| Loss function                 | CTC (mean reduction) |
| Optimizer                     | AdamW              |
| Learning rate                 | 3 × 10⁻⁴          |
| Learning rate schedule        | Linear decay       |
| Warmup steps                  | 500                |
| Batch size (per device)       | 8                  |
| Gradient accumulation steps   | 2                  |
| Effective batch size          | 16                 |
| Number of epochs              | 30                 |
| Mixed precision (FP16)        | Yes (if GPU available) |
| Evaluation strategy           | Every 500 steps    |
| Best model selection metric   | WER (lower is better) |
| Save total limit              | 3 checkpoints      |
| Group by length               | Yes                |

**Regularization:**
- Attention dropout: 0.1
- Hidden dropout: 0.1
- Feature projection dropout: 0.0
- Mask time probability: 0.05 (SpecAugment)
- Layer drop: 0.1

**Data Collation:** A custom DataCollatorCTCWithPadding class handles dynamic padding of both input audio features and label sequences, replacing padding positions in labels with -100 to be ignored by the CTC loss computation.

### 4.8 Evaluation Metrics

We evaluate using two standard ASR metrics:

- **Word Error Rate (WER)**: The edit distance between predicted and reference word sequences, normalized by the reference length. WER captures word-level accuracy including substitutions, insertions, and deletions.
- **Character Error Rate (CER)**: The edit distance at the character level, providing a finer-grained measure of transcription accuracy that is particularly informative for morphologically rich languages like Kalenjin.

Both metrics are computed using the HuggingFace `evaluate` library, with CTC greedy decoding (argmax over logits) applied to generate predictions.

---

## 5. Experimental Setup

### 5.1 Exploratory Data Analysis

Prior to preprocessing, we conducted a comprehensive exploratory data analysis (EDA) of the raw Common Voice corpus, examining:

- **Missing data patterns** across metadata fields
- **Audio duration distributions** with descriptive statistics (mean, median, standard deviation, skewness, kurtosis)
- **Audio feature distributions** including spectral centroid, zero-crossing rate, and spectral rolloff across random samples
- **Linguistic analysis** of transcription text, including word frequency distributions, character frequency analysis, and vocabulary statistics
- **Speaker demographics** including gender distribution and age group representation
- **Data quality metrics** comparing validated, invalidated, and other splits

This analysis informed the threshold selection for quality filtering and identified potential biases in the corpus.

### 5.2 Processing Pipeline Execution

The preprocessing pipeline was executed sequentially across all three splits (train, development, test). For each split, the pipeline:

1. Loaded metadata from TSV files and audio from the clips directory
2. Applied audio processing (resampling, VAD, trimming, normalization)
3. Applied text normalization with Kalenjin-specific rules
4. Computed quality metrics for each sample
5. Filtered samples failing quality thresholds
6. Exported results as JSON metadata and HuggingFace Dataset format

Processing was performed with parallel workers (4 processes) for the dataset mapping step.

### 5.3 Batch Quality Assessment

A batch quality assessment script enables parallel processing of the entire corpus with configurable worker counts and batch sizes, supporting both sequential (for debugging) and parallel execution modes. Results are aggregated per split and overall, with detailed per-sample metrics preserved for analysis.

---

## 6. Results

### 6.1 Preprocessing Results

The preprocessing pipeline successfully processed 99.91% of the raw corpus, with only 21 samples rejected across all splits. The primary reasons for rejection were audio loading failures and edge cases in duration filtering.

**Table 5: Audio quality metrics (post-processing)**

| Metric              | Train       | Development | Test        |
|---------------------|-------------|-------------|-------------|
| Mean SNR (dB)       | 50.7 ± 14.9 | 48.9 ± 13.3 | 55.5 ± 12.1 |
| Mean Duration (s)   | 2.84 ± 1.30 | 3.02 ± 1.42 | 3.89 ± 2.17 |
| Total Duration (h)  | 8.71        | 5.38        | 6.13        |
| Speech Ratio        | ~53%        | ~65%        | ~59%        |

The high SNR values across all splits confirm the overall quality of the Common Voice recordings. The speech ratio metric reveals that VAD-based trimming removed 35–47% of the original audio duration, consisting primarily of leading/trailing silence.

### 6.2 Text Normalization Examples

**Table 6: Text normalization examples**

| Original Text                              | Normalized Text                          |
|--------------------------------------------|------------------------------------------|
| Tomo itinye choruet ne chepto iman?        | tomo itinye coruet ne cepto iman         |
| Ming'in lakwana kosir oleipwotchin inendet | ming'in lakwana kosir oleipwotcin inendet |
| Kamet tinyei lakwengung kenyisiek ata      | kamet tinyei lakwengung kenyisiek ata    |

Notable transformations include the *ch* → *c* orthographic mapping, removal of question marks, and preservation of the linguistically significant apostrophe in *ng'in*.

### 6.3 Model Training Results

*Model training results (WER, CER) will be reported upon completion of the training phase.*

---

## 7. Discussion

### 7.1 Preprocessing Pipeline Design

The modular design of our preprocessing pipeline—with separate AudioProcessor, TextNormalizer, QualityAssessor, DatasetStructurer, and PreprocessingPipeline classes—enables independent testing and modification of each component. This is particularly valuable for low-resource language ASR, where language-specific adaptations (e.g., orthographic rules, phonological constraints) may need iterative refinement.

The Kalenjin-specific text normalization decisions merit discussion. The simplification of *ch* to *c* reflects the observation that in Kalenjin orthography, these represent the same phoneme, and reducing graphemic variation helps the CTC model learn more consistent character-to-sound mappings. The preservation of the apostrophe in *ng'* is critical because it distinguishes the velar nasal /ŋ/ from the sequence /ng/, which carries phonemic contrast in Kalenjin.

### 7.2 Data Quality Considerations

The extremely low rejection rate (0.09%) suggests that the Common Voice community validation process is effective for Kalenjin. However, the quality metrics reveal interesting patterns: the test split has notably longer mean duration (3.89s vs. 2.84s for train) and higher SNR (55.5 dB vs. 50.7 dB), which may introduce distribution mismatch between training and evaluation. This is an inherent characteristic of the Common Voice splitting methodology, which prioritizes speaker disjointness over statistical similarity.

### 7.3 Challenges and Limitations

1. **Data scarcity**: At approximately 20 hours total, the Kalenjin corpus is small by ASR standards. While transfer learning from XLS-R mitigates this, performance may be limited compared to high-resource languages.
2. **Speaker diversity**: Common Voice data may not represent the full dialectal variation within Kalenjin, which encompasses several sub-varieties (Nandi, Kipsigis, Tugen, Keiyo, Marakwet, Pokot, Sabaot).
3. **Domain limitation**: The read-speech nature of Common Voice data may not generalize well to spontaneous conversational speech.
4. **Orthographic standardization**: Kalenjin lacks a fully standardized orthography, and our normalization rules represent one possible convention.

### 7.4 Reproducibility

The entire pipeline is implemented in Python using open-source libraries (librosa, HuggingFace Transformers/Datasets, WebRTC VAD) and is structured as Jupyter notebooks with supporting Python scripts. All configuration parameters are defined as dataclasses with documented defaults, enabling exact reproduction of results.

---

## 8. Conclusion and Future Work

We have presented a comprehensive framework for Kalenjin ASR, encompassing data preprocessing, quality assessment, and model fine-tuning. Our modular preprocessing pipeline processes the Mozilla Common Voice v24.0 Kalenjin corpus into a training-ready format with 23,141 validated samples, applying audio processing, Kalenjin-specific text normalization, and multi-dimensional quality filtering. The fine-tuning configuration leverages Wav2Vec2-XLS-R-300M with CTC loss for character-level speech recognition.

Future work includes:

1. **Training completion and evaluation**: Reporting WER and CER on the test set, with error analysis.
2. **Language model integration**: Adding an n-gram or neural language model for decoding to improve word-level accuracy.
3. **Data augmentation**: Exploring speed perturbation, noise injection, and SpecAugment variations to increase effective training data.
4. **Dialect adaptation**: Investigating sub-variety-specific fine-tuning or multi-task learning across Kalenjin dialects.
5. **Downstream applications**: Deploying the model for practical applications such as voice-based information services, educational tools, and language documentation.
6. **Cross-lingual transfer**: Evaluating transfer to related Nilotic languages (e.g., Maasai, Turkana) that share typological features with Kalenjin.

---

## References

Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., ... & Auli, M. (2022). XLS-R: Self-supervised cross-lingual speech representation learning at scale. *arXiv preprint arXiv:2111.09296*.

Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems*, 33, 12449–12460.

Conneau, A., Baevski, A., Collobert, R., Mohamed, A., & Auli, M. (2021). Unsupervised cross-lingual representation learning for speech recognition. *arXiv preprint arXiv:2006.13979*.

Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. *Interspeech 2019*, 2613–2617.

---

*Manuscript prepared January 2025. Model training results pending.*
