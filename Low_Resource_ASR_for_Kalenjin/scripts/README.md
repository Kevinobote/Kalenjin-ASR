# Revision Scripts for Kalenjin ASR Paper

These scripts generate real numbers for the ablation study and error analysis
tables required by reviewers.

## Prerequisites

```bash
pip install modal
modal setup  # if not already authenticated
```

## Step 1: Explore Volume Structure

Run this first to confirm exact paths for model, dataset, and LM files:

```bash
cd scripts/
modal run explore_volume.py
```

Check the output and update paths in the other scripts if needed:
- `dataset_path` — path to the preprocessed test split
- `model_path` — path to the fine-tuned model checkpoint
- `lm_path` — path to the KenLM .arpa or .binary file

## Step 2: Run Ablation Study

```bash
modal run ablation_study.py
```

This produces `/data/ablation_results.json` with WER/CER for:
- Greedy decoding (no LM)
- Beam search + KenLM (full system)
- No VAD trimming (padded silence)
- No text normalization (raw reference)

## Step 3: Run Error Analysis

```bash
modal run error_analysis.py
```

This produces `/data/error_analysis_results.json` with:
- Error type distribution (morpheme boundary, OOV, phonetic confusion, etc.)
- Per-sample WER/CER in `/data/error_analysis_samples.json`

## Step 4: Update Paper

Use the real numbers from JSON outputs to replace the placeholder values in
`conference_101719.tex` (Table VII: Ablation Study, Table VIII: Error Distribution).

## Notes

- All scripts use the same 200 random test samples (seed=42) for consistency
- The volume name is `mozilla-cv-volume` under workspace `viviannyamoraa/main`
- Model fallback: if volume model fails, scripts load from `RareElf/kalenjin-asr`
