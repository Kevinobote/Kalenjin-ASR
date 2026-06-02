"""
Error Analysis Script for Kalenjin ASR
Runs on Modal to access GPU + volumes.

Categorizes errors into:
- Morpheme boundary (merge/split errors)
- OOV / rare vocabulary
- Phonetic confusion (ch/c, ng'/n, etc.)
- Deletion (short function words dropped)
- Insertion (extra tokens)

Usage: modal run error_analysis.py
"""

import modal

app = modal.App("kalenjin-error-analysis")

volume = modal.Volume.from_name("mozilla-cv-volume", create_modal_client=None)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "jiwer",
        "pyctcdecode",
        "kenlm",
        "librosa",
        "numpy",
        "pandas",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/data": volume},
)
def run_error_analysis():
    import torch
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from datasets import load_from_disk
    from jiwer import wer, cer
    from collections import Counter

    # --- Paths ---
    dataset_path = "/data/kalenjin-dataset/kln"
    model_path = "/data/models"  # Adjust if model is in a subfolder

    # Try loading model from volume first, fall back to HuggingFace
    try:
        print(f"Loading model from volume: {model_path}")
        # List contents to find the right path
        model_dir = Path(model_path)
        print(f"Contents of {model_path}: {list(model_dir.iterdir())}")
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except Exception as e:
        print(f"Volume load failed ({e}), loading from HuggingFace...")
        processor = Wav2Vec2Processor.from_pretrained("RareElf/kalenjin-asr")
        model = Wav2Vec2ForCTC.from_pretrained("RareElf/kalenjin-asr")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    # --- Load test dataset ---
    print(f"Loading dataset from: {dataset_path}")
    dataset_dir = Path(dataset_path)
    print(f"Contents: {list(dataset_dir.iterdir())}")

    # Try loading preprocessed test split
    try:
        test_ds = load_from_disk(str(dataset_dir / "test"))
    except Exception:
        # Maybe it's a single dataset with splits
        ds = load_from_disk(str(dataset_dir))
        test_ds = ds["test"] if "test" in ds else ds

    print(f"Test set size: {len(test_ds)}")

    # --- Run inference on 200 samples ---
    num_samples = min(200, len(test_ds))
    np.random.seed(42)
    indices = np.random.choice(len(test_ds), num_samples, replace=False)

    results = []
    for idx in indices:
        sample = test_ds[int(idx)]

        # Get audio - handle different column names
        if "audio" in sample:
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
        elif "input_values" in sample:
            audio = np.array(sample["input_values"])
            sr = 16000
        else:
            print(f"Available keys: {sample.keys()}")
            raise KeyError("Cannot find audio data in sample")

        # Get reference text
        ref = sample.get("sentence", sample.get("text", sample.get("normalized_text", "")))
        ref = ref.lower().strip()

        # Run inference
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        # Greedy decode
        predicted_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(predicted_ids)[0].lower().strip()

        results.append({
            "index": int(idx),
            "reference": ref,
            "prediction": pred,
            "wer": wer(ref, pred) if ref else 1.0,
            "cer": cer(ref, pred) if ref else 1.0,
        })

    # --- Error categorization ---
    # Kalenjin function words (high frequency, short)
    function_words = {"eng", "ko", "ne", "ak", "si", "ago", "bo", "ke", "che", "ce",
                      "ab", "en", "mi", "ki", "ka", "the", "a", "e", "i", "o"}

    # Phonetic confusion pairs
    confusion_pairs = [("ch", "c"), ("ng'", "n"), ("ng'", "ng"), ("kh", "k"),
                       ("b", "p"), ("d", "t"), ("g", "k")]

    error_counts = Counter()
    total_word_errors = 0

    for r in results:
        ref_words = r["reference"].split()
        pred_words = r["prediction"].split()

        # Simple alignment via edit operations
        # Deletions: words in ref not in pred
        ref_set = set(ref_words)
        pred_set = set(pred_words)

        # Count deletions of function words
        for w in ref_words:
            if w in function_words and w not in pred_words:
                error_counts["deletion_function_word"] += 1
                total_word_errors += 1

        # Check for merge errors (two ref words merged into one pred word)
        for pw in pred_words:
            for i in range(len(ref_words) - 1):
                merged = ref_words[i] + ref_words[i + 1]
                if pw == merged:
                    error_counts["morpheme_boundary_merge"] += 1
                    total_word_errors += 1

        # Check for split errors (one ref word split into two pred words)
        for rw in ref_words:
            for j in range(len(pred_words) - 1):
                split_form = pred_words[j] + pred_words[j + 1]
                if rw == split_form:
                    error_counts["morpheme_boundary_split"] += 1
                    total_word_errors += 1

        # Phonetic confusion: words differ only by confusion pair
        for rw in ref_words:
            # Find closest pred word
            for pw in pred_words:
                if rw != pw and len(rw) > 0 and len(pw) > 0:
                    for orig, confused in confusion_pairs:
                        if rw.replace(orig, confused) == pw or rw.replace(confused, orig) == pw:
                            error_counts["phonetic_confusion"] += 1
                            total_word_errors += 1
                            break

        # OOV: words in reference that are rare (not in function words, length > 6)
        for rw in ref_words:
            if rw not in function_words and len(rw) > 6:
                # Check if prediction got it wrong
                if rw not in pred_words:
                    error_counts["rare_oov"] += 1
                    total_word_errors += 1

        # Insertions: words in pred not traceable to ref
        extra = len(pred_words) - len(ref_words)
        if extra > 0:
            error_counts["insertion"] += extra
            total_word_errors += extra

    # --- Compute percentages ---
    total_errors = sum(error_counts.values()) if error_counts else 1
    morpheme_boundary = error_counts.get("morpheme_boundary_merge", 0) + error_counts.get("morpheme_boundary_split", 0)

    error_distribution = {
        "morpheme_boundary": morpheme_boundary / total_errors * 100,
        "rare_oov": error_counts.get("rare_oov", 0) / total_errors * 100,
        "phonetic_confusion": error_counts.get("phonetic_confusion", 0) / total_errors * 100,
        "deletion_function_word": error_counts.get("deletion_function_word", 0) / total_errors * 100,
        "insertion": error_counts.get("insertion", 0) / total_errors * 100,
    }
    error_distribution["other"] = 100 - sum(error_distribution.values())

    # --- Summary stats ---
    wers = [r["wer"] for r in results]
    cers = [r["cer"] for r in results]

    summary = {
        "num_samples": num_samples,
        "mean_wer": np.mean(wers),
        "mean_cer": np.mean(cers),
        "median_wer": np.median(wers),
        "median_cer": np.median(cers),
        "error_distribution_pct": error_distribution,
        "raw_error_counts": dict(error_counts),
        "total_categorized_errors": total_errors,
    }

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Samples analyzed: {num_samples}")
    print(f"Mean WER: {summary['mean_wer']:.4f}")
    print(f"Mean CER: {summary['mean_cer']:.4f}")
    print(f"\nError Distribution:")
    for etype, pct in sorted(error_distribution.items(), key=lambda x: -x[1]):
        print(f"  {etype:30s}: {pct:5.1f}%")
    print(f"\nRaw counts: {dict(error_counts)}")

    # Save results
    output_path = "/data/error_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save per-sample results
    samples_path = "/data/error_analysis_samples.json"
    with open(samples_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Per-sample results saved to {samples_path}")

    return summary


@app.local_entrypoint()
def main():
    result = run_error_analysis.remote()
    print("\n\nFinal Summary:")
    print(json.dumps(result, indent=2))
    import json
