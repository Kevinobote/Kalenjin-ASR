"""
Combined Error Analysis + Ablation Study for Kalenjin ASR
Single Modal job: loads model once, runs all experiments.

Outputs saved to volume at /data/revision_results/

Usage: modal run combined_revision.py
"""

import modal

app = modal.App("kalenjin-revision-experiments")

volume = modal.Volume.from_name("mozilla-cv-volume")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "jiwer",
        "pyctcdecode",
        "kenlm",
        "librosa",
        "numpy",
        "pandas",
        "soundfile",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/data": volume},
)
def run_all_experiments():
    import torch
    import json
    import re
    import csv
    import numpy as np
    from pathlib import Path
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from jiwer import wer, cer
    from collections import Counter
    import librosa

    # === PATHS ===
    MODEL_PATH = "/data/models/stage2_20260219_2055/final_phd_model"
    LM_PATH = "/data/models/stage2_20260219_2334/kalenjin_lm.bin"
    LM_ARPA_PATH = "/data/models/stage2_20260219_2334/kalenjin_5gram.arpa"
    TEST_TSV = "/data/kalenjin-dataset/kln/test.tsv"
    CLIPS_DIR = "/data/kalenjin-dataset/kln/clips"
    OUTPUT_DIR = Path("/data/revision_results")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # === LOAD MODEL ===
    print("Loading model...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    print(f"Model loaded on {device}")

    # === LOAD TEST DATA ===
    print("Loading test data...")
    test_samples = []
    with open(TEST_TSV, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            test_samples.append({
                "path": row["path"],
                "sentence": row["sentence"],
            })
    print(f"Test set: {len(test_samples)} samples")

    # === TEXT NORMALIZATION ===
    def normalize_kalenjin(text):
        text = text.lower()
        text = text.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
        text = text.replace("ch", "c").replace("kh", "k")
        text = re.sub(r"[^a-z' ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # === BUILD LM DECODER ===
    print("Building LM decoder...")
    from pyctcdecode import build_ctcdecoder

    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    labels = [k for k, v in sorted_vocab]

    # Try binary first, fall back to arpa
    lm_file = LM_PATH if Path(LM_PATH).exists() else LM_ARPA_PATH
    lm_decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_file,
        alpha=0.6,
        beta=1.0,
    )
    print(f"LM decoder built from {lm_file}")

    # === SELECT 200 SAMPLES ===
    np.random.seed(42)
    num_samples = min(200, len(test_samples))
    indices = np.random.choice(len(test_samples), num_samples, replace=False)
    selected = [test_samples[i] for i in indices]
    print(f"Selected {num_samples} samples for evaluation")

    # === INFERENCE HELPER ===
    def run_inference(audio_array, use_lm=False):
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        if use_lm:
            logits_np = logits.cpu().numpy()[0]
            pred = lm_decoder.decode(logits_np, beam_width=100)
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            pred = processor.batch_decode(predicted_ids)[0]
        return pred.lower().strip()

    # === EXPERIMENT 1: GREEDY + LM (200 samples) ===
    print("\n--- Experiment 1: Greedy vs Beam+LM (200 samples) ---")
    results = []
    for i, sample in enumerate(selected):
        audio_path = Path(CLIPS_DIR) / sample["path"]
        if not audio_path.exists():
            # Try .wav in clips_cleaned
            audio_path = Path("/data/kalenjin-dataset/kln/clips_cleaned") / sample["path"].replace(".mp3", ".wav")
        if not audio_path.exists():
            continue

        audio, sr = librosa.load(str(audio_path), sr=16000)
        ref = normalize_kalenjin(sample["sentence"])
        if not ref:
            continue

        greedy_pred = run_inference(audio, use_lm=False)
        lm_pred = run_inference(audio, use_lm=True)

        results.append({
            "file": sample["path"],
            "reference": ref,
            "greedy_pred": greedy_pred,
            "lm_pred": lm_pred,
            "greedy_wer": wer(ref, greedy_pred),
            "greedy_cer": cer(ref, greedy_pred),
            "lm_wer": wer(ref, lm_pred),
            "lm_cer": cer(ref, lm_pred),
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    # Compute aggregate metrics
    refs = [r["reference"] for r in results]
    greedy_preds = [r["greedy_pred"] for r in results]
    lm_preds = [r["lm_pred"] for r in results]

    agg_greedy_wer = wer(refs, greedy_preds)
    agg_greedy_cer = cer(refs, greedy_preds)
    agg_lm_wer = wer(refs, lm_preds)
    agg_lm_cer = cer(refs, lm_preds)

    print(f"\n  Greedy: WER={agg_greedy_wer:.4f}, CER={agg_greedy_cer:.4f}")
    print(f"  Beam+LM: WER={agg_lm_wer:.4f}, CER={agg_lm_cer:.4f}")
    print(f"  LM improvement: {agg_greedy_wer - agg_lm_wer:.4f} WER, {agg_greedy_cer - agg_lm_cer:.4f} CER")

    # === EXPERIMENT 2: NO-VAD ABLATION ===
    print("\n--- Experiment 2: No VAD (raw audio without trimming) ---")
    no_vad_preds = []
    no_vad_refs = []
    for i, sample in enumerate(selected[:200]):
        audio_path = Path(CLIPS_DIR) / sample["path"]
        if not audio_path.exists():
            continue

        # Load raw audio WITHOUT trimming (this is the original audio)
        audio, sr = librosa.load(str(audio_path), sr=16000)
        ref = normalize_kalenjin(sample["sentence"])
        if not ref:
            continue

        pred = run_inference(audio, use_lm=False)
        no_vad_preds.append(pred)
        no_vad_refs.append(ref)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}")

    if no_vad_refs:
        no_vad_wer = wer(no_vad_refs, no_vad_preds)
        no_vad_cer = cer(no_vad_refs, no_vad_preds)
        print(f"  No-VAD (raw clips): WER={no_vad_wer:.4f}, CER={no_vad_cer:.4f}")
    else:
        no_vad_wer, no_vad_cer = None, None

    # === EXPERIMENT 3: NO TEXT NORMALIZATION ===
    print("\n--- Experiment 3: No text normalization (raw text as reference) ---")
    no_norm_results = []
    for i, sample in enumerate(selected[:200]):
        audio_path = Path(CLIPS_DIR) / sample["path"]
        if not audio_path.exists():
            audio_path = Path("/data/kalenjin-dataset/kln/clips_cleaned") / sample["path"].replace(".mp3", ".wav")
        if not audio_path.exists():
            continue

        audio, sr = librosa.load(str(audio_path), sr=16000)

        # Raw text: only lowercase + remove non-alpha (but keep ch, don't map to c)
        raw_text = sample["sentence"].lower().strip()
        raw_text = re.sub(r"[^a-z' ]", "", raw_text)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        if not raw_text:
            continue

        pred = run_inference(audio, use_lm=False)
        no_norm_results.append({"ref": raw_text, "pred": pred})

    if no_norm_results:
        no_norm_wer = wer([r["ref"] for r in no_norm_results], [r["pred"] for r in no_norm_results])
        no_norm_cer = cer([r["ref"] for r in no_norm_results], [r["pred"] for r in no_norm_results])
        print(f"  No-norm: WER={no_norm_wer:.4f}, CER={no_norm_cer:.4f}")
    else:
        no_norm_wer, no_norm_cer = None, None

    # === ERROR ANALYSIS (using LM predictions) ===
    print("\n--- Error Analysis (200 samples, beam+LM) ---")

    function_words = {"eng", "ko", "ne", "ak", "si", "ago", "bo", "ke", "che", "ce",
                      "ab", "en", "mi", "ki", "ka", "a", "e", "i", "o", "u",
                      "eng'", "amu", "noto", "nebo"}

    confusion_pairs = [("c", "ch"), ("ng'", "ng"), ("ng'", "n"),
                       ("b", "p"), ("d", "t"), ("g", "k")]

    error_counts = Counter()

    for r in results:
        ref_words = r["reference"].split()
        pred_words = r["lm_pred"].split()

        # Morpheme boundary: merge
        for pw in pred_words:
            for i in range(len(ref_words) - 1):
                if ref_words[i] + ref_words[i + 1] == pw:
                    error_counts["morpheme_boundary_merge"] += 1

        # Morpheme boundary: split
        for rw in ref_words:
            for j in range(len(pred_words) - 1):
                if pred_words[j] + pred_words[j + 1] == rw:
                    error_counts["morpheme_boundary_split"] += 1

        # Phonetic confusion
        for rw in ref_words:
            for pw in pred_words:
                if rw != pw and len(rw) > 2 and len(pw) > 2:
                    for norm_char, model_char in confusion_pairs:
                        if rw.replace(norm_char, model_char) == pw or pw.replace(model_char, norm_char) == rw:
                            error_counts["phonetic_confusion"] += 1
                            break

        # Deletion of function words
        for rw in ref_words:
            if rw in function_words and rw not in pred_words:
                error_counts["deletion_function_word"] += 1

        # OOV
        for rw in ref_words:
            if rw not in function_words and len(rw) > 6 and rw not in pred_words:
                is_phonetic = any(
                    rw.replace(nc, mc) == pw or pw.replace(mc, nc) == rw
                    for pw in pred_words
                    for nc, mc in confusion_pairs
                )
                if not is_phonetic:
                    error_counts["rare_oov"] += 1

        # Insertions
        extra = max(0, len(pred_words) - len(ref_words))
        error_counts["insertion"] += extra

    total_errors = sum(error_counts.values()) or 1
    morpheme = error_counts.get("morpheme_boundary_merge", 0) + error_counts.get("morpheme_boundary_split", 0)

    error_dist = {
        "morpheme_boundary": round(morpheme / total_errors * 100, 1),
        "rare_oov": round(error_counts.get("rare_oov", 0) / total_errors * 100, 1),
        "phonetic_confusion": round(error_counts.get("phonetic_confusion", 0) / total_errors * 100, 1),
        "deletion_function_word": round(error_counts.get("deletion_function_word", 0) / total_errors * 100, 1),
        "insertion": round(error_counts.get("insertion", 0) / total_errors * 100, 1),
    }

    print(f"\n  Total errors categorized: {total_errors}")
    for etype, pct in sorted(error_dist.items(), key=lambda x: -x[1]):
        print(f"  {etype:<30}: {pct:5.1f}%")

    # === SAVE ALL RESULTS ===
    final_output = {
        "num_samples": len(results),
        "metrics": {
            "greedy_wer": agg_greedy_wer,
            "greedy_cer": agg_greedy_cer,
            "beam_lm_wer": agg_lm_wer,
            "beam_lm_cer": agg_lm_cer,
            "lm_wer_improvement_pp": agg_greedy_wer - agg_lm_wer,
            "lm_wer_improvement_relative": (agg_greedy_wer - agg_lm_wer) / agg_greedy_wer * 100,
        },
        "ablation": {
            "no_vad_wer": no_vad_wer,
            "no_vad_cer": no_vad_cer,
            "no_vad_delta_wer": (no_vad_wer - agg_greedy_wer) if no_vad_wer else None,
            "no_text_norm_wer": no_norm_wer,
            "no_text_norm_cer": no_norm_cer,
            "no_text_norm_delta_wer": (no_norm_wer - agg_greedy_wer) if no_norm_wer else None,
        },
        "error_analysis": {
            "total_errors": total_errors,
            "distribution_pct": error_dist,
            "raw_counts": dict(error_counts),
        },
    }

    # Save summary
    with open(OUTPUT_DIR / "revision_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    # Save per-sample predictions
    with open(OUTPUT_DIR / "predictions_200.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL RESULTS SAVED to /data/revision_results/")
    print("=" * 60)
    print(json.dumps(final_output, indent=2))

    volume.commit()
    return final_output


@app.local_entrypoint()
def main():
    import json
    result = run_all_experiments.remote()
    print("\n\n=== FINAL RESULTS ===")
    print(json.dumps(result, indent=2))

    # Save locally too
    output_path = "/home/obote/Documents/Journals/Low_Resource_ASR_for_Kalenjin/scripts/revision_results.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved locally to: {output_path}")
