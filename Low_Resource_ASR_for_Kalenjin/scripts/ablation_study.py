"""
Ablation Study Script for Kalenjin ASR
Runs on Modal to access GPU + volumes.

Tests the following configurations:
1. Stage 2 model + KenLM beam search (full system) → baseline WER
2. Stage 2 model + greedy decoding (no LM) → isolates LM contribution
3. Stage 2 model with raw audio (no VAD trim) → isolates VAD contribution
4. Stage 2 model with raw text (no normalization) → isolates text norm contribution

Usage: modal run ablation_study.py
"""

import modal

app = modal.App("kalenjin-ablation-study")

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
def run_ablation():
    import torch
    import json
    import re
    import numpy as np
    from pathlib import Path
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from datasets import load_from_disk
    from jiwer import wer, cer
    import librosa

    # --- Paths ---
    dataset_path = "/data/kalenjin-dataset/kln"
    model_path = "/data/models"
    lm_path = None  # Will search for .arpa or .bin file

    # --- Find LM file ---
    data_dir = Path("/data")
    for p in data_dir.rglob("*.arpa"):
        lm_path = str(p)
        break
    if not lm_path:
        for p in data_dir.rglob("*.bin"):
            if "kenlm" in str(p).lower() or "lm" in str(p).lower():
                lm_path = str(p)
                break

    print(f"LM path found: {lm_path}")

    # List all contents to help debug
    print("\n--- Volume structure ---")
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and not str(p).startswith("/data/kalenjin-dataset/kln/audio"):
            print(f"  {p} ({p.stat().st_size / 1024:.1f} KB)")
    print("--- End structure ---\n")

    # --- Load model ---
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except Exception:
        # Try subdirectories
        model_dir = Path(model_path)
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()] if model_dir.exists() else []
        print(f"Model subdirs: {subdirs}")
        if subdirs:
            model_path = str(subdirs[-1])  # Use latest
            processor = Wav2Vec2Processor.from_pretrained(model_path)
            model = Wav2Vec2ForCTC.from_pretrained(model_path)
        else:
            print("Falling back to HuggingFace...")
            processor = Wav2Vec2Processor.from_pretrained("RareElf/kalenjin-asr")
            model = Wav2Vec2ForCTC.from_pretrained("RareElf/kalenjin-asr")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    # --- Load test dataset ---
    dataset_dir = Path(dataset_path)
    try:
        test_ds = load_from_disk(str(dataset_dir / "test"))
    except Exception:
        try:
            ds = load_from_disk(str(dataset_dir))
            test_ds = ds["test"] if "test" in ds else ds
        except Exception:
            # Try loading raw CSV/TSV
            print(f"Dataset dir contents: {list(dataset_dir.iterdir())}")
            raise

    print(f"Test set size: {len(test_ds)}")
    print(f"Columns: {test_ds.column_names}")

    # --- Setup LM decoder if available ---
    lm_decoder = None
    if lm_path:
        try:
            from pyctcdecode import build_ctcdecoder

            vocab = processor.tokenizer.get_vocab()
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            labels = [k for k, v in sorted_vocab]

            lm_decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=lm_path,
                alpha=0.6,
                beta=1.0,
            )
            print("LM decoder built successfully")
        except Exception as e:
            print(f"Failed to build LM decoder: {e}")

    # --- Kalenjin text normalization function ---
    def normalize_kalenjin(text):
        """Apply Kalenjin-specific text normalization."""
        text = text.lower()
        # Apostrophe standardization
        text = text.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
        # Orthographic mapping
        text = text.replace("ch", "c").replace("kh", "k")
        # Keep only [a-z'] and space
        text = re.sub(r"[^a-z' ]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_raw_text(sample):
        """Get text before normalization (original sentence)."""
        return sample.get("sentence", sample.get("text", ""))

    # --- Inference function ---
    def run_inference(audio_array, sr=16000, use_lm=False):
        """Run model inference and return prediction."""
        # Resample if needed
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        if use_lm and lm_decoder:
            # Beam search with LM
            logits_np = logits.cpu().numpy()[0]
            pred = lm_decoder.decode(logits_np, beam_width=100)
        else:
            # Greedy decoding
            predicted_ids = torch.argmax(logits, dim=-1)
            pred = processor.batch_decode(predicted_ids)[0]

        return pred.lower().strip()

    # --- Sample selection (use same 200 for all ablations) ---
    num_samples = min(200, len(test_ds))
    np.random.seed(42)
    indices = np.random.choice(len(test_ds), num_samples, replace=False)

    # --- Ablation 1: Full system (greedy, processed audio + normalized text) ---
    print("\n--- Ablation 1: Greedy decoding (no LM) ---")
    greedy_results = []
    for i, idx in enumerate(indices):
        sample = test_ds[int(idx)]
        if "audio" in sample:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
        elif "input_values" in sample:
            audio = np.array(sample["input_values"], dtype=np.float32)
            sr = 16000
        else:
            continue

        ref = normalize_kalenjin(get_raw_text(sample))
        if not ref:
            continue

        pred = run_inference(audio, sr, use_lm=False)
        greedy_results.append({"ref": ref, "pred": pred})

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    greedy_wer = wer([r["ref"] for r in greedy_results], [r["pred"] for r in greedy_results])
    greedy_cer = cer([r["ref"] for r in greedy_results], [r["pred"] for r in greedy_results])
    print(f"  Greedy WER: {greedy_wer:.4f}, CER: {greedy_cer:.4f}")

    # --- Ablation 2: With LM beam search ---
    lm_wer, lm_cer = None, None
    if lm_decoder:
        print("\n--- Ablation 2: Beam search + KenLM ---")
        lm_results = []
        for i, idx in enumerate(indices):
            sample = test_ds[int(idx)]
            if "audio" in sample:
                audio = np.array(sample["audio"]["array"], dtype=np.float32)
                sr = sample["audio"]["sampling_rate"]
            elif "input_values" in sample:
                audio = np.array(sample["input_values"], dtype=np.float32)
                sr = 16000
            else:
                continue

            ref = normalize_kalenjin(get_raw_text(sample))
            if not ref:
                continue

            pred = run_inference(audio, sr, use_lm=True)
            lm_results.append({"ref": ref, "pred": pred})

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{num_samples}")

        lm_wer = wer([r["ref"] for r in lm_results], [r["pred"] for r in lm_results])
        lm_cer = cer([r["ref"] for r in lm_results], [r["pred"] for r in lm_results])
        print(f"  LM WER: {lm_wer:.4f}, CER: {lm_cer:.4f}")

    # --- Ablation 3: Without VAD trimming (use raw audio without trim) ---
    print("\n--- Ablation 3: No VAD trimming ---")
    no_vad_results = []
    for i, idx in enumerate(indices):
        sample = test_ds[int(idx)]
        if "audio" in sample:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
        elif "input_values" in sample:
            audio = np.array(sample["input_values"], dtype=np.float32)
            sr = 16000
        else:
            continue

        ref = normalize_kalenjin(get_raw_text(sample))
        if not ref:
            continue

        # Add silence padding to simulate no VAD (pad 1s silence on each side)
        silence = np.zeros(16000, dtype=np.float32)
        audio_no_vad = np.concatenate([silence, audio, silence])

        pred = run_inference(audio_no_vad, 16000, use_lm=False)
        no_vad_results.append({"ref": ref, "pred": pred})

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    no_vad_wer = wer([r["ref"] for r in no_vad_results], [r["pred"] for r in no_vad_results])
    no_vad_cer = cer([r["ref"] for r in no_vad_results], [r["pred"] for r in no_vad_results])
    print(f"  No-VAD WER: {no_vad_wer:.4f}, CER: {no_vad_cer:.4f}")

    # --- Ablation 4: Without text normalization (compare against raw text) ---
    print("\n--- Ablation 4: No text normalization (raw reference) ---")
    no_norm_results = []
    for i, idx in enumerate(indices):
        sample = test_ds[int(idx)]
        if "audio" in sample:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
        elif "input_values" in sample:
            audio = np.array(sample["input_values"], dtype=np.float32)
            sr = 16000
        else:
            continue

        # Use raw text (only lowercase, no Kalenjin-specific normalization)
        raw_text = get_raw_text(sample).lower().strip()
        raw_text = re.sub(r"[^a-z' ]", "", raw_text)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        if not raw_text:
            continue

        pred = run_inference(audio, sr, use_lm=False)
        no_norm_results.append({"ref": raw_text, "pred": pred})

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    no_norm_wer = wer([r["ref"] for r in no_norm_results], [r["pred"] for r in no_norm_results])
    no_norm_cer = cer([r["ref"] for r in no_norm_results], [r["pred"] for r in no_norm_results])
    print(f"  No-norm WER: {no_norm_wer:.4f}, CER: {no_norm_cer:.4f}")

    # --- Summary ---
    ablation_results = {
        "num_samples": num_samples,
        "configurations": {
            "greedy_decoding": {
                "description": "Stage 2 model, greedy decoding, processed audio, normalized text",
                "wer": greedy_wer,
                "cer": greedy_cer,
            },
            "beam_search_kenlm": {
                "description": "Stage 2 model + KenLM beam search (full system)",
                "wer": lm_wer,
                "cer": lm_cer,
            },
            "no_vad_trimming": {
                "description": "Stage 2 model, greedy, audio padded with 1s silence (simulating no VAD)",
                "wer": no_vad_wer,
                "cer": no_vad_cer,
            },
            "no_text_normalization": {
                "description": "Stage 2 model, greedy, raw text without ch→c / kh→k mapping",
                "wer": no_norm_wer,
                "cer": no_norm_cer,
            },
        },
        "ablation_deltas": {
            "LM_contribution_wer": (greedy_wer - lm_wer) if lm_wer else None,
            "LM_contribution_cer": (greedy_cer - lm_cer) if lm_cer else None,
            "VAD_contribution_wer": no_vad_wer - greedy_wer,
            "VAD_contribution_cer": no_vad_cer - greedy_cer,
            "TextNorm_contribution_wer": no_norm_wer - greedy_wer,
            "TextNorm_contribution_cer": no_norm_cer - greedy_cer,
        },
    }

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<35} {'WER':>8} {'CER':>8}")
    print("-" * 55)
    for name, cfg in ablation_results["configurations"].items():
        w = f"{cfg['wer']:.4f}" if cfg['wer'] is not None else "N/A"
        c = f"{cfg['cer']:.4f}" if cfg['cer'] is not None else "N/A"
        print(f"{name:<35} {w:>8} {c:>8}")

    print(f"\n{'Component Contribution':<35} {'ΔWER':>8} {'ΔCER':>8}")
    print("-" * 55)
    for name, val in ablation_results["ablation_deltas"].items():
        v = f"{val:+.4f}" if val is not None else "N/A"
        print(f"{name:<35} {v:>8}")

    # Save
    output_path = "/data/ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return ablation_results


@app.local_entrypoint()
def main():
    import json
    result = run_ablation.remote()
    print("\n\nFinal Ablation Results:")
    print(json.dumps(result, indent=2))
