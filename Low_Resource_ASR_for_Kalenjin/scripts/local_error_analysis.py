"""
Local Error Analysis for Kalenjin ASR
Uses existing qualitative_results.json (50 predictions) to categorize error types.
No GPU or Modal needed - runs locally.

Usage: python3 local_error_analysis.py
"""

import json
from collections import Counter

# Load predictions
with open("/home/obote/Documents/Hineni/Kalenjin-ASR/processed_data/qualitative_results.json") as f:
    data = json.load(f)

predictions = data["predictions"]

# Kalenjin function words
function_words = {"eng", "ko", "ne", "ak", "si", "ago", "bo", "ke", "che", "ce",
                  "ab", "en", "mi", "ki", "ka", "the", "a", "e", "i", "o", "u",
                  "eng'", "amu", "noto", "nebo"}

# Phonetic confusion pairs (normalized → model tends to output)
confusion_pairs = [
    ("c", "ch"),      # ch→c normalization reversal
    ("ng'", "ng"),    # velar nasal confusion
    ("ng'", "n"),     # velar nasal dropped
    ("b", "p"),       # voicing confusion
    ("d", "t"),       # voicing confusion
    ("g", "k"),       # voicing confusion
]

error_counts = Counter()
total_word_errors = 0
detailed_examples = {
    "morpheme_boundary": [],
    "rare_oov": [],
    "phonetic_confusion": [],
    "deletion": [],
    "insertion": [],
}

for pred in predictions:
    ref_words = pred["reference"].split()
    # Use beam prediction (full system)
    pred_words = pred["beam_pred"].split()

    # --- Morpheme boundary: merge errors ---
    for pw in pred_words:
        for i in range(len(ref_words) - 1):
            merged = ref_words[i] + ref_words[i + 1]
            if pw == merged:
                error_counts["morpheme_boundary_merge"] += 1
                total_word_errors += 1
                detailed_examples["morpheme_boundary"].append(
                    f"  Merge: '{ref_words[i]}' + '{ref_words[i+1]}' → '{pw}'"
                )

    # --- Morpheme boundary: split errors ---
    for rw in ref_words:
        for j in range(len(pred_words) - 1):
            split_form = pred_words[j] + pred_words[j + 1]
            if rw == split_form:
                error_counts["morpheme_boundary_split"] += 1
                total_word_errors += 1
                detailed_examples["morpheme_boundary"].append(
                    f"  Split: '{rw}' → '{pred_words[j]}' + '{pred_words[j+1]}'"
                )

    # --- Phonetic confusion ---
    for rw in ref_words:
        for pw in pred_words:
            if rw != pw and len(rw) > 2 and len(pw) > 2:
                for norm_char, model_char in confusion_pairs:
                    # Check if the only difference is this confusion pair
                    if rw.replace(norm_char, model_char) == pw:
                        error_counts["phonetic_confusion"] += 1
                        total_word_errors += 1
                        detailed_examples["phonetic_confusion"].append(
                            f"  '{rw}' → '{pw}' ({norm_char}→{model_char})"
                        )
                        break
                    elif pw.replace(model_char, norm_char) == rw:
                        error_counts["phonetic_confusion"] += 1
                        total_word_errors += 1
                        detailed_examples["phonetic_confusion"].append(
                            f"  '{rw}' → '{pw}' ({norm_char}↔{model_char})"
                        )
                        break

    # --- Deletion of function words ---
    for rw in ref_words:
        if rw in function_words and rw not in pred_words:
            error_counts["deletion_function_word"] += 1
            total_word_errors += 1
            detailed_examples["deletion"].append(f"  Deleted: '{rw}'")

    # --- OOV / rare vocabulary ---
    for rw in ref_words:
        if rw not in function_words and len(rw) > 6:
            if rw not in pred_words:
                # Check if it's not just a phonetic confusion (already counted)
                is_phonetic = False
                for pw in pred_words:
                    for norm_char, model_char in confusion_pairs:
                        if rw.replace(norm_char, model_char) == pw or pw.replace(model_char, norm_char) == rw:
                            is_phonetic = True
                            break
                    if is_phonetic:
                        break
                if not is_phonetic:
                    error_counts["rare_oov"] += 1
                    total_word_errors += 1
                    detailed_examples["rare_oov"].append(f"  OOV: '{rw}'")

    # --- Insertions ---
    extra = max(0, len(pred_words) - len(ref_words))
    if extra > 0:
        error_counts["insertion"] += extra
        total_word_errors += extra

# Compute percentages
total = sum(error_counts.values()) or 1
morpheme = error_counts.get("morpheme_boundary_merge", 0) + error_counts.get("morpheme_boundary_split", 0)

results = {
    "morpheme_boundary": {
        "count": morpheme,
        "pct": morpheme / total * 100,
        "merge": error_counts.get("morpheme_boundary_merge", 0),
        "split": error_counts.get("morpheme_boundary_split", 0),
    },
    "rare_oov": {
        "count": error_counts.get("rare_oov", 0),
        "pct": error_counts.get("rare_oov", 0) / total * 100,
    },
    "phonetic_confusion": {
        "count": error_counts.get("phonetic_confusion", 0),
        "pct": error_counts.get("phonetic_confusion", 0) / total * 100,
    },
    "deletion_function_word": {
        "count": error_counts.get("deletion_function_word", 0),
        "pct": error_counts.get("deletion_function_word", 0) / total * 100,
    },
    "insertion": {
        "count": error_counts.get("insertion", 0),
        "pct": error_counts.get("insertion", 0) / total * 100,
    },
}
results["other"] = {"count": 0, "pct": max(0, 100 - sum(r["pct"] for r in results.values()))}

# Print results
print("=" * 60)
print("ERROR ANALYSIS RESULTS (50-sample qualitative evaluation)")
print("=" * 60)
print(f"\nOverall: Greedy WER={data['greedy_wer']:.4f}, Beam+LM WER={data['beam_wer']:.4f}")
print(f"         Greedy CER={data['greedy_cer']:.4f}, Beam+LM CER={data['beam_cer']:.4f}")
print(f"\nTotal categorized errors: {total}")
print(f"\n{'Error Type':<30} {'Count':>6} {'Percentage':>10}")
print("-" * 50)
for etype, vals in sorted(results.items(), key=lambda x: -x[1]["pct"]):
    print(f"{etype:<30} {vals['count']:>6} {vals['pct']:>9.1f}%")

print(f"\n{'='*60}")
print("DETAILED EXAMPLES (first 5 per category)")
print("=" * 60)
for cat, examples in detailed_examples.items():
    if examples:
        print(f"\n--- {cat} ({len(examples)} total) ---")
        for ex in examples[:5]:
            print(ex)

# Save for paper
output = {
    "source": "qualitative_results.json (50 test samples, beam+KenLM decoding)",
    "total_categorized_errors": total,
    "error_distribution": results,
    "raw_counts": dict(error_counts),
}

output_path = "/home/obote/Documents/Hineni/Kalenjin-ASR/processed_data/error_analysis.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {output_path}")
