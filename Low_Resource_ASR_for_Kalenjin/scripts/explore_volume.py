"""
Volume Explorer - Run this FIRST to check exact paths on Modal volume.
This will tell you where the model, dataset, and LM files are stored.

Usage: modal run explore_volume.py
"""

import modal

app = modal.App("kalenjin-explore-volume")

volume = modal.Volume.from_name("mozilla-cv-volume")

image = modal.Image.debian_slim(python_version="3.10")


@app.function(
    image=image,
    timeout=300,
    volumes={"/data": volume},
)
def explore():
    from pathlib import Path

    data_dir = Path("/data")

    print("=" * 60)
    print("MODAL VOLUME STRUCTURE")
    print("=" * 60)

    # Top-level
    print("\n--- Top level ---")
    for p in sorted(data_dir.iterdir()):
        print(f"  {'[DIR]' if p.is_dir() else '[FILE]'} {p.name}")

    # Dataset directory
    dataset_dir = data_dir / "kalenjin-dataset" / "kln"
    if dataset_dir.exists():
        print(f"\n--- {dataset_dir} ---")
        for p in sorted(dataset_dir.iterdir()):
            if p.is_dir():
                print(f"  [DIR] {p.name}/")
                # Show first few files in subdirs
                subfiles = list(p.iterdir())[:5]
                for sf in subfiles:
                    size = sf.stat().st_size / 1024 if sf.is_file() else 0
                    print(f"    {'[DIR]' if sf.is_dir() else f'[FILE {size:.1f}KB]'} {sf.name}")
                if len(list(p.iterdir())) > 5:
                    print(f"    ... and {len(list(p.iterdir())) - 5} more")
            else:
                size = p.stat().st_size / 1024
                print(f"  [FILE {size:.1f}KB] {p.name}")

    # Models directory
    models_dir = data_dir / "models"
    if models_dir.exists():
        print(f"\n--- {models_dir} ---")
        for p in sorted(models_dir.rglob("*")):
            if p.is_file():
                size = p.stat().st_size / (1024 * 1024)
                rel = p.relative_to(models_dir)
                print(f"  [{size:.1f}MB] {rel}")
    else:
        print("\n--- /data/models NOT FOUND ---")
        # Search for model files
        print("Searching for model files (.bin, .safetensors, config.json)...")
        for p in data_dir.rglob("config.json"):
            print(f"  Found config: {p}")
        for p in data_dir.rglob("*.safetensors"):
            print(f"  Found model: {p}")
        for p in data_dir.rglob("*.bin"):
            if p.stat().st_size > 1024 * 1024:  # > 1MB
                print(f"  Found large .bin: {p} ({p.stat().st_size / (1024*1024):.1f}MB)")

    # Search for LM files
    print("\n--- Language Model files ---")
    for ext in ["*.arpa", "*.binary", "*kenlm*", "*lm*"]:
        for p in data_dir.rglob(ext):
            if p.is_file():
                size = p.stat().st_size / (1024 * 1024)
                print(f"  [{size:.1f}MB] {p}")

    if not list(data_dir.rglob("*.arpa")) and not list(data_dir.rglob("*kenlm*")):
        print("  No LM files found!")

    print("\n" + "=" * 60)


@app.local_entrypoint()
def main():
    explore.remote()
