#!/usr/bin/env python3
"""
Dependency Verification Script for Kalenjin ASR Pipeline
"""

import sys
from importlib import import_module

# Define all required dependencies
DEPENDENCIES = {
    'Core': ['numpy', 'pandas', 'scipy', 'pathlib'],
    'Audio': ['librosa', 'soundfile', 'webrtcvad'],
    'Text': ['re', 'unicodedata', 'textdistance'],
    'ML': ['sklearn', 'datasets'],
    'Visualization': ['matplotlib', 'seaborn', 'plotly'],
    'Utilities': ['tqdm', 'yaml', 'json'],
}

OPTIONAL = {
    'Advanced Audio': ['pyannote.audio', 'pydub'],
    'Jupyter': ['IPython', 'ipywidgets'],
}

def check_dependency(package_name):
    """Check if a package can be imported."""
    try:
        import_module(package_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Kalenjin ASR Dependency Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    missing_required = []
    missing_optional = []
    
    # Check required dependencies
    print("Required Dependencies:")
    print("-" * 60)
    for category, packages in DEPENDENCIES.items():
        print(f"\n{category}:")
        for package in packages:
            success, error = check_dependency(package)
            status = "✓" if success else "✗"
            print(f"  {status} {package}")
            if not success:
                all_passed = False
                missing_required.append(package)
    
    # Check optional dependencies
    print("\n" + "=" * 60)
    print("Optional Dependencies:")
    print("-" * 60)
    for category, packages in OPTIONAL.items():
        print(f"\n{category}:")
        for package in packages:
            success, error = check_dependency(package)
            status = "✓" if success else "○"
            print(f"  {status} {package}")
            if not success:
                missing_optional.append(package)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if all_passed:
        print("✓ All required dependencies are installed!")
    else:
        print(f"✗ Missing {len(missing_required)} required package(s):")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print(f"\n○ {len(missing_optional)} optional package(s) not installed:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print(f"\nOptional install: pip install {' '.join(missing_optional)}")
    
    print("\n" + "=" * 60)
    
    # Version info for key packages
    if all_passed:
        print("\nKey Package Versions:")
        print("-" * 60)
        try:
            import numpy as np
            import pandas as pd
            import librosa
            import datasets
            print(f"NumPy:    {np.__version__}")
            print(f"Pandas:   {pd.__version__}")
            print(f"Librosa:  {librosa.__version__}")
            print(f"Datasets: {datasets.__version__}")
        except:
            pass
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
