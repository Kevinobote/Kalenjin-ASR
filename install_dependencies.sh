#!/bin/bash
# Quick dependency installer for existing environment

echo "Installing dependencies in current environment..."

# Core packages
pip install -q matplotlib seaborn plotly
pip install -q webrtcvad pydub
pip install -q textdistance unidecode
pip install -q datasets
pip install -q pyyaml
pip install -q ipywidgets

echo "✓ Installation complete!"
echo ""
echo "Verify with: python verify_env.py"
