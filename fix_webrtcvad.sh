#!/bin/bash
# Fix webrtcvad dependency issue

echo "Fixing webrtcvad in audio_ml environment..."

# Activate audio_ml and install setuptools
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio_ml

echo "Installing setuptools in audio_ml environment..."
pip install setuptools

echo "Reinstalling webrtcvad..."
pip uninstall -y webrtcvad
pip install webrtcvad

echo ""
echo "Testing webrtcvad..."
python -c "import webrtcvad; print('✓ webrtcvad working!')" && echo "SUCCESS" || echo "FAILED - will use fallback VAD"

echo ""
echo "Done!"
