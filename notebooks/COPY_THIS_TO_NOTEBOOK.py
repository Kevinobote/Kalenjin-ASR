"""
Kalenjin ASR Preprocessing Pipeline - Production Imports
Copy this entire cell into your notebook
"""

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE DEPENDENCIES
# ============================================================================
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Audio processing
import librosa
import librosa.display
import soundfile as sf
import scipy.signal
from scipy.stats import zscore, iqr

# Text processing
import re
import unicodedata
from collections import Counter, defaultdict
import string

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import datasets
from datasets import Dataset, DatasetDict, Audio

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parallel processing
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.auto import tqdm
tqdm.pandas()

# ============================================================================
# OPTIONAL DEPENDENCIES (with fallbacks)
# ============================================================================
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except:
    WEBRTC_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except:
    YAML_AVAILABLE = False

try:
    from textdistance import levenshtein
    TEXTDISTANCE_AVAILABLE = True
except:
    TEXTDISTANCE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except:
    PYDUB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FALLBACK IMPLEMENTATIONS
# ============================================================================
class SimpleVAD:
    """Librosa-based VAD fallback."""
    def __init__(self, aggressiveness=2):
        self.top_db = [15, 20, 25, 30][aggressiveness]
    
    def trim_silence(self, audio, sr=16000):
        trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db)
        return trimmed

if not WEBRTC_AVAILABLE:
    webrtcvad = type('webrtcvad', (), {'Vad': SimpleVAD})

# ============================================================================
# ENVIRONMENT SUMMARY
# ============================================================================
print('=' * 70)
print('KALENJIN ASR PREPROCESSING PIPELINE'.center(70))
print('=' * 70)
print(f'\n✓ NumPy:      {np.__version__}')
print(f'✓ Pandas:     {pd.__version__}')
print(f'✓ Librosa:    {librosa.__version__}')
print(f'✓ Datasets:   {datasets.__version__}')
print(f'✓ Matplotlib: {plt.matplotlib.__version__}')
print(f'✓ CPU Cores:  {cpu_count()}')
print(f'\nVAD Method:   {"WebRTC" if WEBRTC_AVAILABLE else "Librosa (fallback)"}')
print('=' * 70)
print('✓ Ready for preprocessing!\n')
