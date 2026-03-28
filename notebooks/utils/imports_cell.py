# Kalenjin ASR Preprocessing Pipeline - Dependency Imports
# This cell handles all imports with graceful fallbacks for optional dependencies

import warnings
warnings.filterwarnings('ignore')

# Core scientific computing
import numpy as np
import pandas as pd
from pathlib import Path
import json
try:
    import yaml
except ImportError:
    print("⚠ yaml not available - using json for config")
    yaml = None

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Audio processing & signal analysis
import librosa
import librosa.display
import soundfile as sf
import scipy.signal
from scipy.stats import zscore, iqr
import webrtcvad

# Optional: pyannote.audio (for advanced VAD)
try:
    import pyannote.audio
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    print("⚠ pyannote.audio not available - using basic VAD only")
    PYANNOTE_AVAILABLE = False

# Optional: pydub (for audio format conversion)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("⚠ pydub not available - limited format support")
    PYDUB_AVAILABLE = False

# Text processing & linguistics
import re
import unicodedata
from collections import Counter, defaultdict
import string
try:
    from textdistance import levenshtein
    TEXTDISTANCE_AVAILABLE = True
except ImportError:
    print("⚠ textdistance not available - using basic text processing")
    TEXTDISTANCE_AVAILABLE = False

# Machine learning & data science
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import datasets
from datasets import Dataset, DatasetDict, Audio

# Visualization & analysis
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠ plotly not available - using matplotlib only")
    PLOTLY_AVAILABLE = False

# Parallel processing
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.auto import tqdm
tqdm.pandas()

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Print summary
print('=' * 60)
print('✓ All core dependencies loaded successfully')
print('=' * 60)
print(f'✓ Available CPU cores: {cpu_count()}')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ Pandas version: {pd.__version__}')
print(f'✓ Librosa version: {librosa.__version__}')
print(f'✓ Datasets version: {datasets.__version__}')
print(f'✓ Matplotlib version: {plt.matplotlib.__version__}')
print('=' * 60)
print('Optional Features:')
print(f'  {"✓" if PYANNOTE_AVAILABLE else "✗"} Advanced VAD (pyannote.audio)')
print(f'  {"✓" if PYDUB_AVAILABLE else "✗"} Format conversion (pydub)')
print(f'  {"✓" if TEXTDISTANCE_AVAILABLE else "✗"} Text distance metrics')
print(f'  {"✓" if PLOTLY_AVAILABLE else "✗"} Interactive plots (plotly)')
print('=' * 60)
