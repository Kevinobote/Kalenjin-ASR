# ============================================================================
# Kalenjin ASR Preprocessing Pipeline - Dependencies
# ============================================================================
# This cell imports all required packages with graceful handling of optionals

import warnings
warnings.filterwarnings('ignore')

# Core scientific computing
import numpy as np
import pandas as pd
from pathlib import Path
import json
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

# Text processing & linguistics
import re
import unicodedata
from collections import Counter, defaultdict
import string

# Machine learning & data science
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import datasets
from datasets import Dataset, DatasetDict, Audio

# Visualization & analysis
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

# Optional dependencies with fallbacks
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    
try:
    from textdistance import levenshtein
    TEXTDISTANCE_AVAILABLE = True
except ImportError:
    TEXTDISTANCE_AVAILABLE = False
    
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import pyannote.audio
    from pyannote.audio import Pipeline as PyannoteePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

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
print('=' * 70)
print('KALENJIN ASR PREPROCESSING PIPELINE'.center(70))
print('=' * 70)
print(f'✓ NumPy:      {np.__version__}')
print(f'✓ Pandas:     {pd.__version__}')
print(f'✓ Librosa:    {librosa.__version__}')
print(f'✓ Datasets:   {datasets.__version__}')
print(f'✓ Matplotlib: {plt.matplotlib.__version__}')
print(f'✓ Seaborn:    {sns.__version__}')
print(f'✓ Plotly:     {px.__version__}')
print(f'✓ CPU Cores:  {cpu_count()}')
print('=' * 70)
print('Optional Features:')
print(f'  {"✓" if YAML_AVAILABLE else "✗"} YAML config support')
print(f'  {"✓" if TEXTDISTANCE_AVAILABLE else "✗"} Text distance metrics')
print(f'  {"✓" if PYDUB_AVAILABLE else "✗"} Audio format conversion (pydub)')
print(f'  {"✓" if PYANNOTE_AVAILABLE else "✗"} Advanced VAD (pyannote.audio)')
print('=' * 70)
print('✓ Environment ready for preprocessing!')
print('=' * 70)
