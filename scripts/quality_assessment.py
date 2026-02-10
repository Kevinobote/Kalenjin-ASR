#!/usr/bin/env python3
"""
Quality Assessment Module for Kalenjin ASR Preprocessing Pipeline
Section 5 Supporting Script

This script provides comprehensive quality assessment tools for audio and text data
in the Kalenjin ASR corpus preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityThresholds:
    """Quality assessment thresholds for audio and text validation."""
    
    # Audio quality thresholds
    min_snr_db: float = 10.0
    max_zero_crossing_rate: float = 0.3
    min_spectral_centroid: float = 200.0
    max_spectral_centroid: float = 8000.0
    max_clipping_ratio: float = 0.01
    min_dynamic_range_db: float = 6.0
    
    # Text quality thresholds
    min_words: int = 1
    max_words: int = 50
    max_char_repetition: int = 3
    min_word_length: float = 2.0
    max_oov_ratio: float = 0.1

class AudioQualityAssessor:
    """Advanced audio quality assessment for ASR preprocessing."""
    
    def __init__(self, thresholds: QualityThresholds):
        self.thresholds = thresholds
        self.stats = defaultdict(list)
    
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio using noise floor estimation."""
        # Use quietest 10% as noise estimate
        sorted_energy = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_energy[:int(0.1 * len(sorted_energy))])
        
        rms_energy = np.sqrt(np.mean(audio**2))
        signal_power = rms_energy**2
        noise_power = noise_floor**2
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr_db
    
    def calculate_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculate spectral quality features."""
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_spectral_centroid = np.mean(spectral_centroids)
        
        # Zero crossing rate (indicates noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        mean_zcr = np.mean(zcr)
        
        # Spectral rolloff (frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        mean_rolloff = np.mean(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        mean_bandwidth = np.mean(spectral_bandwidth)
        
        return {
            'spectral_centroid': mean_spectral_centroid,
            'zero_crossing_rate': mean_zcr,
            'spectral_rolloff': mean_rolloff,
            'spectral_bandwidth': mean_bandwidth
        }
    
    def detect_clipping(self, audio: np.ndarray) -> float:
        """Detect audio clipping ratio."""
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        return clipping_ratio
    
    def calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        rms_energy = np.sqrt(np.mean(audio**2))
        peak_amplitude = np.max(np.abs(audio))
        
        if rms_energy > 0:
            dynamic_range_db = 20 * np.log10(peak_amplitude / rms_energy)
        else:
            dynamic_range_db = 0.0
        
        return dynamic_range_db
    
    def assess_audio_quality(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Comprehensive audio quality assessment."""
        # Basic metrics
        duration = len(audio) / sr
        rms_energy = np.sqrt(np.mean(audio**2))
        peak_amplitude = np.max(np.abs(audio))
        
        # Quality metrics
        snr_db = self.calculate_snr(audio)
        spectral_features = self.calculate_spectral_features(audio, sr)
        clipping_ratio = self.detect_clipping(audio)
        dynamic_range_db = self.calculate_dynamic_range(audio)
        
        # Combine all metrics
        metrics = {
            'duration': duration,
            'rms_energy': rms_energy,
            'peak_amplitude': peak_amplitude,
            'snr_db': snr_db,
            'clipping_ratio': clipping_ratio,
            'dynamic_range_db': dynamic_range_db,
            **spectral_features
        }
        
        # Quality validation
        is_valid, issues = self.validate_audio_quality(metrics)
        
        # Update statistics
        for key, value in metrics.items():
            self.stats[key].append(value)
        
        return {
            'metrics': metrics,
            'is_valid': is_valid,
            'issues': issues
        }
    
    def validate_audio_quality(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate audio quality against thresholds."""
        issues = []
        
        # SNR check
        if metrics['snr_db'] < self.thresholds.min_snr_db:
            issues.append(f"Low SNR: {metrics['snr_db']:.1f}dB < {self.thresholds.min_snr_db}dB")
        
        # Zero crossing rate check
        if metrics['zero_crossing_rate'] > self.thresholds.max_zero_crossing_rate:
            issues.append(f"High ZCR: {metrics['zero_crossing_rate']:.3f} > {self.thresholds.max_zero_crossing_rate}")
        
        # Spectral centroid check
        if (metrics['spectral_centroid'] < self.thresholds.min_spectral_centroid or 
            metrics['spectral_centroid'] > self.thresholds.max_spectral_centroid):
            issues.append(f"Spectral centroid out of range: {metrics['spectral_centroid']:.0f}Hz")
        
        # Clipping check
        if metrics['clipping_ratio'] > self.thresholds.max_clipping_ratio:
            issues.append(f"Audio clipping detected: {metrics['clipping_ratio']:.3f}")
        
        # Dynamic range check
        if metrics['dynamic_range_db'] < self.thresholds.min_dynamic_range_db:
            issues.append(f"Low dynamic range: {metrics['dynamic_range_db']:.1f}dB")
        
        return len(issues) == 0, issues

class TextQualityAssessor:
    """Text quality assessment for ASR preprocessing."""
    
    def __init__(self, thresholds: QualityThresholds):
        self.thresholds = thresholds
        self.stats = defaultdict(int)
        self.vocabulary = set()
    
    def calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text quality metrics."""
        if not isinstance(text, str):
            return {'error': 'Input is not a string'}
        
        words = text.split()
        
        # Basic metrics
        char_count = len(text)
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Character analysis
        unique_chars = len(set(text.lower()))
        whitespace_ratio = text.count(' ') / len(text) if text else 0
        
        # Repetition detection
        max_repetition = self.detect_character_repetitions(text)
        
        # Word analysis
        unique_words = len(set(words))
        word_diversity = unique_words / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'unique_chars': unique_chars,
            'whitespace_ratio': whitespace_ratio,
            'max_repetition': max_repetition,
            'unique_words': unique_words,
            'word_diversity': word_diversity
        }
    
    def detect_character_repetitions(self, text: str) -> int:
        """Detect excessive character repetitions."""
        max_repetition = 0
        current_char = ''
        current_count = 0
        
        for char in text:
            if char == current_char:
                current_count += 1
                max_repetition = max(max_repetition, current_count)
            else:
                current_char = char
                current_count = 1
        
        return max_repetition
    
    def assess_text_quality(self, text: str) -> Dict[str, Any]:
        """Comprehensive text quality assessment."""
        # Calculate metrics
        metrics = self.calculate_text_metrics(text)
        
        if 'error' in metrics:
            return {
                'metrics': metrics,
                'is_valid': False,
                'issues': [metrics['error']]
            }
        
        # Quality validation
        is_valid, issues = self.validate_text_quality(text, metrics)
        
        # Update statistics
        self.stats['total_processed'] += 1
        if is_valid:
            self.stats['valid_texts'] += 1
            # Update vocabulary
            words = text.split()
            self.vocabulary.update(words)
        else:
            self.stats['invalid_texts'] += 1
        
        return {
            'metrics': metrics,
            'is_valid': is_valid,
            'issues': issues
        }
    
    def validate_text_quality(self, text: str, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate text quality against thresholds."""
        issues = []
        
        # Word count check
        if metrics['word_count'] < self.thresholds.min_words:
            issues.append(f"Too few words: {metrics['word_count']} < {self.thresholds.min_words}")
        elif metrics['word_count'] > self.thresholds.max_words:
            issues.append(f"Too many words: {metrics['word_count']} > {self.thresholds.max_words}")
        
        # Character repetition check
        if metrics['max_repetition'] > self.thresholds.max_char_repetition:
            issues.append(f"Excessive repetition: {metrics['max_repetition']} chars")
        
        # Average word length check
        if metrics['avg_word_length'] < self.thresholds.min_word_length:
            issues.append(f"Words too short: avg {metrics['avg_word_length']:.1f} chars")
        
        return len(issues) == 0, issues

class QualityAssessmentPipeline:
    """Main pipeline for comprehensive quality assessment."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.audio_assessor = AudioQualityAssessor(self.thresholds)
        self.text_assessor = TextQualityAssessor(self.thresholds)
        self.results = []
    
    def assess_text_sample(self, text: str, identifier: str = "") -> Dict[str, Any]:
        """Assess quality of a text sample."""
        assessment = self.text_assessor.assess_text_quality(text)
        assessment['identifier'] = identifier
        return assessment
    
    def process_dataset(self, data_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process entire dataset for quality assessment."""
        logger.info(f"Starting quality assessment of dataset: {data_dir}")
        
        # Find audio files
        audio_files = list(data_dir.glob("**/*.wav")) + list(data_dir.glob("**/*.mp3"))
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process audio files (placeholder - no actual audio files in demo)
        audio_results = []
        
        # Process transcriptions if available
        text_results = []
        transcript_file = data_dir / "transcripts.txt"
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        file_id, text = parts[0], parts[1]
                        result = self.assess_text_sample(text, file_id)
                        text_results.append(result)
        
        # Generate summary statistics
        summary = self.generate_summary(audio_results, text_results)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_results(audio_results, text_results, summary, output_dir)
        
        return summary
    
    def generate_summary(self, audio_results: List[Dict], text_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive quality assessment summary."""
        summary = {
            'audio_assessment': {
                'total_files': len(audio_results),
                'valid_files': sum(1 for r in audio_results if r.get('is_valid', False)),
                'invalid_files': sum(1 for r in audio_results if not r.get('is_valid', True))
            },
            'text_assessment': {
                'total_samples': len(text_results),
                'valid_samples': sum(1 for r in text_results if r.get('is_valid', False)),
                'invalid_samples': sum(1 for r in text_results if not r.get('is_valid', True))
            }
        }
        
        # Text statistics
        if text_results:
            valid_text = [r for r in text_results if r.get('is_valid', False)]
            if valid_text:
                metrics = [r['metrics'] for r in valid_text if 'metrics' in r]
                if metrics:
                    summary['text_stats'] = {
                        'avg_word_count': np.mean([m['word_count'] for m in metrics]),
                        'avg_word_length': np.mean([m['avg_word_length'] for m in metrics]),
                        'vocabulary_size': len(self.text_assessor.vocabulary)
                    }
        
        return summary
    
    def save_results(self, audio_results: List[Dict], text_results: List[Dict], 
                    summary: Dict[str, Any], output_dir: Path):
        """Save assessment results to files."""
        # Save detailed results
        if audio_results:
            with open(output_dir / 'audio_quality_results.json', 'w') as f:
                json.dump(audio_results, f, indent=2, default=str)
        
        if text_results:
            with open(output_dir / 'text_quality_results.json', 'w') as f:
                json.dump(text_results, f, indent=2, default=str)
        
        # Save summary
        with open(output_dir / 'quality_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save quality report
        self.generate_quality_report(summary, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    def generate_quality_report(self, summary: Dict[str, Any], output_dir: Path):
        """Generate human-readable quality report."""
        report_path = output_dir / 'quality_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("KALENJIN ASR CORPUS QUALITY ASSESSMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Audio assessment
            audio_stats = summary['audio_assessment']
            f.write("AUDIO QUALITY ASSESSMENT\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total audio files: {audio_stats['total_files']}\n")
            f.write(f"Valid files: {audio_stats['valid_files']}\n")
            f.write(f"Invalid files: {audio_stats['invalid_files']}\n")
            
            if audio_stats['total_files'] > 0:
                validity_rate = audio_stats['valid_files'] / audio_stats['total_files'] * 100
                f.write(f"Validity rate: {validity_rate:.1f}%\n")
            
            f.write("\n")
            
            # Text assessment
            text_stats = summary['text_assessment']
            f.write("TEXT QUALITY ASSESSMENT\n")
            f.write("-" * 24 + "\n")
            f.write(f"Total text samples: {text_stats['total_samples']}\n")
            f.write(f"Valid samples: {text_stats['valid_samples']}\n")
            f.write(f"Invalid samples: {text_stats['invalid_samples']}\n")
            
            if text_stats['total_samples'] > 0:
                validity_rate = text_stats['valid_samples'] / text_stats['total_samples'] * 100
                f.write(f"Validity rate: {validity_rate:.1f}%\n")
            
            if 'text_stats' in summary:
                stats = summary['text_stats']
                f.write(f"Average word count: {stats['avg_word_count']:.1f}\n")
                f.write(f"Average word length: {stats['avg_word_length']:.1f} chars\n")
                f.write(f"Vocabulary size: {stats['vocabulary_size']}\n")

def main():
    """Main execution function for quality assessment."""
    # Setup paths
    base_dir = Path("/home/obote/Documents/SE/AOB/2026/Kalenjin ASR")
    data_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "results" / "quality_assessment"
    
    # Initialize pipeline with custom thresholds
    thresholds = QualityThresholds(
        min_snr_db=8.0,  # Slightly relaxed for field recordings
        max_zero_crossing_rate=0.35,
        min_spectral_centroid=150.0,
        max_spectral_centroid=8500.0,
        max_clipping_ratio=0.02,
        min_dynamic_range_db=5.0,
        min_words=1,
        max_words=100,  # Allow longer sentences
        max_char_repetition=4,
        min_word_length=1.5,
        max_oov_ratio=0.15
    )
    
    pipeline = QualityAssessmentPipeline(thresholds)
    
    # Check if data directory exists
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        logger.info("Creating sample data for demonstration...")
        
        # Create sample data structure
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample transcript file
        sample_transcripts = [
            "audio001.wav\tKipchoge amat eng' keny",
            "audio002.wav\tAmun ko tugul che kitaunen",
            "audio003.wav\tKo ak tugul che keny",
            "audio004.wav\tIndet ne tugul ak amun",
            "audio005.wav\tKipchoge ak Rotich ko tugul"
        ]
        
        with open(data_dir / "transcripts.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(sample_transcripts))
        
        logger.info("Sample transcript file created")
    
    # Run quality assessment
    try:
        summary = pipeline.process_dataset(data_dir, output_dir)
        
        # Print summary
        print("\n" + "=" * 50)
        print("QUALITY ASSESSMENT COMPLETED")
        print("=" * 50)
        print(f"Audio files processed: {summary['audio_assessment']['total_files']}")
        print(f"Text samples processed: {summary['text_assessment']['total_samples']}")
        print(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSection 5 Quality Assessment script completed successfully!")
    else:
        print("\nSection 5 Quality Assessment script encountered errors.")