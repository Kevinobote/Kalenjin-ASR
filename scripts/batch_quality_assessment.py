#!/usr/bin/env python3
"""
Batch Quality Assessment Script for Kalenjin ASR Dataset
Supporting script for Section 5 - Quality Assessment Module

This script processes the entire Kalenjin ASR dataset in batches to assess
audio and text quality efficiently.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from quality_assessment import QualityAssessmentPipeline, QualityThresholds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_quality_assessment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchQualityProcessor:
    """Batch processor for quality assessment of large datasets."""
    
    def __init__(self, data_path: str, output_dir: str, n_jobs: int = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_jobs = n_jobs or min(cpu_count() - 1, 8)
        
        # Initialize quality thresholds
        self.thresholds = QualityThresholds()
        
        logger.info(f"Initialized BatchQualityProcessor with {self.n_jobs} workers")
    
    def load_dataset_splits(self) -> Dict[str, pd.DataFrame]:
        """Load all dataset splits."""
        splits = {}
        
        for split_name in ['train', 'dev', 'test', 'validated', 'invalidated', 'other']:
            split_file = self.data_path / f"{split_name}.tsv"
            if split_file.exists():
                splits[split_name] = pd.read_csv(split_file, sep='\t')
                logger.info(f"Loaded {split_name}: {len(splits[split_name])} samples")
            else:
                logger.warning(f"Split file not found: {split_file}")
        
        return splits
    
    def process_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """Process a batch of samples for quality assessment."""
        pipeline = QualityAssessmentPipeline(self.thresholds)
        results = []
        
        for sample in batch_data:
            try:
                result = pipeline.assess_sample(
                    sample['audio_path'],
                    sample['text'],
                    sample['sample_id']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {sample['sample_id']}: {e}")
                results.append({
                    'sample_id': sample['sample_id'],
                    'audio_path': sample['audio_path'],
                    'text': sample['text'],
                    'audio_quality': {'is_valid': False, 'issues': [f'Processing error: {str(e)}']},
                    'text_quality': {'is_valid': False, 'issues': ['Processing failed']},
                    'overall_valid': False
                })
        
        return results
    
    def create_batches(self, df: pd.DataFrame, batch_size: int = 100) -> List[List[Dict]]:
        """Create batches from dataframe."""
        clips_path = self.data_path / "clips"
        batches = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_data = []
            
            for idx, row in batch_df.iterrows():
                audio_path = clips_path / row['path']
                if audio_path.exists():
                    batch_data.append({
                        'sample_id': f"{idx}_{row['path']}",
                        'audio_path': str(audio_path),
                        'text': row['sentence']
                    })
            
            if batch_data:
                batches.append(batch_data)
        
        return batches
    
    def process_split_parallel(self, split_name: str, df: pd.DataFrame, batch_size: int = 100) -> Dict[str, Any]:
        """Process a dataset split in parallel."""
        logger.info(f"Processing {split_name} split: {len(df)} samples")
        
        # Create batches
        batches = self.create_batches(df, batch_size)
        logger.info(f"Created {len(batches)} batches for {split_name}")
        
        # Process batches in parallel
        all_results = []
        
        if self.n_jobs == 1:
            # Sequential processing for debugging
            for batch in tqdm(batches, desc=f"Processing {split_name}"):
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)
        else:
            # Parallel processing
            with Pool(self.n_jobs) as pool:
                batch_results = list(tqdm(
                    pool.imap(self.process_batch, batches),
                    total=len(batches),
                    desc=f"Processing {split_name}"
                ))
                
                for results in batch_results:
                    all_results.extend(results)
        
        # Generate summary statistics
        summary = self.generate_split_summary(all_results)
        
        # Save results
        output_file = self.output_dir / f"{split_name}_quality_assessment.json"
        self.save_split_results(split_name, all_results, summary, output_file)
        
        logger.info(f"Completed {split_name}: {len(all_results)} samples processed")
        return summary
    
    def generate_split_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for a split."""
        if not results:
            return {'error': 'No results to summarize'}
        
        total_samples = len(results)
        valid_audio = sum(1 for r in results if r['audio_quality']['is_valid'])
        valid_text = sum(1 for r in results if r['text_quality']['is_valid'])
        overall_valid = sum(1 for r in results if r['overall_valid'])
        
        # Collect audio metrics
        audio_metrics = {}
        for metric in ['snr_db', 'spectral_centroid', 'zero_crossing_rate', 'dynamic_range_db']:
            values = []
            for r in results:
                if r['audio_quality']['is_valid'] and 'metrics' in r['audio_quality']:
                    if metric in r['audio_quality']['metrics']:
                        values.append(r['audio_quality']['metrics'][metric])
            
            if values:
                audio_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Collect text metrics
        text_metrics = {}
        for metric in ['word_count', 'avg_word_length', 'word_diversity']:
            values = []
            for r in results:
                if r['text_quality']['is_valid'] and 'metrics' in r['text_quality']:
                    if metric in r['text_quality']['metrics']:
                        values.append(r['text_quality']['metrics'][metric])
            
            if values:
                text_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return {
            'total_samples': total_samples,
            'valid_audio': valid_audio,
            'valid_text': valid_text,
            'overall_valid': overall_valid,
            'audio_validity_rate': valid_audio / total_samples if total_samples > 0 else 0,
            'text_validity_rate': valid_text / total_samples if total_samples > 0 else 0,
            'overall_validity_rate': overall_valid / total_samples if total_samples > 0 else 0,
            'audio_metrics': audio_metrics,
            'text_metrics': text_metrics
        }
    
    def save_split_results(self, split_name: str, results: List[Dict], summary: Dict[str, Any], output_file: Path):
        """Save results for a split."""
        output_data = {
            'split_name': split_name,
            'summary': summary,
            'thresholds': {
                'min_snr_db': self.thresholds.min_snr_db,
                'max_zero_crossing_rate': self.thresholds.max_zero_crossing_rate,
                'min_spectral_centroid': self.thresholds.min_spectral_centroid,
                'max_spectral_centroid': self.thresholds.max_spectral_centroid,
                'max_clipping_ratio': self.thresholds.max_clipping_ratio,
                'min_dynamic_range_db': self.thresholds.min_dynamic_range_db,
                'min_words': self.thresholds.min_words,
                'max_words': self.thresholds.max_words,
                'max_char_repetition': self.thresholds.max_char_repetition,
                'min_word_length': self.thresholds.min_word_length
            },
            'detailed_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
    
    def process_all_splits(self, batch_size: int = 100) -> Dict[str, Any]:
        """Process all dataset splits."""
        logger.info("Starting batch quality assessment for all splits")
        
        # Load all splits
        splits = self.load_dataset_splits()
        
        if not splits:
            logger.error("No dataset splits found")
            return {}
        
        # Process each split
        all_summaries = {}
        
        for split_name, df in splits.items():
            try:
                summary = self.process_split_parallel(split_name, df, batch_size)
                all_summaries[split_name] = summary
            except Exception as e:
                logger.error(f"Error processing {split_name}: {e}")
                all_summaries[split_name] = {'error': str(e)}
        
        # Generate overall summary
        overall_summary = self.generate_overall_summary(all_summaries)
        
        # Save overall summary
        summary_file = self.output_dir / "overall_quality_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        logger.info(f"Overall summary saved to {summary_file}")
        return overall_summary
    
    def generate_overall_summary(self, split_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary across all splits."""
        total_samples = 0
        total_valid_audio = 0
        total_valid_text = 0
        total_overall_valid = 0
        
        for split_name, summary in split_summaries.items():
            if 'error' not in summary:
                total_samples += summary['total_samples']
                total_valid_audio += summary['valid_audio']
                total_valid_text += summary['valid_text']
                total_overall_valid += summary['overall_valid']
        
        return {
            'dataset_summary': {
                'total_samples': total_samples,
                'total_valid_audio': total_valid_audio,
                'total_valid_text': total_valid_text,
                'total_overall_valid': total_overall_valid,
                'overall_audio_validity_rate': total_valid_audio / total_samples if total_samples > 0 else 0,
                'overall_text_validity_rate': total_valid_text / total_samples if total_samples > 0 else 0,
                'overall_validity_rate': total_overall_valid / total_samples if total_samples > 0 else 0
            },
            'split_summaries': split_summaries
        }

def main():
    """Main function for batch quality assessment."""
    parser = argparse.ArgumentParser(description="Batch Quality Assessment for Kalenjin ASR Dataset")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the Kalenjin dataset directory")
    parser.add_argument("--output_dir", type=str, default="quality_assessment_results",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for processing")
    parser.add_argument("--n_jobs", type=int, default=None,
                       help="Number of parallel jobs")
    parser.add_argument("--splits", nargs='+', default=None,
                       help="Specific splits to process (default: all)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchQualityProcessor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs
    )
    
    if args.splits:
        # Process specific splits
        splits = processor.load_dataset_splits()
        for split_name in args.splits:
            if split_name in splits:
                processor.process_split_parallel(split_name, splits[split_name], args.batch_size)
            else:
                logger.error(f"Split '{split_name}' not found")
    else:
        # Process all splits
        overall_summary = processor.process_all_splits(args.batch_size)
        
        # Print summary
        print("\n" + "="*60)
        print("QUALITY ASSESSMENT SUMMARY")
        print("="*60)
        
        if 'dataset_summary' in overall_summary:
            ds = overall_summary['dataset_summary']
            print(f"Total samples processed: {ds['total_samples']:,}")
            print(f"Overall validity rate: {ds['overall_validity_rate']:.2%}")
            print(f"Audio validity rate: {ds['overall_audio_validity_rate']:.2%}")
            print(f"Text validity rate: {ds['overall_text_validity_rate']:.2%}")
            
            print("\nSplit-wise Summary:")
            for split_name, summary in overall_summary['split_summaries'].items():
                if 'error' not in summary:
                    print(f"  {split_name:12}: {summary['total_samples']:>6,} samples, "
                          f"{summary['overall_validity_rate']:>6.1%} valid")

if __name__ == "__main__":
    main()