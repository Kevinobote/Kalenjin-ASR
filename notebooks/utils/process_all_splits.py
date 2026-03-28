"""
Process all dataset splits and save results to JSON
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Helper function to convert numpy types to JSON-serializable types
def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

# Assuming pipeline and DATA_PATH are already defined in your notebook
# Add this code to a new cell:

"""
# Create output directory
Path('../processed_data').mkdir(parents=True, exist_ok=True)

# Process all splits
for split in ['train', 'dev', 'test']:
    print(f"\nProcessing {split} split...")
    
    df = pd.read_csv(DATA_PATH / f'{split}.tsv', sep='\t')
    pairs = [(str(DATA_PATH / 'clips' / row['path']), row['sentence']) 
             for _, row in df.iterrows() 
             if (DATA_PATH / 'clips' / row['path']).exists()]
    
    results = pipeline.run_pipeline(pairs)
    
    # Remove audio arrays and convert numpy types
    results_to_save = {
        'total_samples': results['total_samples'],
        'valid_samples': results['valid_samples'],
        'success_rate': results['success_rate'],
        'metadata': [
            {k: v for k, v in item.items() if k != 'audio'}
            for item in results['processed_data']
        ]
    }
    
    # Convert numpy types to native Python types
    results_to_save = convert_to_json_serializable(results_to_save)
    
    # Save results
    with open(f'../processed_data/{split}_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"{split}: {results['success_rate']:.1%} success ({results['valid_samples']}/{results['total_samples']})")

print("\n✓ All splits processed and saved successfully!")
"""
