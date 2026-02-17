
import os
import json
import random
from pathlib import Path

def split_dataset_files():
    """
    Splits all dataset files in the 'dataset' directory into train and test sets.
    """
    base_dir = Path('/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/dataset')
    
    modalities = ['vision', 'tabular', 'text']
    
    for modality in modalities:
        source_dir = base_dir / modality
        train_dir = base_dir / 'train' / modality
        test_dir = base_dir / 'test' / modality
        
        # Ensure destination directories exist
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_dir.exists():
            print(f"Source directory not found: {source_dir}")
            continue
            
        print(f"Processing modality: {modality}")
        
        for filepath in source_dir.glob('*.json'):
            print(f"  Splitting file: {filepath.name}")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    print(f"    Skipping {filepath.name}: not a JSON list.")
                    continue
                
                # Shuffle the data
                random.shuffle(data)
                
                # Split the data
                split_index = int(len(data) * 0.8)
                train_data = data[:split_index]
                test_data = data[split_index:]
                
                # Write the new files
                train_filepath = train_dir / filepath.name
                test_filepath = test_dir / filepath.name
                
                with open(train_filepath, 'w') as f:
                    json.dump(train_data, f, indent=2)
                
                with open(test_filepath, 'w') as f:
                    json.dump(test_data, f, indent=2)
                    
                print(f"    Created {train_filepath} ({len(train_data)} records)")
                print(f"    Created {test_filepath} ({len(test_data)} records)")

            except Exception as e:
                print(f"    Error processing {filepath.name}: {e}")

if __name__ == '__main__':
    split_dataset_files()
