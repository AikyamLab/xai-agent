
import os
import json
import random
from pathlib import Path

SAMPLES_PER_FILE = 100  # double the original 50
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def split_dataset_files(
    source_base: str = '/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/dataset_full',
    output_base: str = '/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/dataset_2x',
    n_samples: int = SAMPLES_PER_FILE,
    seed: int = RANDOM_SEED,
):
    """
    Samples n_samples records from each file in source_base and splits them
    into train/test sets (80/20) under output_base.

    Originally used to create dataset/ (50 samples/file) from dataset_full/.
    Default parameters create dataset_2x/ with 100 samples/file.
    """
    random.seed(seed)

    source_base = Path(source_base)
    output_base = Path(output_base)

    modalities = ['vision', 'tabular', 'text']

    for modality in modalities:
        source_dir = source_base / modality
        train_dir = output_base / 'train' / modality
        test_dir = output_base / 'test' / modality

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            print(f"Source directory not found: {source_dir}")
            continue

        print(f"Processing modality: {modality}")

        for filepath in sorted(source_dir.glob('*.json')):
            print(f"  Sampling file: {filepath.name}")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    print(f"    Skipping {filepath.name}: not a JSON list.")
                    continue

                sampled = data[:]
                random.shuffle(sampled)

                # 80/20 train/test split
                split_index = int(len(sampled) * TRAIN_RATIO)
                train_data = sampled[:split_index]
                test_data = sampled[split_index:]

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
