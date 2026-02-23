"""
Compute average faithfulness scores per q_type per dataset,
both before (original) and after (improved) reflection.

Data source: outputs/training_data/<modality>/<dataset>/<q_type>/<row_no>/training_datapoint.json
"""

import json
import os
from collections import defaultdict
from pathlib import Path

TRAINING_DATA_ROOT = Path("outputs/training_data")

# {(dataset, q_type): [original_scores]}
original_scores = defaultdict(list)
improved_scores = defaultdict(list)
counts = defaultdict(int)


def collect(root: Path):
    # root / modality / dataset / q_type / row_no / training_datapoint.json
    for modality_dir in sorted(root.iterdir()):
        if not modality_dir.is_dir():
            continue
        for dataset_dir in sorted(modality_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for q_dir in sorted(dataset_dir.iterdir()):
                if not q_dir.is_dir():
                    continue
                q_type = q_dir.name  # e.g. "q1"
                for row_dir in q_dir.iterdir():
                    if not row_dir.is_dir():
                        continue
                    dp_path = row_dir / "training_datapoint.json"
                    if not dp_path.exists():
                        continue
                    try:
                        with open(dp_path) as f:
                            dp = json.load(f)
                        metrics = dp.get("improvement_metrics", {})
                        orig = metrics.get("original_faithfulness")
                        impr = metrics.get("improved_faithfulness")
                        if orig is not None and impr is not None:
                            key = (dataset, q_type)
                            original_scores[key].append(orig)
                            improved_scores[key].append(impr)
                            counts[key] += 1
                    except Exception as e:
                        print(f"  [skip] {dp_path}: {e}")


collect(TRAINING_DATA_ROOT)

# ── Print results ──────────────────────────────────────────────────────────────

all_datasets = sorted({k[0] for k in counts})
all_qtypes   = sorted({k[1] for k in counts}, key=lambda x: int(x[1:]))

# Per dataset × q_type table
print(f"{'Dataset':<22}  {'Q':<5}  {'N':>5}  {'Orig avg':>10}  {'Impr avg':>10}  {'Delta':>8}")
print("-" * 68)

dataset_totals = defaultdict(lambda: {"orig": [], "impr": []})

for dataset in all_datasets:
    for q_type in all_qtypes:
        key = (dataset, q_type)
        if key not in counts:
            continue
        n     = counts[key]
        orig  = sum(original_scores[key]) / n
        impr  = sum(improved_scores[key]) / n
        delta = impr - orig
        print(f"{dataset:<22}  {q_type:<5}  {n:>5}  {orig:>10.4f}  {impr:>10.4f}  {delta:>+8.4f}")
        dataset_totals[dataset]["orig"].extend(original_scores[key])
        dataset_totals[dataset]["impr"].extend(improved_scores[key])

# Per-dataset summary
print()
print(f"{'Dataset (all q)':<22}  {'N':>5}  {'Orig avg':>10}  {'Impr avg':>10}  {'Delta':>8}")
print("-" * 60)
all_orig_list, all_impr_list = [], []
for dataset in all_datasets:
    o_list = dataset_totals[dataset]["orig"]
    i_list = dataset_totals[dataset]["impr"]
    if not o_list:
        continue
    n    = len(o_list)
    orig = sum(o_list) / n
    impr = sum(i_list) / n
    print(f"{dataset:<22}  {n:>5}  {orig:>10.4f}  {impr:>10.4f}  {impr-orig:>+8.4f}")
    all_orig_list.extend(o_list)
    all_impr_list.extend(i_list)

# Grand total
if all_orig_list:
    n    = len(all_orig_list)
    orig = sum(all_orig_list) / n
    impr = sum(all_impr_list) / n
    print("-" * 60)
    print(f"{'OVERALL':<22}  {n:>5}  {orig:>10.4f}  {impr:>10.4f}  {impr-orig:>+8.4f}")
