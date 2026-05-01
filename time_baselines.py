#!/usr/bin/env python3
"""
Timing script: runs Q1-Q10 for multiple baselines across text and tabular datasets.
Reports per-question wall-clock time and writes CSV summary.
"""
import subprocess
import sys
import time
import os
import csv
from datetime import datetime

VLM = "tinker/Qwen3-VL-30B-A3B-Instruct"

# Datasets to test: (name, dataset_pattern, model_path)
DATASETS = [
    # Text
    ("imdb_cnn", "dataset/test/text/imdb_cnn_q{q}.json", "models_to_read/text/imdb_cnn.pth"),
    ("adult_tabnn", "dataset/test/tabular/adult_tabnn_q{q}.json", "models_to_read/tabular/adult_tabnn.pth"),
]

BASELINES = ["cot", "react"]
Q_TYPES = list(range(1, 11))

all_results = []  # (baseline, dataset_name, q, status, elapsed)

for baseline in BASELINES:
    for ds_name, ds_pattern, model_path in DATASETS:
        print(f"\n{'#'*70}")
        print(f"# {baseline.upper()} | {ds_name}")
        print(f"{'#'*70}")

        for q in Q_TYPES:
            dataset = ds_pattern.format(q=q)
            if not os.path.exists(dataset):
                all_results.append((baseline, ds_name, q, "SKIP", 0))
                continue

            cmd = [
                sys.executable, "run_baseline_agent.py",
                "--dataset", dataset,
                "--question_id", "0",
                "--model_url", model_path,
                "--vlm", VLM,
                "--no-improvement", "--no-sf",
            ]
            env = os.environ.copy()
            env["BASELINE_TYPE"] = baseline

            print(f"  {baseline}/{ds_name}/Q{q} ... ", end="", flush=True)

            start = time.time()
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
                elapsed = time.time() - start
                success = "PIPELINE COMPLETE" in proc.stdout
                status = "OK" if success else "FAIL"
            except subprocess.TimeoutExpired:
                elapsed = 300.0
                status = "TIMEOUT"

            all_results.append((baseline, ds_name, q, status, elapsed))
            print(f"{status} {elapsed:.1f}s")

            if status == "FAIL":
                lines = (proc.stdout + proc.stderr).strip().split('\n')
                for line in lines[-5:]:
                    print(f"    | {line}")

# Write CSV
csv_path = f"timing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["baseline", "dataset", "q_type", "status", "time_s"])
    for row in all_results:
        w.writerow(row)

# Print summary table
print(f"\n\n{'='*80}")
print(f"FULL TIMING SUMMARY")
print(f"{'='*80}")
print(f"{'Baseline':<8} {'Dataset':<18} {'Q#':<5} {'Status':<8} {'Time(s)':<8}")
print(f"{'-'*50}")
for baseline, ds_name, q, status, elapsed in all_results:
    if status != "SKIP":
        print(f"{baseline:<8} {ds_name:<18} Q{q:<4} {status:<8} {elapsed:<8.1f}")

# Per-baseline/dataset averages
print(f"\n{'='*80}")
print(f"AVERAGES PER BASELINE × DATASET")
print(f"{'='*80}")
print(f"{'Baseline':<8} {'Dataset':<18} {'OK/Total':<10} {'Avg(s)':<8} {'Total(s)':<10}")
print(f"{'-'*55}")
for baseline in BASELINES:
    for ds_name, _, _ in DATASETS:
        rows = [(s, t) for b, d, q, s, t in all_results if b == baseline and d == ds_name and s != "SKIP"]
        if not rows:
            continue
        ok = sum(1 for s, _ in rows if s == "OK")
        total_t = sum(t for _, t in rows)
        avg_t = total_t / len(rows) if rows else 0
        print(f"{baseline:<8} {ds_name:<18} {ok}/{len(rows):<7} {avg_t:<8.1f} {total_t:<10.1f}")

print(f"\nCSV saved to: {csv_path}")
