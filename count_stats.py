import os
import json
from collections import defaultdict

def count_distribution(root_dir):
    # Overall stats
    stats = {
        "total": 0,
        "passed_first_try": 0,
        "improvement_attempted": 0,
        "actually_improved": 0,
        "worse_after_improvement": 0,
        "same_after_improvement": 0,
        "other": 0
    }
    
    # Nested stats: stats_by_group[dataset][q_id]
    stats_by_group = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "passed_first_try": 0,
        "improvement_attempted": 0,
        "actually_improved": 0,
        "worse_after_improvement": 0,
        "same_after_improvement": 0
    }))

    for root, dirs, files in os.walk(root_dir):
        if "training_datapoint.json" in files:
            file_path = os.path.join(root, "training_datapoint.json")
            
            # Path format: outputs/training_data/{modality}/{dataset}/{q_id}/{row_no}/training_datapoint.json
            parts = root.split(os.sep)
            if len(parts) < 4: continue
            
            # Finding the index of 'training_data' to be robust
            try:
                base_idx = parts.index("training_data")
                modality = parts[base_idx + 1]
                dataset = parts[base_idx + 2]
                q_id = parts[base_idx + 3]
            except (ValueError, IndexError):
                continue

            stats["total"] += 1
            stats_by_group[dataset][q_id]["total"] += 1
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                original = data.get("original", {})
                orig_faithfulness = original.get("explanation_faithfulness", {})
                orig_passed = orig_faithfulness.get("passed", False)
                
                has_improvement = "improvement_metrics" in data or "improved" in data
                
                if orig_passed and not has_improvement:
                    stats["passed_first_try"] += 1
                    stats_by_group[dataset][q_id]["passed_first_try"] += 1
                
                if has_improvement:
                    stats["improvement_attempted"] += 1
                    stats_by_group[dataset][q_id]["improvement_attempted"] += 1
                    
                    metrics = data.get("improvement_metrics", {})
                    delta = metrics.get("faithfulness_delta", 0)
                    
                    if delta > 0:
                        stats["actually_improved"] += 1
                        stats_by_group[dataset][q_id]["actually_improved"] += 1
                    elif delta < 0:
                        stats["worse_after_improvement"] += 1
                        stats_by_group[dataset][q_id]["worse_after_improvement"] += 1
                    else:
                        stats["same_after_improvement"] += 1
                        stats_by_group[dataset][q_id]["same_after_improvement"] += 1
                elif not orig_passed:
                    stats["other"] += 1
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return stats, stats_by_group

if __name__ == "__main__":
    root_path = "outputs/training_data"
    overall, grouped = count_distribution(root_path)
    
    print("=== Overall Statistics ===")
    print(json.dumps(overall, indent=2))
    
    print("\n=== Statistics by Dataset and Question ===")
    # Sort for better readability
    for ds in sorted(grouped.keys()):
        print(f"\nDataset: {ds}")
        for q in sorted(grouped[ds].keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
            s = grouped[ds][q]
            print(f"  {q}: Total: {s['total']}, Passed1st: {s['passed_first_try']}, Improved: {s['actually_improved']}, Worse: {s['worse_after_improvement']}, Same: {s['same_after_improvement']}")
