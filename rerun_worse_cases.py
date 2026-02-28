import os
import json
import random
import subprocess
import shutil
from pathlib import Path

# Config
BASE_DIR = Path("/sfs/ceph/standard/AikyamLab/yuyang/xai_agent/framework/trial_2")
TRAINING_DATA_DIR = BASE_DIR / "outputs" / "training_data"
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_outputs"
VLM_WRAPPER_PATH = BASE_DIR / "vlm_wrapper.py"
VLM_WRAPPER_BACKUP = BASE_DIR / "vlm_wrapper.py.bak"

# Dataset to model mapping (from run_pipeline_batch.py)
DATASET_MODEL_MAP = {
    "stl10_resnet": "vision/stl10_resnet.pth",
    "stl10_densenet": "vision/stl10_densenet.pth",
    "cub_resnet": "vision/cub_resnet.pth",
    "cub_densenet": "vision/cub_densenet.pth",
    "imdb_cnn": "text/imdb_cnn.pth",
    "imdb_2layernn": "text/imdb_2layernn.pth",
    "snli_cnn": "text/snli_cnn.pth",
    "snli_2layernn": "text/snli_2layernn.pth",
    "adult_census": "tabular/adult_census.pth",
    "adult_tabnn": "tabular/adult_tabnn.pth",
    "adult_2layernn": "tabular/adult_2layernn.pth",
    "cancer_2nn": "tabular/cancer_2nn.pth",
    "cancer_tabnn": "tabular/cancer_tabnn.pth",
    "cancer_2layernn": "tabular/cancer_2layernn.pth",
}

def get_worse_cases():
    worse_cases = []
    for root, dirs, files in os.walk(TRAINING_DATA_DIR):
        if "training_datapoint.json" in files:
            file_path = os.path.join(root, "training_datapoint.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                metrics = data.get("improvement_metrics", {})
                delta = metrics.get("faithfulness_delta", 0)
                
                if delta < 0:
                    question = data.get("question", {})
                    dataset_base_name = question.get("dataset_base_name")
                    row_no = question.get("row_no")
                    modality = data.get("modality")
                    
                    # Try to find the actual dataset name from dataset_base_name
                    dataset_name = "_".join(dataset_base_name.split("_")[:-1]) if "_q" in dataset_base_name else dataset_base_name
                    
                    if dataset_name not in DATASET_MODEL_MAP:
                         for ds_key in DATASET_MODEL_MAP.keys():
                             if dataset_base_name.startswith(ds_key):
                                 dataset_name = ds_key
                                 break
                    
                    worse_cases.append({
                        "file": file_path,
                        "dataset_base_name": dataset_base_name,
                        "row_no": row_no,
                        "modality": modality,
                        "dataset_name": dataset_name,
                        "orig_delta": delta,
                        "orig_score": metrics.get("original_faithfulness"),
                        "impr_score": metrics.get("improved_faithfulness")
                    })
            except:
                continue
    return worse_cases

def run_rerun():
    # 1. Get worse cases
    all_worse = get_worse_cases()
    print(f"Found {len(all_worse)} worse cases.")
    
    random.seed(42)
    sample_cases = random.sample(all_worse, min(10, len(all_worse)))
    
    # 2. Backup and modify vlm_wrapper.py
    if not VLM_WRAPPER_BACKUP.exists():
        shutil.copy(VLM_WRAPPER_PATH, VLM_WRAPPER_BACKUP)
        print("Backed up vlm_wrapper.py")
    
    with open(VLM_WRAPPER_PATH, 'r') as f:
        content = f.read()
    
    # Force temperature=0.0 in all places
    new_content = content.replace("temperature=0.1", "temperature=0.0")
    new_content = new_content.replace("temperature: float = 0.1,", "temperature: float = 0.0,")
    
    with open(VLM_WRAPPER_PATH, 'w') as f:
        f.write(new_content)
    print("Set temperature to 0.0 in vlm_wrapper.py")

    # 3. Create test_outputs dir
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_summary = []

    try:
        # 4. Run pipeline for each case
        for i, case in enumerate(sample_cases):
            print(f"\n[{i+1}/10] Rerunning {case['dataset_base_name']} row {case['row_no']} (Orig Delta: {case['orig_delta']:.4f})")
            
            # dataset_path e.g. dataset/train/tabular/adult_2layernn_q1.json
            dataset_file = f"dataset/train/{case['modality']}/{case['dataset_base_name']}.json"
            
            model_path = DATASET_MODEL_MAP.get(case['dataset_name'])
            if not model_path:
                print(f"  WARNING: No model found for dataset_name={case['dataset_name']}, skipping.")
                results_summary.append({
                    "case": f"{case['dataset_base_name']}_{case['row_no']}",
                    "orig_delta": case['orig_delta'],
                    "new_delta": "Skip",
                    "status": "Skipped (no model)"
                })
                continue

            # Look up the positional index of the question in the dataset file
            # (--question_id expects a list index, not row_no)
            dataset_file_abs = BASE_DIR / dataset_file
            question_index = None
            try:
                with open(dataset_file_abs, 'r') as f:
                    dataset_questions = json.load(f)
                if isinstance(dataset_questions, dict):
                    dataset_questions = dataset_questions.get('questions', [])
                for idx, q in enumerate(dataset_questions):
                    if q.get('row_no') == case['row_no']:
                        question_index = idx
                        break
            except Exception as e:
                print(f"  WARNING: Could not load dataset file {dataset_file_abs}: {e}, skipping.")
                results_summary.append({
                    "case": f"{case['dataset_base_name']}_{case['row_no']}",
                    "orig_delta": case['orig_delta'],
                    "new_delta": "Skip",
                    "status": "Skipped (dataset load error)"
                })
                continue

            if question_index is None:
                print(f"  WARNING: row_no={case['row_no']} not found in {dataset_file}, skipping.")
                results_summary.append({
                    "case": f"{case['dataset_base_name']}_{case['row_no']}",
                    "orig_delta": case['orig_delta'],
                    "new_delta": "Skip",
                    "status": "Skipped (row_no not found)"
                })
                continue

            print(f"  row_no={case['row_no']} → question index {question_index}")

            cmd = [
                "python", "xai_pipeline_v2.py",
                "--dataset", dataset_file,
                "--question_id", str(question_index),
                "--model_url", model_path,
                "--output_dir", str(TEST_OUTPUT_DIR),
                "--vlm", "claude-haiku-4-5-20251001",
                "--mode", "train"
            ]

            print(f"Executing: {' '.join(cmd)}")

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
                
                import re
                match = re.search(r'_(q\d+)', case['dataset_base_name'])
                q_type = match.group(1) if match else "q1"
                
                new_dp_path = TEST_OUTPUT_DIR / "training_data" / case['modality'] / case['dataset_name'] / q_type / str(case['row_no']) / "training_datapoint.json"
                
                new_delta = "N/A"
                if new_dp_path.exists():
                    with open(new_dp_path, 'r') as f:
                        new_data = json.load(f)
                    new_metrics = new_data.get("improvement_metrics", {})
                    new_delta = new_metrics.get("faithfulness_delta", 0)
                    
                results_summary.append({
                    "case": f"{case['dataset_base_name']}_{case['row_no']}",
                    "orig_delta": case['orig_delta'],
                    "new_delta": new_delta,
                    "status": "Improved" if isinstance(new_delta, (int, float)) and new_delta > case['orig_delta'] else "Same/Worse"
                })
                
                if proc.returncode != 0:
                    print(f"Pipeline failed (exit {proc.returncode}).")
                    print(f"STDOUT:\n{proc.stdout[-2000:]}")
                    print(f"STDERR:\n{proc.stderr[-2000:]}")
                else:
                    # Print last few lines of stdout for progress visibility
                    last_lines = "\n".join(proc.stdout.splitlines()[-20:])
                    print(f"Pipeline output (last 20 lines):\n{last_lines}")
            except Exception as e:
                print(f"Exception during run: {e}")
                results_summary.append({
                    "case": f"{case['dataset_base_name']}_{case['row_no']}",
                    "orig_delta": case['orig_delta'],
                    "new_delta": "Error",
                    "status": "Failed"
                })

    finally:
        # 5. Restore vlm_wrapper.py
        if VLM_WRAPPER_BACKUP.exists():
            shutil.copy(VLM_WRAPPER_BACKUP, VLM_WRAPPER_PATH)
            os.remove(VLM_WRAPPER_BACKUP)
            print("\nRestored vlm_wrapper.py")

    # 6. Show results
    print("\n" + "="*50)
    print("RERUN RESULTS (Temperature=0.0, Claude-Haiku)")
    print("="*50)
    print(f"{'Case':<30} | {'Orig Delta':<10} | {'New Delta':<10} | {'Status'}")
    print("-" * 70)
    for r in results_summary:
        new_d = f"{r['new_delta']:.4f}" if isinstance(r['new_delta'], (int, float)) else r['new_delta']
        print(f"{r['case']:<30} | {r['orig_delta']:<10.4f} | {new_d:<10} | {r['status']}")

if __name__ == "__main__":
    run_rerun()
