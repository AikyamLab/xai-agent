"""
Re-score the 240 already-answered Q11 questions (q11_outputs/base +
q11_outputs/ckpt) using the newly trained (full-train-split) CAVs, without
re-running the agent pipeline. The Actor's concept_name choice is already
fixed in each complete_results.json; only the CAV-ablation faithfulness
score depends on CAV quality, so this just re-invokes Q11Evaluator.evaluate()
per question with the new CAVs now sitting in concept_level/cavs/.
"""
import glob
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "models_to_read/vision")

import torch

import load_cub_resnet
import load_cub_densenet
from evaluation.explanation_faithfulness.q11_evaluator import Q11Evaluator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODULES = {"cub_resnet": load_cub_resnet, "cub_densenet": load_cub_densenet}
MODEL_CACHE = {}

def get_model(dataset_name):
    if dataset_name not in MODEL_CACHE:
        mod = MODULES[dataset_name]
        mod.set_dataset_root("dataset_full")
        ckpt = f"models_to_read/vision/{dataset_name}.pth"
        model, transform = mod.load_model(ckpt)
        model.eval()
        MODEL_CACHE[dataset_name] = (mod, model, transform)
    return MODEL_CACHE[dataset_name]

evaluator = Q11Evaluator()
results = []

for run in ["base", "ckpt"]:
    files = sorted(glob.glob(f"q11_outputs/{run}/complete_results/vision/*/q11/*/complete_results.json"))
    for f in files:
        d = json.load(open(f))
        q = d["question"]
        dataset_name = f.split("/complete_results/vision/")[1].split("/")[0]
        row_no = q["row_no"]
        predicted_class_idx = q["predicted"]["value"]
        candidate_concepts = q["candidate_concepts"]
        dataset_base_name = q["dataset_base_name"]
        agent_output = {"output": d["results"]["output"]}

        old_faith = d.get("evaluation", {}).get("faithfulness", {})
        old_score = old_faith.get("score")
        old_raw_drop = old_faith.get("details", {}).get("raw_drop")

        mod, model, transform = get_model(dataset_name)
        data = mod.load_data(row_no, split="test")
        image = data["image"]

        result = evaluator.evaluate(
            agent_output=agent_output,
            original_input=image,
            model=model,
            original_prediction={"predicted_class_idx": predicted_class_idx},
            candidate_concepts=candidate_concepts,
            dataset_base_name=dataset_base_name,
            processor=transform,
            device=DEVICE,
        )
        new_raw_drop = result.details.get("raw_drop") if result.details else None
        results.append({
            "run": run, "dataset": dataset_name, "row_no": row_no,
            "old_score": old_score, "new_score": result.score,
            "old_raw_drop": old_raw_drop, "new_raw_drop": new_raw_drop,
            "errors": result.errors,
        })
        print(f"[{run}/{dataset_name}/{row_no}] old={old_score:.4f} new={result.score:.4f} "
              f"(raw_drop old={old_raw_drop:.4f} new={new_raw_drop:.4f})" if old_raw_drop is not None
              else f"[{run}/{dataset_name}/{row_no}] new={result.score:.4f} errors={result.errors}")

json.dump(results, open("concept_level/q11_rescore_results.json", "w"))
print(f"\nDone. {len(results)} questions re-scored.")
