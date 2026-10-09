"""
Run Q11 (concept-level explanation, CUB-only) through the full
Proposer->Actor->Evaluator pipeline and report mean faithfulness broken down
by attribute category (color / shape / pattern).

This is a thin wrapper around the existing run_pipeline_batch.py /
MEA_pipeline.py machinery -- it does not reimplement any pipeline logic.

Usage:
    # Just summarize whatever Q11 evaluation results already exist under --output_dir
    python concept_level/eval_q11.py --output_dir outputs

    # Actually launch the pipeline run first (costs VLM API calls), then summarize
    python concept_level/eval_q11.py --output_dir outputs --run --vlm gemini-2.5-pro
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

MODELS = ["cub_resnet", "cub_densenet"]


def category_group(category: str) -> str:
    if category.endswith("_color"):
        return "color"
    if category.endswith("_shape") or category == "bill_length":
        return "shape"
    if category.endswith("_pattern"):
        return "pattern"
    return "other"


def load_attribute_categories():
    """Return {attribute_id: category_group_string}."""
    concept_to_part = json.load(open("concept_level/concept_to_part.json"))
    return {
        int(attr_id): category_group(meta["category"])
        for attr_id, meta in concept_to_part.items()
    }


def load_row_to_attribute(model):
    """Return {row_no: concept_attribute_id} for this model's Q11 dataset."""
    path = f"dataset_full/vision/{model}_q11.json"
    rows = json.load(open(path))
    return {r["row_no"]: r["concept_attribute_id"] for r in rows}


def run_pipeline(args):
    cmd = [
        sys.executable, "run_pipeline_batch.py",
        "--dataset_dir", "dataset_full",
        "--datasets", *MODELS,
        "--q_types", "11",
        "--question_ids", "all",
        "--output_dir", args.output_dir,
        "--vlm", args.vlm,
    ]
    if args.parallel:
        cmd.append("--parallel")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize(args):
    attr_categories = load_attribute_categories()

    rows = []
    for model in MODELS:
        row_to_attr = load_row_to_attribute(model)
        pattern = os.path.join(args.output_dir, "evaluations", "vision", model, "q11", "*", "evaluation*.json")
        for path in glob.glob(pattern):
            row_no = int(os.path.basename(os.path.dirname(path)))
            with open(path) as f:
                data = json.load(f)
            faithfulness = data.get("faithfulness", {})
            score = faithfulness.get("score")
            if score is None:
                continue
            attribute_id = row_to_attr.get(row_no)
            category = attr_categories.get(attribute_id, "unknown")
            rows.append({
                "model": model,
                "row_no": row_no,
                "attribute_id": attribute_id,
                "category": category,
                "score": score,
                "bbox_iou_with_ground_truth": faithfulness.get("details", {}).get("bbox_iou_with_ground_truth"),
            })

    if not rows:
        print(f"No Q11 evaluation results found under {args.output_dir}/evaluations/vision/*/q11/. "
              f"Run with --run first (or point --output_dir at an existing run).")
        return

    os.makedirs("concept_level", exist_ok=True)
    out_csv = os.path.join("concept_level", "q11_results.csv")
    with open(out_csv, "w") as f:
        f.write("model,row_no,attribute_id,category,score,bbox_iou_with_ground_truth\n")
        for r in rows:
            f.write(f"{r['model']},{r['row_no']},{r['attribute_id']},{r['category']},"
                    f"{r['score']},{r['bbox_iou_with_ground_truth']}\n")
    print(f"Wrote {len(rows)} rows to {out_csv}")

    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r["score"])
        by_category["__overall__"].append(r["score"])

    print(f"\n{'category':<12}{'n':<8}{'mean_faithfulness':<20}")
    print("-" * 40)
    for cat in sorted(by_category.keys() - {"__overall__"}) + ["__overall__"]:
        vals = by_category[cat]
        label = "OVERALL" if cat == "__overall__" else cat
        print(f"{label:<12}{len(vals):<8}{sum(vals) / len(vals):<20.4f}")

    ious = [r["bbox_iou_with_ground_truth"] for r in rows if r["bbox_iou_with_ground_truth"] is not None]
    if ious:
        print(f"\nAgent fidelity: mean bbox IoU with ground-truth concept region = "
              f"{sum(ious) / len(ious):.4f} (n={len(ious)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--vlm", type=str, default="gemini-2.5-pro")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--run", action="store_true", help="Launch the pipeline run before summarizing (costs VLM API calls)")
    args = parser.parse_args()

    if args.run:
        run_pipeline(args)
    summarize(args)


if __name__ == "__main__":
    main()
