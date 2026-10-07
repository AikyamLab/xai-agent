"""
Lightweight robustness check for Q11 (Part D): on a small random sample of
already-evaluated Q11 instances, re-mask the SAME region with BLUR instead of
GRAY-fill and compare the resulting faithfulness scores. If gray-fill and
blur agree, the faithfulness conclusion isn't an artifact of the specific
masking color/texture. Cheaper than standing up SD inpainting + CLIP
filtering -- reuses the existing BLUR strategy already implemented in
evaluation/masking_utils.py (no new masking code).

Requires Q11 evaluation results to already exist under --output_dir (i.e.
eval_q11.py --run has been executed).

Usage:
    python concept_level/robustness_blur_check.py --output_dir outputs --n_samples 25
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cub_model_utils import MODEL_CONFIG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.masking_utils import get_masker, MaskingStrategy
from evaluation.explanation_faithfulness.q11_evaluator import Q11Evaluator

IMAGE_ROOT = "dataset_full/image"
MODELS = ["cub_resnet", "cub_densenet"]
OUT_DIR = "concept_level"
SEED = 42


def load_q11_eval_records(output_dir, model):
    """Return list of dicts with row_no, bbox, p_original, original_class, score
    for every completed Q11 evaluation under output_dir for this model."""
    pattern = os.path.join(output_dir, "evaluations", "vision", model, "q11", "*", "evaluation*.json")
    records = []
    for path in glob.glob(pattern):
        row_no = int(os.path.basename(os.path.dirname(path)))
        with open(path) as f:
            data = json.load(f)
        faithfulness = data.get("faithfulness", {})
        region = faithfulness.get("details", {}).get("region", {})
        bbox = region.get("bounding_box")
        if bbox is None or faithfulness.get("p_original") is None:
            continue
        records.append({
            "model": model,
            "row_no": row_no,
            "bbox": bbox,
            "p_original": faithfulness["p_original"],
            "original_class": int(faithfulness["original_class"]),
            "score_gray_original_run": faithfulness["score"],
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_samples", type=int, default=25)
    args = parser.parse_args()

    all_records = []
    for model in MODELS:
        all_records.extend(load_q11_eval_records(args.output_dir, model))

    if not all_records:
        print(f"No Q11 evaluation results found under {args.output_dir}/evaluations/vision/*/q11/. "
              f"Run eval_q11.py --run first.")
        return

    rng = random.Random(SEED)
    sample = rng.sample(all_records, min(args.n_samples, len(all_records)))

    # Row_no -> image_path, per model, from the Q11 dataset JSONs.
    row_to_path = {}
    for model in MODELS:
        rows = json.load(open(f"dataset_full/vision/{model}_q11.json"))
        row_to_path[model] = {r["row_no"]: r["image_path"] for r in rows}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Q11Evaluator(modality="vision")
    masker_gray = get_masker("vision", strategy=MaskingStrategy.GRAY)
    masker_blur = get_masker("vision", strategy=MaskingStrategy.BLUR)

    model_cache = {}
    results = []
    for rec in sample:
        model_name = rec["model"]
        if model_name not in model_cache:
            cfg = MODEL_CONFIG[model_name]
            m, transform = cfg["module"].load_model(cfg["checkpoint"])
            m.eval()
            model_cache[model_name] = (m, transform)
        model, transform = model_cache[model_name]

        image_path = IMAGE_ROOT + row_to_path[model_name][rec["row_no"]]
        image = Image.open(image_path).convert("RGB")
        region = {"bounding_box": rec["bbox"]}

        masked_gray = masker_gray.mask(image, region, row_no=rec["row_no"], mask_suffix="_robustness_gray")
        masked_blur = masker_blur.mask(image, region, row_no=rec["row_no"], mask_suffix="_robustness_blur")

        pred_gray = evaluator.get_prediction(model, masked_gray, processor=transform, device=device)
        pred_blur = evaluator.get_prediction(model, masked_blur, processor=transform, device=device)

        p_orig = rec["p_original"]
        tgt = rec["original_class"]
        p_mod_gray = float(pred_gray["probabilities"][tgt])
        p_mod_blur = float(pred_blur["probabilities"][tgt])

        results.append({
            "model": model_name,
            "row_no": rec["row_no"],
            "p_original": p_orig,
            "m_gray_recomputed": max(0.0, p_orig - p_mod_gray),
            "m_gray_original_run": rec["score_gray_original_run"],
            "m_blur": max(0.0, p_orig - p_mod_blur),
        })
        print(f"  [{model_name}] row {rec['row_no']}: m_gray={results[-1]['m_gray_recomputed']:.4f} "
              f"(orig run: {rec['score_gray_original_run']:.4f})  m_blur={results[-1]['m_blur']:.4f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "q11_robustness_blur_vs_gray.csv")
    df.to_csv(out_csv, index=False)

    x = df["m_gray_recomputed"].to_numpy()
    y = df["m_blur"].to_numpy()
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    mean_abs_diff = np.abs(x - y).mean()

    print(f"\nn = {len(df)}")
    print(f"Pearson  r = {pearson_r:.4f} (p = {pearson_p:.4g})")
    print(f"Spearman r = {spearman_r:.4f} (p = {spearman_p:.4g})")
    print(f"Mean |gray - blur| = {mean_abs_diff:.4f}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
