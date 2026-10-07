"""
Representation-space concept ablation (Part B): for each CUB test-split
image where a cub112 concept is present, orthogonally project out that
concept's CAV direction from the penultimate activation and measure the
resulting drop in the model's own predicted-class probability. No bounding
box, no masking -- pure forward-pass math on cached activations.

    m_concept_repr = max(0, P(y_hat | x) - P(y_hat | x with CAV direction removed))

Also computes a random-direction negative control (project out a random unit
vector of the same dimensionality instead of the CAV) -- if m_concept_repr is
not meaningfully larger than the random-direction ablation, the CAV isn't
capturing anything the model actually uses.

Usage:
    python concept_level/eval_repr_ablation.py [--auc_threshold 0.7]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from cub_model_utils import (
    MODEL_CONFIG, load_model_and_dataset, extract_activations,
    get_image_ids_for_dataset, head_forward_probs,
)

CUB_ROOT = "dataset_full/image/CUB_200_2011"
OUT_DIR = "concept_level"
MIN_CERTAINTY = 2
SEED = 42


def load_attribute_labels():
    return pd.read_csv(
        os.path.join(CUB_ROOT, "attributes", "image_attribute_labels_clean.txt"),
        sep=r"\s+", header=None,
        names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        engine="python",
    )


def ablate(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove `direction` (unit vector) from every row of `activations` via
    orthogonal projection: h' = h - (h . v) v."""
    v = direction / np.linalg.norm(direction)
    proj = activations @ v  # [N]
    return activations - np.outer(proj, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auc_threshold", type=float, default=0.7,
                         help="Only use CAVs with held-out probe AUC >= this")
    args = parser.parse_args()

    concepts = json.load(open(os.path.join(OUT_DIR, "cub112_concepts.json")))
    quality = pd.read_csv(os.path.join(OUT_DIR, "cav_quality.csv"))
    quality = quality[quality["held_out_auc"] >= args.auc_threshold]

    labels_df = load_attribute_labels()
    labels_df = labels_df[labels_df["certainty_id"] >= MIN_CERTAINTY]

    rng = np.random.default_rng(SEED)
    rows = []

    for model_name in MODEL_CONFIG:
        model_quality = quality[quality["model"] == model_name]
        if model_quality.empty:
            print(f"{model_name}: no CAVs pass AUC threshold {args.auc_threshold}, skipping")
            continue

        print(f"\n=== {model_name}: extracting test-split activations ===")
        model, head_module, dataset = load_model_and_dataset(model_name, split="test")
        image_ids = get_image_ids_for_dataset(dataset, CUB_ROOT)
        activations, _ = extract_activations(model, head_module, dataset)
        print(f"  activations shape: {activations.shape}")

        id_to_row = {img_id: i for i, img_id in enumerate(image_ids)}

        # Original prediction (target class + probability) for every test image, once.
        with torch.no_grad():
            act_t = torch.tensor(activations, dtype=torch.float32)
            probs_orig = head_forward_probs(model, model_name, act_t).numpy()  # [N, C]
        target_classes = probs_orig.argmax(axis=1)  # model's own predicted class
        p_original_all = probs_orig[np.arange(len(target_classes)), target_classes]

        for _, qrow in model_quality.iterrows():
            attribute_id = int(qrow["attribute_id"])
            attr_name = qrow["attribute_name"]
            cav_path = os.path.join(OUT_DIR, "cavs", f"{model_name}_{attribute_id}.npy")
            if not os.path.exists(cav_path):
                continue
            cav = np.load(cav_path)
            random_dir = rng.normal(size=cav.shape)
            random_dir = random_dir / np.linalg.norm(random_dir)

            present = labels_df[
                (labels_df["attribute_id"] == attribute_id) &
                (labels_df["is_present"] == 1) &
                (labels_df["image_id"].isin(id_to_row.keys()))
            ]
            if present.empty:
                continue

            idx = np.array([id_to_row[iid] for iid in present["image_id"]])
            h = activations[idx]
            tgt = target_classes[idx]
            p_orig = p_original_all[idx]

            h_cav_ablated = ablate(h, cav)
            h_rand_ablated = ablate(h, random_dir)
            with torch.no_grad():
                probs_cav = head_forward_probs(
                    model, model_name, torch.tensor(h_cav_ablated, dtype=torch.float32)
                ).numpy()
                probs_rand = head_forward_probs(
                    model, model_name, torch.tensor(h_rand_ablated, dtype=torch.float32)
                ).numpy()
            p_mod_cav = probs_cav[np.arange(len(tgt)), tgt]
            p_mod_rand = probs_rand[np.arange(len(tgt)), tgt]

            m_concept = np.maximum(0.0, p_orig - p_mod_cav)
            m_random = np.maximum(0.0, p_orig - p_mod_rand)

            for i, image_id in enumerate(present["image_id"]):
                rows.append({
                    "model": model_name,
                    "image_id": int(image_id),
                    "attribute_id": attribute_id,
                    "attribute_name": attr_name,
                    "target_class": int(tgt[i]),
                    "p_original": float(p_orig[i]),
                    "p_modified_cav": float(p_mod_cav[i]),
                    "m_concept_repr": float(m_concept[i]),
                    "p_modified_random": float(p_mod_rand[i]),
                    "m_concept_random": float(m_random[i]),
                    "cav_quality_auc": float(qrow["held_out_auc"]),
                })

            print(f"  [{attribute_id:3d}] {attr_name:<45s} n={len(idx):4d} "
                  f"mean m_concept={m_concept.mean():.4f}  mean m_random={m_random.mean():.4f}")

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "repr_ablation_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")
    if not out_df.empty:
        print(f"Overall mean m_concept_repr={out_df['m_concept_repr'].mean():.4f}  "
              f"mean m_concept_random={out_df['m_concept_random'].mean():.4f}")


if __name__ == "__main__":
    main()
