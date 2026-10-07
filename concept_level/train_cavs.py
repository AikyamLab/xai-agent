"""
Train a Concept Activation Vector (CAV) per concept per CUB model.

For each concept in concept_level/cub112_concepts.json, extract penultimate-
layer activations for CUB train-split images, then fit a logistic-regression
probe (activation -> attribute is_present) and take its normalized weight
vector as the CAV. This is used by eval_repr_ablation.py to ablate the
concept direction in representation space (independent of any bounding box).

Usage:
    python concept_level/train_cavs.py
    python concept_level/train_cavs.py --max_train_images 500   # fast partial run
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cub_model_utils import (
    MODEL_CONFIG, load_model_and_dataset, extract_activations,
    get_image_ids_for_dataset,
)

CUB_ROOT = "dataset_full/image/CUB_200_2011"
OUT_DIR = "concept_level"
MIN_CERTAINTY = 2  # exclude only "not visible" (1); keep guessing/probably/definitely
MIN_SUPPORT = 15   # need at least this many positive AND negative examples
VAL_FRACTION = 0.2
SEED = 42


def load_attribute_labels():
    """Return DataFrame with columns image_id, attribute_id, is_present, certainty_id."""
    return pd.read_csv(
        os.path.join(CUB_ROOT, "attributes", "image_attribute_labels_clean.txt"),
        sep=r"\s+", header=None,
        names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        engine="python",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_images", type=int, default=None,
                         help="Use only the first N train-split images (fast partial run "
                              "for smoke-testing; omit for the real full-scale run).")
    args = parser.parse_args()

    concepts = json.load(open(os.path.join(OUT_DIR, "cub112_concepts.json")))
    attribute_ids = sorted(int(a) for a in concepts.keys())

    labels_df = load_attribute_labels()
    labels_df = labels_df[labels_df["certainty_id"] >= MIN_CERTAINTY]

    cav_dir = os.path.join(OUT_DIR, "cavs")
    os.makedirs(cav_dir, exist_ok=True)

    quality_rows = []

    for model_name in MODEL_CONFIG:
        print(f"\n=== {model_name}: extracting train-split activations ===")
        model, head_module, dataset = load_model_and_dataset(model_name, split="train")
        image_ids_full = get_image_ids_for_dataset(dataset, CUB_ROOT)
        if args.max_train_images is not None:
            dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_train_images, len(dataset)))))
            image_ids = image_ids_full[:len(dataset)]
        else:
            image_ids = image_ids_full
        activations, _ = extract_activations(model, head_module, dataset)
        print(f"  activations shape: {activations.shape}")

        id_to_row = {img_id: i for i, img_id in enumerate(image_ids)}
        image_id_set = set(image_ids)

        for attribute_id in attribute_ids:
            attr_name = concepts[str(attribute_id)]["attribute_name"]
            sub = labels_df[
                (labels_df["attribute_id"] == attribute_id) &
                (labels_df["image_id"].isin(image_id_set))
            ]
            if sub.empty:
                continue

            X = np.stack([activations[id_to_row[iid]] for iid in sub["image_id"]])
            y = sub["is_present"].to_numpy()

            n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
            if n_pos < MIN_SUPPORT or n_neg < MIN_SUPPORT:
                quality_rows.append({
                    "model": model_name, "attribute_id": attribute_id,
                    "attribute_name": attr_name, "n_positive": n_pos, "n_negative": n_neg,
                    "held_out_auc": None, "skipped_reason": "insufficient_support",
                })
                continue

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=VAL_FRACTION, random_state=SEED, stratify=y
            )
            clf = LogisticRegression(class_weight="balanced", max_iter=2000)
            clf.fit(X_train, y_train)
            val_scores = clf.decision_function(X_val)
            auc = roc_auc_score(y_val, val_scores)

            cav = clf.coef_[0]
            cav = cav / np.linalg.norm(cav)
            np.save(os.path.join(cav_dir, f"{model_name}_{attribute_id}.npy"), cav)

            quality_rows.append({
                "model": model_name, "attribute_id": attribute_id,
                "attribute_name": attr_name, "n_positive": n_pos, "n_negative": n_neg,
                "held_out_auc": round(float(auc), 4), "skipped_reason": None,
            })
            print(f"  [{attribute_id:3d}] {attr_name:<45s} n_pos={n_pos:4d} n_neg={n_neg:4d} AUC={auc:.4f}")

    quality_df = pd.DataFrame(quality_rows)
    quality_path = os.path.join(OUT_DIR, "cav_quality.csv")
    quality_df.to_csv(quality_path, index=False)
    print(f"\nWrote {quality_path} ({len(quality_df)} rows, "
          f"{quality_df['held_out_auc'].notna().sum()} CAVs trained)")


if __name__ == "__main__":
    main()
