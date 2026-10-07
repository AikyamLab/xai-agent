"""
Build Q11 (concept-level explanation) dataset instances for CUB.

One row per image (reusing the same 120-image benchmark already used by
dataset_full/vision/cub_{model}_q1.json), each carrying a `candidate_concepts`
list: the cub112-style concepts (concept_level/cub112_concepts.json) that are
actually present in that image (ground-truth attribute labels) AND have a
trained CAV passing the quality threshold (concept_level/cav_quality.csv, if
present -- if train_cavs.py hasn't been run yet, falls back to "present +
selected concept" only, with a warning, since there's nothing to filter on).

No bounding boxes, no part-keypoint mapping -- Q11's evaluator scores in
representation space (CAV ablation), not via masking, so none of that spatial
machinery is needed for dataset construction anymore.

Usage:
    python concept_level/build_q11_dataset.py
"""

import json
import os

import pandas as pd

CUB_ROOT = "dataset_full/image/CUB_200_2011"
Q1_JSON = {
    "cub_resnet": "dataset_full/vision/cub_resnet_q1.json",
    "cub_densenet": "dataset_full/vision/cub_densenet_q1.json",
}
OUT_JSON = {
    "cub_resnet": "dataset_full/vision/cub_resnet_q11.json",
    "cub_densenet": "dataset_full/vision/cub_densenet_q11.json",
}

MIN_CERTAINTY = 2  # "guessing" (1) excluded, "probably"(2)/"definitely"(3) kept
AUC_THRESHOLD = 0.7
QUESTION_TEXT = "Which concept was most responsible for the model's prediction?"


def load_attribute_labels(image_ids):
    df = pd.read_csv(
        os.path.join(CUB_ROOT, "attributes", "image_attribute_labels_clean.txt"),
        sep=r"\s+", header=None,
        names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        engine="python",
    )
    df = df[df["image_id"].isin(image_ids)]
    return {
        (row.image_id, row.attribute_id): (row.is_present, row.certainty_id)
        for row in df.itertuples()
    }


def load_cav_quality(model_name):
    """Return {attribute_id: held_out_auc} for CAVs that passed AUC_THRESHOLD
    for this model, or None if cav_quality.csv doesn't exist yet."""
    path = "concept_level/cav_quality.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[(df["model"] == model_name) & (df["held_out_auc"].notna())
             & (df["held_out_auc"] >= AUC_THRESHOLD)]
    return {int(row.attribute_id): float(row.held_out_auc) for row in df.itertuples()}


def main():
    concepts = json.load(open("concept_level/cub112_concepts.json"))
    concepts = {int(k): v for k, v in concepts.items()}

    for model, q1_path in Q1_JSON.items():
        q1_rows = json.load(open(q1_path))
        image_ids = [r["row_no"] + 1 for r in q1_rows]

        attr_labels = load_attribute_labels(set(image_ids))
        cav_ok = load_cav_quality(model)
        if cav_ok is None:
            print(f"WARNING [{model}]: concept_level/cav_quality.csv not found -- "
                  f"candidate lists will NOT be filtered by CAV quality. Run "
                  f"train_cavs.py first for a real run; evaluation will fail per-concept "
                  f"at runtime for any concept without a trained CAV file.")
            usable_attribute_ids = set(concepts.keys())
        else:
            usable_attribute_ids = set(cav_ok.keys())
            print(f"[{model}]: {len(usable_attribute_ids)}/{len(concepts)} concepts "
                  f"have a CAV with AUC >= {AUC_THRESHOLD}")

        out_rows = []
        n_skipped_no_candidates = 0
        for q1_row in q1_rows:
            image_id = q1_row["row_no"] + 1
            candidate_concepts = []
            for attribute_id in sorted(usable_attribute_ids):
                label = attr_labels.get((image_id, attribute_id))
                if label is None:
                    continue
                is_present, certainty_id = label
                if is_present == 1 and certainty_id >= MIN_CERTAINTY:
                    candidate_concepts.append({
                        "attribute_id": attribute_id,
                        "concept_name": concepts[attribute_id]["attribute_name"]
                            .replace("has_", "").replace("::", ": ").replace("_", " "),
                    })

            if not candidate_concepts:
                n_skipped_no_candidates += 1
                continue

            out_rows.append({
                "row_no": q1_row["row_no"],
                "image_path": q1_row["image_path"],
                "modality": "vision",
                "dataset": q1_row["dataset"],
                "model": q1_row["model"],
                "features": q1_row["features"],
                "target": q1_row["target"],
                "predicted": q1_row["predicted"],
                "example": (
                    f"The model predicted the image as {q1_row['predicted']['label']}. "
                    f"{QUESTION_TEXT}"
                ),
                "q_type": 11,
                "q": QUESTION_TEXT,
                "candidate_concepts": candidate_concepts,
            })

        os.makedirs(os.path.dirname(OUT_JSON[model]), exist_ok=True)
        with open(OUT_JSON[model], "w") as f:
            json.dump(out_rows, f, indent=2)

        print(f"{model}: {len(out_rows)} instances written "
              f"({n_skipped_no_candidates} images skipped, no candidate concepts) "
              f"-> {OUT_JSON[model]}")


if __name__ == "__main__":
    main()
