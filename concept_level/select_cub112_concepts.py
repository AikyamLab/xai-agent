"""
Select a reliable concept subset of CUB's 312 attributes, following the
class-level majority-vote protocol from Koh et al. 2020 ("Concept Bottleneck
Models"): for each attribute, for each of the 200 species classes,
majority-vote presence (certainty >= "probably") across all images of that
class; keep the attribute if it is majority-present in at least MIN_CLASSES
classes. Koh et al. report 112 concepts under this protocol; our exact
threshold choices (certainty cutoff, MIN_CLASSES) are a documented
approximation, not a byte-for-byte replication, so the selected count may
differ (~75-110 depending on the certainty threshold) -- what matters is that
the selection rule is principled and pre-registered, not tuned post hoc.

This concept set is used for the representation-space CAV ablation (Part B),
independent of the spatial part-mapping used for the region-based Q11
(Part A, concept_level/concept_to_part.json) -- no bounding box involved here.

Usage:
    python concept_level/select_cub112_concepts.py
"""

import json
import os

import pandas as pd

CUB_ROOT = "dataset_full/image/CUB_200_2011"
OUT_DIR = "concept_level"
MIN_CERTAINTY = 3  # certainty codes: 1=not visible, 2=guessing, 3=probably, 4=definitely
MIN_CLASSES = 10   # attribute must be majority-present in >= this many classes


def parse_attributes(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            attr_id_str, name = line.split(" ", 1)
            out[int(attr_id_str)] = name
    return out


def main():
    attributes = parse_attributes(os.path.join(CUB_ROOT, "attributes", "attributes.txt"))

    labels = pd.read_csv(
        os.path.join(CUB_ROOT, "attributes", "image_attribute_labels_clean.txt"),
        sep=r"\s+", header=None,
        names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        engine="python",
    )
    classes = pd.read_csv(
        os.path.join(CUB_ROOT, "image_class_labels.txt"),
        sep=" ", header=None, names=["image_id", "class_id"],
    )

    labels = labels.merge(classes, on="image_id")
    labels["counted_present"] = (
        (labels["is_present"] == 1) & (labels["certainty_id"] >= MIN_CERTAINTY)
    )

    # Per (attribute_id, class_id): majority vote across that class's images.
    majority = (
        labels.groupby(["attribute_id", "class_id"])["counted_present"]
        .mean()
        .reset_index()
    )
    majority["class_majority_present"] = majority["counted_present"] > 0.5

    n_classes_present = (
        majority[majority["class_majority_present"]]
        .groupby("attribute_id")
        .size()
    )

    selected = sorted(int(a) for a in n_classes_present[n_classes_present >= MIN_CLASSES].index)

    result = {
        str(attr_id): {
            "attribute_name": attributes[attr_id],
            "n_classes_majority_present": int(n_classes_present[attr_id]),
        }
        for attr_id in selected
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "cub112_concepts.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Selected {len(selected)} / {len(attributes)} attributes "
          f"(majority-present in >= {MIN_CLASSES} classes, certainty >= {MIN_CERTAINTY})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
