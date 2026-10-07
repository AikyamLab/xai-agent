"""
Build the attribute -> part-keypoint mapping table for Q11 (concept-level
explanation, CUB-200-2011 only).

CUB attribute names have the fixed form "has_<category>::<value>" (e.g.
"has_wing_color::blue"). There are 30 distinct categories across the 312
attributes; each category is hand-mapped to one (or two, for bilateral parts)
of the 15 CUB part keypoints. Categories with no meaningful spatial referent
(has_size::*, has_shape::*, has_primary_color::*) are left unmapped.

Outputs:
    concept_level/concept_to_part.json   -- {attribute_id: {attribute_name, category, part_candidates}}
    concept_level/unmapped_attributes.json -- {attribute_id: attribute_name} for excluded attrs
"""

import json
import os

CUB_ROOT = "dataset_full/image/CUB_200_2011"
OUT_DIR = "concept_level"

# category (the token between "has_" and "::") -> list of candidate CUB part
# names (in preference order) to try per-image. Two-part lists are for
# bilaterally-annotated parts; per-image bbox generation picks whichever is
# visible, preferring the first candidate.
CATEGORY_TO_PART = {
    "bill_shape": ["beak"],
    "bill_length": ["beak"],
    "bill_color": ["beak"],
    "wing_color": ["left wing", "right wing"],
    "wing_shape": ["left wing", "right wing"],
    "wing_pattern": ["left wing", "right wing"],
    "upperparts_color": ["back"],
    "underparts_color": ["belly"],
    "breast_pattern": ["breast"],
    "breast_color": ["breast"],
    "back_color": ["back"],
    "back_pattern": ["back"],
    "tail_shape": ["tail"],
    "tail_pattern": ["tail"],
    "upper_tail_color": ["tail"],
    "under_tail_color": ["tail"],
    "head_pattern": ["crown"],  # approximation: crown as representative head region
    "throat_color": ["throat"],
    "eye_color": ["left eye", "right eye"],
    "forehead_color": ["forehead"],
    "nape_color": ["nape"],
    "belly_color": ["belly"],
    "belly_pattern": ["belly"],
    "leg_color": ["left leg", "right leg"],
    "crown_color": ["crown"],
    # No spatial referent -> unmapped
    "size": None,
    "shape": None,
    "primary_color": None,
}


def parse_attributes(path):
    """Return list of (attribute_id, attribute_name, category)."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: "<id> <id> <name>" (id duplicated in this file)
            parts = line.split(" ", 2)
            if len(parts) < 3:
                parts = line.split(None, 2)
            attr_id = int(parts[0])
            name = parts[-1]
            assert name.startswith("has_") and "::" in name, f"unexpected attribute format: {name}"
            category = name[len("has_"):].split("::")[0]
            out.append((attr_id, name, category))
    return out


def main():
    attrs_path = os.path.join(CUB_ROOT, "attributes", "attributes.txt")
    attributes = parse_attributes(attrs_path)

    mapped = {}
    unmapped = {}
    unknown_categories = set()

    for attr_id, name, category in attributes:
        if category not in CATEGORY_TO_PART:
            unknown_categories.add(category)
            unmapped[attr_id] = name
            continue
        part_candidates = CATEGORY_TO_PART[category]
        if part_candidates is None:
            unmapped[attr_id] = name
            continue
        mapped[attr_id] = {
            "attribute_name": name,
            "category": category,
            "part_candidates": part_candidates,
        }

    if unknown_categories:
        raise RuntimeError(
            f"Found attribute categories with no entry in CATEGORY_TO_PART: "
            f"{sorted(unknown_categories)}. Update the mapping table."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "concept_to_part.json"), "w") as f:
        json.dump(mapped, f, indent=2)
    with open(os.path.join(OUT_DIR, "unmapped_attributes.json"), "w") as f:
        json.dump(unmapped, f, indent=2)

    print(f"Total attributes: {len(attributes)}")
    print(f"Mapped: {len(mapped)}  Unmapped: {len(unmapped)}")
    print("Wrote concept_level/concept_to_part.json and concept_level/unmapped_attributes.json")


if __name__ == "__main__":
    main()
