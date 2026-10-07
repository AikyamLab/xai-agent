"""
Compare region-based Q11 faithfulness (Part A: gray-fill masking a
human-annotated spatial region) against representation-space concept
ablation (Part B: orthogonal CAV projection), on the (image, concept) pairs
covered by both. This is the key rebuttal evidence: if the two agree, that
supports using the cheaper/agent-compatible local-perturbation proxy as a
faithful stand-in for concept-level faithfulness in general.

Requires concept_level/q11_results.csv (from eval_q11.py) and
concept_level/repr_ablation_results.csv (from eval_repr_ablation.py).

Usage:
    python concept_level/compare_region_vs_repr.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = "concept_level"


def main():
    region_path = os.path.join(OUT_DIR, "q11_results.csv")
    repr_path = os.path.join(OUT_DIR, "repr_ablation_results.csv")
    if not os.path.exists(region_path) or not os.path.exists(repr_path):
        print(f"Missing input(s): need both {region_path} (run eval_q11.py) "
              f"and {repr_path} (run eval_repr_ablation.py)")
        return

    region_df = pd.read_csv(region_path)
    repr_df = pd.read_csv(repr_path)

    # q11_results.csv uses row_no (Q1-benchmark index); CUB image_id = row_no + 1.
    region_df = region_df.copy()
    region_df["image_id"] = region_df["row_no"] + 1

    merged = region_df.merge(
        repr_df, on=["model", "image_id", "attribute_id"], suffixes=("_region", "_repr")
    )

    if merged.empty:
        print("No overlapping (model, image_id, attribute_id) pairs between "
              "region-based Q11 results and representation-space ablation results. "
              "This can happen if the two runs used disjoint concept/image samples -- "
              "widen the Q11 sample or the cub112 concept set to get overlap.")
        return

    x = merged["score"].to_numpy()          # Part A: gray-fill region faithfulness
    y = merged["m_concept_repr"].to_numpy()  # Part B: representation-space ablation

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    mean_abs_diff = np.abs(x - y).mean()

    merged.to_csv(os.path.join(OUT_DIR, "region_vs_repr_merged.csv"), index=False)

    print(f"n = {len(merged)} overlapping (model, image, concept) pairs")
    print(f"Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4g})")
    print(f"Spearman r = {spearman_r:.4f}  (p = {spearman_p:.4g})")
    print(f"Mean |region_score - repr_score| = {mean_abs_diff:.4f}")

    with open(os.path.join(OUT_DIR, "region_vs_repr_correlation.json"), "w") as f:
        import json
        json.dump({
            "n": len(merged),
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
            "mean_abs_diff": float(mean_abs_diff),
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, alpha=0.4, s=18)
    lims = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Region-based faithfulness (gray-fill)")
    ax.set_ylabel("Representation-space faithfulness (CAV ablation)")
    ax.set_title(f"Q11 region vs. representation-space faithfulness\n"
                 f"Pearson r={pearson_r:.2f}, Spearman r={spearman_r:.2f}, n={len(merged)}")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "region_vs_repr_scatter.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
