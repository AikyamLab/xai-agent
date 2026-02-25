#!/usr/bin/env python3
"""
Summarize evaluation scores by modality / q_type / dataset.
Reads all evaluation.json files under:
  <EVAL_ROOT>/<modality>/<dataset>/<q_type>/<row_no>/evaluation.json
"""

import json
from pathlib import Path
from collections import defaultdict

EVAL_ROOT = Path(
    "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2"
    "/dpo_eval_outputs/evaluations"
)

# SCORE_KEYS = ["overall_score", "quality_score", "completeness"]
SCORE_KEYS = ['soft_score']
FAITHFULNESS_KEY = "faithfulness"


def collect(root: Path):
    """Return a nested dict: data[modality][dataset][q_type] = list of records."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for eval_file in sorted(root.rglob("evaluation.json")):
        parts = eval_file.relative_to(root).parts
        # expected: modality / dataset / q_type / row_no / evaluation.json
        if len(parts) != 5:
            continue
        modality, dataset, q_type, row_no, _ = parts

        with open(eval_file) as f:
            record = json.load(f)
        data[modality][dataset][q_type].append(record)

    return data


def avg(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else float("nan")


def summarize(data):
    rows = []

    for modality in sorted(data):
        for dataset in sorted(data[modality]):
            for q_type in sorted(data[modality][dataset]):
                records = data[modality][dataset][q_type]
                n = len(records)

                row = {
                    "modality": modality,
                    "dataset": dataset,
                    "q_type": q_type,
                    "n": n,
                }

                for key in SCORE_KEYS:
                    row[key] = avg([r.get(key) for r in records])

                # faithfulness.score (may be absent)
                row["faithfulness_score"] = avg(
                    [
                        r.get(FAITHFULNESS_KEY, {}).get("score")
                        if isinstance(r.get(FAITHFULNESS_KEY), dict)
                        else None
                        for r in records
                    ]
                )

                # faithfulness pass rate
                passed = [
                    r.get(FAITHFULNESS_KEY, {}).get("passed")
                    for r in records
                    if isinstance(r.get(FAITHFULNESS_KEY), dict)
                    and r[FAITHFULNESS_KEY].get("passed") is not None
                ]
                row["faithfulness_pass_rate"] = (
                    sum(passed) / len(passed) if passed else float("nan")
                )

                rows.append(row)

    return rows


def print_table(rows):
    col_widths = {
        "modality": 10,
        "dataset": 18,
        "q_type": 7,
        "n": 5,
        "faithfulness_score": 18,
        "faithfulness_pass_rate": 22,
    }

    header = (
        f"{'modality':<{col_widths['modality']}}"
        f"{'dataset':<{col_widths['dataset']}}"
        f"{'q_type':<{col_widths['q_type']}}"
        f"{'n':>{col_widths['n']}}"
        f"{'faith_score':>{col_widths['faithfulness_score']}}"
        f"{'faith_pass%':>{col_widths['faithfulness_pass_rate']}}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    prev_key = None
    for row in rows:
        key = (row["modality"], row["dataset"])
        if prev_key and key != prev_key:
            print()
        prev_key = key

        print(
            f"{row['modality']:<{col_widths['modality']}}"
            f"{row['dataset']:<{col_widths['dataset']}}"
            f"{row['q_type']:<{col_widths['q_type']}}"
            f"{row['n']:>{col_widths['n']}}"
            f"{row['faithfulness_score']:>{col_widths['faithfulness_score']}.4f}"
            f"{row['faithfulness_pass_rate']*100:>{col_widths['faithfulness_pass_rate']}.1f}%"
        )

    print(sep)

    # --- aggregate totals per (modality, dataset) ---
    print("\n=== Aggregate by modality / dataset ===\n")
    from itertools import groupby

    agg_header = (
        f"{'modality':<{col_widths['modality']}}"
        f"{'dataset':<{col_widths['dataset']}}"
        f"{'n_total':>{7}}"
        f"{'faith_score':>{col_widths['faithfulness_score']}}"
        f"{'faith_pass%':>{col_widths['faithfulness_pass_rate']}}"
    )
    print("-" * len(agg_header))
    print(agg_header)
    print("-" * len(agg_header))

    for (modality, dataset), group in groupby(rows, key=lambda r: (r["modality"], r["dataset"])):
        group = list(group)
        n_total = sum(r["n"] for r in group)

        def weighted_avg(key):
            total_w = sum(r["n"] for r in group if r[key] == r[key])
            if total_w == 0:
                return float("nan")
            return sum(r[key] * r["n"] for r in group if r[key] == r[key]) / total_w

        print(
            f"{modality:<{col_widths['modality']}}"
            f"{dataset:<{col_widths['dataset']}}"
            f"{n_total:>{7}}"
            f"{weighted_avg('faithfulness_score'):>{col_widths['faithfulness_score']}.4f}"
            f"{weighted_avg('faithfulness_pass_rate')*100:>{col_widths['faithfulness_pass_rate']}.1f}%"
        )

    print("-" * len(agg_header))


if __name__ == "__main__":
    data = collect(EVAL_ROOT)
    rows = summarize(data)
    print_table(rows)
