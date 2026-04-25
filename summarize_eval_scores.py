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
    "/openai_outputs/gpt_5-4_nano_tabular/evaluations"
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

                # soft_score / size_score / size_score_l1 live at faithfulness.details.*
                for detail_key in ("soft_score", "size_score", "size_score_l1"):
                    row[detail_key] = avg(
                        [
                            r.get(FAITHFULNESS_KEY, {}).get("details", {}).get(detail_key)
                            if isinstance(r.get(FAITHFULNESS_KEY), dict)
                            else None
                            for r in records
                        ]
                    )

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
        "faithfulness_score": 12,
        "soft_score": 12,
        "size_score": 12,
        "size_score_l1": 14,
        "faithfulness_pass_rate": 12,
    }

    header = (
        f"{'modality':<{col_widths['modality']}}"
        f"{'dataset':<{col_widths['dataset']}}"
        f"{'q_type':<{col_widths['q_type']}}"
        f"{'n':>{col_widths['n']}}"
        f"{'faith_score':>{col_widths['faithfulness_score']}}"
        f"{'soft_score':>{col_widths['soft_score']}}"
        f"{'size_score':>{col_widths['size_score']}}"
        f"{'size_score_l1':>{col_widths['size_score_l1']}}"
        f"{'faith_pass':>{col_widths['faithfulness_pass_rate']}}"
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

        def fmt(val, width):
            return f"{val:>{width}.4f}" if val == val else f"{'nan':>{width}}"

        print(
            f"{row['modality']:<{col_widths['modality']}}"
            f"{row['dataset']:<{col_widths['dataset']}}"
            f"{row['q_type']:<{col_widths['q_type']}}"
            f"{row['n']:>{col_widths['n']}}"
            + fmt(row['faithfulness_score'], col_widths['faithfulness_score'])
            + fmt(row['soft_score'], col_widths['soft_score'])
            + fmt(row['size_score'], col_widths['size_score'])
            + fmt(row['size_score_l1'], col_widths['size_score_l1'])
            + f"{row['faithfulness_pass_rate']*100:>{col_widths['faithfulness_pass_rate']}.1f}%"
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
        f"{'soft_score':>{col_widths['soft_score']}}"
        f"{'size_score':>{col_widths['size_score']}}"
        f"{'size_score_l1':>{col_widths['size_score_l1']}}"
        f"{'faith_pass':>{col_widths['faithfulness_pass_rate']}}"
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
            + fmt(weighted_avg('faithfulness_score'), col_widths['faithfulness_score'])
            + fmt(weighted_avg('soft_score'), col_widths['soft_score'])
            + fmt(weighted_avg('size_score'), col_widths['size_score'])
            + fmt(weighted_avg('size_score_l1'), col_widths['size_score_l1'])
            + f"{weighted_avg('faithfulness_pass_rate')*100:>{col_widths['faithfulness_pass_rate']}.1f}%"
        )

    print("-" * len(agg_header))

    # --- aggregate totals per q_type ---
    print("\n=== Aggregate by q_type ===\n")

    qtype_header = (
        f"{'q_type':<{col_widths['q_type']}}"
        f"{'n_total':>{7}}"
        f"{'faith_score':>{col_widths['faithfulness_score']}}"
        f"{'soft_score':>{col_widths['soft_score']}}"
        f"{'size_score':>{col_widths['size_score']}}"
        f"{'size_score_l1':>{col_widths['size_score_l1']}}"
        f"{'faith_pass':>{col_widths['faithfulness_pass_rate']}}"
    )
    print("-" * len(qtype_header))
    print(qtype_header)
    print("-" * len(qtype_header))

    sorted_by_qtype = sorted(rows, key=lambda r: r["q_type"])
    for q_type, group in groupby(sorted_by_qtype, key=lambda r: r["q_type"]):
        group = list(group)
        n_total = sum(r["n"] for r in group)

        def weighted_avg_q(key):
            valid_rows = [r for r in group if r[key] == r[key]]
            total_n = sum(r["n"] for r in valid_rows)
            if total_n == 0:
                return float("nan")
            return sum(r[key] * r["n"] for r in valid_rows) / total_n

        print(
            f"{q_type:<{col_widths['q_type']}}"
            f"{n_total:>{7}}"
            + fmt(weighted_avg_q('faithfulness_score'), col_widths['faithfulness_score'])
            + fmt(weighted_avg_q('soft_score'), col_widths['soft_score'])
            + fmt(weighted_avg_q('size_score'), col_widths['size_score'])
            + fmt(weighted_avg_q('size_score_l1'), col_widths['size_score_l1'])
            + f"{weighted_avg_q('faithfulness_pass_rate')*100:>{col_widths['faithfulness_pass_rate']}.1f}%"
        )
    print("-" * len(qtype_header))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize evaluation scores.")
    parser.add_argument(
        "--modality",
        default=None,
        help="Filter by modality (e.g. 'vision', 'tabular'). Omit to include all.",
    )
    args = parser.parse_args()

    data = collect(EVAL_ROOT)

    if args.modality is not None:
        if args.modality not in data:
            print(f"No data found for modality '{args.modality}'. "
                  f"Available: {sorted(data.keys())}")
            raise SystemExit(1)
        data = {args.modality: data[args.modality]}

    rows = summarize(data)
    print_table(rows)
