#!/usr/bin/env python3
"""Collate faithfulness scores from baseline result.json files.

Recursively finds all result.json files under outputs_baselines/ and
reports score distributions by baseline, dataset, question type, etc.

Usage:
    python collate_scores.py                          # default: outputs_baselines/
    python collate_scores.py /path/to/outputs_dir     # custom root
"""

import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def find_result_files(root: Path):
    """Recursively find all result.json files."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f == "result.json":
                yield Path(dirpath) / f


def parse_result(filepath: Path) -> dict | None:
    """Parse a single result.json and return a normalised record."""
    try:
        data = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠ Could not read {filepath}: {e}")
        return None

    # --- Extract baseline name ---
    baseline = data.get("baseline", "")

    # --- Extract question type ---
    q_type_raw = data.get("question_type")
    if q_type_raw is not None:
        q_type = f"q{q_type_raw}"
    else:
        qid = data.get("question_id", "")
        q_type = qid if qid.startswith("q") else ""

    # --- Extract dataset from the filepath ---
    # Typical: .../naive/tabular/adult_2layernn/q1/5/naive/tabular/adult_2layernn/q1/4618/result.json
    rel = str(filepath)
    dataset = ""
    modality = ""
    mod_re = re.search(r"/(vision|tabular|text)/([^/]+)/q\d+", rel)
    if mod_re:
        modality = mod_re.group(1)
        dataset = mod_re.group(2)

    # --- Extract faithfulness evaluation ---
    evaluation = data.get("evaluation", {})
    faith = evaluation.get("faithfulness", {})

    score = faith.get("score")
    passed = faith.get("passed")
    metric_name = faith.get("metric_name", "")
    p_original = faith.get("p_original")
    p_modified = faith.get("p_modified")
    original_class = faith.get("original_class")
    modified_class = faith.get("modified_class")
    errors = faith.get("errors", [])

    explanation = data.get("explanation", "")
    if explanation and len(explanation) > 100:
        explanation = explanation[:100] + "..."

    return {
        "filepath": str(filepath),
        "baseline": baseline,
        "dataset": dataset,
        "modality": modality,
        "q_type": q_type,
        "score": score,
        "passed": passed,
        "metric_name": metric_name,
        "p_original": p_original,
        "p_modified": p_modified,
        "original_class": original_class,
        "modified_class": modified_class,
        "errors": errors,
        "explanation": explanation,
    }


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def stats_line(scores, n_passed, label, label_width=10):
    """Format a stats row."""
    n = len(scores)
    if n == 0:
        return f"{label:<{label_width}} {'—':>5}"
    mean = statistics.mean(scores)
    med = statistics.median(scores)
    sd = statistics.stdev(scores) if n > 1 else 0.0
    mn = min(scores)
    mx = max(scores)
    pp = pct(n_passed, n)
    return (f"{label:<{label_width}} {n:>5} {mean:>8.4f} {med:>8.4f} "
            f"{sd:>8.4f} {mn:>8.4f} {mx:>8.4f} {pp:>7.1f}%")


def bucket_row(label, scores, label_width=10):
    """Format a distribution-bucket row."""
    z  = sum(1 for s in scores if s == 0.0)
    l  = sum(1 for s in scores if 0.0 < s <= 0.1)
    ml = sum(1 for s in scores if 0.1 < s <= 0.3)
    mh = sum(1 for s in scores if 0.3 < s <= 0.5)
    hl = sum(1 for s in scores if 0.5 < s <= 0.7)
    h  = sum(1 for s in scores if 0.7 < s <= 1.0)
    return f"{label:<{label_width}} {z:>6} {l:>7} {ml:>8} {mh:>8} {hl:>8} {h:>7} {len(scores):>6}"


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs_baselines")
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    print(f"Scanning {root} for result.json files...\n")
    results = []
    for fp in find_result_files(root):
        rec = parse_result(fp)
        if rec:
            results.append(rec)

    total = len(results)
    scored = [r for r in results if r["score"] is not None]
    no_score = [r for r in results if r["score"] is None]
    with_errors = [r for r in results if r["errors"]]

    print(f"Total result.json files found: {total}")
    print(f"  With faithfulness score:     {len(scored)}")
    print(f"  Without score (no eval):     {len(no_score)}")
    print(f"  With evaluation errors:      {len(with_errors)}")

    if with_errors:
        print(f"\n  Error summary:")
        err_types = defaultdict(int)
        for r in with_errors:
            for e in r["errors"]:
                err_types[e] += 1
        for err, cnt in sorted(err_types.items(), key=lambda x: -x[1])[:10]:
            print(f"    ({cnt}x) {err[:120]}")

    if no_score:
        print(f"\n  No-score files (first 5):")
        for r in no_score[:5]:
            print(f"    {r['baseline']}  {r['dataset']}  {r['q_type']}  → {r['filepath']}")

    if not scored:
        print("\nNo faithfulness scores found!")
        return

    # Pre-build groupings
    by_bl = defaultdict(list)
    passed_bl = defaultdict(int)
    by_mod = defaultdict(list)
    passed_mod = defaultdict(int)
    by_ds = defaultdict(list)
    passed_ds = defaultdict(int)
    by_qt = defaultdict(list)
    passed_qt = defaultdict(int)

    for r in scored:
        by_bl[r["baseline"]].append(r["score"])
        by_mod[r["modality"]].append(r["score"])
        by_ds[r["dataset"]].append(r["score"])
        by_qt[r["q_type"]].append(r["score"])
        if r["passed"]:
            passed_bl[r["baseline"]] += 1
            passed_mod[r["modality"]] += 1
            passed_ds[r["dataset"]] += 1
            passed_qt[r["q_type"]] += 1

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 1: FAITHFULNESS BY BASELINE")
    print("=" * 80)
    hdr = f"{'Baseline':<10} {'N':>5} {'Mean':>8} {'Median':>8} {'Stdev':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for bl in ["naive", "cot", "react", "tot"]:
        if by_bl.get(bl):
            print(stats_line(by_bl[bl], passed_bl[bl], bl))
    for bl in sorted(by_bl):
        if bl not in ("naive", "cot", "react", "tot") and by_bl[bl]:
            print(stats_line(by_bl[bl], passed_bl[bl], bl))

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 2: FAITHFULNESS BY MODALITY")
    print("=" * 80)
    hdr = f"{'Modality':<10} {'N':>5} {'Mean':>8} {'Median':>8} {'Stdev':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for mod in sorted(by_mod):
        print(stats_line(by_mod[mod], passed_mod[mod], mod or "(unknown)"))

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 3: FAITHFULNESS BY DATASET")
    print("=" * 80)
    hdr = f"{'Dataset':<25} {'N':>5} {'Mean':>8} {'Median':>8} {'Stdev':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for ds in sorted(by_ds):
        print(stats_line(by_ds[ds], passed_ds[ds], ds or "(unknown)", label_width=25))

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 4: FAITHFULNESS BY QUESTION TYPE")
    print("=" * 80)
    hdr = f"{'Q_Type':<8} {'N':>5} {'Mean':>8} {'Median':>8} {'Stdev':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for qt in sorted(by_qt):
        print(stats_line(by_qt[qt], passed_qt[qt], qt or "?", label_width=8))

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 5: FAITHFULNESS BY BASELINE × DATASET")
    print("=" * 80)
    by_bd = defaultdict(list)
    passed_bd = defaultdict(int)
    for r in scored:
        key = (r["baseline"], r["dataset"])
        by_bd[key].append(r["score"])
        if r["passed"]:
            passed_bd[key] += 1

    hdr = f"{'Baseline':<10} {'Dataset':<25} {'N':>5} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for (bl, ds) in sorted(by_bd):
        sc = by_bd[(bl, ds)]
        ps = passed_bd[(bl, ds)]
        n = len(sc)
        print(f"{bl:<10} {ds:<25} {n:>5} {statistics.mean(sc):>8.4f} {statistics.median(sc):>8.4f} "
              f"{min(sc):>8.4f} {max(sc):>8.4f} {pct(ps, n):>7.1f}%")

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 6: FAITHFULNESS BY BASELINE × Q_TYPE")
    print("=" * 80)
    by_bq = defaultdict(list)
    passed_bq = defaultdict(int)
    for r in scored:
        key = (r["baseline"], r["q_type"])
        by_bq[key].append(r["score"])
        if r["passed"]:
            passed_bq[key] += 1

    hdr = f"{'Baseline':<10} {'Q_Type':<8} {'N':>5} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for (bl, qt) in sorted(by_bq):
        sc = by_bq[(bl, qt)]
        ps = passed_bq[(bl, qt)]
        n = len(sc)
        print(f"{bl:<10} {qt:<8} {n:>5} {statistics.mean(sc):>8.4f} {statistics.median(sc):>8.4f} "
              f"{min(sc):>8.4f} {max(sc):>8.4f} {pct(ps, n):>7.1f}%")

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 7: SCORE DISTRIBUTION BUCKETS BY BASELINE")
    print("=" * 80)
    hdr = f"{'Baseline':<10} {'=0':>6} {'(0,.1]':>7} {'(.1,.3]':>8} {'(.3,.5]':>8} {'(.5,.7]':>8} {'(.7,1]':>7} {'Total':>6}"
    print(hdr)
    print("-" * len(hdr))
    for bl in ["naive", "cot", "react", "tot"]:
        if by_bl.get(bl):
            print(bucket_row(bl, by_bl[bl]))

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 8: SCORE DISTRIBUTION BUCKETS BY Q_TYPE")
    print("=" * 80)
    hdr = f"{'Q_Type':<8} {'=0':>6} {'(0,.1]':>7} {'(.1,.3]':>8} {'(.3,.5]':>8} {'(.5,.7]':>8} {'(.7,1]':>7} {'Total':>6}"
    print(hdr)
    print("-" * len(hdr))
    for qt in sorted(by_qt):
        print(bucket_row(qt, by_qt[qt], label_width=8))

    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 9: FULL DETAIL (BASELINE × DATASET × Q_TYPE)")
    print("=" * 80)
    by_full = defaultdict(list)
    for r in scored:
        by_full[(r["baseline"], r["dataset"], r["q_type"])].append(r)

    hdr = f"{'Baseline':<10} {'Dataset':<25} {'QType':<6} {'N':>4} {'Mean':>8} {'Median':>8} {'Pass':>6}"
    print(hdr)
    print("-" * len(hdr))
    prev_bl = None
    for (bl, ds, qt) in sorted(by_full):
        if bl != prev_bl and prev_bl is not None:
            print()
        prev_bl = bl
        rs = by_full[(bl, ds, qt)]
        sc = [r["score"] for r in rs]
        ps = sum(1 for r in rs if r["passed"])
        print(f"{bl:<10} {ds:<25} {qt:<6} {len(sc):>4} {statistics.mean(sc):>8.4f} {statistics.median(sc):>8.4f} {ps:>3}/{len(sc)}")

    # ================================================================
    # Export CSV for further analysis
    csv_path = root / "faithfulness_summary.csv"
    with open(csv_path, "w") as f:
        f.write("baseline,modality,dataset,q_type,score,passed,p_original,p_modified,original_class,modified_class,filepath\n")
        for r in scored:
            f.write(f"{r['baseline']},{r['modality']},{r['dataset']},{r['q_type']},"
                    f"{r['score']},{r['passed']},{r['p_original']},{r['p_modified']},"
                    f"{r['original_class']},{r['modified_class']},{r['filepath']}\n")
    print(f"\n✓ Detailed CSV exported to: {csv_path}")
    print(f"  ({len(scored)} scored rows)")


if __name__ == "__main__":
    main()
