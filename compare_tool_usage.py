"""
Compare tool usage statistics between two model runs (baseline vs trained).

Usage:
  python compare_tool_usage.py [baseline_results_dir] [trained_results_dir]
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Import from the analysis module
sys.path.insert(0, str(Path(__file__).parent))
from analyze_tool_usage import analyze_results_dir

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def collect_stats(results_dir, prompts_dir=None, tol=0):
    """Run analysis and return (global_bbox_counts, per_qtype_bbox_counts, total_bbox)."""
    results_dir = Path(results_dir)
    if prompts_dir is None:
        candidate = results_dir.parent / 'prompts'
        prompts_dir = candidate if candidate.exists() else None

    global_counts, per_qtype, per_dataset, total, skipped, no_prompt, multi = \
        analyze_results_dir(results_dir, prompts_dir, tol=tol, verbose=False)

    # Strip no_bbox for bbox-only analysis
    def bbox_only(d):
        return {k: v for k, v in d.items() if k != 'no_bbox'}

    global_bbox  = bbox_only(global_counts)
    per_qtype_bb = {q: bbox_only(c) for q, c in per_qtype.items()
                    if any(k != 'no_bbox' for k in c)}

    total_bbox = sum(global_bbox.values())
    return global_bbox, per_qtype_bb, total_bbox


def pct(count, total):
    return count / total * 100 if total > 0 else 0.0


def delta_bar(diff):
    """Visual indicator for change."""
    if abs(diff) < 0.5:
        return '  ≈'
    elif diff > 0:
        return f'  ▲{diff:+.1f}%'
    else:
        return f'  ▼{diff:+.1f}%'


def print_comparison(label_a, label_b, counts_a, counts_b, total_a, total_b, title):
    all_keys = sorted(set(counts_a) | set(counts_b))
    if not any(counts_a.get(k, 0) + counts_b.get(k, 0) > 0 for k in all_keys):
        return

    width = 80
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"  {label_a} (n={total_a})  vs  {label_b} (n={total_b})")
    print(f"{'='*width}")
    header = f"  {'Tool':<35}  {label_a:>8}  {label_b:>8}  {'Δ':>8}"
    print(header)
    print(f"  {'-'*35}  {'--------':>8}  {'--------':>8}  {'--------':>8}")

    for key in sorted(all_keys, key=lambda k: -(counts_a.get(k, 0) + counts_b.get(k, 0))):
        ca = counts_a.get(key, 0)
        cb = counts_b.get(key, 0)
        pa = pct(ca, total_a)
        pb = pct(cb, total_b)
        diff = pb - pa
        marker = delta_bar(diff)
        print(f"  {key:<35}  {pa:>6.1f}%  {pb:>6.1f}%{marker}")


def group_by_tool_family(counts):
    """Merge _expanded/_attention variants into a single tool name."""
    merged = defaultdict(int)
    for k, v in counts.items():
        base = k.replace('_expanded', '').replace('_attention', '')
        merged[base] += v
    return dict(merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare tool usage between two runs')
    parser.add_argument('baseline_results', nargs='?',
        default='/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/baseline_outputs/Qwen30B_VL_multi/results')
    parser.add_argument('trained_results', nargs='?',
        default='/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/baseline_outputs/Qwen30B_VL_multi_trained_30step/results')
    parser.add_argument('--tolerance', '-t', type=int, default=0)
    parser.add_argument('--label-a', default='Baseline')
    parser.add_argument('--label-b', default='Trained-30step')
    args = parser.parse_args()

    print(f"Loading baseline:  {args.baseline_results}")
    g_a, qt_a, tot_a = collect_stats(args.baseline_results, tol=args.tolerance)
    print(f"Loading trained:   {args.trained_results}")
    g_b, qt_b, tot_b = collect_stats(args.trained_results, tol=args.tolerance)

    # ── 1. Global comparison (fine-grained) ────────────────────────────────
    print_comparison(
        args.label_a, args.label_b,
        g_a, g_b, tot_a, tot_b,
        'GLOBAL: All bbox outputs (fine-grained)'
    )

    # ── 2. Global comparison (tool family) ─────────────────────────────────
    g_a_fam = group_by_tool_family(g_a)
    g_b_fam = group_by_tool_family(g_b)
    print_comparison(
        args.label_a, args.label_b,
        g_a_fam, g_b_fam, tot_a, tot_b,
        'GLOBAL: Tool family (expanded+attention merged)'
    )

    # ── 3. Per question type ────────────────────────────────────────────────
    all_qtypes = sorted(set(qt_a) | set(qt_b))
    for qtype in all_qtypes:
        ca = qt_a.get(qtype, {})
        cb = qt_b.get(qtype, {})
        tot_qa = sum(ca.values())
        tot_qb = sum(cb.values())
        if tot_qa + tot_qb == 0:
            continue
        print_comparison(
            args.label_a, args.label_b,
            ca, cb, tot_qa, tot_qb,
            f'QUESTION TYPE: {qtype}'
        )

    # ── 4. Summary: biggest shifts ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY: Largest % shifts (Global, tool-family level, Trained - Baseline)")
    print(f"{'='*80}")
    all_keys = sorted(set(g_a_fam) | set(g_b_fam))
    shifts = []
    for k in all_keys:
        pa = pct(g_a_fam.get(k, 0), tot_a)
        pb = pct(g_b_fam.get(k, 0), tot_b)
        shifts.append((k, pa, pb, pb - pa))
    shifts.sort(key=lambda x: -abs(x[3]))

    print(f"  {'Tool':<35}  {args.label_a:>8}  {args.label_b:>8}  {'Δ':>8}")
    print(f"  {'-'*35}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
    for k, pa, pb, diff in shifts:
        marker = delta_bar(diff)
        print(f"  {k:<35}  {pa:>6.1f}%  {pb:>6.1f}%{marker}")


if __name__ == '__main__':
    main()
