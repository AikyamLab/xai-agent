"""
Plot DPO training curves from metrics.jsonl.

Usage:
    # Latest run (auto-detect):
    python training/dpo/plot_metrics.py

    # Specific run directory or metrics file:
    python training/dpo/plot_metrics.py training/dpo/runs/dpo_Qwen3-VL-30B_1234567890/
    python training/dpo/plot_metrics.py training/dpo/runs/dpo_Qwen3-VL-30B_1234567890/metrics.jsonl

Output: training_curves.png saved next to metrics.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = BASE_DIR / "training/dpo/runs"


def load_metrics(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def smooth(values: list[float], window: int = 20) -> list[float]:
    """Simple moving average."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(float(np.mean(values[lo:hi])))
    return result


def find_latest_metrics() -> Path:
    runs = sorted(RUNS_DIR.glob("*/metrics.jsonl"), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"No metrics.jsonl found under {RUNS_DIR}")
    return runs[-1]


def plot(metrics_path: Path):
    records = load_metrics(metrics_path)
    if not records:
        sys.exit("metrics.jsonl is empty")

    steps       = [r["step"]              for r in records]
    train_loss  = [r.get("dpo_loss", None)   for r in records]
    train_margin = [r.get("margin", None)    for r in records]
    train_acc   = [r.get("accuracy", None)   for r in records]

    # Eval points (subset of steps where eval was run)
    eval_steps   = [r["step"] for r in records if "eval_margin" in r]
    eval_margin  = [r["eval_margin"]   for r in records if "eval_margin" in r]
    eval_acc     = [r["eval_accuracy"] for r in records if "eval_accuracy" in r]

    # Drop None
    def clean(xs, ref):
        pairs = [(s, v) for s, v in zip(ref, xs) if v is not None]
        return ([p[0] for p in pairs], [p[1] for p in pairs])

    s_loss,   v_loss   = clean(train_loss,   steps)
    s_margin, v_margin = clean(train_margin, steps)
    s_acc,    v_acc    = clean(train_acc,    steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"DPO Training — {metrics_path.parent.name}", fontsize=13, fontweight="bold")

    colors = {"train": "#4C72B0", "smooth": "#DD4444", "eval": "#2CA02C"}

    # ── Panel 1: Loss ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(s_loss, v_loss, color=colors["train"], alpha=0.3, linewidth=0.8, label="train (raw)")
    ax.plot(s_loss, smooth(v_loss), color=colors["smooth"], linewidth=1.8, label="train (smooth)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("DPO Loss")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Margin ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(s_margin, v_margin, color=colors["train"], alpha=0.3, linewidth=0.8, label="train (raw)")
    ax.plot(s_margin, smooth(v_margin), color=colors["smooth"], linewidth=1.8, label="train (smooth)")
    if eval_steps:
        ax.scatter(eval_steps, eval_margin, color=colors["eval"], zorder=5,
                   s=40, label="eval", marker="D")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="margin=0")
    ax.set_ylabel("Margin (chosen − rejected log-prob)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Accuracy ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(s_acc, v_acc, color=colors["train"], alpha=0.3, linewidth=0.8, label="train (raw)")
    ax.plot(s_acc, smooth(v_acc), color=colors["smooth"], linewidth=1.8, label="train (smooth)")
    if eval_steps:
        ax.scatter(eval_steps, eval_acc, color=colors["eval"], zorder=5,
                   s=40, label="eval", marker="D")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="random baseline")
    ax.set_ylabel("Accuracy (chosen > rejected)")
    ax.set_xlabel("Step")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = metrics_path.parent / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print summary stats
    if v_loss:
        last_n = min(50, len(v_loss))
        print(f"\nLast {last_n} steps summary:")
        print(f"  loss   mean={np.mean(v_loss[-last_n:]):.4f}")
        print(f"  margin mean={np.mean(v_margin[-last_n:]) if v_margin else 'N/A':.4f}")
        print(f"  acc    mean={np.mean(v_acc[-last_n:]) if v_acc else 'N/A':.4f}")
    if eval_steps:
        print(f"\nLatest eval (step {eval_steps[-1]}):")
        print(f"  eval_margin   = {eval_margin[-1]:.4f}")
        print(f"  eval_accuracy = {eval_acc[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Run dir or metrics.jsonl path")
    args = parser.parse_args()

    if args.path is None:
        metrics_path = find_latest_metrics()
    else:
        p = Path(args.path)
        metrics_path = p if p.suffix == ".jsonl" else p / "metrics.jsonl"

    if not metrics_path.exists():
        sys.exit(f"Not found: {metrics_path}")

    print(f"Loading: {metrics_path}")
    plot(metrics_path)


if __name__ == "__main__":
    main()
