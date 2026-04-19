"""
GRPO Training Visualization
Usage: python visualize_grpo.py
Saves: grpo_training_curves.png  (in the script's directory)
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

RUNS = {
    "run_full_reward": Path(
        "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2"
        "/checkpoints/grpo_text_qwen30b_full_reward_fromTab_1/grpo_log.jsonl"
    ),
}

COLORS = {
    "run_full_reward":        "#E07B39",   # orange
}

SMOOTH_W = 5   # rolling average window


# ── Load data ─────────────────────────────────────────────────────────────────

def _parse_line(line: str):
    """Parse one or more concatenated JSON objects from a single line."""
    decoder = json.JSONDecoder()
    line = line.strip()
    pos = 0
    objects = []
    while pos < len(line):
        obj, idx = decoder.raw_decode(line, pos)
        objects.append(obj)
        pos = idx
        while pos < len(line) and line[pos] in " \t":
            pos += 1
    return objects


def load(path: Path):
    rows = [obj for l in path.open() if l.strip() for obj in _parse_line(l)]
    train = [r for r in rows if r["type"] == "train"]
    evals = [r for r in rows if r["type"] == "eval"]
    ckpts = [r for r in rows if r["type"] == "checkpoint"]
    return train, evals, ckpts


def smooth(vals, w):
    if w <= 1 or len(vals) < w:
        return vals
    return np.convolve(vals, np.ones(w) / w, mode="valid")


def smooth_x(xs, w):
    """Corresponding x-axis after convolution (centred)."""
    if w <= 1 or len(xs) < w:
        return xs
    return xs[w // 2: len(xs) - (w - 1 - w // 2)]


# ── Epoch boundary helpers ─────────────────────────────────────────────────────

def epoch_boundaries(train):
    """Return step indices where epoch number increases."""
    bounds = []
    prev = None
    for r in train:
        e = r.get("epoch")
        if prev is not None and e != prev:
            bounds.append(r["step"])
        prev = e
    return bounds


# ── Plotting ──────────────────────────────────────────────────────────────────

def add_epoch_lines(ax, bounds, color, alpha=0.25):
    for b in bounds:
        ax.axvline(b, color=color, lw=1, ls="--", alpha=alpha)


all_data = {name: load(path) for name, path in RUNS.items()}

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("GRPO Training Curves", fontsize=16, fontweight="bold", y=0.98)
plt.subplots_adjust(hspace=0.45, wspace=0.32)

ax = axes.flatten()


# ── 1. Train mean reward ──────────────────────────────────────────────────────
a = ax[0]
a.set_title("Train Mean Reward", fontweight="bold")
for name, (train, evals, _) in all_data.items():
    xs  = np.array([r["step"] for r in train])
    ys  = np.array([r["mean_reward"] for r in train])
    col = COLORS[name]
    a.plot(xs, ys, alpha=0.25, color=col, lw=1)
    sxs = smooth_x(xs, SMOOTH_W)
    sys = smooth(ys, SMOOTH_W)
    a.plot(sxs, sys, color=col, lw=2, label=name)
    add_epoch_lines(a, epoch_boundaries(train), col)
a.set_xlabel("Step"); a.set_ylabel("Reward")
a.legend(fontsize=7, loc="lower right")
a.set_ylim(bottom=0)
a.grid(alpha=0.3)


# ── 2. Eval mean reward ───────────────────────────────────────────────────────
a = ax[1]
a.set_title("Eval Mean Reward", fontweight="bold")
for name, (_, evals, _) in all_data.items():
    if not evals:
        continue
    xs  = [r["step"] for r in evals]
    ys  = [r["eval_mean_reward"] for r in evals]
    std = [r.get("eval_std_reward", 0) for r in evals]
    col = COLORS[name]
    a.errorbar(xs, ys, yerr=std, fmt="o-", color=col, lw=2,
               capsize=4, markersize=5, label=name)
a.set_xlabel("Step"); a.set_ylabel("Reward")
a.legend(fontsize=7)
a.set_ylim(bottom=0)
a.grid(alpha=0.3)


# ── 3. Eval reward by Q-type (run3 only, more data) ──────────────────────────
a = ax[2]
a.set_title("Eval Reward by Q-type (run3)", fontweight="bold")
name3 = "run3 (resumed from ckpt-40)"
evals3 = all_data[name3][1] if name3 in all_data else []
if evals3:
    qtypes = sorted(set(
        int(k.split("_")[1])
        for e in evals3
        for k in e
        if k.startswith("qtype_") and k.endswith("_mean_reward")
    ))
    cmap = plt.cm.tab10
    for i, qt in enumerate(qtypes):
        key = f"qtype_{qt}_mean_reward"
        xs  = [e["step"] for e in evals3 if key in e]
        ys  = [e[key]    for e in evals3 if key in e]
        a.plot(xs, ys, "o-", color=cmap(i), lw=1.5, markersize=4, label=f"Q{qt}")
a.set_xlabel("Step"); a.set_ylabel("Reward")
a.legend(fontsize=7, ncol=2)
a.set_ylim(bottom=0)
a.grid(alpha=0.3)


# ── 4. Zero reward fraction ───────────────────────────────────────────────────
a = ax[3]
a.set_title("Zero-Reward Fraction", fontweight="bold")
for name, (train, _, _) in all_data.items():
    xs  = np.array([r["step"] for r in train])
    ys  = np.array([r["zero_reward_frac"] for r in train])
    col = COLORS[name]
    a.plot(xs, ys, alpha=0.25, color=col, lw=1)
    a.plot(smooth_x(xs, SMOOTH_W), smooth(ys, SMOOTH_W), color=col, lw=2, label=name)
    add_epoch_lines(a, epoch_boundaries(train), col)
a.set_xlabel("Step"); a.set_ylabel("Fraction")
a.legend(fontsize=7)
a.set_ylim(0, 1)
a.grid(alpha=0.3)


# ── 5. Entropy (−mean logprob) ────────────────────────────────────────────────
a = ax[4]
a.set_title("Policy Entropy Approx (−mean logprob)", fontweight="bold")
for name, (train, _, _) in all_data.items():
    xs  = np.array([r["step"] for r in train])
    ys  = np.array([r.get("entropy_approx", float("nan")) for r in train])
    col = COLORS[name]
    a.plot(xs, ys, alpha=0.25, color=col, lw=1)
    a.plot(smooth_x(xs, SMOOTH_W), smooth(ys, SMOOTH_W), color=col, lw=2, label=name)
    add_epoch_lines(a, epoch_boundaries(train), col)
a.set_xlabel("Step"); a.set_ylabel("Entropy proxy")
a.legend(fontsize=7)
a.grid(alpha=0.3)


# ── 6. Reward std (within batch) ──────────────────────────────────────────────
a = ax[5]
a.set_title("Reward Std (within batch)", fontweight="bold")
for name, (train, _, _) in all_data.items():
    xs  = np.array([r["step"] for r in train])
    ys  = np.array([r["reward_std"] for r in train])
    col = COLORS[name]
    a.plot(xs, ys, alpha=0.25, color=col, lw=1)
    a.plot(smooth_x(xs, SMOOTH_W), smooth(ys, SMOOTH_W), color=col, lw=2, label=name)
    add_epoch_lines(a, epoch_boundaries(train), col)
a.set_xlabel("Step"); a.set_ylabel("Std")
a.legend(fontsize=7)
a.grid(alpha=0.3)


# ── 7. Forward loss ───────────────────────────────────────────────────────────
a = ax[6]
a.set_title("Forward Loss (fwd_loss:sum)", fontweight="bold")
for name, (train, _, _) in all_data.items():
    xs  = np.array([r["step"] for r in train])
    ys  = np.array([r.get("fwd_loss:sum", float("nan")) for r in train])
    col = COLORS[name]
    a.plot(xs, ys, alpha=0.3, color=col, lw=1)
    a.plot(smooth_x(xs, SMOOTH_W), smooth(ys, SMOOTH_W), color=col, lw=2, label=name)
    add_epoch_lines(a, epoch_boundaries(train), col)
a.set_xlabel("Step"); a.set_ylabel("Loss")
a.legend(fontsize=7)
a.grid(alpha=0.3)


# ── 8. Mean action tokens ─────────────────────────────────────────────────────
a = ax[7]
a.set_title("Mean Action Tokens per Turn", fontweight="bold")
for name, (train, _, _) in all_data.items():
    xs  = np.array([r["step"] for r in train])
    ys  = np.array([r["mean_action_tokens"] for r in train])
    col = COLORS[name]
    a.plot(xs, ys, alpha=0.25, color=col, lw=1)
    a.plot(smooth_x(xs, SMOOTH_W), smooth(ys, SMOOTH_W), color=col, lw=2, label=name)
    add_epoch_lines(a, epoch_boundaries(train), col)
a.set_xlabel("Step"); a.set_ylabel("Tokens")
a.legend(fontsize=7)
a.grid(alpha=0.3)


# ── 9. Step wall time ─────────────────────────────────────────────────────────
a = ax[8]
a.set_title("Step Wall Time (elapsed_s)", fontweight="bold")
for name, (train, _, _) in all_data.items():
    xs  = np.array([r["step"] for r in train])
    ys  = np.array([r["elapsed_s"] for r in train])
    col = COLORS[name]
    a.scatter(xs, ys, s=18, color=col, alpha=0.6, label=name)
    a.axhline(np.median(ys), color=col, lw=1.5, ls="--",
              alpha=0.8, label=f"{name} median={np.median(ys):.0f}s")
a.set_xlabel("Step"); a.set_ylabel("Seconds")
a.legend(fontsize=6)
a.grid(alpha=0.3)


# ── Legend patches for epoch lines ────────────────────────────────────────────
for a_ in ax:
    ylim = a_.get_ylim()
    if ylim[1] > ylim[0]:
        a_.set_ylim(ylim)   # keep stable

fig.text(0.5, 0.01,
         "Dashed vertical lines = epoch boundaries  |  Shaded area = raw values, solid line = rolling avg (w=5)",
         ha="center", fontsize=8, color="#555555")

out = Path(__file__).parent / "grpo_text_training_curves_full_reward_fromTab_1.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
