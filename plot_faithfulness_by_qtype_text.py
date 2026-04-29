import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def collect_scores(base_dir):
    base = Path(base_dir) / "evaluations" / "text"
    if not base.exists():
        print(f"  NOT FOUND: {base}")
        return {}
    scores = defaultdict(list)
    for dataset in base.iterdir():
        if not dataset.is_dir():
            continue
        for qtype in dataset.iterdir():
            if not qtype.is_dir():
                continue
            q = qtype.name
            for sample in qtype.iterdir():
                eval_file = sample / "evaluation.json"
                if not eval_file.exists():
                    continue
                try:
                    with open(eval_file) as f:
                        data = json.load(f)
                    soft = data.get("faithfulness", {}).get("details", {}).get("soft_score")
                    if soft is not None:
                        scores[q].append(float(soft))
                except Exception:
                    pass
    return {q: sum(v) / len(v) for q, v in scores.items() if v}

paths = {
    "Base Qwen30B VL": "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/baseline_outputs/Qwen30B_VL_text",
    "Trained Qwen30B VL": "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/trained_outputs/Qwen30B_VL_text_full_reward_2_fromTab",
}

all_scores = {name: collect_scores(path) for name, path in paths.items()}

qtypes = [f"q{i}" for i in range(1, 11)]
labels = [f"Q{i}" for i in range(1, 11)]
model_names = list(all_scores.keys())

_palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']
colors = [_palette[i % len(_palette)] for i in range(len(model_names))]

x = np.arange(len(qtypes))
n_models = len(model_names)
width = min(0.8 / n_models, 0.25)
offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

fig, ax = plt.subplots(figsize=(16, 7))

for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [all_scores[name].get(q, float('nan')) for q in qtypes]
    bars = ax.bar(x + offsets[i], vals, width, label=name, color=color, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6.5,
                    color='#333333')

ax.set_xlabel('Question Type', fontsize=12)
ax.set_ylabel('Mean Faithfulness Score (Soft Score)', fontsize=12)
ax.set_title('Tabular Faithfulness Score by Question Type\n(Qwen30B-VL: Baseline vs GRPO-Trained)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = '/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/faithfulness_text_baseline.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
