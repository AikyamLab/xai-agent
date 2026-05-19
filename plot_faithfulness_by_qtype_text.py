import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TRIAL2_DIR = Path(__file__).parent.resolve()
MODALITY    = "text"

REF_DIR = "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/trained_outputs/Qwen3.6_text_step700"


def collect_scores(base_dir):
    base = Path(base_dir) / "evaluations" / MODALITY
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


def collect_tool_baseline_q1(modality):
    """Return {tool_name: {"q1": mean_score}}."""
    base = TRIAL2_DIR / "outputs" / "baseline_tool_only_full" / modality
    if not base.exists():
        return {}
    tool_scores = defaultdict(list)
    for tool_dir in sorted(base.iterdir()):
        if not tool_dir.is_dir():
            continue
        for result_file in tool_dir.glob("*.json"):
            try:
                data = json.load(open(result_file))
                results = data.get("results", [])
                scores_n = [r["score"] for r in results if r.get("score") is not None]
                if scores_n:
                    tool_scores[tool_dir.name].append(sum(scores_n) / len(scores_n))
            except Exception:
                pass
    result = {}
    for tool, vals in tool_scores.items():
        if vals:
            result[tool] = {"q1": sum(vals) / len(vals)}
    return result


paths = {
    "Base Qwen3.6 35B": "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/baseline_outputs/Qwen3.6_text_noImage",
    "GPT 5.4 nano": "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/openai_outputs/gpt_5-4_nano_text",
    "Trained Qwen 30B": "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/trained_outputs/Qwen30B_VL_text_mixed",
    "Trained Qwen3.6 35B": "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/trained_outputs/Qwen3.6_text_noimage"
}

model_scores = {name: collect_scores(path) for name, path in paths.items()}
tool_scores   = collect_tool_baseline_q1(MODALITY)

# Tools appear first (leftmost at Q1), then models
all_scores = {**tool_scores, **model_scores}

qtypes     = [f"q{i}" for i in range(1, 11)]
labels     = [f"Q{i}" for i in range(1, 11)]
tool_list  = list(tool_scores.keys())
model_list = list(model_scores.keys())
n_t, n_m   = len(tool_list), len(model_list)

_model_palette = ['#4C72B0', '#DD8452', '#55A868', '#F4B98B']
_tool_palette  = ['#AECDE8', '#F4B98A', '#A7D5B1', '#E8A0A0', '#C3BAE4', '#C9B8A8']

# ── Layout constants ──────────────────────────────────────────────────────────
BAR_W       = 0.14   # each bar width
BAR_GAP     = 0.02   # gap between adjacent bars within a section
SECTION_GAP = 0.10   # gap between tool section and model section at Q1
GROUP_GAP   = 0.28   # gap between Q groups

tool_sec_w  = n_t * BAR_W + max(n_t - 1, 0) * BAR_GAP
model_sec_w = n_m * BAR_W + max(n_m - 1, 0) * BAR_GAP
q1_w        = tool_sec_w + SECTION_GAP + model_sec_w

# Q-group centers: Q1 is wider, Q2-Q10 are model-section width
q_centers = np.zeros(10)
for i in range(1, 10):
    prev_half = q1_w / 2 if i == 1 else model_sec_w / 2
    q_centers[i] = q_centers[i - 1] + prev_half + GROUP_GAP + model_sec_w / 2

# Per-entry offsets relative to group center
q1_left      = -q1_w / 2
tool_off     = {t: q1_left + j * (BAR_W + BAR_GAP) + BAR_W / 2
                for j, t in enumerate(tool_list)}
model_off_q1 = {m: q1_left + tool_sec_w + SECTION_GAP + j * (BAR_W + BAR_GAP) + BAR_W / 2
                for j, m in enumerate(model_list)}
model_off    = {m: -model_sec_w / 2 + j * (BAR_W + BAR_GAP) + BAR_W / 2
                for j, m in enumerate(model_list)}

fig, ax = plt.subplots(figsize=(16, 6))

# ── Tool bars (Q1 only) ───────────────────────────────────────────────────────
for j, name in enumerate(tool_list):
    yp = all_scores[name].get('q1', float('nan'))
    if np.isnan(yp):
        continue
    xp    = q_centers[0] + tool_off[name]
    color = _tool_palette[j % len(_tool_palette)]
    ax.bar(xp, yp, BAR_W, label=name, color=color,
           alpha=0.85, edgecolor='grey', linewidth=0.5, hatch='//')
    ax.text(xp, yp + 0.012, f'{yp:.3f}', ha='center', va='bottom',
            fontsize=6, color='#555555')

# ── Model bars (all Q types) ──────────────────────────────────────────────────
for j, name in enumerate(model_list):
    color   = _model_palette[j % len(_model_palette)]
    xp_list = [q_centers[qi] + (model_off_q1[name] if qi == 0 else model_off[name])
               for qi in range(10)]
    yp_list = [all_scores[name].get(q, float('nan')) for q in qtypes]
    valid   = [(xp, yp) for xp, yp in zip(xp_list, yp_list) if not np.isnan(yp)]
    if not valid:
        continue
    vx, vy = zip(*valid)
    ax.bar(vx, vy, BAR_W, label=name, color=color, alpha=0.85,
           edgecolor='white', linewidth=0.5)
    for xp, yp in valid:
        ax.text(xp, yp + 0.012, f'{yp:.3f}', ha='center', va='bottom',
                fontsize=6, color='#333333')

ax.set_xlabel('Question Type', fontsize=12)
ax.set_ylabel('Mean Faithfulness Score (Soft Score)', fontsize=12)
ax.set_title('Text Faithfulness Score by Question Type\n(Qwen30B-VL: Baseline vs GRPO-Trained)',
             fontsize=13, fontweight='bold')
ax.set_xticks(q_centers)
ax.set_xticklabels(labels, fontsize=11)
ax.set_xlim(q_centers[0] - q1_w / 2 - 0.2,
            q_centers[-1] + model_sec_w / 2 + 0.2)
ax.set_ylim(0, 1.20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
ax.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = '/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/faithfulness_text_mixed.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
