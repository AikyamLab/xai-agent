"""
Prepare trajectory-level DPO data for XAI ProposerAgent + ActorAgent (same model).

Produces two kinds of pairs, mixed into one JSONL:

  PROPOSER pairs  (role="proposer"):
    prompt          = base_prompt (PromptBuilder output, no reflection)
    chosen          = full trajectory: original strategy + execution
                      → reflection (proposer_reflection + actor_reflection)
                      → improved strategy + execution + faithfulness      [Type B]
                    OR original strategy + execution + faithfulness        [Type A, no reflection]
    rejected        = full trajectory (same structure but degraded)        [Type C]

  ACTOR pairs  (role="actor"):
    prompt          = base actor prompt (PromptBuilder, original strategy)
    chosen          = full trajectory: original explanation
                      → critic's feedback (actor_reflection)
                      → improved explanation + faithfulness                [Type B]
                    OR original explanation + faithfulness                 [Type A, no reflection]
    rejected        = full trajectory (same structure but degraded)        [Type C]

Both use cross-instance B×C and A×C pairing matched by q_type (same modality).

Quality tiers:
  high   delta>0.3 AND imp_faith>0.5  → weight=2.0
  medium 0.1<delta≤0.3                → weight=1.0
  a      passed 1st attempt (no improved) → weight=1.0
"""

import inspect
import json
import glob
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR         = Path(__file__).resolve().parents[2]
DATA_DIR         = BASE_DIR / "outputs/training_data"
TOOL_OUTPUTS_DIR = BASE_DIR / "outputs/tool_outputs"
RESULTS_DIR      = BASE_DIR / "outputs/results"
OUT_DIR          = Path(__file__).resolve().parent / "data"

# PromptBuilders live in BASE_DIR/prompts
sys.path.insert(0, str(BASE_DIR))
from prompts import get_prompt_builder  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────────────

# Minimal system prompt — full task description lives in the user message
# (produced by PromptBuilder), matching how ProposerAgent calls invoke_vlm_for_json.
SYSTEM_PROMPT = "You are a helpful AI assistant."

CANONICAL_STRATEGY_KEYS = [
    "strategy_type", "reasoning", "confidence",
    "autonomous_tasks", "tool_selection",
]

QUALITY_WEIGHTS = {"high": 2.0, "medium": 1.0, "a": 1.0}

# Each Type C sample can be used as 'rejected' at most this many times.
# Prevents a single rare C record from dominating the dataset (e.g. Q2/Q5 have only 1 C).
MAX_C_REUSE = 20


# ── datapoint_id helpers ────────────────────────────────────────────────────────

def extract_model_from_id(datapoint_id: str) -> str:
    m = re.match(r"^(.+)_q\d+_", datapoint_id)
    return m.group(1).rsplit("_", 1)[-1] if m else datapoint_id

def extract_dataset_model_from_id(datapoint_id: str) -> str:
    """Return dataset_model prefix, e.g. 'cancer_tabnn' from 'cancer_tabnn_q2_84'."""
    m = re.match(r"^(.+)_q\d+_", datapoint_id)
    return m.group(1) if m else datapoint_id

def extract_row_no_from_id(datapoint_id: str) -> str:
    """Return numeric row_no, e.g. '84' from 'cancer_tabnn_q2_84_20260219_030550'."""
    m = re.search(r"_q\d+_(\d+)", datapoint_id)
    return m.group(1) if m else ""


def extract_qtype_from_id(datapoint_id: str) -> int:
    m = re.search(r"_q(\d+)_", datapoint_id)
    return int(m.group(1)) if m else 0


# ── Strategy helpers ────────────────────────────────────────────────────────────

def clean_strategy(strategy: dict) -> dict:
    return {k: v for k, v in strategy.items() if k in CANONICAL_STRATEGY_KEYS}


def extract_tools_from_strategy(strategy: dict) -> list:
    """Get selected_tools list from canonical strategy."""
    tool_sel = strategy.get("tool_selection", {})
    if isinstance(tool_sel, dict):
        return tool_sel.get("selected_tools", [])
    return []


# ── Prompt context reconstruction ───────────────────────────────────────────────

def get_flip_target_class(question: dict, pred_label: str) -> str:
    """For Q5-Q7: derive flip target from question text, fallback to binary opposite."""
    q_text = question.get("question", question.get("example", ""))
    m = re.search(r"flip.*?into\s+(\w+)", q_text, re.IGNORECASE)
    if m:
        return m.group(1)
    opposites = {
        "malignant": "benign", "benign": "malignant",
        ">50k": "<=50k", "<=50k": ">50k",
        "positive": "negative", "negative": "positive",
    }
    return opposites.get(pred_label.lower(), "a different prediction")


def build_context_for_promptbuilder(question: dict, model_str: str) -> dict:
    """Reconstruct the context dict passed to PromptBuilder.build_proposer_prompt()."""
    q_type   = question.get("q_type", 1)
    modality = question.get("modality", "tabular")
    features = question.get("features", {})
    predicted = question.get("predicted", {})
    target    = question.get("target", {})

    pred_label = (predicted.get("label", str(predicted.get("value", "?")))
                  if isinstance(predicted, dict) else str(predicted))
    pred_value = predicted.get("value", 0) if isinstance(predicted, dict) else 0
    tgt_label  = (target.get("label", str(target.get("value", "?")))
                  if isinstance(target, dict) else str(target))

    if isinstance(features, dict):
        feat_str   = ", ".join(f"{k}={v}" for k, v in list(features.items())[:10])
        data_desc  = f"Features: {feat_str}"
        feat_names = list(features.keys())
    elif isinstance(features, list):
        data_desc  = f"{len(features)} instances"
        feat_names = []
    else:
        data_desc  = str(features)[:400]
        feat_names = []

    context = {
        "user_question": question.get("question", question.get("q", "")),
        "model_info": {
            "model_name":   model_str,
            "model_type":   model_str,
            "architecture": model_str,
            "num_classes":  2,
            "device":       "cuda",
        },
        "prediction": {
            "predicted_class":     pred_label,
            "predicted_class_idx": pred_value,
            "confidence":          0.0,   # not stored in training data
            "top5_predictions":    [],
        },
        "input_data":      features,
        "feature_names":   feat_names,
        "data_description": data_desc,
        "ground_truth":    tgt_label,
    }

    if q_type in [5, 6, 7]:
        context["target_class"] = get_flip_target_class(question, pred_label)

    # Multi-instance support (Q4, Q9, Q10)
    if question.get("is_multi_instance"):
        context["num_instances"] = question.get("num_instances", 1)
        if isinstance(features, list):
            context["predictions"]   = [{"predicted_class": pred_label}] * len(features)
            context["input_paths"]   = [""] * len(features)
            context["image_indices"] = list(range(len(features)))
            descs = []
            for i, f in enumerate(features[:3]):
                if isinstance(f, dict):
                    row = ", ".join(f"{k}={v}" for k, v in list(f.items())[:8])
                    descs.append(f"Instance {chr(65+i)}: {row}")
            context["data_description"] = "\n".join(descs)
        if q_type in [8, 9, 10]:
            context["ground_truth"] = tgt_label

    return context


def build_base_prompt(d: dict, model_str: str) -> str:
    """Call the actual PromptBuilder for this record's q_type and modality."""
    q        = d["question"]
    q_type   = q.get("q_type", 1)
    modality = q.get("modality", "tabular")
    context  = build_context_for_promptbuilder(q, model_str)

    pb  = get_prompt_builder(q_type, modality)
    sig = inspect.signature(pb.build_proposer_prompt)
    if "for_react_agent" in sig.parameters:
        return pb.build_proposer_prompt(context, for_react_agent=False)
    return pb.build_proposer_prompt(context)


# ── Shared formatting helpers ────────────────────────────────────────────────────

def fmt_autonomous_results(auto: dict) -> str:
    """Format autonomous_results, stripping raw_response from each subtask."""
    if not auto:
        return ""
    clean = {}
    for k, v in auto.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "raw_response"}
        else:
            clean[k] = v
    return json.dumps(clean, indent=2)


def fmt_tool_importance(scores: dict) -> str:
    if not scores:
        return "  (none)"
    return "\n".join(f"  {tool}: {score:+.4f}" for tool, score in scores.items())


def fmt_proposer_reflection(pr) -> str:
    if isinstance(pr, str):
        try:
            pr = json.loads(pr)
        except Exception:
            return pr[:800]
    if not isinstance(pr, dict):
        return "(none)"
    lines = []
    for tool, info in pr.items():
        if isinstance(info, dict):
            eff   = info.get("effectiveness", "?")
            score = info.get("importance_score", None)
            rec   = info.get("recommendation", "?")
            s_str = f"{score:+.4f}" if isinstance(score, (int, float)) else str(score)
            lines.append(f"  {tool}: effectiveness={eff}, score={s_str}, recommendation={rec}")
        else:
            lines.append(f"  {tool}: {str(info)[:80]}")
    return "\n".join(lines)


def fmt_actor_reflection(ac) -> str:
    if isinstance(ac, str):
        try:
            ac = json.loads(ac)
        except Exception:
            return ac[:800]
    if isinstance(ac, dict):
        return ac.get("analysis", json.dumps(ac, indent=2))[:800]
    return "(none)"


def _fmt_faith(faith: dict) -> str:
    score  = faith.get("score", 0) or 0.0
    passed = faith.get("passed", False)
    p_orig = faith.get("p_original")
    p_mod  = faith.get("p_modified")
    s = f"**Faithfulness Score:** {score:.4f} ({'passed' if passed else 'failed'})"
    if p_orig is not None and p_mod is not None:
        s += f"\n- P(original class): {p_orig:.4f}\n- P(modified class): {p_mod:.4f}"
    return s


# ── Proposer completions ─────────────────────────────────────────────────────────

def build_proposer_completion_a(d: dict) -> str:
    """Type A completion: first-pass strategy that already passed the threshold (no reflection)."""
    orig        = d["original"]
    orig_strat  = clean_strategy(orig["strategy"])

    expl        = orig.get("explanation", {})
    expl_output = expl.get("output", {})
    expl_text   = expl.get("explanation", "")
    auto_res    = expl.get("autonomous_results", {})
    auto_str    = fmt_autonomous_results(auto_res)

    c = (
        f"```json\n{json.dumps(orig_strat, indent=2)}\n```\n\n"
        f"## Execution Result\n\n"
        f"**Structured Output:**\n"
        f"```json\n{json.dumps(expl_output, indent=2)}\n```\n\n"
        f"**Explanation:**\n{expl_text}\n"
    )
    if auto_str:
        c += f"\n**Autonomous Task Results:**\n```json\n{auto_str}\n```\n"
    c += "\n" + _fmt_faith(orig.get("explanation_faithfulness", {}))
    return c


def build_proposer_completion_full(d: dict) -> str:
    """Full trajectory for Type B (chosen) and Type C (rejected):
    initial strategy + execution → reflection → improved strategy + execution + faithfulness.
    """
    orig = d["original"]
    imp  = d["improved"]

    # ── original ──
    orig_strat  = clean_strategy(orig["strategy"])
    orig_expl   = orig.get("explanation", {})
    orig_output = orig_expl.get("output", {})
    orig_text   = orig_expl.get("explanation", "")
    orig_auto_s = fmt_autonomous_results(orig_expl.get("autonomous_results", {}))

    # ── reflection ──
    sf        = d.get("strategy_faithfulness", {})
    ti_scores = sf.get("tool_importance_scores", {})
    pr_str    = fmt_proposer_reflection(d.get("proposer_reflection", {}))
    ac_str    = fmt_actor_reflection(d.get("actor_reflection", {}))

    # ── improved ──
    imp_strat  = clean_strategy(imp["strategy"])
    imp_expl   = imp.get("explanation", {})
    imp_output = imp_expl.get("output", {})
    imp_text   = imp_expl.get("explanation", "")
    imp_auto_s = fmt_autonomous_results(imp_expl.get("autonomous_results", {}))

    c = (
        f"## Initial Strategy\n"
        f"```json\n{json.dumps(orig_strat, indent=2)}\n```\n\n"
        f"## Initial Execution Result\n\n"
        f"**Structured Output:**\n"
        f"```json\n{json.dumps(orig_output, indent=2)}\n```\n\n"
        f"**Explanation:**\n{orig_text}\n"
    )
    if orig_auto_s:
        c += f"\n**Autonomous Task Results:**\n```json\n{orig_auto_s}\n```\n"
    c += "\n" + _fmt_faith(orig.get("explanation_faithfulness", {}))

    c += (
        f"\n\n## Reflection\n\n"
        f"**Tool Importance Scores:**\n{fmt_tool_importance(ti_scores)}\n\n"
        f"**Critic's Feedback on Strategy:**\n{pr_str}\n\n"
        f"**Actor Analysis:**\n{ac_str}\n\n"
        f"## Improved Strategy\n"
        f"```json\n{json.dumps(imp_strat, indent=2)}\n```\n\n"
        f"## Improved Execution Result\n\n"
        f"**Structured Output:**\n"
        f"```json\n{json.dumps(imp_output, indent=2)}\n```\n\n"
        f"**Explanation:**\n{imp_text}\n"
    )
    if imp_auto_s:
        c += f"\n**Autonomous Task Results:**\n```json\n{imp_auto_s}\n```\n"
    c += "\n" + _fmt_faith(imp.get("explanation_faithfulness", {}))

    return c


# ── Actor helpers ───────────────────────────────────────────────────────────────

def load_tool_outputs(datapoint_id: str, modality: str) -> dict:
    """Load tool_outputs.json for a given datapoint (returns {} if not found)."""
    dataset_model = extract_dataset_model_from_id(datapoint_id)
    q_type        = extract_qtype_from_id(datapoint_id)
    row_no        = extract_row_no_from_id(datapoint_id)
    path = (TOOL_OUTPUTS_DIR / modality / dataset_model
            / f"q{q_type}" / str(row_no) / "tool_outputs.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_results_for_actor(tool_outputs: dict, strategy: dict, expl: dict) -> dict:
    """Build the `results` dict expected by build_actor_prompt.

    Filters tool_outputs to the tools selected in the strategy.
    Falls back to all non-autonomous results if none match.
    Uses autonomous_results from the stored explanation (more accurate than tool_outputs).
    """
    selected_tools = extract_tools_from_strategy(strategy)
    tool_results = {}
    for tool in selected_tools:
        key = tool.lower().replace(" ", "_").replace("-", "_")
        if key in tool_outputs:
            tool_results[key] = tool_outputs[key]
    if not tool_results:
        tool_results = {k: v for k, v in tool_outputs.items() if k != "autonomous_tasks"}

    autonomous_results = expl.get("autonomous_results",
                                  tool_outputs.get("autonomous_tasks", {}))
    return {"tool_results": tool_results, "autonomous_results": autonomous_results}


def load_improved_tool_results(datapoint_id: str, modality: str) -> dict:
    """Load tool_results from result_improved.json (returns {} if not found)."""
    dataset_model = extract_dataset_model_from_id(datapoint_id)
    q_type        = extract_qtype_from_id(datapoint_id)
    row_no        = extract_row_no_from_id(datapoint_id)
    path = (RESULTS_DIR / modality / dataset_model
            / f"q{q_type}" / str(row_no) / "result_improved.json")
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.get("tool_results", {}).items()
            if k != "autonomous_tasks"}


def fmt_improved_tool_results(tool_results: dict) -> str:
    """Format improved tool results for inclusion in actor completion."""
    if not tool_results:
        return "  (no tool results)"
    lines = []
    for tool_name, result in tool_results.items():
        if not isinstance(result, dict):
            continue
        success = result.get("success", False)
        method  = result.get("method", tool_name)
        lines.append(f"\n### {tool_name} ({method}): {'success' if success else 'failed'}")
        if not success:
            lines.append(f"  Error: {result.get('error', 'unknown')}")
            continue
        stats = result.get("statistics", {})
        if stats:
            lines.append(f"```json\n{json.dumps(stats, indent=2)}\n```")
        desc = result.get("description", "")
        if desc and not stats:
            lines.append(f"  {desc[:300]}")
    return "\n".join(lines) if lines else "  (no tool results)"


def build_actor_prompt_for_attempt(d: dict, attempt: str, model_str: str,
                                   modality: str) -> str:
    """Build the base actor prompt for 'original' or 'improved' attempt."""
    q      = d["question"]
    q_type = q.get("q_type", 1)
    dp_id  = d.get("datapoint_id", "")

    context      = build_context_for_promptbuilder(q, model_str)
    strategy     = d[attempt]["strategy"]
    expl         = d[attempt].get("explanation", {})
    tool_outputs = load_tool_outputs(dp_id, modality)
    results      = build_results_for_actor(tool_outputs, strategy, expl)

    pb = get_prompt_builder(q_type, modality)
    return pb.build_actor_prompt(context, strategy, results)


def build_actor_completion_a(d: dict) -> str:
    """Type A actor completion: original explanation that passed (no reflection)."""
    expl    = d["original"].get("explanation", {})
    output  = expl.get("output", {})
    text    = expl.get("explanation", "")
    auto_s  = fmt_autonomous_results(expl.get("autonomous_results", {}))

    c = (
        f"```json\n{json.dumps(output, indent=2)}\n```\n\n"
        f"**Explanation:**\n{text}\n"
    )
    if auto_s:
        c += f"\n**Autonomous Task Results:**\n```json\n{auto_s}\n```\n"
    c += "\n" + _fmt_faith(d["original"].get("explanation_faithfulness", {}))
    return c


def build_actor_completion_full(d: dict, modality: str = "tabular") -> str:
    """Full actor trajectory for Type B (chosen) and Type C (rejected):
    initial explanation → reflection → improved tool execution → improved explanation.
    """
    # ── original ──
    orig_expl   = d["original"].get("explanation", {})
    orig_output = orig_expl.get("output", {})
    orig_text   = orig_expl.get("explanation", "")
    orig_auto_s = fmt_autonomous_results(orig_expl.get("autonomous_results", {}))

    # ── tool importance scores (from strategy_faithfulness) ──
    sf        = d.get("strategy_faithfulness", {})
    ti_scores = sf.get("tool_importance_scores", {})

    # ── actor reflection (critic's feedback) ──
    actor_refl = d.get("actor_reflection", "")
    if isinstance(actor_refl, str):
        try:
            ar = json.loads(actor_refl)
            actor_refl = ar.get("analysis", actor_refl) if isinstance(ar, dict) else actor_refl
        except Exception:
            pass
    elif isinstance(actor_refl, dict):
        actor_refl = actor_refl.get("analysis", json.dumps(actor_refl, indent=2))

    # ── improved tool results (from result_improved.json) ──
    dp_id            = d.get("datapoint_id", "")
    imp_tool_results = load_improved_tool_results(dp_id, modality)
    imp_tool_str     = fmt_improved_tool_results(imp_tool_results)

    # ── improved explanation ──
    imp_expl   = d["improved"].get("explanation", {})
    imp_output = imp_expl.get("output", {})
    imp_text   = imp_expl.get("explanation", "")
    imp_auto_s = fmt_autonomous_results(imp_expl.get("autonomous_results", {}))

    c = (
        f"## Initial Explanation\n\n"
        f"```json\n{json.dumps(orig_output, indent=2)}\n```\n\n"
        f"**Explanation:**\n{orig_text}\n"
    )
    if orig_auto_s:
        c += f"\n**Autonomous Task Results:**\n```json\n{orig_auto_s}\n```\n"
    c += "\n" + _fmt_faith(d["original"].get("explanation_faithfulness", {}))

    c += (
        f"\n\n## Reflection\n\n"
        f"**Tool Importance Scores:**\n{fmt_tool_importance(ti_scores)}\n\n"
        f"**Critic's Feedback:**\n{actor_refl}\n\n"
        f"## Improved Tool Execution\n{imp_tool_str}\n\n"
        f"## Improved Explanation\n\n"
        f"```json\n{json.dumps(imp_output, indent=2)}\n```\n\n"
        f"**Explanation:**\n{imp_text}\n"
    )
    if imp_auto_s:
        c += f"\n**Autonomous Task Results:**\n```json\n{imp_auto_s}\n```\n"
    c += "\n" + _fmt_faith(d["improved"].get("explanation_faithfulness", {}))

    return c


# ── Pair loading ────────────────────────────────────────────────────────────────

def load_trajectory_pairs(
    data_dir: Path,
    modality: str = "tabular",
    seed: int = 42,
) -> list[dict]:
    """B×C and A×C trajectory pairs matched by q_type (same modality, cross-model).

    Type A: no `improved` field; original passed faithfulness threshold (passed=True).
            prompt    = base_prompt (PromptBuilder only).
            chosen    = original strategy + execution + faithfulness (passed).
    Type B: has `improved`; meaningful delta (high: >0.3 & imp>0.5, medium: 0.1<delta≤0.3).
            prompt    = base_prompt.
            chosen    = full trajectory: original → reflection → improved (better).
    Type C: has `improved`; delta < 0 (faithfulness degraded).
            prompt    = base_prompt.
            rejected  = full trajectory: original → reflection → improved (worse).
    """
    pattern = str(data_dir / modality / "**/*.json")
    files   = glob.glob(pattern, recursive=True)

    type_a        = defaultdict(list)
    type_b_high   = defaultdict(list)
    type_b_medium = defaultdict(list)
    type_c        = defaultdict(list)
    skipped_low   = 0
    build_errors  = 0

    for f in files:
        with open(f) as fp:
            d = json.load(fp)

        dp_id  = d.get("datapoint_id", "")
        model  = extract_model_from_id(dp_id)
        q_type = extract_qtype_from_id(dp_id)
        key    = q_type   # match by q_type only; modality already filtered by directory

        # Pre-validate PromptBuilder works for this record
        try:
            build_base_prompt(d, model)
        except Exception:
            build_errors += 1
            continue

        if "improved" not in d:
            # Type A: passed on first attempt
            orig_faith = d["original"]["explanation_faithfulness"]
            if orig_faith.get("passed", False):
                orig_s = orig_faith.get("score", 0) or 0.0
                type_a[key].append({
                    "d":       d,
                    "model":   model,
                    "q_type":  q_type,
                    "orig_s":  orig_s,
                    "dataset": d["question"].get("dataset", "unknown"),
                })
            continue

        orig_s = d["original"]["explanation_faithfulness"].get("score", 0) or 0.0
        imp_s  = d["improved"]["explanation_faithfulness"].get("score", 0) or 0.0
        delta  = imp_s - orig_s

        record = {
            "d":       d,
            "model":   model,
            "q_type":  q_type,
            "delta":   delta,
            "orig_s":  orig_s,
            "imp_s":   imp_s,
            "dataset": d["question"].get("dataset", "unknown"),
        }

        if delta > 0.3 and imp_s > 0.5:
            type_b_high[key].append(record)
        elif 0.1 < delta <= 0.3:
            type_b_medium[key].append(record)
        elif 0 < delta <= 0.1:
            skipped_low += 1
        elif delta < 0:
            type_c[key].append(record)

    rng = random.Random(seed)
    pairs = []
    skipped_no_c = 0
    pair_errors  = 0

    # ── Proposer B×C pairs ───────────────────────────────────────────────────────
    def make_pair_bc(b_rec, c_rec, quality: str) -> dict | None:
        try:
            prompt_chosen   = build_base_prompt(b_rec["d"], b_rec["model"])
            chosen          = build_proposer_completion_full(b_rec["d"])
            prompt_rejected = build_base_prompt(c_rec["d"], c_rec["model"])
            rejected        = build_proposer_completion_full(c_rec["d"])
        except Exception:
            return None
        return {
            "prompt_chosen":     prompt_chosen,
            "prompt_rejected":   prompt_rejected,
            "chosen":            chosen,
            "rejected":          rejected,
            "weight":            QUALITY_WEIGHTS[quality],
            "quality":           quality,
            "role":              "proposer",
            "delta_chosen":      round(b_rec["delta"], 4),
            "delta_rejected":    round(c_rec["delta"], 4),
            "imp_faith_chosen":  round(b_rec["imp_s"], 4),
            "orig_faith_chosen": round(b_rec["orig_s"], 4),
            "q_type":            b_rec["q_type"],
            "model":             b_rec["model"],
            "dataset_chosen":    b_rec["dataset"],
            "dataset_rejected":  c_rec["dataset"],
        }

    for tier_name, tier_dict in [("high", type_b_high), ("medium", type_b_medium)]:
        for key, b_list in tier_dict.items():
            c_list = type_c.get(key, [])
            if not c_list:
                skipped_no_c += len(b_list)
                continue
            c_usage = defaultdict(int)
            for b_rec in rng.sample(b_list, len(b_list)):
                available = [i for i in range(len(c_list)) if c_usage[i] < MAX_C_REUSE]
                if not available:
                    break
                idx   = rng.choice(available)
                c_usage[idx] += 1
                pair  = make_pair_bc(b_rec, c_list[idx], tier_name)
                if pair is None:
                    pair_errors += 1
                else:
                    pairs.append(pair)

    # ── Proposer A×C pairs ───────────────────────────────────────────────────────
    def make_pair_ac(a_rec, c_rec) -> dict | None:
        try:
            # chosen: first-pass (no reflection) → passed
            prompt_chosen   = build_base_prompt(a_rec["d"], a_rec["model"])
            chosen          = build_proposer_completion_a(a_rec["d"])
            # rejected: full degraded trajectory
            prompt_rejected = build_base_prompt(c_rec["d"], c_rec["model"])
            rejected        = build_proposer_completion_full(c_rec["d"])
        except Exception:
            return None
        return {
            "prompt_chosen":     prompt_chosen,
            "prompt_rejected":   prompt_rejected,
            "chosen":            chosen,
            "rejected":          rejected,
            "weight":            QUALITY_WEIGHTS["a"],
            "quality":           "a",
            "role":              "proposer",
            "delta_chosen":      round(a_rec["orig_s"], 4),
            "delta_rejected":    round(c_rec["delta"], 4),
            "imp_faith_chosen":  round(a_rec["orig_s"], 4),
            "orig_faith_chosen": round(a_rec["orig_s"], 4),
            "q_type":            a_rec["q_type"],
            "model":             a_rec["model"],
            "dataset_chosen":    a_rec["dataset"],
            "dataset_rejected":  c_rec["dataset"],
        }

    for key, a_list in type_a.items():
        c_list = type_c.get(key, [])
        if not c_list:
            skipped_no_c += len(a_list)
            continue
        c_usage = defaultdict(int)
        for a_rec in rng.sample(a_list, len(a_list)):
            available = [i for i in range(len(c_list)) if c_usage[i] < MAX_C_REUSE]
            if not available:
                break
            idx   = rng.choice(available)
            c_usage[idx] += 1
            pair  = make_pair_ac(a_rec, c_list[idx])
            if pair is None:
                pair_errors += 1
            else:
                pairs.append(pair)

    proposer_count = len(pairs)

    # ── Actor B×C pairs ──────────────────────────────────────────────────────────
    def make_pair_actor_bc(b_rec, c_rec, quality: str) -> dict | None:
        try:
            # Both sides use base actor prompt (original strategy); full trajectory in completion
            prompt_chosen   = build_actor_prompt_for_attempt(
                b_rec["d"], "original", b_rec["model"], modality)
            chosen          = build_actor_completion_full(b_rec["d"], modality)
            prompt_rejected = build_actor_prompt_for_attempt(
                c_rec["d"], "original", c_rec["model"], modality)
            rejected        = build_actor_completion_full(c_rec["d"], modality)
        except Exception:
            return None
        return {
            "prompt_chosen":     prompt_chosen,
            "prompt_rejected":   prompt_rejected,
            "chosen":            chosen,
            "rejected":          rejected,
            "weight":            QUALITY_WEIGHTS[quality],
            "quality":           quality,
            "role":              "actor",
            "delta_chosen":      round(b_rec["delta"], 4),
            "delta_rejected":    round(c_rec["delta"], 4),
            "imp_faith_chosen":  round(b_rec["imp_s"], 4),
            "orig_faith_chosen": round(b_rec["orig_s"], 4),
            "q_type":            b_rec["q_type"],
            "model":             b_rec["model"],
            "dataset_chosen":    b_rec["dataset"],
            "dataset_rejected":  c_rec["dataset"],
        }

    for tier_name, tier_dict in [("high", type_b_high), ("medium", type_b_medium)]:
        for key, b_list in tier_dict.items():
            c_list = type_c.get(key, [])
            if not c_list:
                continue
            c_usage = defaultdict(int)
            for b_rec in rng.sample(b_list, len(b_list)):
                available = [i for i in range(len(c_list)) if c_usage[i] < MAX_C_REUSE]
                if not available:
                    break
                idx  = rng.choice(available)
                c_usage[idx] += 1
                pair = make_pair_actor_bc(b_rec, c_list[idx], tier_name)
                if pair is None:
                    pair_errors += 1
                else:
                    pairs.append(pair)

    # ── Actor A×C pairs ──────────────────────────────────────────────────────────
    def make_pair_actor_ac(a_rec, c_rec) -> dict | None:
        try:
            # chosen: base prompt (original strategy) → original explanation (passed, no reflection)
            prompt_chosen   = build_actor_prompt_for_attempt(
                a_rec["d"], "original", a_rec["model"], modality)
            chosen          = build_actor_completion_a(a_rec["d"])
            # rejected: base prompt (original strategy) → full degraded trajectory
            prompt_rejected = build_actor_prompt_for_attempt(
                c_rec["d"], "original", c_rec["model"], modality)
            rejected        = build_actor_completion_full(c_rec["d"], modality)
        except Exception:
            return None
        return {
            "prompt_chosen":     prompt_chosen,
            "prompt_rejected":   prompt_rejected,
            "chosen":            chosen,
            "rejected":          rejected,
            "weight":            QUALITY_WEIGHTS["a"],
            "quality":           "a",
            "role":              "actor",
            "delta_chosen":      round(a_rec["orig_s"], 4),
            "delta_rejected":    round(c_rec["delta"], 4),
            "imp_faith_chosen":  round(a_rec["orig_s"], 4),
            "orig_faith_chosen": round(a_rec["orig_s"], 4),
            "q_type":            a_rec["q_type"],
            "model":             a_rec["model"],
            "dataset_chosen":    a_rec["dataset"],
            "dataset_rejected":  c_rec["dataset"],
        }

    for key, a_list in type_a.items():
        c_list = type_c.get(key, [])
        if not c_list:
            continue
        c_usage = defaultdict(int)
        for a_rec in rng.sample(a_list, len(a_list)):
            available = [i for i in range(len(c_list)) if c_usage[i] < MAX_C_REUSE]
            if not available:
                break
            idx  = rng.choice(available)
            c_usage[idx] += 1
            pair = make_pair_actor_ac(a_rec, c_list[idx])
            if pair is None:
                pair_errors += 1
            else:
                pairs.append(pair)

    actor_count = len(pairs) - proposer_count
    total_c     = sum(len(v) for v in type_c.values())

    print(f"  Total files:                         {len(files)}")
    print(f"  PromptBuilder errors (skipped):      {build_errors}")
    print(f"  Type A (passed 1st attempt):         {sum(len(v) for v in type_a.values())}")
    print(f"  Type B high  (delta>0.3, imp>0.5):   {sum(len(v) for v in type_b_high.values())}")
    print(f"  Type B medium (0.1<delta≤0.3):       {sum(len(v) for v in type_b_medium.values())}")
    print(f"  Type B excluded (delta≤0.1):         {skipped_low}")
    print(f"  Type C (delta<0):                    {total_c}")
    print(f"  Matching: by q_type only (cross-model, same modality)")
    print(f"  Skipped (no matching Type C):        {skipped_no_c}")
    print(f"  Build errors:                        {pair_errors}")
    print(f"  Proposer pairs (B×C + A×C):          {proposer_count}")
    print(f"  Actor pairs    (B×C + A×C):          {actor_count}")
    print(f"  Total pairs:                         {len(pairs)}")
    return pairs


# ── Output ──────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Written {len(records)} records → {path}")


def print_stats(pairs: list):
    if not pairs:
        return
    import statistics
    by_q    = defaultdict(int)
    by_qual = defaultdict(int)
    by_role = defaultdict(int)
    for p in pairs:
        by_q[p["q_type"]] += 1
        by_qual[p["quality"]] += 1
        by_role[p.get("role", "proposer")] += 1

    print(f"\nPair stats (n={len(pairs)}):")
    print("  Role:    " + "  ".join(f"{k}={v}" for k, v in sorted(by_role.items())))
    print("  Quality: " + "  ".join(f"{k}={v}" for k, v in sorted(by_qual.items())))
    print("  By q_type: " + "  ".join(f"Q{k}={v}" for k, v in sorted(by_q.items())))

    # Approx token count: max(chosen_seq, rejected_seq) where seq = system+prompt+completion
    sys_len = len(SYSTEM_PROMPT)
    lens = []
    for p in pairs:
        c_toks = (sys_len + len(p["prompt_chosen"])  + len(p["chosen"]))  / 3.5
        r_toks = (sys_len + len(p["prompt_rejected"]) + len(p["rejected"])) / 3.5
        lens.append(max(c_toks, r_toks))
    sl = sorted(lens)
    n  = len(sl)
    print(f"  Approx max-sequence tokens (per datum):")
    print(f"    mean={statistics.mean(lens):.0f}  "
          f"p75={sl[3*n//4]:.0f}  "
          f"p90={sl[9*n//10]:.0f}  "
          f"p95={sl[19*n//20]:.0f}  "
          f"max={sl[-1]:.0f}")


# ── Main ────────────────────────────────────────────────────────────────────────

def prepare(
    eval_ratio: float = 0.1,
    seed:       int   = 42,
    modality:   str   = "tabular",
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading trajectory pairs ({modality}) ...")
    pairs = load_trajectory_pairs(DATA_DIR, modality, seed)

    if not pairs:
        print("No pairs found.")
        return

    random.seed(seed)
    random.shuffle(pairs)

    n_eval    = max(1, int(len(pairs) * eval_ratio))
    eval_set  = pairs[:n_eval]
    train_set = pairs[n_eval:]

    print("\nWriting splits ...")
    write_jsonl(OUT_DIR / "train.jsonl", train_set)
    write_jsonl(OUT_DIR / "eval.jsonl",  eval_set)
    print_stats(pairs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval_ratio", type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--modality",   type=str,   default="tabular")
    args = p.parse_args()
    prepare(args.eval_ratio, args.seed, args.modality)
