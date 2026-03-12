"""
Tool-Only Baseline for Q1 (Tabular)

For each tabular Q1 dataset and each XAI tool, directly uses the tool's
top-ranked feature as the agent answer and evaluates with Q1Evaluator.

No LLM/VLM is involved — the tool output is used verbatim as the explanation.

Usage:
    cd /standard/AikyamLab/yuyang/xai_agent/framework/trial_2
    python baseline_tool_only_q1.py
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure we run from the trial_2 directory so relative imports work
TRIAL2_DIR = Path(__file__).parent.resolve()
os.chdir(TRIAL2_DIR)
sys.path.insert(0, str(TRIAL2_DIR))

from DataModelLoader import DataModelLoader
from xai_tools_native import (
    SHAPTabularTool,
    LIMETabularTool,
    IntegratedGradientsTabularTool,
    SensitivityAnalysisTabularTool,
)
from evaluation.explanation_faithfulness.q1_evaluator import Q1Evaluator
from evaluation.masking_utils import set_masking_output_dir

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = TRIAL2_DIR / "dataset" / "test" / "tabular"
OUTPUT_DIR  = TRIAL2_DIR / "outputs" / "baseline_tool_only_q1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_masking_output_dir(str(OUTPUT_DIR))

# (dataset_file, model_name, split_for_load)
# Adult:  row_no = original DataFrame index → loader ignores split, uses orig_to_pos
# Cancer: row_no = positional index into test set (Q1 uses split='test')
DATASET_CONFIGS = [
    {
        "file":       "adult_2layernn_q1.json",
        "model_name": "adult_2layernn",
        "split":      "test",
        "dataset_key": "adult_2layernn_q1",   # used as dataset_base_name
    },
    {
        "file":       "adult_tabnn_q1.json",
        "model_name": "adult_tabnn",
        "split":      "test",
        "dataset_key": "adult_tabnn_q1",
    },
    {
        "file":       "cancer_2layernn_q1.json",
        "model_name": "cancer_2layernn",
        "split":      "test",
        "dataset_key": "cancer_2layernn_q1",
    },
    {
        "file":       "cancer_tabnn_q1.json",
        "model_name": "cancer_tabnn",
        "split":      "test",
        "dataset_key": "cancer_tabnn_q1",
    },
]

TOOL_NAMES = ["shap", "lime", "integrated_gradients", "sensitivity_analysis"]

# Number of top features to use as the agent's answer.
# 1 = top-1 only; N > 1 = top-N features masked together.
TOP_N = 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_tools(loader: DataModelLoader) -> Dict[str, Any]:
    """Instantiate one of each tabular tool sharing the same DataModelLoader."""
    return {
        "shap":                   SHAPTabularTool(data_model_loader=loader),
        "lime":                   LIMETabularTool(data_model_loader=loader),
        "integrated_gradients":   IntegratedGradientsTabularTool(data_model_loader=loader),
        "sensitivity_analysis":   SensitivityAnalysisTabularTool(data_model_loader=loader),
    }


def get_top_features(tool_name: str, tool_result: dict, top_n: int = 1) -> List[str]:
    """Extract the top-N highest-importance feature names from a tool's result dict."""
    stats = tool_result.get("statistics", {})

    if tool_name in ("shap", "lime", "integrated_gradients"):
        items = stats.get("feature_importance", [])
    elif tool_name == "sensitivity_analysis":
        items = stats.get("feature_sensitivity", [])
    else:
        return []

    if not items:
        return []

    # Tools already sort items by abs(score) descending; take the first top_n.
    return [item["feature"] for item in items[:top_n] if item.get("feature")]


def run_tool(tool, tool_name: str, target_class: int) -> Optional[dict]:
    """Run a tool and return the parsed JSON result, or None on failure."""
    try:
        raw = tool.run(target_class=target_class)
        result = json.loads(raw)
        if not result.get("success"):
            print(f"    [WARN] tool={tool_name} returned success=False: {result.get('error','')[:120]}")
            return None
        return result
    except Exception as e:
        print(f"    [ERROR] tool={tool_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(cfg: dict) -> Dict[str, List[dict]]:
    """
    Evaluate all tools on all samples for one dataset config.

    Returns:
        {tool_name: [per_sample_result_dict, ...]}
    """
    dataset_file = DATASET_DIR / cfg["file"]
    model_name   = cfg["model_name"]
    split        = cfg["split"]
    dataset_key  = cfg["dataset_key"]

    print(f"\n{'='*70}")
    print(f"Dataset: {cfg['file']}  (model={model_name})")
    print(f"{'='*70}")

    with open(dataset_file) as f:
        samples = json.load(f)

    print(f"Loading model '{model_name}' ...")
    loader = DataModelLoader(model_name=model_name, modality="tabular")
    model  = loader.get_model()
    device = loader.device

    tools = make_tools(loader)
    evaluator = Q1Evaluator(modality="tabular")

    # {tool_name: [result_dict, ...]}
    tool_results: Dict[str, List[dict]] = {t: [] for t in TOOL_NAMES}

    for idx, sample in enumerate(samples):
        row_no   = sample["row_no"]
        features_raw = sample.get("features", {})

        print(f"\n  Sample {idx+1}/{len(samples)}  row_no={row_no}")

        # Load this sample into the loader (updates internal state)
        try:
            loader.load_sample(row_no, split=split)
        except Exception as e:
            print(f"    [ERROR] load_sample({row_no}): {e}")
            for t in TOOL_NAMES:
                tool_results[t].append({
                    "row_no": row_no, "score": None, "passed": None,
                    "error": f"load_sample failed: {e}"
                })
            continue

        # Preprocessed feature tensor [N]
        features_tensor = loader.get_current_features()
        if features_tensor is None:
            print(f"    [ERROR] get_current_features() returned None")
            for t in TOOL_NAMES:
                tool_results[t].append({
                    "row_no": row_no, "score": None, "passed": None,
                    "error": "get_current_features() is None"
                })
            continue

        feature_names = loader.get_feature_names() or []

        # Get original model prediction
        try:
            orig_pred = evaluator.get_prediction(model, features_tensor, processor=None, device=str(device))
            predicted_class = orig_pred["predicted_class_idx"]
            print(f"    predicted_class={predicted_class}  confidence={orig_pred['confidence']:.4f}")
        except Exception as e:
            print(f"    [ERROR] get_prediction: {e}")
            for t in TOOL_NAMES:
                tool_results[t].append({
                    "row_no": row_no, "score": None, "passed": None,
                    "error": f"get_prediction failed: {e}"
                })
            continue

        # Run each tool and evaluate
        for tool_name in TOOL_NAMES:
            tool = tools[tool_name]

            # Run the tool
            tool_result = run_tool(tool, tool_name, target_class=predicted_class)
            if tool_result is None:
                tool_results[tool_name].append({
                    "row_no": row_no, "score": None, "passed": None,
                    "error": "tool execution failed"
                })
                continue

            # Extract top-N features
            top_features = get_top_features(tool_name, tool_result, top_n=TOP_N)
            if not top_features:
                print(f"    [{tool_name}] no top features extracted")
                tool_results[tool_name].append({
                    "row_no": row_no, "score": None, "passed": None,
                    "error": "no top features extracted"
                })
                continue

            print(f"    [{tool_name}] top_{TOP_N}_features={top_features}")

            # Build agent_output in the format Q1Evaluator expects
            agent_output = {"output": {"feature_keys": top_features}}

            # Evaluate
            try:
                eval_result = evaluator.evaluate(
                    agent_output=agent_output,
                    original_input=features_tensor,
                    model=model,
                    original_prediction=orig_pred,
                    # masker kwargs
                    dataset_base_name=dataset_key,
                    row_no=row_no,
                    tool_name=tool_name,
                    feature_names=feature_names,
                    original_features=loader.get_current_features_dict() or features_raw,
                    device=str(device),
                )
                score  = eval_result.score
                passed = eval_result.passed
                errors = eval_result.errors
                print(f"      score={score:.4f}  passed={passed}  "
                      f"p_orig={eval_result.p_original:.4f}  p_mod={eval_result.p_modified:.4f}")
                if errors:
                    print(f"      errors: {errors}")

                tool_results[tool_name].append({
                    "row_no":      row_no,
                    "top_features": top_features,
                    "score":       score,
                    "passed":      passed,
                    "p_original":  eval_result.p_original,
                    "p_modified":  eval_result.p_modified,
                    "raw_drop":    eval_result.details.get("raw_drop"),
                    "region_ratio": eval_result.details.get("region_ratio"),
                    "class_changed": eval_result.details.get("class_changed"),
                    "errors":      errors,
                })
            except Exception as e:
                tb = traceback.format_exc()
                print(f"      [ERROR] evaluate: {e}")
                tool_results[tool_name].append({
                    "row_no": row_no, "top_features": top_features,
                    "score": None, "passed": None,
                    "error": str(e), "traceback": tb
                })

    return tool_results


def summarize(tool_results: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Compute mean score, mean raw_drop, pass rate, and error rate per tool."""
    summary = {}
    for tool_name, results in tool_results.items():
        valid   = [r for r in results if r["score"] is not None]
        n_total = len(results)
        n_valid = len(valid)
        n_err   = n_total - n_valid

        if valid:
            scores      = [r["score"] for r in valid]
            raw_drops   = [r["raw_drop"] for r in valid if r.get("raw_drop") is not None]
            mean_score  = sum(scores) / len(scores)
            mean_raw_drop = sum(raw_drops) / len(raw_drops) if raw_drops else None
            pass_rate   = sum(r["passed"] for r in valid) / len(valid)
        else:
            mean_score    = None
            mean_raw_drop = None
            pass_rate     = None

        summary[tool_name] = {
            "n_total":      n_total,
            "n_valid":      n_valid,
            "n_errors":     n_err,
            "mean_score":   mean_score,
            "mean_raw_drop": mean_raw_drop,
            "pass_rate":    pass_rate,
        }
    return summary


def print_combined_table(all_summaries: Dict[str, Dict[str, dict]]):
    """Print all datasets × tools in a single box-drawing table."""
    # Column widths
    W_DS   = 17  # Dataset
    W_TOOL = 22  # Tool
    W_MS   = 12  # Mean Score
    W_RD   = 10  # Raw Drop
    W_PR   = 11  # Pass Rate

    def h_line(left, mid, right, fill="─"):
        return (left
                + fill * (W_DS   + 2) + mid
                + fill * (W_TOOL + 2) + mid
                + fill * (W_MS   + 2) + mid
                + fill * (W_RD   + 2) + mid
                + fill * (W_PR   + 2) + right)

    top    = h_line("┌", "┬", "┐")
    sep    = h_line("├", "┼", "┤")
    bottom = h_line("└", "┴", "┘")

    def row(ds, tool, ms, rd, pr):
        return (f"│ {ds:<{W_DS}} │ {tool:<{W_TOOL}} │ {ms:>{W_MS}} │"
                f" {rd:>{W_RD}} │ {pr:>{W_PR}} │")

    print(f"\nTop-{TOP_N} results overall (TOP_N={TOP_N})，per-dataset per-tool means\n")
    print(top)
    print(row("Dataset", "Tool", "Mean Score", "Raw Drop", "Pass Rate"))
    print(sep)

    dataset_names = list(all_summaries.keys())
    for i, ds_name in enumerate(dataset_names):
        summary = all_summaries[ds_name]
        # strip trailing "_q1" for display
        ds_label = ds_name.replace("_q1", "")
        tool_names = list(summary.keys())
        for j, tool_name in enumerate(tool_names):
            s = summary[tool_name]
            ms = f"{s['mean_score']:.4f}"    if s["mean_score"]    is not None else "N/A"
            rd = f"{s['mean_raw_drop']:.4f}" if s["mean_raw_drop"] is not None else "N/A"
            pr = f"{s['pass_rate']:.0%}"     if s["pass_rate"]     is not None else "N/A"
            print(row(ds_label, tool_name, ms, rd, pr))
            # separator between rows, but thicker between datasets
            is_last_tool    = (j == len(tool_names) - 1)
            is_last_dataset = (i == len(dataset_names) - 1)
            if is_last_tool and is_last_dataset:
                print(bottom)
            elif is_last_tool:
                print(sep)
            else:
                print(row("", "", "", "", "").replace("│", "│").replace(
                    " " * (W_DS + 2), " " * (W_DS + 2)))
                # thin inner row separator
                print("│" + " " * (W_DS + 2)
                      + "├" + "─" * (W_TOOL + 2)
                      + "┼" + "─" * (W_MS   + 2)
                      + "┼" + "─" * (W_RD   + 2)
                      + "┼" + "─" * (W_PR   + 2) + "┤")


def main():
    all_summaries = {}

    for cfg in DATASET_CONFIGS:
        name = cfg["dataset_key"]
        try:
            tool_results = evaluate_dataset(cfg)
        except Exception as e:
            print(f"\n[FATAL] {name}: {e}")
            traceback.print_exc()
            continue

        summary = summarize(tool_results)
        all_summaries[name] = summary

        # Save per-dataset results to JSON
        out_path = OUTPUT_DIR / f"{name}_top{TOP_N}_results.json"
        with open(out_path, "w") as f:
            json.dump({"dataset": name, "tool_results": tool_results, "summary": summary}, f, indent=2)
        print(f"\n  Results saved to: {out_path}")

    # ── Per-dataset combined box table ────────────────────────────────────────
    if all_summaries:
        print_combined_table(all_summaries)

    # ── Overall summary across all datasets ──────────────────────────────────
    print(f"\n{'─'*74}")
    print(f"OVERALL  (top-{TOP_N}, mean of per-dataset means)")
    print(f"{'─'*74}")
    print(f"  {'Tool':<22}  {'Mean Score':>10}  {'Raw Drop':>9}  {'Pass Rate':>9}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*9}  {'-'*9}")

    for tool_name in TOOL_NAMES:
        scores = [
            s[tool_name]["mean_score"]
            for s in all_summaries.values()
            if s.get(tool_name, {}).get("mean_score") is not None
        ]
        raw_drops = [
            s[tool_name]["mean_raw_drop"]
            for s in all_summaries.values()
            if s.get(tool_name, {}).get("mean_raw_drop") is not None
        ]
        pass_rates = [
            s[tool_name]["pass_rate"]
            for s in all_summaries.values()
            if s.get(tool_name, {}).get("pass_rate") is not None
        ]
        ms = f"{sum(scores)/len(scores):.4f}"         if scores     else "N/A"
        rd = f"{sum(raw_drops)/len(raw_drops):.4f}"   if raw_drops  else "N/A"
        pr = f"{sum(pass_rates)/len(pass_rates):.0%}" if pass_rates else "N/A"
        print(f"  {tool_name:<22}  {ms:>10}  {rd:>9}  {pr:>9}")

    # Save overall summary
    overall_path = OUTPUT_DIR / f"overall_summary_top{TOP_N}.json"
    with open(overall_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nOverall summary saved to: {overall_path}")


if __name__ == "__main__":
    main()
