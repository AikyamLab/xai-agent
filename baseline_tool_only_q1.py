"""
Tool-Only Baseline for Q1 — All Modalities (Text, Vision, Tabular)

For each modality × dataset × tool, directly uses the tool's output as the
explanation and evaluates faithfulness with Q1Evaluator.  No LLM/VLM involved.

Explanation budget:
  - Tabular / Text : top 25% of features / words by absolute importance
  - Vision         : suggested_bounding_box from the tool (derived from top 1%
                     of heatmap pixels inside each tool implementation)

Results are saved to:
  outputs/baseline_tool_only/<dataset_key>_results.json   (per dataset)
  outputs/baseline_tool_only/overall_summary.json         (aggregate)

Usage:
    cd /standard/AikyamLab/yuyang/xai_agent/framework/trial_2
    python baseline_tool_only_q1.py
"""

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

TRIAL2_DIR = Path(__file__).parent.resolve()
os.chdir(TRIAL2_DIR)
sys.path.insert(0, str(TRIAL2_DIR))

from DataModelLoader import DataModelLoader
from xai_tools_native import (
    # Tabular
    SHAPTabularTool, LIMETabularTool, IntegratedGradientsTabularTool,
    SensitivityAnalysisTabularTool, GuidedBackpropTabularTool, SmoothGradTabularTool,
    # Text
    LIMETextTool, IntegratedGradientsTextTool, SHAPTextTool,
    SensitivityAnalysisTextTool, GuidedBackpropTextTool, SmoothGradTextTool,
    # Vision
    GradCAMTool, IntegratedGradientsTool, LIMETool, SHAPTool,
    GuidedBackpropTool, SmoothGradTool,
)
from evaluation.explanation_faithfulness.q1_evaluator import Q1Evaluator
from evaluation.masking_utils import set_masking_output_dir

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = TRIAL2_DIR / "outputs" / "baseline_tool_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_masking_output_dir(str(OUTPUT_DIR))

# Fraction of features/words to mask (top 25%)
TOP_FRAC = 0.25

TABULAR_DIR = TRIAL2_DIR / "dataset" / "test" / "tabular"
TEXT_DIR    = TRIAL2_DIR / "dataset" / "test" / "text"
VISION_DIR  = TRIAL2_DIR / "dataset" / "test" / "vision"

# (dataset_file, model_name, split, dataset_key)
# Adult:  row_no = original DataFrame index → loader handles via orig_to_pos
# Cancer: row_no = positional index into test set (Q1 uses split='test')
TABULAR_DATASETS = [
    {"file": "adult_2layernn_q1.json",  "model_name": "adult_2layernn",  "split": "test", "dataset_key": "adult_2layernn_q1"},
    {"file": "adult_tabnn_q1.json",     "model_name": "adult_tabnn",     "split": "test", "dataset_key": "adult_tabnn_q1"},
    {"file": "cancer_2layernn_q1.json", "model_name": "cancer_2layernn", "split": "test", "dataset_key": "cancer_2layernn_q1"},
    {"file": "cancer_tabnn_q1.json",    "model_name": "cancer_tabnn",    "split": "test", "dataset_key": "cancer_tabnn_q1"},
]

TEXT_DATASETS = [
    {"file": "imdb_2layernn_q1.json", "model_name": "imdb_2layernn", "dataset_key": "imdb_2layernn_q1"},
    {"file": "imdb_cnn_q1.json",      "model_name": "imdb_cnn",      "dataset_key": "imdb_cnn_q1"},
    {"file": "snli_2layernn_q1.json", "model_name": "snli_2layernn", "dataset_key": "snli_2layernn_q1"},
    {"file": "snli_cnn_q1.json",      "model_name": "snli_cnn",      "dataset_key": "snli_cnn_q1"},
]

VISION_DATASETS = [
    {"file": "cub_densenet_q1.json",   "model_name": "cub_densenet",   "dataset_key": "cub_densenet_q1"},
    {"file": "cub_resnet_q1.json",     "model_name": "cub_resnet",     "dataset_key": "cub_resnet_q1"},
    {"file": "stl10_densenet_q1.json", "model_name": "stl10_densenet", "dataset_key": "stl10_densenet_q1"},
    {"file": "stl10_resnet_q1.json",   "model_name": "stl10_resnet",   "dataset_key": "stl10_resnet_q1"},
]

TABULAR_TOOL_NAMES = ["shap", "lime", "integrated_gradients", "sensitivity_analysis", "guided_backprop", "smoothgrad"]
TEXT_TOOL_NAMES    = ["lime", "integrated_gradients", "shap", "sensitivity_analysis", "guided_backprop", "smoothgrad"]
VISION_TOOL_NAMES  = ["gradcam", "integrated_gradients", "lime", "shap", "guided_backprop", "smoothgrad"]


# ─────────────────────────────────────────────────────────────────────────────
# Tool factories
# ─────────────────────────────────────────────────────────────────────────────

def make_tabular_tools(loader: DataModelLoader) -> Dict[str, Any]:
    return {
        "shap":                 SHAPTabularTool(data_model_loader=loader),
        "lime":                 LIMETabularTool(data_model_loader=loader),
        "integrated_gradients": IntegratedGradientsTabularTool(data_model_loader=loader),
        "sensitivity_analysis": SensitivityAnalysisTabularTool(data_model_loader=loader),
        "guided_backprop":      GuidedBackpropTabularTool(data_model_loader=loader),
        "smoothgrad":           SmoothGradTabularTool(data_model_loader=loader),
    }


def make_text_tools(loader: DataModelLoader) -> Dict[str, Any]:
    return {
        "lime":                 LIMETextTool(data_model_loader=loader),
        "integrated_gradients": IntegratedGradientsTextTool(data_model_loader=loader),
        "shap":                 SHAPTextTool(data_model_loader=loader),
        "sensitivity_analysis": SensitivityAnalysisTextTool(data_model_loader=loader),
        "guided_backprop":      GuidedBackpropTextTool(data_model_loader=loader),
        "smoothgrad":           SmoothGradTextTool(data_model_loader=loader),
    }


def make_vision_tools(loader: DataModelLoader) -> Dict[str, Any]:
    return {
        "gradcam":              GradCAMTool(data_model_loader=loader),
        "integrated_gradients": IntegratedGradientsTool(data_model_loader=loader),
        "lime":                 LIMETool(data_model_loader=loader),
        "shap":                 SHAPTool(data_model_loader=loader),
        "guided_backprop":      GuidedBackpropTool(data_model_loader=loader),
        "smoothgrad":           SmoothGradTool(data_model_loader=loader),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from tool outputs
# ─────────────────────────────────────────────────────────────────────────────

def get_top_features_tabular(tool_name: str, tool_result: dict, n_features: int) -> List[str]:
    """Return top ceil(TOP_FRAC * n_features) feature names from a tabular tool result."""
    top_n = max(1, math.ceil(TOP_FRAC * n_features))
    stats = tool_result.get("statistics", {})
    if tool_name in ("shap", "lime", "integrated_gradients"):
        items = stats.get("feature_importance", [])
        key   = "feature"
    elif tool_name == "sensitivity_analysis":
        items = stats.get("feature_sensitivity", [])
        key   = "feature"
    elif tool_name in ("guided_backprop", "smoothgrad"):
        # These tools store only top-10 in statistics (pre-sorted)
        items = stats.get("top_features", [])
        key   = "feature"
    else:
        return []
    return [item[key] for item in items[:top_n] if item.get(key)]


def get_top_words_text(tool_name: str, tool_result: dict, n_words: int) -> List[str]:
    """Return top ceil(TOP_FRAC * n_words) word/token strings from a text tool result."""
    top_n = max(1, math.ceil(TOP_FRAC * n_words))
    stats = tool_result.get("statistics", {})
    if tool_name == "lime":
        items = stats.get("word_importance", [])
        key   = "word"
    elif tool_name == "integrated_gradients":
        items = stats.get("token_importance", [])
        key   = "token"
    elif tool_name == "shap":
        items = stats.get("word_importance", [])
        key   = "word"
    elif tool_name == "sensitivity_analysis":
        items = stats.get("word_sensitivity", [])
        key   = "word"
    elif tool_name in ("guided_backprop", "smoothgrad"):
        # These tools store only top-10 (pre-sorted)
        items = stats.get("top_tokens", [])
        key   = "token"
    else:
        return []
    return [item[key] for item in items[:top_n] if item.get(key)]


def get_bounding_box_vision(tool_result: dict) -> Optional[List[int]]:
    """Return the tool's suggested bounding box [x1, y1, x2, y2] (top-1% pixels)."""
    return tool_result.get("suggested_bounding_box")


def run_tool(tool: Any, tool_name: str, target_class: int) -> Optional[dict]:
    """Run a tool and return the parsed JSON result, or None on failure."""
    try:
        raw    = tool.run(target_class=target_class)
        result = json.loads(raw)
        if not result.get("success"):
            print(f"    [WARN] {tool_name} success=False: {result.get('error', '')[:120]}")
            return None
        return result
    except Exception as e:
        print(f"    [ERROR] {tool_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Common result recording
# ─────────────────────────────────────────────────────────────────────────────

def _record_result(results_list: list, row_no: int, explanation: Any, eval_result: Any) -> None:
    score  = eval_result.score
    passed = eval_result.passed
    errors = eval_result.errors
    print(f"      score={score:.4f}  passed={passed}  "
          f"p_orig={eval_result.p_original:.4f}  p_mod={eval_result.p_modified:.4f}")
    if errors:
        print(f"      errors: {errors}")
    results_list.append({
        "row_no":       row_no,
        "explanation":  explanation,
        "score":        score,
        "passed":       passed,
        "p_original":   eval_result.p_original,
        "p_modified":   eval_result.p_modified,
        "raw_drop":     eval_result.details.get("raw_drop"),
        "region_ratio": eval_result.details.get("region_ratio"),
        "class_changed": eval_result.details.get("class_changed"),
        "errors":       errors,
    })


def _record_error(results_list: list, row_no: int, error: str) -> None:
    results_list.append({"row_no": row_no, "score": None, "passed": None, "error": error})


# ─────────────────────────────────────────────────────────────────────────────
# Tabular dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tabular_dataset(cfg: dict, skip_tools: Optional[set] = None) -> Dict[str, List[dict]]:
    dataset_file = TABULAR_DIR / cfg["file"]
    model_name   = cfg["model_name"]
    split        = cfg["split"]
    dataset_key  = cfg["dataset_key"]

    print(f"\n{'='*70}")
    print(f"[TABULAR] {cfg['file']}  model={model_name}")
    print(f"{'='*70}")

    with open(dataset_file) as f:
        samples = json.load(f)

    print(f"Loading model '{model_name}' ...")
    loader    = DataModelLoader(model_name=model_name, modality="tabular")
    model     = loader.get_model()
    device    = loader.device
    active_tools = [t for t in TABULAR_TOOL_NAMES if not (skip_tools and t in skip_tools)]
    tools     = make_tabular_tools(loader)
    evaluator = Q1Evaluator(modality="tabular")

    tool_results: Dict[str, List[dict]] = {t: [] for t in active_tools}

    for idx, sample in enumerate(samples):
        row_no       = sample["row_no"]
        features_raw = sample.get("features", {})

        print(f"\n  [{idx+1}/{len(samples)}] row_no={row_no}")

        try:
            loader.load_sample(row_no, split=split)
        except Exception as e:
            print(f"    [ERROR] load_sample: {e}")
            for t in TABULAR_TOOL_NAMES:
                _record_error(tool_results[t], row_no, f"load_sample: {e}")
            continue

        features_tensor = loader.get_current_features()
        if features_tensor is None:
            print(f"    [ERROR] get_current_features() is None")
            for t in active_tools:
                _record_error(tool_results[t], row_no, "features is None")
            continue

        n_features    = features_tensor.shape[-1]
        feature_names = loader.get_feature_names() or []

        try:
            orig_pred       = evaluator.get_prediction(model, features_tensor, processor=None, device=str(device))
            predicted_class = orig_pred["predicted_class_idx"]
            print(f"    class={predicted_class}  conf={orig_pred['confidence']:.4f}  n_feat={n_features}")
        except Exception as e:
            print(f"    [ERROR] get_prediction: {e}")
            for t in active_tools:
                _record_error(tool_results[t], row_no, f"get_prediction: {e}")
            continue

        for tool_name in active_tools:
            tool_result = run_tool(tools[tool_name], tool_name, predicted_class)
            if tool_result is None:
                tool_results[tool_name].append({"row_no": row_no, "score": None, "passed": None, "error": "tool failed"})
                continue

            top_features = get_top_features_tabular(tool_name, tool_result, n_features)
            if not top_features:
                print(f"    [{tool_name}] no features")
                tool_results[tool_name].append({"row_no": row_no, "score": None, "passed": None, "error": "no features"})
                continue

            print(f"    [{tool_name}] top_features={top_features}")
            agent_output = {"output": {"feature_keys": top_features}}

            try:
                eval_result = evaluator.evaluate(
                    agent_output=agent_output,
                    original_input=features_tensor,
                    model=model,
                    original_prediction=orig_pred,
                    dataset_base_name=dataset_key,
                    row_no=row_no,
                    tool_name=tool_name,
                    feature_names=feature_names,
                    original_features=loader.get_raw_features_dict() or loader.get_current_features_dict() or features_raw,
                    device=str(device),
                    processor=loader.get_processor(),
                    feature_modes=loader.get_feature_modes(),
                )
                _record_result(tool_results[tool_name], row_no, top_features, eval_result)
            except Exception as e:
                print(f"      [ERROR] evaluate: {e}")
                tool_results[tool_name].append({
                    "row_no": row_no, "explanation": top_features,
                    "score": None, "passed": None, "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    return tool_results


# ─────────────────────────────────────────────────────────────────────────────
# Text dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_text_dataset(cfg: dict, skip_tools: Optional[set] = None) -> Dict[str, List[dict]]:
    dataset_file = TEXT_DIR / cfg["file"]
    model_name   = cfg["model_name"]
    dataset_key  = cfg["dataset_key"]

    print(f"\n{'='*70}")
    print(f"[TEXT] {cfg['file']}  model={model_name}")
    print(f"{'='*70}")

    with open(dataset_file) as f:
        samples = json.load(f)

    print(f"Loading model '{model_name}' ...")
    loader        = DataModelLoader(model_name=model_name, modality="text")
    model         = loader.get_model()
    processor     = loader.get_processor()
    device        = loader.device
    loader_module = loader.loader_module
    active_tools  = [t for t in TEXT_TOOL_NAMES if not (skip_tools and t in skip_tools)]
    tools         = make_text_tools(loader)
    evaluator     = Q1Evaluator(modality="text")

    tool_results: Dict[str, List[dict]] = {t: [] for t in active_tools}

    for idx, sample in enumerate(samples):
        row_no   = sample["row_no"]
        features = sample.get("features", {})

        print(f"\n  [{idx+1}/{len(samples)}] row_no={row_no}")

        # Determine text format: NLI (premise/hypothesis) vs single text
        is_nli = "premise" in features and "hypothesis" in features
        if is_nli:
            text_input     = {"premise": features["premise"], "hypothesis": features["hypothesis"]}
            original_input = {"premise": features["premise"], "hypothesis": features["hypothesis"]}
            # Tools perturb only the premise for NLI; use premise word count as budget
            n_words = len(features["premise"].split())
        else:
            text_input     = features.get("text", "")
            original_input = text_input
            n_words        = len(text_input.split())

        if not text_input:
            print(f"    [ERROR] empty text")
            for t in active_tools:
                _record_error(tool_results[t], row_no, "empty text")
            continue

        # Load sample into loader so all tools can access current_sample_data
        try:
            sample_data = loader_module.load_data(text_input)
            loader.current_sample_data = sample_data
        except Exception as e:
            print(f"    [ERROR] load_data: {e}")
            for t in active_tools:
                _record_error(tool_results[t], row_no, f"load_data: {e}")
            continue

        try:
            orig_pred       = evaluator.get_prediction(model, original_input, processor=processor, device=str(device))
            predicted_class = orig_pred["predicted_class_idx"]
            print(f"    class={predicted_class}  conf={orig_pred['confidence']:.4f}  n_words={n_words}")
        except Exception as e:
            print(f"    [ERROR] get_prediction: {e}")
            for t in active_tools:
                _record_error(tool_results[t], row_no, f"get_prediction: {e}")
            continue

        for tool_name in active_tools:
            tool_result = run_tool(tools[tool_name], tool_name, predicted_class)
            if tool_result is None:
                tool_results[tool_name].append({"row_no": row_no, "score": None, "passed": None, "error": "tool failed"})
                continue

            top_words = get_top_words_text(tool_name, tool_result, n_words)
            if not top_words:
                print(f"    [{tool_name}] no words")
                tool_results[tool_name].append({"row_no": row_no, "score": None, "passed": None, "error": "no words"})
                continue

            print(f"    [{tool_name}] top_words={top_words}")
            agent_output = {"output": {"text_spans": top_words}}

            try:
                eval_result = evaluator.evaluate(
                    agent_output=agent_output,
                    original_input=original_input,
                    model=model,
                    original_prediction=orig_pred,
                    processor=processor,
                    device=str(device),
                    dataset_base_name=dataset_key,
                    row_no=row_no,
                    tool_name=tool_name,
                )
                _record_result(tool_results[tool_name], row_no, top_words, eval_result)
            except Exception as e:
                print(f"      [ERROR] evaluate: {e}")
                tool_results[tool_name].append({
                    "row_no": row_no, "explanation": top_words,
                    "score": None, "passed": None, "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    return tool_results


# ─────────────────────────────────────────────────────────────────────────────
# Vision dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vision_dataset(cfg: dict, skip_tools: Optional[set] = None) -> Dict[str, List[dict]]:
    dataset_file = VISION_DIR / cfg["file"]
    model_name   = cfg["model_name"]
    dataset_key  = cfg["dataset_key"]

    print(f"\n{'='*70}")
    print(f"[VISION] {cfg['file']}  model={model_name}")
    print(f"{'='*70}")

    with open(dataset_file) as f:
        samples = json.load(f)

    print(f"Loading model '{model_name}' ...")
    loader       = DataModelLoader(model_name=model_name, modality="vision")
    model        = loader.get_model()
    processor    = loader.get_processor()
    device       = loader.device
    active_tools = [t for t in VISION_TOOL_NAMES if not (skip_tools and t in skip_tools)]
    tools        = make_vision_tools(loader)
    evaluator    = Q1Evaluator(modality="vision")

    tool_results: Dict[str, List[dict]] = {t: [] for t in active_tools}

    for idx, sample in enumerate(samples):
        row_no = sample["row_no"]

        print(f"\n  [{idx+1}/{len(samples)}] row_no={row_no}")

        try:
            loader.load_sample(row_no, split="test")
        except Exception as e:
            print(f"    [ERROR] load_sample: {e}")
            for t in active_tools:
                _record_error(tool_results[t], row_no, f"load_sample: {e}")
            continue

        # Processed PIL image (after Resize+CenterCrop); same coordinate space as tool bboxes
        original_input = loader.get_current_image()
        if original_input is None:
            print(f"    [ERROR] get_current_image() is None")
            for t in active_tools:
                _record_error(tool_results[t], row_no, "image is None")
            continue

        try:
            orig_pred       = evaluator.get_prediction(model, original_input, processor=processor, device=str(device))
            predicted_class = orig_pred["predicted_class_idx"]
            print(f"    class={predicted_class}  conf={orig_pred['confidence']:.4f}")
        except Exception as e:
            print(f"    [ERROR] get_prediction: {e}")
            for t in active_tools:
                _record_error(tool_results[t], row_no, f"get_prediction: {e}")
            continue

        for tool_name in active_tools:
            tool_result = run_tool(tools[tool_name], tool_name, predicted_class)
            if tool_result is None:
                tool_results[tool_name].append({"row_no": row_no, "score": None, "passed": None, "error": "tool failed"})
                continue

            bbox = get_bounding_box_vision(tool_result)
            if bbox is None:
                print(f"    [{tool_name}] no bounding box")
                tool_results[tool_name].append({"row_no": row_no, "score": None, "passed": None, "error": "no bbox"})
                continue

            print(f"    [{tool_name}] bbox={bbox}")
            agent_output = {"output": {"bounding_box": bbox}}

            try:
                eval_result = evaluator.evaluate(
                    agent_output=agent_output,
                    original_input=original_input,
                    model=model,
                    original_prediction=orig_pred,
                    processor=processor,
                    device=str(device),
                    dataset_base_name=dataset_key,
                    row_no=row_no,
                    tool_name=tool_name,
                )
                _record_result(tool_results[tool_name], row_no, bbox, eval_result)
            except Exception as e:
                print(f"      [ERROR] evaluate: {e}")
                tool_results[tool_name].append({
                    "row_no": row_no, "explanation": bbox,
                    "score": None, "passed": None, "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    return tool_results


# ─────────────────────────────────────────────────────────────────────────────
# Summarization
# ─────────────────────────────────────────────────────────────────────────────

def summarize(tool_results: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Compute per-tool mean score, mean raw_drop, pass rate, and error rate."""
    summary = {}
    for tool_name, results in tool_results.items():
        valid   = [r for r in results if r.get("score") is not None]
        n_total = len(results)
        n_valid = len(valid)
        if valid:
            scores     = [r["score"] for r in valid]
            raw_drops  = [r["raw_drop"] for r in valid if r.get("raw_drop") is not None]
            mean_score = sum(scores) / len(scores)
            mean_rd    = sum(raw_drops) / len(raw_drops) if raw_drops else None
            pass_rate  = sum(r["passed"] for r in valid) / len(valid)
        else:
            mean_score = mean_rd = pass_rate = None
        summary[tool_name] = {
            "n_total":      n_total,
            "n_valid":      n_valid,
            "n_errors":     n_total - n_valid,
            "mean_score":   mean_score,
            "mean_raw_drop": mean_rd,
            "pass_rate":    pass_rate,
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def print_combined_table(all_summaries: Dict[str, Dict[str, dict]]) -> None:
    W_DS   = 22
    W_TOOL = 22
    W_MS   = 12
    W_RD   = 10
    W_PR   = 11

    def hline(l, m, r, fill="─"):
        return (l + fill*(W_DS+2) + m + fill*(W_TOOL+2) + m
                + fill*(W_MS+2) + m + fill*(W_RD+2) + m + fill*(W_PR+2) + r)

    def row(ds, tool, ms, rd, pr):
        return (f"│ {ds:<{W_DS}} │ {tool:<{W_TOOL}} │ {ms:>{W_MS}} │"
                f" {rd:>{W_RD}} │ {pr:>{W_PR}} │")

    top    = hline("┌", "┬", "┐")
    sep    = hline("├", "┼", "┤")
    bottom = hline("└", "┴", "┘")

    print(f"\nTool-only baseline Q1  (tabular/text: top {int(TOP_FRAC*100)}% features, vision: top-1% bbox)\n")
    print(top)
    print(row("Dataset", "Tool", "Mean Score", "Raw Drop", "Pass Rate"))
    print(sep)

    ds_names = list(all_summaries.keys())
    for i, ds_name in enumerate(ds_names):
        summary   = all_summaries[ds_name]
        tool_list = list(summary.keys())
        for j, tool_name in enumerate(tool_list):
            s  = summary[tool_name]
            ms = f"{s['mean_score']:.4f}"    if s["mean_score"]    is not None else "N/A"
            rd = f"{s['mean_raw_drop']:.4f}" if s["mean_raw_drop"] is not None else "N/A"
            pr = f"{s['pass_rate']:.0%}"     if s["pass_rate"]     is not None else "N/A"
            print(row(ds_name, tool_name, ms, rd, pr))
            is_last_tool    = (j == len(tool_list) - 1)
            is_last_dataset = (i == len(ds_names) - 1)
            if is_last_tool and is_last_dataset:
                print(bottom)
            elif is_last_tool:
                print(sep)
            else:
                # thin separator within a dataset block
                print("│" + " "*(W_DS+2)
                      + "├" + "─"*(W_TOOL+2) + "┼" + "─"*(W_MS+2)
                      + "┼" + "─"*(W_RD+2)   + "┼" + "─"*(W_PR+2) + "┤")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def save_by_modality_tool(
    modality: str,
    name: str,
    tool_results: Dict[str, List[dict]],
    summary: Dict[str, dict],
) -> None:
    """Save results to outputs/baseline_tool_only/{modality}/{tool}/{dataset}.json"""
    for tool_name, results in tool_results.items():
        tool_dir = OUTPUT_DIR / modality / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        out = tool_dir / f"{name}.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "dataset":  name,
                    "modality": modality,
                    "tool":     tool_name,
                    "results":  results,
                    "summary":  summary[tool_name],
                },
                f, indent=2,
            )
    print(f"\n  Saved: {OUTPUT_DIR / modality}/<tool>/{name}.json  ({len(tool_results)} tools)")


def load_existing_tool_results(modality: str, dataset_key: str, tool_names: List[str]) -> Dict[str, List[dict]]:
    """Load per-tool results already saved to disk.  Returns only tools with n_valid > 0."""
    existing: Dict[str, List[dict]] = {}
    for tool_name in tool_names:
        path = OUTPUT_DIR / modality / tool_name / f"{dataset_key}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                s = data.get("summary", {})
                if s.get("n_valid", 0) > 0:
                    existing[tool_name] = data["results"]
                    print(f"  [RESUME] {modality}/{tool_name}/{dataset_key}: loaded {s['n_valid']}/{s['n_total']} valid results")
            except Exception:
                pass
    return existing


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip tools whose output already has n_valid > 0")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Root of test datasets (default: <trial2>/dataset/test)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <trial2>/outputs/baseline_tool_only)")
    args, _ = parser.parse_known_args()
    resume = args.resume

    global TABULAR_DIR, TEXT_DIR, VISION_DIR, OUTPUT_DIR
    if args.dataset_dir:
        base = Path(args.dataset_dir)
        TABULAR_DIR = base / "tabular"
        TEXT_DIR    = base / "text"
        VISION_DIR  = base / "vision"
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        set_masking_output_dir(str(OUTPUT_DIR))

    all_summaries: Dict[str, Dict[str, dict]] = {}

    # ── Tabular ──────────────────────────────────────────────────────────────
    for cfg in TABULAR_DATASETS:
        name = cfg["dataset_key"]
        existing = load_existing_tool_results("tabular", name, TABULAR_TOOL_NAMES) if resume else {}
        skip_tools = set(existing.keys())
        try:
            tool_results = evaluate_tabular_dataset(cfg, skip_tools=skip_tools)
        except Exception as e:
            print(f"\n[FATAL] {name}: {e}")
            traceback.print_exc()
            tool_results = {}
        tool_results.update(existing)
        if not tool_results:
            continue
        summary = summarize(tool_results)
        all_summaries[name] = summary
        save_by_modality_tool("tabular", name, tool_results, summary)

    # ── Text ─────────────────────────────────────────────────────────────────
    for cfg in TEXT_DATASETS:
        name = cfg["dataset_key"]
        existing = load_existing_tool_results("text", name, TEXT_TOOL_NAMES) if resume else {}
        skip_tools = set(existing.keys())
        try:
            tool_results = evaluate_text_dataset(cfg, skip_tools=skip_tools)
        except Exception as e:
            print(f"\n[FATAL] {name}: {e}")
            traceback.print_exc()
            tool_results = {}
        tool_results.update(existing)
        if not tool_results:
            continue
        summary = summarize(tool_results)
        all_summaries[name] = summary
        save_by_modality_tool("text", name, tool_results, summary)

    # ── Vision ───────────────────────────────────────────────────────────────
    for cfg in VISION_DATASETS:
        name = cfg["dataset_key"]
        existing = load_existing_tool_results("vision", name, VISION_TOOL_NAMES) if resume else {}
        skip_tools = set(existing.keys())
        try:
            tool_results = evaluate_vision_dataset(cfg, skip_tools=skip_tools)
        except Exception as e:
            print(f"\n[FATAL] {name}: {e}")
            traceback.print_exc()
            tool_results = {}
        tool_results.update(existing)
        if not tool_results:
            continue
        summary = summarize(tool_results)
        all_summaries[name] = summary
        save_by_modality_tool("vision", name, tool_results, summary)

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_summaries:
        print_combined_table(all_summaries)

    # ── Overall per-tool means across all datasets ────────────────────────────
    all_tool_names = sorted({t for s in all_summaries.values() for t in s})
    print(f"\n{'─'*74}")
    print(f"OVERALL  (mean of per-dataset means, all modalities)")
    print(f"{'─'*74}")
    print(f"  {'Tool':<24}  {'Mean Score':>10}  {'Raw Drop':>9}  {'Pass Rate':>9}")
    print(f"  {'-'*24}  {'-'*10}  {'-'*9}  {'-'*9}")

    for tool_name in all_tool_names:
        scores = [s[tool_name]["mean_score"]    for s in all_summaries.values() if tool_name in s and s[tool_name]["mean_score"]    is not None]
        rdrops = [s[tool_name]["mean_raw_drop"] for s in all_summaries.values() if tool_name in s and s[tool_name]["mean_raw_drop"] is not None]
        prates = [s[tool_name]["pass_rate"]     for s in all_summaries.values() if tool_name in s and s[tool_name]["pass_rate"]     is not None]
        ms = f"{sum(scores)/len(scores):.4f}"         if scores else "N/A"
        rd = f"{sum(rdrops)/len(rdrops):.4f}"         if rdrops else "N/A"
        pr = f"{sum(prates)/len(prates):.0%}"         if prates else "N/A"
        print(f"  {tool_name:<24}  {ms:>10}  {rd:>9}  {pr:>9}")

    overall_path = OUTPUT_DIR / "overall_summary.json"
    with open(overall_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nOverall summary saved to: {overall_path}")


if __name__ == "__main__":
    main()
