"""
Smoke-test for all six text XAI tools on imdb_2layernn and snli_2layernn.
Loads samples the same way the pipeline does (via loader_module.load_data(text_input)).
"""
import sys, json, os
sys.path.insert(0, "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2")
os.chdir("/standard/AikyamLab/yuyang/xai_agent/framework/trial_2")

import torch, numpy as np
from DataModelLoader import DataModelLoader
from xai_tools_native import (
    LIMETextTool,
    IntegratedGradientsTextTool,
    SHAPTextTool,
    SensitivityAnalysisTextTool,
    GuidedBackpropTextTool,
    SmoothGradTextTool,
)
from xai_tools import set_output_dir
from pathlib import Path


def load_text_sample(loader: DataModelLoader, json_path: str, idx: int = 0):
    """Load a sample the same way the pipeline does: read features from JSON, pass text to load_data."""
    with open(json_path) as f:
        samples = json.load(f)
    entry = samples[idx]
    features = entry["features"]
    # features may have 'review_text' or 'text'
    text_input = features.get("review_text") or features.get("text") or features.get("premise", "")
    data = loader.loader_module.load_data(text_input)
    loader.current_sample_data = data
    return entry


def check_result(tool_name: str, raw: str) -> bool:
    try:
        r = json.loads(raw)
    except Exception as e:
        print(f"  [FAIL] {tool_name}: JSON decode error — {e}")
        return False

    if not r.get("success", False):
        print(f"  [FAIL] {tool_name}: success=False — {r.get('error', '?')}")
        return False

    stats = r.get("statistics", {})
    # Each tool uses a different key for its ranked list
    top = (stats.get("top_tokens")
           or stats.get("top_features")
           or stats.get("top_words")
           or stats.get("token_importance")
           or stats.get("word_importance")
           or stats.get("word_sensitivity")
           or [])

    def get_score(item):
        for k in ("importance", "weight", "attribution_score", "shap_value", "prob_drop"):
            if k in item:
                return item[k]
        return None

    def get_label(item):
        for k in ("token", "word", "feature"):
            if k in item:
                return item[k]
        return "?"

    all_scores = sorted([s for t in top if isinstance(t, dict) for s in [get_score(t)] if s is not None])

    if not top:
        print(f"  [FAIL] {tool_name}: no ranked items found in statistics")
        return False

    if len(all_scores) >= 2 and all_scores[0] == all_scores[-1]:
        print(f"  [FAIL] {tool_name}: all importance scores identical ({all_scores[0]:.6f})")
        return False

    top1 = top[0]
    imp_val = get_score(top1)
    imp_str = f"{imp_val:.4f}" if isinstance(imp_val, (int, float)) else str(imp_val)
    n_unique = len(set(round(s, 6) for s in all_scores))
    print(f"  [OK]   {tool_name}: top='{get_label(top1)}' imp={imp_str}, unique={n_unique}/{len(all_scores)}")
    return True


def run_tools(loader: DataModelLoader, target_class: int, label: str):
    print(f"\n{'='*60}")
    print(f"Model: {label}  |  target_class={target_class}")
    print(f"text[:60]: {(loader.current_sample_data or {}).get('text','')[:60]}")
    print(f"{'='*60}")

    ctx = dict(data_model_loader=loader)
    tools = [
        ("lime",                 LIMETextTool(**ctx)),
        ("integrated_gradients", IntegratedGradientsTextTool(**ctx)),
        ("shap",                 SHAPTextTool(**ctx)),
        ("sensitivity_analysis", SensitivityAnalysisTextTool(**ctx)),
        ("guided_backprop",      GuidedBackpropTextTool(**ctx)),
        ("smoothgrad",           SmoothGradTextTool(**ctx)),
    ]

    passed = 0
    for name, tool in tools:
        try:
            raw = tool.run(target_class=target_class, image_id=f"test_{name}")
            ok = check_result(name, raw)
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: exception — {e}")
            traceback.print_exc()
            ok = False
        passed += ok

    print(f"\n  {passed}/{len(tools)} tools passed")
    return passed == len(tools)


def main():
    out_dir = Path("/tmp/test_text_tools_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(out_dir))

    all_pass = True

    # ── imdb_2layernn ──────────────────────────────────────────────
    print("\nLoading imdb_2layernn ...")
    loader_imdb = DataModelLoader("imdb_2layernn", "text")
    entry = load_text_sample(
        loader_imdb,
        "dataset_full/test/text/imdb_2layernn_q1.json",
        idx=0,
    )
    target = entry["target"]["value"]
    all_pass &= run_tools(loader_imdb, target_class=int(target), label="imdb_2layernn")

    # ── snli_2layernn ──────────────────────────────────────────────
    print("\nLoading snli_2layernn ...")
    loader_snli = DataModelLoader("snli_2layernn", "text")
    with open("dataset_full/test/text/snli_2layernn_q1.json") as f:
        snli_samples = json.load(f)
    snli_entry = snli_samples[0]
    snli_feat = snli_entry["features"]
    # NLI: pass the full features dict so load_data can handle premise+hypothesis
    snli_data = loader_snli.loader_module.load_data(snli_feat)
    loader_snli.current_sample_data = snli_data
    target_snli = snli_entry["target"]["value"]
    all_pass &= run_tools(loader_snli, target_class=int(target_snli), label="snli_2layernn")

    print("\n" + ("ALL PASSED" if all_pass else "SOME FAILED"))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
