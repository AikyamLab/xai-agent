#!/usr/bin/env python3
"""
Baseline Pipeline Runner (Legacy Custom-Agent Path)

Runs baseline agents (naive, cot, react, tot) on the same benchmark data
as xai_pipeline_v2.py so results are directly comparable.

Note:
    This file is retained for legacy/explicit experiments that require the
    custom Naive/CoT/ReAct/ToT agent implementations.
    The canonical baseline route is:
      run_baselines.sh -> run_baseline_agent_batch.py -> run_baseline_agent.py
    which uses the standard pipeline backbone with --no-improvement --no-sf.

Usage:
    python run_baseline.py --baseline cot --dataset dataset/test/vision/stl10_resnet_q1.json \\
        --question_id 0 --model_url vision/stl10_resnet.pth

    python run_baseline.py --baseline react --dataset dataset/test/text/imdb_cnn_q1.json \\
        --question_id 0 --model_url text/imdb_cnn.pth

    # Run all baselines at once
    python run_baseline.py --baseline all --dataset dataset/test/vision/stl10_resnet_q1.json \\
        --question_id 0 --model_url vision/stl10_resnet.pth
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Reuse existing infrastructure
from vlm_wrapper import VisionLanguageModel, create_vlm
from DataModelLoader import DataModelLoader
from question_templates_new import get_question_template, QuestionTemplate
from evaluation import get_evaluator, get_masker, EvaluationResult, set_masking_output_dir

# Baseline agents
from baselines import NaiveAgent, CoTAgent, ReActAgent, ToTAgent
from baselines.tool_executor import ToolExecutor


# ---------------------------------------------------------------------------
# Reuse loader logic from xai_pipeline_v2
# ---------------------------------------------------------------------------

def load_model_loader_module(model_path: str, models_dir: str):
    """Dynamically load the loader module for a model (copied from xai_pipeline_v2)."""
    model_path = Path(model_path)
    model_name_stem = model_path.stem
    loader_module_name = f"load_{model_name_stem}.py"

    if model_path.is_absolute():
        loader_path = model_path.parent / loader_module_name
    else:
        loader_path = Path(models_dir) / model_path.parent / loader_module_name

    if not loader_path.exists():
        raise FileNotFoundError(f"Loader module not found: {loader_path}")

    spec = importlib.util.spec_from_file_location(loader_path.stem, loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    for func_name in ["load_model", "load_data", "predict"]:
        if not hasattr(loader_module, func_name):
            raise AttributeError(f"Loader module {loader_path} missing: {func_name}")

    return loader_module


# ---------------------------------------------------------------------------
# Baseline Pipeline
# ---------------------------------------------------------------------------

class BaselinePipeline:
    """
    End-to-end pipeline that mirrors XAIPipelineV2 but uses baseline agents.

    Reuses:
    - VLM initialization (vlm_wrapper.py)
    - DataModelLoader for model/data loading
    - Question template system (question_templates_new.py)
    - Evaluation framework (evaluation/)
    """

    BASELINE_CLASSES = {
        "naive": NaiveAgent,
        "cot": CoTAgent,
        "react": ReActAgent,
        "tot": ToTAgent,
    }

    def __init__(
        self,
        vlm_model_id: str = "Qwen3.6-35B-A3B",
        output_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        mode: str = "test",
    ):
        self.mode = mode
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs_baselines")
        if dataset_dir is None:
            dataset_dir = os.path.join(os.getcwd(), "dataset")
        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), "models_to_read")

        self.output_dir = Path(output_dir).resolve()
        self.dataset_dir = Path(dataset_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_masking_output_dir(str(self.output_dir))

        print("\n" + "=" * 70)
        print("INITIALIZING BASELINE PIPELINE")
        print("=" * 70)
        print(f"Mode: {self.mode}")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Models directory: {self.models_dir}")
        print(f"Output directory: {self.output_dir}")

        # Initialize VLM
        print("\nInitializing VLM...")
        self.vlm = create_vlm(model_id=vlm_model_id)

        # Agents will be created per-run (after tool executor is ready)
        self.data_model_loader: Optional[DataModelLoader] = None
        self.tool_executor: Optional[ToolExecutor] = None

        print("Baseline Pipeline initialized!\n")

    # ------------------------------------------------------------------
    # Question loading (mirrors XAIPipelineV2._load_question)
    # ------------------------------------------------------------------

    def _load_question(
        self, dataset_path: str, question_id: str
    ) -> Tuple[Dict[str, Any], QuestionTemplate]:
        """Load and normalize a question from a benchmark JSON."""
        print("\n=== Loading Question ===")

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        if isinstance(dataset, list):
            questions = dataset
        elif isinstance(dataset, dict):
            questions = dataset.get("questions", [dataset] if "q_type" in dataset else [])
        else:
            questions = []

        question = None
        try:
            idx = int(question_id)
            if 0 <= idx < len(questions):
                question = questions[idx]
        except ValueError:
            for q in questions:
                if q.get("question_id") == question_id:
                    question = q
                    break

        if question is None:
            raise ValueError(f"Question '{question_id}' not found")

        # Normalize
        question["question"] = question.get("example", question.get("q", ""))
        modality = question.get("modality")
        question["question_id"] = f"q{question_id}"

        q_type = question.get("q_type")
        question["modality"] = modality

        # Detect multi-instance
        row_no = question.get("row_no")
        has_ab = "instance_A" in question and "instance_B" in question
        is_multi = q_type in [4, 9, 10] or isinstance(row_no, list) or has_ab

        if is_multi:
            question["is_multi_instance"] = True
            if has_ab:
                ia = question["instance_A"]
                ib = question["instance_B"]
                image_indices = [
                    ia.get("features", {}).get("image_index"),
                    ib.get("features", {}).get("image_index"),
                ]
                targets = [ia.get("target", {}), ib.get("target", {})]
                preds = [ia.get("prediction", {}), ib.get("prediction", {})]
                question["is_q4_format"] = True
            else:
                image_indices = question.get(
                    "image_indices",
                    question.get("row_idx", question.get("row_no", [])),
                )
                if not isinstance(image_indices, list):
                    image_indices = [image_indices]
                targets = question.get("target", [])
                preds = question.get("predicted", [])
                if not isinstance(targets, list):
                    targets = [targets]
                if not isinstance(preds, list):
                    preds = [preds]

            question["image_indices"] = image_indices
            question["num_instances"] = len(image_indices)
            question["targets"] = targets
            question["predictions"] = preds
        else:
            question["is_multi_instance"] = False
            question["num_instances"] = 1

        template = get_question_template(q_type, modality)
        print(f"  Q{q_type} ({modality}) — {template.template[:60]}...")
        return question, template

    # ------------------------------------------------------------------
    # Model + data loading (mirrors XAIPipelineV2._load_model_and_data)
    # ------------------------------------------------------------------

    def _load_model_and_data(
        self,
        question: Dict[str, Any],
        model_url: Optional[str],
        image_path: Optional[str] = None,
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[str], Any]:
        """Load target model and input data."""
        print("\n=== Loading Model and Data ===")

        modality = question.get("modality")
        model_info = None
        prediction = None
        loaded_data_path = None
        input_tensor = None
        loader_module = None

        try:
            if model_url:
                if not os.path.isabs(model_url):
                    local = self.models_dir / model_url
                    if local.exists():
                        model_url = str(local)

                loader_module = load_model_loader_module(model_url, str(self.models_dir))
                model, processor = loader_module.load_model(model_url)

                extra_info = (
                    loader_module.get_model_info(model)
                    if hasattr(loader_module, "get_model_info")
                    else {}
                )

                model_info = {
                    "success": True,
                    "model": model,
                    "processor": processor,
                    "model_name": Path(model_url).name,
                    "model_type": "local_pth",
                    "model_path": model_url,
                    "device": str(
                        loader_module.DEVICE
                        if hasattr(loader_module, "DEVICE")
                        else "cpu"
                    ),
                    "architecture": model.__class__.__name__,
                    "num_classes": extra_info.get("num_classes"),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "label_map": extra_info.get("label_map", {}),
                    "feature_names": extra_info.get("feature_names") or [],
                }

                if modality == "tabular":
                    feats = question.get("features", {})
                    if isinstance(feats, dict) and feats:
                        model_info["feature_names"] = list(feats.keys())

                # DataModelLoader
                self.data_model_loader = DataModelLoader(
                    model_name=Path(model_url).stem,
                    modality=modality,
                    model_path=model_url,
                    models_dir=str(self.models_dir),
                )

                # Initialize ToolExecutor for baselines that use tools
                self.tool_executor = ToolExecutor(
                    data_model_loader=self.data_model_loader,
                    output_dir=str(self.output_dir),
                )

            # Load data per modality
            if modality == "vision" and loader_module:
                if image_path and os.path.exists(image_path):
                    from PIL import Image
                    input_tensor = Image.open(image_path).convert("RGB")
                    loaded_data_path = image_path
                    self.data_model_loader.current_sample_data = {
                        "image": input_tensor,
                        "image_path": image_path,
                    }
                else:
                    features = question.get("features", {})
                    sample_idx = features.get(
                        "image_index",
                        question.get("metadata", {}).get(
                            "row_no", question.get("row_no", 0)
                        ),
                    )
                    split = question.get("split", "test")
                    data = self.data_model_loader.load_sample(index=sample_idx, split=split)
                    input_tensor = data.get("image")
                    loaded_data_path = f"dataset_index_{sample_idx}"

                if self.data_model_loader:
                    prediction = self.data_model_loader.predict(input_tensor)

            elif modality == "text" and loader_module:
                features = question.get("features", {})
                if "premise" in features and "hypothesis" in features:
                    text_input = features
                elif "review_text" in features:
                    text_input = features.get("review_text", "")
                else:
                    text_input = features

                data = loader_module.load_data(text_input)
                self.data_model_loader.current_sample_data = data

                if "premise" in features:
                    input_tensor = {"premise": features["premise"], "hypothesis": features.get("hypothesis")}
                else:
                    input_tensor = data.get("text", text_input)

                self.data_model_loader.current_text_data = input_tensor
                self.data_model_loader.current_data_type = "text"
                loaded_data_path = "text_input"

                if model_info and model_info.get("model"):
                    prediction = loader_module.predict(
                        model=model_info["model"],
                        text_input=data,
                        tokenizer=model_info.get("processor"),
                    )

            elif modality == "tabular" and loader_module:
                row_no = question.get("row_no", 0)
                split = question.get("split", "test")
                q_type = question.get("q_type", 1)
                if q_type in (8, 9, 10):
                    split = "full"

                data = self.data_model_loader.load_sample(index=row_no, split=split)
                input_tensor = data.get("features")
                loaded_data_path = f"tabular_index_{row_no}"

                if model_info and data.get("feature_names"):
                    model_info["feature_names"] = data["feature_names"]
                    question["feature_names"] = data["feature_names"]

                if model_info and model_info.get("model"):
                    prediction = loader_module.predict(
                        model=model_info["model"],
                        input_data=input_tensor,
                        preprocessor=model_info.get("processor"),
                    )

        except Exception as e:
            print(f"Warning: Failed to load model/data: {e}")
            import traceback
            traceback.print_exc()

        return model_info, prediction, loaded_data_path, input_tensor

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        baseline_name: str,
        question_dataset_path: str,
        question_id: str,
        target_model_url: Optional[str] = None,
        image_path: Optional[str] = None,
        evaluate_faithfulness: bool = True,
        faithfulness_threshold: float = 0.1,
        tot_branches: int = 3,
        react_max_iter: int = 6,
    ) -> Dict[str, Any]:
        """
        Run a single baseline on a single question.

        Args:
            baseline_name: One of "naive", "cot", "react", "tot", "all"
            question_dataset_path: Path to benchmark JSON
            question_id: Question index
            target_model_url: Model path
            image_path: Optional direct image path
            evaluate_faithfulness: Run evaluator
            faithfulness_threshold: Passing threshold
            tot_branches: Number of ToT branches (K)
            react_max_iter: Max ReAct iterations

        Returns:
            Results dict (or dict of results for "all")
        """
        dataset_base_name = Path(question_dataset_path).stem

        # Load question
        question, template = self._load_question(question_dataset_path, question_id)
        question["dataset_base_name"] = dataset_base_name

        # Load model & data
        model_info, prediction, data_path, input_tensor = self._load_model_and_data(
            question, target_model_url, image_path
        )

        # Determine baselines to run
        if baseline_name == "all":
            baselines_to_run = ["naive", "cot", "react", "tot"]
        else:
            baselines_to_run = [baseline_name]

        all_results = {}
        for bname in baselines_to_run:
            print(f"\n{'='*70}")
            print(f"RUNNING BASELINE: {bname.upper()}")
            print(f"{'='*70}")

            agent = self._create_agent(bname, tot_branches, react_max_iter)

            # Run baseline
            result = agent.run(
                question=question,
                model_info=model_info,
                prediction=prediction,
                input_path=data_path,
            )

            # Evaluate faithfulness
            if evaluate_faithfulness and model_info and model_info.get("model"):
                evaluation = self._evaluate(
                    result, question, template, input_tensor,
                    prediction, model_info, faithfulness_threshold
                )
                result["evaluation"] = evaluation
            else:
                result["evaluation"] = {"status": "skipped"}

            # Save result with evaluation data
            agent.save_baseline_result(result, question)

            all_results[bname] = result

        if baseline_name == "all":
            # Save comparison
            self._save_comparison(all_results, question)
            return all_results
        else:
            return all_results[baseline_name]

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _create_agent(self, name: str, tot_branches: int = 3, react_max_iter: int = 6):
        """Create a baseline agent instance."""
        if name == "naive":
            return NaiveAgent(vlm=self.vlm, output_dir=str(self.output_dir))
        elif name == "cot":
            agent = CoTAgent(vlm=self.vlm, output_dir=str(self.output_dir))
            if self.tool_executor:
                agent.set_tool_executor(self.tool_executor)
            return agent
        elif name == "react":
            agent = ReActAgent(
                vlm=self.vlm,
                output_dir=str(self.output_dir),
                max_iterations=react_max_iter,
            )
            if self.tool_executor:
                agent.set_tool_executor(self.tool_executor)
            return agent
        elif name == "tot":
            agent = ToTAgent(
                vlm=self.vlm,
                output_dir=str(self.output_dir),
                num_branches=tot_branches,
            )
            if self.tool_executor:
                agent.set_tool_executor(self.tool_executor)
            return agent
        else:
            raise ValueError(f"Unknown baseline: {name}. Choose from: naive, cot, react, tot, all")

    # ------------------------------------------------------------------
    # Evaluation (reuses existing evaluators)
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        result: Dict[str, Any],
        question: Dict[str, Any],
        template: QuestionTemplate,
        input_tensor: Any,
        prediction: Optional[Dict[str, Any]],
        model_info: Dict[str, Any],
        threshold: float,
    ) -> Dict[str, Any]:
        """Evaluate baseline output using the same evaluators as the 3-agent system."""
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")

        print(f"\n  Evaluating faithfulness (Q{q_type})...")

        try:
            evaluator = get_evaluator(q_type, modality=modality)

            # Derive ground truth
            gt = question.get("ground_truth")
            if gt is None and prediction:
                gt = prediction.get("ground_truth_idx")
            if gt is None:
                target = question.get("target", {})
                if isinstance(target, dict):
                    gt = target.get("value")

            # Extract expected_class from agent output (for Q6+)
            expected_class = None
            if isinstance(result, dict):
                output_data = result.get('output', {})
                if isinstance(output_data, dict):
                    expected_class = output_data.get('expected_class')

            eval_result = evaluator.evaluate(
                agent_output=result,
                original_input=input_tensor,
                model=model_info["model"],
                original_prediction=prediction,
                ground_truth=gt,
                processor=model_info.get("processor"),
                device=model_info.get("device", "cuda"),
                class_names=model_info.get("label_map", {}),
                feature_names=model_info.get("feature_names", []),
                expected_class=expected_class,
            )

            # faithfulness = {
            #     "score": eval_result.score if eval_result else None,
            #     "passed": (
            #         eval_result.passed if eval_result else False
            #     ),
            #     "details": {
            #         "p_original": eval_result.p_original if eval_result else None,
            #         "p_modified": eval_result.p_modified if eval_result else None,
            #         "threshold": threshold,
            #     },
            # }

            faithfulness = eval_result.to_dict()

            status = "PASSED" if faithfulness["passed"] else "FAILED"
            print(f"  Faithfulness: {faithfulness['score']:.4f} ({status})")
            return {"faithfulness": faithfulness}

            # return 

        except Exception as e:
            print(f"  Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return {"faithfulness": {"score": None, "error": str(e)}}

    # ------------------------------------------------------------------
    # Comparison output
    # ------------------------------------------------------------------

    def _save_comparison(self, results: Dict[str, Dict], question: Dict[str, Any]):
        """Save side-by-side comparison of all baselines."""
        import re

        dataset_base = question.get("dataset_base_name", "unknown")
        row_no = question.get("row_no", question.get("question_id", 0))
        modality = question.get("modality", "vision")
        q_type = question.get("q_type", 1)

        match = re.match(r"(.+?)_(q\d+)(?:_.*)?$", dataset_base)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base
            q_type_str = f"q{q_type}"

        save_dir = (
            self.output_dir / "comparison" / modality / dataset_name
            / q_type_str / str(row_no)
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        comparison = {
            "question_id": question.get("question_id"),
            "question_type": q_type,
            "modality": modality,
            "baselines": {},
        }

        for bname, result in results.items():
            faith = result.get("evaluation", {}).get("faithfulness", {})
            comparison["baselines"][bname] = {
                "faithfulness_score": faith.get("score"),
                "faithfulness_passed": faith.get("passed"),
                "output_preview": str(result.get("output", ""))[:300],
                "explanation_preview": str(result.get("explanation", ""))[:300],
            }

        filepath = save_dir / "comparison.json"
        with open(filepath, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n  Comparison saved to: {filepath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run legacy custom baseline agents (Naive / CoT / ReAct / ToT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single baseline
    python run_baseline.py --baseline cot \\
        --dataset dataset/test/vision/stl10_resnet_q1.json \\
        --question_id 0 --model_url vision/stl10_resnet.pth

    # All baselines at once
    python run_baseline.py --baseline all \\
        --dataset dataset/test/vision/stl10_resnet_q1.json \\
        --question_id 0 --model_url vision/stl10_resnet.pth

    # Text modality
    python run_baseline.py --baseline react \\
        --dataset dataset/test/text/imdb_cnn_q1.json \\
        --question_id 0 --model_url text/imdb_cnn.pth

Available baselines: naive, cot, react, tot, all
""",
    )

    parser.add_argument(
        "--baseline", type=str, required=True,
        choices=["naive", "cot", "react", "tot", "all"],
        help="Baseline to run",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to benchmark JSON")
    parser.add_argument("--question_id", type=str, required=True, help="Question index")
    parser.add_argument("--model_url", type=str, default=None, help="Model path")
    parser.add_argument(
        "--vlm", type=str, default="Qwen3.6-35B-A3B",
        help="VLM model ID",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--faithfulness_threshold", type=float, default=0.1,
    )
    parser.add_argument("--tot_branches", type=int, default=3, help="ToT branch count")
    parser.add_argument("--react_max_iter", type=int, default=6, help="ReAct max iterations")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test"], default="test",
    )

    args = parser.parse_args()

    pipeline = BaselinePipeline(
        vlm_model_id=args.vlm,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        mode=args.mode,
    )

    results = pipeline.run(
        baseline_name=args.baseline,
        question_dataset_path=args.dataset,
        question_id=args.question_id,
        target_model_url=args.model_url,
        image_path=args.image_path,
        evaluate_faithfulness=not args.no_eval,
        faithfulness_threshold=args.faithfulness_threshold,
        tot_branches=args.tot_branches,
        react_max_iter=args.react_max_iter,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)

    if args.baseline == "all":
        for bname, res in results.items():
            faith = res.get("evaluation", {}).get("faithfulness", {})
            score = faith.get("score", "N/A")
            passed = faith.get("passed", "N/A")
            print(f"  {bname:>8}: faithfulness={score}  passed={passed}")
    else:
        faith = results.get("evaluation", {}).get("faithfulness", {})
        print(f"  Baseline: {args.baseline}")
        print(f"  Faithfulness Score: {faith.get('score', 'N/A')}")
        print(f"  Faithfulness Passed: {faith.get('passed', 'N/A')}")
        explanation = results.get("explanation", "N/A")
        if isinstance(explanation, str) and len(explanation) > 100:
            explanation = explanation[:100] + "..."
        print(f"  Explanation: {explanation}")


if __name__ == "__main__":
    main()
