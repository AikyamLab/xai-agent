"""
Base Baseline Agent

Provides shared functionality for all baseline agents:
- VLM invocation (reuses the existing BaseAgent helpers)
- Context building (reuses PromptBuilder infrastructure)
- Output parsing & saving
- Evaluation hooks (reuses per-question evaluators)
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from agents.base_agent import BaseAgent


class BaseBaseline(BaseAgent):
    """
    Abstract base class for all baseline agents.

    Inherits from BaseAgent to reuse:
        - invoke_vlm()
        - parse_json_response()
        - save_json()

    Subclasses implement ``run()`` with a specific reasoning strategy.
    """

    # Human-readable label used in logs and output directories.
    baseline_name: str = "baseline"

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        if agent_name is None:
            agent_name = self.__class__.__name__
        super().__init__(vlm=vlm, output_dir=output_dir, agent_name=agent_name)

    # ------------------------------------------------------------------
    # Context helpers (mirror ProposerAgent._build_context)
    # ------------------------------------------------------------------

    def build_context(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        input_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the standard context dict consumed by PromptBuilders."""
        modality = question.get("modality", "vision")
        clean_model_info = self._clean_model_info(model_info)

        context: Dict[str, Any] = {
            "user_question": question.get("question", ""),
            "model_info": clean_model_info,
            "prediction": prediction or {},
        }

        if modality == "vision":
            context["image_path"] = input_path
            context["image_description"] = ""
        elif modality == "text":
            features = question.get("features", {})
            if isinstance(features, dict):
                text_content = features.get("text", features.get("premise", ""))
                if "premise" in features:
                    preview = (
                        f"Premise: {features['premise'][:200]}\n"
                        f"Hypothesis: {features.get('hypothesis', '')[:200]}"
                    )
                else:
                    preview = text_content[:400] + ("..." if len(text_content) > 400 else "")
            else:
                text_content = str(features)
                preview = text_content[:400]
            context["text_input"] = text_content
            context["text_length"] = len(text_content)
            context["text_description"] = preview
        elif modality == "tabular":
            features = question.get("features", {})
            context["input_data"] = features
            context["feature_names"] = (
                list(features.keys()) if isinstance(features, dict) else []
            )
            if isinstance(features, dict):
                feat_str = ", ".join(
                    f"{k}={v}" for k, v in list(features.items())[:10]
                )
                context["data_description"] = f"Features: {feat_str}"
            else:
                context["data_description"] = str(features)[:400]

        # Counterfactual questions need target_class
        if question.get("q_type") in [5, 6, 7]:
            context["target_class"] = question.get(
                "target_class", "a different prediction"
            )

        # Spurious feature questions need ground_truth
        if question.get("q_type") in [8, 9, 10]:
            target_info = question.get("target", {})
            if isinstance(target_info, list):
                context["ground_truth"] = [
                    (
                        t.get("label", t.get("value", "Unknown"))
                        if isinstance(t, dict)
                        else t
                    )
                    for t in target_info
                ]
            elif isinstance(target_info, dict):
                context["ground_truth"] = target_info.get(
                    "label", target_info.get("value", "Unknown")
                )
            else:
                context["ground_truth"] = target_info

        return context

    def build_context_multi(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        input_paths: List[str],
    ) -> Dict[str, Any]:
        """Build context for multi-instance questions (Q4, Q9, Q10)."""
        modality = question.get("modality", "vision")
        clean_model_info = self._clean_model_info(model_info)

        context: Dict[str, Any] = {
            "user_question": question.get("question", ""),
            "model_info": clean_model_info,
            "num_instances": len(predictions),
            "predictions": predictions,
            "input_paths": input_paths,
            "modality": modality,
        }

        for i, pred in enumerate(predictions):
            context[f"prediction_{i}"] = pred
        for i, path in enumerate(input_paths):
            context[f"image_path_{i}"] = path

        if predictions:
            context["prediction"] = predictions[0]
            context["prediction_A"] = predictions[0] if len(predictions) > 0 else {}
            context["prediction_B"] = predictions[1] if len(predictions) > 1 else {}
        if input_paths:
            context["image_path"] = input_paths[0]
            context["image_path_A"] = input_paths[0] if len(input_paths) > 0 else ""
            context["image_path_B"] = input_paths[1] if len(input_paths) > 1 else ""

        # Text / tabular instance data
        if modality in ("text", "tabular"):
            features_list = question.get("features", [])
            image_indices = question.get(
                "image_indices", question.get("row_no", [])
            )
            targets = question.get("targets", question.get("target", []))
            preds_q = question.get("predictions", question.get("predicted", []))

            instance_data = []
            for i in range(
                len(features_list) if isinstance(features_list, list) else 0
            ):
                feat = features_list[i]
                inst: Dict[str, Any] = {
                    "index": image_indices[i] if i < len(image_indices) else i
                }
                if modality == "text":
                    if "text" in feat:
                        inst["text"] = feat["text"]
                    elif "premise" in feat:
                        inst["premise"] = feat["premise"]
                        inst["hypothesis"] = feat.get("hypothesis", "")
                elif modality == "tabular":
                    inst["features"] = feat
                if isinstance(targets, list) and i < len(targets):
                    inst["target"] = targets[i]
                if isinstance(preds_q, list) and i < len(preds_q):
                    inst["predicted"] = preds_q[i]
                instance_data.append(inst)

            context["instance_data"] = instance_data

        return context

    # ------------------------------------------------------------------
    # Output format helpers
    # ------------------------------------------------------------------

    def get_output_format_instruction(self, modality: str, q_type: int) -> str:
        """Return the modality-specific JSON output format string."""
        if modality == "vision":
            base = '"bounding_box": [x_min, y_min, x_max, y_max]'
        elif modality == "text":
            base = '"start_index": int, "end_index": int'
        elif modality == "tabular":
            base = '"feature_key": "string"'
        else:
            base = '"output": {...}'

        if q_type == 5:
            return f'{base}, "prediction_after_masking": "yes" or "no", "explanation": "..."'
        elif q_type == 6:
            return f'"change_plan": {{"action": "change|delete|swap|add", {base}, "new_value": "..."}}, "expected_class": "...", "explanation": "..."'
        elif q_type == 7:
            return f'{base}, "predicted_class_after_change": "...", "explanation": "..."'
        else:
            return f'{base}, "description": "...", "explanation": "..."'

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_model_info(model_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not model_info:
            return {}
        return {
            "model_name": model_info.get("model_name", "Unknown"),
            "architecture": model_info.get("architecture", "Unknown"),
            "num_classes": model_info.get("num_classes", "Unknown"),
        }

    def save_baseline_result(
        self,
        result: Dict[str, Any],
        question: Dict[str, Any],
        suffix: str = "",
    ) -> Path:
        """Save result JSON under a structured directory."""
        modality = question.get("modality", "vision")
        dataset_base = question.get("dataset_base_name", "unknown")
        row_no = question.get("row_no", question.get("question_id", 0))

        match = re.match(r"(.+?)_(q\d+)(?:_.*)?$", dataset_base)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base
            q_type_str = f"q{question.get('q_type', 1)}"

        save_dir = (
            self.output_dir
            / self.baseline_name
            / modality
            / dataset_name
            / q_type_str
            / str(row_no)
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        fname = f"result{suffix}.json"
        filepath = save_dir / fname
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  [{self.baseline_name}] Result saved to: {filepath}")
        return filepath
