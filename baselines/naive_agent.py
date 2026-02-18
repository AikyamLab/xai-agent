"""
Naive Agent Baseline

The simplest possible baseline: asks the VLM to answer the XAI question
directly with **zero** reasoning scaffolding, no chain-of-thought, no tools.
Just a single prompt → single JSON answer.

This establishes a lower bound on performance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from baselines.base_baseline import BaseBaseline
from question_templates_new import get_question_template, get_prompt_builder


class NaiveAgent(BaseBaseline):
    """
    Naive Agent — single VLM call, no tools, no reasoning scaffolding.

    Pipeline:
        1. Build a minimal prompt with question + model info + prediction
        2. Ask the VLM to answer in structured JSON
        3. Parse and return
    """

    baseline_name = "naive"

    def run(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None,
        # Multi-instance
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the naive baseline.

        Returns a result dict with the same schema as the 3-agent system
        so it can be evaluated with the same evaluators.
        """
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")
        is_multi = question.get("is_multi_instance", False)

        print("\n" + "=" * 70)
        print(f"NAIVE AGENT: Q{q_type} ({modality})")
        print("=" * 70)

        # Build prompt
        if is_multi and input_paths and predictions:
            prompt = self._build_naive_prompt_multi(
                question, model_info, predictions, input_paths
            )
            images = [p for p in input_paths if p and not p.startswith("text_") and not p.startswith("tabular_")]
        else:
            prompt = self._build_naive_prompt(
                question, model_info, prediction, input_path
            )
            images = (
                [input_path]
                if input_path
                and modality == "vision"
                and not input_path.startswith("text_")
                and not input_path.startswith("tabular_")
                else None
            )

        # Single VLM call
        print("  Calling VLM (single pass, no tools)...")
        response = self.invoke_vlm(prompt, images)
        parsed = self.parse_json_response(response)

        if not parsed:
            print("  Warning: failed to parse VLM response, wrapping raw text")
            parsed = {"explanation": response[:2000], "raw_parse_failed": True}

        # Wrap in standard result format
        result = {
            "baseline": "naive",
            "question_id": question.get("question_id", "unknown"),
            "question_type": q_type,
            "output": parsed.get("output", parsed),
            "explanation": parsed.get("explanation", ""),
            **parsed,
            "tool_results": {},
            "visualization_paths": [],
        }

        self.save_baseline_result(result, question)
        print(f"  Naive agent complete for Q{q_type}")
        return result

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_naive_prompt(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        input_path: Optional[str],
    ) -> str:
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")
        clean_model = self._clean_model_info(model_info)
        pred = prediction or {}

        output_fmt = self.get_output_format_instruction(modality, q_type)

        prompt = f"""You are an AI model explanation expert. Answer the following question about a machine learning model's prediction.

## Question
{question.get("question", "")}

## Model Information
- Model: {clean_model.get("model_name", "Unknown")}
- Architecture: {clean_model.get("architecture", "Unknown")}
- Number of classes: {clean_model.get("num_classes", "Unknown")}

## Prediction
- Predicted class: {pred.get("predicted_class_name", pred.get("predicted_class_idx", "Unknown"))}
- Confidence: {pred.get("confidence", 0.0):.4f}
- Top predictions: {pred.get("top5_predictions", "N/A")}

## Modality: {modality}
"""
        # Add modality-specific data
        if modality == "text":
            features = question.get("features", {})
            if isinstance(features, dict):
                if "premise" in features:
                    prompt += f"\n## Input Text\nPremise: {features['premise']}\nHypothesis: {features.get('hypothesis', '')}\n"
                else:
                    prompt += f"\n## Input Text\n{features.get('text', features.get('review_text', str(features)[:500]))}\n"
        elif modality == "tabular":
            features = question.get("features", {})
            if isinstance(features, dict):
                feat_str = "\n".join(f"  {k}: {v}" for k, v in features.items())
                prompt += f"\n## Input Features\n{feat_str}\n"

        prompt += f"""
## Instructions
Answer the question directly based on your understanding. Provide your answer as valid JSON.

## Required JSON format
{{
    "output": {{ {output_fmt} }},
    "explanation": "Your explanation here"
}}

JSON Response:"""

        return prompt

    def _build_naive_prompt_multi(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        input_paths: List[str],
    ) -> str:
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")
        clean_model = self._clean_model_info(model_info)
        output_fmt = self.get_output_format_instruction(modality, q_type)

        # Instance sections
        inst_sections = []
        for i, pred in enumerate(predictions):
            label = chr(65 + i)  # A, B, C, ...
            section = (
                f"### Instance {label}\n"
                f"- Predicted class: {pred.get('predicted_class_name', pred.get('predicted_class_idx', 'Unknown'))}\n"
                f"- Confidence: {pred.get('confidence', 0.0):.4f}\n"
            )
            inst_sections.append(section)

        prompt = f"""You are an AI model explanation expert. Answer the following question about a machine learning model's predictions on multiple instances.

## Question
{question.get("question", "")}

## Model Information
- Model: {clean_model.get("model_name", "Unknown")}
- Architecture: {clean_model.get("architecture", "Unknown")}
- Number of classes: {clean_model.get("num_classes", "Unknown")}

## Instances
{chr(10).join(inst_sections)}

## Modality: {modality}

## Instructions
Answer the question directly. Provide your answer as valid JSON with output for each instance.

## Required JSON format
{{
    {", ".join(f'"input_{chr(65+i)}": {{ {output_fmt} }}' for i in range(len(predictions)))},
    "explanation": "Your explanation here"
}}

JSON Response:"""

        return prompt
