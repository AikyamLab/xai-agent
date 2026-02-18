"""
Chain-of-Thought (CoT) Baseline Agent

Single-pass reasoning with explicit step-by-step thinking.
Uses ALL available XAI tools (like the full pipeline) but replaces
the Proposer→Actor→Critic orchestration with a single CoT prompt
that asks the VLM to reason through tool results step-by-step.

Differences from the 3-agent system:
- No separate strategy-planning step (Proposer)
- No separate feature-extraction VLM call
- No Critic evaluation / reflection loop
- Single VLM call with "think step-by-step" instruction
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from baselines.base_baseline import BaseBaseline
from baselines.tool_executor import ToolExecutor
from question_templates_new import get_question_template, get_prompt_builder


class CoTAgent(BaseBaseline):
    """
    Chain-of-Thought Agent — one VLM call with step-by-step reasoning.

    Pipeline:
        1. Run a default set of XAI tools (all available)
        2. Build a single CoT prompt with tool results + question
        3. VLM reasons step-by-step and produces final JSON answer
    """

    baseline_name = "cot"

    # Default tools to run per modality (uses all applicable tools)
    DEFAULT_TOOLS = {
        "vision": ["gradcam", "lime", "shap", "integrated_gradients"],
        "text": ["lime", "shap"],
        "tabular": ["lime", "shap"],
    }

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ):
        super().__init__(vlm=vlm, output_dir=output_dir, agent_name="CoTAgent")
        self.tool_executor = tool_executor

    def set_tool_executor(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor

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
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")
        is_multi = question.get("is_multi_instance", False)

        print("\n" + "=" * 70)
        print(f"COT AGENT: Q{q_type} ({modality})")
        print("=" * 70)

        # Step 1: Execute XAI tools
        all_tool_results: Dict[str, Any] = {}
        if self.tool_executor:
            target_class = (prediction or {}).get("predicted_class_idx", 0)
            tools_to_run = self._select_tools(modality)

            if is_multi and input_paths and predictions:
                # Run tools per instance
                for i, (path, pred) in enumerate(zip(input_paths, predictions)):
                    tc = pred.get("predicted_class_idx", 0) if pred else 0
                    inst_results = self.tool_executor.run_tools(
                        tool_names=tools_to_run,
                        target_class=tc,
                        image_path=path,
                        image_id=f"cot_inst{i}",
                        question=question,
                    )
                    all_tool_results[f"instance_{i}"] = inst_results
            else:
                all_tool_results = self.tool_executor.run_tools(
                    tool_names=tools_to_run,
                    target_class=target_class,
                    image_path=input_path or "",
                    image_id="cot",
                    question=question,
                )
        else:
            print("  Warning: No tool executor — running CoT without XAI tools")

        # Step 2: Build CoT prompt
        if is_multi and input_paths and predictions:
            prompt = self._build_cot_prompt_multi(
                question, model_info, predictions, input_paths, all_tool_results
            )
            images = [
                p for p in input_paths
                if p and not p.startswith("text_") and not p.startswith("tabular_")
            ]
        else:
            prompt = self._build_cot_prompt(
                question, model_info, prediction, input_path, all_tool_results
            )
            images = (
                [input_path]
                if input_path and modality == "vision"
                and not input_path.startswith("text_")
                and not input_path.startswith("tabular_")
                else None
            )

        # Step 3: Single VLM call
        print("  Calling VLM (Chain-of-Thought, single pass)...")
        response = self.invoke_vlm(prompt, images)
        parsed = self.parse_json_response(response)

        if not parsed:
            print("  Warning: failed to parse VLM response")
            parsed = {"explanation": response[:2000], "raw_parse_failed": True}

        # Wrap result
        result = {
            "baseline": "cot",
            "question_id": question.get("question_id", "unknown"),
            "question_type": q_type,
            "output": parsed.get("output", parsed),
            "explanation": parsed.get("explanation", ""),
            **parsed,
            "tool_results": all_tool_results,
            "visualization_paths": [],
        }

        self.save_baseline_result(result, question)
        print(f"  CoT agent complete for Q{q_type}")
        return result

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def _select_tools(self, modality: str) -> List[str]:
        """Select tools to run, filtered by what's actually available."""
        desired = self.DEFAULT_TOOLS.get(modality, ["lime", "shap"])
        if self.tool_executor:
            available = set(self.tool_executor.list_tools())
            return [t for t in desired if t in available]
        return desired

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_cot_prompt(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        input_path: Optional[str],
        tool_results: Dict[str, Any],
    ) -> str:
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")
        clean_model = self._clean_model_info(model_info)
        pred = prediction or {}
        output_fmt = self.get_output_format_instruction(modality, q_type)

        # Format tool results
        tool_section = ""
        if self.tool_executor and tool_results:
            tool_section = f"""
## XAI Tool Results
{self.tool_executor.format_tool_results(tool_results)}
"""

        # Modality-specific data section
        data_section = self._build_data_section(question, modality)

        prompt = f"""You are an expert XAI analyst. You have been given a question about a machine learning model's prediction, along with results from XAI analysis tools.

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
{data_section}{tool_section}
## Instructions
Think through this step-by-step:

1. **Understand the question**: What exactly is being asked about the model's prediction?
2. **Analyze the XAI tool results**: What do the attribution maps, importance scores, and coordinates tell us?
3. **Identify the key region/feature**: Based on the tool outputs, which specific part of the input is most relevant to answering this question?
4. **Formulate your answer**: Provide a clear, specific answer with precise coordinates/indices.

Think step by step, then provide your final answer as valid JSON.

## Required JSON format
{{
    "reasoning": "Your step-by-step reasoning here",
    "output": {{ {output_fmt} }},
    "explanation": "Your final concise explanation"
}}

JSON Response:"""

        return prompt

    def _build_cot_prompt_multi(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        input_paths: List[str],
        tool_results: Dict[str, Any],
    ) -> str:
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")
        clean_model = self._clean_model_info(model_info)
        output_fmt = self.get_output_format_instruction(modality, q_type)

        # Instance sections
        inst_sections = []
        for i, pred in enumerate(predictions):
            label = chr(65 + i)
            section = (
                f"### Instance {label}\n"
                f"- Predicted: {pred.get('predicted_class_name', pred.get('predicted_class_idx', 'Unknown'))}\n"
                f"- Confidence: {pred.get('confidence', 0.0):.4f}\n"
            )
            inst_key = f"instance_{i}"
            if inst_key in tool_results and self.tool_executor:
                section += f"\n**Tool Results:**\n{self.tool_executor.format_tool_results(tool_results[inst_key])}\n"
            inst_sections.append(section)

        prompt = f"""You are an expert XAI analyst. Answer the following multi-instance question about a model's predictions.

## Question
{question.get("question", "")}

## Model Information
- Model: {clean_model.get("model_name", "Unknown")}
- Architecture: {clean_model.get("architecture", "Unknown")}

## Instances
{chr(10).join(inst_sections)}

## Instructions
Think step-by-step:
1. Analyze each instance's XAI results independently
2. Compare the instances to identify similarities and differences
3. Formulate your answer addressing each instance

## Required JSON format
{{
    {", ".join(f'"input_{chr(65+i)}": {{ {output_fmt} }}' for i in range(len(predictions)))},
    "reasoning": "Your step-by-step reasoning",
    "explanation": "Your comparative explanation"
}}

JSON Response:"""

        return prompt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_data_section(self, question: Dict[str, Any], modality: str) -> str:
        if modality == "text":
            features = question.get("features", {})
            if isinstance(features, dict):
                if "premise" in features:
                    return f"\n## Input Text\nPremise: {features['premise']}\nHypothesis: {features.get('hypothesis', '')}\n"
                else:
                    text = features.get("text", features.get("review_text", str(features)[:500]))
                    return f"\n## Input Text\n{text}\n"
        elif modality == "tabular":
            features = question.get("features", {})
            if isinstance(features, dict):
                feat_str = "\n".join(f"  {k}: {v}" for k, v in features.items())
                return f"\n## Input Features\n{feat_str}\n"
        return ""
