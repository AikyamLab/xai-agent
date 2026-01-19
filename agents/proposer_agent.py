"""
Proposer Agent for XAI Agent Framework

Responsible for:
- Analyzing the question type and modality
- Selecting appropriate XAI tools
- Creating execution strategies
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base_agent import BaseAgent


class ProposerAgent(BaseAgent):
    """
    Proposer Agent - Strategy Planning

    Generates XAI analysis strategies by:
    - Analyzing the question type and modality
    - Selecting appropriate XAI tools
    - Creating execution plans
    """

    def __init__(
        self,
        vlm: Any,
        data_model_loader: Any = None,
        models_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Proposer Agent.

        Args:
            vlm: VisionLanguageModel instance
            data_model_loader: DataModelLoader instance
            models_dir: Directory containing models
            output_dir: Output directory
        """
        super().__init__(vlm, output_dir, "ProposerAgent")

        self.data_model_loader = data_model_loader

        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), "models_to_read")
        self.models_dir = Path(models_dir).resolve()

        self.strategy_dir = self.output_dir / "strategies"
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Models directory: {self.models_dir}")

    def run(
        self,
        question: Dict[str, Any],
        question_template: Any,
        model_info: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - propose analysis strategy.

        Args:
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder instance
            model_info: Model information
            input_path: Path to input (image, text file, etc.)
            prediction: Model prediction results

        Returns:
            Strategy dictionary
        """
        return self.propose_strategy(
            question, question_template, model_info, input_path, prediction
        )[0]

    def propose_strategy(
        self,
        question: Dict[str, Any],
        question_template: Any,
        model_info: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Propose analysis strategy using VLM.

        Args:
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder instance
            model_info: Model information
            input_path: Path to input data
            prediction: Model prediction results

        Returns:
            Tuple of (strategy dict, saved model metadata path)
        """
        print("\n" + "=" * 70)
        print("PROPOSER AGENT: Planning Strategy")
        print("=" * 70)

        saved_model_metadata_path: Optional[str] = None
        q_type = question.get('q_type')
        modality = question.get('modality', 'vision')

        # Get clean model info
        clean_model_info = self._get_clean_model_info_dict(model_info)

        try:
            # Get PromptBuilder
            prompt_builder = self._get_prompt_builder(question_template, q_type, modality)

            if prompt_builder is not None:
                print(f"  Using PromptBuilder: {prompt_builder.__class__.__name__}")

                # Build context
                context = self._build_context(
                    question=question,
                    model_info=clean_model_info,
                    prediction=prediction or {},
                    input_path=input_path
                )

                # Generate strategy
                strategy = self._generate_strategy_with_prompt_builder(
                    prompt_builder=prompt_builder,
                    context=context
                )
            else:
                # Fallback to rule-based strategy
                print("  Using fallback strategy generation")
                question_type = self._get_question_type_string(q_type)
                strategy = self._get_strategy_by_question_type(question_type, modality)

            # Validate strategy
            if not strategy.get('selected_tools'):
                print("Warning: Strategy missing selected_tools, using default")
                return self._get_default_strategy(modality), saved_model_metadata_path

            # Save strategy
            self._save_strategy(strategy, question.get('question_id', 'unknown'))

            print(f"\nStrategy proposed: {strategy.get('strategy_type', 'unknown')}")
            print(f"  Selected {len(strategy.get('selected_tools', []))} tools")

            return strategy, saved_model_metadata_path

        except Exception as e:
            print(f"Warning: Strategy generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_strategy(modality), saved_model_metadata_path

    def _get_prompt_builder(
        self,
        question_template: Any,
        q_type: Optional[int],
        modality: str
    ) -> Any:
        """Get appropriate PromptBuilder"""
        # If question_template is already a PromptBuilder
        if hasattr(question_template, 'build_proposer_prompt'):
            return question_template

        # If question_template has a prompt_builder attribute
        if hasattr(question_template, 'prompt_builder') and question_template.prompt_builder:
            return question_template.prompt_builder

        # Try to import from prompts module
        try:
            from prompts import get_prompt_builder
            return get_prompt_builder(q_type, modality)
        except ImportError:
            pass

        return None

    def _build_context(
        self,
        question: Dict[str, Any],
        model_info: Dict[str, Any],
        prediction: Dict[str, Any],
        input_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build context dictionary for PromptBuilder"""
        modality = question.get("modality", "vision")

        context = {
            "user_question": question.get("question", ""),
            "model_info": model_info,
            "prediction": prediction,
        }

        if modality == "vision":
            context["image_path"] = input_path
            context["image_description"] = ""
        elif modality == "text":
            context["text_input"] = question.get("text_input", "")
            context["text_length"] = len(context["text_input"])
        elif modality == "tabular":
            context["input_data"] = question.get("features", {})
            context["feature_names"] = list(question.get("features", {}).keys())

        # Add target_class for counterfactual questions (Q5-Q7)
        if question.get("q_type") in [5, 6, 7]:
            context["target_class"] = question.get("target_class", "a different prediction")

        # Add ground_truth for spurious feature questions (Q8-Q10)
        if question.get("q_type") in [8, 9, 10]:
            target_info = question.get("target", {})
            context["ground_truth"] = target_info.get("label", target_info.get("value", "Unknown"))

        return context

    def _generate_strategy_with_prompt_builder(
        self,
        prompt_builder: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategy using PromptBuilder"""
        try:
            # Build prompt
            import inspect
            sig = inspect.signature(prompt_builder.build_proposer_prompt)
            if 'for_react_agent' in sig.parameters:
                prompt = prompt_builder.build_proposer_prompt(context, for_react_agent=False)
            else:
                prompt = prompt_builder.build_proposer_prompt(context)

            print(f"\n  Generated proposer prompt ({len(prompt)} chars)")

            # Call VLM
            response = self.invoke_vlm(prompt)
            print(f"  VLM Response preview: {response[:300]}...")

            # Parse response
            strategy = self.parse_json_response(response)

            # Convert tool_selection format if needed
            if not strategy.get('selected_tools') and strategy.get('tool_selection'):
                strategy = self._convert_tool_selection(strategy)

            return strategy if strategy.get('selected_tools') else self._get_default_strategy()

        except Exception as e:
            print(f"  Warning: VLM call failed: {e}")
            return self._get_default_strategy()

    def _convert_tool_selection(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tool_selection format to selected_tools format"""
        tool_selection = strategy.get('tool_selection', {})
        selected_tools = []

        for i, tool_name in enumerate(tool_selection.get('selected_tools', [])):
            tool_params = tool_selection.get('tool_params', {}).get(tool_name, {})
            priority = tool_params.pop('priority', i + 1) if isinstance(tool_params, dict) else i + 1
            selected_tools.append({
                "tool_name": tool_name,
                "priority": priority,
                "reasoning": tool_selection.get('reasoning', ''),
                "parameters": tool_params if isinstance(tool_params, dict) else {}
            })

        strategy['selected_tools'] = selected_tools
        return strategy

    def _get_question_type_string(self, q_type: Optional[int]) -> str:
        """Map question type ID to string"""
        if q_type is None:
            return "general"
        if 1 <= q_type <= 4:
            return "feature_attribution"
        elif 5 <= q_type <= 7:
            return "counterfactual"
        elif 8 <= q_type <= 10:
            return "spurious_features"
        return "general"

    def _get_strategy_by_question_type(
        self,
        question_type: str,
        modality: str = "vision"
    ) -> Dict[str, Any]:
        """Generate rule-based strategy"""
        # Tool priorities by question type and modality
        tool_priorities = {
            "vision": {
                "feature_attribution": ["gradcam", "integrated_gradients", "shap"],
                "counterfactual": ["gradcam", "lime", "integrated_gradients"],
                "spurious_features": ["integrated_gradients", "shap", "lime"],
                "general": ["gradcam", "integrated_gradients"]
            },
            "text": {
                "feature_attribution": ["lime", "shap", "integrated_gradients"],
                "counterfactual": ["lime", "shap"],
                "spurious_features": ["shap", "lime"],
                "general": ["lime", "shap"]
            },
            "tabular": {
                "feature_attribution": ["shap", "lime"],
                "counterfactual": ["shap", "lime"],
                "spurious_features": ["shap", "lime"],
                "general": ["shap", "lime"]
            }
        }

        modality_tools = tool_priorities.get(modality, tool_priorities["vision"])
        priority_list = modality_tools.get(question_type, modality_tools["general"])

        selected_tools = []
        for i, tool_name in enumerate(priority_list):
            selected_tools.append({
                "tool_name": tool_name,
                "priority": i + 1,
                "reasoning": f"Rule-based selection for {question_type}",
                "parameters": {}
            })

        return {
            "strategy_type": "tools",
            "reasoning": f"Rule-based selection for {question_type} on {modality}",
            "selected_tools": selected_tools,
            "autonomous_tasks": []
        }

    def _get_default_strategy(self, modality: str = "vision") -> Dict[str, Any]:
        """Get default fallback strategy"""
        default_tools = {
            "vision": [{"tool_name": "gradcam", "priority": 1, "reasoning": "Default", "parameters": {}}],
            "text": [{"tool_name": "lime", "priority": 1, "reasoning": "Default", "parameters": {}}],
            "tabular": [{"tool_name": "shap", "priority": 1, "reasoning": "Default", "parameters": {}}]
        }

        return {
            "strategy_type": "tools",
            "reasoning": f"Default strategy for {modality}",
            "selected_tools": default_tools.get(modality, default_tools["vision"]),
            "autonomous_tasks": []
        }

    def _get_clean_model_info_dict(self, model_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create clean dict representation of model info"""
        if not model_info:
            return {
                "model_name": "Unknown",
                "model_type": "Unknown",
                "architecture": "Unknown",
                "num_classes": "Unknown",
                "device": "Unknown"
            }

        return {
            "model_name": model_info.get("model_name", "Unknown"),
            "model_type": model_info.get("model_type", "Unknown"),
            "architecture": model_info.get("architecture", "Unknown"),
            "num_classes": model_info.get("num_classes", "Unknown"),
            "device": str(model_info.get("device", "Unknown"))
        }

    def _save_strategy(self, strategy: Dict[str, Any], question_id: str):
        """Save strategy to file"""
        filepath = self.save_json(strategy, f"strategy_{question_id}", "strategies")
        print(f"Strategy saved to: {filepath}")
