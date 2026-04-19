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
        self.tool_registry = None  # Set via set_tool_registry() after ActorAgent.initialize_tools()

        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), "models_to_read")
        self.models_dir = Path(models_dir).resolve()

        self.strategy_dir = self.output_dir / "strategies"
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Models directory: {self.models_dir}")

    def set_tool_registry(self, tool_registry: Any) -> None:
        """
        Provide the proposer with the actual tool registry so that
        build_proposer_prompt receives a live available_tools dict
        instead of the hardcoded fallback lists in each PromptBuilder.

        Call this after ActorAgent.initialize_tools() in the pipeline:
            self.proposer.set_tool_registry(self.actor.tool_registry)
        """
        self.tool_registry = tool_registry

    def _get_tool_info_for_context(self, modality: str) -> Dict[str, str]:
        """
        Return {tool_name: description} for tools that are actually available
        in the registry for the given modality.

        Falls back to an empty dict when no registry is set; prompt builders
        will then use their own hardcoded fallback values.
        """
        if self.tool_registry is not None:
            return self.tool_registry.get_tool_descriptions()
        return {}

    def run(
        self,
        question: Dict[str, Any],
        question_template: Any,
        model_info: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None,
        prediction: Optional[Dict[str, Any]] = None,
        # Multi-instance parameters (for Q4, Q9, Q10)
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - propose analysis strategy.

        Args:
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder instance
            model_info: Model information
            input_path: Path to input (single instance)
            prediction: Model prediction results (single instance)
            input_paths: List of paths for multi-instance questions
            predictions: List of predictions for multi-instance questions

        Returns:
            Strategy dictionary
        """
        return self.propose_strategy(
            question, question_template, model_info, input_path, prediction,
            input_paths=input_paths, predictions=predictions
        )[0]

    def propose_strategy(
        self,
        question: Dict[str, Any],
        question_template: Any,
        model_info: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None,
        prediction: Optional[Dict[str, Any]] = None,
        # Multi-instance parameters
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Propose analysis strategy using VLM.

        Handles both single-instance and multi-instance questions.
        For multi-instance (Q4, Q9, Q10), uses build_proposer_prompt_multi if available.

        Args:
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder instance
            model_info: Model information
            input_path: Path to input data (single instance)
            prediction: Model prediction results (single instance)
            input_paths: List of paths for multi-instance questions
            predictions: List of predictions for multi-instance questions

        Returns:
            Tuple of (strategy dict, saved model metadata path)
        """
        is_multi_instance = question.get('is_multi_instance', False)
        num_instances = question.get('num_instances', 1)
        q_type = question.get('q_type')

        print("\n" + "=" * 70)
        if is_multi_instance:
            print(f"PROPOSER AGENT: Planning Strategy for Q{q_type} ({num_instances} instances)")
        else:
            print("PROPOSER AGENT: Planning Strategy")
        print("=" * 70)

        saved_model_metadata_path: Optional[str] = None
        modality = question.get('modality', 'vision')

        # Get clean model info
        clean_model_info = self._get_clean_model_info_dict(model_info)

        # Get PromptBuilder
        prompt_builder = self._get_prompt_builder(question_template, q_type, modality)
        if prompt_builder is None:
            raise RuntimeError(f"No PromptBuilder found for Q{q_type} modality={modality}")
        print(f"  Using PromptBuilder: {prompt_builder.__class__.__name__}")

        if is_multi_instance and input_paths and predictions:
            # Multi-instance: build context with all instances
            context = self._build_context_multi(
                question=question,
                model_info=clean_model_info,
                predictions=predictions,
                input_paths=input_paths
            )

            # Build instances list for multi-instance prompt
            instances = [
                {
                    'prediction': pred,
                    'path': path,
                    'ground_truth': pred.get('ground_truth_name')
                }
                for pred, path in zip(predictions, input_paths)
            ]

            # Use multi-instance prompt if available
            if hasattr(prompt_builder, 'build_proposer_prompt_multi'):
                print(f"  Using build_proposer_prompt_multi for {num_instances} instances")
                strategy = self._generate_strategy_multi(
                    prompt_builder=prompt_builder,
                    context=context,
                    instances=instances,
                    question=question
                )
            else:
                raise RuntimeError(
                    f"PromptBuilder {prompt_builder.__class__.__name__} missing build_proposer_prompt_multi "
                    f"for multi-instance Q{q_type}"
                )

            # Mark as multi-instance strategy
            strategy['is_multi_instance'] = True
            strategy['num_instances'] = num_instances
        else:
            # Single instance: original logic
            context = self._build_context(
                question=question,
                model_info=clean_model_info,
                prediction=prediction or {},
                input_path=input_path
            )

            strategy = self._generate_strategy_with_prompt_builder(
                prompt_builder=prompt_builder,
                context=context,
                question=question
            )

        # Pure-reasoning strategy is valid: actor will skip tool execution and use VLM reasoning directly
        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            print("  [Proposer] Pure-reasoning strategy: no tools or autonomous tasks selected. "
                  "Actor will generate explanation via direct VLM reasoning.")

        # Save strategy
        self._save_strategy(strategy, question)

        print(f"\nStrategy proposed: {strategy.get('strategy_type', 'unknown')}")
        print(f"  Selected {len(strategy.get('selected_tools', []))} tools")

        return strategy, saved_model_metadata_path

    def _get_prompt_builder(
        self,
        question_template: Any,
        q_type: Optional[int],
        modality: str
    ) -> Any:
        """Get appropriate PromptBuilder"""
        # If question_template has a prompt_builder attribute (QuestionTemplate dataclass)
        if hasattr(question_template, 'prompt_builder') and question_template.prompt_builder:
            return question_template.prompt_builder

        # If question_template is already a PromptBuilder (has build_proposer_prompt method)
        # but is NOT a QuestionTemplate dataclass
        if hasattr(question_template, 'build_proposer_prompt') and not hasattr(question_template, 'prompt_builder'):
            return question_template

        from prompts import get_prompt_builder
        return get_prompt_builder(q_type, modality)

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
            "available_tools": self._get_tool_info_for_context(modality),
        }

        if modality == "vision":
            context["image_path"] = input_path
        elif modality == "text":
            features = question.get("features", {})
            if isinstance(features, dict):
                text_content = features.get("text", features.get("premise", ""))
                if "premise" in features:
                    preview = f"Premise: {features['premise'][:200]}\nHypothesis: {features.get('hypothesis', '')[:200]}"
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
            context["feature_names"] = list(features.keys()) if isinstance(features, dict) else []
            if isinstance(features, dict):
                feat_names = list(features.keys())
                context["data_description"] = f"Feature names: {', '.join(feat_names)}"
            else:
                context["data_description"] = str(features)[:400]

        # Add target_class / queried_part / part_to_change for counterfactual questions (Q5-Q7)
        if question.get("q_type") in [5, 6, 7]:
            extracted = self._extract_question_context_fields(question)
            context["target_class"] = extracted.get("target_class", question.get("target_class", "a different prediction"))
            if "queried_part" in extracted:
                context["queried_part"] = extracted["queried_part"]
            if "part_to_change" in extracted:
                context["part_to_change"] = extracted["part_to_change"]

        # Add ground_truth for spurious feature questions (Q8-Q10)
        if question.get("q_type") in [8, 9, 10]:
            context["ground_truth"] = prediction.get('ground_truth_name')

        return context

    def _build_context_multi(
        self,
        question: Dict[str, Any],
        model_info: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        input_paths: List[str]
    ) -> Dict[str, Any]:
        """Build context dictionary for multi-instance questions (Q4, Q9, Q10)."""
        modality = question.get("modality", "vision")
        num_instances = len(predictions)

        context = {
            "user_question": question.get("question", ""),
            "model_info": model_info,
            "num_instances": num_instances,
            "available_tools": self._get_tool_info_for_context(modality),
        }

        # Add all predictions
        context["predictions"] = predictions
        if predictions:
            context["prediction"] = predictions[0]  # For compatibility

        # Add all input paths
        context["input_paths"] = input_paths
        if input_paths:
            context["image_path"] = input_paths[0]  # For compatibility

        # Add indexed access (prediction_0, prediction_1, etc.)
        for i, pred in enumerate(predictions):
            context[f"prediction_{i}"] = pred

        for i, path in enumerate(input_paths):
            if modality == "vision":
                context[f"image_path_{i}"] = path
                context[f"image_description_{i}"] = f"Image at index {question.get('image_indices', [])[i] if i < len(question.get('image_indices', [])) else 'unknown'}"

        # For Q4 vision: set _A/_B aliases for the A/B prompt template
        if question.get('q_type') == 4 and modality == "vision":
            context['image_description_A'] = context.get('image_description_0', 'Image A')
            context['image_description_B'] = context.get('image_description_1', 'Image B')

        # Add targets from question
        targets = question.get('targets', [])
        context["targets"] = targets
        for i, target in enumerate(targets):
            context[f"target_{i}"] = target

        # Add ground_truth for spurious feature questions (Q8-Q10)
        if question.get("q_type") in [8, 9, 10]:
            context["ground_truth"] = targets

        # Add text/tabular content descriptions for proposer prompts
        if modality in ('text', 'tabular'):
            features_list = question.get('features', [])
            image_indices = question.get('image_indices', question.get('row_no', []))
            q_type = question.get('q_type')
            if isinstance(features_list, list):
                descriptions = []
                for i, feat in enumerate(features_list):
                    idx = image_indices[i] if i < len(image_indices) else i
                    if modality == 'text':
                        text = feat.get('text', feat.get('premise', ''))
                        preview = text[:150] + "..." if len(text) > 150 else text
                        descriptions.append(f"Instance {chr(65+i)} (index {idx}): \"{preview}\"")
                    elif modality == 'tabular':
                        # Proposer only needs instance labels/indices; feature names listed once below
                        descriptions.append(f"Instance {chr(65+i)} (index {idx})")
                desc_key = 'text_description' if modality == 'text' else 'data_description'
                if modality == 'tabular' and features_list and isinstance(features_list[0], dict):
                    feat_names = list(features_list[0].keys())
                    context[desc_key] = "\n".join(descriptions) + f"\nFeature names: {', '.join(feat_names)}"
                else:
                    context[desc_key] = "\n".join(descriptions)
                # For Q4: also expose per-instance _A/_B keys used by the A/B prompt template
                if q_type == 4:
                    if len(descriptions) >= 1:
                        context[desc_key + '_A'] = descriptions[0]
                    if len(descriptions) >= 2:
                        context[desc_key + '_B'] = descriptions[1]

        return context

    def _generate_strategy_multi(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        instances: List[Dict[str, Any]],
        question: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate strategy using multi-instance prompt builder."""
        prompt = prompt_builder.build_proposer_prompt_multi(context, instances)
        print(f"\n  Generated multi-instance proposer prompt ({len(prompt)} chars)")

        if question:
            self._save_prompt(prompt, question, "proposer_prompt")

        strategy = self.invoke_vlm_for_json(prompt)

        # Convert tool_selection format if needed
        if not strategy.get('selected_tools') and strategy.get('tool_selection'):
            strategy = self._convert_tool_selection(strategy)

        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            print("  [Proposer] Pure-reasoning multi-instance strategy: actor will use direct VLM reasoning.")

        return strategy

    def _generate_strategy_with_prompt_builder(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        question: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate strategy using PromptBuilder"""
        # Build prompt
        import inspect
        sig = inspect.signature(prompt_builder.build_proposer_prompt)
        if 'for_react_agent' in sig.parameters:
            prompt = prompt_builder.build_proposer_prompt(context, for_react_agent=False)
        else:
            prompt = prompt_builder.build_proposer_prompt(context)

        print(f"\n  Generated proposer prompt ({len(prompt)} chars)")

        if question:
            self._save_prompt(prompt, question, "proposer_prompt")

        strategy = self.invoke_vlm_for_json(prompt)

        # Convert tool_selection format if needed
        if not strategy.get('selected_tools') and strategy.get('tool_selection'):
            strategy = self._convert_tool_selection(strategy)

        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            print("  [Proposer] Pure-reasoning strategy: actor will use direct VLM reasoning.")

        return strategy

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
                # NOTE: 'parameters' is stored here but intentionally NOT applied at runtime.
                # ActorAgent._get_fixed_tool_params() provides deterministic fixed values instead.
                # To re-enable LLM-driven params, pass tool_spec['parameters'] in actor_agent._execute_tools().
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

    def _save_strategy(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        suffix: str = ""
    ):
        """Save strategy to file"""
        import re
        # Extract naming components from question
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))
        modality = question.get('modality', 'vision')

        # Extract dataset_name and q_type from dataset_base_name
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        # Format: /strategies/{modality}/{dataset_name}/{q_type}/{question_id}/strategy_r{rollout_id}.json
        rollout_id = question.get('rollout_id')
        rollout_pfx = f"_r{rollout_id}" if rollout_id is not None else ""
        filename = f"strategy{rollout_pfx}{suffix}" if (rollout_pfx or suffix) else "strategy"
        subdir = f"strategies/{modality}/{dataset_name}/{q_type_str}/{row_no}"
        filepath = self.save_json(strategy, filename, subdir)
        print(f"Strategy saved to: {filepath}")

    def run_with_reflection(
        self,
        question: Dict[str, Any],
        question_template: Any,
        model_info: Optional[Dict[str, Any]],
        input_path: Optional[str],
        prediction: Optional[Dict[str, Any]],
        proposer_reflection: str,
        original_strategy: Dict[str, Any],
        # Multi-instance parameters (for Q9, Q10)
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate improved strategy based on Critic's reflection.

        Args:
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder instance
            model_info: Model information
            input_path: Path to input data (single instance)
            prediction: Model prediction results (single instance)
            proposer_reflection: JSON string with feedback from Critic
            original_strategy: The original strategy that was evaluated
            input_paths: List of paths for multi-instance questions
            predictions: List of predictions for multi-instance questions

        Returns:
            Improved strategy dictionary
        """
        print("\n" + "=" * 70)
        print("PROPOSER AGENT: Re-planning Strategy with Reflection")
        print("=" * 70)

        q_type = question.get('q_type')
        modality = question.get('modality', 'vision')
        clean_model_info = self._get_clean_model_info_dict(model_info)

        # Get PromptBuilder
        prompt_builder = self._get_prompt_builder(question_template, q_type, modality)
        if prompt_builder is None:
            raise RuntimeError(f"No PromptBuilder found for Q{q_type} modality={modality}")

        # Build base context (multi-instance or single-instance)
        is_multi = question.get('is_multi_instance', False) and input_paths and predictions
        if is_multi:
            context = self._build_context_multi(
                question=question,
                model_info=clean_model_info,
                predictions=predictions,
                input_paths=input_paths
            )
        else:
            context = self._build_context(
                question=question,
                model_info=clean_model_info,
                prediction=prediction or {},
                input_path=input_path
            )

        # Add reflection information to context
        context['previous_strategy'] = original_strategy
        context['critic_feedback'] = proposer_reflection

        # Generate improved strategy with reflection
        strategy = self._generate_strategy_with_reflection(
            prompt_builder=prompt_builder,
            context=context,
            proposer_reflection=proposer_reflection,
            original_strategy=original_strategy,
            question=question
        )

        # Pure-reasoning strategy is valid: no tools/tasks means actor uses direct VLM reasoning
        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            print("  [Proposer] Pure-reasoning improved strategy: actor will use direct VLM reasoning.")

        # Save improved strategy
        self._save_strategy(strategy, question, suffix="_improved")

        print(f"\nImproved strategy proposed: {strategy.get('strategy_type', 'unknown')}")
        print(f"  Selected {len(strategy.get('selected_tools', []))} tools")

        return strategy

    # =========================================================================
    # Q4 Specific Methods (instance_A / instance_B format)
    # =========================================================================

    def run_q4(
        self,
        question: Dict[str, Any],
        question_template: Any,
        model_info: Optional[Dict[str, Any]] = None,
        instances: List[Dict[str, Any]] = None,  # [{'prediction': ..., 'path': ..., 'label': 'A/B'}]
        proposer_reflection: Optional[str] = None,
        original_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run Proposer for Q4 contrastive instances.

        Args:
            question: Question dict with instance_A, instance_B
            question_template: Q4ContrastiveInstancesPromptBuilder
            model_info: Model information
            instances: [{'prediction': pred_A, 'path': path_A, 'label': 'A'},
                        {'prediction': pred_B, 'path': path_B, 'label': 'B'}]

        Returns:
            Strategy dict
        """
        print("\n" + "=" * 70)
        print("PROPOSER AGENT: Planning Q4 Strategy (Instance A vs B)")
        print("=" * 70)

        clean_model_info = self._get_clean_model_info_dict(model_info)
        modality = question.get('modality', 'vision')

        # Get Q4 PromptBuilder
        prompt_builder = self._get_prompt_builder(question_template, 4, modality)
        if prompt_builder is None:
            raise RuntimeError(f"No PromptBuilder found for Q4 modality={modality}")
        print(f"  Using PromptBuilder: {prompt_builder.__class__.__name__}")

        # Build Q4-specific context
        context = self._build_context_q4(question, clean_model_info, instances)

        # Use build_proposer_prompt_multi (base class provides default implementation)
        base_prompt = prompt_builder.build_proposer_prompt_multi(context, instances)

        # If reflection provided, append it to the base prompt (same pattern as _generate_strategy_with_reflection)
        if proposer_reflection and original_strategy:
            print("  Applying critic reflection to Q4 strategy generation...")
            original_tools = [t.get('tool_name') for t in original_strategy.get('selected_tools', [])]
            reflection_prompt = f"""

You are the Proposer Agent. Your previous strategy did not achieve satisfactory explanation faithfulness.

## Previous Strategy
- Tools Used: {original_tools}
- Reasoning: {original_strategy.get('reasoning', 'N/A')}

## Critic's Feedback on Your Strategy
{proposer_reflection}

## Your Task
Based on the critic's feedback, generate an IMPROVED strategy for this contrastive Q4 question. Consider:
1. Which tools to keep based on their effectiveness across both instances A and B
2. Which tools to remove (low importance scores)
3. Which tools to add for better contrastive coverage
4. How to adjust tool priorities

Generate a new strategy in the same JSON format as before.
"""
            prompt = f"{base_prompt}\n\n{reflection_prompt}"
        else:
            prompt = base_prompt

        print(f"\n  Generated Q4 proposer prompt ({len(prompt)} chars)")
        self._save_prompt(prompt, question, "proposer_prompt")

        strategy = self.invoke_vlm_for_json(prompt)

        # Convert format if needed
        if not strategy.get('selected_tools') and strategy.get('tool_selection'):
            strategy = self._convert_tool_selection(strategy)

        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            print("  [Proposer] Pure-reasoning Q4 strategy: actor will use direct VLM reasoning.")

        # Mark as Q4 strategy
        strategy['is_q4'] = True
        strategy['num_instances'] = 2
        strategy['instance_labels'] = ['A', 'B']

        # Save strategy
        self._save_strategy(strategy, question)

        print(f"\nQ4 Strategy proposed: {strategy.get('strategy_type', 'unknown')}")
        print(f"  Selected {len(strategy.get('selected_tools', []))} tools")

        return strategy

    def _build_context_q4(
        self,
        question: Dict[str, Any],
        model_info: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context for Q4 contrastive instances."""
        instance_a = question.get('instance_A', {})
        instance_b = question.get('instance_B', {})

        modality = question.get('modality', 'vision')
        context = {
            "user_question": question.get("question", question.get("example", "")),
            "model_info": model_info,
            "num_instances": 2,
            "available_tools": self._get_tool_info_for_context(modality),
        }

        # Add predictions
        if instances and len(instances) >= 2:
            context["prediction_A"] = instances[0].get('prediction', {})
            context["prediction_B"] = instances[1].get('prediction', {})
            context["predictions"] = [
                instances[0].get('prediction', {}),
                instances[1].get('prediction', {})
            ]
            context["image_path_A"] = instances[0].get('path', '')
            context["image_path_B"] = instances[1].get('path', '')

        # Add original instance info
        context["instance_A"] = instance_a
        context["instance_B"] = instance_b

        # Add modality-appropriate descriptions
        if modality == 'vision':
            context["image_description_A"] = f"Image at index {instance_a.get('features', {}).get('image_index', 'unknown')}"
            context["image_description_B"] = f"Image at index {instance_b.get('features', {}).get('image_index', 'unknown')}"
        elif modality == 'text':
            context["text_description_A"] = str(instance_a.get('features', instance_a))
            context["text_description_B"] = str(instance_b.get('features', instance_b))
        else:  # tabular
            context["data_description_A"] = str(instance_a.get('features', instance_a))
            context["data_description_B"] = str(instance_b.get('features', instance_b))

        return context

    def _generate_strategy_with_reflection(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        proposer_reflection: str,
        original_strategy: Dict[str, Any],
        question: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate improved strategy using reflection feedback.

        Args:
            prompt_builder: PromptBuilder instance
            context: Context dictionary
            proposer_reflection: Critic's feedback for Proposer
            original_strategy: Original strategy

        Returns:
            Improved strategy dictionary
        """
        # Build reflection-aware prompt
        import json as _json
        original_tools = [t.get('tool_name') for t in original_strategy.get('selected_tools', [])]
        original_auto_tasks = original_strategy.get('autonomous_tasks', [])

        reflection_prompt = f"""You are the Proposer Agent. Your previous strategy did not achieve satisfactory explanation faithfulness.

## Previous Strategy
- Tools Used: {original_tools}
- Autonomous Tasks: {_json.dumps(original_auto_tasks, ensure_ascii=False)}
- Reasoning: {original_strategy.get('reasoning', 'N/A')}

## Critic's Feedback on Your Strategy
{proposer_reflection}

## Your Task
Based on the critic's feedback, generate an IMPROVED strategy. Consider:
1. Which tools to keep based on their effectiveness
2. Which tools to remove (low importance scores)
3. Which tools to add for better coverage
4. How to adjust tool priorities

Generate a new strategy in the same JSON format as before.
"""

        # Build base proposer prompt
        import inspect
        sig = inspect.signature(prompt_builder.build_proposer_prompt)
        if 'for_react_agent' in sig.parameters:
            base_prompt = prompt_builder.build_proposer_prompt(context, for_react_agent=False)
        else:
            base_prompt = prompt_builder.build_proposer_prompt(context)

        # Combine with reflection
        full_prompt = f"{base_prompt}\n\n{reflection_prompt}"

        print(f"\n  Generated reflection-aware prompt ({len(full_prompt)} chars)")

        if question:
            self._save_prompt(full_prompt, question, "proposer_prompt_improved")

        strategy = self.invoke_vlm_for_json(full_prompt)

        # Convert format if needed
        if not strategy.get('selected_tools') and strategy.get('tool_selection'):
            strategy = self._convert_tool_selection(strategy)

        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            print("  [Proposer] Pure-reasoning reflection strategy: actor will use direct VLM reasoning.")

        # Mark as improved
        strategy['_improved'] = True
        strategy['_original_strategy'] = original_strategy

        return strategy
