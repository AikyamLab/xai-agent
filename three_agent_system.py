"""
Three Agent System - Native Implementation (No LangChain)

Implements the three-agent architecture without LangChain:
1. ProposerAgent: Strategy planning
2. ActorAgent: XAI tool execution
3. CriticAgent: Result evaluation
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

import torch

from vlm_wrapper import VisionLanguageModel
from question_templates import (
    QuestionTemplate,
    QuestionCategory,
    Modality,
    PromptBuilder,
    PROMPT_BUILDER_MAP,
    # Vision PromptBuilders
    VisionFeatureAttributionPromptBuilder,
    VisionCounterfactualPromptBuilder,
    VisionSpuriousFeaturesPromptBuilder,
    # Text PromptBuilders
    TextFeatureAttributionPromptBuilder,
    TextCounterfactualPromptBuilder,
    TextSpuriousFeaturesPromptBuilder,
    # Tabular PromptBuilders
    TabularFeatureAttributionPromptBuilder,
    TabularCounterfactualPromptBuilder,
    TabularSpuriousFeaturesPromptBuilder,
)
from xai_tools_native import XAIToolRegistry, create_xai_tools
from xai_tools import get_available_tools
from DataModelLoader import DataModelLoader


class ProposerAgent:
    """
    Proposer Agent - Native Implementation

    Generates XAI analysis strategies by:
    - Analyzing the question type and modality
    - Selecting appropriate XAI tools
    - Creating execution plans
    """

    def __init__(
        self,
        vlm: VisionLanguageModel,
        data_model_loader: DataModelLoader,
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
        self.vlm = vlm
        self.data_model_loader = data_model_loader

        # Set default directories
        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), "models_to_read")
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.models_dir = Path(models_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.strategy_dir = self.output_dir / "strategies"
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

        print("Proposer Agent initialized")
        print(f"  Models directory: {self.models_dir}")
        print(f"  Output directory: {self.output_dir}")

    def propose_strategy(
        self,
        question: Dict[str, Any],
        question_template: QuestionTemplate,
        model_info: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Propose analysis strategy using VLM.

        Args:
            question: Question dictionary
            question_template: QuestionTemplate instance
            model_info: Model information
            image_path: Path to image (for vision)
            prediction: Model prediction results

        Returns:
            Tuple of (strategy dict, saved model metadata path)
        """
        print("\n" + "=" * 70)
        print("PROPOSER AGENT: Planning Strategy")
        print("=" * 70)

        saved_model_metadata_path: Optional[str] = None

        # Map q_type to question type string
        q_type = question.get('q_type')
        question_type_str = self._get_question_type_string(q_type)

        # Get clean model info for prompt (as dict for PromptBuilder)
        clean_model_info_dict = self._get_clean_model_info_dict(model_info)

        try:
            # Get PromptBuilder from question_template or create one based on modality/category
            prompt_builder = self._get_prompt_builder(question_template, q_type)

            if prompt_builder is not None:
                print(f"  Using PromptBuilder: {prompt_builder.__class__.__name__}")

                # Build context for PromptBuilder
                context = self._build_proposer_context(
                    question=question,
                    model_info=clean_model_info_dict,
                    prediction=prediction or {},
                    image_path=image_path
                )

                # Generate strategy using PromptBuilder
                strategy = self._generate_strategy_with_prompt_builder(
                    prompt_builder=prompt_builder,
                    context=context
                )
            else:
                # Fallback to simple strategy generation
                print("  Using fallback strategy generation (no PromptBuilder)")
                clean_model_info = self._get_clean_model_info(model_info)
                strategy = self._generate_strategy(
                    question=question.get('question', ''),
                    question_type=question_type_str,
                    model_info=clean_model_info,
                    modality=str(question_template.modality.value) if isinstance(question_template.modality, Modality) else str(question_template.modality)
                )

            # Validate strategy
            if not strategy.get('selected_tools'):
                print("Warning: Strategy missing selected_tools, using default")
                return self._get_default_strategy(question_template.modality), saved_model_metadata_path

            # Save strategy
            self._save_strategy(strategy, question.get('question_id', 'unknown'))

            print(f"\nStrategy proposed: {strategy.get('strategy_type', 'unknown')}")
            print(f"  Selected {len(strategy.get('selected_tools', []))} tools")

            return strategy, saved_model_metadata_path

        except Exception as e:
            print(f"Warning: Strategy generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_strategy(question_template.modality), saved_model_metadata_path

    def _get_prompt_builder(
        self,
        question_template: QuestionTemplate,
        q_type: Optional[int] = None
    ) -> Optional[PromptBuilder]:
        """
        Get appropriate PromptBuilder based on modality and category.

        Args:
            question_template: QuestionTemplate instance
            q_type: Question type (1-10)

        Returns:
            PromptBuilder instance or None
        """
        # First, check if question_template has a prompt_builder
        if question_template.prompt_builder is not None:
            return question_template.prompt_builder

        # Otherwise, create one based on modality and category
        modality = question_template.modality
        category = question_template.category

        # Convert enums to strings for PROMPT_BUILDER_MAP lookup
        modality_str = modality.value if isinstance(modality, Modality) else str(modality)
        category_str = category.value if isinstance(category, QuestionCategory) else str(category)

        # Build the key for PROMPT_BUILDER_MAP
        key = f"{modality_str}_{category_str}"

        builder_class = PROMPT_BUILDER_MAP.get(key)
        if builder_class is None:
            print(f"  Warning: No PromptBuilder found for {key}")
            return None

        # Get attribution_type from question_template if available
        attribution_type = question_template.attribution_type

        # For feature_attribution builders, pass attribution_type
        if category_str == "feature_attribution" and attribution_type:
            try:
                return builder_class(attribution_type=attribution_type)
            except TypeError:
                return builder_class()
        else:
            return builder_class()

    def _build_proposer_context(
        self,
        question: Dict[str, Any],
        model_info: Dict[str, Any],
        prediction: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context dictionary for PromptBuilder.

        Args:
            question: Question dictionary
            model_info: Model information dict
            prediction: Prediction results
            image_path: Path to image

        Returns:
            Context dictionary for PromptBuilder
        """
        context = {
            "user_question": question.get("question", ""),
            "model_info": model_info,
            "prediction": prediction,
            "image_path": image_path,
            "image_description": "",  # Can be filled by VLM if needed
        }

        # Add modality-specific fields
        modality = question.get("modality", "vision")

        if modality == "text":
            context["text_input"] = question.get("text_input", "")
            context["text_length"] = len(context["text_input"])
        elif modality == "tabular":
            context["input_data"] = question.get("features", {})
            context["feature_names"] = list(question.get("features", {}).keys())
            context["feature_types"] = {}  # Can be inferred if needed

        # Add target_class for counterfactual questions
        if question.get("q_type") in [5, 6, 7]:
            # Try to get target class from question
            context["target_class"] = question.get("target_class", "a different prediction")

        # Add ground_truth for spurious feature questions
        if question.get("q_type") in [8, 9, 10]:
            target_info = question.get("target", {})
            context["ground_truth"] = target_info.get("label", target_info.get("value", "Unknown"))

        return context

    def _generate_strategy_with_prompt_builder(
        self,
        prompt_builder: PromptBuilder,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate strategy using PromptBuilder.

        Args:
            prompt_builder: PromptBuilder instance
            context: Context dictionary

        Returns:
            Strategy dictionary
        """
        # Build proposer prompt using the PromptBuilder
        try:
            # Check if build_proposer_prompt accepts for_react_agent parameter
            import inspect
            sig = inspect.signature(prompt_builder.build_proposer_prompt)
            if 'for_react_agent' in sig.parameters:
                prompt = prompt_builder.build_proposer_prompt(context, for_react_agent=False)
            else:
                prompt = prompt_builder.build_proposer_prompt(context)
        except Exception as e:
            print(f"  Warning: PromptBuilder.build_proposer_prompt failed: {e}")
            prompt = prompt_builder.build_proposer_prompt(context)

        print(f"\n  Generated proposer prompt ({len(prompt)} chars)")
        print(f"  Prompt preview: {prompt[:300]}...")

        # Call VLM with the prompt
        try:
            response = self.vlm.invoke(prompt)
            print(f"  VLM Response: {response[:500]}...")

            # Parse the response
            strategy = self._parse_strategy(response)

            # Convert tool_selection format to selected_tools format if needed
            if not strategy.get('selected_tools') and strategy.get('tool_selection'):
                tool_selection = strategy['tool_selection']
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

        except Exception as e:
            error_msg = str(e)
            print(f"  Warning: VLM call failed: {error_msg}")

            # Handle CUDA/NVML errors specifically
            if "nvml" in error_msg.lower() or "CUDA" in error_msg:
                print("  Detected CUDA/NVML error. This may be due to GPU memory pressure.")
                print("  Suggestion: Try clearing CUDA cache or running with fewer GPU resources.")
                # Try to clear CUDA cache
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

            available_tools = get_available_tools()
            return self._get_strategy_by_question_type("feature_attribution", available_tools)

    def _get_clean_model_info_dict(self, model_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create clean dict representation of model info for PromptBuilder."""
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

    def _get_question_type_string(self, q_type: Optional[int]) -> str:
        """Map question type ID to string."""
        if q_type is None:
            return "general"
        if 1 <= q_type <= 4:
            return "feature_attribution"
        elif 5 <= q_type <= 7:
            return "counterfactual"
        elif 8 <= q_type <= 10:
            return "spurious_features"
        return "general"

    def _generate_strategy(
        self,
        question: str,
        question_type: str,
        model_info: str,
        modality: str
    ) -> Dict[str, Any]:
        """
        Generate strategy using VLM.

        Args:
            question: User's question
            question_type: Type of question
            model_info: Model information string
            modality: Data modality

        Returns:
            Strategy dictionary
        """
        available_tools = get_available_tools()

        tool_descriptions = {
            "gradcam": "Visualizes which regions of the image the model focuses on",
            "integrated_gradients": "Provides pixel-level feature attribution",
            "lime": "Creates local interpretable explanations by perturbing input",
            "shap": "Uses Shapley values to explain model predictions",
            "object_detection": "Detects and localizes objects in the image"
        }

        tools_info = "\n".join([
            f"- {tool}: {tool_descriptions.get(tool, 'XAI analysis tool')}"
            for tool in available_tools
        ])

        prompt = f"""You are an XAI (Explainable AI) strategy planner. Select the most appropriate XAI tools.

User Question: {question}
Question Type: {question_type}
Model Info: {model_info}
Modality: {modality}

Available XAI Tools:
{tools_info}

Tool Selection Guidelines:
- For "feature_attribution" questions: Use "gradcam" or "integrated_gradients"
- For "counterfactual" questions: Use "gradcam" and "lime"
- For "spurious_features" questions: Use "integrated_gradients" and "shap"
- For general questions: Use "gradcam" as a starting point

Respond with ONLY a JSON object in this exact format (no other text):
{{
    "strategy_type": "tools",
    "reasoning": "Brief explanation of why these tools were selected",
    "selected_tools": [
        {{
            "tool_name": "<tool_name>",
            "priority": 1,
            "reasoning": "Why this tool is appropriate",
            "parameters": {{}}
        }}
    ],
    "autonomous_tasks": []
}}

JSON Response:"""

        print("\n  Generating strategy via VLM...")
        print(f"  Question type: {question_type}")

        try:
            response = self.vlm.invoke(prompt)
            print(f"  VLM Response: {response[:500]}...")

            # Parse the response
            strategy = self._parse_strategy(response)

            # If parsing failed, use rule-based selection
            if not strategy.get('selected_tools'):
                print("  Warning: Could not parse VLM response, using rule-based selection")
                strategy = self._get_strategy_by_question_type(question_type, available_tools)

            return strategy

        except Exception as e:
            error_msg = str(e)
            print(f"  Warning: VLM call failed: {error_msg}, using rule-based selection")

            # Handle CUDA/NVML errors specifically
            if "nvml" in error_msg.lower() or "CUDA" in error_msg:
                print("  Detected CUDA/NVML error. Attempting to recover...")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

            return self._get_strategy_by_question_type(question_type, available_tools)

    def _get_strategy_by_question_type(
        self,
        question_type: str,
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Generate strategy based on question type (fallback).

        Args:
            question_type: Type of question
            available_tools: List of available tool names

        Returns:
            Strategy dictionary
        """
        tool_priorities = {
            "feature_attribution": ["gradcam", "integrated_gradients", "shap"],
            "counterfactual": ["gradcam", "lime", "integrated_gradients"],
            "spurious_features": ["integrated_gradients", "shap", "lime"],
            "general": ["gradcam", "integrated_gradients"]
        }

        priority_list = tool_priorities.get(question_type, tool_priorities["general"])

        selected_tools = []
        for i, tool_name in enumerate(priority_list):
            if tool_name in available_tools:
                selected_tools.append({
                    "tool_name": tool_name,
                    "priority": i + 1,
                    "reasoning": f"Selected based on question type: {question_type}",
                    "parameters": {}
                })

        # Ensure at least one tool is selected
        if not selected_tools and available_tools:
            selected_tools.append({
                "tool_name": available_tools[0],
                "priority": 1,
                "reasoning": "Default fallback tool",
                "parameters": {}
            })

        return {
            "strategy_type": "tools",
            "reasoning": f"Rule-based selection for {question_type} question type",
            "selected_tools": selected_tools,
            "autonomous_tasks": []
        }

    def _get_clean_model_info(self, model_info: Optional[Dict[str, Any]]) -> str:
        """Create clean string representation of model info."""
        if not model_info:
            return "No model loaded"

        clean_info = {
            "model_name": model_info.get("model_name", "Unknown"),
            "model_type": model_info.get("model_type", "Unknown"),
            "architecture": model_info.get("architecture", "Unknown"),
            "num_classes": model_info.get("num_classes", "Unknown"),
            "device": model_info.get("device", "Unknown")
        }

        return json.dumps(clean_info, indent=2)

    def _parse_strategy(self, output: str) -> Dict[str, Any]:
        """Parse strategy from VLM output."""
        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return self._get_default_strategy()
        except json.JSONDecodeError:
            return self._get_default_strategy()

    def _get_default_strategy(self, modality: Optional[Modality] = None) -> Dict[str, Any]:
        """
        Get default fallback strategy based on modality.

        Args:
            modality: Data modality (vision, text, tabular)

        Returns:
            Default strategy dictionary
        """
        # Convert modality to string
        modality_str = modality.value if isinstance(modality, Modality) else str(modality) if modality else "vision"

        # Define default tools per modality
        default_tools_by_modality = {
            "vision": [
                {
                    "tool_name": "gradcam",
                    "priority": 1,
                    "reasoning": "Standard visualization tool for images",
                    "parameters": {}
                }
            ],
            "text": [
                {
                    "tool_name": "lime",
                    "priority": 1,
                    "reasoning": "Standard text explanation tool",
                    "parameters": {}
                },
                {
                    "tool_name": "shap",
                    "priority": 2,
                    "reasoning": "Game-theoretic word importance",
                    "parameters": {}
                }
            ],
            "tabular": [
                {
                    "tool_name": "shap",
                    "priority": 1,
                    "reasoning": "Best tool for tabular feature importance",
                    "parameters": {}
                },
                {
                    "tool_name": "lime",
                    "priority": 2,
                    "reasoning": "Local interpretable explanation",
                    "parameters": {}
                }
            ]
        }

        selected_tools = default_tools_by_modality.get(modality_str, default_tools_by_modality["vision"])

        return {
            "strategy_type": "tools",
            "reasoning": f"Default strategy for {modality_str} modality",
            "selected_tools": selected_tools,
            "autonomous_tasks": []
        }

    def _save_strategy(self, strategy: Dict[str, Any], question_id: str):
        """Save strategy to file."""
        strategy_file = self.strategy_dir / f"strategy_{question_id}.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        print(f"Strategy saved to: {strategy_file}")


class ActorAgent:
    """
    Actor Agent - Native Implementation

    Executes XAI tools and generates explanations:
    - Executes selected tools from strategy
    - Uses VLM reasoning to analyze tool visualizations
    - Extracts features (bounding boxes, importance) through VLM reasoning
    - Generates structured explanations

    IMPORTANT: Feature extraction is done by VLM, not by the tools.
    Tools only provide raw results (visualizations, statistics).
    """

    def __init__(
        self,
        vlm: VisionLanguageModel,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Actor Agent.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory
        """
        self.vlm = vlm

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.output_dir = Path(output_dir).resolve()
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.tool_registry: Optional[XAIToolRegistry] = None

        print("Actor Agent initialized")
        print(f"  Output directory: {self.output_dir}")

    def initialize_tools(
        self,
        model: Any,
        model_type: str,
        processor: Any,
        modality: str = "vision"
    ):
        """
        Initialize XAI tools with model.

        Args:
            model: PyTorch model
            model_type: Model type
            processor: Image processor
            modality: Data modality
        """
        self.tool_registry = create_xai_tools(
            model=model,
            model_type=model_type,
            processor=processor,
            modality=modality,
            output_dir=str(self.output_dir / "xai_outputs")
        )
        print("XAI Tools initialized for Actor Agent")

    def execute_and_explain(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: QuestionTemplate,
        image_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy and generate explanation.

        This method implements a three-step process:
        1. Execute XAI tools to get raw results (visualizations, statistics)
        2. Use VLM reasoning to extract features from tool results
        3. Generate structured explanation output

        Args:
            strategy: Strategy from Proposer
            question: Question dictionary
            question_template: QuestionTemplate instance
            image_path: Path to image
            model_info: Model information
            prediction: Prediction results

        Returns:
            Result dictionary with explanation
        """
        print("\n" + "=" * 70)
        print("ACTOR AGENT: Executing Strategy")
        print("=" * 70)

        if self.tool_registry is None:
            print("Warning: XAI tools not initialized")
            return {
                "explanation": "XAI tools not available",
                "extracted_features": {},
                "confidence": 0.0,
                "error": "Tools not initialized"
            }

        try:
            # Step 1: Execute XAI tools to get raw results
            print("  Step 1: Executing XAI tools...")
            tool_results = self._execute_tools(
                strategy=strategy,
                image_path=image_path or "",
                prediction=prediction or {}
            )

            # Step 2: Use VLM to extract features from tool results
            print("  Step 2: Extracting features via VLM reasoning...")
            extracted_features = self._extract_features_via_vlm(
                tool_results=tool_results,
                image_path=image_path,
                question=question,
                question_template=question_template,
                prediction=prediction or {}
            )

            # Step 3: Generate structured explanation
            print("  Step 3: Generating explanation...")
            # Get PromptBuilder for Actor prompt
            prompt_builder = self._get_prompt_builder(question_template)

            if prompt_builder is not None:
                print(f"  Using PromptBuilder for explanation: {prompt_builder.__class__.__name__}")

                # Build context for Actor prompt
                context = self._build_actor_context(
                    question=question,
                    model_info=model_info or {},
                    prediction=prediction or {},
                    image_path=image_path
                )

                # Build results dict for Actor prompt (with extracted features)
                results_for_prompt = {
                    "tool_results": tool_results.get('tool_results', {}),
                    "extracted_features": extracted_features,
                    "autonomous_results": {}
                }

                # Generate explanation using PromptBuilder
                parsed_result = self._generate_explanation_with_prompt_builder(
                    prompt_builder=prompt_builder,
                    context=context,
                    strategy=strategy,
                    results=results_for_prompt,
                    tool_results=tool_results
                )

                # Ensure extracted_features from VLM is in the result
                if not parsed_result.get('extracted_features') or not parsed_result['extracted_features'].get('responsible_regions'):
                    parsed_result['extracted_features'] = extracted_features
            else:
                # Use VLM-extracted features directly
                parsed_result = {
                    "explanation": extracted_features.get('explanation', 'XAI analysis completed.'),
                    "extracted_features": extracted_features,
                    "tool_results": tool_results.get('tool_results', {}),
                    "tool_results_summary": tool_results.get('tool_results_summary', ''),
                    "visualization_paths": tool_results.get('visualization_paths', []),
                    "confidence": extracted_features.get('confidence', 0.7)
                }

            # Save tool outputs
            if parsed_result.get('tool_results'):
                self._save_tool_outputs(
                    parsed_result['tool_results'],
                    question.get('question_id', 'unknown')
                )

            # Add metadata
            parsed_result['question_id'] = question.get('question_id', 'unknown')
            parsed_result['question_type'] = question.get('q_type', 'unknown')

            # Save results
            self._save_results(parsed_result)

            print(f"\nExplanation generated")
            return parsed_result

        except Exception as e:
            print(f"Warning: Tool execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "explanation": f"Tool execution failed: {str(e)}",
                "extracted_features": {
                    "responsible_regions": [],
                    "importance_distribution": {},
                    "key_visual_features": []
                },
                "confidence": 0.0,
                "error": str(e)
            }

    def _get_prompt_builder(self, question_template: QuestionTemplate) -> Optional[PromptBuilder]:
        """
        Get appropriate PromptBuilder based on question_template.

        Args:
            question_template: QuestionTemplate instance

        Returns:
            PromptBuilder instance or None
        """
        # First, check if question_template has a prompt_builder
        if question_template.prompt_builder is not None:
            return question_template.prompt_builder

        # Otherwise, create one based on modality and category
        modality = question_template.modality
        category = question_template.category

        # Convert enums to strings
        modality_str = modality.value if isinstance(modality, Modality) else str(modality)
        category_str = category.value if isinstance(category, QuestionCategory) else str(category)

        # Build the key for PROMPT_BUILDER_MAP
        key = f"{modality_str}_{category_str}"

        builder_class = PROMPT_BUILDER_MAP.get(key)
        if builder_class is None:
            return None

        # Get attribution_type from question_template if available
        attribution_type = question_template.attribution_type

        # For feature_attribution builders, pass attribution_type
        if category_str == "feature_attribution" and attribution_type:
            try:
                return builder_class(attribution_type=attribution_type)
            except TypeError:
                return builder_class()
        else:
            return builder_class()

    def _build_actor_context(
        self,
        question: Dict[str, Any],
        model_info: Dict[str, Any],
        prediction: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context dictionary for Actor PromptBuilder.

        Args:
            question: Question dictionary
            model_info: Model information dict
            prediction: Prediction results
            image_path: Path to image

        Returns:
            Context dictionary for PromptBuilder
        """
        # Ensure model_info has required fields
        clean_model_info = {
            "model_name": model_info.get("model_name", "Unknown") if model_info else "Unknown",
            "model_type": model_info.get("model_type", "Unknown") if model_info else "Unknown",
            "architecture": model_info.get("architecture", "Unknown") if model_info else "Unknown",
            "num_classes": model_info.get("num_classes", "Unknown") if model_info else "Unknown",
        }

        context = {
            "user_question": question.get("question", ""),
            "model_info": clean_model_info,
            "prediction": prediction,
            "image_path": image_path,
        }

        # Add modality-specific fields
        modality = question.get("modality", "vision")

        if modality == "text":
            context["text_input"] = question.get("text_input", "")
        elif modality == "tabular":
            context["input_data"] = question.get("features", {})
            context["feature_names"] = list(question.get("features", {}).keys())

        # Add target_class for counterfactual questions
        if question.get("q_type") in [5, 6, 7]:
            context["target_class"] = question.get("target_class", "a different prediction")

        # Add ground_truth for spurious feature questions
        if question.get("q_type") in [8, 9, 10]:
            target_info = question.get("target", {})
            context["ground_truth"] = target_info.get("label", target_info.get("value", "Unknown"))

        return context

    def _generate_explanation_with_prompt_builder(
        self,
        prompt_builder: PromptBuilder,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanation using PromptBuilder.

        Args:
            prompt_builder: PromptBuilder instance
            context: Context dictionary
            strategy: Strategy dictionary
            results: Results dictionary for prompt
            tool_results: Raw tool execution results

        Returns:
            Parsed result dictionary
        """
        try:
            # Build actor prompt using the PromptBuilder
            prompt = prompt_builder.build_actor_prompt(context, strategy, results)

            print(f"\n  Generated actor prompt ({len(prompt)} chars)")
            print(f"  Prompt preview: {prompt[:300]}...")

            # Call VLM with the prompt
            response = self.vlm.invoke(prompt)
            print(f"  VLM Response: {response[:500]}...")

            # Parse the response
            parsed = self._parse_actor_response(response)

            # Merge with tool_results
            parsed['tool_results'] = tool_results.get('tool_results', {})
            parsed['tool_results_summary'] = tool_results.get('tool_results_summary', '')
            parsed['visualization_paths'] = tool_results.get('visualization_paths', [])

            return parsed

        except Exception as e:
            print(f"  Warning: Actor PromptBuilder failed: {e}")
            # Return tool_results as fallback
            return tool_results

    def _parse_actor_response(self, response: str) -> Dict[str, Any]:
        """Parse Actor's VLM response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except json.JSONDecodeError:
            pass

        # If parsing fails, return a basic structure with the response as explanation
        return {
            "explanation": response[:500] if response else "No explanation generated",
            "extracted_features": {
                "responsible_regions": [],
                "importance_distribution": {},
                "key_visual_features": []
            },
            "confidence": 0.5
        }

    def _execute_tools(
        self,
        strategy: Dict[str, Any],
        image_path: str,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute XAI tools based on strategy.

        Args:
            strategy: Strategy dictionary
            image_path: Path to image
            prediction: Prediction results

        Returns:
            Result dictionary
        """
        all_bboxes = []
        all_viz_paths = []
        tool_summaries = []
        tool_outputs = {}

        # Get target class from prediction
        target_class = prediction.get('predicted_class_idx', 0) if prediction else 0

        # Execute each tool in the strategy
        for tool_spec in strategy.get('selected_tools', []):
            tool_name = tool_spec.get('tool_name', 'gradcam')

            if self.tool_registry is None:
                continue

            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                print(f"  Warning: Tool '{tool_name}' not found")
                continue

            try:
                print(f"  Executing {tool_name}...")

                # Execute tool
                result_str = tool.run(
                    image_path=image_path,
                    target_class=target_class,
                    image_id=f"direct_{tool_name}"
                )
                result = json.loads(result_str)

                # Store full tool output
                tool_outputs[tool_name] = result

                if result.get('success'):
                    # Collect bounding boxes
                    bboxes = result.get('bounding_boxes', [])
                    for bbox in bboxes:
                        bbox_with_source = bbox.copy()
                        bbox_with_source['source_tool'] = tool_name
                        all_bboxes.append(bbox_with_source)

                    # Collect visualization paths
                    viz_path = result.get('visualization_path')
                    if viz_path:
                        all_viz_paths.append({
                            'tool': tool_name,
                            'path': viz_path
                        })

                    tool_summaries.append(f"{tool_name}: {result.get('summary', 'completed')}")
                    print(f"    {tool_name} completed successfully")
                else:
                    tool_summaries.append(f"{tool_name}: failed - {result.get('error', 'unknown')}")
                    print(f"    {tool_name} failed: {result.get('error')}")

            except Exception as e:
                tool_summaries.append(f"{tool_name}: exception - {str(e)}")
                tool_outputs[tool_name] = {"success": False, "error": str(e)}
                print(f"    {tool_name} exception: {e}")

        # Process and deduplicate bounding boxes
        unique_bboxes = self._deduplicate_bboxes(all_bboxes)

        # Calculate importance distribution
        importance_stats = self._calculate_importance_stats(tool_outputs)
        primary_contribution = unique_bboxes[0].get('importance', 0.8) if unique_bboxes else 0.5
        secondary_contribution = (
            sum(b.get('importance', 0) for b in unique_bboxes[1:]) / max(len(unique_bboxes) - 1, 1)
            if len(unique_bboxes) > 1 else 0.15
        )

        return {
            "explanation": "XAI analysis identified the most important regions for the model's prediction.",
            "extracted_features": {
                "responsible_regions": unique_bboxes,
                "importance_distribution": {
                    "primary_region_contribution": round(primary_contribution, 4),
                    "secondary_regions_contribution": round(secondary_contribution, 4),
                    "background_contribution": round(
                        max(0, 1.0 - primary_contribution - secondary_contribution), 4
                    ),
                    "per_tool_stats": importance_stats
                },
                "key_visual_features": [b.get('label', 'region') for b in unique_bboxes[:3]],
                "visualization_paths": all_viz_paths
            },
            "tool_results": tool_outputs,
            "tool_results_summary": "; ".join(tool_summaries),
            "visualization_paths": all_viz_paths,
            "confidence": 0.8 if unique_bboxes else 0.3
        }

    def _deduplicate_bboxes(self, bboxes: List[Dict]) -> List[Dict]:
        """Remove near-duplicate bounding boxes."""
        # Sort by importance
        bboxes.sort(key=lambda x: x.get('importance', 0), reverse=True)

        unique_bboxes = []
        for bbox in bboxes:
            is_duplicate = False
            for existing in unique_bboxes:
                if self._bbox_overlap(bbox.get('bbox', {}), existing.get('bbox', {})) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_bboxes.append(bbox)

        return unique_bboxes[:5]  # Keep top 5

    def _bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate IoU between two bounding boxes."""
        if not bbox1 or not bbox2:
            return 0.0

        x1 = max(bbox1.get('x1', 0), bbox2.get('x1', 0))
        y1 = max(bbox1.get('y1', 0), bbox2.get('y1', 0))
        x2 = min(bbox1.get('x2', 0), bbox2.get('x2', 0))
        y2 = min(bbox1.get('y2', 0), bbox2.get('y2', 0))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1.get('x2', 0) - bbox1.get('x1', 0)) * (bbox1.get('y2', 0) - bbox1.get('y1', 0))
        area2 = (bbox2.get('x2', 0) - bbox2.get('x1', 0)) * (bbox2.get('y2', 0) - bbox2.get('y1', 0))
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_importance_stats(self, tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract importance statistics from tool outputs."""
        importance_stats = {}
        for tool_name, output in tool_outputs.items():
            if isinstance(output, dict) and output.get('success'):
                imp_dist = output.get('importance_distribution', {})
                if imp_dist:
                    importance_stats[tool_name] = imp_dist
        return importance_stats

    def _extract_features_via_vlm(
        self,
        tool_results: Dict[str, Any],
        image_path: Optional[str],
        question: Dict[str, Any],
        question_template: QuestionTemplate,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use VLM reasoning to extract features from tool results.

        This is the KEY method that delegates feature extraction to the VLM
        instead of doing it algorithmically. The VLM analyzes visualizations
        and extracts:
        - Responsible regions (bounding boxes)
        - Importance distribution
        - Key visual features

        Args:
            tool_results: Raw results from XAI tools
            image_path: Path to original image
            question: Question dictionary
            question_template: QuestionTemplate instance
            prediction: Prediction results

        Returns:
            Extracted features dictionary
        """
        try:
            # Collect visualization paths from tool results
            viz_paths = tool_results.get('visualization_paths', [])
            tool_outputs = tool_results.get('tool_results', {})

            # Build a description of tool results for the VLM
            tool_descriptions = []
            statistics_summary = {}

            for tool_name, output in tool_outputs.items():
                if isinstance(output, dict) and output.get('success'):
                    desc = output.get('description', f'{tool_name} analysis completed')
                    tool_descriptions.append(f"- {tool_name}: {desc}")

                    # Collect statistics
                    stats = output.get('statistics', {})
                    if stats:
                        statistics_summary[tool_name] = stats

            # Build VLM prompt for feature extraction
            prompt = self._build_feature_extraction_prompt(
                tool_descriptions=tool_descriptions,
                statistics_summary=statistics_summary,
                question=question,
                question_template=question_template,
                prediction=prediction,
                image_path=image_path
            )

            # Prepare images for VLM (original + visualizations)
            images_to_analyze = []
            if image_path and os.path.exists(image_path):
                images_to_analyze.append(image_path)

            # Add visualization images
            for viz_info in viz_paths:
                viz_path = viz_info.get('path', '') if isinstance(viz_info, dict) else str(viz_info)
                if viz_path and os.path.exists(viz_path):
                    images_to_analyze.append(viz_path)

            # Call VLM with images and prompt
            if images_to_analyze:
                # Use VLM multimodal capability
                response = self._call_vlm_with_images(prompt, images_to_analyze)
            else:
                # Fallback to text-only
                response = self.vlm.invoke(prompt)

            # Parse VLM response to extract features
            extracted_features = self._parse_vlm_feature_response(response)

            # Add raw statistics from tools
            extracted_features['tool_statistics'] = statistics_summary

            return extracted_features

        except Exception as e:
            print(f"Warning: VLM feature extraction failed: {e}")
            import traceback
            traceback.print_exc()

            # Return default structure on failure
            return {
                "responsible_regions": [],
                "importance_distribution": {
                    "primary_region_contribution": 0.5,
                    "secondary_regions_contribution": 0.3,
                    "background_contribution": 0.2
                },
                "key_visual_features": [],
                "explanation": "Feature extraction via VLM encountered an error.",
                "confidence": 0.3,
                "error": str(e)
            }

    def _build_feature_extraction_prompt(
        self,
        tool_descriptions: List[str],
        statistics_summary: Dict[str, Any],
        question: Dict[str, Any],
        question_template: QuestionTemplate,
        prediction: Dict[str, Any],
        image_path: Optional[str]
    ) -> str:
        """
        Build a prompt for VLM to extract features from XAI tool outputs.

        Args:
            tool_descriptions: List of tool description strings
            statistics_summary: Dictionary of statistics per tool
            question: Question dictionary
            question_template: QuestionTemplate instance
            prediction: Prediction results
            image_path: Path to original image

        Returns:
            Prompt string for VLM
        """
        # Get question category
        category = question_template.category
        category_str = category.value if isinstance(category, QuestionCategory) else str(category)

        # Get predicted class info
        pred_class = prediction.get('predicted_class', 'Unknown')
        pred_conf = prediction.get('confidence', 0.0)

        # Build statistics text
        stats_text = ""
        for tool_name, stats in statistics_summary.items():
            stats_text += f"\n{tool_name} statistics:\n"
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    stats_text += f"  - {key}: {value}\n"
                elif isinstance(value, list) and len(value) <= 5:
                    stats_text += f"  - {key}: {value}\n"

        # Category-specific instructions
        category_instructions = self._get_category_specific_instructions(category_str)

        prompt = f"""You are an expert XAI analyst. Analyze the XAI visualization results and extract structured features.

## Context
- User Question: {question.get('question', 'What influenced the model prediction?')}
- Question Category: {category_str}
- Model Prediction: {pred_class} (confidence: {pred_conf:.2%})

## XAI Tool Results
{chr(10).join(tool_descriptions) if tool_descriptions else "No tool descriptions available."}

## Statistics from Tools
{stats_text if stats_text else "No detailed statistics available."}

## Your Task
{category_instructions}

Analyze the provided XAI visualizations (heatmaps, attention maps, etc.) and extract:
1. **Responsible Regions**: Identify the image regions that most influenced the model's prediction.
   For each region, provide:
   - A descriptive label (e.g., "dog's face", "central object", "top-left corner")
   - Approximate bounding box as percentage of image (x1, y1, x2, y2 where 0-100)
   - Importance score (0.0 to 1.0)

2. **Importance Distribution**: Estimate how much each region type contributes:
   - Primary region contribution (0.0 to 1.0)
   - Secondary regions contribution (0.0 to 1.0)
   - Background contribution (0.0 to 1.0)
   (These should sum to approximately 1.0)

3. **Key Visual Features**: List the main visual features (textures, shapes, colors) that the model focused on.

4. **Explanation**: Provide a clear, concise explanation of why the model made this prediction based on the XAI analysis.

5. **Confidence**: Your confidence in this analysis (0.0 to 1.0).

Respond with ONLY a JSON object in this exact format:
{{
    "responsible_regions": [
        {{
            "label": "descriptive name of region",
            "bbox": {{"x1": 10, "y1": 20, "x2": 60, "y2": 80}},
            "importance": 0.85,
            "description": "Brief description of what this region contains"
        }}
    ],
    "importance_distribution": {{
        "primary_region_contribution": 0.65,
        "secondary_regions_contribution": 0.25,
        "background_contribution": 0.10
    }},
    "key_visual_features": ["feature1", "feature2", "feature3"],
    "explanation": "Clear explanation of the model's decision based on XAI analysis.",
    "confidence": 0.85
}}

JSON Response:"""

        return prompt

    def _get_category_specific_instructions(self, category: str) -> str:
        """Get category-specific instructions for feature extraction."""
        instructions = {
            "feature_attribution": (
                "Focus on identifying which image regions (visual features) contributed most "
                "to the prediction. Look at the heatmaps to see where the model 'looked' most intensely. "
                "Highlight both positive contributions (supporting the prediction) and any notable "
                "negative regions (areas that could have changed the prediction)."
            ),
            "counterfactual": (
                "Focus on identifying regions that, if changed, would alter the prediction. "
                "Look for areas where small modifications could flip the model's decision. "
                "Consider what minimal changes would be needed to change the output class."
            ),
            "spurious_features": (
                "Focus on identifying any suspicious or potentially spurious features the model "
                "might be relying on. Look for attention on background elements, watermarks, "
                "or features unrelated to the actual object of interest. These could indicate "
                "the model is using shortcuts rather than meaningful features."
            )
        }
        return instructions.get(category, instructions["feature_attribution"])

    def _call_vlm_with_images(
        self,
        prompt: str,
        image_paths: List[str]
    ) -> str:
        """
        Call VLM with multiple images for analysis.

        Args:
            prompt: Text prompt
            image_paths: List of image paths to analyze

        Returns:
            VLM response string
        """
        try:
            # Try to use VLM's multimodal capability
            if hasattr(self.vlm, 'invoke_with_images'):
                return self.vlm.invoke_with_images(prompt, image_paths)
            elif hasattr(self.vlm, 'invoke_multimodal'):
                return self.vlm.invoke_multimodal(prompt, image_paths)
            else:
                # Load images and include in prompt description
                image_descriptions = []
                for i, path in enumerate(image_paths):
                    try:
                        img = Image.open(path)
                        w, h = img.size
                        image_descriptions.append(
                            f"Image {i+1}: {os.path.basename(path)} ({w}x{h} pixels)"
                        )
                    except Exception:
                        image_descriptions.append(f"Image {i+1}: {os.path.basename(path)}")

                # Add image info to prompt
                enhanced_prompt = (
                    f"[Note: Analyzing {len(image_paths)} images: "
                    f"{', '.join(image_descriptions)}]\n\n{prompt}"
                )

                return self.vlm.invoke(enhanced_prompt)

        except Exception as e:
            print(f"Warning: VLM multimodal call failed: {e}")
            return self.vlm.invoke(prompt)

    def _parse_vlm_feature_response(self, response: str) -> Dict[str, Any]:
        """
        Parse VLM response to extract features.

        Args:
            response: VLM response string

        Returns:
            Extracted features dictionary
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                # Validate and clean the parsed result
                features = {
                    "responsible_regions": [],
                    "importance_distribution": {
                        "primary_region_contribution": 0.5,
                        "secondary_regions_contribution": 0.3,
                        "background_contribution": 0.2
                    },
                    "key_visual_features": [],
                    "explanation": "",
                    "confidence": 0.5
                }

                # Extract responsible regions
                if 'responsible_regions' in parsed:
                    regions = parsed['responsible_regions']
                    if isinstance(regions, list):
                        for region in regions:
                            if isinstance(region, dict):
                                cleaned_region = {
                                    "label": region.get("label", "region"),
                                    "bbox": region.get("bbox", {}),
                                    "importance": float(region.get("importance", 0.5)),
                                    "description": region.get("description", "")
                                }
                                features["responsible_regions"].append(cleaned_region)

                # Extract importance distribution
                if 'importance_distribution' in parsed:
                    imp_dist = parsed['importance_distribution']
                    if isinstance(imp_dist, dict):
                        features["importance_distribution"] = {
                            "primary_region_contribution": float(imp_dist.get("primary_region_contribution", 0.5)),
                            "secondary_regions_contribution": float(imp_dist.get("secondary_regions_contribution", 0.3)),
                            "background_contribution": float(imp_dist.get("background_contribution", 0.2))
                        }

                # Extract key visual features
                if 'key_visual_features' in parsed:
                    kvf = parsed['key_visual_features']
                    if isinstance(kvf, list):
                        features["key_visual_features"] = [str(f) for f in kvf]

                # Extract explanation
                if 'explanation' in parsed:
                    features["explanation"] = str(parsed['explanation'])

                # Extract confidence
                if 'confidence' in parsed:
                    features["confidence"] = float(parsed['confidence'])

                return features

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Warning: Failed to parse VLM response: {e}")

        # Fallback: try to extract information from text response
        return {
            "responsible_regions": [],
            "importance_distribution": {
                "primary_region_contribution": 0.5,
                "secondary_regions_contribution": 0.3,
                "background_contribution": 0.2
            },
            "key_visual_features": [],
            "explanation": response[:500] if response else "Unable to extract features.",
            "confidence": 0.3
        }

    def _save_tool_outputs(self, tool_outputs: Dict[str, Any], question_id: str):
        """Save tool outputs to file."""
        tool_outputs_dir = self.output_dir / "tool_outputs"
        tool_outputs_dir.mkdir(parents=True, exist_ok=True)

        output_file = tool_outputs_dir / f"tool_outputs_{question_id}.json"
        with open(output_file, 'w') as f:
            json.dump(tool_outputs, f, indent=2)

        print(f"Tool outputs saved to: {output_file}")

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        results_file = self.results_dir / f"result_{results['question_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")


class CriticAgent:
    """
    Critic Agent - Native Implementation

    Evaluates Actor's results:
    - Assesses quality of explanations
    - Calculates evaluation metrics
    - Provides feedback
    """

    def __init__(
        self,
        vlm: VisionLanguageModel,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Critic Agent.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory
        """
        self.vlm = vlm

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.output_dir = Path(output_dir).resolve()
        self.eval_dir = self.output_dir / "evaluations"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        print("Critic Agent initialized")
        print(f"  Output directory: {self.output_dir}")

    def evaluate(
        self,
        results: Dict[str, Any],
        question: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate Actor's results.

        Args:
            results: Results from Actor
            question: Original question
            ground_truth: Ground truth labels (if available)

        Returns:
            Evaluation dictionary
        """
        print("\n" + "=" * 70)
        print("CRITIC AGENT: Evaluating Results")
        print("=" * 70)

        # Calculate basic metrics
        evaluation = self._calculate_metrics(results, ground_truth)

        # Save evaluation
        eval_file = self.eval_dir / f"eval_{results.get('question_id', 'unknown')}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2)

        print(f"Evaluation saved to: {eval_file}")
        print(f"  Overall Rating: {evaluation['overall_rating']}")

        return evaluation

    def _calculate_metrics(
        self,
        results: Dict[str, Any],
        ground_truth: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics.

        Args:
            results: Results from Actor
            ground_truth: Ground truth labels

        Returns:
            Evaluation metrics dictionary
        """
        # Extract features for evaluation
        extracted_features = results.get('extracted_features', {})
        regions = extracted_features.get('responsible_regions', [])
        tool_results = results.get('tool_results', {})

        # Calculate quality score based on available results
        quality_score = 0.0
        completeness = 0.0

        # Check if tools executed successfully
        successful_tools = sum(
            1 for r in tool_results.values()
            if isinstance(r, dict) and r.get('success')
        )
        total_tools = len(tool_results)

        if total_tools > 0:
            quality_score = successful_tools / total_tools
            completeness = min(1.0, len(regions) / 3)  # Expect at least 3 regions

        # Calculate clarity (based on having visualization paths)
        viz_paths = extracted_features.get('visualization_paths', [])
        clarity = min(1.0, len(viz_paths) / max(total_tools, 1))

        # Overall rating
        overall_score = (quality_score + completeness + clarity) / 3

        if overall_score >= 0.8:
            overall_rating = "Excellent"
        elif overall_score >= 0.6:
            overall_rating = "Good"
        elif overall_score >= 0.4:
            overall_rating = "Fair"
        else:
            overall_rating = "Needs Improvement"

        return {
            "quality_score": round(quality_score, 2),
            "completeness": round(completeness, 2),
            "clarity": round(clarity, 2),
            "overall_score": round(overall_score, 2),
            "overall_rating": overall_rating,
            "metrics": {
                "successful_tools": successful_tools,
                "total_tools": total_tools,
                "regions_found": len(regions),
                "visualizations_generated": len(viz_paths)
            },
            "comments": self._generate_comments(quality_score, completeness, clarity)
        }

    def _generate_comments(
        self,
        quality_score: float,
        completeness: float,
        clarity: float
    ) -> str:
        """Generate evaluation comments."""
        comments = []

        if quality_score >= 0.8:
            comments.append("All XAI tools executed successfully.")
        elif quality_score >= 0.5:
            comments.append("Most XAI tools executed successfully.")
        else:
            comments.append("Some XAI tools failed to execute.")

        if completeness >= 0.8:
            comments.append("Found sufficient important regions.")
        elif completeness >= 0.5:
            comments.append("Found some important regions.")
        else:
            comments.append("Few important regions identified.")

        if clarity >= 0.8:
            comments.append("Clear visualizations generated.")
        elif clarity >= 0.5:
            comments.append("Some visualizations available.")
        else:
            comments.append("Limited visualizations generated.")

        return " ".join(comments)
