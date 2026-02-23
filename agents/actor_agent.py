"""
Actor Agent for XAI Agent Framework

Responsible for:
- Executing XAI tools based on strategy
- Using VLM reasoning to analyze tool outputs
- Generating structured explanations
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent


class ActorAgent(BaseAgent):
    """
    Actor Agent - XAI Tool Execution and Explanation Generation

    Implements a three-step process:
    1. Execute XAI tools to get raw results (visualizations, statistics)
    2. Use VLM reasoning to extract features from tool results
    3. Generate structured explanation output
    """

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Actor Agent.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory
        """
        super().__init__(vlm, output_dir, "ActorAgent")

        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.tool_registry: Optional[Any] = None
        self.data_model_loader: Optional[Any] = None

    def initialize_tools(self, data_model_loader: Any):
        """
        Initialize XAI tools with a data model loader.

        Args:
            data_model_loader: An instance of DataModelLoader.
        """
        # Store data_model_loader for use in multi-instance methods (Q4, Q9, Q10)
        self.data_model_loader = data_model_loader

        from xai_tools_native import create_xai_tools
        self.tool_registry = create_xai_tools(
            data_model_loader=data_model_loader,
            output_dir=str(self.output_dir / "xai_outputs")
        )
        print("XAI Tools initialized for Actor Agent")

    def run(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        # Multi-instance parameters (for Q4, Q9, Q10)
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - execute strategy and generate explanation.

        Args:
            strategy: Strategy from Proposer
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder
            input_path: Path to input data (single instance)
            model_info: Model information
            prediction: Prediction results (single instance)
            input_paths: List of paths for multi-instance questions
            predictions: List of predictions for multi-instance questions

        Returns:
            Result dictionary with explanation
        """
        return self.execute_and_explain(
            strategy, question, question_template,
            input_path, model_info, prediction,
            input_paths=input_paths, predictions=predictions
        )

    def execute_and_explain(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        # Multi-instance parameters
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy and generate explanation.

        Handles both single-instance and multi-instance questions.
        For multi-instance (Q4, Q9, Q10):
        - Executes tools on each instance separately (loop)
        - Combines results and generates comparative explanation (one VLM call)

        Args:
            strategy: Strategy from Proposer
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder
            input_path: Path to input (single instance)
            model_info: Model information
            prediction: Prediction results (single instance)
            input_paths: List of paths for multi-instance
            predictions: List of predictions for multi-instance

        Returns:
            Result dictionary with explanation and output
        """
        is_multi_instance = question.get('is_multi_instance', False)
        num_instances = question.get('num_instances', 1)
        q_type = question.get('q_type')
        modality = question.get('modality', 'vision')

        print("\n" + "=" * 70)
        if is_multi_instance:
            print(f"ACTOR AGENT: Executing Strategy for Q{q_type} ({num_instances} instances)")
        else:
            print("ACTOR AGENT: Executing Strategy")
        print("=" * 70)

        if is_multi_instance and input_paths and predictions:
            # Multi-instance execution
            return self._execute_multi_instance(
                strategy=strategy,
                question=question,
                question_template=question_template,
                input_paths=input_paths,
                model_info=model_info,
                predictions=predictions
            )
        else:
            # Single instance execution (original logic)
            return self._execute_single_instance(
                strategy=strategy,
                question=question,
                question_template=question_template,
                input_path=input_path,
                model_info=model_info,
                prediction=prediction
            )

    def _execute_single_instance(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute strategy for single instance (original logic)."""
        modality = question.get('modality', 'vision')

        # Pure-reasoning mode: no tools and no autonomous tasks
        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            return self._run_pure_reasoning(
                question=question,
                question_template=question_template,
                strategy=strategy,
                input_path=input_path,
                model_info=model_info,
                prediction=prediction
            )

        # Step 1: Execute XAI tools
        print("  Step 1: Executing XAI tools...")
        tool_results = self._execute_tools(
            strategy=strategy,
            input_path=input_path or "",
            prediction=prediction or {},
            modality=modality,
            question=question
        )

        # Step 1.5: Execute autonomous tasks (if any)
        print("  Step 1.5: Executing autonomous tasks...")
        autonomous_results = self._execute_autonomous_tasks(
            strategy=strategy,
            input_path=input_path or "",
            prediction=prediction or {},
            question=question,
            tool_results=tool_results
        )

        # Merge autonomous results into tool_results for unified storage
        if autonomous_results:
            tool_results['autonomous_results'] = autonomous_results
            tool_results['tool_results']['autonomous_tasks'] = autonomous_results
            self._save_tool_outputs(tool_results['tool_results'], question)

        # Step 2: Extract features via VLM
        print("  Step 2: Extracting features via VLM reasoning...")
        extracted_features = self._extract_features_via_vlm(
            tool_results=tool_results,
            input_path=input_path,
            question=question,
            question_template=question_template,
            prediction=prediction or {}
        )

        # Step 3: Generate structured explanation
        print("  Step 3: Generating explanation...")
        prompt_builder = self._get_prompt_builder(question_template, question)

        context = self._build_context(question, model_info, prediction, input_path)
        results_for_prompt = {
            "tool_results": tool_results.get('tool_results', {}),
            "extracted_features": extracted_features,
            "autonomous_results": autonomous_results
        }

        parsed_result = self._generate_explanation_with_prompt_builder(
            prompt_builder=prompt_builder,
            context=context,
            strategy=strategy,
            results=results_for_prompt,
            tool_results=tool_results
        )

        # Add metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = question.get('q_type', 'unknown')
        parsed_result['tool_results'] = tool_results.get('tool_results', {})
        parsed_result['autonomous_results'] = autonomous_results
        parsed_result['visualization_paths'] = tool_results.get('visualization_paths', [])

        # Save results
        self._save_results(parsed_result, question)

        print(f"\nExplanation generated")
        return parsed_result

    def _execute_multi_instance(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_paths: List[str],
        model_info: Optional[Dict[str, Any]] = None,
        predictions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy for multi-instance questions (Q4, Q9, Q10).

        Approach:
        1. Execute XAI tools on each instance separately (loop)
        2. Combine all tool results
        3. One VLM call to generate comparative explanation
        """
        modality = question.get('modality', 'vision')
        num_instances = len(input_paths)
        q_type = question.get('q_type')

        # Pure-reasoning mode: no tools and no autonomous tasks
        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            return self._run_pure_reasoning(
                question=question,
                question_template=question_template,
                strategy=strategy,
                input_paths=input_paths,
                model_info=model_info,
                predictions=predictions
            )

        # Step 1: Execute XAI tools on each instance
        all_tool_results = []
        all_viz_paths = []

        # Get image indices and split for reloading samples per instance
        image_indices = question.get('image_indices', [])
        # Breast Cancer Q8/Q9/Q10 (spurious): row_idx is an index into the full dataset (569 samples),
        # not the test-split (114 samples). Mirror is_spurious logic from load_breast_cancer.py.
        _dataset = str(question.get('dataset', '')).lower().replace(' ', '_')
        _is_breast_cancer_spurious = (
            'breast_cancer' in _dataset
            and question.get('q_type', 1) in (8, 9, 10)
        )
        split = ('full' if _is_breast_cancer_spurious else question.get('split', 'test'))

        for i, (input_path, prediction) in enumerate(zip(input_paths, predictions or [{}] * num_instances)):
            print(f"  Step 1.{i+1}: Executing XAI tools on Instance {i}...")

            # Reload this instance's sample in data_model_loader so XAI tools
            # operate on the correct data (tools use data_model_loader.get_current_*())
            if self.data_model_loader and i < len(image_indices):
                try:
                    if modality == 'text':
                        # Text loaders expect text content, not integer indices.
                        # Reload from the question's features array.
                        features_list = question.get('features', [])
                        if isinstance(features_list, list) and i < len(features_list):
                            feat = features_list[i]
                            if 'premise' in feat and 'hypothesis' in feat:
                                text_input = feat
                            elif 'text' in feat:
                                text_input = feat.get('text', '')
                            elif 'review_text' in feat:
                                text_input = feat.get('review_text', '')
                            else:
                                text_input = feat
                            data = self.data_model_loader.loader_module.load_data(text_input)
                            self.data_model_loader.current_sample_data = data
                        else:
                            print(f"  Warning: No features entry for text instance {i}")
                    else:
                        self.data_model_loader.load_sample(index=image_indices[i], split=split)
                except Exception as e:
                    raise RuntimeError(f"Failed to reload sample {image_indices[i]} for instance {i}: {e}") from e

            # Modify question for this instance (for output naming)
            instance_question = question.copy()
            instance_question['instance_index'] = i
            instance_question['instance_suffix'] = f"_inst{i}"

            tool_results = self._execute_tools(
                strategy=strategy,
                input_path=input_path,
                prediction=prediction,
                modality=modality,
                question=instance_question
            )

            all_tool_results.append(tool_results)
            all_viz_paths.extend(tool_results.get('visualization_paths', []))

        # Combine tool results
        combined_tool_results = {
            'instances': all_tool_results,
            'tool_results': {f'instance_{i}': tr.get('tool_results', {}) for i, tr in enumerate(all_tool_results)},
            'visualization_paths': all_viz_paths,
            'tool_results_summary': "; ".join([
                f"Instance {i}: {tr.get('tool_results_summary', 'no results')}"
                for i, tr in enumerate(all_tool_results)
            ])
        }

        # Step 1.5: Execute autonomous tasks (if any)
        print("  Step 1.5: Executing autonomous tasks...")
        autonomous_results = self._execute_autonomous_tasks(
            strategy=strategy,
            input_path=input_paths[0] if input_paths else "",
            prediction=predictions[0] if predictions else {},
            question=question,
            tool_results=combined_tool_results
        )

        if autonomous_results:
            combined_tool_results['autonomous_results'] = autonomous_results
            combined_tool_results['tool_results']['autonomous_tasks'] = autonomous_results

        # Step 2: Extract comparative features via VLM
        print("  Step 2: Extracting comparative features via VLM...")
        extracted_features = self._extract_features_multi(
            tool_results=combined_tool_results,
            input_paths=input_paths,
            question=question,
            question_template=question_template,
            predictions=predictions
        )

        # Step 3: Generate comparative explanation
        print("  Step 3: Generating comparative explanation...")
        prompt_builder = self._get_prompt_builder(question_template, question)

        context = self._build_context_multi(question, model_info, predictions, input_paths)
        results_for_prompt = {
            "tool_results": combined_tool_results['tool_results'],
            "extracted_features": extracted_features,
            "autonomous_results": autonomous_results,
            "instances": [{'prediction': p, 'path': path} for p, path in zip(predictions, input_paths)]
        }

        # Use multi-instance prompt
        parsed_result = self._generate_explanation_multi(
            prompt_builder=prompt_builder,
            context=context,
            strategy=strategy,
            results=results_for_prompt,
            tool_results=combined_tool_results,
            instances=results_for_prompt['instances']
        )

        # Ensure multi-instance output format
        if not parsed_result.get('output') or not isinstance(parsed_result.get('output'), dict):
            parsed_result['output'] = {}

        # Add per-instance outputs if not present (always use letter-based keys: input_A, input_B, input_C...)
        for i in range(num_instances):
            key = f'input_{chr(ord("A") + i)}'
            if key not in parsed_result['output']:
                parsed_result['output'][key] = extracted_features.get(f'output_{i}', {})

        # Add metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = q_type
        parsed_result['is_multi_instance'] = True
        parsed_result['num_instances'] = num_instances
        parsed_result['tool_results'] = combined_tool_results['tool_results']
        parsed_result['autonomous_results'] = autonomous_results
        parsed_result['visualization_paths'] = all_viz_paths

        # Save results
        self._save_results(parsed_result, question)

        print(f"\nMulti-instance explanation generated for {num_instances} instances")
        return parsed_result

    def _extract_features_multi(
        self,
        tool_results: Dict[str, Any],
        input_paths: List[str],
        question: Dict[str, Any],
        question_template: Any,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features for multi-instance comparison (Q9, Q10)."""
        num_instances = len(input_paths)
        modality = question.get('modality', 'vision')
        q_type = question.get('q_type', 9)

        # Get image size from tool results
        image_width, image_height = 224, 224
        for inst_results in tool_results.get('instances', []):
            for tool_name, result in inst_results.get('tool_results', {}).items():
                if isinstance(result, dict) and result.get('success'):
                    img_size = result.get('original_image_size', {})
                    if img_size:
                        image_width = img_size.get('width', image_width)
                        image_height = img_size.get('height', image_height)
                        break
            if image_width != 224:
                break

        # Build size constraint
        if modality == "vision":
            output_format = '"bounding_box": [x_min, y_min, x_max, y_max]'
            size_constraint = f"""
**CRITICAL IMAGE SIZE CONSTRAINT:**
- Image dimensions: {image_width} x {image_height} pixels
- ALL coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- Ensure x_max <= {image_width} and y_max <= {image_height}"""
        elif modality == "text":
            output_format = '"start_index": int, "end_index": int'
            size_constraint = ""
        else:
            output_format = '"feature_key": "string"'
            size_constraint = ""

        # Build per-instance summaries with detailed tool statistics
        instance_sections = []
        for i in range(num_instances):
            pred = predictions[i] if i < len(predictions) else {}
            inst_results = tool_results.get('instances', [{}])[i] if i < len(tool_results.get('instances', [])) else {}

            pred_class = pred.get('predicted_class_name', pred.get('predicted_class_idx', 'Unknown'))
            confidence = pred.get('confidence', 0.0)

            section = f"""### Instance {i}
- Model Prediction: {pred_class} (confidence: {confidence:.2%})

**Tool Results Summary:**
{inst_results.get('tool_results_summary', 'No tools executed')}

**Detailed Tool Statistics:**
{self._format_tool_statistics(inst_results)}"""
            instance_sections.append(section)

        prompt = f"""You are an expert XAI analyst. Analyze the XAI results and extract key features for {num_instances} instances.

## Context
- Question: {question.get('question', '')}
- Question Type: Q{q_type}
- Modality: {modality}
{size_constraint}

## Instance Analysis
{chr(10).join(instance_sections)}

## Your Task
For EACH instance, identify THE SINGLE MOST IMPORTANT region/feature that causes its prediction.

**CRITICAL: Your response MUST be valid JSON with this exact structure:**
{{
    {', '.join([f'"output_{i}": {{{output_format}, "description": "explanation for instance {i}"}}' for i in range(num_instances)])},
    "comparison": "Brief explanation of key differences between instances"
}}

**Important Guidelines:**
- For vision: Provide bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates
  - MUST respect image bounds: x in [0, {image_width}], y in [0, {image_height}]
  - Use the top_attention_coords from tool results to determine the region
- For text: Provide character indices (start_index, end_index)
- For tabular: Provide the exact feature name
- **If quantitative XAI tool statistics are unavailable or incomplete, use the autonomous analysis results above to infer the answer. If no autonomous results exist either, perform your own direct reasoning based on the model prediction and context.**


Respond with ONLY valid JSON:"""

        return self.invoke_vlm_for_json(prompt)

    def _format_tool_summary(self, tool_results: Dict[str, Any]) -> str:
        """Format tool results as brief summary."""
        summaries = []
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get('success'):
                bbox = result.get('suggested_bounding_box')
                if bbox:
                    summaries.append(f"{tool_name}: bbox={bbox}")
                else:
                    summaries.append(f"{tool_name}: success")
        return ", ".join(summaries) if summaries else "no results"

    def _build_context_multi(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        input_paths: List[str]
    ) -> Dict[str, Any]:
        """Build context for multi-instance comparison."""
        modality = question.get("modality", "vision")

        clean_model_info = {}
        if model_info:
            clean_model_info = {
                "model_name": model_info.get("model_name", "Unknown"),
                "architecture": model_info.get("architecture", "Unknown"),
                "num_classes": model_info.get("num_classes", "Unknown"),
            }

        context = {
            "user_question": question.get("question", ""),
            "model_info": clean_model_info,
            "num_instances": len(predictions),
            "predictions": predictions,
            "input_paths": input_paths,
            "modality": modality,
        }

        # Add indexed access
        for i, pred in enumerate(predictions):
            context[f"prediction_{i}"] = pred
        for i, path in enumerate(input_paths):
            context[f"image_path_{i}"] = path

        # For compatibility with existing prompts
        if predictions:
            context["prediction"] = predictions[0]
            context["prediction_A"] = predictions[0] if len(predictions) > 0 else {}
            context["prediction_B"] = predictions[1] if len(predictions) > 1 else {}
        if input_paths:
            context["image_path"] = input_paths[0]
            context["image_path_A"] = input_paths[0] if len(input_paths) > 0 else ""
            context["image_path_B"] = input_paths[1] if len(input_paths) > 1 else ""

        # Add text/tabular instance data for VLM prompt
        if modality in ('text', 'tabular'):
            features_list = question.get('features', [])
            image_indices = question.get('image_indices', question.get('row_no', []))
            targets = question.get('targets', question.get('target', []))
            preds = question.get('predictions', question.get('predicted', []))

            instance_data = []
            for i in range(len(features_list) if isinstance(features_list, list) else 0):
                feat = features_list[i]
                inst = {"index": image_indices[i] if i < len(image_indices) else i}
                if modality == 'text':
                    if 'text' in feat:
                        inst["text"] = feat["text"]
                    elif 'premise' in feat:
                        inst["premise"] = feat["premise"]
                        inst["hypothesis"] = feat.get("hypothesis", "")
                elif modality == 'tabular':
                    inst["features"] = feat
                if isinstance(targets, list) and i < len(targets):
                    inst["target"] = targets[i]
                if isinstance(preds, list) and i < len(preds):
                    inst["predicted"] = preds[i]
                instance_data.append(inst)

            context["instance_data"] = instance_data

        return context

    def _generate_explanation_multi(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        tool_results: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate explanation using multi-instance prompt builder."""
        prompt = prompt_builder.build_actor_prompt_multi(context, strategy, results, instances)
        print(f"  Generated multi-instance actor prompt ({len(prompt)} chars)")

        return self.invoke_vlm_for_json(prompt)

    # =========================================================================
    # Q4 Specific Methods (instance_A / instance_B format)
    # =========================================================================

    def _reload_q4_instance(self, inst: Dict[str, Any], modality: str):
        """Reload a Q4 instance into data_model_loader before tool execution."""
        if not self.data_model_loader:
            return
        split = inst.get('split', 'test')
        try:
            if modality == 'text':
                feat = inst.get('features', {})
                if 'premise' in feat and 'hypothesis' in feat:
                    text_input = feat
                elif 'text' in feat:
                    text_input = feat['text']
                else:
                    text_input = feat
                data = self.data_model_loader.loader_module.load_data(text_input)
                self.data_model_loader.current_sample_data = data
                print(f"    Reloaded text instance: row_no={inst.get('image_index')}")
            elif modality == 'tabular' and inst.get('image_index') is not None:
                self.data_model_loader.load_sample(index=inst['image_index'], split=split)
                print(f"    Reloaded tabular instance: row_no={inst['image_index']}")
            elif modality == 'vision' and inst.get('image_index') is not None:
                self.data_model_loader.load_sample(index=inst['image_index'], split=split)
                print(f"    Reloaded vision instance: image_index={inst['image_index']}")
        except Exception as e:
            raise RuntimeError(f"Failed to reload Q4 instance: {e}") from e

    def run_q4(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        instances: List[Dict[str, Any]],  # [{'prediction': ..., 'path': ..., 'label': 'A/B'}]
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy for Q4 contrastive instances.

        Approach:
        1. Execute XAI tools on Instance A
        2. Execute XAI tools on Instance B
        3. One VLM call to generate comparative explanation

        Args:
            strategy: Strategy from Proposer
            question: Question dict with instance_A, instance_B
            question_template: Q4ContrastiveInstancesPromptBuilder
            instances: [{'prediction': pred_A, 'path': path_A, 'label': 'A'}, ...]
            model_info: Model information

        Returns:
            Result dict with output.input_A and output.input_B
        """
        print("\n" + "=" * 70)
        print("ACTOR AGENT: Executing Q4 Strategy (Instance A vs B)")
        print("=" * 70)

        modality = question.get('modality', 'vision')

        # Pure-reasoning mode: no tools and no autonomous tasks
        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            return self._run_pure_reasoning(
                question=question,
                question_template=question_template,
                strategy=strategy,
                model_info=model_info,
                instances=instances
            )

        # Breast Cancer Q4 uses full-dataset indices (is_spurious pattern from load_breast_cancer.py)
        if modality == 'tabular':
            _dataset = str(question.get('dataset', '')).lower().replace(' ', '_')
            if 'breast_cancer' in _dataset:
                for inst in instances:
                    inst['split'] = 'full'

        # ═══════════════════════════════════════════════════════
        # Step 1: Execute XAI tools on Instance A
        # ═══════════════════════════════════════════════════════
        inst_a = instances[0]
        print(f"  Step 1: Executing XAI tools on Instance A...")

        # CRITICAL: Reload the correct sample for Instance A before tool execution
        # This ensures XAI tools operate on A's data, not B's
        self._reload_q4_instance(inst_a, modality)

        question_a = question.copy()
        question_a['instance_suffix'] = '_A'

        tool_results_a = self._execute_tools(
            strategy=strategy,
            input_path=inst_a['path'],
            prediction=inst_a['prediction'],
            modality=modality,
            question=question_a
        )
        print(f"    Instance A tools complete: {tool_results_a.get('tool_results_summary', 'N/A')}")

        # ═══════════════════════════════════════════════════════
        # Step 2: Execute XAI tools on Instance B
        # ═══════════════════════════════════════════════════════
        inst_b = instances[1]
        print(f"  Step 2: Executing XAI tools on Instance B...")

        # CRITICAL: Reload the correct sample for Instance B before tool execution
        self._reload_q4_instance(inst_b, modality)

        question_b = question.copy()
        question_b['instance_suffix'] = '_B'

        tool_results_b = self._execute_tools(
            strategy=strategy,
            input_path=inst_b['path'],
            prediction=inst_b['prediction'],
            modality=modality,
            question=question_b
        )
        print(f"    Instance B tools complete: {tool_results_b.get('tool_results_summary', 'N/A')}")

        # ═══════════════════════════════════════════════════════
        # Step 3: Combine tool results
        # ═══════════════════════════════════════════════════════
        combined_tool_results = {
            'instance_A': tool_results_a.get('tool_results', {}),
            'instance_B': tool_results_b.get('tool_results', {}),
            'tool_results': {
                'instance_A': tool_results_a.get('tool_results', {}),
                'instance_B': tool_results_b.get('tool_results', {})
            },
            'visualization_paths': (
                tool_results_a.get('visualization_paths', []) +
                tool_results_b.get('visualization_paths', [])
            ),
            'tool_results_summary': (
                f"Instance A: {tool_results_a.get('tool_results_summary', 'N/A')}; "
                f"Instance B: {tool_results_b.get('tool_results_summary', 'N/A')}"
            )
        }

        # ═══════════════════════════════════════════════════════
        # Step 3.5: Execute autonomous tasks (if any)
        # ═══════════════════════════════════════════════════════
        print("  Step 3.5: Executing autonomous tasks...")
        autonomous_results = self._execute_autonomous_tasks(
            strategy=strategy,
            input_path=inst_a['path'],
            prediction=inst_a['prediction'],
            question=question,
            tool_results=combined_tool_results
        )

        if autonomous_results:
            combined_tool_results['autonomous_results'] = autonomous_results
            combined_tool_results['tool_results']['autonomous_tasks'] = autonomous_results

        # ═══════════════════════════════════════════════════════
        # Step 4: Extract comparative features via VLM
        # ═══════════════════════════════════════════════════════
        print("  Step 3: Extracting comparative features...")
        extracted_features = self._extract_features_q4(
            tool_results_a=tool_results_a,
            tool_results_b=tool_results_b,
            instances=instances,
            question=question
        )

        # ═══════════════════════════════════════════════════════
        # Step 5: Generate Q4 explanation (one VLM call)
        # ═══════════════════════════════════════════════════════
        print("  Step 4: Generating Q4 comparative explanation...")

        prompt_builder = self._get_prompt_builder(question_template, question)
        context = self._build_context_q4(question, model_info, instances)

        results_for_prompt = {
            "tool_results": combined_tool_results['tool_results'],
            "extracted_features": extracted_features,
            "autonomous_results": autonomous_results
        }

        # Use Q4-specific actor prompt
        parsed_result = self._generate_explanation_q4(
            prompt_builder=prompt_builder,
            context=context,
            strategy=strategy,
            results=results_for_prompt,
            tool_results=combined_tool_results,
            instances=instances
        )

        # Ensure Q4 output format
        if 'output' not in parsed_result:
            parsed_result['output'] = {}

        if 'input_A' not in parsed_result['output']:
            parsed_result['output']['input_A'] = extracted_features.get('output_A', {})
        if 'input_B' not in parsed_result['output']:
            parsed_result['output']['input_B'] = extracted_features.get('output_B', {})

        # Add metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = 4
        parsed_result['is_q4'] = True
        parsed_result['tool_results'] = combined_tool_results['tool_results']
        parsed_result['autonomous_results'] = autonomous_results
        parsed_result['visualization_paths'] = combined_tool_results['visualization_paths']

        # Save results
        self._save_results(parsed_result, question)

        print(f"\nQ4 Explanation generated")
        return parsed_result

    def _extract_features_q4(
        self,
        tool_results_a: Dict[str, Any],
        tool_results_b: Dict[str, Any],
        instances: List[Dict[str, Any]],
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features for Q4 comparison (Instance A vs B)."""
        modality = question.get('modality', 'vision')
        pred_a = instances[0].get('prediction', {})
        pred_b = instances[1].get('prediction', {})

        # Get image size from tool results
        image_width, image_height = 224, 224
        for tool_name, result in tool_results_a.get('tool_results', {}).items():
            if isinstance(result, dict) and result.get('success'):
                img_size = result.get('original_image_size', {})
                if img_size:
                    image_width = img_size.get('width', image_width)
                    image_height = img_size.get('height', image_height)
                    break

        # Build size constraint based on modality
        if modality == "vision":
            output_format = '"bounding_box": [x_min, y_min, x_max, y_max]'
            size_constraint = f"""
**CRITICAL IMAGE SIZE CONSTRAINT:**
- Image dimensions: {image_width} x {image_height} pixels
- ALL coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- Example valid bounding box for this image: [10, 20, 80, 70]
- Ensure x_max <= {image_width} and y_max <= {image_height}. Do not hallucinate coordinates outside this range."""
        elif modality == "text":
            output_format = '"start_index": int, "end_index": int'
            text_input = question.get('text_input', '')
            text_length = len(text_input) if text_input else 100
            size_constraint = f"""
**CRITICAL TEXT LENGTH CONSTRAINT:**
- Text length: {text_length} characters
- ALL indices MUST be within: [0, {text_length}]"""
        else:
            output_format = '"feature_key": "string"'
            size_constraint = ""

        # Format predictions
        pred_a_class = pred_a.get('predicted_class_name', pred_a.get('predicted_class_idx', 'Unknown'))
        pred_a_conf = pred_a.get('confidence', 0.0)
        pred_b_class = pred_b.get('predicted_class_name', pred_b.get('predicted_class_idx', 'Unknown'))
        pred_b_conf = pred_b.get('confidence', 0.0)

        prompt = f"""You are an expert XAI analyst. Analyze the XAI results and extract key features for two instances with DIFFERENT predictions.

## Context
- Question: {question.get('question', 'Why are instances A and B given different predictions?')}
- Question Type: Q4 (Contrastive Instances)
- Modality: {modality}
{size_constraint}

## Instance A Analysis
- Model Prediction: {pred_a_class} (confidence: {pred_a_conf:.2%})

**Tool Results Summary:**
{tool_results_a.get('tool_results_summary', 'No tools executed')}

**Detailed Tool Statistics:**
{self._format_tool_statistics(tool_results_a)}

## Instance B Analysis
- Model Prediction: {pred_b_class} (confidence: {pred_b_conf:.2%})

**Tool Results Summary:**
{tool_results_b.get('tool_results_summary', 'No tools executed')}

**Detailed Tool Statistics:**
{self._format_tool_statistics(tool_results_b)}

## Your Task
For EACH instance, identify THE SINGLE MOST IMPORTANT region/feature that causes its prediction.
Explain why these regions lead to DIFFERENT predictions.

**CRITICAL: Your response MUST be valid JSON with this exact structure:**
{{
    "output_A": {{
        {output_format},
        "description": "2-3 sentences explaining why this region causes prediction A"
    }},
    "output_B": {{
        {output_format},
        "description": "2-3 sentences explaining why this region causes prediction B"
    }},
    "comparison": "Brief explanation of why these regions lead to different predictions"
}}

**Important Guidelines:**
- For vision: Provide bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates
  - MUST respect image bounds: x in [0, {image_width}], y in [0, {image_height}]
  - Use the top_attention_coords from tool results to determine the region
- For text: Provide character indices (start_index, end_index)
- For tabular: Provide the exact feature name

Respond with ONLY valid JSON:"""

        return self.invoke_vlm_for_json(prompt)

    def _compute_bbox_from_tool_results(
        self,
        tool_results: Dict[str, Any],
        image_size: Tuple[int, int]
    ) -> List[int]:
        """Compute bounding box from tool results."""
        width, height = image_size
        all_xs = []
        all_ys = []

        for tool_name, result in tool_results.get('tool_results', {}).items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            # Check for suggested_bounding_box first
            if 'suggested_bounding_box' in result:
                return result['suggested_bounding_box']

            stats = result.get('statistics', {})
            top_coords = (
                stats.get('top_attention_coords') or
                stats.get('top_importance_coords') or
                stats.get('top_gradient_coords')
            )

            if top_coords:
                for coord in top_coords:
                    x, y = coord.get('x', 0), coord.get('y', 0)
                    if 0 <= x < width and 0 <= y < height:
                        all_xs.append(x)
                        all_ys.append(y)

        if all_xs and all_ys:
            padding = max(width, height) // 10
            x_min = max(0, min(all_xs) - padding)
            y_min = max(0, min(all_ys) - padding)
            x_max = min(width, max(all_xs) + padding)
            y_max = min(height, max(all_ys) + padding)
            return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def _build_context_q4(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        instances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context for Q4."""
        modality = question.get('modality', 'vision')
        clean_model_info = {}
        if model_info:
            clean_model_info = {
                "model_name": model_info.get("model_name", "Unknown"),
                "architecture": model_info.get("architecture", "Unknown"),
                "num_classes": model_info.get("num_classes", "Unknown"),
            }

        context = {
            "user_question": question.get("question", question.get("example", "")),
            "model_info": clean_model_info,
            "num_instances": 2,
            "modality": modality,
        }

        if instances and len(instances) >= 2:
            context["prediction_A"] = instances[0].get('prediction', {})
            context["prediction_B"] = instances[1].get('prediction', {})
            context["image_path_A"] = instances[0].get('path', '')
            context["image_path_B"] = instances[1].get('path', '')

        # Add text/tabular instance data for VLM prompt
        if modality in ('text', 'tabular'):
            instance_a = question.get('instance_A', {})
            instance_b = question.get('instance_B', {})
            instance_data = []
            for label, inst in [('A', instance_a), ('B', instance_b)]:
                feat = inst.get('features', {})
                entry = {"label": label, "row_no": inst.get('row_no')}
                if modality == 'text':
                    if 'text' in feat:
                        entry["text"] = feat["text"]
                    elif 'premise' in feat:
                        entry["premise"] = feat["premise"]
                        entry["hypothesis"] = feat.get("hypothesis", "")
                elif modality == 'tabular':
                    entry["features"] = feat
                entry["target"] = inst.get('target', {})
                entry["predicted"] = inst.get('prediction', {})
                instance_data.append(entry)
            context["instance_data"] = instance_data

        return context

    def _generate_explanation_q4(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        tool_results: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate Q4 explanation using the prompt builder."""
        # Use build_actor_prompt_multi for Q4
        prompt = prompt_builder.build_actor_prompt_multi(context, strategy, results, instances)
        print(f"  Generated Q4 actor prompt ({len(prompt)} chars)")

        print("=" * 70)
        print(f"\n DEBUG Q4 EXPLANATION PROMPT:\n{prompt}")
        print("=" * 70)

        last_error = None
        for attempt in range(1, 4):
            response = self.invoke_vlm(prompt)
            print("=" * 70)
            print(f"\n DEBUG Q4 EXPLANATION RESPONSE (attempt {attempt}/3):\n{response}")
            print("=" * 70)
            try:
                return self.parse_json_response(response)
            except RuntimeError as e:
                last_error = e
                if attempt < 3:
                    print(f"  JSON parse failed (attempt {attempt}/3), retrying in 2s...")
                    time.sleep(2)
        raise last_error

    def _execute_tools(
        self,
        strategy: Dict[str, Any],
        input_path: str,
        prediction: Dict[str, Any],
        modality: str = "vision",
        question: Optional[Dict[str, Any]] = None,
        save_suffix: str = ""
    ) -> Dict[str, Any]:
        """Execute XAI tools based on strategy"""
        import re
        from xai_tools import set_output_dir

        all_viz_paths = []
        tool_summaries = []
        tool_outputs = {}

        target_class = prediction.get('predicted_class_idx', 0)

        # Build image_id and set xai_outputs directory based on question info
        # Directory structure: /xai_outputs/{modality}/{dataset_name}/{q_type}/{question_id}/
        # e.g., /xai_outputs/vision/stl10_resnet/q1/0/
        if question:
            dataset_base_name = question.get('dataset_base_name', 'unknown')
            row_no = question.get('row_no', question.get('question_id', 0))
            q_modality = question.get('modality', modality)

            # Extract dataset_name and q_type from dataset_base_name
            # e.g., "stl10_resnet_q1_test" -> dataset_name="stl10_resnet", q_type="q1"
            match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
            if match:
                dataset_name = match.group(1)  # e.g., "stl10_resnet"
                q_type_str = match.group(2)    # e.g., "q1"
            else:
                dataset_name = dataset_base_name
                q_type_str = f"q{question.get('q_type', 1)}"

            # Get instance suffix for multi-instance questions (Q4, Q9, Q10)
            instance_suffix = question.get('instance_suffix', '')

            # Set nested output directory for xai_outputs: /{modality}/{dataset_name}/{q_type}/{question_id}/
            # For multi-instance: /{modality}/{dataset_name}/{q_type}/{question_id}/{instance_suffix}/
            if instance_suffix:
                xai_output_dir = self.output_dir / "xai_outputs" / q_modality / dataset_name / q_type_str / str(row_no) / instance_suffix.strip('_')
            else:
                xai_output_dir = self.output_dir / "xai_outputs" / q_modality / dataset_name / q_type_str / str(row_no)
            xai_output_dir.mkdir(parents=True, exist_ok=True)
            set_output_dir(str(xai_output_dir))

            image_id_prefix = f"{dataset_name}_{q_type_str}_{row_no}{instance_suffix}"
        else:
            image_id_prefix = "direct"

        for tool_spec in strategy.get('selected_tools', []):
            tool_name = tool_spec.get('tool_name', 'gradcam')

            if self.tool_registry is None:
                raise RuntimeError(f"Tool registry not initialized. Cannot execute tool '{tool_name}'.")

            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                print(f"  Warning: Tool '{tool_name}' not found in registry (proposer may have hallucinated tool name), skipping.")
                continue

            print(f"  Executing {tool_name}...")
            result_str = tool.run(
                image_path=input_path,
                target_class=target_class,
                image_id=f"{image_id_prefix}_{tool_name}"
            )
            result = json.loads(result_str)

            # Compute suggested_bounding_box from tool statistics (same logic as _format_tool_statistics)
            if result.get('success'):
                stats = result.get('statistics', {})
                top_coords = (
                    stats.get('top_attention_coords') or
                    stats.get('top_importance_coords') or
                    stats.get('top_gradient_coords')
                )
                if top_coords and len(top_coords) > 0:
                    xs = [c.get('x', 0) for c in top_coords if isinstance(c, dict)]
                    ys = [c.get('y', 0) for c in top_coords if isinstance(c, dict)]
                    if xs and ys:
                        result['suggested_bounding_box'] = [min(xs), min(ys), max(xs), max(ys)]

            tool_outputs[tool_name] = result

            if result.get('success'):
                viz_path = result.get('visualization_path')
                if viz_path:
                    all_viz_paths.append({'tool': tool_name, 'path': viz_path})
                tool_summaries.append(f"{tool_name}: success")
            else:
                raise RuntimeError(f"Tool '{tool_name}' returned failure: {result.get('error', 'unknown error')}")

            # Free intermediate activation tensors left by XAI tools after each run
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Save tool outputs
        if question:
            self._save_tool_outputs(tool_outputs, question, suffix=save_suffix)

        return {
            "tool_results": tool_outputs,
            "tool_results_summary": "; ".join(tool_summaries),
            "visualization_paths": all_viz_paths,
        }

    def _execute_autonomous_tasks(
        self,
        strategy: Dict[str, Any],
        input_path: str,
        prediction: Dict[str, Any],
        question: Dict[str, Any],
        tool_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute autonomous tasks using VLM's own reasoning capabilities.

        Args:
            strategy: Strategy containing autonomous_tasks
            input_path: Path to input data
            prediction: Model prediction results
            question: Question dictionary
            tool_results: Results from XAI tools (for image size info)

        Returns:
            Dictionary mapping task_type to task results
        """
        autonomous_tasks = strategy.get('autonomous_tasks', [])

        if not autonomous_tasks:
            print("  No autonomous tasks specified in strategy")
            return {}

        print(f"\n  Executing {len(autonomous_tasks)} autonomous task(s)...")
        autonomous_results = {}
        modality = question.get('modality', 'vision')

        for i, task in enumerate(autonomous_tasks):
            task_type = task.get('task_type', 'unknown')
            query = task.get('query', '')
            expected_output = task.get('expected_output', '')

            print(f"    [{i+1}/{len(autonomous_tasks)}] Executing '{task_type}' task...")

            # Build prompt directly from task specification
            prompt = self._build_autonomous_task_prompt(
                task_type=task_type,
                query=query,
                expected_output=expected_output,
                prediction=prediction,
                question=question,
                modality=modality,
                tool_results=tool_results
            )

            # Prepare images for VLM (vision modality)
            images = []
            if modality == "vision" and input_path and os.path.exists(input_path):
                images.append(input_path)

            # Call VLM and parse JSON, with retry on parse failure
            parsed = self.invoke_vlm_for_json(prompt, images if images else None)

            autonomous_results[task_type] = {
                "success": True,
                "task_type": task_type,
                "query": query,
                "result": parsed,
                "raw_response": str(parsed)[:1000]
            }
            print(f"        '{task_type}': completed")

        return autonomous_results

    def _build_autonomous_task_prompt(
        self,
        task_type: str,
        query: str,
        expected_output: str,
        prediction: Dict[str, Any],
        question: Dict[str, Any],
        modality: str,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for autonomous task execution.

        Directly uses task_type, query, and expected_output from proposer's strategy.
        Includes modality-specific constraints (image size, text length, etc.)
        """
        pred_class = prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))
        confidence = prediction.get('confidence', 0.0)
        top_predictions = prediction.get('top_predictions', [])

        # Format top predictions
        top_pred_str = ""
        if top_predictions:
            top_pred_str = "\n".join([
                f"  - {p.get('class_name', p.get('class_idx', 'Unknown'))}: {p.get('confidence', 0):.2%}"
                for p in top_predictions[:5]
            ])

        # Build modality-specific constraints
        size_constraint = self._build_modality_constraint(modality, question, tool_results)

        prompt = f"""You are an expert AI analyst. Perform the following autonomous reasoning task.

## Task Type: {task_type}

## Context
- Question: {question.get('question', '')}
- Model Prediction: {pred_class} (confidence: {confidence:.2%})
- Modality: {modality}

## Top-5 Model Predictions
{top_pred_str if top_pred_str else "Not available"}
{size_constraint}

## Your Task
{query}

## Expected Output
{expected_output}

## Response Format
Provide your analysis as a JSON object. Include:
- Your findings/results matching the expected output format
- "explanation": Brief explanation of your analysis
- "confidence": Your confidence score (0.0-1.0)

JSON Response:"""

        return prompt

    def _build_modality_constraint(
        self,
        modality: str,
        question: Dict[str, Any],
        tool_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build modality-specific constraints for autonomous task prompts."""

        if modality == "vision":
            # Extract image size from tool results
            image_width, image_height = 224, 224  # Default
            if tool_results:
                for tool_name, result in tool_results.get('tool_results', {}).items():
                    if isinstance(result, dict) and result.get('success'):
                        img_size = result.get('original_image_size', {})
                        if img_size:
                            image_width = img_size.get('width', image_width)
                            image_height = img_size.get('height', image_height)
                            break

            min_width = max(int(image_width * 0.1), 10)
            min_height = max(int(image_height * 0.1), 10)

            return f"""
## CRITICAL IMAGE CONSTRAINTS
- Image dimensions: {image_width} x {image_height} pixels
- ALL bounding box coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- MINIMUM bounding box size: {min_width}x{min_height} pixels
- Format bounding boxes as [x_min, y_min, x_max, y_max]
- Ensure x_max > x_min and y_max > y_min"""

        elif modality == "text":
            features = question.get('features', {})
            # Handle multi-instance (Q9/Q10): features is a list of dicts
            if isinstance(features, list):
                labels = "ABCDEFGHIJ"
                image_indices = question.get('image_indices', question.get('row_no', []))
                targets = question.get('targets', question.get('target', []))
                preds_list = question.get('predictions', question.get('predicted', []))
                lines = []
                all_texts = []
                for i, feat in enumerate(features):
                    label = labels[i] if i < len(labels) else str(i)
                    idx = image_indices[i] if i < len(image_indices) else i
                    target_info = targets[i] if isinstance(targets, list) and i < len(targets) else {}
                    pred_info = preds_list[i] if isinstance(preds_list, list) and i < len(preds_list) else {}
                    target_lbl = target_info.get('label', target_info.get('value', '?')) if isinstance(target_info, dict) else target_info
                    pred_lbl = pred_info.get('label', pred_info.get('value', '?')) if isinstance(pred_info, dict) else pred_info
                    lines.append(f"\n### Instance {label} (index {idx}) — true: {target_lbl}, predicted: {pred_lbl}")
                    if isinstance(feat, dict):
                        text = feat.get('text', feat.get('premise', ''))
                        if 'premise' in feat:
                            lines.append(f"Premise: {feat['premise']}")
                            lines.append(f"Hypothesis: {feat.get('hypothesis', '')}")
                        elif text:
                            lines.append(f"```\n{text}\n```")
                        all_texts.append(text)
                    elif isinstance(feat, str):
                        lines.append(f"```\n{feat}\n```")
                        all_texts.append(feat)
                text_content_section = "\n## INPUT TEXTS (ALL INSTANCES)\n" + "\n".join(lines)
                max_text_length = max((len(t) for t in all_texts), default=100)
                return f"""{text_content_section}

## CRITICAL TEXT CONSTRAINTS
- ALL indices MUST be within the bounds of each instance's text
- Format text spans as {{"start_index": int, "end_index": int}}
- Ensure end_index > start_index"""

            # Extract actual text content for NLI (premise+hypothesis) or single-text tasks
            if isinstance(features, dict) and 'premise' in features and 'hypothesis' in features:
                premise = features.get('premise', '')
                hypothesis = features.get('hypothesis', '')
                text_input = f"Premise: {premise} Hypothesis: {hypothesis}"
                text_content_section = f"""
## INPUT TEXT
- Premise: {premise}
- Hypothesis: {hypothesis}"""
            else:
                text_input = question.get('text_input', '')
                if not text_input and isinstance(features, dict):
                    text_input = features.get('review_text', features.get('text', ''))
                if not text_input and isinstance(features, str):
                    text_input = features
                text_content_section = f"""
## INPUT TEXT
{text_input}""" if text_input else ""

            text_length = len(text_input) if text_input else 100

            return f"""{text_content_section}

## CRITICAL TEXT CONSTRAINTS
- Text length: {text_length} characters
- ALL indices MUST be within: [0, {text_length}]
- Format text spans as {{"start_index": int, "end_index": int}}
- Ensure end_index > start_index"""

        elif modality == "tabular":
            features = question.get('features', {})
            # Handle multi-instance (Q9/Q10): features is a list of dicts
            if isinstance(features, list) and features:
                # Use first instance's keys as representative feature names
                first_feat = features[0] if isinstance(features[0], dict) else {}
                feature_names = list(first_feat.keys())
                labels = "ABCDEFGHIJ"
                image_indices = question.get('image_indices', question.get('row_no', []))
                targets = question.get('targets', question.get('target', []))
                preds_list = question.get('predictions', question.get('predicted', []))
                lines = []
                for i, feat in enumerate(features):
                    label = labels[i] if i < len(labels) else str(i)
                    idx = image_indices[i] if i < len(image_indices) else i
                    target_info = targets[i] if isinstance(targets, list) and i < len(targets) else {}
                    pred_info = preds_list[i] if isinstance(preds_list, list) and i < len(preds_list) else {}
                    target_lbl = target_info.get('label', target_info.get('value', '?')) if isinstance(target_info, dict) else target_info
                    pred_lbl = pred_info.get('label', pred_info.get('value', '?')) if isinstance(pred_info, dict) else pred_info
                    feat_str = ", ".join(f"{k}={v}" for k, v in list(feat.items())[:10]) if isinstance(feat, dict) else str(feat)
                    lines.append(f"- Instance {label} (index {idx}) — true: {target_lbl}, predicted: {pred_lbl}: {feat_str}")
                instance_section = "\n## INSTANCE DATA\n" + "\n".join(lines)
                return f"""{instance_section}

## CRITICAL TABULAR CONSTRAINTS
- Available features: {feature_names}{'...' if len(feature_names) > 20 else ''}
- Use exact feature names as they appear above
- Format as {{"feature_key": "feature_name"}}"""
            else:
                feature_names = list(features.keys()) if isinstance(features, dict) and features else []
                feat_preview = (
                    ", ".join(f"{k}={repr(v)}" for k, v in list(features.items())[:15])
                    if isinstance(features, dict) else ""
                )

                return f"""
## CRITICAL TABULAR CONSTRAINTS
- Available features: {feature_names}
- Current values: {feat_preview}
- Use exact feature names as they appear above
- new_value can be numeric (e.g. 40) or a string category (e.g. "Exec-managerial")
- Format as {{"feature_key": "feature_name", "action": "change", "new_value": <value>}}"""

        return ""

    def _save_tool_outputs(self, tool_outputs: Dict[str, Any], question: Dict[str, Any], suffix: str = ""):
        """Save tool outputs to file."""
        import re
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))
        modality = question.get('modality', 'vision')
        instance_suffix = question.get('instance_suffix', '')

        # Extract dataset_name and q_type from dataset_base_name
        # e.g., "stl10_resnet_q1_test" -> dataset_name="stl10_resnet", q_type="q1"
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)  # e.g., "stl10_resnet"
            q_type_str = match.group(2)    # e.g., "q1"
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        # Directory structure: /tool_outputs/{modality}/{dataset_name}/{q_type}/{question_id}/
        # For multi-instance: /tool_outputs/{modality}/{dataset_name}/{q_type}/{question_id}/{instance}/
        if instance_suffix:
            tool_outputs_dir = self.output_dir / "tool_outputs" / modality / dataset_name / q_type_str / str(row_no) / instance_suffix.strip('_')
        else:
            tool_outputs_dir = self.output_dir / "tool_outputs" / modality / dataset_name / q_type_str / str(row_no)
        tool_outputs_dir.mkdir(parents=True, exist_ok=True)

        filename = f"tool_outputs{suffix}.json"
        output_file = tool_outputs_dir / filename

        with open(output_file, 'w') as f:
            json.dump(tool_outputs, f, indent=2)

        print(f"Tool outputs saved to: {output_file}")

    def _extract_features_via_vlm(
        self,
        tool_results: Dict[str, Any],
        input_path: Optional[str],
        question: Dict[str, Any],
        question_template: Any,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use VLM to extract features from tool results"""
        modality = question.get('modality', 'vision')
        q_type = question.get('q_type', 1)

        # Extract image size from tool results (for validation)
        image_size = None
        if modality == "vision":
            for tool_name, tool_result in tool_results.get('tool_results', {}).items():
                if isinstance(tool_result, dict) and tool_result.get('success'):
                    img_size = tool_result.get('original_image_size', {})
                    if img_size:
                        image_size = (img_size.get('width', 224), img_size.get('height', 224))
                        break

        # Build extraction prompt
        prompt = self._build_feature_extraction_prompt(
            tool_results=tool_results,
            question=question,
            prediction=prediction,
            modality=modality,
            q_type=q_type
        )

        # Collect images for VLM (only actual image files, not HTML/text artifacts)
        IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
        images = []
        if modality == "vision" and input_path and os.path.exists(input_path):
            images.append(input_path)
        for viz_info in tool_results.get('visualization_paths', []):
            viz_path = viz_info.get('path', '') if isinstance(viz_info, dict) else str(viz_info)
            if viz_path and os.path.exists(viz_path):
                ext = os.path.splitext(viz_path)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    images.append(viz_path)

        # Call VLM and parse, with retry on parse failure
        last_error = None
        for attempt in range(1, 4):
            response = self.invoke_vlm(prompt, images if images else None)
            try:
                return self._parse_feature_response(
                    response, modality, q_type,
                    tool_results=tool_results,
                    image_size=image_size
                )
            except RuntimeError as e:
                last_error = e
                if attempt < 3:
                    print(f"  Feature extraction parse failed (attempt {attempt}/3), retrying in 2s...")
                    time.sleep(2)
        raise last_error

    def _build_feature_extraction_prompt(
        self,
        tool_results: Dict[str, Any],
        question: Dict[str, Any],
        prediction: Dict[str, Any],
        modality: str,
        q_type: int
    ) -> str:
        """Build prompt for feature extraction"""
        pred_class = prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))
        confidence = prediction.get('confidence', 0.0)

        # Extract image size from tool results (for vision modality)
        image_width, image_height = 224, 224  # Default assumption
        if modality == "vision":
            # Try to get actual image size from any successful tool result
            for tool_name, tool_result in tool_results.get('tool_results', {}).items():
                if isinstance(tool_result, dict) and tool_result.get('success'):
                    img_size = tool_result.get('original_image_size', {})
                    if img_size:
                        image_width = img_size.get('width', image_width)
                        image_height = img_size.get('height', image_height)
                        break

        # Get output format based on modality
        if modality == "vision":
            output_format = '"bounding_box": [x_min, y_min, x_max, y_max]'
            size_constraint = f"""
**CRITICAL IMAGE SIZE CONSTRAINT:**
- Image dimensions: {image_width} x {image_height} pixels
- ALL coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- Example valid bounding box for this image: [10, 20, 80, 70]
- Ensure x_max <= {image_width} and y_max <= {image_height}. Do not hallucinate coordinates outside this range."""
        elif modality == "text":
            output_format = '"start_index": int, "end_index": int'
            text_input = question.get('text_input', '')
            text_length = len(text_input) if text_input else 100
            size_constraint = f"""
**CRITICAL TEXT LENGTH CONSTRAINT:**
- Text length: {text_length} characters
- ALL indices MUST be within: [0, {text_length}]"""
        else:
            output_format = '"feature_key": "string"'
            size_constraint = ""

        # Format autonomous results if available
        autonomous_summary = self._format_autonomous_results(tool_results)

        prompt = f"""You are an expert XAI analyst. Analyze the XAI results and extract the key feature.

## Context
- Question: {question.get('question', '')}
- Question Type: Q{q_type}
- Modality: {modality}
- Model Prediction: {pred_class} (confidence: {confidence:.2%})
{size_constraint}

## Tool Results Summary
{tool_results.get('tool_results_summary', 'No tools executed')}

## Detailed Tool Statistics
{self._format_tool_statistics(tool_results)}
{autonomous_summary}

## Your Task
Based on the XAI analysis, identify THE SINGLE MOST IMPORTANT region/feature that answers the question.

**CRITICAL: Your response MUST be valid JSON with this exact structure:**
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this region/feature is important",
    "confidence": confidence score
}}

**Important Guidelines:**
- For vision: Provide bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates
  - MUST respect image bounds: x in [0, {image_width}], y in [0, {image_height}]
  - Use the top_attention_coords from tool results to determine the region
- For text: Provide character indices (start_index, end_index)
- For tabular: Provide the feature/column name as feature_key
- Focus on the SINGLE most important region/feature, not multiple
- **If quantitative XAI tool statistics are unavailable or incomplete, use the autonomous analysis results above to infer the answer. If no autonomous results exist either, perform your own direct reasoning based on the model prediction and context.**

JSON Response:"""

        return prompt

    def _format_tool_statistics(self, tool_results: Dict[str, Any]) -> str:
        """Format detailed tool statistics for the prompt"""
        lines = []

        for tool_name, result in tool_results.get('tool_results', {}).items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            stats = result.get('statistics', {})


            method = result.get('method', tool_name)

            lines.append(f"\n### {tool_name}:")

            # Image size (common to all vision tools)
            img_size = result.get('original_image_size', {})
            if img_size:
                lines.append(f"- Image size: {img_size.get('width')}x{img_size.get('height')}")

            # ========== Tool-specific formatting ==========

            # GradCAM / Integrated Gradients / Guided Backprop - coordinate-based tools
            top_coords = (
                stats.get('top_attention_coords') or
                stats.get('top_importance_coords') or
                stats.get('top_gradient_coords') or
                stats.get('top_impact_coords')
            )
            if top_coords and len(top_coords) > 0:
                xs = [c.get('x', 0) for c in top_coords if isinstance(c, dict)]
                ys = [c.get('y', 0) for c in top_coords if isinstance(c, dict)]
                if xs and ys:
                    lines.append(f"- High attention region: x=[{min(xs)}-{max(xs)}], y=[{min(ys)}-{max(ys)}]")
                    lines.append(f"- Top 3 attention points: {[(c.get('x'), c.get('y')) for c in top_coords[:3]]}")

            # Attention/importance metrics (GradCAM, IG, etc.)
            if 'mean_attention' in stats:
                lines.append(f"- Mean attention: {stats['mean_attention']:.3f}")
            if 'high_attention_ratio' in stats:
                lines.append(f"- High attention ratio: {stats['high_attention_ratio']:.2%}")
            if 'mean_importance' in stats:
                lines.append(f"- Mean importance: {stats['mean_importance']:.3f}")
            if 'high_importance_ratio' in stats:
                lines.append(f"- High importance ratio: {stats['high_importance_ratio']:.2%}")
            if 'mean_gradient' in stats:
                lines.append(f"- Mean gradient: {stats['mean_gradient']:.4f}")

            # Layer CAM specific
            if 'mean_activation' in stats:
                lines.append(f"- Mean activation: {stats['mean_activation']:.3f}")
                lines.append(f"- Max activation: {stats.get('max_activation', 0):.3f}")
            if 'layer_name' in stats:
                lines.append(f"- Target layer: {stats['layer_name']}")

            # LIME specific - segment-based explanations with bounding boxes
            if 'top_positive_segments' in stats:
                lines.append(f"- Positive segments: {stats.get('num_positive_segments', 0)}")
                lines.append(f"- Negative segments: {stats.get('num_negative_segments', 0)}")
                lines.append(f"- Important region ratio: {stats.get('important_region_ratio', 0):.2%}")
                lines.append(f"- Max positive weight: {stats.get('max_positive_weight', 0):.4f}")
                top_pos = stats.get('top_positive_segments', [])[:3]
                if top_pos:
                    lines.append(f"- **TOP POSITIVE SEGMENTS (most important for prediction):**")
                    for s in top_pos:
                        bbox = s.get('bbox')
                        bbox_str = f"bbox={bbox}" if bbox else "bbox=N/A"
                        lines.append(f"  - Segment {s.get('segment_id')}: weight={s.get('weight', 0):.3f}, {bbox_str}")
                # Check for suggested_bounding_box at result level
                suggested_bbox = result.get('suggested_bounding_box')
                if suggested_bbox:
                    lines.append(f"- **SUGGESTED IMPORTANT REGION:** bbox={suggested_bbox}")

            # SHAP specific
            if 'total_positive_shap' in stats:
                lines.append(f"- Total positive SHAP: {stats['total_positive_shap']:.4f}")
                lines.append(f"- Total negative SHAP: {stats.get('total_negative_shap', 0):.4f}")
                lines.append(f"- High impact ratio: {stats.get('high_impact_ratio', 0):.2%}")

            # Object Detection specific - bounding boxes
            detections = result.get('detections', [])
            if detections:
                lines.append(f"- **DETECTED OBJECTS ({len(detections)} total):**")
                for det in detections[:5]:  # Top 5 detections
                    cls_name = det.get('class_name', 'unknown')
                    conf = det.get('confidence', 0)
                    bbox = det.get('bbox', {})
                    if bbox:
                        lines.append(f"  - {cls_name} (conf={conf:.2f}): bbox=[{bbox.get('x1')}, {bbox.get('y1')}, {bbox.get('x2')}, {bbox.get('y2')}]")
                class_counts = stats.get('class_counts', {})
                if class_counts:
                    lines.append(f"- Class distribution: {class_counts}")

            # Sensitivity Analysis specific
            if 'sensitivity_score' in stats:
                lines.append(f"- Sensitivity score: {stats['sensitivity_score']:.4f}")
                lines.append(f"- Original probability: {stats.get('original_probability', 0):.4f}")
                lines.append(f"- Final probability: {stats.get('final_probability', 0):.4f}")
                lines.append(f"- Probability drop: {stats.get('probability_drop', 0):.4f}")

            # Include description if available
            description = result.get('description', '')
            if description and len(lines) <= 3:  # Only add description if we didn't extract much
                lines.append(f"- Description: {description[:200]}")

        return "\n".join(lines) if lines else "No detailed statistics available."

    def _format_autonomous_results(self, tool_results: Dict[str, Any]) -> str:
        """Format autonomous task results for inclusion in prompts."""
        autonomous_results = tool_results.get('autonomous_results', {})

        if not autonomous_results:
            return ""

        lines = ["\n## Autonomous Reasoning Results"]

        for task_type, result in autonomous_results.items():
            if not isinstance(result, dict):
                continue

            success = result.get('success', False)
            query = result.get('query', '')

            lines.append(f"\n### {task_type.upper()} Task")
            lines.append(f"- Query: {query[:200]}...")
            lines.append(f"- Status: {'Success' if success else 'Failed'}")

            if success and 'result' in result:
                task_result = result['result']
                if isinstance(task_result, dict):
                    # Include all structured data from the result
                    for key, value in task_result.items():
                        if key == 'explanation':
                            lines.append(f"- Explanation: {value}")
                        elif key == 'confidence':
                            lines.append(f"- Confidence: {value}")
                        else:
                            # Serialize structured data (objects, discriminative_features, etc.)
                            lines.append(f"- {key}: {json.dumps(value)}")
                elif isinstance(task_result, str):
                    lines.append(f"- Result: {task_result}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _parse_feature_response(
        self,
        response: str,
        modality: str,
        q_type: int,
        tool_results: Optional[Dict[str, Any]] = None,
        image_size: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """Parse VLM feature extraction response with validation"""
        parsed = self.parse_json_response(response)

        if not parsed:
            raise RuntimeError(
                f"Failed to parse VLM response for feature extraction. "
                f"Response preview: {response[:500]}"
            )

        # Ensure output field exists
        if 'output' not in parsed:
            parsed['output'] = self._extract_output_from_parsed(parsed, modality)

        """
        # Validate and fix bounding box for vision modality
        if modality == "vision" and image_size:
            parsed['output'] = self._validate_and_fix_bounding_box(
                parsed.get('output', {}),
                image_size,
                tool_results
            )
        """

        return {
            "output": parsed.get('output', {}),
            "explanation": parsed.get('explanation', ''),
            "confidence": parsed.get('confidence', 0.5),
            "raw_response": response[:500]
        }

    def _validate_and_fix_bounding_box(
        self,
        output: Dict[str, Any],
        image_size: tuple,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate bounding box and fix if out of bounds or too small"""
        width, height = image_size
        bbox = output.get('bounding_box', [])

        # Minimum size: at least 10% of image dimensions
        min_width = max(int(width * 0.1), 10)
        min_height = max(int(height * 0.1), 10)

        if not bbox or len(bbox) != 4:
            # No bounding box, try to compute from tool results
            print(f"[ActorAgent] No valid bbox provided, computing from tool results...")
            return self._compute_bbox_from_tools(tool_results, image_size)

        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Check if bbox is completely out of bounds
        if x_min >= width or y_min >= height or x_max <= 0 or y_max <= 0:
            print(f"[ActorAgent] WARNING: VLM returned out-of-bounds bbox {bbox} for {width}x{height} image")
            print(f"[ActorAgent] Computing bounding box from tool results instead...")
            return self._compute_bbox_from_tools(tool_results, image_size)

        # Check if bbox is too small
        if bbox_width < min_width or bbox_height < min_height:
            print(f"[ActorAgent] WARNING: VLM returned too-small bbox {bbox} ({bbox_width}x{bbox_height} pixels)")
            print(f"[ActorAgent] Minimum required size: {min_width}x{min_height} pixels")

            # First, try to use object detection bbox if available
            obj_det_bbox = self._get_object_detection_bbox(tool_results, image_size)
            if obj_det_bbox:
                print(f"[ActorAgent] Using object detection bbox instead: {obj_det_bbox}")
                return {"bounding_box": obj_det_bbox}

            # Otherwise, expand the small bbox around its center
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # Expand to minimum size
            half_w = max(bbox_width / 2, min_width / 2)
            half_h = max(bbox_height / 2, min_height / 2)

            x_min = max(0, int(center_x - half_w))
            x_max = min(width, int(center_x + half_w))
            y_min = max(0, int(center_y - half_h))
            y_max = min(height, int(center_y + half_h))

            print(f"[ActorAgent] Expanded bbox to: [{x_min}, {y_min}, {x_max}, {y_max}]")
            return {"bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)]}

        # Clip to image bounds
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(1, min(x_max, width))
        y_max = max(1, min(y_max, height))

        # Ensure min < max
        if x_min >= x_max:
            x_max = min(x_min + min_width, width)
        if y_min >= y_max:
            y_max = min(y_min + min_height, height)

        return {"bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)]}

    def _get_object_detection_bbox(
        self,
        tool_results: Optional[Dict[str, Any]],
        image_size: tuple
    ) -> Optional[List[int]]:
        """Get the best object detection bounding box if available"""
        if not tool_results:
            return None

        width, height = image_size
        best_detection = None
        best_confidence = 0

        for tool_name, result in tool_results.get('tool_results', {}).items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            detections = result.get('detections', [])
            for det in detections:
                conf = det.get('confidence', 0)
                bbox = det.get('bbox', {})
                if bbox and conf > best_confidence:
                    best_confidence = conf
                    best_detection = bbox

        if best_detection:
            x1 = max(0, min(best_detection.get('x1', 0), width))
            y1 = max(0, min(best_detection.get('y1', 0), height))
            x2 = max(0, min(best_detection.get('x2', width), width))
            y2 = max(0, min(best_detection.get('y2', height), height))
            if x2 > x1 and y2 > y1:
                return [int(x1), int(y1), int(x2), int(y2)]

        return None

    def _compute_bbox_from_tools(
        self,
        tool_results: Optional[Dict[str, Any]],
        image_size: tuple
    ) -> Dict[str, Any]:
        """Compute bounding box directly from tool results, prioritizing object detection"""
        width, height = image_size

        # Minimum size: at least 10% of image dimensions
        min_width = max(int(width * 0.1), 10)
        min_height = max(int(height * 0.1), 10)

        if not tool_results:
            # Default to center region (at least 50% of image)
            margin_x = width // 4
            margin_y = height // 4
            return {"bounding_box": [margin_x, margin_y, width - margin_x, height - margin_y]}

        # PRIORITY 1: Use object detection bbox if available
        obj_det_bbox = self._get_object_detection_bbox(tool_results, image_size)
        if obj_det_bbox:
            print(f"[ActorAgent] Using object detection bbox: {obj_det_bbox}")
            return {"bounding_box": obj_det_bbox}

        # PRIORITY 2: Compute from attention/importance coordinates
        all_xs = []
        all_ys = []

        for tool_name, result in tool_results.get('tool_results', {}).items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            stats = result.get('statistics', {})

            # Get top coordinates from various tool types
            top_coords = (
                stats.get('top_attention_coords') or
                stats.get('top_importance_coords') or
                stats.get('top_gradient_coords')
            )

            if top_coords:
                for coord in top_coords:
                    x, y = coord.get('x', 0), coord.get('y', 0)
                    if 0 <= x < width and 0 <= y < height:
                        all_xs.append(x)
                        all_ys.append(y)

        if all_xs and all_ys:
            # Compute center of attention region
            center_x = sum(all_xs) / len(all_xs)
            center_y = sum(all_ys) / len(all_ys)

            # Calculate spread of attention points
            x_spread = max(all_xs) - min(all_xs)
            y_spread = max(all_ys) - min(all_ys)

            # Ensure minimum size with extra padding (20% of image or spread, whichever is larger)
            half_w = max(x_spread / 2 + 5, min_width, width * 0.1)
            half_h = max(y_spread / 2 + 5, min_height, height * 0.1)

            x_min = max(0, int(center_x - half_w))
            y_min = max(0, int(center_y - half_h))
            x_max = min(width, int(center_x + half_w))
            y_max = min(height, int(center_y + half_h))

            print(f"[ActorAgent] Computed expanded bbox from attention coords: [{x_min}, {y_min}, {x_max}, {y_max}]")
            return {"bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)]}
            

    def _extract_output_from_parsed(self, parsed: Dict, modality: str) -> Dict:
        """Try to extract output from various parsed formats"""
        if modality == "vision":
            # Look for bounding_box in various places
            if 'bounding_box' in parsed:
                return {"bounding_box": parsed['bounding_box']}
            if 'responsible_regions' in parsed and parsed['responsible_regions']:
                region = parsed['responsible_regions'][0]
                bbox = region.get('bbox', region.get('bounding_box'))
                if bbox:
                    if isinstance(bbox, dict):
                        return {"bounding_box": [bbox.get('x1', 0), bbox.get('y1', 0),
                                                  bbox.get('x2', 100), bbox.get('y2', 100)]}
                    return {"bounding_box": bbox}
        elif modality == "text":
            if 'start_index' in parsed:
                return {"start_index": parsed['start_index'], "end_index": parsed.get('end_index', 0)}
        elif modality == "tabular":
            if 'feature_key' in parsed:
                return {"feature_key": parsed['feature_key']}

        return {}

    def _get_default_features(self, modality: str) -> Dict[str, Any]:
        """Return default features when extraction fails"""
        default_output = {
            "vision": {"bounding_box": [0, 0, 100, 100]},
            "text": {"start_index": 0, "end_index": 10},
            "tabular": {"feature_key": "unknown"}
        }
        return {
            "output": default_output.get(modality, {}),
            "explanation": "Feature extraction failed, using default.",
            "confidence": 0.1
        }

    def _format_output_from_features(
        self,
        features: Dict[str, Any],
        modality: str
    ) -> Dict[str, Any]:
        """Format output from extracted features"""
        return features.get('output', self._get_default_features(modality)['output'])

    def _get_prompt_builder(self, question_template: Any, question: Dict) -> Any:
        """Get prompt builder from template or create one"""
        if hasattr(question_template, 'build_actor_prompt'):
            return question_template

        if hasattr(question_template, 'prompt_builder') and question_template.prompt_builder:
            return question_template.prompt_builder

        from prompts import get_prompt_builder
        return get_prompt_builder(question.get('q_type', 1), question.get('modality', 'vision'))

    def _build_context(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        input_path: Optional[str]
    ) -> Dict[str, Any]:
        """Build context for prompt builder"""
        modality = question.get('modality', 'vision')

        context = {
            "user_question": question.get("question", ""),
            "model_info": model_info or {},
            "prediction": prediction or {},
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
            context["text_description"] = preview
            # Build single-instance data for prompt inclusion
            context["modality"] = modality
            context["instance_data_single"] = {
                "text": features.get("text", "") if isinstance(features, dict) else str(features),
                "premise": features.get("premise", "") if isinstance(features, dict) else "",
                "hypothesis": features.get("hypothesis", "") if isinstance(features, dict) else "",
                "features": features,
            }
        elif modality == "tabular":
            features = question.get("features", {})
            context["input_data"] = features
            context["modality"] = modality
            if isinstance(features, dict):
                feat_str = ", ".join(f"{k}={v}" for k, v in list(features.items())[:10])
                context["data_description"] = f"Features: {feat_str}"
                context["instance_data_single"] = {"features": features}
            else:
                context["data_description"] = str(features)[:400]
                context["instance_data_single"] = {"features": {}}

        return context

    def _generate_explanation_with_prompt_builder(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanation using prompt builder"""
        prompt = prompt_builder.build_actor_prompt(context, strategy, results)

        parsed = self.invoke_vlm_for_json(prompt)

        # Merge with tool results
        parsed['tool_results'] = tool_results.get('tool_results', {})
        parsed['visualization_paths'] = tool_results.get('visualization_paths', [])
        return parsed

    # =========================================================================
    # Pure-Reasoning (No-Tools) Methods
    # =========================================================================

    def _build_no_tools_prompt(
        self,
        question: Dict[str, Any],
        prediction: Dict[str, Any],
        modality: str,
        q_type: int,
        image_size: Tuple[int, int] = (224, 224),
        input_data: Any = None,
        all_input_datas: Optional[List] = None,
        all_predictions: Optional[List[Dict]] = None
    ) -> str:
        """
        Build prompt for VLM to analyze input directly without any XAI tool results.
        Supports vision, text, and tabular modalities.

        Output format is driven by QUESTION_OUTPUT_SCHEMAS so it automatically
        covers all Q-type x modality combinations without per-case hardcoding.

        For multi-instance Q types (Q4, Q9, Q10), pass all_input_datas and
        all_predictions so the VLM can reason about every instance.
        """
        from prompts.output_schemas import get_output_schema

        # Determine if we're in multi-instance mode
        is_multi = bool(all_input_datas and len(all_input_datas) > 1)
        instance_labels = [chr(ord('A') + i) for i in range(len(all_input_datas))] if is_multi else []

        _pred = prediction or {}
        pred_class = _pred.get('predicted_class_name', _pred.get('predicted_class_idx', 'Unknown'))
        confidence = _pred.get('confidence', 0.0)

        # Canonical output schema for this Q-type x modality
        schema = get_output_schema(q_type, modality)
        schema_str = json.dumps(schema, indent=4)

        # Build modality-specific context sections
        if modality == "vision":
            image_width, image_height = image_size
            size_constraint = f"""
**CRITICAL IMAGE SIZE CONSTRAINT:**
- Image dimensions: {image_width} x {image_height} pixels
- ALL bounding_box coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- Example valid bounding box: [10, 20, 80, 70]"""
            if is_multi:
                pred_lines = []
                for lbl, pred in zip(instance_labels, all_predictions or []):
                    cls = pred.get('predicted_class_name', pred.get('predicted_class_idx', '?'))
                    conf = pred.get('confidence', 0.0)
                    pred_lines.append(f"  - Instance {lbl}: {cls} (confidence: {conf:.2%})")
                input_section = (
                    "\n## Instances (images provided in order)\n"
                    + "\n".join(pred_lines) + "\n"
                )
            else:
                input_section = ""  # Single image passed separately
            guidelines = (
                "- Replace every placeholder value with your actual analysis result\n"
                "- Bounding box must stay within image bounds"
            )

        elif modality == "text":
            def _extract_text(data) -> str:
                if isinstance(data, str):
                    return data
                if isinstance(data, dict):
                    premise = data.get('premise', '')
                    hypothesis = data.get('hypothesis', '')
                    return f"Premise: {premise}\nHypothesis: {hypothesis}" if premise else str(data)
                return ''

            if is_multi:
                parts = []
                max_len = 0
                for lbl, data, pred in zip(instance_labels,
                                           all_input_datas,
                                           all_predictions or [{}] * len(all_input_datas)):
                    txt = _extract_text(data)
                    max_len = max(max_len, len(txt))
                    cls = pred.get('predicted_class_name', pred.get('predicted_class_idx', '?'))
                    display = txt[:500] + "..." if len(txt) > 500 else txt
                    parts.append(f"### Instance {lbl} (prediction: {cls})\n```\n{display}\n```")
                text_length = max_len if max_len > 0 else 100
                size_constraint = f"""
**CRITICAL TEXT LENGTH CONSTRAINT (longest instance: {text_length} chars):**
- start_index MUST be >= 0, end_index MUST be <= length of that instance's text, end_index > start_index"""
                input_section = "\n## Input Instances\n" + "\n\n".join(parts) + "\n"
            else:
                text_input = question.get('text_input', '')
                if not text_input and input_data:
                    text_input = _extract_text(input_data)
                text_length = len(text_input) if text_input else 100
                size_constraint = f"""
**CRITICAL TEXT LENGTH CONSTRAINT:**
- Text length: {text_length} characters
- start_index MUST be >= 0, end_index MUST be <= {text_length}, end_index > start_index"""
                display_text = text_input[:1000] + "..." if len(text_input) > 1000 else text_input
                input_section = f"\n## Input Text\n```\n{display_text}\n```\n"

            guidelines = (
                "- Replace every placeholder value with your actual analysis result\n"
                "- start_index / end_index must be valid character positions in each instance's text"
            )

        else:  # tabular
            size_constraint = ""

            def _extract_features(data) -> dict:
                return data if isinstance(data, dict) else {}

            if is_multi:
                parts = []
                all_features = {}
                q_features = question.get('features', {})
                for i, (lbl, data, pred) in enumerate(zip(
                        instance_labels,
                        all_input_datas,
                        all_predictions or [{}] * len(all_input_datas))):
                    feats = _extract_features(data)
                    if not feats:
                        if isinstance(q_features, list) and i < len(q_features):
                            feats = q_features[i] if isinstance(q_features[i], dict) else {}
                        elif isinstance(q_features, dict):
                            feats = q_features
                    all_features = feats  # same schema across instances
                    cls = pred.get('predicted_class_name', pred.get('predicted_class_idx', '?'))
                    feat_lines = "\n".join(f"  - {k}: {v}" for k, v in feats.items())
                    parts.append(f"### Instance {lbl} (prediction: {cls})\n{feat_lines}")
                input_section = (
                    "\n## Input Instances\n"
                    + "\n\n".join(parts)
                    + f"\n\n**Available feature names:** {list(all_features.keys())}\n"
                )
            else:
                features = question.get('features', {})
                if not features and isinstance(input_data, dict):
                    features = input_data
                if features:
                    feat_lines = "\n".join(f"  - {k}: {v}" for k, v in features.items())
                    input_section = (
                        f"\n## Input Features\n{feat_lines}\n"
                        f"\n**Available feature names:** {list(features.keys())}\n"
                    )
                else:
                    input_section = ""

            guidelines = (
                "- feature_key must be the EXACT name of one of the available features listed above\n"
                "- Replace every placeholder value with your actual analysis result"
            )

        # Prediction summary line (single-instance or first instance)
        pred_summary = f"- Model Prediction: {pred_class} (confidence: {confidence:.2%})"
        if is_multi and modality != "vision":
            pred_summary = f"- Instances: {len(all_input_datas)} (predictions shown per instance below)"

        prompt = f"""You are an expert XAI analyst. Analyze the input DIRECTLY and answer the question about the model's prediction.

## Context
- Question: {question.get('question', question.get('q', 'What is most responsible for the prediction?'))}
- Question Type: Q{q_type}
- Modality: {modality}
{pred_summary}
{size_constraint}
{input_section}
## IMPORTANT NOTE
**NO XAI tool results are available.** Analyze the input DIRECTLY.

## Required Output Format
Your response MUST be valid JSON matching this EXACT structure (replace placeholder values with your analysis):

{schema_str}

**Guidelines:**
{guidelines}

Respond with ONLY valid JSON:"""

        return prompt

    def _prepare_images_for_vlm(
        self,
        input_tensor: Any,
        input_path: str,
        modality: str,
        input_tensors: Optional[List] = None,
        input_paths: Optional[List[str]] = None
    ) -> Optional[List]:
        """Prepare images for VLM invocation.

        For multi-instance Q types (Q4, Q9, Q10), pass input_tensors / input_paths
        to include all instances.  Falls back to the single-instance parameters
        when the lists are not provided.
        """
        if modality != "vision":
            return None

        from PIL import Image
        import os

        def _tensor_to_pil(t) -> Optional[Any]:
            """Convert a single tensor to PIL, or return None."""
            try:
                import torch, numpy as np
                if isinstance(t, torch.Tensor):
                    t = t.detach().cpu()
                    if t.dim() == 4:
                        t = t[0]
                    if t.dim() == 3:
                        arr = t.permute(1, 2, 0).numpy()
                        mean = np.array([0.485, 0.456, 0.406])
                        std  = np.array([0.229, 0.224, 0.225])
                        arr = (arr * std + mean) * 255
                        return Image.fromarray(arr.clip(0, 255).astype('uint8'))
            except Exception:
                pass
            return None

        def _load_single(tensor, path) -> Optional[Any]:
            if isinstance(tensor, Image.Image):
                return tensor.convert("RGB")
            if path and os.path.exists(path):
                return Image.open(path).convert("RGB")
            img = _tensor_to_pil(tensor)
            return img

        # Collect source list: prefer multi-instance lists
        tensors = input_tensors if input_tensors else [input_tensor]
        paths   = input_paths   if input_paths   else [input_path]

        images = []
        for t, p in zip(tensors, paths):
            img = _load_single(t, p)
            if img is not None:
                images.append(img)

        return images if images else None

    def _parse_no_tools_response(
        self,
        response: str,
        modality: str,
        q_type: int,
        image_size: Tuple[int, int] = (224, 224),
        text_length: int = 100,
        available_features: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse VLM response for no-tools analysis. Supports all modalities."""
        import re

        def clip_bounding_box(bbox: List, width: int, height: int) -> List:
            """Clip bounding box coordinates to valid image bounds."""
            if not bbox or len(bbox) != 4:
                return [0, 0, width, height]
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0, min(int(x_min), width - 1))
            y_min = max(0, min(int(y_min), height - 1))
            x_max = max(x_min + 1, min(int(x_max), width))
            y_max = max(y_min + 1, min(int(y_max), height))
            return [x_min, y_min, x_max, y_max]

        def clip_text_indices(start: int, end: int, length: int) -> Tuple[int, int]:
            """Clip text indices to valid range."""
            start = max(0, min(int(start), length - 1))
            end = max(start + 1, min(int(end), length))
            return start, end

        width, height = image_size

        def clip_region_dict(d: Dict) -> Dict:
            """Clip/validate a single region dict in-place based on modality."""
            if not isinstance(d, dict):
                return d
            if modality == "vision" and 'bounding_box' in d:
                d['bounding_box'] = clip_bounding_box(d['bounding_box'], width, height)
            elif modality == "text":
                if 'start_index' in d and 'end_index' in d:
                    s, e = clip_text_indices(d['start_index'], d['end_index'], text_length)
                    d['start_index'], d['end_index'] = s, e
            elif modality == "tabular" and available_features:
                if 'feature_key' in d and d['feature_key'] not in available_features:
                    d['feature_key'] = available_features[0] if available_features else "unknown"
            return d

        def validate_and_fix_output(parsed: Dict, modality: str) -> Dict:
            """Validate and fix output based on modality."""
            if 'output' not in parsed:
                return parsed

            output = parsed['output']

            # Direct region fields (Q1-Q3, Q8)
            clip_region_dict(output)

            # Nested single-dict structures
            for nested_key in ('change_plan', 'masked_region',
                               'correct_instance_features', 'wrong_instance_features'):
                if nested_key in output and isinstance(output[nested_key], dict):
                    clip_region_dict(output[nested_key])

            # Multi-instance: all input_* keys (Q4, Q9)
            for key, val in output.items():
                if key.startswith('input_') and isinstance(val, dict):
                    clip_region_dict(val)

            parsed['output'] = output
            return parsed

        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())

                if 'output' in parsed:
                    return validate_and_fix_output(parsed, modality)

                # Try to restructure if output fields are at top level
                if 'change_plan' in parsed:
                    return validate_and_fix_output({
                        "output": {"change_plan": parsed['change_plan']},
                        "explanation": parsed.get('explanation', 'VLM direct analysis'),
                        "confidence": parsed.get('confidence', 0.5)
                    }, modality)
                elif modality == "vision" and 'bounding_box' in parsed:
                    clipped_bbox = clip_bounding_box(parsed['bounding_box'], width, height)
                    return {
                        "output": {"bounding_box": clipped_bbox},
                        "explanation": parsed.get('explanation', 'VLM direct analysis'),
                        "confidence": parsed.get('confidence', 0.5)
                    }
                elif modality == "text" and 'start_index' in parsed and 'end_index' in parsed:
                    start, end = clip_text_indices(parsed['start_index'], parsed['end_index'], text_length)
                    return {
                        "output": {"start_index": start, "end_index": end},
                        "explanation": parsed.get('explanation', 'VLM direct analysis'),
                        "confidence": parsed.get('confidence', 0.5)
                    }
                elif modality == "tabular" and 'feature_key' in parsed:
                    feature_key = parsed['feature_key']
                    if available_features and feature_key not in available_features:
                        feature_key = available_features[0] if available_features else "unknown"
                    return {
                        "output": {"feature_key": feature_key},
                        "explanation": parsed.get('explanation', 'VLM direct analysis'),
                        "confidence": parsed.get('confidence', 0.5)
                    }

            except json.JSONDecodeError:
                pass

        # Try direct parse
        try:
            parsed = json.loads(response)
            if 'output' in parsed:
                return validate_and_fix_output(parsed, modality)
        except json.JSONDecodeError:
            pass

        raise RuntimeError(f"Failed to parse no-tools VLM response. Response preview: {response[:300]}")

    def _run_pure_reasoning(
        self,
        question: Dict[str, Any],
        question_template: Any,
        strategy: Dict[str, Any],
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        instances: Optional[List[Dict[str, Any]]] = None,
        suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Execute pure-reasoning mode: no XAI tools, direct VLM analysis.
        Used when proposer selects no tools and no autonomous tasks.

        Covers single instance, multi-instance (Q9/Q10), and Q4 (A vs B).
        """
        modality = question.get('modality', 'vision')
        q_type = question.get('q_type', 1)

        print("  [Actor] Pure-reasoning mode: direct VLM analysis (no tools)")

        # Normalize to lists for unified handling
        if instances:
            # Q4: instances = [{'prediction': ..., 'path': ..., 'label': 'A'}, ...]
            _paths = [inst['path'] for inst in instances]
            _preds = [inst.get('prediction', {}) for inst in instances]
        elif input_paths and predictions:
            _paths = input_paths
            _preds = predictions
        else:
            _paths = [input_path] if input_path else []
            _preds = [prediction or {}]

        is_multi = len(_paths) > 1
        _pred = _preds[0] if _preds else {}

        # Prepare images for vision first so we know the exact size VLM will see
        images = self._prepare_images_for_vlm(
            input_tensor=None,
            input_path=_paths[0] if _paths else "",
            modality=modality,
            input_tensors=None,
            input_paths=_paths if is_multi else None
        )

        # Get image size from the prepared PIL images (what VLM actually sees)
        image_size = (224, 224)
        if modality == "vision" and images:
            image_size = images[0].size  # (width, height) of what VLM receives

        # Build prompt (pass placeholder list for multi so is_multi is detected)
        all_input_datas = [None] * len(_paths) if is_multi else None
        all_predictions = _preds if is_multi else None

        prompt = self._build_no_tools_prompt(
            question=question,
            prediction=_pred,
            modality=modality,
            q_type=q_type,
            image_size=image_size,
            input_data=None,
            all_input_datas=all_input_datas,
            all_predictions=all_predictions
        )

        # Compute parsing parameters (independent of response, compute once)
        text_length = 100
        available_features = None

        if modality == "text":
            text_input = question.get('text_input', '')
            if text_input:
                text_length = len(text_input)
            else:
                features_list = question.get('features', [])
                if isinstance(features_list, list):
                    max_len = 0
                    for feat in features_list:
                        if isinstance(feat, dict):
                            txt = feat.get('premise', '') + feat.get('hypothesis', '')
                            max_len = max(max_len, len(txt))
                        elif isinstance(feat, str):
                            max_len = max(max_len, len(feat))
                    text_length = max_len or 100
                elif isinstance(features_list, dict):
                    txt = features_list.get('premise', '') + features_list.get('hypothesis', '')
                    text_length = len(txt) or 100
        elif modality == "tabular":
            features = question.get('features', {})
            if isinstance(features, list) and features:
                features = features[0] if isinstance(features[0], dict) else {}
            if isinstance(features, dict):
                available_features = list(features.keys()) if features else None

        # Invoke VLM and parse, with retry on parse failure
        last_error = None
        parsed_result = None
        for attempt in range(1, 4):
            response = self.invoke_vlm(prompt, images)
            try:
                parsed_result = self._parse_no_tools_response(
                    response=response,
                    modality=modality,
                    q_type=q_type,
                    image_size=image_size,
                    text_length=text_length,
                    available_features=available_features
                )
                break
            except RuntimeError as e:
                last_error = e
                if attempt < 3:
                    print(f"  Pure-reasoning parse failed (attempt {attempt}/3), retrying in 2s...")
                    time.sleep(2)
        else:
            raise last_error

        # Attach metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = q_type
        parsed_result['tool_results'] = {}
        parsed_result['autonomous_results'] = {}
        parsed_result['visualization_paths'] = []
        parsed_result['pure_reasoning'] = True
        if is_multi:
            parsed_result['is_multi_instance'] = True
            parsed_result['num_instances'] = len(_paths)
        if instances:
            parsed_result['is_q4'] = True

        # Save results
        self._save_results(parsed_result, question, suffix=suffix)

        print(f"  [Actor] Pure-reasoning complete.")
        return parsed_result

    def _create_error_result(self, error: str, question: Dict) -> Dict[str, Any]:
        """Create error result"""
        modality = question.get('modality', 'vision')
        return {
            "explanation": f"Execution failed: {error}",
            "output": self._get_default_features(modality)['output'],
            "confidence": 0.0,
            "error": error,
            "question_id": question.get('question_id', 'unknown')
        }

    def _save_results(
        self,
        results: Dict[str, Any],
        question: Dict[str, Any],
        suffix: str = ""
    ):
        """Save results to file"""
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

        # Format: /results/{modality}/{dataset_name}/{q_type}/{question_id}/result.json
        filename = f"result{suffix}" if suffix else "result"
        subdir = f"results/{modality}/{dataset_name}/{q_type_str}/{row_no}"
        filepath = self.save_json(results, filename, subdir)
        print(f"Results saved to: {filepath}")

    def run_with_reflection(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: Optional[str],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        actor_reflection: str,
        original_results: Dict[str, Any],
        # Multi-instance parameters (for Q9, Q10)
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate improved explanation based on Critic's reflection.

        Args:
            strategy: Strategy (may be improved) from Proposer
            question: Question dictionary
            question_template: QuestionTemplate instance
            input_path: Path to input data (single instance)
            model_info: Model information
            prediction: Model prediction results (single instance)
            actor_reflection: JSON string with feedback from Critic
            original_results: The original results that were evaluated
            input_paths: List of paths for multi-instance questions
            predictions: List of predictions for multi-instance questions

        Returns:
            Improved results dictionary
        """
        is_multi = question.get('is_multi_instance', False) and input_paths and predictions

        print("\n" + "=" * 70)
        if is_multi:
            print(f"ACTOR AGENT: Re-executing with Reflection (Multi-Instance, {len(input_paths)} instances)")
        else:
            print("ACTOR AGENT: Re-executing with Reflection")
        print("=" * 70)

        modality = question.get('modality', 'vision')

        if is_multi:
            return self._run_with_reflection_multi(
                strategy=strategy,
                question=question,
                question_template=question_template,
                input_paths=input_paths,
                model_info=model_info,
                predictions=predictions,
                actor_reflection=actor_reflection,
                original_results=original_results
            )

        # --- Single-instance reflection path (original logic) ---

        # Pure-reasoning mode: no tools and no autonomous tasks
        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            return self._run_pure_reasoning(
                question=question,
                question_template=question_template,
                strategy=strategy,
                input_path=input_path,
                model_info=model_info,
                prediction=prediction,
                suffix="_improved"
            )

        # Step 1: Execute XAI tools (with potentially new strategy)
        print("  Step 1: Executing XAI tools...")
        tool_results = self._execute_tools(
            strategy=strategy,
            input_path=input_path or "",
            prediction=prediction or {},
            modality=modality,
            question=question,
            save_suffix="_improved"
        )

        # Step 1.5: Execute autonomous tasks if any
        autonomous_results = self._execute_autonomous_tasks(
            strategy=strategy,
            input_path=input_path or "",
            prediction=prediction or {},
            question=question,
            tool_results=tool_results
        )

        if autonomous_results:
            tool_results['autonomous_results'] = autonomous_results
            tool_results['tool_results']['autonomous_tasks'] = autonomous_results
            self._save_tool_outputs(tool_results['tool_results'], question, suffix="_improved")

        # Step 2: Extract features with reflection guidance
        print("  Step 2: Extracting features with reflection...")
        extracted_features = self._extract_features_with_reflection(
            tool_results=tool_results,
            input_path=input_path,
            question=question,
            question_template=question_template,
            prediction=prediction or {},
            actor_reflection=actor_reflection,
            original_results=original_results
        )

        # Step 3: Generate improved explanation
        print("  Step 3: Generating improved explanation...")
        prompt_builder = self._get_prompt_builder(question_template, question)

        context = self._build_context(question, model_info, prediction, input_path)
        results_for_prompt = {
            "tool_results": tool_results.get('tool_results', {}),
            "extracted_features": extracted_features,
            "autonomous_results": autonomous_results
        }

        parsed_result = self._generate_explanation_with_reflection(
            prompt_builder=prompt_builder,
            context=context,
            strategy=strategy,
            results=results_for_prompt,
            tool_results=tool_results,
            actor_reflection=actor_reflection,
            original_results=original_results
        )

        # Add metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = question.get('q_type', 'unknown')
        parsed_result['tool_results'] = tool_results.get('tool_results', {})
        parsed_result['autonomous_results'] = autonomous_results
        parsed_result['visualization_paths'] = tool_results.get('visualization_paths', [])
        parsed_result['_improved'] = True

        # Save improved results
        self._save_results(parsed_result, question, suffix="_improved")

        print(f"\nImproved explanation generated")
        return parsed_result

    def _run_with_reflection_multi(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_paths: List[str],
        model_info: Optional[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        actor_reflection: str,
        original_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Multi-instance reflection path for Q9/Q10.

        Same structure as _execute_multi_instance but with reflection-aware
        feature extraction and explanation generation.
        """
        modality = question.get('modality', 'vision')
        num_instances = len(input_paths)
        q_type = question.get('q_type')

        # Pure-reasoning mode: no tools and no autonomous tasks
        if not strategy.get('selected_tools') and not strategy.get('autonomous_tasks'):
            return self._run_pure_reasoning(
                question=question,
                question_template=question_template,
                strategy=strategy,
                input_paths=input_paths,
                model_info=model_info,
                predictions=predictions,
                suffix="_improved"
            )

        # Step 1: Execute XAI tools on each instance (same as _execute_multi_instance)
        all_tool_results = []
        all_viz_paths = []

        image_indices = question.get('image_indices', [])
        # Breast Cancer Q8/Q9/Q10 (spurious): row_idx is an index into the full dataset (569 samples),
        # not the test-split (114 samples). Mirror is_spurious logic from load_breast_cancer.py.
        _dataset = str(question.get('dataset', '')).lower().replace(' ', '_')
        _is_breast_cancer_spurious = (
            'breast_cancer' in _dataset
            and question.get('q_type', 1) in (8, 9, 10)
        )
        split = ('full' if _is_breast_cancer_spurious else question.get('split', 'test'))

        for i, (input_path, prediction) in enumerate(zip(input_paths, predictions or [{}] * num_instances)):
            print(f"  Step 1.{i+1}: Executing XAI tools on Instance {i}...")

            # Reload this instance's sample in data_model_loader so XAI tools
            # operate on the correct data (mirrors _execute_multi_instance logic)
            if self.data_model_loader and i < len(image_indices):
                try:
                    if modality == 'text':
                        features_list = question.get('features', [])
                        if isinstance(features_list, list) and i < len(features_list):
                            feat = features_list[i]
                            if 'premise' in feat and 'hypothesis' in feat:
                                text_input = feat
                            elif 'text' in feat:
                                text_input = feat.get('text', '')
                            elif 'review_text' in feat:
                                text_input = feat.get('review_text', '')
                            else:
                                text_input = feat
                            data = self.data_model_loader.loader_module.load_data(text_input)
                            self.data_model_loader.current_sample_data = data
                        else:
                            print(f"  Warning: No features entry for text instance {i}")
                    else:
                        self.data_model_loader.load_sample(index=image_indices[i], split=split)
                except Exception as e:
                    raise RuntimeError(f"Failed to reload sample {image_indices[i]} for instance {i}: {e}") from e

            instance_question = question.copy()
            instance_question['instance_index'] = i
            instance_question['instance_suffix'] = f"_inst{i}"

            tool_results = self._execute_tools(
                strategy=strategy,
                input_path=input_path,
                prediction=prediction,
                modality=modality,
                question=instance_question,
                save_suffix="_improved"
            )

            all_tool_results.append(tool_results)
            all_viz_paths.extend(tool_results.get('visualization_paths', []))

        # Combine tool results
        combined_tool_results = {
            'instances': all_tool_results,
            'tool_results': {f'instance_{i}': tr.get('tool_results', {}) for i, tr in enumerate(all_tool_results)},
            'visualization_paths': all_viz_paths,
            'tool_results_summary': "; ".join([
                f"Instance {i}: {tr.get('tool_results_summary', 'no results')}"
                for i, tr in enumerate(all_tool_results)
            ])
        }

        # Step 1.5: Execute autonomous tasks (if any)
        print("  Step 1.5: Executing autonomous tasks...")
        autonomous_results = self._execute_autonomous_tasks(
            strategy=strategy,
            input_path=input_paths[0] if input_paths else "",
            prediction=predictions[0] if predictions else {},
            question=question,
            tool_results=combined_tool_results
        )

        if autonomous_results:
            combined_tool_results['autonomous_results'] = autonomous_results
            combined_tool_results['tool_results']['autonomous_tasks'] = autonomous_results

        # Step 2: Extract features with reflection guidance
        # Use _extract_features_multi but inject reflection context
        print("  Step 2: Extracting comparative features with reflection...")
        extracted_features = self._extract_features_multi(
            tool_results=combined_tool_results,
            input_paths=input_paths,
            question=question,
            question_template=question_template,
            predictions=predictions
        )

        # Step 3: Generate improved explanation with reflection
        print("  Step 3: Generating improved comparative explanation with reflection...")
        prompt_builder = self._get_prompt_builder(question_template, question)

        context = self._build_context_multi(question, model_info, predictions, input_paths)
        results_for_prompt = {
            "tool_results": combined_tool_results['tool_results'],
            "extracted_features": extracted_features,
            "autonomous_results": autonomous_results,
            "instances": [{'prediction': p, 'path': path} for p, path in zip(predictions, input_paths)]
        }

        parsed_result = self._generate_explanation_with_reflection(
            prompt_builder=prompt_builder,
            context=context,
            strategy=strategy,
            results=results_for_prompt,
            tool_results=combined_tool_results,
            actor_reflection=actor_reflection,
            original_results=original_results
        )

        # Ensure multi-instance output format
        if 'output' not in parsed_result:
            parsed_result['output'] = {}

        for i in range(num_instances):
            key = f'input_{chr(ord("A") + i)}'
            if key not in parsed_result['output']:
                parsed_result['output'][key] = extracted_features.get(f'output_{i}', {})

        # Add metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = q_type
        parsed_result['is_multi_instance'] = True
        parsed_result['num_instances'] = num_instances
        parsed_result['tool_results'] = combined_tool_results['tool_results']
        parsed_result['autonomous_results'] = autonomous_results
        parsed_result['visualization_paths'] = all_viz_paths
        parsed_result['_improved'] = True

        # Save improved results
        self._save_results(parsed_result, question, suffix="_improved")

        print(f"\nImproved multi-instance explanation generated for {num_instances} instances")
        return parsed_result

    def _extract_features_with_reflection(
        self,
        tool_results: Dict[str, Any],
        input_path: Optional[str],
        question: Dict[str, Any],
        question_template: Any,
        prediction: Dict[str, Any],
        actor_reflection: str,
        original_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract features using VLM with reflection guidance.

        Args:
            tool_results: Results from XAI tools
            input_path: Path to input
            question: Question dict
            question_template: QuestionTemplate
            prediction: Model prediction
            actor_reflection: Critic's feedback
            original_results: Previous results

        Returns:
            Extracted features dict
        """
        modality = question.get('modality', 'vision')
        q_type = question.get('q_type', 1)

        # Get image size for validation
        image_size = None
        if modality == "vision":
            for tool_name, tool_result in tool_results.get('tool_results', {}).items():
                if isinstance(tool_result, dict) and tool_result.get('success'):
                    img_size = tool_result.get('original_image_size', {})
                    if img_size:
                        image_size = (img_size.get('width', 224), img_size.get('height', 224))
                        break

        # Build reflection-aware feature extraction prompt
        prompt = self._build_feature_extraction_prompt_with_reflection(
            tool_results=tool_results,
            question=question,
            prediction=prediction,
            modality=modality,
            q_type=q_type,
            actor_reflection=actor_reflection,
            original_results=original_results
        )

        # Collect images for VLM (only actual image files, not HTML/text artifacts)
        IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
        images = []
        if modality == "vision" and input_path and os.path.exists(input_path):
            images.append(input_path)
        for viz_info in tool_results.get('visualization_paths', []):
            viz_path = viz_info.get('path', '') if isinstance(viz_info, dict) else str(viz_info)
            if viz_path and os.path.exists(viz_path):
                ext = os.path.splitext(viz_path)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    images.append(viz_path)

        # Call VLM and parse, with retry on parse failure
        last_error = None
        for attempt in range(1, 4):
            response = self.invoke_vlm(prompt, images if images else None)
            try:
                return self._parse_feature_response(
                    response, modality, q_type,
                    tool_results=tool_results,
                    image_size=image_size
                )
            except RuntimeError as e:
                last_error = e
                if attempt < 3:
                    print(f"  Feature extraction parse failed (attempt {attempt}/3), retrying in 2s...")
                    time.sleep(2)
        raise last_error

    def _build_feature_extraction_prompt_with_reflection(
        self,
        tool_results: Dict[str, Any],
        question: Dict[str, Any],
        prediction: Dict[str, Any],
        modality: str,
        q_type: int,
        actor_reflection: str,
        original_results: Dict[str, Any]
    ) -> str:
        """
        Build feature extraction prompt incorporating critic's feedback.

        Args:
            tool_results: XAI tool results
            question: Question dict
            prediction: Model prediction
            modality: Data modality
            q_type: Question type
            actor_reflection: Critic's feedback
            original_results: Previous results

        Returns:
            Prompt string
        """
        pred_class = prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))
        confidence = prediction.get('confidence', 0.0)

        # Get image dimensions for vision
        image_width, image_height = 224, 224
        if modality == "vision":
            for tool_name, tool_result in tool_results.get('tool_results', {}).items():
                if isinstance(tool_result, dict) and tool_result.get('success'):
                    img_size = tool_result.get('original_image_size', {})
                    if img_size:
                        image_width = img_size.get('width', image_width)
                        image_height = img_size.get('height', image_height)
                        break

        # Get output format
        if modality == "vision":
            output_format = '"bounding_box": [x_min, y_min, x_max, y_max]'
            size_constraint = f"""
**IMAGE BOUNDS:** {image_width} x {image_height} pixels
All coordinates must be within: x in [0, {image_width}], y in [0, {image_height}]"""
        elif modality == "text":
            output_format = '"start_index": int, "end_index": int'
            text_input = question.get('text_input', '')
            size_constraint = f"**TEXT LENGTH:** {len(text_input)} characters"
        else:
            output_format = '"feature_key": "string"'
            size_constraint = ""

        # Get original output for reference
        original_output = original_results.get('output', {})

        prompt = f"""You are an expert XAI analyst. Your previous analysis had issues. Improve it based on feedback.

## Context
- Question: {question.get('question', '')}
- Question Type: Q{q_type}
- Modality: {modality}
- Model Prediction: {pred_class} (confidence: {confidence:.2%})
{size_constraint}

## Previous Output (had issues)
{original_output}

## Critic's Feedback on Your Previous Explanation
{actor_reflection}

## Current Tool Results Summary
{tool_results.get('tool_results_summary', 'No tools executed')}

## Detailed Tool Statistics
{self._format_tool_statistics(tool_results)}

## Your Task
Based on the critic's feedback, provide an IMPROVED identification of the key region/feature.
Pay special attention to:
1. Region accuracy - use tool statistics to guide your selection
2. Explanation clarity - be specific about why this region matters
3. Any specific issues mentioned in the feedback

**Output valid JSON:**
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why, addressing critic's feedback",
    "confidence": 0.0-1.0
}}

JSON Response:"""

        return prompt

    def _generate_explanation_with_reflection(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        tool_results: Dict[str, Any],
        actor_reflection: str,
        original_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate improved explanation based on Critic's reflection feedback.

        Builds an independent reflection prompt (not appended to base prompt)
        that focuses on the feedback and improvement guidance.

        Args:
            prompt_builder: PromptBuilder instance
            context: Context dict
            strategy: Strategy dict
            results: Current results (includes extracted_features from Step 2)
            tool_results: Tool execution results
            actor_reflection: Critic's feedback (JSON string or dict)
            original_results: Previous results

        Returns:
            Parsed result dict
        """
        from prompts.output_schemas import get_output_schema, schema_to_prompt_string

        prediction = context.get('prediction', {})
        pred_class = prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))
        confidence = prediction.get('confidence', 0.0)
        modality = prompt_builder.modality if hasattr(prompt_builder, 'modality') else 'vision'
        q_type = prompt_builder.q_type if hasattr(prompt_builder, 'q_type') else 1

        # Get Q-type-specific output schema
        output_schema = get_output_schema(q_type, modality)
        output_schema_str = schema_to_prompt_string(output_schema)

        # Build size constraint for coordinate validation
        size_constraint = ""
        if modality == "vision":
            image_width, image_height = 224, 224
            for tool_name, tool_result in tool_results.get('tool_results', {}).items():
                if isinstance(tool_result, dict) and tool_result.get('success'):
                    img_size = tool_result.get('original_image_size', {})
                    if img_size:
                        image_width = img_size.get('width', image_width)
                        image_height = img_size.get('height', image_height)
                        break
            size_constraint = f"""
**IMAGE BOUNDS:** {image_width} x {image_height} pixels
All bounding_box coordinates must be within: x in [0, {image_width}], y in [0, {image_height}]"""
        elif modality == "text":
            text_input = context.get('text_input', '')
            if text_input:
                size_constraint = f"**TEXT LENGTH:** {len(text_input)} characters. Indices must be within [0, {len(text_input)}]."

        # Get previous output and explanation
        original_output = original_results.get('output', {})
        original_explanation = original_results.get('explanation', 'N/A')

        # Get extracted features from Step 2 reflection
        extracted_features = results.get('extracted_features', {})

        prompt = f"""You are an expert XAI analyst. Your previous explanation had issues. Generate an IMPROVED explanation based on the critic's feedback.

## Question
{context.get('user_question', '')}

## Context
- Question Type: Q{q_type}
- Modality: {modality}
- Model Prediction: {pred_class} (confidence: {confidence:.2%})
{size_constraint}

## Previous Output (had issues)
{original_output}

## Previous Explanation (had issues)
{original_explanation}

## Critic's Feedback
{actor_reflection}

## Updated Extracted Features (from re-analysis)
{extracted_features}

## Current Tool Results Summary
{tool_results.get('tool_results_summary', 'No tools executed')}

## Detailed Tool Statistics
{self._format_tool_statistics(tool_results)}

## Your Task
Based on the critic's feedback and updated features, generate an IMPROVED explanation.
Pay special attention to:
1. Address EVERY specific issue mentioned in the critic's feedback
2. Use the updated extracted features to guide your region/feature selection
3. Ensure region accuracy matches tool statistics
4. Provide clear reasoning for why this region/feature matters

**CRITICAL: Your output MUST follow this EXACT JSON schema for Q{q_type} ({modality}):**
{output_schema_str}

JSON Response:"""

        parsed = self.invoke_vlm_for_json(prompt)

        parsed['tool_results'] = tool_results.get('tool_results', {})
        parsed['visualization_paths'] = tool_results.get('visualization_paths', [])
        return parsed
