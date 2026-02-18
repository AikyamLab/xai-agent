"""
Tool Attribution Evaluator for Strategy Faithfulness

Computes importance scores for each XAI tool by measuring how much
the explanation faithfulness changes when tools are included vs excluded.

Method:
- For N tools, evaluate all 2^N configurations (tool combinations)
- Tool i importance = avg(faithfulness with tool i) - avg(faithfulness without tool i)
- Each tool appears in exactly 2^(N-1) configurations
- Caches intermediate results to avoid redundant computations
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import random
import time

from .cache_manager import CacheManager


@dataclass
class ToolImportanceResult:
    """Result for a single tool's importance evaluation"""
    tool_name: str
    importance_score: float  # Average faithfulness difference
    evaluated_configs: List[List[int]]  # Configs used to compute this score
    config_faithfulness_diffs: List[float]  # Faithfulness diff for each config
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyFaithfulnessResult:
    """Complete strategy faithfulness evaluation result"""
    # Basic info
    question_id: str
    modality: str
    timestamp: str
    
    # Original faithfulness
    original_explanation_faithfulness: float
    faithfulness_threshold: float
    faithfulness_passed: bool
    
    # Tool importance results
    tool_importance_scores: Dict[str, float]  # {tool_name: importance}
    tool_importance_details: List[ToolImportanceResult]
    
    # All evaluated configurations with their results
    evaluated_configs: Dict[str, Dict[str, Any]]  # {config_str: result}
    
    # Statistics
    total_configs_evaluated: int
    configs_from_cache: int
    evaluation_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "modality": self.modality,
            "timestamp": self.timestamp,
            "original_explanation_faithfulness": self.original_explanation_faithfulness,
            "faithfulness_threshold": self.faithfulness_threshold,
            "faithfulness_passed": self.faithfulness_passed,
            "tool_importance_scores": self.tool_importance_scores,
            "tool_importance_details": [t.to_dict() for t in self.tool_importance_details],
            "evaluated_configs": self.evaluated_configs,
            "total_configs_evaluated": self.total_configs_evaluated,
            "configs_from_cache": self.configs_from_cache,
            "evaluation_time_seconds": self.evaluation_time_seconds
        }


class ToolAttributionEvaluator:
    """
    Evaluates strategy faithfulness using tool attribution.
    
    For each tool, computes its importance by measuring the faithfulness
    drop when that tool (and subsequent tools) are removed from the strategy.
    """
    
    def __init__(
        self,
        cache_dir: str,
        output_dir: str
    ):
        """
        Initialize the evaluator.
        
        Args:
            cache_dir: Directory for caching intermediate results
            output_dir: Directory for saving final results
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_manager = CacheManager(str(self.cache_dir))
        
        # Will be set before evaluation
        self.actor = None
        self.critic = None
    
    def set_agents(self, actor, critic):
        """Set the actor and critic agents for evaluation"""
        self.actor = actor
        self.critic = critic
    
    def compute_tool_importance(
        self,
        original_strategy: Dict[str, Any],
        original_faithfulness: float,
        original_tool_results: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: str,
        model_info: Dict[str, Any],
        prediction: Dict[str, Any],
        input_tensor: Any,
        faithfulness_threshold: float = 0.1,
        processor: Any = None,
        device: str = "cuda",
        max_samples: Optional[int] = None,
        # Multi-instance parameters (for Q4, Q9, Q10)
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        input_tensors: Optional[List[Any]] = None,
        ground_truths: Optional[List[Any]] = None
    ) -> StrategyFaithfulnessResult:
        """
        Compute importance scores for each tool in the strategy.

        Method: Evaluate all 2^N tool configurations, then compute each tool's
        importance as: avg(faithfulness with tool) - avg(faithfulness without tool)

        OPTIMIZATION: Reuses existing tool execution results. Only re-runs
        feature extraction and explanation generation with filtered tool results.

        Args:
            original_strategy: The strategy from Proposer
            original_faithfulness: Original explanation faithfulness score
            original_tool_results: Already executed tool results from Actor
            question: Question dictionary
            question_template: QuestionTemplate instance
            input_path: Path to input data
            model_info: Model information
            prediction: Model prediction
            input_tensor: Original input tensor for evaluation
            faithfulness_threshold: Threshold for passing
            processor: Data processor
            device: Device for computation

        Returns:
            StrategyFaithfulnessResult with all tool importance scores
        """
        start_time = time.time()

        # Get XAI tools from selected_tools
        xai_tools = original_strategy.get('selected_tools', [])

        # Get autonomous tasks and convert them to tool format
        autonomous_tasks = original_strategy.get('autonomous_tasks', [])
        autonomous_tool_entries = []
        for task in autonomous_tasks:
            task_type = task.get('task_type', 'unknown')
            autonomous_tool_entries.append({
                'tool_name': f'autonomous_{task_type}',
                'is_autonomous': True,
                'task_type': task_type,
                'priority': 100 + len(autonomous_tool_entries),  # Lower priority than XAI tools
                'reasoning': task.get('query', '')
            })

        # Combine XAI tools and autonomous tasks
        tools = xai_tools + autonomous_tool_entries
        n_tools = len(tools)
        question_id = question.get('question_id', 'unknown')
        row_no = question.get('row_no', question_id)
        modality = question.get('modality', 'vision')
        dataset_base_name = question.get('dataset_base_name', 'unknown')

        # Update cache directory based on question info
        # Format: /strategy_faithfulness_cache/{modality}/{dataset_name}/{q_type}/
        import re
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        # Update cache directory for this question
        question_cache_dir = self.cache_dir / modality / dataset_name / q_type_str
        question_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = CacheManager(str(question_cache_dir))

        total_possible = 2 ** n_tools
        print(f"\n  Computing tool importance for {n_tools} tools...")
        print(f"  Total possible configurations: 2^{n_tools} = {total_possible}")
        print(f"  (Reusing existing tool results - only re-running feature extraction)")

        if n_tools == 0:
            return self._create_empty_result(
                question_id, modality, original_faithfulness,
                faithfulness_threshold, time.time() - start_time
            )

        # Generate configurations: full enumeration or random sampling
        if max_samples is not None and max_samples < total_possible:
            all_configs = self._sample_configs(n_tools, max_samples)
            print(f"  Sampled {len(all_configs)} configs from {total_possible} possible")
        else:
            all_configs = self._generate_all_configs(n_tools)
            print(f"  Generated {len(all_configs)} configurations (full enumeration)")

        # Track all evaluated configs and their faithfulness scores
        all_evaluated_configs: Dict[str, Dict] = {}
        config_faithfulness: Dict[tuple, float] = {}
        configs_from_cache = 0

        # Evaluate all configurations
        for config in all_configs:
            config_tuple = tuple(config)
            config_str = str(config_tuple)

            # Check cache
            cached = self.cache_manager.get(config_tuple, question_id)

            if cached is not None:
                faithfulness_score = cached.get('faithfulness_score', 0.0)
                if faithfulness_score is None:
                    faithfulness_score = 0.0
                configs_from_cache += 1
                print(f"    Config {config}: cached (faith={faithfulness_score:.4f})")
            else:
                # Evaluate this config by filtering tool results (not re-executing)
                result = self._evaluate_config_with_filtered_results(
                    config=config,
                    original_strategy=original_strategy,
                    original_tool_results=original_tool_results,
                    question=question,
                    question_template=question_template,
                    input_path=input_path,
                    model_info=model_info,
                    prediction=prediction,
                    input_tensor=input_tensor,
                    processor=processor,
                    device=device,
                    input_paths=input_paths,
                    predictions=predictions,
                    input_tensors=input_tensors,
                    ground_truths=ground_truths
                )

                faithfulness_score = result.get('faithfulness_score', 0.0)
                if faithfulness_score is None:
                    faithfulness_score = 0.0

                # Cache the result
                self.cache_manager.set(config_tuple, question_id, result)

                print(f"    Config {config}: computed (faith={faithfulness_score:.4f})")

                cached = result

            # Store results
            all_evaluated_configs[config_str] = cached
            config_faithfulness[config_tuple] = faithfulness_score

        # Compute importance for each tool using Shapley-value-like method
        # Tool importance = avg(faithfulness with tool) - avg(faithfulness without tool)
        tool_importance_results: List[ToolImportanceResult] = []

        for i, tool in enumerate(tools):
            tool_name = tool.get('tool_name', f'tool_{i}')

            # Get configs where tool i is included (config[i] == 1)
            configs_with_tool = [c for c in all_configs if c[i] == 1]
            # Get configs where tool i is excluded (config[i] == 0)
            configs_without_tool = [c for c in all_configs if c[i] == 0]

            # Calculate average faithfulness for each set
            faith_with = [config_faithfulness[tuple(c)] for c in configs_with_tool]
            faith_without = [config_faithfulness[tuple(c)] for c in configs_without_tool]

            avg_with = sum(faith_with) / len(faith_with) if faith_with else 0.0
            avg_without = sum(faith_without) / len(faith_without) if faith_without else 0.0

            # Importance = how much better is faithfulness when tool is included
            importance = avg_with - avg_without

            # Store detailed diffs for each pair
            config_diffs = []
            evaluated_configs_for_tool = []
            for c_with in configs_with_tool:
                # Find corresponding config without tool (same except position i)
                c_without = c_with.copy()
                c_without[i] = 0
                diff = config_faithfulness[tuple(c_with)] - config_faithfulness[tuple(c_without)]
                config_diffs.append(diff)
                evaluated_configs_for_tool.append(c_with)

            tool_importance_results.append(ToolImportanceResult(
                tool_name=tool_name,
                importance_score=importance,
                evaluated_configs=evaluated_configs_for_tool,
                config_faithfulness_diffs=config_diffs
            ))

            print(f"    Tool '{tool_name}': importance={importance:.4f} "
                  f"(avg_with={avg_with:.4f}, avg_without={avg_without:.4f})")

        # Build final result
        tool_importance_scores = {
            r.tool_name: r.importance_score
            for r in tool_importance_results
        }

        elapsed_time = time.time() - start_time

        result = StrategyFaithfulnessResult(
            question_id=question_id,
            modality=modality,
            timestamp=datetime.now().isoformat(),
            original_explanation_faithfulness=original_faithfulness,
            faithfulness_threshold=faithfulness_threshold,
            faithfulness_passed=original_faithfulness >= faithfulness_threshold,
            tool_importance_scores=tool_importance_scores,
            tool_importance_details=tool_importance_results,
            evaluated_configs=all_evaluated_configs,
            total_configs_evaluated=len(all_evaluated_configs),
            configs_from_cache=configs_from_cache,
            evaluation_time_seconds=elapsed_time
        )

        print(f"\n  Strategy faithfulness evaluation complete:")
        if max_samples is not None and max_samples < total_possible:
            print(f"    Sampled configs: {result.total_configs_evaluated}/{total_possible}")
        else:
            print(f"    Total configs evaluated: {result.total_configs_evaluated} (2^{n_tools})")
        print(f"    From cache: {configs_from_cache}")
        print(f"    Time: {elapsed_time:.2f}s")

        return result

    def _generate_all_configs(self, n_tools: int) -> List[List[int]]:
        """
        Generate all 2^n tool configurations.

        Args:
            n_tools: Number of tools

        Returns:
            List of all possible config masks (1=include tool, 0=exclude)
            Example for n=2: [[0,0], [0,1], [1,0], [1,1]]
        """
        if n_tools == 0:
            return [[]]

        configs = []
        for i in range(2 ** n_tools):
            config = []
            for j in range(n_tools):
                # Extract bit j from i
                config.append((i >> j) & 1)
            configs.append(config)

        return configs

    def _sample_configs(self, n_tools: int, max_samples: int) -> List[List[int]]:
        """
        Randomly sample tool configurations instead of full 2^n enumeration.

        Always includes all-zeros (no tools baseline) and all-ones (all tools)
        as anchors, then randomly samples the rest.

        Args:
            n_tools: Number of tools
            max_samples: Total number of configs to return (including anchors)

        Returns:
            List of sampled config masks
        """
        if n_tools == 0:
            return [[]]

        all_zeros = [0] * n_tools
        all_ones = [1] * n_tools
        configs = [all_zeros, all_ones]

        if max_samples <= 2:
            return configs

        # Sample from remaining configs (exclude index 0=all-zeros and 2^n-1=all-ones)
        remaining_indices = list(range(1, 2 ** n_tools - 1))
        sample_count = min(max_samples - 2, len(remaining_indices))
        sampled_indices = random.sample(remaining_indices, sample_count)

        for idx in sampled_indices:
            config = [(idx >> j) & 1 for j in range(n_tools)]
            configs.append(config)

        return configs

    def _generate_configs_for_tool(
        self,
        tool_idx: int,
        n_tools: int
    ) -> List[List[int]]:
        """
        Generate configurations where tool at tool_idx is excluded.

        This is now a helper method - main computation uses _generate_all_configs.

        Args:
            tool_idx: Index of the tool to evaluate
            n_tools: Total number of tools

        Returns:
            List of config masks where config[tool_idx] = 0
        """
        all_configs = self._generate_all_configs(n_tools)
        return [c for c in all_configs if c[tool_idx] == 0]
    
    def _evaluate_config_with_filtered_results(
        self,
        config: List[int],
        original_strategy: Dict[str, Any],
        original_tool_results: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: str,
        model_info: Dict[str, Any],
        prediction: Dict[str, Any],
        input_tensor: Any,
        processor: Any = None,
        device: str = "cuda",
        # Multi-instance parameters
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        input_tensors: Optional[List[Any]] = None,
        ground_truths: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a specific tool configuration by filtering existing tool results.

        OPTIMIZATION: Does NOT re-execute XAI tools. Instead, filters the
        existing tool_results based on config mask, then only re-runs
        feature extraction and explanation generation via Actor's internal methods.

        Supports both single-instance and multi-instance (Q4, Q9, Q10) questions.

        Args:
            config: Tool mask (1=include tool result, 0=exclude)
            original_strategy: Original strategy
            original_tool_results: Already executed tool results
            question: Question dict
            question_template: QuestionTemplate instance
            input_path: Path to input (first instance for multi-instance)
            model_info: Model info
            prediction: Model prediction (first instance for multi-instance)
            input_tensor: Input tensor for evaluation (first instance for multi-instance)
            processor: Data processor
            device: Device for computation
            input_paths: List of paths for multi-instance
            predictions: List of predictions for multi-instance
            input_tensors: List of input tensors for multi-instance
            ground_truths: List of ground truths for multi-instance

        Returns:
            Dict with filtered_tool_results, explanation, and faithfulness_score
        """
        if self.actor is None or self.critic is None:
            raise RuntimeError("Actor and Critic must be set before evaluation")

        is_multi = question.get('is_multi_instance', False) and input_paths and predictions

        # Get XAI tools from strategy
        xai_tools = original_strategy.get('selected_tools', [])
        xai_tool_names = [t.get('tool_name', f'tool_{i}') for i, t in enumerate(xai_tools)]

        # Get autonomous tasks and create tool names for them
        autonomous_tasks = original_strategy.get('autonomous_tasks', [])
        autonomous_tool_names = [f'autonomous_{task.get("task_type", "unknown")}' for task in autonomous_tasks]

        # Combined tool names (XAI tools first, then autonomous tasks)
        tool_names = xai_tool_names + autonomous_tool_names

        # Filter tool results based on config mask
        # config[i] = 1 means include, config[i] = 0 means exclude
        filtered_tool_results = {}
        filtered_autonomous_results = {}
        included_tools = []

        tool_results_dict = original_tool_results.get('tool_results', {})

        # For multi-instance, tool_results are nested: {instance_0: {gradcam: ...}, ...}
        # Flatten by merging from the first instance for tool name lookup
        if is_multi:
            flat_tool_results = {}
            for inst_key, inst_results in tool_results_dict.items():
                if isinstance(inst_results, dict):
                    for tool_name, tool_result in inst_results.items():
                        if tool_name not in flat_tool_results:
                            flat_tool_results[tool_name] = tool_result
            autonomous_results_dict = flat_tool_results.get('autonomous_tasks', {})
        else:
            flat_tool_results = tool_results_dict
            autonomous_results_dict = tool_results_dict.get('autonomous_tasks', {})

        for i, tool_name in enumerate(tool_names):
            if i < len(config) and config[i] == 1:
                if tool_name.startswith('autonomous_'):
                    # This is an autonomous task
                    task_type = tool_name.replace('autonomous_', '')
                    if task_type in autonomous_results_dict:
                        filtered_autonomous_results[task_type] = autonomous_results_dict[task_type]
                        included_tools.append(tool_name)
                else:
                    # Regular XAI tool
                    if tool_name in flat_tool_results:
                        filtered_tool_results[tool_name] = flat_tool_results[tool_name]
                        included_tools.append(tool_name)

        # Add autonomous_tasks to filtered_tool_results if any are included
        if filtered_autonomous_results:
            filtered_tool_results['autonomous_tasks'] = filtered_autonomous_results

        # Create a name for the tool configuration being evaluated
        tool_config_name = "-".join(included_tools) if included_tools else "no_tools"

        # Handle empty tools case - use VLM direct vision analysis
        if not included_tools:
            return self._evaluate_config_no_tools(
                config=config,
                question=question,
                question_template=question_template,
                input_path=input_path,
                model_info=model_info,
                prediction=prediction,
                input_tensor=input_tensor,
                processor=processor,
                device=device,
                input_paths=input_paths,
                predictions=predictions,
                input_tensors=input_tensors,
                ground_truths=ground_truths
            )

        # For multi-instance, also build per-instance filtered tool results
        if is_multi:
            filtered_multi_tool_results = {}
            for inst_key, inst_results in tool_results_dict.items():
                if isinstance(inst_results, dict):
                    filtered_inst = {}
                    for tool_name in included_tools:
                        if not tool_name.startswith('autonomous_') and tool_name in inst_results:
                            filtered_inst[tool_name] = inst_results[tool_name]
                    if filtered_autonomous_results:
                        filtered_inst['autonomous_tasks'] = filtered_autonomous_results
                    filtered_multi_tool_results[inst_key] = filtered_inst

        # Build filtered tool_results structure (same format as _execute_tools output)
        filtered_results_structure = {
            'tool_results': filtered_multi_tool_results if is_multi else filtered_tool_results,
            'visualization_paths': [
                v for v in original_tool_results.get('visualization_paths', [])
                if isinstance(v, dict) and v.get('tool') in included_tools
            ],
            'tool_results_summary': "; ".join([f"{t}: success" for t in included_tools])
        }

        if is_multi:
            # Multi-instance path: use actor's multi-instance methods
            extracted_features = self.actor._extract_features_multi(
                tool_results=filtered_results_structure,
                input_paths=input_paths,
                question=question,
                question_template=question_template,
                predictions=predictions
            )

            prompt_builder = self.actor._get_prompt_builder(question_template, question)
            context = self.actor._build_context_multi(question, model_info, predictions, input_paths)

            results_for_prompt = {
                "tool_results": filtered_results_structure['tool_results'],
                "extracted_features": extracted_features,
                "instances": [{'prediction': p, 'path': path} for p, path in zip(predictions, input_paths)]
            }

            if hasattr(prompt_builder, 'build_actor_prompt_multi'):
                parsed_result = self.actor._generate_explanation_multi(
                    prompt_builder=prompt_builder,
                    context=context,
                    strategy=original_strategy,
                    results=results_for_prompt,
                    tool_results=filtered_results_structure,
                    instances=results_for_prompt['instances']
                )
            else:
                parsed_result = self.actor._generate_explanation_with_prompt_builder(
                    prompt_builder=prompt_builder,
                    context=context,
                    strategy=original_strategy,
                    results=results_for_prompt,
                    tool_results=filtered_results_structure
                )
        else:
            # Single-instance path (original logic)
            extracted_features = self.actor._extract_features_via_vlm(
                tool_results=filtered_results_structure,
                input_path=input_path,
                question=question,
                question_template=question_template,
                prediction=prediction or {}
            )

            prompt_builder = self.actor._get_prompt_builder(question_template, question)
            context = self.actor._build_context(question, model_info, prediction, input_path)

            results_for_prompt = {
                "tool_results": filtered_tool_results,
                "extracted_features": extracted_features,
                "autonomous_results": filtered_autonomous_results
            }

            parsed_result = self.actor._generate_explanation_with_prompt_builder(
                prompt_builder=prompt_builder,
                context=context,
                strategy=original_strategy,
                results=results_for_prompt,
                tool_results=filtered_results_structure
            )

        # Add metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = question.get('q_type', 'unknown')
        parsed_result['tool_results'] = filtered_tool_results
        parsed_result['visualization_paths'] = filtered_results_structure.get('visualization_paths', [])

        # Save result to results/ directory (parallel to actor.run() behaviour)
        self.actor._save_results(parsed_result, question, suffix=f"_{tool_config_name}")

        # Run Critic to evaluate faithfulness
        critic_kwargs = {
            'results': parsed_result,
            'question': question,
            'original_input': input_tensor,
            'original_prediction': prediction,
            'processor': processor,
            'device': device,
            'tool_name': tool_config_name,
            # Required for tabular masking (feature_key lookup) and class resolution
            'feature_names': model_info.get('feature_names', []) if model_info else [],
            'class_names': model_info.get('label_map', model_info.get('class_names', {})) if model_info else {},
        }
        if is_multi:
            critic_kwargs['inputs'] = input_tensors
            critic_kwargs['predictions'] = predictions
            critic_kwargs['ground_truths'] = ground_truths

        evaluation = self.critic.run(**critic_kwargs)

        faithfulness_score = evaluation.get('faithfulness', {}).get('score', 0.0)
        if faithfulness_score is None:
            faithfulness_score = 0.0

        return {
            "config": config,
            "included_tools": included_tools,
            "explanation": parsed_result,
            "evaluation": evaluation,
            "faithfulness_score": faithfulness_score
        }

    def _evaluate_config_no_tools(
        self,
        config: List[int],
        question: Dict[str, Any],
        question_template: Any,
        input_path: str,
        model_info: Dict[str, Any],
        prediction: Dict[str, Any],
        input_tensor: Any,
        processor: Any = None,
        device: str = "cuda",
        # Multi-instance parameters
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        input_tensors: Optional[List[Any]] = None,
        ground_truths: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the all-zeros config by having VLM analyze images directly
        WITHOUT any XAI tool results. This provides a true baseline.

        Args:
            config: Tool mask (should be all zeros)
            question: Question dict
            question_template: QuestionTemplate instance
            input_path: Path to input data
            model_info: Model info
            prediction: Model prediction
            input_tensor: Input tensor/image for evaluation
            processor: Data processor
            device: Device for computation
            input_paths: List of paths for multi-instance
            predictions: List of predictions for multi-instance
            input_tensors: List of input tensors for multi-instance
            ground_truths: List of ground truths for multi-instance

        Returns:
            Dict with explanation and faithfulness_score from VLM direct analysis
        """
        if self.actor is None or self.critic is None:
            raise RuntimeError("Actor and Critic must be set before evaluation")

        modality = question.get('modality', 'vision')
        q_type = question.get('q_type', 1)

        print(f"      Evaluating no-tools config: VLM direct {modality} analysis...")

        # Determine whether this is a multi-instance question (Q4, Q9, Q10)
        is_multi = (question.get('is_multi_instance', False)
                    and input_tensors and len(input_tensors) > 1)

        # Get actual image size (use first instance as reference)
        image_size = self._get_image_size(input_tensor, input_path, modality)

        # Build prompt for VLM to analyze input directly (no tool results)
        prompt = self._build_no_tools_prompt(
            question=question,
            prediction=prediction,
            modality=modality,
            q_type=q_type,
            image_size=image_size,
            input_data=input_tensor,
            # Multi-instance: expose ALL instances so VLM can reason about all of them
            all_input_datas=input_tensors if is_multi else None,
            all_predictions=predictions if is_multi else None
        )

        # Prepare images for VLM (vision only); include all instances for multi-instance Q types
        images = self._prepare_images_for_vlm(
            input_tensor, input_path, modality,
            input_tensors=input_tensors if is_multi else None,
            input_paths=input_paths if is_multi else None
        )

        # Invoke VLM directly
        response = self.actor.invoke_vlm(prompt, images)

        # Prepare parsing parameters based on modality
        text_length = 100
        available_features = None

        if modality == "text":
            def _get_text_len(t) -> int:
                if isinstance(t, str):
                    return len(t)
                if isinstance(t, dict):
                    p = t.get('premise', '')
                    h = t.get('hypothesis', '')
                    return len(p) + len(h) + 1
                return 0
            if is_multi and input_tensors:
                # Use max length across all instances for conservative clipping
                text_length = max((_get_text_len(t) for t in input_tensors), default=100) or 100
            else:
                text_input = question.get('text_input', '')
                if not text_input and isinstance(input_tensor, str):
                    text_input = input_tensor
                elif not text_input and isinstance(input_tensor, dict):
                    premise = input_tensor.get('premise', '')
                    hypothesis = input_tensor.get('hypothesis', '')
                    text_input = f"{premise} {hypothesis}"
                text_length = len(text_input) if text_input else 100

        elif modality == "tabular":
            features = question.get('features', {})
            if not features and isinstance(input_tensor, dict):
                features = input_tensor
            available_features = list(features.keys()) if features else None

        # Parse response
        parsed_result = self._parse_no_tools_response(
            response=response,
            modality=modality,
            q_type=q_type,
            image_size=image_size,
            text_length=text_length,
            available_features=available_features
        )

        if not parsed_result:
            raise RuntimeError(f"Failed to parse no-tools VLM response. Response preview: {response[:500]}")

        # Add metadata
        parsed_result['question_id'] = question.get('question_id', 'unknown')
        parsed_result['question_type'] = q_type
        parsed_result['tool_results'] = {}
        parsed_result['visualization_paths'] = []
        parsed_result['no_tools_baseline'] = True

        # Save result to results/ directory (same as actor.run() does)
        self.actor._save_results(parsed_result, question, suffix="_no_tools")

        # Run Critic to evaluate faithfulness
        is_multi = question.get('is_multi_instance', False) and input_paths and predictions
        critic_kwargs = {
            'results': parsed_result,
            'question': question,
            'original_input': input_tensor,
            'original_prediction': prediction,
            'processor': processor,
            'device': device,
            'tool_name': "no_tools",
            # Required for tabular masking (feature_key lookup) and class resolution
            'feature_names': model_info.get('feature_names', []) if model_info else [],
            'class_names': model_info.get('label_map', model_info.get('class_names', {})) if model_info else {},
        }
        if is_multi:
            critic_kwargs['inputs'] = input_tensors
            critic_kwargs['predictions'] = predictions
            critic_kwargs['ground_truths'] = ground_truths

        evaluation = self.critic.run(**critic_kwargs)

        faithfulness_score = evaluation.get('faithfulness', {}).get('score', 0.0)
        if faithfulness_score is None:
            faithfulness_score = 0.0

        print(f"      No-tools baseline faithfulness: {faithfulness_score:.4f}")

        return {
            "config": config,
            "included_tools": [],
            "explanation": parsed_result,
            "evaluation": evaluation,
            "faithfulness_score": faithfulness_score,
            "no_tools_baseline": True
        }

    def _get_image_size(
        self,
        input_tensor: Any,
        input_path: str,
        modality: str
    ) -> Tuple[int, int]:
        """Get the actual image size from input."""
        if modality != "vision":
            return (224, 224)

        from PIL import Image
        import os

        # Try to get size from PIL Image
        if isinstance(input_tensor, Image.Image):
            return input_tensor.size  # (width, height)

        # Try to get size from file
        if input_path and os.path.exists(input_path):
            with Image.open(input_path) as img:
                return img.size

        # Try to get size from torch tensor
        if hasattr(input_tensor, 'shape'):
            shape = input_tensor.shape
            if len(shape) == 4:  # BCHW
                return (shape[3], shape[2])
            elif len(shape) == 3:  # CHW
                return (shape[2], shape[1])

        # Default fallback
        return (224, 224)

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
        covers all Q-type × modality combinations without per-case hardcoding.

        For multi-instance Q types (Q4, Q9, Q10), pass all_input_datas and
        all_predictions so the VLM can reason about every instance.
        """
        from prompts.output_schemas import get_output_schema

        # Determine if we're in multi-instance mode
        is_multi = bool(all_input_datas and len(all_input_datas) > 1)
        instance_labels = [chr(ord('A') + i) for i in range(len(all_input_datas))] if is_multi else []

        pred_class = prediction.get('predicted_class_name',
                                    prediction.get('predicted_class_idx', 'Unknown'))
        confidence = prediction.get('confidence', 0.0)

        # Canonical output schema for this Q-type × modality
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
            # Images are passed separately; note labels if multi-instance
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
                for lbl, data, pred in zip(instance_labels,
                                           all_input_datas,
                                           all_predictions or [{}] * len(all_input_datas)):
                    feats = _extract_features(data) or question.get('features', {})
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
            # Per-instance predictions already shown in input_section
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

        def _tensor_to_pil(t) -> Optional["Image.Image"]:
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

        def _load_single(tensor, path) -> Optional["Image.Image"]:
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
                images.append(img.resize((224, 224)))

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
        import json

        def clip_bounding_box(bbox: List, width: int, height: int) -> List:
            """Clip bounding box coordinates to valid image bounds."""
            if not bbox or len(bbox) != 4:
                return [0, 0, width, height]
            x_min, y_min, x_max, y_max = bbox
            # Clip to valid range
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

        # width/height defined here so both clip_region_dict and validate_and_fix_output
        # can access them as a closure variable
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
            """Validate and fix output based on modality.

            Handles all region-containing structures across Q types:
            - Q1-Q3, Q8: direct bounding_box / start+end / feature_key
            - Q5, Q7: direct fields + nested masked_region
            - Q6: nested change_plan
            - Q4, Q9: nested input_A / input_B / input_C ...
            - Q10: nested correct_instance_features / wrong_instance_features
            """
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

                # Validate structure
                if 'output' in parsed:
                    return validate_and_fix_output(parsed, modality)

                # Try to restructure if output fields are at top level
                if 'change_plan' in parsed:
                    # Q6-style: change_plan surfaced at top level
                    return validate_and_fix_output({
                        "output": {"change_plan": parsed['change_plan']},
                        "explanation": parsed.get('explanation', 'VLM direct analysis'),
                        "confidence": parsed.get('confidence', 0.5)
                    }, modality)
                elif modality == "vision" and 'bounding_box' in parsed:
                    width, height = image_size
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

        print(f"      Warning: Failed to parse no-tools response: {response[:200]}...")
        return None

    def _get_default_no_tools_result(
        self,
        modality: str,
        q_type: int
    ) -> Dict[str, Any]:
        """Return default result when VLM parsing fails."""
        if modality == "vision":
            output = {"bounding_box": [56, 56, 168, 168]}  # Center region
        elif modality == "text":
            output = {"start_index": 0, "end_index": 50}
        else:
            output = {"feature_key": "unknown"}

        return {
            "output": output,
            "explanation": "Default fallback - VLM direct analysis did not produce valid output",
            "confidence": 0.1
        }

    def _create_empty_result(
        self,
        question_id: str,
        modality: str,
        original_faithfulness: float,
        threshold: float,
        elapsed_time: float
    ) -> StrategyFaithfulnessResult:
        """Create empty result when no tools are available"""
        return StrategyFaithfulnessResult(
            question_id=question_id,
            modality=modality,
            timestamp=datetime.now().isoformat(),
            original_explanation_faithfulness=original_faithfulness,
            faithfulness_threshold=threshold,
            faithfulness_passed=original_faithfulness >= threshold,
            tool_importance_scores={},
            tool_importance_details=[],
            evaluated_configs={},
            total_configs_evaluated=0,
            configs_from_cache=0,
            evaluation_time_seconds=elapsed_time
        )
    
    def save_result(
        self,
        result: StrategyFaithfulnessResult,
        question: Dict[str, Any]
    ) -> str:
        """
        Save strategy faithfulness result to file.

        Args:
            result: StrategyFaithfulnessResult to save
            question: Question dict for naming

        Returns:
            Path to saved file
        """
        import re
        modality = question.get('modality', 'vision')
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))

        # Extract dataset_name and q_type from dataset_base_name
        # e.g., "stl10_resnet_q1_test" -> dataset_name="stl10_resnet", q_type="q1"
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)  # e.g., "stl10_resnet"
            q_type_str = match.group(2)    # e.g., "q1"
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        # Directory structure: /strategy_faithfulness/{modality}/{dataset_name}/{q_type}/{question_id}/
        save_dir = self.output_dir / modality / dataset_name / q_type_str / str(row_no)
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = "strategy_faithfulness.json"
        filepath = save_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        print(f"  Strategy faithfulness saved to: {filepath}")
        return str(filepath)
