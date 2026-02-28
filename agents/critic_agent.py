"""
Critic Agent for XAI Agent Framework

Responsible for:
- Evaluating explanation faithfulness
- Computing question-specific metrics
- Validating agent outputs against model behavior

This agent implements the evaluation pipeline for all 10 question types.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base_agent import BaseAgent


class CriticAgent(BaseAgent):
    """
    Critic Agent - Explanation Faithfulness Evaluation

    Evaluates Actor's results by:
    1. Parsing agent outputs to extract identified regions/features
    2. Applying masking or modifications to inputs
    3. Running model on original and modified inputs
    4. Computing faithfulness metrics
    """

    def __init__(
        self,
        vlm: Any,
        model: Optional[nn.Module] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Critic Agent.

        Args:
            vlm: VisionLanguageModel instance
            model: Target model being explained (optional, can be set later)
            output_dir: Output directory
        """
        super().__init__(vlm, output_dir, "CriticAgent")

        self.model = model
        self.eval_dir = self.output_dir / "evaluations"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Evaluators for each question type (lazily loaded)
        self._evaluators: Dict[int, Any] = {}

    def set_model(self, model: nn.Module):
        """Set the target model for evaluation"""
        self.model = model

    def run(
        self,
        results: Dict[str, Any],
        question: Dict[str, Any],
        original_input: Any = None,
        original_prediction: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Any] = None,
        tool_name: Optional[str] = None,  # For strategy faithfulness file naming
        suffix: str = "",  # For _improved file naming
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point - evaluate explanation faithfulness.

        Args:
            results: Results from Actor Agent
            question: Original question dictionary
            original_input: Original input data (image tensor, text, tabular)
            original_prediction: Original model prediction
            ground_truth: Ground truth label (if available)
            tool_name: Name of tool config being evaluated (for file naming)
            suffix: Suffix for output filenames (e.g., "_improved")
            **kwargs: Additional arguments (processor, tokenizer, etc.)

        Returns:
            Evaluation dictionary with metrics
        """
        return self.evaluate(
            results, question, original_input,
            original_prediction, ground_truth, tool_name=tool_name, suffix=suffix, **kwargs
        )

    def evaluate(
        self,
        results: Dict[str, Any],
        question: Dict[str, Any],
        original_input: Any = None,
        original_prediction: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Any] = None,
        tool_name: Optional[str] = None,
        suffix: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate Actor's results for explanation faithfulness.

        Args:
            results: Results from Actor Agent
            question: Original question
            original_input: Original input data
            original_prediction: Original prediction results
            ground_truth: Ground truth labels
            tool_name: Name of tool config being evaluated (for file naming)
            suffix: Suffix for output filenames (e.g., "_improved")
            **kwargs: Additional arguments

        Returns:
            Evaluation dictionary
        """
        print("\n" + "=" * 70)
        print("CRITIC AGENT: Evaluating Results")
        print("=" * 70)

        q_type = question.get('q_type', 1)
        modality = question.get('modality', 'vision')

        evaluation = {
            "question_id": results.get('question_id', 'unknown'),
            "question_type": q_type,
            "modality": modality,
        }

        # Calculate basic quality metrics
        quality_metrics = self._calculate_quality_metrics(results)
        evaluation.update(quality_metrics)

        # Evaluate explanation faithfulness if model and input are available
        if self.model is not None and original_input is not None:
            # Pass question info for masked input naming
            dataset_base_name = question.get('dataset_base_name', 'unknown')
            row_no = question.get('row_no', results.get('question_id', 0))

            # Extract Q5/Q6/Q7-specific kwargs from the question's example field
            q_specific = self._extract_q_specific_kwargs(q_type, question, modality, original_input)
            # Merge into kwargs without overwriting already-provided values
            for k, v in q_specific.items():
                kwargs.setdefault(k, v)

            faithfulness_result = self.evaluate_faithfulness(
                q_type=q_type,
                agent_output=results,
                original_input=original_input,
                original_prediction=original_prediction or {},
                modality=modality,
                ground_truth=ground_truth,
                dataset_base_name=dataset_base_name,
                row_no=row_no,
                tool_name=tool_name,
                mask_suffix=suffix,
                **kwargs
            )
            evaluation["faithfulness"] = faithfulness_result
        else:
            evaluation["faithfulness"] = {
                "score": None,
                "error": "Model or input not provided for faithfulness evaluation"
            }

        # Save evaluation with nested directory structure
        # Format: /evaluations/{modality}/{dataset_name}/{q_type}/{question_id}/evaluation.json
        import re
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', results.get('question_id', 0))

        # Extract dataset_name and q_type from dataset_base_name
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        filename = f"evaluation{suffix}"
        if tool_name:
            filename = f"evaluation_{tool_name}{suffix}"

        subdir = f"evaluations/{modality}/{dataset_name}/{q_type_str}/{row_no}"
        filepath = self.save_json(evaluation, filename, subdir)
        print(f"Evaluation saved to: {filepath}")
        print(f"  Quality Score: {evaluation.get('quality_score', 'N/A')}")
        print(f"  Faithfulness Score: {evaluation.get('faithfulness', {}).get('score', 'N/A')}")

        return evaluation

    def evaluate_faithfulness(
        self,
        q_type: int,
        agent_output: Dict[str, Any],
        original_input: Any,
        original_prediction: Dict[str, Any],
        modality: str = "vision",
        ground_truth: Optional[Any] = None,
        tool_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate explanation faithfulness for a specific question type.

        Args:
            q_type: Question type (1-10)
            agent_output: Parsed output from agent
            original_input: Original input data
            original_prediction: Original prediction
            modality: Data modality
            ground_truth: Ground truth (for Q8-Q10)
            tool_name: Name of tool config being evaluated (for file naming)
            **kwargs: Additional arguments

        Returns:
            Faithfulness evaluation result
        """
        evaluator = self._get_evaluator(q_type, modality)
        if evaluator is None:
            raise RuntimeError(f"No evaluator found for Q{q_type} modality={modality}")

        # Pass tool_name through kwargs
        kwargs['tool_name'] = tool_name

        result = evaluator.evaluate(
            agent_output=agent_output,
            original_input=original_input,
            model=self.model,
            original_prediction=original_prediction,
            ground_truth=ground_truth,
            q_type=q_type,
            **kwargs
        )

        return result.to_dict() if hasattr(result, 'to_dict') else result

    def _get_evaluator(self, q_type: int, modality: str) -> Any:
        """Get or create evaluator for question type"""
        key = (q_type, modality)

        if key not in self._evaluators:
            from evaluation import get_evaluator
            self._evaluators[key] = get_evaluator(q_type, modality)

        return self._evaluators.get(key)

    def _calculate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic quality metrics for execution"""
        tool_results = results.get('tool_results', {})
        output = results.get('output', {})

        # Tool execution quality
        # Handle both regular tools and nested autonomous_tasks
        successful_tools = 0
        total_tools = 0

        for tool_name, r in tool_results.items():
            if not isinstance(r, dict):
                continue

            if tool_name == 'autonomous_tasks':
                # autonomous_tasks is a nested structure with sub-tools like 'grounding', 'comparison'
                for sub_tool_name, sub_result in r.items():
                    if isinstance(sub_result, dict):
                        total_tools += 1
                        if sub_result.get('success'):
                            successful_tools += 1
            else:
                # Regular tool
                total_tools += 1
                if r.get('success'):
                    successful_tools += 1

        quality_score = successful_tools / total_tools if total_tools > 0 else 0.0

        # Output completeness
        has_output = bool(output)
        has_explanation = bool(results.get('explanation'))
        has_confidence = 'confidence' in results

        completeness = sum([has_output, has_explanation, has_confidence]) / 3

        # Overall score
        overall_score = (quality_score + completeness) / 2

        if overall_score >= 0.8:
            rating = "Excellent"
        elif overall_score >= 0.6:
            rating = "Good"
        elif overall_score >= 0.4:
            rating = "Fair"
        else:
            rating = "Needs Improvement"

        return {
            "quality_score": round(quality_score, 2),
            "completeness": round(completeness, 2),
            "overall_score": round(overall_score, 2),
            "overall_rating": rating,
            "execution_metrics": {
                "successful_tools": successful_tools,
                "total_tools": total_tools,
                "has_output": has_output,
                "has_explanation": has_explanation
            }
        }

    # =========================================================================
    # Q4 Specific Methods (instance_A / instance_B format)
    # =========================================================================

    def run_q4(
        self,
        results: Dict[str, Any],
        question: Dict[str, Any],
        inputs: Dict[str, Any],        # {'A': image_A, 'B': image_B}
        predictions: Dict[str, Any],   # {'A': pred_A, 'B': pred_B}
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate Q4 explanation faithfulness.

        Uses Q4Evaluator.evaluate_multi() which expects:
        - inputs: [input_A, input_B]
        - predictions: [pred_A, pred_B]
        - agent_output with output.input_A and output.input_B

        Args:
            results: Results from Actor with output.input_A, output.input_B
            question: Question dict
            inputs: {'A': PIL.Image, 'B': PIL.Image}
            predictions: {'A': pred_dict, 'B': pred_dict}
            **kwargs: processor, device, etc.

        Returns:
            Evaluation dict with gap reduction score
        """
        print("\n" + "=" * 70)
        print("CRITIC AGENT: Evaluating Q4 Results")
        print("=" * 70)

        modality = question.get('modality', 'vision')

        evaluation = {
            "question_id": results.get('question_id', 'unknown'),
            "question_type": 4,
            "modality": modality,
        }

        # Calculate basic quality metrics
        quality_metrics = self._calculate_quality_metrics(results)
        evaluation.update(quality_metrics)

        # Evaluate Q4 faithfulness
        if self.model is not None and inputs.get('A') is not None and inputs.get('B') is not None:
            # Get Q4 evaluator
            evaluator = self._get_evaluator(4, modality)

            if evaluator is None:
                raise RuntimeError(f"Q4 evaluator not found for modality={modality}")

            # Convert to lists for evaluate_multi
            inputs_list = [inputs['A'], inputs['B']]
            predictions_list = [predictions['A'], predictions['B']]

            # Get question naming info
            dataset_base_name = question.get('dataset_base_name', 'unknown')
            row_no = question.get('pair_id', question.get('row_no', question.get('question_id', 0)))

            # Call Q4Evaluator.evaluate_multi
            result = evaluator.evaluate_multi(
                agent_output=results,
                inputs=inputs_list,
                model=self.model,
                predictions=predictions_list,
                processor=kwargs.get('processor'),
                device=kwargs.get('device', 'cuda'),
                dataset_base_name=dataset_base_name,
                row_no=row_no,
                tool_name=kwargs.get('tool_name')
            )

            evaluation["faithfulness"] = result.to_dict() if hasattr(result, 'to_dict') else result

            # Print summary
            faith_score = evaluation["faithfulness"].get("score")
            details = evaluation["faithfulness"].get("details", {})
            gap_orig = details.get('gap_original', 'N/A')
            gap_mod = details.get('gap_modified', 'N/A')
            gap_red = details.get('gap_reduction', 'N/A')

            if isinstance(gap_orig, (int, float)):
                print(f"  Gap Original: {gap_orig:.4f}")
            if isinstance(gap_mod, (int, float)):
                print(f"  Gap Modified: {gap_mod:.4f}")
            if isinstance(gap_red, (int, float)):
                print(f"  Gap Reduction: {gap_red:.4f}")
            print(f"  Score: {faith_score} (1 = gap reduced, 0 = gap increased/same)")
        else:
            evaluation["faithfulness"] = {
                "score": None,
                "error": "Model or inputs not provided"
            }

        # Save evaluation (suffix e.g. "_improved" for second-pass evaluations)
        suffix = kwargs.get('suffix', '')
        import re
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('pair_id', question.get('row_no', question.get('question_id', 0)))

        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = "q4"

        subdir = f"evaluations/{modality}/{dataset_name}/{q_type_str}/{row_no}"
        filepath = self.save_json(evaluation, f"evaluation{suffix}", subdir)
        print(f"Q4 Evaluation saved to: {filepath}")

        return evaluation

    def generate_reflections(
        self,
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        question: Dict[str, Any],
        faithfulness_result: Dict[str, Any],
        tool_importance_scores: Dict[str, float],
        threshold: float = 0.1
    ) -> Tuple[str, str]:
        """
        Generate two separate reflections for Proposer and Actor agents.

        Uses VLM to analyze the strategy and explanation, then produces
        targeted feedback for each agent to improve their outputs.

        Args:
            strategy: Strategy from Proposer Agent
            results: Results from Actor Agent
            question: Original question dictionary
            faithfulness_result: Faithfulness evaluation result
            tool_importance_scores: Dict of {tool_name: importance_score}
            threshold: Faithfulness threshold

        Returns:
            Tuple of (proposer_reflection_json_str, actor_reflection_json_str)
            These are JSON strings that can be directly passed to the agents.
        """
        from prompts.critic_reflection_prompt import (
            build_critic_reflection_prompt,
            parse_dual_reflection,
            format_tool_importance_details
        )

        print("\n  Generating reflections for Proposer and Actor...")

        # Format tool importance details
        tool_importance_details = format_tool_importance_details(tool_importance_scores)

        # Build prompt
        prompt = build_critic_reflection_prompt(
            question=question,
            strategy=strategy,
            explanation=results,
            faithfulness_result=faithfulness_result,
            tool_importance_details=tool_importance_details,
            threshold=threshold
        )

        # Save prompt
        self._save_prompt(prompt, question, "critic_prompt")

        # Call VLM
        response = self.invoke_vlm(prompt)

        # Parse response to extract both reflections
        proposer_reflection, actor_reflection = parse_dual_reflection(response)

        print(f"    Proposer reflection generated ({len(proposer_reflection)} chars)")
        print(f"    Actor reflection generated ({len(actor_reflection)} chars)")

        return proposer_reflection, actor_reflection

    def save_reflections(
        self,
        proposer_reflection: str,
        actor_reflection: str,
        question: Dict[str, Any]
    ) -> str:
        """
        Save reflections to file.

        Args:
            proposer_reflection: JSON string for Proposer
            actor_reflection: JSON string for Actor
            question: Question dict for naming

        Returns:
            Path to saved file
        """
        import json
        import re

        modality = question.get('modality', 'vision')
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))

        # Extract dataset_name and q_type from dataset_base_name
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"

        # Try to parse JSON strings for cleaner storage
        try:
            proposer_dict = json.loads(proposer_reflection)
        except json.JSONDecodeError:
            proposer_dict = {"raw": proposer_reflection}

        try:
            actor_dict = json.loads(actor_reflection)
        except json.JSONDecodeError:
            actor_dict = {"raw": actor_reflection}

        reflections = {
            "proposer_reflection": proposer_dict,
            "actor_reflection": actor_dict
        }

        filename = "reflections"
        subdir = f"reflections/{modality}/{dataset_name}/{q_type_str}/{row_no}"
        filepath = self.save_json(reflections, filename, subdir)

        print(f"  Reflections saved to: {filepath}")
        return filepath

    def batch_evaluate(
        self,
        results_list: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        inputs: Optional[List[Any]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate multiple results in batch.

        Args:
            results_list: List of Actor results
            questions: List of questions
            inputs: List of original inputs
            predictions: List of predictions

        Returns:
            Aggregated evaluation results
        """
        evaluations = []

        for i, (results, question) in enumerate(zip(results_list, questions)):
            original_input = inputs[i] if inputs else None
            prediction = predictions[i] if predictions else None

            eval_result = self.evaluate(
                results=results,
                question=question,
                original_input=original_input,
                original_prediction=prediction,
                **kwargs
            )
            evaluations.append(eval_result)

        # Aggregate metrics
        quality_scores = [e.get('quality_score', 0) for e in evaluations]
        faithfulness_scores = [
            e.get('faithfulness', {}).get('score', 0)
            for e in evaluations
            if e.get('faithfulness', {}).get('score') is not None
        ]

        return {
            "individual_evaluations": evaluations,
            "aggregate": {
                "num_evaluated": len(evaluations),
                "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "avg_faithfulness_score": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
                "num_faithfulness_evaluated": len(faithfulness_scores)
            }
        }

    # =========================================================================
    # Q5/Q6/Q7 Specific Parsing Helpers
    # =========================================================================

    def _extract_q_specific_kwargs(
        self, q_type: int, question: Dict, modality: str, original_input: Any
    ) -> Dict:
        """
        Extract Q5/Q6/Q7-specific kwargs by parsing the question's example field.

        Q5 needs 'queried_region', Q6 needs 'expected_class', Q7 needs 'part_to_change'.
        These are embedded in the natural language 'example' field of each benchmark entry.
        """
        kwargs: Dict = {}
        example = question.get('example', '')
        if not example:
            return kwargs

        if q_type == 5:
            region = self._parse_queried_region(example, modality, original_input, question)
            if region is not None:
                kwargs['queried_region'] = region
        elif q_type == 6:
            expected_class = self._parse_expected_class(example, modality)
            if expected_class is not None:
                kwargs['expected_class'] = expected_class
        elif q_type == 7:
            region = self._parse_queried_region(example, modality, original_input, question)
            if region is not None:
                kwargs['part_to_change'] = region

        return kwargs

    def _parse_queried_region(
        self, example: str, modality: str, original_input: Any, question: Dict
    ) -> Optional[Dict]:
        """
        Parse the queried region from the example sentence.

        Text: extracts word from 'mask the word X' / 'remove/change the word X',
              finds its character position in source text.
        Tabular: extracts feature name from 'mask the X feature' / 'remove/change X, how'.
        Vision: returns None (evaluator resolves region from context).
        """
        if modality == 'text':
            m = re.search(
                r"(?:mask|remove/change|remove|change)\s+the\s+word\s+'([^']+)'",
                example, re.IGNORECASE
            )
            if not m:
                return None
            word = m.group(1)
            source_text = self._get_text_from_input(original_input, question)
            if source_text:
                idx = source_text.find(word)
                if idx >= 0:
                    return {"start_index": idx, "end_index": idx + len(word)}
            return None

        elif modality == 'tabular':
            # Q5: "mask the <feature> feature"
            m = re.search(r"mask\s+the\s+(.+?)\s+feature", example, re.IGNORECASE)
            if m:
                return {"feature_key": m.group(1).strip()}
            # Q7: "remove/change <feature>, how" or "remove/change <feature>?"
            m = re.search(
                r"remove/change\s+(.+?)(?:\s*,|\s*\?|$)", example, re.IGNORECASE
            )
            if m:
                return {"feature_key": m.group(1).strip()}
            return None

        # Vision: don't attempt to parse from text
        return None

    def _parse_expected_class(self, example: str, modality: str) -> Optional[str]:
        """
        Parse the target/expected class from a Q6 example sentence.

        Text: "flip the model's prediction to <class>"
        Tabular: "flip the model into <class>"
        """
        # Text pattern (handles apostrophe variants)
        m = re.search(
            r"flip the model['\u2019]?s prediction to\s+([^?.\n]+)",
            example, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        # Tabular pattern
        m = re.search(r"flip the model into\s+([^?.\n]+)", example, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _get_text_from_input(self, original_input: Any, question: Dict) -> Optional[str]:
        """
        Extract the source text string used for word-position search (Q5/Q7 text).

        For NLI inputs the TextMasker operates on the premise, so we return premise.
        """
        # Prefer structured features from the benchmark question dict
        features = question.get('features', {})
        if isinstance(features, dict):
            if 'premise' in features:
                return features['premise']
            if 'text' in features:
                return features['text']
        # Fall back to original_input
        if isinstance(original_input, str):
            return original_input
        if isinstance(original_input, dict):
            if 'premise' in original_input:
                return original_input['premise']
            if 'text' in original_input:
                return original_input['text']
        return None
