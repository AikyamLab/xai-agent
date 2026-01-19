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
            **kwargs: Additional arguments (processor, tokenizer, etc.)

        Returns:
            Evaluation dictionary with metrics
        """
        return self.evaluate(
            results, question, original_input,
            original_prediction, ground_truth, **kwargs
        )

    def evaluate(
        self,
        results: Dict[str, Any],
        question: Dict[str, Any],
        original_input: Any = None,
        original_prediction: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Any] = None,
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
            faithfulness_result = self.evaluate_faithfulness(
                q_type=q_type,
                agent_output=results,
                original_input=original_input,
                original_prediction=original_prediction or {},
                modality=modality,
                ground_truth=ground_truth,
                **kwargs
            )
            evaluation["faithfulness"] = faithfulness_result
        else:
            evaluation["faithfulness"] = {
                "score": None,
                "error": "Model or input not provided for faithfulness evaluation"
            }

        # Save evaluation
        filepath = self.save_json(evaluation, f"eval_{results.get('question_id', 'unknown')}", "evaluations")
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
            **kwargs: Additional arguments

        Returns:
            Faithfulness evaluation result
        """
        try:
            evaluator = self._get_evaluator(q_type, modality)
            if evaluator is None:
                return {"score": None, "error": f"No evaluator for Q{q_type}"}

            result = evaluator.evaluate(
                agent_output=agent_output,
                original_input=original_input,
                model=self.model,
                original_prediction=original_prediction,
                ground_truth=ground_truth,
                **kwargs
            )

            return result.to_dict() if hasattr(result, 'to_dict') else result

        except Exception as e:
            print(f"  Warning: Faithfulness evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"score": None, "error": str(e)}

    def _get_evaluator(self, q_type: int, modality: str) -> Any:
        """Get or create evaluator for question type"""
        key = (q_type, modality)

        if key not in self._evaluators:
            try:
                from evaluation import get_evaluator
                self._evaluators[key] = get_evaluator(q_type, modality)
            except ImportError as e:
                print(f"  Warning: Could not import evaluator: {e}")
                return None

        return self._evaluators.get(key)

    def _calculate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic quality metrics for execution"""
        tool_results = results.get('tool_results', {})
        output = results.get('output', {})

        # Tool execution quality
        successful_tools = sum(
            1 for r in tool_results.values()
            if isinstance(r, dict) and r.get('success')
        )
        total_tools = len(tool_results)
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

    def evaluate_strategy_faithfulness(
        self,
        strategy: Dict[str, Any],
        expected_behavior: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate strategy faithfulness.

        TODO: This is a placeholder for future implementation.

        Args:
            strategy: Strategy from Proposer Agent
            expected_behavior: Expected behavior specification

        Returns:
            Strategy faithfulness evaluation
        """
        # TODO: Implement strategy faithfulness evaluation
        # This should evaluate whether the selected tools and approach
        # are appropriate for the question type
        raise NotImplementedError(
            "Strategy faithfulness evaluation not yet implemented. "
            "This is a TODO placeholder for future development."
        )

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
