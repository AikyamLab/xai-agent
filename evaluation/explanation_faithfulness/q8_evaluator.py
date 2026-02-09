"""
Q8 Evaluator: Irrelevant Parts Causing Wrong Prediction

Metric: 1 if P_correct_modified > P_correct_original, else 0
Success if masking spurious part improves correct class probability
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q8Evaluator(BaseEvaluator):
    """Evaluator for Q8: Irrelevant parts causing wrong prediction"""

    @property
    def question_type(self) -> int:
        return 8

    @property
    def metric_name(self) -> str:
        return "Correct Class Improvement"

    @property
    def metric_formula(self) -> str:
        return "1 if P_correct_modified > P_correct_original else 0"

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q8: Check if masking spurious part improves correct class probability.

        Args:
            agent_output: Agent output with identified spurious region
            original_input: Original input data
            model: Target model
            original_prediction: Original (wrong) prediction
            **kwargs: ground_truth, processor, device

        Returns:
            EvaluationResult with improvement score (1 or 0)
        """
        try:
            # Get ground truth (correct class)
            ground_truth = kwargs.get('ground_truth')
            if ground_truth is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Ground truth not provided"]
                )

            # Convert ground truth to class index
            class_names = kwargs.get('class_names')
            correct_class_idx = self._get_class_index(ground_truth, class_names)

            if correct_class_idx < 0:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Could not determine correct class index"]
                )

            # Extract spurious region from agent output
            region = self.extract_region(agent_output)
            if region is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Could not extract spurious region from agent output"]
                )

            # Get original probability for correct class
            original_probs = original_prediction.get('probabilities')
            if original_probs is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Original probabilities not available"]
                )

            p_correct_original = float(original_probs[correct_class_idx])

            # Mask spurious region (use GRAY for neutral masking)
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            masked_input = masker.mask(
                original_input, region,
                dataset_base_name=kwargs.get('dataset_base_name'),
                row_no=kwargs.get('row_no'),
                tool_name=kwargs.get('tool_name'),
                mask_suffix=kwargs.get('mask_suffix', '')
            )

            # Get prediction on masked input
            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            modified_prediction = self.get_prediction(model, masked_input, processor, device)
            modified_probs = modified_prediction.get('probabilities')

            if modified_probs is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Modified probabilities not available"]
                )

            p_correct_modified = float(modified_probs[correct_class_idx])

            # Check if correct class probability improved
            improved = p_correct_modified > p_correct_original
            score = 1.0 if improved else 0.0

            return EvaluationResult(
                score=score,
                passed=improved,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                p_original=p_correct_original,
                p_modified=p_correct_modified,
                details={
                    "region": region,
                    "ground_truth": ground_truth,
                    "correct_class_idx": correct_class_idx,
                    "p_correct_original": p_correct_original,
                    "p_correct_modified": p_correct_modified,
                    "improvement": p_correct_modified - p_correct_original,
                    "interpretation": "1 = masking spurious part improved correct class probability"
                }
            )

        except Exception as e:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                errors=[str(e)]
            )

    def _get_class_index(self, class_ref: Any, class_names: list = None) -> int:
        """Convert class reference to index"""
        if isinstance(class_ref, int):
            return class_ref
        if isinstance(class_ref, str):
            if class_names:
                try:
                    return class_names.index(class_ref)
                except ValueError:
                    pass
            try:
                return int(class_ref)
            except ValueError:
                pass
        return -1
