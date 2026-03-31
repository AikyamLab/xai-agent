"""
Q8 Evaluator: Irrelevant Parts Causing Wrong Prediction

Metric: 1 if P_correct_modified - P_correct_original > threshold, else 0
Success if masking spurious part improves correct class probability
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker

threshold = 0.1

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
        return "1 if P_correct_modified - P_correct_original > threshold else 0"

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

        err = self.validate_region(region, original_input, **kwargs)
        if err:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                errors=[err]
            )

        # Get original probability for correct class
        original_probs = original_prediction.get('probabilities')
        if original_probs is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Original probabilities not available"]
            )

        original_probs = self._normalize_probs(original_probs)
        p_correct_original = float(original_probs[correct_class_idx])

        # Mask using modality-appropriate default strategy
        masker = get_masker(self.modality, preprocessor=kwargs.get('processor'))
        masked_input = masker.mask(
            original_input, region,
            dataset_base_name=kwargs.get('dataset_base_name'),
            row_no=kwargs.get('row_no'),
            tool_name=kwargs.get('tool_name'),
            mask_suffix=kwargs.get('mask_suffix', ''),
            feature_names=kwargs.get('feature_names', []),
            original_features=kwargs.get('original_features', {})
        )

        # Get prediction on masked input
        processor = kwargs.get('processor')
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        modified_prediction = self.get_prediction(model, masked_input, processor, device)
        modified_probs = self._normalize_probs(modified_prediction.get('probabilities'))

        if modified_probs is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Modified probabilities not available"]
            )

        p_correct_modified = float(modified_probs[correct_class_idx])

        # Passed logic unchanged: binary improved > threshold or not
        improvement = p_correct_modified - p_correct_original
        improved = improvement > threshold

        # Soft score: normalized improvement (proportion of improvable space used)
        room = 1.0 - p_correct_original
        if room > 1e-8:
            soft_score = max(0.0, min(1.0, improvement / room))
        else:
            # Already near-perfect probability; any improvement is full score
            soft_score = 1.0 if improved else 0.0

        # Size penalty: penalize large masked regions
        region_ratio = self.compute_region_ratio(region, original_input)
        size_penalty = 1.0 - region_ratio
        size_score = soft_score * size_penalty
        score = soft_score

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
                "improvement": improvement,
                "threshold": threshold,
                "room_for_improvement": room,
                "soft_score": soft_score,
                "size_score": size_score,
                "size_score_l1": self._size_score_l1(soft_score, region_ratio, penalize_large=True),
                "region_ratio": region_ratio,
                "size_penalty": size_penalty,
                "interpretation": "score = normalized_improvement; size_score = score * (1 - region_ratio)"
            }
        )

    def _get_class_index(self, class_ref: Any, class_names=None) -> int:
        """Convert class reference to index. class_names may be a list or dict."""
        if isinstance(class_ref, int):
            return class_ref
        if isinstance(class_ref, str):
            if class_names:
                try:
                    if isinstance(class_names, dict):
                        # label_map: {idx: name} or {name: idx}
                        for k, v in class_names.items():
                            if v == class_ref:
                                return int(k)
                            if k == class_ref:
                                return int(v)
                    else:
                        return class_names.index(class_ref)
                except (ValueError, TypeError):
                    pass
            try:
                return int(class_ref)
            except ValueError:
                pass
        return -1
