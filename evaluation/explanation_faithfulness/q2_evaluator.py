"""
Q2 Evaluator: Least Responsible Part

Metric: 1 - |P_original - P_modified|
Higher score = better (the identified part was indeed unimportant, causing minimal change)
Score range: [0, 1], threshold: 0.95 (i.e., |probability_drop| < 0.05)
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker


class Q2Evaluator(BaseEvaluator):
    """Evaluator for Q2: Least responsible part identification"""

    @property
    def question_type(self) -> int:
        return 2

    @property
    def metric_name(self) -> str:
        return "Stability Score"

    @property
    def metric_formula(self) -> str:
        return "1 - |P_original - P_modified|"

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q2: Mask the identified least-responsible part.

        A smaller probability change indicates the part was indeed unimportant.

        Args:
            agent_output: Agent output with identified region
            original_input: Original input data
            model: Target model
            original_prediction: Original prediction
            **kwargs: processor, device, etc.

        Returns:
            EvaluationResult with inverse probability drop score
        """
        try:
            region = self.extract_region(agent_output)
            if region is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metric_name=self.metric_name,
                    metric_formula=self.metric_formula,
                    errors=["Could not extract region from agent output"]
                )

            # Get original probability
            original_class = original_prediction.get('predicted_class_idx', 0)
            original_probs = original_prediction.get('probabilities')
            if original_probs is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metric_name=self.metric_name,
                    metric_formula=self.metric_formula,
                    errors=["Original probabilities not available"]
                )
            original_probs = self._normalize_probs(original_probs)
            p_original = float(original_probs[original_class])

            # Mask using modality-appropriate default strategy
            masker = get_masker(self.modality)
            masked_input = masker.mask(
                original_input, region,
                dataset_base_name=kwargs.get('dataset_base_name'),
                row_no=kwargs.get('row_no'),
                tool_name=kwargs.get('tool_name'),
                mask_suffix=kwargs.get('mask_suffix', ''),
                feature_names=kwargs.get('feature_names', [])
            )

            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            modified_prediction = self.get_prediction(model, masked_input, processor, device)
            modified_probs = modified_prediction.get('probabilities')
            if modified_probs is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metric_name=self.metric_name,
                    metric_formula=self.metric_formula,
                    errors=["Modified probabilities not available"]
                )
            p_modified = float(modified_probs[original_class])

            # Calculate metric: 1 - |P_original - P_modified|
            # For least responsible, we want minimal change (positive or negative)
            # Higher score (closer to 1) = better
            probability_drop = p_original - p_modified
            soft_score = 1.0 - abs(probability_drop)

            # Size penalty: penalize small regions (reward finding large unimportant areas)
            region_ratio = self.compute_region_ratio(region, original_input)
            size_penalty = region_ratio
            score = soft_score * size_penalty

            # Passed if soft_score >= 0.95 (i.e., |probability_drop| < 0.05)
            threshold = 0.95
            passed = soft_score >= threshold

            return EvaluationResult(
                score=score,
                passed=passed,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                p_original=p_original,
                p_modified=p_modified,
                original_class=str(original_class),
                details={
                    "region": region,
                    "probability_drop": probability_drop,
                    "soft_score": soft_score,
                    "region_ratio": region_ratio,
                    "size_penalty": size_penalty,
                    "threshold": threshold,
                    "interpretation": "score = (1 - |P_orig - P_mod|) * region_ratio"
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
