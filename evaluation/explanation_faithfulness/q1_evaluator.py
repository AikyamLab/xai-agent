"""
Q1 Evaluator: Most Responsible Part

Metric: P_original - P_modified
Higher score = better (the identified part was indeed important)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q1Evaluator(BaseEvaluator):
    """Evaluator for Q1: Most responsible part identification"""

    @property
    def question_type(self) -> int:
        return 1

    @property
    def metric_name(self) -> str:
        return "Probability Drop"

    @property
    def metric_formula(self) -> str:
        return "P_original - P_modified"

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q1: Mask the identified most-responsible part and measure probability drop.

        A larger drop indicates the part was indeed important (good explanation).

        Args:
            agent_output: Agent output with identified region
            original_input: Original input data
            model: Target model
            original_prediction: Original prediction
            **kwargs: processor, device, etc.

        Returns:
            EvaluationResult with probability drop score
        """
        try:
            # Extract region from agent output
            region = self.extract_region(agent_output)
            if region is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metric_name=self.metric_name,
                    metric_formula=self.metric_formula,
                    errors=["Could not extract region from agent output"]
                )

            # Get original probability for predicted class
            original_class = original_prediction.get('predicted_class_idx', 0)
            original_probs = original_prediction.get('probabilities')
            if original_probs is not None:
                p_original = float(original_probs[original_class])
            else:
                p_original = original_prediction.get('confidence', 1.0)

            # Create masker and mask the identified region (use GRAY for neutral masking)
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            masked_input = masker.mask(original_input, region)

            # Get prediction on masked input
            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            modified_prediction = self.get_prediction(model, masked_input, processor, device)
            modified_probs = modified_prediction.get('probabilities')

            if modified_probs is not None:
                p_modified = float(modified_probs[original_class])
            else:
                p_modified = modified_prediction.get('confidence', 0.0)

            # Calculate metric: P_original - P_modified
            score = p_original - p_modified

            # Determine if passed (significant drop indicates important region)
            # Threshold: at least 0.1 probability drop
            passed = score > 0.1

            return EvaluationResult(
                score=score,
                passed=passed,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                p_original=p_original,
                p_modified=p_modified,
                original_class=str(original_class),
                modified_class=str(modified_prediction.get('predicted_class_idx')),
                details={
                    "region": region,
                    "threshold": 0.1,
                    "interpretation": "Higher score = agent correctly identified important region"
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
