"""
Q2 Evaluator: Least Responsible Part

Metric: -(P_original - P_modified)
Lower absolute difference = better (the identified part was indeed unimportant)
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q2Evaluator(BaseEvaluator):
    """Evaluator for Q2: Least responsible part identification"""

    @property
    def question_type(self) -> int:
        return 2

    @property
    def metric_name(self) -> str:
        return "Inverse Probability Drop"

    @property
    def metric_formula(self) -> str:
        return "-(P_original - P_modified)"

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
            p_original = float(original_probs[original_class]) if original_probs is not None else original_prediction.get('confidence', 1.0)

            # Mask and get new prediction (use GRAY for neutral masking)
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            masked_input = masker.mask(original_input, region)

            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            modified_prediction = self.get_prediction(model, masked_input, processor, device)
            modified_probs = modified_prediction.get('probabilities')
            p_modified = float(modified_probs[original_class]) if modified_probs is not None else modified_prediction.get('confidence', 0.0)

            # Calculate metric: -(P_original - P_modified)
            # For least responsible, we want small change, so negative of drop is our score
            # Higher score (closer to 0) = better
            probability_drop = p_original - p_modified
            score = -probability_drop  # Negate so that small drop = high score

            # Passed if probability change is minimal (< 0.05)
            passed = abs(probability_drop) < 0.05

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
                    "threshold": 0.05,
                    "interpretation": "Score closer to 0 = agent correctly identified unimportant region"
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
