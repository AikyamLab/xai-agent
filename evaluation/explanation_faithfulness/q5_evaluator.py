"""
Q5 Evaluator: Would Masking Change Prediction?

Metric: 1 if agent's prediction matches actual outcome, else 0
Accuracy of agent's yes/no prediction
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q5Evaluator(BaseEvaluator):
    """Evaluator for Q5: Would masking change the prediction?"""

    @property
    def question_type(self) -> int:
        return 5

    @property
    def metric_name(self) -> str:
        return "Prediction Accuracy"

    @property
    def metric_formula(self) -> str:
        return "1 if agent_prediction == actual_outcome else 0"

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q5: Check if agent correctly predicted whether masking changes prediction.

        Args:
            agent_output: Agent output with prediction_changes (1 or 0)
            original_input: Original input data
            model: Target model
            original_prediction: Original prediction
            **kwargs: queried_region, processor, device

        Returns:
            EvaluationResult with accuracy score (1 or 0)
        """
        try:
            # Get agent's prediction (1 = yes changes, 0 = no)
            output_data = agent_output.get('output', {})
            agent_says_changes = output_data.get('prediction_changes', 0)
            agent_says_changes = int(agent_says_changes)

            # Get the queried region (the part being asked about)
            queried_region = kwargs.get('queried_region')
            if queried_region is None:
                # Try to extract from context
                queried_region = self._get_queried_region(kwargs.get('context', {}))

            if queried_region is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Queried region not provided"]
                )

            # Actually mask and check (use GRAY for neutral masking)
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            masked_input = masker.mask(original_input, queried_region)

            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            modified_prediction = self.get_prediction(model, masked_input, processor, device)

            # Check if prediction actually changed
            original_class = original_prediction.get('predicted_class_idx', 0)
            modified_class = modified_prediction.get('predicted_class_idx', 0)

            actually_changed = int(original_class != modified_class)

            # Score: did agent correctly predict?
            correct = (agent_says_changes == actually_changed)
            score = 1.0 if correct else 0.0

            return EvaluationResult(
                score=score,
                passed=correct,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                original_class=str(original_class),
                modified_class=str(modified_class),
                details={
                    "queried_region": queried_region,
                    "agent_says_changes": agent_says_changes,
                    "actually_changed": actually_changed,
                    "interpretation": "1 = agent correctly predicted, 0 = wrong prediction"
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

    def _get_queried_region(self, context: Dict) -> Dict:
        """Extract queried region from context"""
        queried_part = context.get('queried_part', {})
        if isinstance(queried_part, dict):
            return queried_part
        return None
