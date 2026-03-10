"""
Q5 Evaluator: Would Masking Change Prediction?

Metric: 1 if agent's prediction matches actual outcome, else 0
Accuracy of agent's yes/no prediction
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker


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
        # Get agent's prediction (1 = yes changes, 0 = no)
        output_data = agent_output.get('output', {})
        agent_says_changes = output_data.get('prediction_changes', 0)
        agent_says_changes = int(agent_says_changes)

        # Get the queried region (the part being asked about)
        queried_region = kwargs.get('queried_region')
        if not queried_region:
            # Try to extract from context
            queried_region = self._get_queried_region(kwargs.get('context', {}))
        if not queried_region:
            # For vision Q5, use the agent's own masked_region output
            queried_region = output_data.get('masked_region')

        if not queried_region:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Queried region not provided"]
            )

        # Mask using modality-appropriate default strategy
        masker = get_masker(self.modality, preprocessor=kwargs.get('processor'))
        masked_input = masker.mask(
            original_input, queried_region,
            dataset_base_name=kwargs.get('dataset_base_name'),
            row_no=kwargs.get('row_no'),
            tool_name=kwargs.get('tool_name'),
            mask_suffix=kwargs.get('mask_suffix', ''),
            feature_names=kwargs.get('feature_names', []),
            original_features=kwargs.get('original_features', {})
        )

        processor = kwargs.get('processor')
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        modified_prediction = self.get_prediction(model, masked_input, processor, device)

        # Check if prediction actually changed
        original_class = original_prediction.get('predicted_class_idx', 0)
        modified_class = modified_prediction.get('predicted_class_idx', 0)

        actually_changed = int(original_class != modified_class)

        # Passed logic unchanged: binary correct/wrong
        correct = (agent_says_changes == actually_changed)

        # Soft score: measure how well the direction matches
        original_probs = original_prediction.get('probabilities')
        modified_probs = modified_prediction.get('probabilities')
        if original_probs is not None and modified_probs is not None:
            original_probs = self._normalize_probs(original_probs)
            modified_probs = self._normalize_probs(modified_probs)
            p_orig = float(original_probs[original_class])
            p_mod = float(modified_probs[original_class])
            prob_drop = p_orig - p_mod  # positive = class weakened

            if agent_says_changes == 1:
                # Agent said "changes": reward by how much the original class dropped
                score = max(0.0, prob_drop)
            else:
                # Agent said "doesn't change": reward by how stable the prediction was
                score = max(0.0, 1.0 - abs(prob_drop))
        else:
            p_orig = None
            p_mod = None
            prob_drop = None
            score = 1.0 if correct else 0.0

        return EvaluationResult(
            score=score,
            passed=correct,
            metric_name=self.metric_name,
            metric_formula=self.metric_formula,
            original_class=str(original_class),
            modified_class=str(modified_class),
            details={
                "soft_score": score,
                "queried_region": queried_region,
                "agent_says_changes": agent_says_changes,
                "actually_changed": actually_changed,
                "p_original_class_before_mask": p_orig,
                "p_original_class_after_mask": p_mod,
                "prob_drop": prob_drop,
                "binary_correct": correct,
                "interpretation": "Soft score: agent says changes -> prob drop; agent says stable -> 1-|drop|"
            }
        )

    def _get_queried_region(self, context: Dict) -> Optional[Dict]:
        """Extract queried region from context"""
        queried_part = context.get('queried_part', {})
        if isinstance(queried_part, dict) and queried_part:
            return queried_part
        return None
