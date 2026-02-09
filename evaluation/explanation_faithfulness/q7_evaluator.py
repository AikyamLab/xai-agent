"""
Q7 Evaluator: Predict Change Outcome

Metric: 1 if R1_modified == agent_predicted_class, else 0
Success if agent correctly predicted the new class after modification
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q7Evaluator(BaseEvaluator):
    """Evaluator for Q7: Predict how prediction changes after modification"""

    @property
    def question_type(self) -> int:
        return 7

    @property
    def metric_name(self) -> str:
        return "Prediction Accuracy"

    @property
    def metric_formula(self) -> str:
        return "1 if R1_modified == agent_predicted_class else 0"

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q7: Check if agent correctly predicted the class after modification.

        Args:
            agent_output: Agent output with changed_class prediction
            original_input: Original input data
            model: Target model
            original_prediction: Original prediction
            **kwargs: part_to_change, processor, device

        Returns:
            EvaluationResult with accuracy score (1 or 0)
        """
        try:
            # Get agent's predicted class after change
            output_data = agent_output.get('output', {})
            agent_predicted_class = output_data.get('changed_class')

            if agent_predicted_class is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Agent did not provide changed_class prediction"]
                )

            # Get the part to change (should be specified in context)
            part_to_change = kwargs.get('part_to_change')
            if part_to_change is None:
                # Use the most important region from original analysis
                part_to_change = self._get_important_region(original_prediction, kwargs)

            if part_to_change is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Part to change not specified"]
                )

            # Apply modification (mask the important part, use GRAY for neutral masking)
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            modified_input = masker.mask(
                original_input, part_to_change,
                dataset_base_name=kwargs.get('dataset_base_name'),
                row_no=kwargs.get('row_no'),
                tool_name=kwargs.get('tool_name'),
                mask_suffix=kwargs.get('mask_suffix', '')
            )

            # Get actual prediction after modification
            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            modified_prediction = self.get_prediction(model, modified_input, processor, device)
            actual_modified_class = modified_prediction.get('predicted_class_idx', -1)

            # Convert agent's prediction to index for comparison
            class_names = kwargs.get('class_names')
            agent_predicted_idx = self._get_class_index(agent_predicted_class, class_names)

            # Check if agent correctly predicted
            correct = (actual_modified_class == agent_predicted_idx)
            score = 1.0 if correct else 0.0

            return EvaluationResult(
                score=score,
                passed=correct,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                original_class=str(original_prediction.get('predicted_class_idx')),
                modified_class=str(actual_modified_class),
                details={
                    "part_to_change": part_to_change,
                    "agent_predicted_class": agent_predicted_class,
                    "agent_predicted_idx": agent_predicted_idx,
                    "actual_modified_class": actual_modified_class,
                    "interpretation": "1 = agent correctly predicted the new class"
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

    def _get_important_region(self, prediction: Dict, kwargs: Dict) -> Dict:
        """Get the important region to modify"""
        # Try to get from kwargs
        region = kwargs.get('region')
        if region:
            return region

        # Try to construct from context
        context = kwargs.get('context', {})
        queried_part = context.get('part_to_change', {})
        if isinstance(queried_part, dict) and queried_part:
            return queried_part

        return None

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
