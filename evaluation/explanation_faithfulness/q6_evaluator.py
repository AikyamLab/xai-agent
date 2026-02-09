"""
Q6 Evaluator: How to Flip Prediction

Metric: 1 if R1_modified == expected_class, else 0
Success if applying the change plan actually flips to target class
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q6Evaluator(BaseEvaluator):
    """Evaluator for Q6: How to flip prediction to target class"""

    @property
    def question_type(self) -> int:
        return 6

    @property
    def metric_name(self) -> str:
        return "Flip Success"

    @property
    def metric_formula(self) -> str:
        return "1 if R1_modified == expected_class else 0"

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q6: Apply change plan and check if prediction flips to target.

        Args:
            agent_output: Agent output with change_plan
            original_input: Original input data
            model: Target model
            original_prediction: Original prediction
            **kwargs: expected_class, processor, device

        Returns:
            EvaluationResult with flip success score (1 or 0)
        """
        try:
            # Get change plan from agent output
            output_data = agent_output.get('output', {})
            change_plan = output_data.get('change_plan', {})

            if not change_plan:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["No change plan provided by agent"]
                )

            # Get target/expected class
            expected_class = kwargs.get('expected_class')
            if expected_class is None:
                expected_class = kwargs.get('target_class')
            if expected_class is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Expected/target class not provided"]
                )

            # Convert expected_class to index if it's a string
            expected_class_idx = self._get_class_index(expected_class, kwargs.get('class_names'))

            # Extract region from change plan
            region = self._extract_region_from_change_plan(change_plan)
            if region is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Could not extract region from change plan"]
                )

            # Apply modification (for now, we use masking as approximation)
            # TODO: For more sophisticated evaluation, implement actual change application
            action = change_plan.get('action', 'delete')

            if action in ['delete', 'change']:
                # Mask the region (use GRAY for neutral masking)
                masker = get_masker(self.modality, MaskingStrategy.GRAY)
                modified_input = masker.mask(
                    original_input, region,
                    dataset_base_name=kwargs.get('dataset_base_name'),
                    row_no=kwargs.get('row_no'),
                    tool_name=kwargs.get('tool_name'),
                    mask_suffix=kwargs.get('mask_suffix', '')
                )
            else:
                # For add/swap, we'd need more complex logic
                # For now, treat as masking (use GRAY for neutral masking)
                masker = get_masker(self.modality, MaskingStrategy.GRAY)
                modified_input = masker.mask(
                    original_input, region,
                    dataset_base_name=kwargs.get('dataset_base_name'),
                    row_no=kwargs.get('row_no'),
                    tool_name=kwargs.get('tool_name'),
                    mask_suffix=kwargs.get('mask_suffix', '')
                )

            # Get prediction on modified input
            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            modified_prediction = self.get_prediction(model, modified_input, processor, device)
            modified_class = modified_prediction.get('predicted_class_idx', -1)

            # Check if flipped to expected class
            flipped_correctly = (modified_class == expected_class_idx)
            score = 1.0 if flipped_correctly else 0.0

            return EvaluationResult(
                score=score,
                passed=flipped_correctly,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                original_class=str(original_prediction.get('predicted_class_idx')),
                modified_class=str(modified_class),
                details={
                    "change_plan": change_plan,
                    "expected_class": expected_class,
                    "expected_class_idx": expected_class_idx,
                    "flipped_correctly": flipped_correctly,
                    "interpretation": "1 = change plan successfully flipped to target class"
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

    def _extract_region_from_change_plan(self, change_plan: Dict) -> Dict:
        """Extract region from change plan based on modality"""
        if self.modality == "vision":
            bbox = change_plan.get('bounding_box')
            return {"bounding_box": bbox} if bbox else None
        elif self.modality == "text":
            start = change_plan.get('start_index')
            end = change_plan.get('end_index')
            return {"start_index": start, "end_index": end} if start is not None else None
        else:
            key = change_plan.get('feature_key')
            return {"feature_key": key} if key else None

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
            # Try to parse as integer
            try:
                return int(class_ref)
            except ValueError:
                pass
        return -1
