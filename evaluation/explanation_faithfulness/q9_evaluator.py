"""
Q9 Evaluator: Shared Feature in Misclassified Inputs

Metric: 1 if ALL inputs improve (P_correct_modified > P_correct_original for all), else 0
Success if removing shared feature improves all predictions
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..base_evaluator import MultiInstanceEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q9Evaluator(MultiInstanceEvaluator):
    """Evaluator for Q9: Shared feature in misclassified inputs"""

    @property
    def question_type(self) -> int:
        return 9

    @property
    def metric_name(self) -> str:
        return "All Improved"

    @property
    def metric_formula(self) -> str:
        return "1 if ALL (P_correct_modified > P_correct_original) else 0"

    def evaluate_multi(
        self,
        agent_output: Dict[str, Any],
        inputs: List[Any],
        model: nn.Module,
        predictions: List[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q9: Check if removing shared feature improves ALL inputs.

        Args:
            agent_output: Agent output with regions for each input
            inputs: List of misclassified inputs
            model: Target model
            predictions: List of predictions (all wrong)
            **kwargs: ground_truths (list), processor, device

        Returns:
            EvaluationResult with all-improved score (1 or 0)
        """
        try:
            ground_truths = kwargs.get('ground_truths', kwargs.get('ground_truth', []))
            if isinstance(ground_truths, (str, int)):
                ground_truths = [ground_truths] * len(inputs)

            if len(inputs) < 2:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Need at least 2 inputs"]
                )

            if len(ground_truths) < len(inputs):
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Ground truths not provided for all inputs"]
                )

            # Extract regions for all inputs
            output_data = agent_output.get('output', {})
            regions = {}
            for key in output_data:
                if key.startswith('input_'):
                    regions[key] = self._extract_instance_region(output_data[key])

            if len(regions) < len(inputs):
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Regions not provided for all inputs"]
                )

            # Evaluate each input
            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            class_names = kwargs.get('class_names')

            improvements = []
            details_per_input = []

            for i, (input_data, pred) in enumerate(zip(inputs, predictions)):
                input_key = f"input_{chr(ord('A') + i)}"
                region = regions.get(input_key)

                if region is None:
                    improvements.append(False)
                    details_per_input.append({"error": "Region not found"})
                    continue

                # Get correct class for this input
                correct_class_idx = self._get_class_index(ground_truths[i], class_names)

                # Get original probability for correct class
                original_probs = pred.get('probabilities')
                if original_probs is None:
                    improvements.append(False)
                    details_per_input.append({"error": "Probabilities not available"})
                    continue

                p_correct_original = float(original_probs[correct_class_idx])

                # Mask and get new prediction (use GRAY for neutral masking)
                # Use instance suffix for multi-instance saving (e.g., _A, _B, _C...)
                instance_suffix = f"_{chr(ord('A') + i)}"
                masker = get_masker(self.modality, MaskingStrategy.GRAY)
                masked_input = masker.mask(
                    input_data, region,
                    dataset_base_name=kwargs.get('dataset_base_name'),
                    row_no=kwargs.get('row_no'),
                    tool_name=kwargs.get('tool_name'),
                    instance_suffix=instance_suffix,
                    mask_suffix=kwargs.get('mask_suffix', '')
                )
                modified_pred = self.get_prediction(model, masked_input, processor, device)
                modified_probs = modified_pred.get('probabilities')

                if modified_probs is None:
                    improvements.append(False)
                    details_per_input.append({"error": "Modified probabilities not available"})
                    continue

                p_correct_modified = float(modified_probs[correct_class_idx])

                improved = p_correct_modified > p_correct_original
                improvements.append(improved)
                details_per_input.append({
                    "input_key": input_key,
                    "correct_class": ground_truths[i],
                    "p_correct_original": p_correct_original,
                    "p_correct_modified": p_correct_modified,
                    "improved": improved
                })

            # All must improve
            all_improved = all(improvements)
            score = 1.0 if all_improved else 0.0

            return EvaluationResult(
                score=score,
                passed=all_improved,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                details={
                    "num_inputs": len(inputs),
                    "num_improved": sum(improvements),
                    "all_improved": all_improved,
                    "shared_feature_description": agent_output.get('shared_feature_description', ''),
                    "per_input_details": details_per_input,
                    "interpretation": "1 = removing shared feature improved ALL inputs"
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

    def _extract_instance_region(self, instance_data: Dict) -> Dict:
        """Extract region from instance data"""
        if self.modality == "vision":
            bbox = instance_data.get('bounding_box')
            return {"bounding_box": bbox} if bbox else None
        elif self.modality == "text":
            start = instance_data.get('start_index')
            end = instance_data.get('end_index')
            return {"start_index": start, "end_index": end} if start is not None else None
        else:
            key = instance_data.get('feature_key')
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
            try:
                return int(class_ref)
            except ValueError:
                pass
        return -1
