"""
Q4 Evaluator: Contrastive Instances (Why A != B)

Metric: 1 if (P_A_a1_orig - P_B_a1_orig) - (P_A_a1_mod - P_B_a1_mod) > 0, else 0
Success if masking reduces the probability gap between instances
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..base_evaluator import MultiInstanceEvaluator, EvaluationResult
from ..masking_utils import get_masker, MaskingStrategy


class Q4Evaluator(MultiInstanceEvaluator):
    """Evaluator for Q4: Why instances A and B have different predictions"""

    @property
    def question_type(self) -> int:
        return 4

    @property
    def metric_name(self) -> str:
        return "Gap Reduction"

    @property
    def metric_formula(self) -> str:
        return "1 if (gap_original - gap_modified) > 0 else 0"

    def evaluate_multi(
        self,
        agent_output: Dict[str, Any],
        inputs: List[Any],
        model: nn.Module,
        predictions: List[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q4: Mask identified parts in both inputs and check if gap reduces.

        Args:
            agent_output: Agent output with regions for input_A and input_B
            inputs: [input_A, input_B]
            model: Target model
            predictions: [prediction_A, prediction_B]
            **kwargs: processor, device, etc.

        Returns:
            EvaluationResult with gap reduction score
        """
        try:
            if len(inputs) < 2 or len(predictions) < 2:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Need at least 2 inputs and predictions"]
                )

            # Extract regions for both inputs
            output_data = agent_output.get('output', {})
            region_a = self._extract_instance_region(output_data.get('input_A', {}))
            region_b = self._extract_instance_region(output_data.get('input_B', {}))

            if region_a is None or region_b is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Could not extract regions for both instances"]
                )

            input_a, input_b = inputs[0], inputs[1]
            pred_a, pred_b = predictions[0], predictions[1]

            # Get A's original class (the reference class for comparison)
            class_a1 = pred_a.get('predicted_class_idx', 0)

            # Get original probabilities for class A1 in both instances
            probs_a = pred_a.get('probabilities')
            probs_b = pred_b.get('probabilities')

            if probs_a is None or probs_b is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Probabilities not available for both instances"]
                )

            p_a_a1_orig = float(probs_a[class_a1])
            p_b_a1_orig = float(probs_b[class_a1])
            gap_original = p_a_a1_orig - p_b_a1_orig

            # Mask both inputs (use GRAY for neutral masking)
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            masked_a = masker.mask(
                input_a, region_a,
                dataset_base_name=kwargs.get('dataset_base_name'),
                row_no=kwargs.get('row_no'),
                tool_name=kwargs.get('tool_name'),
                instance_suffix='_A',
                mask_suffix=kwargs.get('mask_suffix', '')
            )
            masked_b = masker.mask(
                input_b, region_b,
                dataset_base_name=kwargs.get('dataset_base_name'),
                row_no=kwargs.get('row_no'),
                tool_name=kwargs.get('tool_name'),
                instance_suffix='_B',
                mask_suffix=kwargs.get('mask_suffix', '')
            )

            # Get predictions on masked inputs
            processor = kwargs.get('processor')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            mod_pred_a = self.get_prediction(model, masked_a, processor, device)
            mod_pred_b = self.get_prediction(model, masked_b, processor, device)

            mod_probs_a = mod_pred_a.get('probabilities')
            mod_probs_b = mod_pred_b.get('probabilities')

            if mod_probs_a is None or mod_probs_b is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Modified probabilities not available"]
                )

            p_a_a1_mod = float(mod_probs_a[class_a1])
            p_b_a1_mod = float(mod_probs_b[class_a1])
            gap_modified = p_a_a1_mod - p_b_a1_mod

            # Calculate metric: gap reduced?
            gap_reduction = gap_original - gap_modified
            passed = gap_reduction > 0
            score = 1.0 if passed else 0.0

            return EvaluationResult(
                score=score,
                passed=passed,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                # For Q4, p_original/p_modified represent the gap values
                p_original=gap_original,
                p_modified=gap_modified,
                original_class=f"class_{class_a1}",
                modified_class=f"class_{class_a1}",
                details={
                    "region_a": region_a,
                    "region_b": region_b,
                    "class_a1": class_a1,
                    "gap_original": gap_original,
                    "gap_modified": gap_modified,
                    "gap_reduction": gap_reduction,
                    "p_a_a1_orig": p_a_a1_orig,
                    "p_b_a1_orig": p_b_a1_orig,
                    "p_a_a1_mod": p_a_a1_mod,
                    "p_b_a1_mod": p_b_a1_mod,
                    "interpretation": "1 = masking reduced the probability gap (good)"
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
        """Extract region from instance-specific data"""
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
