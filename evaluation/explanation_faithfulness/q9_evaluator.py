"""
Q9 Evaluator: Shared Feature in Misclassified Inputs

Metric: 1 if ALL inputs improve (P_correct_modified > P_correct_original for all), else 0
Success if removing shared feature improves all predictions
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..base_evaluator import MultiInstanceEvaluator, EvaluationResult
from ..masking_utils import get_masker


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

            # New list format: {"instances": [{...}, {...}, ...]}
            instances_list = output_data.get('instances')
            if isinstance(instances_list, list):
                for i, inst_data in enumerate(instances_list):
                    key = f"input_{chr(ord('A') + i)}"
                    regions[key] = self._extract_instance_region(inst_data)
            else:
                # Legacy named-key format: input_A, input_B, input_C, ...
                for key in output_data:
                    if key.startswith('input_'):
                        regions[key] = self._extract_instance_region(output_data[key])

            if len(regions) < len(inputs):
                # Q9 identifies a SHARED feature across all inputs.  If the model
                # provided fewer instance entries than inputs (e.g. schema defaulted
                # to 2 while 3 inputs exist), broadcast the last available region to
                # fill the gap rather than failing outright.
                if len(regions) >= 1:
                    last_region = regions[f"input_{chr(ord('A') + len(regions) - 1)}"]
                    for i in range(len(regions), len(inputs)):
                        regions[f"input_{chr(ord('A') + i)}"] = last_region
                else:
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

                err = self.validate_region(region, input_data, **kwargs)
                if err:
                    improvements.append(False)
                    details_per_input.append({"error": f"Bad agent region: {err}"})
                    continue

                # Get correct class for this input
                correct_class_idx = self._get_class_index(ground_truths[i], class_names)

                # Get original probability for correct class
                original_probs = pred.get('probabilities')
                if original_probs is None:
                    improvements.append(False)
                    details_per_input.append({"error": "Probabilities not available"})
                    continue

                original_probs = self._normalize_probs(original_probs)
                p_correct_original = float(original_probs[correct_class_idx])

                # Mask and get new prediction (use GRAY for neutral masking)
                # Use instance suffix for multi-instance saving (e.g., _A, _B, _C...)
                instance_suffix = f"_{chr(ord('A') + i)}"
                masker = get_masker(self.modality, preprocessor=kwargs.get('processor'))
                # For multi-instance Q9, original_features may be a list; pick per-instance
                raw_orig = kwargs.get('original_features', {})
                per_inst_orig = raw_orig[i] if isinstance(raw_orig, list) and i < len(raw_orig) else raw_orig
                masked_input = masker.mask(
                    input_data, region,
                    dataset_base_name=kwargs.get('dataset_base_name'),
                    row_no=kwargs.get('row_no'),
                    tool_name=kwargs.get('tool_name'),
                    instance_suffix=instance_suffix,
                    mask_suffix=kwargs.get('mask_suffix', ''),
                    feature_names=kwargs.get('feature_names', []),
                    original_features=per_inst_orig
                )
                modified_pred = self.get_prediction(model, masked_input, processor, device)
                modified_probs = self._normalize_probs(modified_pred.get('probabilities'))

                if modified_probs is None:
                    improvements.append(False)
                    details_per_input.append({"error": "Modified probabilities not available"})
                    continue

                p_correct_modified = float(modified_probs[correct_class_idx])

                improved = p_correct_modified > p_correct_original
                improvements.append(improved)

                # Per-input normalized improvement
                improvement_amount = p_correct_modified - p_correct_original
                room = 1.0 - p_correct_original
                if room > 1e-8:
                    norm_improvement = max(0.0, min(1.0, improvement_amount / room))
                else:
                    norm_improvement = 1.0 if improved else 0.0

                # Per-input region ratio
                input_region_ratio = self.compute_region_ratio(region, input_data)

                details_per_input.append({
                    "input_key": input_key,
                    "correct_class": ground_truths[i],
                    "p_correct_original": p_correct_original,
                    "p_correct_modified": p_correct_modified,
                    "improvement": improvement_amount,
                    "room_for_improvement": room,
                    "normalized_improvement": norm_improvement,
                    "region_ratio": input_region_ratio,
                    "improved": improved
                })

            # Passed logic unchanged: all must improve
            all_improved = all(improvements)

            # Soft score: mean normalized improvement across all instances
            norm_scores = [d["normalized_improvement"] for d in details_per_input if "normalized_improvement" in d]
            soft_score = sum(norm_scores) / len(norm_scores) if norm_scores else 0.0

            # Size penalty: average region ratio across all instances
            region_ratios = [d["region_ratio"] for d in details_per_input if "region_ratio" in d]
            region_ratio = sum(region_ratios) / len(region_ratios) if region_ratios else 0.0
            size_penalty = 1.0 - region_ratio
            score = soft_score * size_penalty

            return EvaluationResult(
                score=score,
                passed=all_improved,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                details={
                    "num_inputs": len(inputs),
                    "num_improved": sum(improvements),
                    "all_improved": all_improved,
                    "soft_score": soft_score,
                    "region_ratio": region_ratio,
                    "size_penalty": size_penalty,
                    "shared_feature_description": agent_output.get('shared_feature_description', ''),
                    "per_input_details": details_per_input,
                    "interpretation": "score = mean_normalized_improvement * (1 - avg_region_ratio)"
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
            # New multi-span format
            spans = instance_data.get('spans')
            if spans and isinstance(spans, list):
                return {"spans": spans}
            # Legacy single-span
            start = instance_data.get('start_index')
            end = instance_data.get('end_index')
            if start is not None and end is not None:
                return {"spans": [{"start_index": start, "end_index": end}]}
            return None
        else:
            # New multi-key format
            keys = instance_data.get('feature_keys')
            if keys and isinstance(keys, list):
                return {"feature_keys": keys}
            # Legacy single-key
            key = instance_data.get('feature_key')
            return {"feature_keys": [key]} if key else None

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
