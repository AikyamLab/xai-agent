"""
Q1 Evaluator: Most Responsible Part

Metric: P_original - P_modified
Higher score = better (the identified part was indeed important)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker


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
            if original_probs is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metric_name=self.metric_name,
                    metric_formula=self.metric_formula,
                    errors=["Original probabilities not available"]
                )
            # Handle dict probabilities (e.g. {"negative": 0.9, "positive": 0.1})
            if isinstance(original_probs, dict):
                original_probs = [original_probs[k] for k in sorted(original_probs.keys())]
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

            # Get prediction on masked input
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

            # Calculate metric: max(0, P_original - P_modified)
            # Clamp negative values (masking increased prob = bad explanation = 0)
            raw_drop = p_original - p_modified
            soft_score = max(0.0, raw_drop)

            # Size penalty: penalize large masked regions
            region_ratio = self.compute_region_ratio(region, original_input)
            size_penalty = 1.0 - region_ratio
            score = soft_score * size_penalty

            # Check if class changed after masking
            modified_class = modified_prediction.get('predicted_class_idx')
            class_changed = modified_class != original_class

            # Determine if passed:
            # 1. Significant probability drop (> 0.5), OR
            # 2. Class changed (strong evidence that the masked region was important)
            passed = (raw_drop > 0.5) or class_changed

            return EvaluationResult(
                score=score,
                passed=passed,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                p_original=p_original,
                p_modified=p_modified,
                original_class=str(original_class),
                modified_class=str(modified_class),
                details={
                    "region": region,
                    "raw_drop": raw_drop,
                    "soft_score": soft_score,
                    "region_ratio": region_ratio,
                    "size_penalty": size_penalty,
                    "threshold": 0.5,
                    "class_changed": class_changed,
                    "interpretation": "score = max(0, P_orig - P_mod) * (1 - region_ratio)"
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
