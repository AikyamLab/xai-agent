"""
Q3 Evaluator: Distinctive Parts (Top-1 vs Top-2)

Metric: 1 if (P_top2_modified > P_top1_modified), else 0
Success if masking causes rank flip between top-1 and top-2
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker


class Q3Evaluator(BaseEvaluator):
    """Evaluator for Q3: Parts distinguishing from next-best alternative"""

    @property
    def question_type(self) -> int:
        return 3

    @property
    def metric_name(self) -> str:
        return "Rank Flip"

    @property
    def metric_formula(self) -> str:
        return "1 if (P_top2_modified > P_top1_modified) else 0"

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q3: Check if masking causes top-1 and top-2 to swap.

        Success means the identified part was truly decisive for choosing top-1 over top-2.

        Args:
            agent_output: Agent output with identified region
            original_input: Original input data
            model: Target model
            original_prediction: Original prediction with top-k classes
            **kwargs: processor, device, etc.

        Returns:
            EvaluationResult with rank flip score (1 or 0)
        """
        region = self.extract_region(agent_output)
        if region is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                errors=["Could not extract region from agent output"]
            )

        err = self.validate_region(region, original_input, **kwargs)
        if err:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                errors=[err]
            )

        # Get original top-1 and top-2 classes
        original_probs = original_prediction.get('probabilities')
        if original_probs is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Original probabilities not available"]
            )

        # Get top-2 classes
        original_probs = self._normalize_probs(original_probs)
        if hasattr(original_probs, 'argsort'):
            sorted_indices = original_probs.argsort()[::-1]
        else:
            import numpy as np
            sorted_indices = np.argsort(original_probs)[::-1]

        top1_class = int(sorted_indices[0])
        top2_class = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_class
        p_top1_original = float(original_probs[top1_class])
        p_top2_original = float(original_probs[top2_class])

        # Mask using modality-appropriate default strategy
        masker = get_masker(self.modality, preprocessor=kwargs.get('processor'), feature_modes=kwargs.get('feature_modes'))
        masked_input = masker.mask(
            original_input, region,
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
        modified_probs = self._normalize_probs(modified_prediction.get('probabilities'))

        if modified_probs is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Modified probabilities not available"]
            )

        # Probabilities of original top1/top2 classes after masking
        p_top1_modified = float(modified_probs[top1_class])
        p_top2_modified = float(modified_probs[top2_class])

        # Monitor current top-1 and top-2 after masking (irrespective of original)
        if hasattr(modified_probs, 'argsort'):
            modified_sorted_indices = modified_probs.argsort()[::-1]
        else:
            import numpy as np
            modified_sorted_indices = np.argsort(modified_probs)[::-1]

        modified_top1_class = int(modified_sorted_indices[0])
        modified_top2_class = int(modified_sorted_indices[1]) if len(modified_sorted_indices) > 1 else modified_top1_class
        modified_top1_prob = float(modified_probs[modified_top1_class])
        modified_top2_prob = float(modified_probs[modified_top2_class])

        # Check if rank flipped: top-2 now has higher probability than top-1
        rank_flipped = p_top2_modified > p_top1_modified

        # Soft score: how much of the original gap was closed/reversed
        # gap_orig > 0 (top1 > top2), gap_mod can be negative (flipped)
        gap_orig = p_top1_original - p_top2_original
        gap_mod = p_top1_modified - p_top2_modified
        if gap_orig > 1e-8:
            # 0 = no change, 0.5 = tied, 1.0 = fully flipped to same magnitude
            soft_score = max(0.0, min(1.0, (gap_orig - gap_mod) / (2.0 * gap_orig)))
        else:
            # Original classes already tied; any flip is perfect
            soft_score = 1.0 if rank_flipped else 0.0

        # Size penalty: penalize large masked regions
        region_ratio = self.compute_region_ratio(region, original_input)
        size_penalty = 1.0 - region_ratio
        size_score = soft_score * size_penalty
        score = soft_score

        # Passed logic unchanged: binary rank flip
        passed = rank_flipped

        return EvaluationResult(
            score=score,
            passed=passed,
            metric_name=self.metric_name,
            metric_formula=self.metric_formula,
            p_original=p_top1_original,
            p_modified=p_top1_modified,
            original_class=str(top1_class),
            modified_class=str(modified_top1_class),
            details={
                "region": region,
                # Original top-1 and top-2
                "original_top1_class": top1_class,
                "original_top2_class": top2_class,
                "original_top1_prob": p_top1_original,
                "original_top2_prob": p_top2_original,
                # Probabilities of original classes after masking
                "original_top1_prob_after_mask": p_top1_modified,
                "original_top2_prob_after_mask": p_top2_modified,
                # Current top-1 and top-2 after masking (may be different classes)
                "modified_top1_class": modified_top1_class,
                "modified_top2_class": modified_top2_class,
                "modified_top1_prob": modified_top1_prob,
                "modified_top2_prob": modified_top2_prob,
                # Analysis
                "gap_original": gap_orig,
                "gap_modified": gap_mod,
                "soft_score": soft_score,
                "size_score": size_score,
                "size_score_l1": self._size_score_l1(soft_score, region_ratio, penalize_large=True),
                "region_ratio": region_ratio,
                "size_penalty": size_penalty,
                "top1_class_changed": modified_top1_class != top1_class,
                "top2_class_changed": modified_top2_class != top2_class,
                "rank_flipped": rank_flipped,
                "interpretation": "score = gap_reduction_ratio; size_score = score * (1 - region_ratio)"
            }
        )
