"""
Q7 Evaluator: Predict Change Outcome

Metric: 1 if R1_modified == agent_predicted_class, else 0
Success if agent correctly predicted the new class after modification
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_evaluator import BaseEvaluator, EvaluationResult
from ..masking_utils import get_masker


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
        # Get agent's predicted class after change
        output_data = agent_output.get('output', {})
        agent_predicted_class = output_data.get('changed_class')

        if agent_predicted_class is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Agent did not provide changed_class prediction"]
            )

        # Extract original class probabilities
        original_probs = original_prediction.get('probabilities')
        original_class = original_prediction.get('predicted_class_idx', 0)
        if original_probs is not None:
            if isinstance(original_probs, dict):
                original_probs = [original_probs[k] for k in sorted(original_probs.keys())]
            p_original = float(original_probs[original_class])
        else:
            p_original = None

        # Get the part to change (should be specified in context)
        part_to_change = kwargs.get('part_to_change')
        if part_to_change is None:
            # Use the most important region from original analysis
            part_to_change = self._get_important_region(original_prediction, kwargs)

        # For vision Q7, the agent identifies the region itself and returns masked_region.
        # Use it as fallback when no pre-specified region is available.
        if part_to_change is None and self.modality == 'vision':
            masked_region = output_data.get('masked_region', {})
            if isinstance(masked_region, dict) and masked_region.get('bounding_box'):
                err = self.validate_region(masked_region, original_input, **kwargs)
                if err:
                    return EvaluationResult(
                        score=0.0,
                        passed=False,
                        metric_name=self.metric_name,
                        metric_formula=self.metric_formula,
                        errors=[f"Agent masked_region invalid: {err}"]
                    )
                part_to_change = masked_region

        if part_to_change is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Part to change not specified"]
            )

        # Mask using modality-appropriate default strategy
        masker = get_masker(self.modality, preprocessor=kwargs.get('processor'))
        modified_input = masker.mask(
            original_input, part_to_change,
            dataset_base_name=kwargs.get('dataset_base_name'),
            row_no=kwargs.get('row_no'),
            tool_name=kwargs.get('tool_name'),
            mask_suffix=kwargs.get('mask_suffix', ''),
            feature_names=kwargs.get('feature_names', []),
            original_features=kwargs.get('original_features', {})
        )

        # Get actual prediction after modification
        processor = kwargs.get('processor')
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        modified_prediction = self.get_prediction(model, modified_input, processor, device)
        actual_modified_class = modified_prediction.get('predicted_class_idx', -1)
        modified_probs = modified_prediction.get('probabilities')
        p_modified = float(modified_probs[original_class]) if modified_probs is not None else None

        # Convert agent's prediction to index for comparison
        class_names = kwargs.get('class_names')
        agent_predicted_idx = self._get_class_index(agent_predicted_class, class_names)

        # Passed logic unchanged: binary correct/wrong
        correct = (actual_modified_class == agent_predicted_idx)

        # Soft score: probability of agent's predicted class in modified output
        if modified_probs is not None and agent_predicted_idx >= 0:
            p_agent_predicted = float(modified_probs[agent_predicted_idx])
            score = max(0.0, min(1.0, p_agent_predicted))
        else:
            p_agent_predicted = None
            score = 1.0 if correct else 0.0

        return EvaluationResult(
            score=score,
            passed=correct,
            metric_name=self.metric_name,
            metric_formula=self.metric_formula,
            original_class=str(original_prediction.get('predicted_class_idx')),
            modified_class=str(actual_modified_class),
            p_original=p_original,
            p_modified=p_modified,
            details={
                "part_to_change": part_to_change,
                "agent_predicted_class": agent_predicted_class,
                "agent_predicted_idx": agent_predicted_idx,
                "actual_modified_class": actual_modified_class,
                "p_agent_predicted_in_modified": p_agent_predicted,
                "binary_correct": correct,
                "interpretation": "Soft score: P(agent_predicted_class) in modified output, range [0,1]"
            }
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

    def _get_class_index(self, class_ref: Any, class_names=None) -> int:
        """Convert class reference to index.

        Handles both list and dict label_map formats.
        dict format: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        list format: ['negative', 'positive']
        Falls back to symbol-normalized comparison so "greater than 50K" matches ">50K".
        """
        if isinstance(class_ref, int):
            return class_ref
        if isinstance(class_ref, str):
            if class_names:
                items = class_names.items() if isinstance(class_names, dict) else enumerate(class_names)
                # Pass 1: exact case-insensitive match
                for idx, name in items:
                    if str(name).lower() == class_ref.lower():
                        return int(idx)
                # Pass 2: symbol-normalized match (">50K" == "greater than 50K")
                norm_ref = self._normalize_label(class_ref)
                items = class_names.items() if isinstance(class_names, dict) else enumerate(class_names)
                for idx, name in items:
                    if self._normalize_label(str(name)) == norm_ref:
                        return int(idx)
            try:
                return int(class_ref)
            except ValueError:
                pass
        return -1

    @staticmethod
    def _normalize_label(s: str) -> str:
        """Expand comparison symbols to words for fuzzy label matching."""
        s = s.lower().strip()
        for sym, word in [('<=', 'less than or equal to '), ('>=', 'greater than or equal to '),
                           ('<', 'less than '), ('>', 'greater than ')]:
            s = s.replace(sym, word)
        return ' '.join(s.split())
