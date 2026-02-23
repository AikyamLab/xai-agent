"""
Q6 Evaluator: How to Flip Prediction

Metric: 1 if R1_modified == expected_class, else 0
Success if applying the change plan actually flips to target class
"""

from typing import Any, Dict, Optional

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
        # Get change plan from agent output.
        # Vision always returns a single dict; text/tabular may return a list of dicts.
        output_data = agent_output.get('output', {})
        change_plan_raw = output_data.get('change_plan', {})

        # Normalise to a uniform list of plan dicts for sequential application.
        if isinstance(change_plan_raw, dict):
            change_plans = [change_plan_raw] if change_plan_raw else []
        elif isinstance(change_plan_raw, list):
            change_plans = [p for p in change_plan_raw if isinstance(p, dict)]
        else:
            change_plans = []

        if not change_plans:
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

        # Apply each plan sequentially.  For text/tabular the agent may propose
        # several feature/span changes; each is applied on top of the previous result.
        modified_input = original_input
        region_ratio = 0.0
        for plan in change_plans:
            region = self._extract_region_from_change_plan(plan)
            if region is None:
                continue
            action = plan.get('action', 'change')
            new_value = plan.get('new_value')
            modified_input = self._apply_change_plan(
                modified_input, region, action, new_value, kwargs
            )
            region_ratio += self.compute_region_ratio(region, original_input)
        region_ratio = min(1.0, region_ratio)

        # Get prediction on modified input
        processor = kwargs.get('processor')
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        modified_prediction = self.get_prediction(model, modified_input, processor, device)
        modified_class = modified_prediction.get('predicted_class_idx', -1)
        modified_probs = modified_prediction.get('probabilities')
        original_probs = original_prediction.get('probabilities')
        original_class_idx = original_prediction.get('predicted_class_idx', 0)

        # expected_class_idx == -1 means "any different class" (e.g. vision Q6 has no
        # specific target class — just "flip to a different class").
        if expected_class_idx == -1:
            # Pass if the predicted class changed at all
            flipped_correctly = (modified_class != original_class_idx)
            # p_original / p_modified track the *original* class probability so we can
            # measure how much the model's confidence in the original class dropped.
            p_original = float(original_probs[original_class_idx]) if original_probs is not None else None
            p_modified = float(modified_probs[original_class_idx]) if modified_probs is not None else None
            # Soft score: probability drop of original class (higher = more convincing flip)
            if p_original is not None and p_modified is not None:
                soft_score = max(0.0, p_original - p_modified)
            else:
                soft_score = 1.0 if flipped_correctly else 0.0
            metric_formula = "1 if R1_modified != original_class else 0"
            interpretation = "score = P_drop(original_class) * (1 - region_ratio)"
        else:
            # Specific target class requested
            flipped_correctly = (modified_class == expected_class_idx)
            # p_original: target class probability BEFORE modification
            p_original = float(original_probs[expected_class_idx]) if original_probs is not None else None
            # Soft score: probability of target class after modification
            if modified_probs is not None:
                p_modified = float(modified_probs[expected_class_idx])
                soft_score = max(0.0, min(1.0, p_modified))
            else:
                p_modified = None
                soft_score = 1.0 if flipped_correctly else 0.0
            metric_formula = self.metric_formula
            interpretation = "score = P(target_class) * (1 - region_ratio)"

        # Size penalty: penalize larger modified regions
        size_penalty = 1.0 - region_ratio
        score = soft_score * size_penalty

        return EvaluationResult(
            score=score,
            passed=flipped_correctly,
            metric_name=self.metric_name,
            metric_formula=metric_formula,
            original_class=str(original_class_idx),
            modified_class=str(modified_class),
            p_original=p_original,
            p_modified=p_modified,
            details={
                "change_plan": change_plan_raw,
                "expected_class": expected_class,
                "expected_class_idx": expected_class_idx,
                "soft_score": soft_score,
                "region_ratio": region_ratio,
                "size_penalty": size_penalty,
                "flipped_correctly": flipped_correctly,
                "interpretation": interpretation
            }
        )

    def _apply_change_plan(self, original_input, region, action, new_value, kwargs) -> Any:
        """Apply the agent's change plan to produce a modified input.

        Vision  : SD inpainting with new_value as prompt (falls back to gray fill)
        Text    : Replace span with new_value for 'change'; delete for 'delete'
        Tabular : Set feature to new_value float; fall back to mean-fill if unparseable
        """
        mask_kwargs = dict(
            dataset_base_name=kwargs.get('dataset_base_name'),
            row_no=kwargs.get('row_no'),
            tool_name=kwargs.get('tool_name'),
            mask_suffix=kwargs.get('mask_suffix', ''),
            feature_names=kwargs.get('feature_names', []),
        )

        if self.modality == 'vision':
            # Use SD inpainting when a text prompt is provided
            if new_value:
                try:
                    from ..sd_inpainting import generate_counterfactual_image
                    bbox = region.get('bounding_box')
                    prompt = self._clean_sd_prompt(str(new_value))
                    result = generate_counterfactual_image(original_input, bbox, prompt)
                    # SD inpainting bypasses masker.mask(), so save explicitly
                    masker = get_masker(self.modality, MaskingStrategy.GRAY)
                    masker.save_from_kwargs(result, **mask_kwargs)
                    return result
                except Exception as e:
                    print(f"[Q6] SD inpainting failed ({e}), falling back to gray fill")
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            return masker.mask(original_input, region, **mask_kwargs)

        elif self.modality == 'text':
            # For 'change': replace span with new_value
            # For 'delete' (or any other action): remove span
            if action == 'change' and new_value:
                result = self._replace_text_span(original_input, region, str(new_value))
                # Text replacement bypasses masker.mask(), so save explicitly
                masker = get_masker(self.modality, MaskingStrategy.DELETE)
                masker.save_from_kwargs(result, **mask_kwargs)
                return result
            else:
                masker = get_masker(self.modality, MaskingStrategy.DELETE)
                return masker.mask(original_input, region, **mask_kwargs)

        else:  # tabular
            # Set the feature to the agent-provided new_value
            if new_value is not None:
                modified = self._set_tabular_feature(
                    original_input, region['feature_key'], new_value,
                    kwargs.get('feature_names', [])
                )
                if modified is not None:
                    # Direct feature-set bypasses masker.mask(), so save explicitly
                    masker = get_masker(self.modality, MaskingStrategy.GRAY)
                    masker.save_from_kwargs(modified, **mask_kwargs)
                    return modified
            # Fallback: mean-fill
            masker = get_masker(self.modality, MaskingStrategy.GRAY)
            return masker.mask(original_input, region, **mask_kwargs)

    def _replace_text_span(self, original_input, region, new_value: str):
        """Replace text[start:end] with new_value, handling plain text and NLI dicts."""
        start = region.get('start_index', 0)
        end = region.get('end_index', 0)
        if isinstance(original_input, dict) and 'premise' in original_input:
            text = original_input['premise']
            new_text = text[:start] + new_value + text[end:]
            result = original_input.copy()
            result['premise'] = new_text
            return result
        elif isinstance(original_input, str):
            return original_input[:start] + new_value + original_input[end:]
        # Unknown text format — return unchanged
        return original_input

    def _set_tabular_feature(self, original_input, feature_key: str, new_value, feature_names):
        """Set a tabular feature to new_value.

        Supports two modes:
        - Numeric: new_value is castable to float; feature_key is a column name/index.
        - Categorical (one-hot fallback): new_value is a string category and feature_key is the
          raw pre-encoding column name (e.g. feature_key='occupation', new_value='Exec-managerial').
          Looks up '{feature_key}_{new_value}' in feature_names, sets it to 1, and resets all
          other '{feature_key}_*' sibling columns to 0.
        Returns None if the feature cannot be resolved.
        """
        import torch
        import numpy as np

        feature_names_list = list(feature_names) if feature_names else []

        # --- Try numeric conversion ---
        try:
            val = float(str(new_value))
            is_numeric = True
        except (ValueError, TypeError):
            val = None
            is_numeric = False

        # --- Categorical one-hot fallback ---
        # Triggered when new_value is a string (e.g. 'Exec-managerial') and the raw column
        # name (e.g. 'occupation') is not directly in the post-encoding feature_names list.
        if not is_numeric and feature_names_list:
            # Try both underscore and space as separator (datasets vary)
            target_col = f"{feature_key}_{new_value}"
            alt_target = f"{feature_key.replace('_', ' ')}_{new_value}" if '_' in feature_key else target_col
            target_col = target_col if target_col in feature_names_list else alt_target
            if target_col in feature_names_list:
                target_idx = feature_names_list.index(target_col)
                prefix = f"{feature_key}_"
                sibling_indices = [i for i, n in enumerate(feature_names_list) if n.startswith(prefix)]

                if isinstance(original_input, torch.Tensor):
                    result = original_input.clone()
                    for idx in sibling_indices:
                        result[idx] = 0.0
                    result[target_idx] = 1.0
                    return result

                if isinstance(original_input, np.ndarray):
                    result = original_input.copy()
                    for idx in sibling_indices:
                        result[idx] = 0.0
                    result[target_idx] = 1.0
                    return result
            # String new_value but no matching one-hot column found
            return None

        if not is_numeric:
            return None

        # --- Numeric path ---
        if isinstance(original_input, dict):
            result = original_input.copy()
            if feature_key in result:
                result[feature_key] = val
                return result
            return None

        if isinstance(original_input, torch.Tensor):
            try:
                col_idx = int(feature_key)
            except ValueError:
                col_idx = self._resolve_feature_col(feature_key, feature_names_list)
                if col_idx is None:
                    return None
            result = original_input.clone()
            result[col_idx] = torch.tensor(val, dtype=result.dtype)
            return result

        if isinstance(original_input, np.ndarray):
            try:
                col_idx = int(feature_key)
            except ValueError:
                col_idx = self._resolve_feature_col(feature_key, feature_names_list)
                if col_idx is None:
                    return None
            result = original_input.copy()
            result[col_idx] = val
            return result

        return None

    @staticmethod
    def _resolve_feature_col(feature_key: str, feature_names_list: list):
        """Return column index for feature_key, trying space↔underscore normalisation.

        Returns None if no match is found.
        """
        if feature_key in feature_names_list:
            return feature_names_list.index(feature_key)
        # Try swapping underscores ↔ spaces (e.g. 'mean_compactness' vs 'mean compactness')
        alt_key = feature_key.replace('_', ' ') if '_' in feature_key else feature_key.replace(' ', '_')
        if alt_key in feature_names_list:
            return feature_names_list.index(alt_key)
        return None

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
            # Try to parse as integer
            try:
                return int(class_ref)
            except ValueError:
                pass
        return -1

    @staticmethod
    def _normalize_label(s: str) -> str:
        """Expand comparison symbols to words for fuzzy label matching."""
        s = s.lower().strip()
        # Order matters: longest symbols first
        for sym, word in [('<=', 'less than or equal to '), ('>=', 'greater than or equal to '),
                           ('<', 'less than '), ('>', 'greater than ')]:
            s = s.replace(sym, word)
        return ' '.join(s.split())

    @staticmethod
    def _clean_sd_prompt(prompt: str) -> str:
        """Clean and normalize a text prompt for Stable Diffusion inpainting.

        Removes leading/trailing whitespace, collapses internal whitespace,
        strips surrounding quotes, and truncates to 200 characters to stay
        within SD token limits.
        """
        import re
        prompt = prompt.strip().strip('"\'')
        prompt = re.sub(r'\s+', ' ', prompt)
        return prompt[:200]
