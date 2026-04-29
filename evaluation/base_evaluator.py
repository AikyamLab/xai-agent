"""
Base Evaluator classes for XAI Agent Framework

Defines the abstract interface for explanation faithfulness evaluation.
"""

import html as _html_module
import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class EvaluationResult:
    """
    Result of an explanation faithfulness evaluation.
    """
    # Core metrics
    score: float  # The primary faithfulness metric
    passed: bool  # Binary success indicator

    # Metric details
    metric_name: str = ""
    metric_formula: str = ""

    # Intermediate values for debugging
    p_original: Optional[float] = None  # Original probability
    p_modified: Optional[float] = None  # Probability after modification
    original_class: Optional[str] = None
    modified_class: Optional[str] = None

    # Additional details
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "score": self.score,
            "passed": self.passed,
            "metric_name": self.metric_name,
            "metric_formula": self.metric_formula,
            "p_original": self.p_original,
            "p_modified": self.p_modified,
            "original_class": self.original_class,
            "modified_class": self.modified_class,
            "details": self.details,
            "errors": self.errors
        }


class BaseEvaluator(ABC):
    """
    Abstract base class for explanation faithfulness evaluators.

    Each question type (Q1-Q10) has its own evaluator that:
    1. Parses the agent's output to extract identified regions/features
    2. Applies masking or modification to the input
    3. Runs the model on original and modified inputs
    4. Computes the faithfulness metric
    """

    _REFUSAL_MARKERS: frozenset = frozenset({
        'unknown', 'cannot', "can't", "n/a", 'none', 'not available',
        'refuse', "i don't", "i can't", 'unsure', 'unclear'
    })

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize typographic/encoding variants so span matching is robust.

        Applied to both the span and the source text before any comparison so
        that all existing fallbacks also benefit.  Only converts losslessly
        equivalent characters; will not introduce false positives.

        Handles:
          - HTML entities: &amp; &#39; &quot; &nbsp; …
          - Smart/curly quotes → straight: \u201c\u201d → " , \u2018\u2019 → '
          - Em/en dash → hyphen: \u2014\u2013 → -
          - Ellipsis character → three dots: \u2026 → ...
          - Non-breaking space → regular space: \u00a0 → space
          - Zero-width chars removed: \u200b \ufeff
        """
        # HTML entities first (may produce quote/amp/etc that we then normalise)
        text = _html_module.unescape(text)
        # Curly/smart quotes → straight
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2032', "'")   # prime ′
        # Dashes
        text = text.replace('\u2014', '-').replace('\u2013', '-')
        # Ellipsis
        text = text.replace('\u2026', '...')
        # Non-breaking space
        text = text.replace('\u00a0', ' ')
        # Zero-width chars
        text = text.replace('\u200b', '').replace('\ufeff', '')
        return text

    def __init__(self, modality: str = "vision", size_lambda: float = 0.0):
        """
        Initialize evaluator.

        Args:
            modality: Data modality ("vision", "text", "tabular")
            size_lambda: Weight for L1 size penalty. 0.0 (default) disables it.
        """
        self.modality = modality
        self.size_lambda = size_lambda

    @property
    @abstractmethod
    def question_type(self) -> int:
        """Return the question type this evaluator handles (1-10)"""
        pass

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the name of the evaluation metric"""
        pass

    @property
    @abstractmethod
    def metric_formula(self) -> str:
        """Return the formula for the evaluation metric"""
        pass

    @abstractmethod
    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate the faithfulness of an agent's explanation.

        Args:
            agent_output: Parsed output from the agent (must match output schema)
            original_input: Original input data (image tensor, text, or tabular data)
            model: The target model being explained
            original_prediction: Original prediction results
            **kwargs: Additional arguments (e.g., processor, tokenizer, ground_truth)

        Returns:
            EvaluationResult with score, passed status, and details
        """
        pass

    def _size_score_l1(self, soft_score: float, region_ratio: float, penalize_large: bool = True) -> float:
        """
        Compute size-penalized score using L1 loss.

        size_score_l1 = max(0, soft_score - size_lambda * region_ratio)      (penalize_large=True)
        size_score_l1 = max(0, soft_score - size_lambda * (1 - region_ratio)) (penalize_large=False)
        """
        if self.size_lambda == 0.0:
            return soft_score
        penalty = region_ratio if penalize_large else (1.0 - region_ratio)
        return max(0.0, soft_score - self.size_lambda * penalty)

    def compute_region_ratio(
        self,
        region: Dict[str, Any],
        original_input: Any
    ) -> float:
        """
        Compute the fraction of the input covered by the masked region.

        Returns:
            Float in [0, 1]. 0 = nothing masked, 1 = entire input masked.
        """
        try:
            if self.modality == "vision":
                bbox = region.get("bounding_box")
                if bbox is None:
                    return 0.0
                x_min, y_min, x_max, y_max = bbox
                region_area = max(0, x_max - x_min) * max(0, y_max - y_min)
                # Determine image dimensions
                if PIL_AVAILABLE and isinstance(original_input, Image.Image):
                    w, h = original_input.size
                elif isinstance(original_input, torch.Tensor):
                    if original_input.dim() == 4:
                        h, w = original_input.shape[2], original_input.shape[3]
                    elif original_input.dim() == 3:
                        h, w = original_input.shape[1], original_input.shape[2]
                    else:
                        return 0.0
                elif isinstance(original_input, np.ndarray):
                    h, w = original_input.shape[:2]
                else:
                    return 0.0
                image_area = w * h
                return region_area / image_area if image_area > 0 else 0.0

            elif self.modality == "text":
                # Determine text
                if isinstance(original_input, dict) and 'premise' in original_input:
                    text = original_input['premise']
                elif isinstance(original_input, str):
                    text = original_input
                else:
                    return 0.0
                text_len = len(text)
                if text_len == 0:
                    return 0.0
                text_lower = text.lower()
                # Count characters covered by all occurrences of each text_span
                text_spans = region.get("text_spans")
                span_text = region.get("span_text")  # Q6 single-span
                if text_spans or span_text:
                    spans_list = text_spans if text_spans else ([span_text] if span_text else [])
                    masked_chars = 0
                    for span in spans_list:
                        if not isinstance(span, str) or not span:
                            continue
                        span_lower = span.lower()
                        span_len = len(span)
                        idx = 0
                        while True:
                            pos = text_lower.find(span_lower, idx)
                            if pos == -1:
                                break
                            masked_chars += span_len
                            idx = pos + span_len
                    return min(1.0, masked_chars / text_len)
                # Legacy index-based spans (used by Q5/Q7 internal regions)
                legacy_spans = region.get("spans")
                if legacy_spans and isinstance(legacy_spans, list):
                    total_len = sum(
                        max(0, s.get("end_index", 0) - s.get("start_index", 0))
                        for s in legacy_spans
                    )
                    return min(1.0, total_len / text_len)
                start = region.get("start_index", 0)
                end = region.get("end_index", 0)
                span_len = max(0, end - start)
                return span_len / text_len

            elif self.modality == "tabular":
                if isinstance(original_input, dict):
                    total = len(original_input)
                elif isinstance(original_input, torch.Tensor):
                    total = original_input.shape[-1]
                elif isinstance(original_input, np.ndarray):
                    total = original_input.shape[-1]
                else:
                    return 0.0
                # Multi-key format
                keys = region.get("feature_keys")
                if keys and isinstance(keys, list):
                    return min(1.0, len(keys) / total) if total > 0 else 0.0
                # Legacy single-key
                return 1.0 / total if total > 0 else 0.0

        except Exception:
            return 0.0
        return 0.0

    def extract_region(self, agent_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the identified region from agent output.

        Args:
            agent_output: Agent output dictionary

        Returns:
            Region dictionary with modality-specific fields, or None
        """
        output_data = agent_output.get("output", {})

        if self.modality == "vision":
            bbox = output_data.get("bounding_box")
            if bbox:
                return {"bounding_box": bbox}
        elif self.modality == "text":
            text_spans = output_data.get("text_spans")
            if text_spans and isinstance(text_spans, list):
                return {"text_spans": text_spans}
        elif self.modality == "tabular":
            # New multi-key format
            keys = output_data.get("feature_keys")
            if keys and isinstance(keys, list):
                return {"feature_keys": keys}
            # Legacy single-key format
            key = output_data.get("feature_key")
            if key:
                return {"feature_keys": [key]}

        return None

    def get_prediction(
        self,
        model: nn.Module,
        input_data: Any,
        processor: Any = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Run model prediction on input data.

        Args:
            model: PyTorch model
            input_data: Input tensor or data (image tensor, text string, dict for NLI, etc.)
            processor: Optional preprocessor (image transform, tokenizer, etc.)
            device: Device to run on

        Returns:
            Prediction dictionary with class and probabilities
        """
        model.eval()
        with torch.no_grad():
            # Handle NLI dict input: {'premise': '...', 'hypothesis': '...'}
            if isinstance(input_data, dict) and 'premise' in input_data and 'hypothesis' in input_data:
                if processor is None:
                    raise ValueError("Processor (tokenizer) required for NLI text input")
                n_forward_params = len(inspect.signature(model.forward).parameters)
                if n_forward_params >= 2:
                    # Dual-input model (e.g. CNN_SNLI): tokenize premise and hypothesis separately
                    premise_ids = processor(input_data['premise'])
                    hypothesis_ids = processor(input_data['hypothesis'])
                    premise_tensor = torch.tensor([premise_ids], dtype=torch.long).to(device)
                    hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long).to(device)
                    outputs = model(premise_tensor, hypothesis_tensor)
                else:
                    # Single-input model (e.g. TwoLayerNN_SNLI): tokenize as combined sequence
                    # trained on raw concatenation without "Premise:"/"Hypothesis:" prefix labels
                    combined = f"{input_data['premise']} {input_data['hypothesis']}"
                    combined_ids = processor(combined)
                    input_tensor = torch.tensor([combined_ids], dtype=torch.long).to(device)
                    outputs = model(input_tensor)
            else:
                # If input is already a tensor, use directly (skip processor)
                # This handles tabular data where features are pre-processed tensors
                if isinstance(input_data, torch.Tensor):
                    input_tensor = input_data
                elif processor is not None:
                    input_tensor = processor(input_data)
                else:
                    raise ValueError("Must provide processor or tensor input")

                # Convert list (e.g. from tokenizer) to tensor
                if isinstance(input_tensor, list):
                    input_tensor = torch.tensor([input_tensor], dtype=torch.long)

                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                elif input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)

                input_tensor = input_tensor.to(device)
                if input_tensor.dtype not in (torch.long, torch.int):
                    input_tensor = input_tensor.float()
                outputs = model(input_tensor)

            # Handle binary classification (single logit output)
            if outputs.shape[-1] == 1:
                # Check if model already applies sigmoid (by inspecting final layers)
                has_sigmoid = self._model_has_final_sigmoid(model)
                if has_sigmoid:
                    prob_pos = outputs.item()
                else:
                    prob_pos = torch.sigmoid(outputs).item()
                probs = torch.tensor([1.0 - prob_pos, prob_pos])
            else:
                probs = torch.softmax(outputs, dim=-1).squeeze()

            top_prob, top_class = probs.max(dim=-1)

            return {
                "predicted_class_idx": top_class.item(),
                "confidence": top_prob.item(),
                "probabilities": probs.cpu().numpy(),
                "logits": outputs.squeeze().cpu().numpy()
            }

    @staticmethod
    def _model_has_final_sigmoid(model: nn.Module) -> bool:
        """Check if model applies sigmoid/softmax as its final activation."""
        modules = list(model.modules())
        for m in reversed(modules):
            if isinstance(m, (nn.Sigmoid, nn.Softmax)):
                return True
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                return False
        return False

    @staticmethod
    def _normalize_probs(probs):
        """Convert dict probabilities to list, sorted by key for consistent indexing."""
        if isinstance(probs, dict):
            return [probs[k] for k in sorted(probs.keys())]
        return probs

    # ------------------------------------------------------------------
    # Agent-output validation helpers
    # ------------------------------------------------------------------

    def _get_image_dims(self, input_data: Any) -> Tuple[int, int]:
        """Return (width, height) from an image input, or (0, 0) if unknown."""
        if PIL_AVAILABLE and isinstance(input_data, Image.Image):
            return input_data.size  # (width, height)
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 4:
                return input_data.shape[3], input_data.shape[2]
            if input_data.dim() == 3:
                return input_data.shape[2], input_data.shape[1]
        if isinstance(input_data, np.ndarray) and input_data.ndim >= 2:
            return input_data.shape[1], input_data.shape[0]
        return 0, 0

    def _get_text_len(self, input_data: Any) -> int:
        """Return character length of text input."""
        if isinstance(input_data, str):
            return len(input_data)
        if isinstance(input_data, dict):
            return (len(input_data.get('premise', ''))
                    + len(input_data.get('hypothesis', ''))
                    + len(input_data.get('text', '')))
        return 0

    def validate_region(self, region: Any, original_input: Any, **kwargs) -> str:
        """Check if a standard agent-output region dict is valid.

        Returns an error string describing the problem if the region is bad
        (empty, hallucinated, refusal, or out-of-bounds).  Returns empty
        string "" if the region is acceptable.

        This method only catches *bad agent outputs* — pipeline errors that
        prevent a region from being formed at all should raise ValueError /
        RuntimeError instead.
        """
        if not isinstance(region, dict):
            return f"Region is not a dict: {type(region).__name__}"

        if self.modality == "vision":
            bbox = region.get("bounding_box")
            if not bbox:
                return "Missing bounding_box in agent response"
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return f"Invalid bounding_box format: {bbox}"
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                return f"Non-numeric bounding_box values: {bbox}"
            if x2 <= x1 or y2 <= y1:
                return f"Degenerate bounding box (x2<=x1 or y2<=y1): {bbox}"
            img_w, img_h = self._get_image_dims(original_input)
            if img_w > 0 and img_h > 0:
                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                    return (f"Bounding box {bbox} out of image bounds "
                            f"[0,0,{img_w},{img_h}]")
            return ""

        elif self.modality == "text":
            text_spans = region.get("text_spans")
            span_text = region.get("span_text")  # Q6 single-span format
            if not text_spans and not span_text:
                return "Empty or missing text_spans in agent response"
            spans_to_check = text_spans if text_spans else [span_text]
            if not isinstance(spans_to_check, list):
                return f"'text_spans' is not a list: {spans_to_check}"
            # Get actual text for hallucination check
            if isinstance(original_input, str):
                actual_text = original_input
            elif isinstance(original_input, dict):
                actual_text = (original_input.get('premise', '')
                               + original_input.get('hypothesis', '')
                               + original_input.get('text', ''))
            else:
                actual_text = ""
            # Pre-normalize both sides for all comparisons below.
            # _normalize_text is lossless (encoding variants only), so applying
            # it upfront cannot introduce false positives and means every
            # subsequent fallback automatically handles HTML entities, curly
            # quotes, em-dashes, ellipsis chars, non-breaking spaces, etc.
            actual_text_low = self._normalize_text(actual_text.lower())
            for span in spans_to_check:
                if not isinstance(span, str) or not span.strip():
                    return f"text_span entry is not a non-empty string: {span!r}"
                span_low = self._normalize_text(span.lower())
                if any(m in span_low for m in self._REFUSAL_MARKERS):
                    return f"Refusal/unknown marker in text_span: {span!r}"
                if actual_text:
                    # Exact match first
                    found = span_low in actual_text_low
                    if not found:
                        # Whitespace-flexible fallback: spaces in the span may correspond
                        # to no whitespace in the source (e.g. HTML artefacts like
                        # `"Title"word` vs agent output `"Title" word`).
                        tokens = re.split(r'\s+', span_low.strip())
                        pattern = r'\s*'.join(re.escape(t) for t in tokens if t)
                        try:
                            found = bool(re.search(pattern, actual_text_low))
                        except re.error:
                            found = False
                    if not found:
                        # Contraction-expansion fallback: agents sometimes expand
                        # contractions (e.g. "isn't" → "not", because "is not" ⊇ "not").
                        # Normalise n't / n\u2019t → " not" in both strings before retrying.
                        _nt = re.compile(r"n['\u2019]t\b", re.IGNORECASE)
                        exp_text = _nt.sub(" not", actual_text_low)
                        exp_span = _nt.sub(" not", span_low)
                        found = exp_span in exp_text
                        if not found:
                            # Also try whitespace-flexible on expanded strings
                            tokens2 = re.split(r'\s+', exp_span.strip())
                            pattern2 = r'\s*'.join(re.escape(t) for t in tokens2 if t)
                            try:
                                found = bool(re.search(pattern2, exp_text))
                            except re.error:
                                found = False
                    if not found:
                        # Trailing-punctuation fallback: agents sometimes truncate a
                        # sentence and append a terminal period (e.g. the original has
                        # "memory, as anyone…" but the span ends with "memory.").
                        span_stripped = re.sub(r'[.!?]+$', '', span_low).strip()
                        if span_stripped and span_stripped in actual_text_low:
                            found = True
                        if not found and span_stripped:
                            # whitespace-flexible on stripped span
                            tokens3 = re.split(r'\s+', span_stripped)
                            pattern3 = r'\s*'.join(re.escape(t) for t in tokens3 if t)
                            try:
                                found = bool(re.search(pattern3, actual_text_low))
                            except re.error:
                                found = False
                    if not found:
                        # Leading/trailing punctuation fallback: strip all
                        # non-alphanumeric characters from both ends of the span
                        # (agents may add wrapping quotes, parentheses, leading
                        # commas, trailing colons, etc.).  Internal punctuation
                        # and spaces are preserved so the core phrase still matches.
                        span_depunct = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', span_low)
                        if span_depunct and span_depunct != span_low:
                            found = span_depunct in actual_text_low
                            if not found:
                                tokens4 = re.split(r'\s+', span_depunct.strip())
                                pattern4 = r'\s*'.join(re.escape(t) for t in tokens4 if t)
                                try:
                                    found = bool(re.search(pattern4, actual_text_low))
                                except re.error:
                                    found = False
                    if not found:
                        return f"text_span {span!r} not found in original text (hallucinated)"
            return ""

        else:  # tabular
            keys = region.get("feature_keys")
            if not keys:
                return "Empty or missing feature_keys in agent response"
            if not isinstance(keys, list):
                return f"'feature_keys' is not a list: {keys}"
            for k in keys:
                k_low = str(k).lower().strip()
                if any(m in k_low for m in self._REFUSAL_MARKERS):
                    return f"Refusal/unknown marker in feature key: {k!r}"
            available = set(kwargs.get('feature_names', []))
            if available:
                # Helpers -------------------------------------------------------
                def _strip_cond(s: str) -> str:
                    """Strip trailing comparison conditions.
                    e.g. 'worst concave points > 0.71' → 'worst concave points'
                    """
                    return re.sub(r'\s*[><=!]+[\s\d.]+$', '', str(s)).strip()

                def _norm(s: str) -> str:
                    # Normalise separators: underscore, hyphen, and '=' (VLMs
                    # often write categorical features as 'occupation=Craft-repair')
                    # are all collapsed to '-' so they compare equal.
                    return s.lower().replace('_', '-').replace('=', '-')

                avail_normed = {_norm(f): f for f in available}

                def key_in_dataset(k: str) -> bool:
                    k = _strip_cond(k)
                    # Also try replacing '=' with '_' for the raw-string checks
                    # below (handles 'occupation=Craft-repair' → 'occupation_Craft-repair')
                    k_eq = k.replace('=', '_')
                    # 1. Exact match
                    if k in available or k_eq in available:
                        return True
                    # 2. Forward prefix: k is base column, available has k_Category (one-hot)
                    if any(f.startswith(k + '_') or f.startswith(k_eq + '_') for f in available):
                        return True
                    # 3. Reverse prefix: k is one-hot (base_Category), available has base
                    if any(k.startswith(f + '_') or k.startswith(f + '-')
                           or k_eq.startswith(f + '_') or k_eq.startswith(f + '-')
                           for f in available):
                        return True
                    # 4. Normalised exact match (case-insensitive, hyphen↔underscore↔equals)
                    k_norm = _norm(k)
                    if k_norm in avail_normed:
                        return True
                    # 5. Normalised forward prefix
                    if any(fn.startswith(k_norm + '-') for fn in avail_normed):
                        return True
                    # 6. Normalised reverse prefix
                    if any(k_norm.startswith(fn + '-') or k_norm.startswith(fn + '_')
                           for fn in avail_normed):
                        return True
                    return False

                if not any(key_in_dataset(k) for k in keys):
                    return f"All feature keys are hallucinated (not in dataset): {keys}"
            return ""

    def compute_probability_difference(
        self,
        original_probs: Any,
        modified_probs: Any,
        target_class: int
    ) -> float:
        """
        Compute probability difference for a target class.

        Args:
            original_probs: Original probability distribution
            modified_probs: Modified probability distribution
            target_class: Class index to compare

        Returns:
            P_original[target_class] - P_modified[target_class]
        """
        if hasattr(original_probs, '__getitem__'):
            p_orig = float(original_probs[target_class])
            p_mod = float(modified_probs[target_class])
        else:
            p_orig = float(original_probs)
            p_mod = float(modified_probs)

        return p_orig - p_mod


class MultiInstanceEvaluator(BaseEvaluator):
    """
    Base class for evaluators handling multiple instances (Q4, Q9, Q10).
    """

    @abstractmethod
    def evaluate_multi(
        self,
        agent_output: Dict[str, Any],
        inputs: List[Any],
        model: nn.Module,
        predictions: List[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate with multiple instances.

        Args:
            agent_output: Agent output with multi-instance regions
            inputs: List of input data
            model: Target model
            predictions: List of prediction results
            **kwargs: Additional arguments

        Returns:
            EvaluationResult
        """
        pass

    def evaluate(
        self,
        agent_output: Dict[str, Any],
        original_input: Any,
        model: nn.Module,
        original_prediction: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Default evaluate wraps single input as list and calls evaluate_multi.
        """
        # For multi-instance evaluators, inputs should be passed via kwargs
        inputs = kwargs.pop("inputs", [original_input])
        predictions = kwargs.pop("predictions", [original_prediction])
        return self.evaluate_multi(agent_output, inputs, model, predictions, **kwargs)
