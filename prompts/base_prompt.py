"""
Base Prompt Builder classes and data structures for XAI Agent Framework

This module defines the abstract base classes and common data structures
used across all question-specific prompt builders.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .output_size_config import OutputSizeConfig


class QuestionCategory(Enum):
    """Categories of explanation questions"""
    FEATURE_ATTRIBUTION = "feature_attribution"
    COUNTERFACTUAL = "counterfactual"
    SPURIOUS_FEATURES = "spurious_features"


class Modality(Enum):
    """Data modality types"""
    VISION = "vision"
    TEXT = "text"
    TABULAR = "tabular"


class AttributionType(Enum):
    """Types of feature attribution"""
    MOST = "most"
    LEAST = "least"
    DISTINCTIVE = "distinctive"
    CONTRASTIVE = "contrastive"


@dataclass
class BoundingBox:
    """
    Bounding box representation for vision modality.
    Uses [x_min, y_min, x_max, y_max] format (pixel coordinates).
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 1.0
    label: Optional[str] = None

    def to_list(self) -> List[int]:
        """Convert to [x_min, y_min, x_max, y_max] list"""
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "confidence": self.confidence,
            "label": self.label
        }

    @classmethod
    def from_list(cls, coords: List[int], confidence: float = 1.0, label: str = None) -> "BoundingBox":
        """Create from [x_min, y_min, x_max, y_max] list"""
        return cls(
            x_min=coords[0],
            y_min=coords[1],
            x_max=coords[2],
            y_max=coords[3],
            confidence=confidence,
            label=label
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "BoundingBox":
        """Create from dictionary"""
        return cls(
            x_min=data.get("x_min", data.get("x1", 0)),
            y_min=data.get("y_min", data.get("y1", 0)),
            x_max=data.get("x_max", data.get("x2", 0)),
            y_max=data.get("y_max", data.get("y2", 0)),
            confidence=data.get("confidence", 1.0),
            label=data.get("label")
        )

    def is_valid(self) -> bool:
        """Check if bounding box coordinates are valid"""
        return self.x_max > self.x_min and self.y_max > self.y_min


@dataclass
class ChangePlan:
    """
    Change plan for counterfactual questions (Q6).
    Describes how to modify input to flip prediction.
    """
    action: str  # "change", "delete", "swap", "add"
    new_value: Optional[Any] = None

    # For vision
    bounding_box: Optional[List[int]] = None

    # For text
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    # For tabular
    feature_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format based on modality"""
        result = {"action": self.action}
        if self.new_value is not None:
            result["new_value"] = self.new_value
        if self.bounding_box is not None:
            result["bounding_box"] = self.bounding_box
        if self.start_index is not None:
            result["start_index"] = self.start_index
        if self.end_index is not None:
            result["end_index"] = self.end_index
        if self.feature_key is not None:
            result["feature_key"] = self.feature_key
        return result


@dataclass
class ExtractionField:
    """Defines a field to be extracted from agent output"""
    name: str
    field_type: str
    description: str
    required: bool = True
    default: Any = None


class PromptBuilder(ABC):
    """
    Abstract base class for building prompts for different question types.

    Each question type (Q1-Q10) should have its own PromptBuilder implementation
    that handles all three modalities (vision, text, tabular).
    """

    def __init__(self, modality: str = "vision", output_size_config=None):
        """
        Initialize PromptBuilder.

        Args:
            modality: Data modality ("vision", "text", "tabular")
            output_size_config: Optional OutputSizeConfig instance.
        """
        self.modality = modality
        if isinstance(modality, str):
            self.modality_enum = Modality(modality)
        else:
            self.modality_enum = modality
        self.output_size_config = output_size_config

    @property
    @abstractmethod
    def question_type(self) -> int:
        """Return the question type number (1-10)"""
        pass

    @property
    @abstractmethod
    def question_category(self) -> QuestionCategory:
        """Return the question category"""
        pass

    @property
    @abstractmethod
    def question_template(self) -> str:
        """Return the question template string"""
        pass

    @abstractmethod
    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build prompt for Proposer Agent to select XAI tools/strategy.

        Args:
            context: Dictionary containing:
                - user_question: str
                - model_info: Dict with model details
                - prediction: Dict with prediction results
                - image_path/text_input/input_data: Input data

        Returns:
            Prompt string for Proposer Agent
        """
        pass

    @abstractmethod
    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Build prompt for Actor Agent to generate explanation.

        Args:
            context: Same as build_proposer_prompt
            strategy: Strategy from Proposer Agent
            results: Results from XAI tool execution

        Returns:
            Prompt string for Actor Agent
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the expected output schema for this question type and modality.

        Returns:
            Dictionary describing expected output format
        """
        pass

    def get_output_example(self) -> Dict[str, Any]:
        """
        Get an example of expected output format.
        Override in subclasses for specific examples.

        Returns:
            Example output dictionary
        """
        schema = self.get_output_schema()
        return schema.get("example", {})

    def validate_output(self, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that output matches expected schema.

        Args:
            output: Output dictionary from Agent

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        schema = self.get_output_schema()

        # Check required fields based on modality
        modality_schema = schema.get(self.modality, schema)
        if isinstance(modality_schema, dict):
            for key, expected_type in modality_schema.items():
                if key not in output:
                    errors.append(f"Missing required field: {key}")

        return len(errors) == 0, errors

    def _format_model_info(self, model_info: Dict[str, Any]) -> str:
        """Format model information for prompt"""
        return (
            f"Model: {model_info.get('model_name', 'Unknown')}\n"
            f"Architecture: {model_info.get('architecture', 'Unknown')}\n"
            f"Num Classes: {model_info.get('num_classes', 'Unknown')}"
        )

    def _format_prediction(self, prediction: Dict[str, Any]) -> str:
        """Format prediction information for prompt"""
        pred_class = prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))
        confidence = prediction.get('confidence', 0.0)
        class_name = prediction.get('predicted_class_name', '')

        result = f"Predicted Class: {pred_class}"
        if class_name:
            result += f" ({class_name})"
        result += f"\nConfidence: {confidence:.4f}"

        top5 = prediction.get('top5_predictions', [])
        if top5:
            result += f"\nTop-5 Predictions: {top5}"

        return result

    def _get_available_tools_description(
        self,
        context: Dict[str, Any],
        fallback_description: str
    ) -> str:
        """
        Return a formatted tool-description block for use in proposer prompts.

        If context['available_tools'] is populated (by ProposerAgent after
        calling set_tool_registry), format the real tool names and descriptions.
        Otherwise fall back to the hardcoded per-question description string.
        """
        available_tools: Dict[str, str] = context.get('available_tools', {})
        if available_tools:
            lines = [f"   - {name}: {desc}" for name, desc in available_tools.items()]
            return "\n".join(lines)
        return fallback_description

    def _get_available_tools_list(
        self,
        context: Dict[str, Any],
        fallback_list: str
    ) -> str:
        """
        Return a JSON array string of available tool names for use in proposer
        prompt JSON examples.

        If context['available_tools'] is populated, build the list from the
        real registry names; otherwise use the hardcoded fallback string.
        """
        import json as _json
        available_tools: Dict[str, str] = context.get('available_tools', {})
        if available_tools:
            return _json.dumps(list(available_tools.keys()))
        return fallback_list

    def _get_modality_specific_output_format(self) -> str:
        """Get modality-specific output format description"""
        if self.modality == "vision":
            return '"bounding_box": [x_min, y_min, x_max, y_max]  // pixel coordinates, integers'
        elif self.modality == "text":
            return '"spans": [{"start_index": int, "end_index": int}]  // character positions; list one or more spans'
        elif self.modality == "tabular":
            return '"feature_keys": ["feature_name"]  // column/feature names; list one or more features'
        else:
            return '"output": {...}'

    def _build_output_size_constraint(self, context: Dict[str, Any]) -> str:
        """
        Build an output size constraint string to inject into actor prompts.
        Returns empty string if no size constraint is configured for this modality.
        """
        cfg = self.output_size_config
        if not cfg:
            return ""

        pct = cfg.fixed_percentage
        pct_str = f"{pct * 100:.0f}%"
        modality = self.modality

        if modality == "tabular" and cfg.apply_to_tabular:
            n_target = context.get('n_target_features')
            total = context.get('total_features')
            if n_target and total:
                return (
                    f"\n- **OUTPUT SIZE CONSTRAINT**: Return EXACTLY {n_target} feature_keys "
                    f"(top {pct_str} of {total} total features)"
                )

        elif modality == "text" and cfg.apply_to_text:
            target_chars = context.get('target_chars')
            text_length = context.get('text_length')
            if target_chars and text_length:
                return (
                    f"\n- **OUTPUT SIZE CONSTRAINT**: Spans must cover approximately "
                    f"{target_chars} characters total (top {pct_str} of {text_length} chars)"
                )

        # Vision area constraint is embedded in _build_image_size_constraint
        return ""

    # ============================================================
    # Common helper methods for formatting tool results
    # These methods are shared across all prompt builders
    # ============================================================

    def _get_image_size_from_results(self, tool_results: Dict[str, Any]) -> tuple:
        """Extract image size from tool results"""
        # Handle multi-instance format: check inside instance dicts
        is_multi = any(k.startswith('instance_') for k in tool_results.keys())
        if is_multi:
            for inst_key in tool_results:
                if inst_key.startswith('instance_') and isinstance(tool_results[inst_key], dict):
                    for tool_name, result in tool_results[inst_key].items():
                        if isinstance(result, dict) and result.get('success'):
                            img_size = result.get('original_image_size', {})
                            if img_size:
                                return img_size.get('width', 224), img_size.get('height', 224)
            return 224, 224

        for tool_name, result in tool_results.items():
            # Skip autonomous_tasks
            if tool_name == 'autonomous_tasks':
                continue
            if isinstance(result, dict) and result.get('success'):
                img_size = result.get('original_image_size', {})
                if img_size:
                    return img_size.get('width', 224), img_size.get('height', 224)
        return 224, 224  # Default fallback

    def _format_tool_results_summary(self, tool_results: Dict[str, Any]) -> str:
        """Format tool results summary for prompt"""
        if not tool_results:
            return "No tool results available."

        # Detect multi-instance format: {"instance_0": {tools...}, "instance_1": {tools...}}
        is_multi = any(k.startswith('instance_') for k in tool_results.keys())

        lines = []
        if is_multi:
            for inst_key in sorted(k for k in tool_results if k.startswith('instance_')):
                inst_results = tool_results[inst_key]
                lines.append(f"\n**{inst_key}:**")
                for tool_name, result in inst_results.items():
                    if tool_name == 'autonomous_tasks':
                        continue
                    if isinstance(result, dict):
                        success = result.get('success', False)
                        summary = result.get('summary', result.get('description', 'N/A'))
                        lines.append(f"- {tool_name}: {'Success' if success else 'Failed'} - {summary}")
        else:
            for tool_name, result in tool_results.items():
                if tool_name == 'autonomous_tasks':
                    continue
                if isinstance(result, dict):
                    success = result.get('success', False)
                    summary = result.get('summary', result.get('description', 'N/A'))
                    lines.append(f"- {tool_name}: {'Success' if success else 'Failed'} - {summary}")
        return "\n".join(lines) if lines else "No tool results available."

    def _format_detailed_statistics(self, tool_results: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Format detailed statistics from tool results including coordinates"""
        if not tool_results:
            return "No detailed statistics available."

        # Detect multi-instance format: {"instance_0": {tools...}, "instance_1": {tools...}}
        is_multi = any(k.startswith('instance_') for k in tool_results.keys())

        if is_multi:
            all_lines = []
            for inst_key in sorted(k for k in tool_results if k.startswith('instance_')):
                inst_results = tool_results[inst_key]
                all_lines.append(f"\n## {inst_key}")
                inst_text = self._format_detailed_statistics_single(inst_results, context)
                if inst_text and inst_text != "No detailed statistics available.":
                    all_lines.append(inst_text)
                else:
                    all_lines.append("No detailed statistics for this instance.")
            return "\n".join(all_lines) if all_lines else "No detailed statistics available."

        return self._format_detailed_statistics_single(tool_results, context)

    def _format_autonomous_tasks(self, autonomous_tasks: Dict[str, Any]) -> str:
        """Format autonomous task results."""
        lines = ["\n## Autonomous Reasoning Results"]
        for task_type, task_result in autonomous_tasks.items():
            if not isinstance(task_result, dict):
                continue
            success = task_result.get('success', False)
            query = task_result.get('query', '')
            lines.append(f"\n### {task_type.upper()} Task")
            lines.append(f"- Query: {query}")
            lines.append(f"- Status: {'Success' if success else 'Failed'}")
            if success and 'result' in task_result:
                result_data = task_result['result']
                if isinstance(result_data, dict):
                    import json as _json
                    for key, value in result_data.items():
                        if key == 'explanation':
                            lines.append(f"- Explanation: {value}")
                        elif key == 'confidence':
                            lines.append(f"- Confidence: {value}")
                        else:
                            lines.append(f"- {key}: {_json.dumps(value)}")
                elif isinstance(result_data, str):
                    lines.append(f"- Result: {result_data}")
                elif 'text_response' in task_result:
                    lines.append(f"- Result: {task_result['text_response']}")
        return "\n".join(lines)

    def _format_detailed_statistics_single(self, tool_results: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Format detailed statistics from a single set of tool results."""
        if not tool_results:
            return "No detailed statistics available."

        # Compute trimming limits from context
        cfg = self.output_size_config
        _n_top_tabular = 5   # default
        _pct_text = None     # if set, compute per-tool from num_tokens/words
        _sort_ascending = context.get('feature_sort_direction') == 'ascending' if context else False

        if cfg and context:
            if self.modality == 'tabular' and cfg.apply_to_tabular:
                _n_top_tabular = context.get('n_target_features', 5)
            elif self.modality == 'text' and cfg.apply_to_text:
                _pct_text = cfg.fixed_percentage

        lines = []
        object_detection_bboxes = []  # Collect object detection bboxes for priority

        for tool_name, result in tool_results.items():
            # Skip autonomous_tasks - they are formatted separately
            if tool_name == 'autonomous_tasks':
                continue
            if not isinstance(result, dict) or not result.get('success'):
                continue

            lines.append(f"\n### {tool_name}:")

            # Image size
            img_size = result.get('original_image_size', {})
            img_width = img_size.get('width', 224) if img_size else 224
            img_height = img_size.get('height', 224) if img_size else 224
            if img_size:
                lines.append(f"- Image size: {img_width}x{img_height} pixels")

            # Handle object detection results - HIGHEST PRIORITY for bounding boxes
            detections = result.get('detections', [])
            if detections:
                lines.append("- **DETECTED OBJECTS:**")
                for det in detections[:5]:  # Limit to top 5
                    class_name = det.get('class_name', 'unknown')
                    conf = det.get('confidence', 0)
                    bbox = det.get('bbox', {})
                    if bbox:
                        bbox_list = [bbox.get('x1'), bbox.get('y1'), bbox.get('x2'), bbox.get('y2')]
                        lines.append(f"  - {class_name} (conf={conf:.2f}): bbox={bbox_list}**")
                        object_detection_bboxes.append({
                            'class': class_name,
                            'bbox': bbox_list,
                            'confidence': conf
                        })

            # Handle statistics
            stats = result.get('statistics', {})
            if stats:
                # ── Tabular: feature_importance (SHAP / LIME / IG tabular) ───
                feature_importance = stats.get('feature_importance', [])
                if feature_importance and isinstance(feature_importance[0], dict) and 'feature' in feature_importance[0]:
                    fi0 = feature_importance[0]
                    if 'shap_value' in fi0:
                        val_key = 'shap_value'
                    elif 'attribution_score' in fi0:
                        val_key = 'attribution_score'
                    else:
                        val_key = 'weight'
                    sorted_fi = sorted(feature_importance,
                                       key=lambda x: abs(x.get(val_key, 0)), reverse=not _sort_ascending)
                    top_pos = [f for f in sorted_fi if f.get('direction', '') == 'positive'][:_n_top_tabular]
                    top_neg = [f for f in sorted_fi if f.get('direction', '') == 'negative'][:_n_top_tabular]
                    feat_rank_label = "Least important" if _sort_ascending else "Top important"
                    if top_pos:
                        lines.append(f"- {feat_rank_label} positive features ({val_key}) [sorted by LOWEST attribution]:" if _sort_ascending else f"- {feat_rank_label} positive features ({val_key}):")
                        for fi in top_pos:
                            lines.append(f"  - {fi['feature']}: {fi.get(val_key, 0):+.4f}")
                    if top_neg:
                        lines.append(f"- {feat_rank_label} negative features ({val_key}) [sorted by LOWEST attribution]:" if _sort_ascending else f"- {feat_rank_label} negative features ({val_key}):")
                        for fi in top_neg:
                            lines.append(f"  - {fi['feature']}: {fi.get(val_key, 0):+.4f}")
                    pos_val = stats.get('max_positive_weight', stats.get('max_positive',
                              stats.get('top_positive', 'N/A')))
                    neg_val = stats.get('max_negative_weight', stats.get('max_negative',
                              stats.get('top_negative', 'N/A')))
                    lines.append(f"- Max positive: {pos_val}  Max negative: {neg_val}")
                    lines.append(f"- Total features: {stats.get('num_features', len(feature_importance))}")

                # ── Tabular: feature_sensitivity (Sensitivity Analysis) ────────
                feature_sensitivity = stats.get('feature_sensitivity', [])
                if feature_sensitivity:
                    sorted_fs = sorted(feature_sensitivity,
                                       key=lambda x: abs(x.get('prob_drop', 0)), reverse=not _sort_ascending)
                    fs_rank_label = "Least sensitive features (lowest prob_drop) [sorted by LOWEST sensitivity]:" if _sort_ascending else "Most sensitive features (prob_drop):"
                    lines.append(f"- {fs_rank_label}")
                    for fs in sorted_fs[:_n_top_tabular]:
                        lines.append(
                            f"  - {fs['feature']}: drop={fs.get('prob_drop', 0):+.4f}"
                            f"  (orig={fs.get('original_prob', 0):.4f} → mod={fs.get('modified_prob', 0):.4f})"
                        )
                    lines.append(f"- Max prob_drop: {stats.get('max_prob_drop', 'N/A')}  "
                                 f"Max prob_increase: {stats.get('max_prob_increase', 'N/A')}")

                # ── Text: token_importance (IG text) ──────────────────────────
                token_importance = stats.get('token_importance', [])
                if token_importance:
                    _n_top_text_ti = max(1, math.ceil(_pct_text * stats.get('num_tokens', len(token_importance)))) if _pct_text else 5
                    sorted_ti = sorted(token_importance,
                                       key=lambda x: abs(x.get('attribution_score', 0)), reverse=not _sort_ascending)
                    top_pos = [t for t in sorted_ti if t.get('direction', '') == 'positive'][:_n_top_text_ti]
                    top_neg = [t for t in sorted_ti if t.get('direction', '') == 'negative'][:_n_top_text_ti]
                    tok_rank_label = "Least important" if _sort_ascending else "Top important"
                    tok_sort_note = " [sorted by LOWEST attribution]" if _sort_ascending else ""
                    if top_pos:
                        lines.append(f"- {tok_rank_label} positive tokens (attribution){tok_sort_note}:")
                        for ti in top_pos:
                            lines.append(f"  - [{ti.get('token_index', '?')}] '{ti['token']}': {ti.get('attribution_score', 0):+.4f}")
                    if top_neg:
                        lines.append(f"- {tok_rank_label} negative tokens (attribution){tok_sort_note}:")
                        for ti in top_neg:
                            lines.append(f"  - [{ti.get('token_index', '?')}] '{ti['token']}': {ti.get('attribution_score', 0):+.4f}")
                    lines.append(f"- Total tokens: {stats.get('num_tokens', len(token_importance))}  "
                                 f"Max positive: {stats.get('max_positive', 'N/A')}  "
                                 f"Max negative: {stats.get('max_negative', 'N/A')}")

                # ── Text: word_importance (LIME text / SHAP text) ─────────────
                word_importance = stats.get('word_importance', [])
                if word_importance:
                    wi0 = word_importance[0]
                    wi_val_key = 'shap_value' if 'shap_value' in wi0 else 'weight'
                    _n_top_text_wi = max(1, math.ceil(_pct_text * stats.get('num_important_words', stats.get('num_words', len(word_importance))))) if _pct_text else 5
                    sorted_wi = sorted(word_importance,
                                       key=lambda x: abs(x.get(wi_val_key, 0)), reverse=not _sort_ascending)
                    top_pos = [w for w in sorted_wi if w.get('direction', '') == 'positive'][:_n_top_text_wi]
                    top_neg = [w for w in sorted_wi if w.get('direction', '') == 'negative'][:_n_top_text_wi]
                    word_rank_label = "Least important" if _sort_ascending else "Top important"
                    word_sort_note = " [sorted by LOWEST attribution]" if _sort_ascending else ""
                    if top_pos:
                        lines.append(f"- {word_rank_label} positive words ({wi_val_key}){word_sort_note}:")
                        for wi in top_pos:
                            lines.append(f"  - '{wi['word']}': {wi.get(wi_val_key, 0):+.4f}")
                    if top_neg:
                        lines.append(f"- {word_rank_label} negative words ({wi_val_key}){word_sort_note}:")
                        for wi in top_neg:
                            lines.append(f"  - '{wi['word']}': {wi.get(wi_val_key, 0):+.4f}")
                    pos_val = stats.get('max_positive_weight', stats.get('top_positive', 'N/A'))
                    neg_val = stats.get('max_negative_weight', stats.get('top_negative', 'N/A'))
                    lines.append(f"- Words: {stats.get('num_important_words', stats.get('num_words', len(word_importance)))}  "
                                 f"Max positive: {pos_val}  Max negative: {neg_val}")

                # ── Text: word_sensitivity (Sensitivity Analysis text) ─────────
                word_sensitivity = stats.get('word_sensitivity', [])
                if word_sensitivity:
                    _n_top_text_ws = max(1, math.ceil(_pct_text * stats.get('num_important_words', stats.get('num_words', len(word_sensitivity))))) if _pct_text else 10
                    sorted_ws = sorted(word_sensitivity,
                                       key=lambda x: abs(x.get('prob_drop', 0)), reverse=not _sort_ascending)
                    ws_rank_label = "Least sensitive words (lowest prob_drop) [sorted by LOWEST sensitivity]:" if _sort_ascending else "Most sensitive words (prob_drop):"
                    lines.append(f"- {ws_rank_label}")
                    for ws in sorted_ws[:_n_top_text_ws]:
                        lines.append(
                            f"  - [{ws.get('word_index', '?')}] '{ws['word']}': drop={ws.get('prob_drop', 0):+.4f}"
                            f"  (orig={ws.get('original_prob', 0):.4f} → masked={ws.get('masked_prob', 0):.4f})"
                        )
                    lines.append(f"- Max prob_drop: {stats.get('max_prob_drop', 'N/A')}  "
                                 f"Max prob_increase: {stats.get('max_prob_increase', 'N/A')}")

                # ── Vision: LIME image segments ────────────────────────────────
                top_positive_segments = stats.get('top_positive_segments', [])
                if top_positive_segments:
                    if _sort_ascending:
                        lines.append("- **LIME POSITIVE SEGMENTS (HIGH attribution — these are the MOST responsible regions; your answer for least responsible should be OUTSIDE these):**")
                    else:
                        lines.append("- **LIME POSITIVE SEGMENTS:**")
                    for seg in top_positive_segments[:5]:
                        seg_id = seg.get('segment_id', '?')
                        weight = seg.get('weight', 0)
                        bbox = seg.get('bbox', [])
                        if bbox:
                            lines.append(f"  - Segment {seg_id}: weight={weight:.4f}, bbox={bbox}**")
                        else:
                            lines.append(f"  - Segment {seg_id}: weight={weight:.4f}")
                    important_ratio = stats.get('important_region_ratio', 0)
                    if important_ratio:
                        lines.append(f"- Important region ratio: {important_ratio:.2%}")

                top_negative_segments = stats.get('top_negative_segments', [])
                if top_negative_segments:
                    lines.append("- **LIME NEGATIVE SEGMENTS (oppose prediction):**")
                    for seg in top_negative_segments[:3]:
                        seg_id = seg.get('segment_id', '?')
                        weight = seg.get('weight', 0)
                        bbox = seg.get('bbox', [])
                        if bbox:
                            lines.append(f"  - Segment {seg_id}: weight={weight:.4f}, bbox={bbox}")

                # ── Vision: attention bbox ─────────────────────────────────────
                # Prefer pre-computed heatmap bbox (bbox_from_explanation logic,
                # top-k% pixels). Fall back to bounding box of top_coords peaks.
                # NOTE: suggested_bounding_box and top_*_coords always represent
                # HIGH-attribution (most responsible) regions from the tool.
                # For Q2 (least responsible), these are the OPPOSITE of the answer.
                suggested_bbox = result.get('suggested_bounding_box')
                if suggested_bbox and len(suggested_bbox) == 4:
                    bw = suggested_bbox[2] - suggested_bbox[0]
                    bh = suggested_bbox[3] - suggested_bbox[1]
                    if _sort_ascending:
                        lines.append(f"- HIGH-attribution bbox (most responsible region): {suggested_bbox} ({bw}x{bh} pixels)")
                    else:
                        lines.append(f"- Attention bbox: {suggested_bbox} ({bw}x{bh} pixels)")
                else:
                    top_coords = (
                        stats.get('top_attention_coords') or
                        stats.get('top_importance_coords') or
                        stats.get('top_gradient_coords') or
                        stats.get('top_impact_coords')
                    )
                    if top_coords and len(top_coords) > 0:
                        xs = [c.get('x', 0) for c in top_coords]
                        ys = [c.get('y', 0) for c in top_coords]
                        if xs and ys:
                            raw_x_min, raw_x_max = min(xs), max(xs)
                            raw_y_min, raw_y_max = min(ys), max(ys)
                            raw_width = raw_x_max - raw_x_min
                            raw_height = raw_y_max - raw_y_min
                            if _sort_ascending:
                                lines.append(f"- HIGH-attribution bbox (most responsible region): [{raw_x_min}, {raw_y_min}, {raw_x_max}, {raw_y_max}] ({raw_width}x{raw_height} pixels)")
                            else:
                                lines.append(f"- Attention bbox: [{raw_x_min}, {raw_y_min}, {raw_x_max}, {raw_y_max}] ({raw_width}x{raw_height} pixels)")

                # ── Vision: sensitivity analysis (perturbation curve) ──────────
                if 'perturbation_levels' in stats and 'probabilities' in stats:
                    orig_prob = stats.get('original_probability', 'N/A')
                    final_prob = stats.get('final_probability', 'N/A')
                    prob_drop = stats.get('probability_drop', 'N/A')
                    lines.append(f"- Sensitivity: orig_prob={orig_prob}  final_prob={final_prob}  drop={prob_drop}")

                # ── Vision: scalar summary metrics ─────────────────────────────
                if 'mean_attention' in stats:
                    lines.append(f"- Mean attention: {stats['mean_attention']:.3f}")
                if 'max_attention' in stats:
                    lines.append(f"- Max attention: {stats['max_attention']:.3f}")
                if 'high_attention_ratio' in stats:
                    lines.append(f"- High attention ratio: {stats['high_attention_ratio']:.2%}")
                if 'mean_importance' in stats:
                    lines.append(f"- Mean importance: {stats['mean_importance']:.3f}")
                if 'mean_gradient' in stats:
                    lines.append(f"- Mean gradient: {stats['mean_gradient']:.3f}")
                if 'mean_impact' in stats:
                    lines.append(f"- Mean impact: {stats['mean_impact']:.3f}")
                if 'max_impact' in stats:
                    lines.append(f"- Max impact: {stats['max_impact']:.3f}")
                if 'high_impact_ratio' in stats:
                    lines.append(f"- High impact ratio: {stats['high_impact_ratio']:.2%}")

        return "\n".join(lines) if lines else "No detailed statistics available."

    def _build_image_size_constraint(self, tool_results: Dict[str, Any]) -> str:
        """Build image size constraint section for vision modality"""
        if self.modality != "vision":
            return ""

        image_width, image_height = self._get_image_size_from_results(tool_results)
        min_width = max(int(image_width * 0.1), 10)
        min_height = max(int(image_height * 0.1), 10)

        area_constraint = ""
        cfg = self.output_size_config
        if cfg and cfg.apply_to_vision:
            pct = cfg.fixed_percentage
            target_area = int(image_width * image_height * pct)
            area_constraint = (
                f"\n- **OUTPUT SIZE CONSTRAINT**: Bounding box area must be ~{target_area} px² "
                f"({pct * 100:.0f}% of {image_width}×{image_height} image)\n"
                f"  - Aim for (x_max - x_min) * (y_max - y_min) ≈ {target_area}"
            )

        return f"""
## CRITICAL IMAGE SIZE AND BOUNDING BOX CONSTRAINTS
- Image dimensions: {image_width} x {image_height} pixels
- ALL bounding box coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}] {area_constraint}
"""

    def _format_autonomous_results(self, autonomous_results: Dict[str, Any]) -> str:
        """Format autonomous task results for prompt"""
        if not autonomous_results:
            return ""

        lines = []
        for task_type, result in autonomous_results.items():
            if not isinstance(result, dict):
                continue

            success = result.get('success', False)
            query = result.get('query', '')

            lines.append(f"\n### {task_type.upper()} Task")
            lines.append(f"- Query: {query}")
            lines.append(f"- Status: {'Success' if success else 'Failed'}")

            if success and 'result' in result:
                task_result = result['result']
                if isinstance(task_result, dict):
                    import json as _json
                    # Include all structured data from the result
                    for key, value in task_result.items():
                        if key == 'explanation':
                            lines.append(f"- Explanation: {value}")
                        elif key == 'confidence':
                            lines.append(f"- Confidence: {value}")
                        else:
                            lines.append(f"- {key}: {_json.dumps(value)}")
                elif isinstance(task_result, str):
                    lines.append(f"- Result: {task_result}")

        return "\n".join(lines) if lines else ""

    def _format_results_comprehensive(self, results: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Comprehensive formatting of results including tool summaries, detailed stats, and autonomous results"""
        tool_results = results.get('tool_results', {})
        autonomous_results = results.get('autonomous_results', {})

        sections = []

        # Tool summary
        tool_summary = self._format_tool_results_summary(tool_results)
        if tool_summary and tool_summary != "No tool results available.":
            sections.append(f"### XAI Tool Results Summary\n{tool_summary}")

        # Detailed statistics
        detailed_stats = self._format_detailed_statistics(tool_results, context)
        if detailed_stats and detailed_stats != "No detailed statistics available.":
            sections.append(f"### Detailed Statistics\n{detailed_stats}")

        # Autonomous reasoning results
        autonomous_str = self._format_autonomous_results(autonomous_results)
        if autonomous_str:
            sections.append(f"### Autonomous Reasoning Results\n{autonomous_str}")

        return "\n\n".join(sections) if sections else "Analysis completed."

    def _format_instance_data_section(self, context: Dict[str, Any]) -> str:
        """Format text/tabular instance data for inclusion in prompts.

        Returns a '## Instance Data' section string, or empty string for vision.
        """
        instance_data = context.get('instance_data')
        if not instance_data:
            return ""

        modality = context.get('modality', 'vision')
        if modality == 'vision':
            return ""

        lines = ["\n## Instance Data"]
        labels = "ABCDEFGHIJ"
        for i, inst in enumerate(instance_data):
            label = labels[i] if i < len(labels) else str(i)
            idx = inst.get('index', inst.get('row_no', '?'))
            target = inst.get('target', {})
            predicted = inst.get('predicted', {})
            target_lbl = target.get('label', target.get('value', '?')) if isinstance(target, dict) else target
            pred_lbl = predicted.get('label', predicted.get('value', '?')) if isinstance(predicted, dict) else predicted

            lines.append(f"\n### Instance {label} (index {idx}) — true: {target_lbl}, predicted: {pred_lbl}")

            if modality == 'text':
                if 'text' in inst:
                    text = inst['text']
                    # Truncate very long texts for prompt
                    if len(text) > 800:
                        text = text[:800] + "..."
                    lines.append(f"```\n{text}\n```")
                elif 'premise' in inst:
                    lines.append(f"Premise: {inst['premise']}")
                    lines.append(f"Hypothesis: {inst.get('hypothesis', '')}")
            elif modality == 'tabular':
                feat = inst.get('features', {})
                feat_str = ", ".join(f"{k}={v}" for k, v in feat.items())
                lines.append(f"Features: {feat_str}")

        return "\n".join(lines)

    def _format_single_instance_data_section(self, context: Dict[str, Any]) -> str:
        """Format single-instance text/tabular data for inclusion in actor prompts.

        Uses 'instance_data_single' from context (set by actor._build_context).
        Returns '## Input Data' section string, or empty string for vision.
        """
        inst = context.get('instance_data_single')
        if not inst:
            return ""

        modality = context.get('modality', 'vision')
        if modality == 'vision':
            return ""

        lines = ["\n## Input Data"]
        if modality == 'text':
            text = inst.get('text', '')
            premise = inst.get('premise', '')
            if premise:
                lines.append(f"Premise: {premise}")
                lines.append(f"Hypothesis: {inst.get('hypothesis', '')}")
            elif text:
                if len(text) > 800:
                    text = text[:800] + "..."
                lines.append(f"```\n{text}\n```")
        elif modality == 'tabular':
            feat = inst.get('features', {})
            if isinstance(feat, dict) and feat:
                feat_str = ", ".join(f"{k}={v}" for k, v in feat.items())
                lines.append(f"Features: {feat_str}")

        return "\n".join(lines)


class MultiInstancePromptBuilder(PromptBuilder):
    """
    Base class for questions involving multiple instances (Q4, Q9, Q10).
    """

    @abstractmethod
    def build_proposer_prompt_multi(
        self,
        context: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for Proposer with multiple instances.

        Args:
            context: Common context
            instances: List of instance dictionaries
        """
        pass

    @abstractmethod
    def build_actor_prompt_multi(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for Actor with multiple instances.
        """
        pass
