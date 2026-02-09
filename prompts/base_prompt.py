"""
Base Prompt Builder classes and data structures for XAI Agent Framework

This module defines the abstract base classes and common data structures
used across all question-specific prompt builders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


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
class TextSpan:
    """
    Text span representation for text modality.
    Uses character-level start_index and end_index.
    """
    start_index: int
    end_index: int
    text: Optional[str] = None
    importance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "text": self.text,
            "importance": self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TextSpan":
        """Create from dictionary"""
        return cls(
            start_index=data["start_index"],
            end_index=data["end_index"],
            text=data.get("text"),
            importance=data.get("importance", 1.0)
        )

    def is_valid(self) -> bool:
        """Check if text span indices are valid"""
        return self.end_index > self.start_index and self.start_index >= 0


@dataclass
class TabularFeature:
    """
    Tabular feature representation for tabular modality.
    """
    feature_key: str
    value: Optional[Any] = None
    importance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "feature_key": self.feature_key,
            "value": self.value,
            "importance": self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TabularFeature":
        """Create from dictionary"""
        return cls(
            feature_key=data["feature_key"],
            value=data.get("value"),
            importance=data.get("importance", 1.0)
        )


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

    def __init__(self, modality: str = "vision"):
        """
        Initialize PromptBuilder.

        Args:
            modality: Data modality ("vision", "text", "tabular")
        """
        self.modality = modality
        if isinstance(modality, str):
            self.modality_enum = Modality(modality)
        else:
            self.modality_enum = modality

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

    def _get_modality_specific_output_format(self) -> str:
        """Get modality-specific output format description"""
        if self.modality == "vision":
            return '"bounding_box": [x_min, y_min, x_max, y_max]  // pixel coordinates, integers'
        elif self.modality == "text":
            return '"start_index": int, "end_index": int  // character positions'
        elif self.modality == "tabular":
            return '"feature_key": "string"  // column/feature name'
        else:
            return '"output": {...}'

    # ============================================================
    # Common helper methods for formatting tool results
    # These methods are shared across all prompt builders
    # ============================================================

    def _get_image_size_from_results(self, tool_results: Dict[str, Any]) -> tuple:
        """Extract image size from tool results"""
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

        lines = []
        for tool_name, result in tool_results.items():
            # Skip autonomous_tasks - they are formatted separately
            if tool_name == 'autonomous_tasks':
                continue
            if isinstance(result, dict):
                success = result.get('success', False)
                summary = result.get('summary', result.get('description', 'N/A'))
                lines.append(f"- {tool_name}: {'Success' if success else 'Failed'} - {summary}")
        return "\n".join(lines) if lines else "No tool results available."

    def _format_detailed_statistics(self, tool_results: Dict[str, Any]) -> str:
        """Format detailed statistics from tool results including coordinates"""
        if not tool_results:
            return "No detailed statistics available."

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

            # Handle statistics with attention coordinates
            stats = result.get('statistics', {})
            if stats:
                # Handle LIME segments with bounding boxes
                top_positive_segments = stats.get('top_positive_segments', [])
                if top_positive_segments:
                    lines.append("- **LIME POSITIVE SEGMENTS:**")
                    for seg in top_positive_segments[:5]:  # Top 5 segments
                        seg_id = seg.get('segment_id', '?')
                        weight = seg.get('weight', 0)
                        bbox = seg.get('bbox', [])
                        if bbox:
                            lines.append(f"  - Segment {seg_id}: weight={weight:.4f}, bbox={bbox}**")
                        else:
                            lines.append(f"  - Segment {seg_id}: weight={weight:.4f}")

                    # Show important region ratio
                    important_ratio = stats.get('important_region_ratio', 0)
                    if important_ratio:
                        lines.append(f"- Important region ratio: {important_ratio:.2%}")

                top_negative_segments = stats.get('top_negative_segments', [])
                if top_negative_segments:
                    lines.append("- **LIME NEGATIVE SEGMENTS (oppose prediction):**")
                    for seg in top_negative_segments[:3]:  # Top 3 negative
                        seg_id = seg.get('segment_id', '?')
                        weight = seg.get('weight', 0)
                        bbox = seg.get('bbox', [])
                        if bbox:
                            lines.append(f"  - Segment {seg_id}: weight={weight:.4f}, bbox={bbox}")

                # Top attention/importance/gradient coordinates
                top_coords = (
                    stats.get('top_attention_coords') or
                    stats.get('top_importance_coords') or
                    stats.get('top_gradient_coords')
                )
                if top_coords and len(top_coords) > 0:
                    # Calculate bounding box from top coordinates WITH PADDING
                    xs = [c.get('x', 0) for c in top_coords]
                    ys = [c.get('y', 0) for c in top_coords]
                    if xs and ys:
                        # Raw attention bbox (often too small!)
                        raw_x_min, raw_x_max = min(xs), max(xs)
                        raw_y_min, raw_y_max = min(ys), max(ys)
                        raw_width = raw_x_max - raw_x_min
                        raw_height = raw_y_max - raw_y_min

                        # Add padding to make it a reasonable region (at least 15% of image dimension)
                        min_width = max(int(img_width * 0.15), 15)
                        min_height = max(int(img_height * 0.15), 15)

                        # Calculate center and expand
                        center_x = (raw_x_min + raw_x_max) / 2
                        center_y = (raw_y_min + raw_y_max) / 2

                        expanded_half_w = max(raw_width / 2, min_width / 2)
                        expanded_half_h = max(raw_height / 2, min_height / 2)

                        expanded_x_min = max(0, int(center_x - expanded_half_w))
                        expanded_x_max = min(img_width, int(center_x + expanded_half_w))
                        expanded_y_min = max(0, int(center_y - expanded_half_h))
                        expanded_y_max = min(img_height, int(center_y + expanded_half_h))

                        # Check if expansion actually happened
                        was_expanded = (raw_width < min_width or raw_height < min_height)

                        if was_expanded:
                            lines.append(f"- Raw attention peaks: x=[{raw_x_min}, {raw_x_max}], y=[{raw_y_min}, {raw_y_max}] (only {raw_width}x{raw_height} pixels - TOO SMALL!)")
                            lines.append(f"- EXPANDED attention bbox: [{expanded_x_min}, {expanded_y_min}, {expanded_x_max}, {expanded_y_max}]**")
                        else:
                            lines.append(f"- Attention bbox: [{expanded_x_min}, {expanded_y_min}, {expanded_x_max}, {expanded_y_max}] ({raw_width}x{raw_height} pixels)")

                # Other useful stats
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

        # Add summary recommendation at the end
        """
        if object_detection_bboxes:
            best_det = max(object_detection_bboxes, key=lambda x: x['confidence'])
            lines.append(f"\n### ** RECOMMENDATION: Use object detection bbox {best_det['bbox']} for '{best_det['class']}' (highest confidence)**")
        """

        # Format autonomous task results
        autonomous_tasks = tool_results.get('autonomous_tasks', {})
        if autonomous_tasks:
            lines.append("\n## Autonomous Reasoning Results")
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
                        # Include all structured data from the result
                        for key, value in result_data.items():
                            if key == 'explanation':
                                lines.append(f"- Explanation: {value}")
                            elif key == 'confidence':
                                lines.append(f"- Confidence: {value}")
                            else:
                                # Serialize structured data (objects, discriminative_features, etc.)
                                lines.append(f"- {key}: {_json.dumps(value)}")
                    elif isinstance(result_data, str):
                        lines.append(f"- Result: {result_data}")
                    elif 'text_response' in task_result:
                        lines.append(f"- Result: {task_result['text_response']}")

        return "\n".join(lines) if lines else "No detailed statistics available."

    def _format_extracted_features(self, features: Dict[str, Any]) -> str:
        """Format extracted features for prompt"""
        if not features:
            return "No pre-extracted features available."

        lines = []

        # Handle the format with 'output' key (from VLM feature extraction)
        output = features.get('output', {})
        if output:
            if 'bounding_box' in output:
                bbox = output['bounding_box']
                if isinstance(bbox, list) and len(bbox) == 4:
                    lines.append(f"- Pre-extracted bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            if 'start_index' in output:
                lines.append(f"- Pre-extracted text span: [{output['start_index']}, {output.get('end_index', 0)}]")
            if 'feature_key' in output:
                lines.append(f"- Pre-extracted feature: {output['feature_key']}")

        # Handle explanation from feature extraction
        explanation = features.get('explanation', '')
        if explanation:
            lines.append(f"- Extraction reasoning: {explanation}")

        # Handle confidence
        confidence = features.get('confidence', 0)
        if confidence:
            lines.append(f"- Extraction confidence: {confidence:.2f}")

        # Legacy format support
        if 'responsible_regions' in features:
            for i, region in enumerate(features['responsible_regions'][:3]):
                label = region.get('label', f'Region {i+1}')
                importance = region.get('importance', 0)
                bbox = region.get('bbox', region.get('bounding_box'))
                if bbox:
                    lines.append(f"- {label}: importance={importance:.2f}, bbox={bbox}")
                else:
                    lines.append(f"- {label}: importance={importance:.2f}")

        if 'key_visual_features' in features:
            lines.append(f"- Key features: {', '.join(features['key_visual_features'][:5])}")

        if 'importance_distribution' in features:
            dist = features['importance_distribution']
            if 'primary_region_contribution' in dist:
                lines.append(f"- Primary region contribution: {dist['primary_region_contribution']:.2%}")
            if 'secondary_regions_contribution' in dist:
                lines.append(f"- Secondary regions contribution: {dist['secondary_regions_contribution']:.2%}")

        return "\n".join(lines) if lines else "No pre-extracted features available."

    def _build_image_size_constraint(self, tool_results: Dict[str, Any]) -> str:
        """Build image size constraint section for vision modality"""
        if self.modality != "vision":
            return ""

        image_width, image_height = self._get_image_size_from_results(tool_results)
        min_width = max(int(image_width * 0.1), 10)
        min_height = max(int(image_height * 0.1), 10)
        return f"""
## CRITICAL IMAGE SIZE AND BOUNDING BOX CONSTRAINTS
- Image dimensions: {image_width} x {image_height} pixels
- ALL bounding box coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- **MINIMUM bounding box size: {min_width}x{min_height} pixels (10% of image dimensions)**

**CRITICAL WARNING**:
- Bounding boxes smaller than {min_width}x{min_height} pixels are INVALID
- Tiny bboxes (like 4x1 pixels) will cause evaluation to FAIL
- Always prefer larger, semantically meaningful regions over tiny attention points
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

    def _format_results_comprehensive(self, results: Dict[str, Any]) -> str:
        """Comprehensive formatting of results including tool summaries, detailed stats, and extracted features"""
        tool_results = results.get('tool_results', {})
        extracted = results.get('extracted_features', {})
        autonomous_results = results.get('autonomous_results', {})

        sections = []

        # Tool summary
        tool_summary = self._format_tool_results_summary(tool_results)
        if tool_summary and tool_summary != "No tool results available.":
            sections.append(f"### XAI Tool Results Summary\n{tool_summary}")

        # Detailed statistics
        detailed_stats = self._format_detailed_statistics(tool_results)
        if detailed_stats and detailed_stats != "No detailed statistics available.":
            sections.append(f"### Detailed Statistics\n{detailed_stats}")

        # Autonomous reasoning results
        autonomous_str = self._format_autonomous_results(autonomous_results)
        if autonomous_str:
            sections.append(f"### Autonomous Reasoning Results\n{autonomous_str}")

        return "\n\n".join(sections) if sections else "Analysis completed."


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
