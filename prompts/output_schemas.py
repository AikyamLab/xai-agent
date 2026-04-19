"""
Standardized Output Schemas for XAI Agent Framework

Defines the expected JSON output formats for each question type (Q1-Q10)
across all modalities (vision, text, tabular).

These schemas are used for:
1. Instructing agents on required output format
2. Validating agent outputs
3. Parsing outputs for evaluation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json


@dataclass
class OutputSchema:
    """Schema definition for agent outputs"""
    q_type: int
    modality: str
    required_fields: Dict[str, str]  # field_name -> type description
    optional_fields: Dict[str, str] = None
    example: Dict[str, Any] = None
    description: str = ""

    def to_prompt_format(self) -> str:
        """Convert schema to prompt-friendly format"""
        lines = ["{"]
        for field, type_desc in self.required_fields.items():
            lines.append(f'    "{field}": {type_desc},')
        if self.optional_fields:
            for field, type_desc in self.optional_fields.items():
                lines.append(f'    "{field}": {type_desc},  // optional')
        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# Q1: Which part of the input was MOST responsible for the prediction?
# =============================================================================

Q1_SCHEMA_VISION = {
    "output": {
        "bounding_box": [0, 0, 100, 100]  # [x_min, y_min, x_max, y_max]
    },
    "explanation": "string"
}

Q1_SCHEMA_TEXT = {
    "output": {
        "text_spans": ["exact phrase from input text"]
    },
    "explanation": "string"
}

Q1_SCHEMA_TABULAR = {
    "output": {
        "feature_keys": ["feature_name"]
    },
    "explanation": "string"
}

# =============================================================================
# Q2: Which part of the input was LEAST responsible for the prediction?
# =============================================================================

Q2_SCHEMA_VISION = Q1_SCHEMA_VISION.copy()
Q2_SCHEMA_TEXT = Q1_SCHEMA_TEXT.copy()
Q2_SCHEMA_TABULAR = Q1_SCHEMA_TABULAR.copy()

# =============================================================================
# Q3: Which specific parts distinguish prediction from next-best alternative?
# =============================================================================

Q3_SCHEMA_VISION = Q1_SCHEMA_VISION.copy()
Q3_SCHEMA_TEXT = Q1_SCHEMA_TEXT.copy()
Q3_SCHEMA_TABULAR = Q1_SCHEMA_TABULAR.copy()

# =============================================================================
# Q4: Why are instances A and B given different predictions?
# =============================================================================

Q4_SCHEMA_VISION = {
    "output": {
        "input_A": "concise feature phrase, e.g. 'fur texture and primate facial features'",
        "input_B": "concise feature phrase, e.g. 'wings and fuselage body'"
    },
    "explanation": "string"
}

Q4_SCHEMA_TEXT = {
    "output": {
        "input_A": {
            "text_spans": ["exact phrase from instance A text"]
        },
        "input_B": {
            "text_spans": ["exact phrase from instance B text"]
        }
    },
    "explanation": "string"
}

Q4_SCHEMA_TABULAR = {
    "output": {
        "input_A": {
            "feature_keys": ["feature_name_1", "feature_name_2"]  # exactly N features (top 25% of total)
        },
        "input_B": {
            "feature_keys": ["feature_name_1", "feature_name_2"]  # exactly N features (top 25% of total)
        }
    },
    "explanation": "string"
}

# =============================================================================
# Q5: If we mask a certain part, would the prediction change?
# =============================================================================

Q5_SCHEMA_VISION = {
    "output": {
        "prediction_changes": 1,  # 1 = Yes, 0 = No
        "masked_region": {
            "bounding_box": [0, 0, 100, 100]  # [x_min, y_min, x_max, y_max]
        }
    },
    "explanation": "string"
}

Q5_SCHEMA_TEXT = {
    "output": {
        "prediction_changes": 1  # 1 = Yes, 0 = No
    },
    "explanation": "string"
}
Q5_SCHEMA_TABULAR = Q5_SCHEMA_TEXT.copy()

# =============================================================================
# Q6: How should the instance change to flip prediction to [expected_class]?
# =============================================================================

Q6_SCHEMA_VISION = {
    "output": {
        "change_plan": {
            "bounding_box": [0, 0, 100, 100],
            "action": "change",  # "change" | "delete" | "swap" | "add"
            "new_value": "optional description"
        }
    },
    "explanation": "string"
}

Q6_SCHEMA_TEXT = {
    "output": {
        "change_plan": [
            {
                "span_text": "exact phrase to change from input text",
                "action": "change",  # "change" | "delete" | "swap" | "add"
                "new_value": "replacement text"
            }
        ]
    },
    "explanation": "string"
}

Q6_SCHEMA_TABULAR = {
    "output": {
        "change_plan": [
            {
                "feature_key": "string",
                "action": "change",  # "change" | "delete" | "swap"
                "new_value": "new feature value"
            }
        ]
    },
    "explanation": "string"
}

# =============================================================================
# Q7: If we remove/change one important part, how would prediction change?
# =============================================================================

Q7_SCHEMA_VISION = {
    "output": {
        "changed_class": "string",  # Predicted class after change
        "masked_region": {
            "bounding_box": [0, 0, 100, 100]  # [x_min, y_min, x_max, y_max]
        }
    },
    "explanation": "string"
}

Q7_SCHEMA_TEXT = {
    "output": {
        "changed_class": "string"
    },
    "explanation": "string"
}
Q7_SCHEMA_TABULAR = Q7_SCHEMA_TEXT.copy()

# =============================================================================
# Q8: Is there any irrelevant part causing the model's wrong prediction?
# =============================================================================

Q8_SCHEMA_VISION = Q1_SCHEMA_VISION.copy()
Q8_SCHEMA_TEXT = Q1_SCHEMA_TEXT.copy()
Q8_SCHEMA_TABULAR = Q1_SCHEMA_TABULAR.copy()

# =============================================================================
# Q9: What shared feature makes multiple misclassified inputs difficult?
# =============================================================================

Q9_SCHEMA_VISION = {
    "output": {
        "instances": [
            {"bounding_box": [0, 0, 100, 100]},
            {"bounding_box": [0, 0, 100, 100]}
        ]
    },
    "shared_feature_description": "string",
    "explanation": "string"
}

Q9_SCHEMA_TEXT = {
    "output": {
        "instances": [
            {"text_spans": ["exact phrase from instance text"]},
            {"text_spans": ["exact phrase from instance text"]}
        ]
    },
    "shared_feature_description": "string",
    "explanation": "string"
}

Q9_SCHEMA_TABULAR = {
    "output": {
        "instances": [
            {"feature_keys": ["feature_name"]},
            {"feature_keys": ["feature_name"]}
        ]
    },
    "shared_feature_description": "string",
    "explanation": "string"
}

# =============================================================================
# Q10: Why are two similar instances given different predictions (one correct, one wrong)?
# =============================================================================

Q10_SCHEMA_VISION = {
    "output": {
        "correct_instance_features": "concise feature phrase, e.g. 'clear object outline and distinct color pattern'",
        "wrong_instance_features": "concise feature phrase, e.g. 'blurred edges and noisy background'"
    },
    "explanation": "string"
}

Q10_SCHEMA_TEXT = {
    "output": {
        "correct_instance_features": {
            "text_spans": ["exact phrase from correct instance text"]
        },
        "wrong_instance_features": {
            "text_spans": ["exact phrase from wrong instance text"]
        }
    },
    "explanation": "string"
}

Q10_SCHEMA_TABULAR = {
    "output": {
        "correct_instance_features": {
            "feature_keys": ["feature_name_1", "feature_name_2"]  # exactly N features (top 25% of total)
        },
        "wrong_instance_features": {
            "feature_keys": ["feature_name_1", "feature_name_2"]  # exactly N features (top 25% of total)
        }
    },
    "explanation": "string"
}

# =============================================================================
# Master Schema Registry
# =============================================================================

QUESTION_OUTPUT_SCHEMAS = {
    1: {"vision": Q1_SCHEMA_VISION, "text": Q1_SCHEMA_TEXT, "tabular": Q1_SCHEMA_TABULAR},
    2: {"vision": Q2_SCHEMA_VISION, "text": Q2_SCHEMA_TEXT, "tabular": Q2_SCHEMA_TABULAR},
    3: {"vision": Q3_SCHEMA_VISION, "text": Q3_SCHEMA_TEXT, "tabular": Q3_SCHEMA_TABULAR},
    4: {"vision": Q4_SCHEMA_VISION, "text": Q4_SCHEMA_TEXT, "tabular": Q4_SCHEMA_TABULAR},
    5: {"vision": Q5_SCHEMA_VISION, "text": Q5_SCHEMA_TEXT, "tabular": Q5_SCHEMA_TABULAR},
    6: {"vision": Q6_SCHEMA_VISION, "text": Q6_SCHEMA_TEXT, "tabular": Q6_SCHEMA_TABULAR},
    7: {"vision": Q7_SCHEMA_VISION, "text": Q7_SCHEMA_TEXT, "tabular": Q7_SCHEMA_TABULAR},
    8: {"vision": Q8_SCHEMA_VISION, "text": Q8_SCHEMA_TEXT, "tabular": Q8_SCHEMA_TABULAR},
    9: {"vision": Q9_SCHEMA_VISION, "text": Q9_SCHEMA_TEXT, "tabular": Q9_SCHEMA_TABULAR},
    10: {"vision": Q10_SCHEMA_VISION, "text": Q10_SCHEMA_TEXT, "tabular": Q10_SCHEMA_TABULAR},
}


def get_output_schema(q_type: int, modality: str) -> Dict[str, Any]:
    """
    Get the output schema for a specific question type and modality.

    Args:
        q_type: Question type (1-10)
        modality: Data modality ("vision", "text", "tabular")

    Returns:
        Schema dictionary
    """
    if q_type not in QUESTION_OUTPUT_SCHEMAS:
        raise ValueError(f"Invalid q_type: {q_type}. Must be 1-10.")

    schemas = QUESTION_OUTPUT_SCHEMAS[q_type]
    if modality not in schemas:
        raise ValueError(f"Invalid modality: {modality}. Must be 'vision', 'text', or 'tabular'.")

    return schemas[modality]


def schema_to_prompt_string(schema: Dict[str, Any], indent: int = 4) -> str:
    """
    Convert a schema dictionary to a formatted string for prompts.

    Args:
        schema: Schema dictionary
        indent: Indentation level

    Returns:
        Formatted string representation
    """
    return json.dumps(schema, indent=indent)


def validate_output(output: Dict[str, Any], q_type: int, modality: str) -> Tuple[bool, List[str]]:
    """
    Validate an agent output against the expected schema.

    Args:
        output: Agent output dictionary
        q_type: Question type (1-10)
        modality: Data modality

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    schema = get_output_schema(q_type, modality)

    # Check for required top-level fields
    if "output" not in output:
        errors.append("Missing required field: 'output'")
        return False, errors

    # Validate based on question type
    output_data = output.get("output", {})
    schema_output = schema.get("output", {})

    # Q1, Q2, Q3, Q8: (Possibly multiple) region output
    if q_type in [1, 2, 3, 8]:
        if modality == "vision":
            if "bounding_box" not in output_data:
                errors.append("Missing required field: 'output.bounding_box'")
            elif not isinstance(output_data["bounding_box"], list) or len(output_data["bounding_box"]) != 4:
                errors.append("'bounding_box' must be a list of 4 integers [x_min, y_min, x_max, y_max]")
        elif modality == "text":
            if "text_spans" not in output_data:
                errors.append("Missing required field: 'output.text_spans' (list of phrase strings)")
        elif modality == "tabular":
            # Accept new multi-key format or legacy single-key
            if "feature_keys" not in output_data and "feature_key" not in output_data:
                errors.append("Missing required field: 'output.feature_keys' (list of feature names)")

    # Q4: two-instance output with named keys
    elif q_type == 4:
        if "input_A" not in output_data:
            errors.append("Missing required field: 'output.input_A'")
        if "input_B" not in output_data:
            errors.append("Missing required field: 'output.input_B'")

    # Q9: shared-feature output as ordered list of instances
    elif q_type == 9:
        if "instances" not in output_data:
            errors.append("Missing required field: 'output.instances' (list of per-instance regions)")

    # Q5: Yes/No prediction
    elif q_type == 5:
        if "prediction_changes" not in output_data:
            errors.append("Missing required field: 'output.prediction_changes'")
        elif output_data["prediction_changes"] not in [0, 1]:
            errors.append("'prediction_changes' must be 0 or 1")

    # Q6: Change plan (may be a single dict or a list of dicts for multi-feature tabular changes)
    elif q_type == 6:
        if "change_plan" not in output_data:
            errors.append("Missing required field: 'output.change_plan'")
        else:
            change_plan = output_data["change_plan"]
            if isinstance(change_plan, list):
                first = change_plan[0] if change_plan else {}
                if "action" not in first:
                    errors.append("Missing required field: 'output.change_plan[0].action'")
            elif isinstance(change_plan, dict):
                if "action" not in change_plan:
                    errors.append("Missing required field: 'output.change_plan.action'")

    # Q7: Changed class prediction
    elif q_type == 7:
        if "changed_class" not in output_data:
            errors.append("Missing required field: 'output.changed_class'")

    # Q10: Feature comparison
    elif q_type == 10:
        if modality == "vision":
            if "correct_instance_features" not in output_data:
                errors.append("Missing required field: 'output.correct_instance_features'")
            if "wrong_instance_features" not in output_data:
                errors.append("Missing required field: 'output.wrong_instance_features'")
        else:
            if "correct_instance_features" not in output_data:
                errors.append("Missing required field: 'output.correct_instance_features'")
            if "wrong_instance_features" not in output_data:
                errors.append("Missing required field: 'output.wrong_instance_features'")

    return len(errors) == 0, errors


def extract_region_from_output(output: Dict[str, Any], q_type: int, modality: str) -> Optional[Dict[str, Any]]:
    """
    Extract the identified region from agent output for evaluation.

    Args:
        output: Agent output dictionary
        q_type: Question type
        modality: Data modality

    Returns:
        Region dictionary or None if not found
    """
    output_data = output.get("output", {})

    if modality == "vision":
        bbox = output_data.get("bounding_box")
        if bbox:
            return {"bounding_box": bbox}
        # For change_plan
        if "change_plan" in output_data:
            return {"bounding_box": output_data["change_plan"].get("bounding_box")}

    elif modality == "text":
        text_spans = output_data.get("text_spans")
        if text_spans and isinstance(text_spans, list):
            return {"text_spans": text_spans}
        # change_plan (Q6)
        if "change_plan" in output_data:
            cp = output_data["change_plan"]
            if isinstance(cp, list) and cp:
                span_text = cp[0].get("span_text")
                if span_text:
                    return {"span_text": span_text}
            elif isinstance(cp, dict):
                span_text = cp.get("span_text")
                if span_text:
                    return {"span_text": span_text}

    elif modality == "tabular":
        # New multi-key format
        keys = output_data.get("feature_keys")
        if keys and isinstance(keys, list):
            return {"feature_keys": keys}
        # Legacy single-key format
        key = output_data.get("feature_key")
        if key:
            return {"feature_keys": [key]}
        if "change_plan" in output_data:
            cp = output_data["change_plan"]
            if isinstance(cp, list) and cp:
                key = cp[0].get("feature_key")
                return {"feature_keys": [key]} if key else None
            elif isinstance(cp, dict):
                key = cp.get("feature_key")
                return {"feature_keys": [key]} if key else None

    return None


def extract_multi_instance_regions(
    output: Dict[str, Any],
    q_type: int,
    modality: str
) -> Dict[str, Dict[str, Any]]:
    """
    Extract regions from multi-instance output (Q4, Q9).

    Args:
        output: Agent output dictionary
        q_type: Question type
        modality: Data modality

    Returns:
        Dictionary mapping instance names to regions
    """
    output_data = output.get("output", {})
    regions = {}

    def _extract_one(value, modality):
        if modality == "vision":
            return {"bounding_box": value.get("bounding_box")}
        elif modality == "text":
            text_spans = value.get("text_spans")
            if text_spans and isinstance(text_spans, list):
                return {"text_spans": text_spans}
        elif modality == "tabular":
            keys = value.get("feature_keys")
            if keys and isinstance(keys, list):
                return {"feature_keys": keys}
            top = value.get("top_features")
            if top and isinstance(top, list):
                return {"feature_keys": top}
            if value.get("feature_key"):
                return {"feature_keys": [value.get("feature_key")]}
        return None

    # Q9: new list format {"instances": [{...}, {...}, ...]}
    instances_list = output_data.get("instances")
    if isinstance(instances_list, list):
        for i, value in enumerate(instances_list):
            key = f"input_{chr(ord('A') + i)}"
            r = _extract_one(value, modality)
            if r is not None:
                regions[key] = r
        return regions

    # Q4 and legacy Q9: named keys input_A, input_B, ...
    for key, value in output_data.items():
        if key.startswith("input_"):
            r = _extract_one(value, modality)
            if r is not None:
                regions[key] = r

    return regions
