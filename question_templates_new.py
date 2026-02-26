"""
Question Templates - Unified Interface (New Implementation)

This module provides a unified interface that integrates:
- Per-question prompt templates (prompts/q1_*.py - q10_*.py)
- Standardized output schemas
- Evaluation pipeline integration

Usage:
    from question_templates_new import (
        get_question_template,
        get_prompt_builder,
        QuestionTemplate,
        QUESTION_REGISTRY
    )

    # Get template for Q1 with vision modality
    template = get_question_template(q_type=1, modality="vision")

    # Get prompt builder
    builder = get_prompt_builder(q_type=1, modality="vision")

    # Build prompts
    proposer_prompt = builder.build_proposer_prompt(context)
    actor_prompt = builder.build_actor_prompt(context, strategy, results)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum

# Import from new modular structure
from prompts.base_prompt import (
    PromptBuilder,
    QuestionCategory,
    Modality,
    AttributionType,
    BoundingBox,
    ExtractionField,
)

from prompts.output_schemas import (
    get_output_schema,
    validate_output,
    QUESTION_OUTPUT_SCHEMAS,
)

# Import all question-specific builders
from prompts.q1_most_responsible import Q1MostResponsiblePromptBuilder
from prompts.q2_least_responsible import Q2LeastResponsiblePromptBuilder
from prompts.q3_distinctive import Q3DistinctivePromptBuilder
from prompts.q4_contrastive_instances import Q4ContrastiveInstancesPromptBuilder
from prompts.q5_mask_prediction import Q5MaskPredictionPromptBuilder
from prompts.q6_flip_prediction import Q6FlipPredictionPromptBuilder
from prompts.q7_change_prediction import Q7ChangePredictionPromptBuilder
from prompts.q8_irrelevant_parts import Q8IrrelevantPartsPromptBuilder
from prompts.q9_shared_feature import Q9SharedFeaturePromptBuilder
from prompts.q10_similar_different import Q10SimilarDifferentPromptBuilder


# =============================================================================
# Question Registry
# =============================================================================

PROMPT_BUILDER_MAP: Dict[int, Type[PromptBuilder]] = {
    1: Q1MostResponsiblePromptBuilder,
    2: Q2LeastResponsiblePromptBuilder,
    3: Q3DistinctivePromptBuilder,
    4: Q4ContrastiveInstancesPromptBuilder,
    5: Q5MaskPredictionPromptBuilder,
    6: Q6FlipPredictionPromptBuilder,
    7: Q7ChangePredictionPromptBuilder,
    8: Q8IrrelevantPartsPromptBuilder,
    9: Q9SharedFeaturePromptBuilder,
    10: Q10SimilarDifferentPromptBuilder,
}

# Question metadata
QUESTION_METADATA = {
    1: {
        "name": "Most Responsible",
        "template": "Which part of the input was most responsible for the model's prediction?",
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "metric": "P_original - P_modified",
        "description": "Identify the most important region/feature for the prediction"
    },
    2: {
        "name": "Least Responsible",
        "template": "Which part of the input was least responsible for the model's prediction?",
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "metric": "-(P_original - P_modified)",
        "description": "Identify the least important region/feature for the prediction"
    },
    3: {
        "name": "Distinctive",
        "template": "Which specific parts distinguish the prediction from the next-best alternative?",
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "metric": "Rank flip: 1 if P_top2_mod > P_top1_mod",
        "description": "Identify features that distinguish top-1 from top-2 prediction"
    },
    4: {
        "name": "Contrastive Instances",
        "template": "Why are instances A and B given different predictions?",
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "metric": "Gap reduction: (gap_orig - gap_mod) > 0",
        "description": "Explain why two instances get different predictions"
    },
    5: {
        "name": "Mask Prediction",
        "template": "If we mask a certain part, would the prediction change?",
        "category": QuestionCategory.COUNTERFACTUAL,
        "metric": "Accuracy: agent_prediction == actual_outcome",
        "description": "Predict whether masking would change the prediction"
    },
    6: {
        "name": "Flip Prediction",
        "template": "How should the instance change to flip the model prediction to [expected_class]?",
        "category": QuestionCategory.COUNTERFACTUAL,
        "metric": "Flip success: R1_modified == expected_class",
        "description": "Propose changes to flip prediction to target class"
    },
    7: {
        "name": "Change Prediction",
        "template": "If we remove/change one important part, how would the prediction change?",
        "category": QuestionCategory.COUNTERFACTUAL,
        "metric": "Accuracy: R1_modified == agent_predicted_class",
        "description": "Predict the new class after modifying important part"
    },
    8: {
        "name": "Irrelevant Parts",
        "template": "Is there any irrelevant part causing the model's wrong prediction?",
        "category": QuestionCategory.SPURIOUS_FEATURES,
        "metric": "Improvement: P_correct_modified > P_correct_original",
        "description": "Identify spurious features causing misclassification"
    },
    9: {
        "name": "Shared Feature",
        "template": "What shared feature makes multiple misclassified inputs difficult?",
        "category": QuestionCategory.SPURIOUS_FEATURES,
        "metric": "All improved: ALL (P_correct_mod > P_correct_orig)",
        "description": "Find shared spurious feature across misclassified inputs"
    },
    10: {
        "name": "Similar Different",
        "template": "Why are two similar instances given different predictions (one correct, one wrong)?",
        "category": QuestionCategory.SPURIOUS_FEATURES,
        "metric": "-Sim(F_correct, F_wrong)",
        "description": "Explain why similar instances have different outcomes"
    },
}


@dataclass
class QuestionTemplate:
    """
    Question template combining metadata, prompt builder, and output schema.

    This is a lightweight wrapper that provides:
    - Question metadata (type, category, template text)
    - Access to the prompt builder
    - Output schema for validation
    """
    q_type: int
    modality: str
    template: str
    category: QuestionCategory
    prompt_builder: PromptBuilder
    attribution_type: Optional[str] = None

    @property
    def output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema"""
        return get_output_schema(self.q_type, self.modality)

    def validate_output(self, output: Dict[str, Any]) -> tuple:
        """Validate agent output against schema"""
        return validate_output(output, self.q_type, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer Agent"""
        return self.prompt_builder.build_proposer_prompt(context)

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor Agent"""
        return self.prompt_builder.build_actor_prompt(context, strategy, results)

    def build_actor_prompt_multi(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Actor Agent with multiple instances (Q4, Q9, Q10)"""
        return self.prompt_builder.build_actor_prompt_multi(context, strategy, results, instances)

    def build_proposer_prompt_multi(
        self,
        context: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Proposer Agent with multiple instances (Q4, Q9, Q10)"""
        return self.prompt_builder.build_proposer_prompt_multi(context, instances)


def get_prompt_builder(q_type: int, modality: str = "vision") -> PromptBuilder:
    """
    Get the prompt builder for a specific question type.

    Args:
        q_type: Question type (1-10)
        modality: Data modality ("vision", "text", "tabular")

    Returns:
        PromptBuilder instance

    Raises:
        ValueError: If q_type is invalid
    """
    if q_type not in PROMPT_BUILDER_MAP:
        raise ValueError(f"Invalid q_type: {q_type}. Must be 1-10.")

    builder_class = PROMPT_BUILDER_MAP[q_type]
    return builder_class(modality=modality)


def get_question_template(q_type: int, modality: str = "vision") -> QuestionTemplate:
    """
    Get a complete question template for a specific question type.

    Args:
        q_type: Question type (1-10)
        modality: Data modality ("vision", "text", "tabular")

    Returns:
        QuestionTemplate instance

    Raises:
        ValueError: If q_type is invalid
    """
    if q_type not in QUESTION_METADATA:
        raise ValueError(f"Invalid q_type: {q_type}. Must be 1-10.")

    metadata = QUESTION_METADATA[q_type]
    prompt_builder = get_prompt_builder(q_type, modality)

    # Determine attribution type for Q1-Q4
    attribution_type = None
    if q_type == 1:
        attribution_type = "most"
    elif q_type == 2:
        attribution_type = "least"
    elif q_type == 3:
        attribution_type = "distinctive"
    elif q_type == 4:
        attribution_type = "contrastive"

    return QuestionTemplate(
        q_type=q_type,
        modality=modality,
        template=metadata["template"],
        category=metadata["category"],
        prompt_builder=prompt_builder,
        attribution_type=attribution_type
    )


def list_question_types() -> List[Dict[str, Any]]:
    """
    List all available question types with metadata.

    Returns:
        List of question metadata dictionaries
    """
    return [
        {
            "q_type": q_type,
            "name": meta["name"],
            "template": meta["template"],
            "category": meta["category"].value,
            "metric": meta["metric"],
            "description": meta["description"]
        }
        for q_type, meta in QUESTION_METADATA.items()
    ]


# =============================================================================
# Backward Compatibility Exports
# =============================================================================

# For backward compatibility with old imports
QUESTION_REGISTRY = QUESTION_METADATA


# Legacy class aliases (for compatibility with old code)
class VisionFeatureAttributionPromptBuilder(Q1MostResponsiblePromptBuilder):
    """Legacy alias for Vision Feature Attribution"""
    def __init__(self, attribution_type: str = "most"):
        if attribution_type == "most":
            super().__init__(modality="vision")
        elif attribution_type == "least":
            # Use Q2 for least
            pass


class VisionCounterfactualPromptBuilder(Q6FlipPredictionPromptBuilder):
    """Legacy alias for Vision Counterfactual"""
    pass


class VisionSpuriousFeaturesPromptBuilder(Q8IrrelevantPartsPromptBuilder):
    """Legacy alias for Vision Spurious Features"""
    pass


# Text modality aliases
class TextFeatureAttributionPromptBuilder(Q1MostResponsiblePromptBuilder):
    """Legacy alias for Text Feature Attribution"""
    def __init__(self, attribution_type: str = "most"):
        super().__init__(modality="text")


class TextCounterfactualPromptBuilder(Q6FlipPredictionPromptBuilder):
    """Legacy alias for Text Counterfactual"""
    def __init__(self):
        super().__init__(modality="text")


class TextSpuriousFeaturesPromptBuilder(Q8IrrelevantPartsPromptBuilder):
    """Legacy alias for Text Spurious Features"""
    def __init__(self):
        super().__init__(modality="text")


# Tabular modality aliases
class TabularFeatureAttributionPromptBuilder(Q1MostResponsiblePromptBuilder):
    """Legacy alias for Tabular Feature Attribution"""
    def __init__(self, attribution_type: str = "most"):
        super().__init__(modality="tabular")


class TabularCounterfactualPromptBuilder(Q6FlipPredictionPromptBuilder):
    """Legacy alias for Tabular Counterfactual"""
    def __init__(self):
        super().__init__(modality="tabular")


class TabularSpuriousFeaturesPromptBuilder(Q8IrrelevantPartsPromptBuilder):
    """Legacy alias for Tabular Spurious Features"""
    def __init__(self):
        super().__init__(modality="tabular")


__all__ = [
    # Main interfaces
    "get_question_template",
    "get_prompt_builder",
    "list_question_types",
    "QuestionTemplate",
    "PROMPT_BUILDER_MAP",
    "QUESTION_METADATA",
    "QUESTION_REGISTRY",
    # Base classes
    "PromptBuilder",
    "QuestionCategory",
    "Modality",
    "AttributionType",
    # Data classes
    "BoundingBox",
    "ExtractionField",
    # Output schema utilities
    "get_output_schema",
    "validate_output",
    # Legacy compatibility exports
    "VisionFeatureAttributionPromptBuilder",
    "VisionCounterfactualPromptBuilder",
    "VisionSpuriousFeaturesPromptBuilder",
    "TextFeatureAttributionPromptBuilder",
    "TextCounterfactualPromptBuilder",
    "TextSpuriousFeaturesPromptBuilder",
    "TabularFeatureAttributionPromptBuilder",
    "TabularCounterfactualPromptBuilder",
    "TabularSpuriousFeaturesPromptBuilder",
]
