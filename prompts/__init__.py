"""
Prompts module for XAI Agent Framework

Contains per-question prompt builders with standardized output formats.
"""

from .base_prompt import (
    PromptBuilder,
    QuestionCategory,
    Modality,
    AttributionType,
    BoundingBox,
    TextSpan,
    TabularFeature,
    ExtractionField,
)

from .output_schemas import (
    OutputSchema,
    get_output_schema,
    validate_output,
    QUESTION_OUTPUT_SCHEMAS,
)

# Import all question-specific prompt builders
from .q1_most_responsible import Q1MostResponsiblePromptBuilder
from .q2_least_responsible import Q2LeastResponsiblePromptBuilder
from .q3_distinctive import Q3DistinctivePromptBuilder
from .q4_contrastive_instances import Q4ContrastiveInstancesPromptBuilder
from .q5_mask_prediction import Q5MaskPredictionPromptBuilder
from .q6_flip_prediction import Q6FlipPredictionPromptBuilder
from .q7_change_prediction import Q7ChangePredictionPromptBuilder
from .q8_irrelevant_parts import Q8IrrelevantPartsPromptBuilder
from .q9_shared_feature import Q9SharedFeaturePromptBuilder
from .q10_similar_different import Q10SimilarDifferentPromptBuilder

# Mapping from q_type to PromptBuilder class
PROMPT_BUILDER_MAP = {
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

def get_prompt_builder(q_type: int, modality: str = "vision") -> PromptBuilder:
    """
    Get the appropriate PromptBuilder for a question type.

    Args:
        q_type: Question type (1-10)
        modality: Data modality ("vision", "text", "tabular")

    Returns:
        PromptBuilder instance
    """
    builder_class = PROMPT_BUILDER_MAP.get(q_type)
    if builder_class is None:
        raise ValueError(f"No PromptBuilder found for q_type={q_type}")
    return builder_class(modality=modality)

__all__ = [
    # Base classes
    "PromptBuilder",
    "QuestionCategory",
    "Modality",
    "AttributionType",
    "BoundingBox",
    "TextSpan",
    "TabularFeature",
    "ExtractionField",
    # Schema utilities
    "OutputSchema",
    "get_output_schema",
    "validate_output",
    "QUESTION_OUTPUT_SCHEMAS",
    # Question builders
    "Q1MostResponsiblePromptBuilder",
    "Q2LeastResponsiblePromptBuilder",
    "Q3DistinctivePromptBuilder",
    "Q4ContrastiveInstancesPromptBuilder",
    "Q5MaskPredictionPromptBuilder",
    "Q6FlipPredictionPromptBuilder",
    "Q7ChangePredictionPromptBuilder",
    "Q8IrrelevantPartsPromptBuilder",
    "Q9SharedFeaturePromptBuilder",
    "Q10SimilarDifferentPromptBuilder",
    # Utilities
    "PROMPT_BUILDER_MAP",
    "get_prompt_builder",
]
