"""
Evaluation module for XAI Agent Framework

Contains evaluation pipelines for:
- Explanation faithfulness: Validates that agent explanations match model behavior
"""

from .base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
)

from .masking_utils import (
    MaskingStrategy,
    VisionMasker,
    TextMasker,
    TabularMasker,
    get_masker,
    set_masking_output_dir,
    get_masking_output_dir,
)

from .explanation_faithfulness import (
    Q1Evaluator,
    Q2Evaluator,
    Q3Evaluator,
    Q4Evaluator,
    Q5Evaluator,
    Q6Evaluator,
    Q7Evaluator,
    Q8Evaluator,
    Q9Evaluator,
    Q10Evaluator,
    get_evaluator,
    EVALUATOR_MAP,
)

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "MaskingStrategy",
    "VisionMasker",
    "TextMasker",
    "TabularMasker",
    "get_masker",
    "set_masking_output_dir",
    "get_masking_output_dir",
    "Q1Evaluator",
    "Q2Evaluator",
    "Q3Evaluator",
    "Q4Evaluator",
    "Q5Evaluator",
    "Q6Evaluator",
    "Q7Evaluator",
    "Q8Evaluator",
    "Q9Evaluator",
    "Q10Evaluator",
    "get_evaluator",
    "EVALUATOR_MAP",
]
