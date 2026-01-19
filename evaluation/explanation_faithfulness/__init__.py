"""
Explanation Faithfulness Evaluators

Contains per-question-type evaluators that validate agent explanations
against actual model behavior through masking experiments.
"""

from .q1_evaluator import Q1Evaluator
from .q2_evaluator import Q2Evaluator
from .q3_evaluator import Q3Evaluator
from .q4_evaluator import Q4Evaluator
from .q5_evaluator import Q5Evaluator
from .q6_evaluator import Q6Evaluator
from .q7_evaluator import Q7Evaluator
from .q8_evaluator import Q8Evaluator
from .q9_evaluator import Q9Evaluator
from .q10_evaluator import Q10Evaluator

EVALUATOR_MAP = {
    1: Q1Evaluator,
    2: Q2Evaluator,
    3: Q3Evaluator,
    4: Q4Evaluator,
    5: Q5Evaluator,
    6: Q6Evaluator,
    7: Q7Evaluator,
    8: Q8Evaluator,
    9: Q9Evaluator,
    10: Q10Evaluator,
}


def get_evaluator(q_type: int, modality: str = "vision"):
    """
    Get the appropriate evaluator for a question type.

    Args:
        q_type: Question type (1-10)
        modality: Data modality ("vision", "text", "tabular")

    Returns:
        Evaluator instance
    """
    evaluator_class = EVALUATOR_MAP.get(q_type)
    if evaluator_class is None:
        raise ValueError(f"No evaluator found for q_type={q_type}")
    return evaluator_class(modality=modality)


__all__ = [
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
    "EVALUATOR_MAP",
    "get_evaluator",
]
