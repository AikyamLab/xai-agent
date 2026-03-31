"""
OutputSizeConfig: controls fixed percentage output size for XAI agents.
"""
from dataclasses import dataclass


@dataclass
class OutputSizeConfig:
    """
    Controls how many features/tokens/area the agent should return.

    Attributes:
        fixed_percentage: Fraction of total to return (e.g. 0.25 = top 25%).
        apply_to_tabular: Whether to apply to tabular modality.
        apply_to_text:    Whether to apply to text modality.
        apply_to_vision:  Whether to apply to vision modality (default False).
    """
    fixed_percentage: float = 0.25
    apply_to_tabular: bool = True
    apply_to_text: bool = True
    apply_to_vision: bool = False
