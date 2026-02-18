"""
Baselines Package for XAI Agent Framework

Provides baseline reasoning strategies:
- NaiveAgent: Direct VLM answer with no XAI tools
- CoTAgent: Chain-of-Thought reasoning (single-pass, no tools)
- ReActAgent: Reasoning + Acting loop (interleaved tool use)
- ToTAgent: Tree of Thought (branching exploration with scoring)
"""

from baselines.naive_agent import NaiveAgent
from baselines.cot_agent import CoTAgent
from baselines.react_agent import ReActAgent
from baselines.tot_agent import ToTAgent
from baselines.base_baseline import BaseBaseline
from baselines.tool_executor import ToolExecutor

__all__ = [
    "BaseBaseline",
    "ToolExecutor",
    "NaiveAgent",
    "CoTAgent",
    "ReActAgent",
    "ToTAgent",
]
