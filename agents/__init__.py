"""
Agents module for XAI Agent Framework

Contains the three-agent architecture:
- ProposerAgent: Strategy planning and tool selection
- ActorAgent: XAI tool execution and explanation generation
- CriticAgent: Explanation faithfulness evaluation
"""

from .base_agent import BaseAgent
from .proposer_agent import ProposerAgent
from .actor_agent import ActorAgent
from .critic_agent import CriticAgent

__all__ = [
    "BaseAgent",
    "ProposerAgent",
    "ActorAgent",
    "CriticAgent",
]
