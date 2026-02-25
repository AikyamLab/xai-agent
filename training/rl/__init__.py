"""
GRPO (Group Relative Policy Optimization) Training Framework for XAI Agent

This package implements GRPO-based RL training for the XAI pipeline using
Tinker as the training backend (Qwen/Qwen3-VL-30B-A3B-Instruct):

  - Policy model:    Tinker-hosted Qwen3-VL-30B with LoRA (cloud training)
  - Environment:     Local tool execution + faithfulness evaluation
  - Reflection:      Deterministic env output injected as additional MDP turns

Core MDP structure:
  Turn 0: Model generates XAI strategy (Proposer role)
  Turn 1: Model generates explanation  (Actor role) → faithfulness evaluation
          if score >= threshold → done, reward = score
          else → reflection injected by env (no gradient)
  Turn 2: Model improves strategy given reflection (Proposer + reflection)
  Turn 3: Model regenerates explanation → final evaluation → done, reward = final_score

GRPO advantage: A_k = (r_k - mean(group_rewards)) / std(group_rewards)
Loss: Tinker importance_sampling loss  (IS = p_θ / p_old per token)
"""

# ── Environment ───────────────────────────────────────────────────────────────
from .env import XAIEnv, XAIMultiTurnEnv, Observation, StepResult

# ── Rollout (Tinker SamplingClient) ──────────────────────────────────────────
from .rollout import (
    Trajectory,
    TrajectoryGroup,
    Transition,
    do_rollout,
    do_group_rollout,
)

# ── GRPO Trainer (Tinker TrainingClient) ─────────────────────────────────────
from .grpo_trainer import (
    GRPOConfig,
    GRPOTrainer,
    compute_advantages,
    assemble_datum,
    assemble_datums_for_batch,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
from .dataset import XAIRLDataset, infer_modality

__all__ = [
    # env
    "XAIEnv",
    "XAIMultiTurnEnv",
    "Observation",
    "StepResult",
    # rollout
    "Trajectory",
    "TrajectoryGroup",
    "Transition",
    "do_rollout",
    "do_group_rollout",
    # grpo_trainer
    "GRPOConfig",
    "GRPOTrainer",
    "compute_advantages",
    "assemble_datum",
    "assemble_datums_for_batch",
    # dataset
    "XAIRLDataset",
    "infer_modality",
]
