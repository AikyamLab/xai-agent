"""
GRPO (Group Relative Policy Optimization) Training Framework for XAI Agent

Uses MEAPipeline directly for all XAI logic (prompt building, tool
execution, Q-type routing, improvement loop, faithfulness evaluation).
RLSamplingVLM is injected as the pipeline VLM to capture token trajectories.

Core flow:
  1. RLSamplingVLM wraps Tinker SamplingClient; records every VLM call as RLTransition.
  2. MEAPipeline(vlm=rl_vlm) runs the full XAI pipeline per episode.
  3. XAIRLEnv calls pipeline.run() and extracts faithfulness score as reward.
  4. do_group_rollout() (sync) collects K trajectories per question.
  5. GRPOTrainer assembles Tinker Datums and runs async forward_backward + optim_step.

GRPO advantage: A_k = (r_k - mean(group_rewards)) / std(group_rewards)
Loss: Tinker importance_sampling loss  (IS = p_θ / p_old per token)
"""

# ── VLM ───────────────────────────────────────────────────────────────────────
from .rl_vlm import RLSamplingVLM, RLTransition

# ── Environment ───────────────────────────────────────────────────────────────
from .env import XAIRLEnv

# ── Rollout ───────────────────────────────────────────────────────────────────
from .rollout import (
    Trajectory,
    TrajectoryGroup,
    Transition,
    do_group_rollout,
)

# ── GRPO Trainer ──────────────────────────────────────────────────────────────
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
    # rl_vlm
    "RLSamplingVLM",
    "RLTransition",
    # env
    "XAIRLEnv",
    # rollout
    "Trajectory",
    "TrajectoryGroup",
    "Transition",
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
