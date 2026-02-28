"""
GRPO Trainer for XAI Agent (Tinker backend)

Implements Group Relative Policy Optimization (GRPO) using Tinker's
async training infrastructure:
  - Rollout: tinker.SamplingClient  (from training_client.save_weights_and_get_sampling_client_async())
  - Training: forward_backward_async() + optim_step_async(), pipelined on same clock cycle
  - Loss function: "importance_sampling"

Algorithm per batch:
  1. Get current-policy SamplingClient via save_weights_and_get_sampling_client_async()
  2. Sample K trajectories per question via do_group_rollout()
  3. Compute group-normalized advantages: A_k = (r_k - mean) / (std + ε)
  4. Assemble tinker.Datum per (trajectory, turn):
       Datum(model_input=prompt_tokens, loss_fn_inputs={
           target_tokens: action_token_ids,
           logprobs:      sampling_logprobs,   # IS denominator
           advantages:    [A_k] * n_tokens,
       })
  5. await forward_backward_async(datums, "importance_sampling")
     await optim_step_async(AdamParams(...))   ← same clock cycle as above
  6. await save_weights_and_get_sampling_client_async() for next rollout

All train_step / fit / save_checkpoint methods are async coroutines.
Call with asyncio.run(trainer.fit(...)) or from within an async context.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .rollout import Trajectory, TrajectoryGroup, Transition, do_group_rollout
from .env import XAIRLEnv


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    """
    Hyperparameters for GRPO training with Tinker.

    Attributes:
        learning_rate:   AdamW learning rate (typical: 1e-5 for RL fine-tuning)
        kl_coef:         KL penalty coefficient. 0.0 disables KL (pure GRPO).
                         When >0, a KL reference client must be passed to GRPOTrainer.
        num_rollouts:    K rollouts per question. Larger K → more stable but slower.
        max_new_tokens:  Max tokens generated per turn during rollout.
        temperature:     Sampling temperature (cookbook recommends T=1.0 for post-trained models).
        batch_size:      Questions per training step.
        num_epochs:      Training epochs over the dataset.
        save_every:      Save Tinker checkpoint every N steps.
        eval_every:      Run evaluation callback every N steps.
        lora_rank:       LoRA rank (passed to create_lora_training_client_async).
        output_dir:      Directory for logs and checkpoints.
    """
    learning_rate: float = 1e-5
    kl_coef: float = 0.0
    num_rollouts: int = 4
    max_new_tokens: int = 512
    temperature: float = 1.0
    batch_size: int = 4
    num_epochs: int = 3
    save_every: int = 50
    eval_every: int = 20
    lora_rank: int = 32
    output_dir: str = "checkpoints/grpo"


# ── Advantage computation ─────────────────────────────────────────────────────

def compute_advantages(group: TrajectoryGroup, eps: float = 1e-8) -> List[float]:
    """
    Compute GRPO group-normalized advantages.

    A_k = (r_k - mean(r)) / (std(r) + ε)

    Matches Tinker cookbook's compute_advantages() from rl/data_processing.py.

    Args:
        group: TrajectoryGroup with K trajectories
        eps:   Stability constant

    Returns:
        List[float] of K advantage values (also stored in group.advantages)
    """
    rewards = [t.total_reward for t in group.trajectories]
    mean_r = sum(rewards) / len(rewards)
    var_r  = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r  = math.sqrt(var_r) + eps

    advantages = [(r - mean_r) / std_r for r in rewards]
    group.advantages = advantages
    return advantages


# ── Datum assembly ────────────────────────────────────────────────────────────

def assemble_datum(
    transition: Transition,
    advantage: float,
    tokenizer,
) -> Optional[Any]:
    """
    Assemble one tinker.Datum for a single Transition.

    Tinker's importance_sampling loss requires all four tensors (input_sequence,
    target_tokens, logprobs, advantages) to have the same length N-1, where
    N = len(prompt) + len(action).

    Standard causal-LM shift:
      model_input  = full_sequence[:-1]   (prefix the model conditions on)
      target_tokens = full_sequence[1:]   (what the model predicts at each position)
      logprobs      = [0.0]*(n_prompt-1) + action_logprobs
                      (0 masks out prompt positions; no IS correction there)
      advantages    = [0.0]*(n_prompt-1) + [adv]*n_action
                      (0 masks out prompt positions; loss only on action tokens)

    Returns None for empty actions (skipped during training).
    """
    import torch
    import tinker
    from tinker import TensorData

    if not transition.action_token_ids:
        return None

    prompt_ids  = transition.prompt_token_ids
    action_ids  = transition.action_token_ids
    n_prompt    = len(prompt_ids)
    n_action    = len(action_ids)

    # Full sequence (prompt + action), then apply causal shift
    full_ids = prompt_ids + action_ids          # length N = n_prompt + n_action

    model_input   = tinker.ModelInput.from_ints(full_ids[:-1])   # length N-1
    targets       = full_ids[1:]                                   # length N-1

    # Mask prompt positions with 0; action positions carry real values
    full_logprobs  = [0.0] * (n_prompt - 1) + list(transition.action_logprobs)  # N-1
    full_advantages = [0.0] * (n_prompt - 1) + [advantage] * n_action           # N-1

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(targets, dtype=torch.int32)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(full_logprobs, dtype=torch.float32)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(full_advantages, dtype=torch.float32)
            ),
        },
    )
    return datum


def assemble_datums_for_batch(
    trajectory_groups: List[TrajectoryGroup],
    tokenizer,
) -> List[Any]:
    """
    Build a flat list of tinker.Datum objects from all trajectory groups.

    One Datum per (trajectory, transition) pair. Groups must already have
    advantages populated (call compute_advantages first).
    """
    datums = []
    for group in trajectory_groups:
        if not group.advantages:
            compute_advantages(group)
        for traj, adv in zip(group.trajectories, group.advantages):
            for transition in traj.transitions:
                d = assemble_datum(transition, adv, tokenizer)
                if d is not None:
                    datums.append(d)
    return datums


# ── GRPO Trainer ──────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    GRPO training loop using Tinker's async TrainingClient.

    All public methods are async coroutines. Call via::

        asyncio.run(trainer.fit(dataset, env, sampling_client))

    Args:
        training_client:  tinker.TrainingClient (created via create_lora_training_client_async)
        tokenizer:        Tokenizer from training_client.get_tokenizer()
        cfg:              GRPOConfig
        kl_reference_client: Optional tinker.SamplingClient for KL penalty.
                             Required when cfg.kl_coef > 0.
    """

    def __init__(
        self,
        training_client,
        tokenizer,
        cfg: GRPOConfig,
        kl_reference_client=None,
    ):
        self.training_client = training_client
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.kl_reference_client = kl_reference_client

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._log_file = open(Path(cfg.output_dir) / "grpo_log.jsonl", "a")

    async def train_step(
        self,
        trajectory_groups: List[TrajectoryGroup],
    ) -> Dict[str, Any]:
        """
        One async GRPO training step.

        Pipelines forward_backward_async and optim_step_async so they land
        on the same server clock cycle (following Tinker cookbook pattern).

        Args:
            trajectory_groups: Batch of TrajectoryGroups (advantages already set)

        Returns:
            Metrics dict
        """
        import tinker

        for group in trajectory_groups:
            if not group.advantages:
                compute_advantages(group)

        datums = assemble_datums_for_batch(trajectory_groups, self.tokenizer)
        if not datums:
            print("  [train_step] No valid datums, skipping.")
            return {"step": self._step, "n_datums": 0}

        adam_params = tinker.AdamParams(
            learning_rate=self.cfg.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

        # ── Pipelined forward-backward + optim (same clock cycle) ────────────
        # Note: importance_sampling does not support loss_fn_config / clip_eps.
        # Use ppo loss if clipping is needed.
        fwd_future  = await self.training_client.forward_backward_async(
            datums,
            loss_fn="importance_sampling",
        )
        optim_future = await self.training_client.optim_step_async(adam_params)

        # Consume results
        fwd_result   = await fwd_future.result_async()
        optim_result = await optim_future.result_async()

        # Collect metrics
        all_rewards = [r for g in trajectory_groups for r in g.rewards]
        metrics: Dict[str, Any] = {
            "step":        self._step,
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "max_reward":  max(all_rewards) if all_rewards else 0.0,
            "min_reward":  min(all_rewards) if all_rewards else 0.0,
            "n_datums":    len(datums),
            "n_groups":    len(trajectory_groups),
        }
        if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
            metrics.update(optim_result.metrics)

        return metrics

    async def fit(
        self,
        dataset,              # XAIRLDataset
        env: "XAIRLEnv",      # XAIRLEnv instance (pipeline + rl_vlm)
        sampling_client,      # Initial Tinker SamplingClient (current policy)
        eval_fn=None,         # Optional async Callable(training_client, step, sampling_client)
    ):
        """
        Full async GRPO training loop.

        The pipeline's VLM calls are captured synchronously by RLSamplingVLM
        (via .result()), so do_group_rollout() is synchronous and is called
        directly from within this async coroutine.

        Args:
            dataset:         XAIRLDataset
            env:             XAIRLEnv wrapping XAIPipelineV2 with RLSamplingVLM
            sampling_client: Initial current-policy Tinker SamplingClient
            eval_fn:         Optional async callback(training_client, step, sampling_client)
        """
        print(f"\n{'='*60}")
        print(f"GRPO Training (Tinker backend)")
        print(f"  Questions:     {len(dataset)}")
        print(f"  Batch size:    {self.cfg.batch_size}")
        print(f"  Rollouts/Q:    {self.cfg.num_rollouts}")
        print(f"  Epochs:        {self.cfg.num_epochs}")
        print(f"  LR:            {self.cfg.learning_rate}")
        print(f"  KL coef:       {self.cfg.kl_coef}")
        print(f"  LoRA rank:     {self.cfg.lora_rank}")
        print(f"{'='*60}\n")

        for epoch in range(self.cfg.num_epochs):
            print(f"\n── Epoch {epoch+1}/{self.cfg.num_epochs} ──")
            batches = dataset.get_batches(self.cfg.batch_size)

            for batch_idx, question_batch in enumerate(batches):
                t0 = time.time()
                print(
                    f"\n[Step {self._step}] Sampling {len(question_batch)} question(s) × "
                    f"{self.cfg.num_rollouts} rollouts..."
                )

                # ── 1. Collect rollouts (sync, via RLSamplingVLM.result()) ────
                trajectory_groups: List[TrajectoryGroup] = []

                for q_idx, question in enumerate(question_batch):
                    print(
                        f"  Q{q_idx+1}/{len(question_batch)}: "
                        f"q_type={question.get('q_type')} "
                        f"row={question.get('row_no', '?')}"
                    )
                    group = do_group_rollout(
                        env=env,
                        question=question,
                        sampling_client=sampling_client,
                        num_rollouts=self.cfg.num_rollouts,
                    )
                    compute_advantages(group)
                    trajectory_groups.append(group)

                    rewards = group.rewards
                    print(
                        f"    rewards={[f'{r:.3f}' for r in rewards]}  "
                        f"mean={sum(rewards)/len(rewards):.3f}"
                    )

                # ── 2. Async training step ────────────────────────────────────
                print(f"\n  Training step...")
                metrics = await self.train_step(trajectory_groups)

                # ── 3. Refresh sampling client to current LoRA weights ────────
                sampling_client = await self.training_client.save_weights_and_get_sampling_client_async()

                elapsed = time.time() - t0
                print(
                    f"  Step {self._step}: "
                    f"reward={metrics.get('mean_reward', 0):.4f}  "
                    f"datums={metrics.get('n_datums', 0)}  "
                    f"time={elapsed:.1f}s"
                )
                self._log_file.write(json.dumps(metrics) + "\n")
                self._log_file.flush()
                self._step += 1

                # ── 4. Checkpoint ─────────────────────────────────────────────
                if self._step % self.cfg.save_every == 0:
                    await self.save_checkpoint(f"step_{self._step}")

                # ── 5. Evaluation ─────────────────────────────────────────────
                if eval_fn is not None and self._step % self.cfg.eval_every == 0:
                    await eval_fn(self.training_client, self._step, sampling_client)

        await self.save_checkpoint("final")
        self._log_file.close()
        print("\nTraining complete.")

    async def save_checkpoint(self, tag: str):
        """Save the current LoRA weights via Tinker."""
        name = f"checkpoint_{tag}"
        try:
            if hasattr(self.training_client, "save_state_async"):
                await self.training_client.save_state_async(name)
            else:
                self.training_client.save_state(name)
            print(f"  Checkpoint saved: {name}")
        except Exception as e:
            print(f"  [WARNING] Checkpoint save failed ({name}): {e}")
