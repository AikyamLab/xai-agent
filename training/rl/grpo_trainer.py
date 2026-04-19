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

Metrics logged to grpo_log.jsonl
  type="train"  — every training step
  type="checkpoint" — every checkpoint save (includes tinker_path)
  type="eval"   — every eval callback result
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import queue as _queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .rollout import (
    Trajectory, TrajectoryGroup, Transition,
    do_group_rollout, do_group_rollout_parallel,
)
from .env import XAIRLEnv


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    """
    Hyperparameters for GRPO training with Tinker.

    Attributes:
        learning_rate:      AdamW learning rate (typical: 1e-5 for RL fine-tuning)
        kl_coef:            KL penalty coefficient. 0.0 disables KL (pure GRPO).
                            When >0, a KL reference client must be passed to GRPOTrainer.
        num_rollouts:       Default K rollouts per question.
        rollouts_override:  Per-Q-type rollout count override.
                            Keys are q_type ints; values override num_rollouts for that type.
                            Example: {1: 8, 3: 8, 8: 8, 9: 8} uses K=8 for Q1/Q3/Q8/Q9.
        stratify_by_qtype:  If True, each batch is assembled by round-robin interleaving
                            across Q-types so every batch contains a balanced mix.
                            If False, falls back to plain shuffled batches.
        max_new_tokens:     Max tokens generated per turn during rollout.
        temperature:        Sampling temperature (cookbook recommends T=1.0 for post-trained models).
        batch_size:         Questions per training step.
        num_epochs:         Training epochs over the dataset.
        save_every:         Save Tinker checkpoint every N steps.
        eval_every:         Run evaluation callback every N steps.
        lora_rank:          LoRA rank (passed to create_lora_training_client_async).
        output_dir:         Directory for logs and checkpoints.
    """
    learning_rate: float = 1e-5
    kl_coef: float = 0.0
    start_step: int = 0
    num_rollouts: int = 4
    rollouts_override: Dict[int, int] = field(default_factory=dict)
    stratify_by_qtype: bool = True
    max_new_tokens: int = 512
    temperature: float = 1.0
    batch_size: int = 4
    num_epochs: int = 3
    save_every: int = 50
    eval_every: int = 20
    lora_rank: int = 32
    output_dir: str = "checkpoints/grpo"
    num_workers: int = 1
    """Parallel rollout workers per batch step.
    1 = sequential (original).  Set to batch_size for full question-level
    parallelism.  Each worker owns an independent (RLSamplingVLM, pipeline,
    env) from a pre-created pool — no shared mutable state between workers.
    """
    parallel_rollouts: bool = False
    """Run the K rollouts for each question in parallel (each rollout gets its
    own env checked out from _rollout_pool).  Requires env_factory to be
    provided to GRPOTrainer.  Has no effect when num_rollouts == 1."""
    max_rollout_workers: int = 0
    """Rollout env pool size for parallel_rollouts.  0 = auto (batch_size ×
    max_K across all Q-types).  Acts as a hard cap on total concurrent Tinker
    API calls coming from rollouts.  E.g. 64 limits to 64 simultaneous
    rollouts across all questions in the batch."""


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


# ── Batch statistics ──────────────────────────────────────────────────────────

def _compute_batch_stats(trajectory_groups: List[TrajectoryGroup]) -> Dict[str, Any]:
    """
    Compute diagnostic statistics from a batch of trajectory groups.

    Computes:
      entropy_approx    — proxy for policy entropy: -mean(token logprobs).
                          Higher = more random policy.
      mean_logprob      — mean per-token log-probability under old policy.
      reward_std        — std of total rewards across all trajectories.
      zero_reward_frac  — fraction of trajectories that received reward=0.
      mean_episode_len  — mean number of VLM turns per episode.
      mean_action_tokens— mean number of generated tokens per turn.
      advantage_mean    — should be ~0 by GRPO normalisation.
      advantage_std     — should be ~1 by GRPO normalisation.
    """
    all_rewards      = [t.total_reward
                        for g in trajectory_groups
                        for t in g.trajectories]
    all_logprobs     = [lp
                        for g in trajectory_groups
                        for t in g.trajectories
                        for tr in t.transitions
                        for lp in tr.action_logprobs]
    all_action_lens  = [len(tr.action_token_ids)
                        for g in trajectory_groups
                        for t in g.trajectories
                        for tr in t.transitions]
    all_episode_lens = [len(t.transitions)
                        for g in trajectory_groups
                        for t in g.trajectories]
    all_advantages   = [a
                        for g in trajectory_groups
                        for a in (g.advantages or [])]

    n = len(all_rewards)
    mean_r = sum(all_rewards) / n if n else 0.0
    var_r  = sum((r - mean_r) ** 2 for r in all_rewards) / n if n else 0.0

    stats: Dict[str, Any] = {
        "reward_std":       math.sqrt(var_r),
        "zero_reward_frac": sum(1 for r in all_rewards if r == 0.0) / n if n else 0.0,
        "mean_episode_len": (sum(all_episode_lens) / len(all_episode_lens)
                             if all_episode_lens else 0.0),
        "mean_action_tokens": (sum(all_action_lens) / len(all_action_lens)
                               if all_action_lens else 0.0),
    }

    if all_logprobs:
        mean_lp = sum(all_logprobs) / len(all_logprobs)
        stats["mean_logprob"]   = mean_lp
        stats["entropy_approx"] = -mean_lp   # ≈ token-level entropy

    if all_advantages:
        mean_adv = sum(all_advantages) / len(all_advantages)
        var_adv  = sum((a - mean_adv) ** 2 for a in all_advantages) / len(all_advantages)
        stats["advantage_mean"] = mean_adv
        stats["advantage_std"]  = math.sqrt(var_adv)

    return stats


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
        env_factory=None,
        kl_reference_client=None,
    ):
        self.training_client = training_client
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.kl_reference_client = kl_reference_client

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._log_file = open(Path(cfg.output_dir) / "grpo_log.jsonl", "a")

        # ── Parallel rollout env pool ──────────────────────────────────────────
        # Pre-create num_workers independent (rl_vlm + pipeline + env) triples.
        # Workers check out an env, run do_group_rollout, then return it.
        # The primary `env` passed to fit() is still used for sequential mode
        # and for the eval callback.
        self._env_pool: Optional[_queue.Queue] = None
        if env_factory is not None and cfg.num_workers > 1 and not cfg.parallel_rollouts:
            self._env_pool = _queue.Queue()
            print(
                f"  [GRPOTrainer] Creating {cfg.num_workers} rollout worker "
                f"env(s) for parallel rollout..."
            )
            for _ in range(cfg.num_workers):
                self._env_pool.put(env_factory())
            print(f"  [GRPOTrainer] Rollout worker pool ready.")

        # ── Rollout-level env pool ─────────────────────────────────────────────
        # When parallel_rollouts=True, each of the K rollouts for a question
        # runs in its own thread with an independent env checked out here.
        # Pool size defaults to batch_size × max_K (one slot per concurrent
        # rollout); max_rollout_workers overrides this cap.
        self._rollout_pool: Optional[_queue.Queue] = None
        self._rollout_max_workers: int = 0
        if env_factory is not None and cfg.parallel_rollouts:
            max_k = max(list(cfg.rollouts_override.values()) + [cfg.num_rollouts])
            auto_size = cfg.batch_size * max_k
            pool_size = cfg.max_rollout_workers if cfg.max_rollout_workers > 0 else auto_size
            self._rollout_max_workers = pool_size
            print(
                f"  [GRPOTrainer] Creating {pool_size} rollout env(s) "
                f"for parallel rollouts within questions "
                f"(auto={auto_size}, cap={cfg.max_rollout_workers or 'none'})..."
            )
            self._rollout_pool = _queue.Queue()
            for _ in range(pool_size):
                self._rollout_pool.put(env_factory())
            print(f"  [GRPOTrainer] Rollout pool ready.")

    def _write_log(self, record: Dict[str, Any]):
        """Write one JSON record to grpo_log.jsonl and flush."""
        class _Enc(json.JSONEncoder):
            def default(self, o):
                if hasattr(o, "item"):
                    return o.item()
                try:
                    return float(o)
                except (TypeError, ValueError):
                    return str(o)
        self._log_file.write(json.dumps(record, cls=_Enc) + "\n")
        self._log_file.flush()

    def _pool_rollout(
        self,
        question: Dict[str, Any],
        sampling_client,
        num_rollouts: int,
    ) -> TrajectoryGroup:
        """
        Run rollouts for one question, using whichever pool is configured.

        When _rollout_pool is set (parallel_rollouts=True):
            K rollouts run concurrently via do_group_rollout_parallel, each
            with its own env checked out from _rollout_pool.
        Otherwise:
            One env is checked out from _env_pool and K rollouts run serially
            (original behaviour).

        Safe to call from ThreadPoolExecutor worker threads in both cases
        — no shared mutable state between concurrent invocations.
        """
        if self._rollout_pool is not None:
            return do_group_rollout_parallel(
                rollout_env_pool=self._rollout_pool,
                question=question,
                sampling_client=sampling_client,
                num_rollouts=num_rollouts,
                max_workers=self._rollout_max_workers,
            )
        env = self._env_pool.get()
        try:
            return do_group_rollout(env, question, sampling_client, num_rollouts)
        finally:
            self._env_pool.put(env)

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
            Metrics dict (type="train")
        """
        import tinker

        for group in trajectory_groups:
            if not group.advantages:
                compute_advantages(group)

        datums = assemble_datums_for_batch(trajectory_groups, self.tokenizer)
        if not datums:
            print("  [train_step] No valid datums, skipping.")
            return {"type": "train", "step": self._step, "n_datums": 0}

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

        # ── Reward metrics ────────────────────────────────────────────────────
        all_rewards = [r for g in trajectory_groups for r in g.rewards]
        mean_r = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        metrics: Dict[str, Any] = {
            "type":        "train",
            "step":        self._step,
            "mean_reward": mean_r,
            "max_reward":  max(all_rewards) if all_rewards else 0.0,
            "min_reward":  min(all_rewards) if all_rewards else 0.0,
            "n_datums":    len(datums),
            "n_groups":    len(trajectory_groups),
        }

        # ── Rollout diagnostics (entropy, episode length, etc.) ───────────────
        metrics.update(_compute_batch_stats(trajectory_groups))

        # ── Loss / grad_norm from Tinker forward-backward ─────────────────────
        # loss_fn_outputs contains per-token TensorData — skip, not loggable.
        if fwd_result:
            if hasattr(fwd_result, "metrics") and fwd_result.metrics:
                for k, v in fwd_result.metrics.items():
                    metrics[f"fwd_{k}"] = v

        # ── Optim metrics (grad_norm if Tinker returns it) ────────────────────
        if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
            metrics.update(optim_result.metrics)

        return metrics

    async def fit(
        self,
        dataset,              # XAIRLDataset
        env: "XAIRLEnv",      # XAIRLEnv instance (pipeline + rl_vlm)
        sampling_client,      # Initial Tinker SamplingClient (current policy)
        eval_fn=None,         # Optional async Callable -> Optional[Dict[str, Any]]
    ):
        """
        Full async GRPO training loop.

        eval_fn signature:
            async def eval_fn(training_client, step, sampling_client)
                -> Optional[Dict[str, Any]]
        If eval_fn returns a dict, it is written to grpo_log.jsonl with
        type="eval" and step=<current_step>.

        Args:
            dataset:         XAIRLDataset
            env:             XAIRLEnv wrapping XAIPipelineV2 with RLSamplingVLM
            sampling_client: Initial current-policy Tinker SamplingClient
            eval_fn:         Optional async callback, returns metrics dict or None
        """
        print(f"\n{'='*60}")
        print(f"GRPO Training (Tinker backend)")
        print(f"  Questions:     {len(dataset)}")
        print(f"  Batch size:    {self.cfg.batch_size}")
        print(f"  Rollouts/Q:    {self.cfg.num_rollouts}"
              + (f"  (override: {self.cfg.rollouts_override})" if self.cfg.rollouts_override else ""))
        print(f"  Stratify:      {self.cfg.stratify_by_qtype}")
        print(f"  Epochs:        {self.cfg.num_epochs}")
        print(f"  LR:            {self.cfg.learning_rate}")
        print(f"  KL coef:       {self.cfg.kl_coef}")
        print(f"  LoRA rank:     {self.cfg.lora_rank}")
        print(f"  Save every:    {self.cfg.save_every} steps")
        print(f"  Eval every:    {self.cfg.eval_every} steps")
        print(f"{'='*60}\n")

        for epoch in range(self.cfg.num_epochs):
            print(f"\n── Epoch {epoch+1}/{self.cfg.num_epochs} ──")
            if self.cfg.stratify_by_qtype:
                batches = dataset.get_stratified_batches(self.cfg.batch_size)
            else:
                batches = dataset.get_batches(self.cfg.batch_size)

            for batch_idx, question_batch in enumerate(batches):
                if self._step < self.cfg.start_step:
                    print(f"  [resume] Skipping step {self._step} (< start_step={self.cfg.start_step})")
                    self._step += 1
                    continue

                t0 = time.time()
                print(
                    f"\n[Step {self._step}] Batch {batch_idx+1}: "
                    f"{len(question_batch)} question(s)"
                )

                # ── 1. Collect rollouts ───────────────────────────────────────
                trajectory_groups: List[TrajectoryGroup] = []

                if self._env_pool is not None or self._rollout_pool is not None:
                    # ── Parallel: ThreadPoolExecutor + env/rollout pool ───────
                    future_to_meta: Dict[Any, tuple] = {}
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.cfg.num_workers
                    ) as executor:
                        for q_idx, question in enumerate(question_batch):
                            q_type = question.get("q_type")
                            num_rollouts = self.cfg.rollouts_override.get(
                                q_type, self.cfg.num_rollouts
                            )
                            print(
                                f"  Q{q_idx+1}/{len(question_batch)}: "
                                f"q_type={q_type} "
                                f"row={question.get('row_no', '?')} "
                                f"K={num_rollouts} [parallel]"
                            )
                            fut = executor.submit(
                                self._pool_rollout,
                                question,
                                sampling_client,
                                num_rollouts,
                            )
                            future_to_meta[fut] = (q_idx, question)

                        for fut in concurrent.futures.as_completed(future_to_meta):
                            q_idx, question = future_to_meta[fut]
                            try:
                                group = fut.result()
                            except Exception as exc:
                                print(f"    Q{q_idx+1} rollout error: {exc}")
                                continue

                            if all(t.data_load_failed for t in group.trajectories):
                                print(
                                    f"    Skipping Q{q_idx+1} "
                                    f"(data-load failure on all rollouts)."
                                )
                                continue

                            compute_advantages(group)
                            trajectory_groups.append(group)
                            rewards = group.rewards
                            print(
                                f"    Q{q_idx+1} rewards={[f'{r:.3f}' for r in rewards]}  "
                                f"mean={sum(rewards)/len(rewards):.3f}"
                            )

                else:
                    # ── Sequential questions ───────────────────────────────────
                    for q_idx, question in enumerate(question_batch):
                        q_type = question.get("q_type")
                        num_rollouts = self.cfg.rollouts_override.get(
                            q_type, self.cfg.num_rollouts
                        )
                        print(
                            f"  Q{q_idx+1}/{len(question_batch)}: "
                            f"q_type={q_type} "
                            f"row={question.get('row_no', '?')} "
                            f"K={num_rollouts}"
                        )
                        if self._rollout_pool is not None:
                            # rollout-parallel only (num_workers=1)
                            group = self._pool_rollout(
                                question, sampling_client, num_rollouts
                            )
                        else:
                            group = do_group_rollout(
                                env=env,
                                question=question,
                                sampling_client=sampling_client,
                                num_rollouts=num_rollouts,
                            )

                        if all(t.data_load_failed for t in group.trajectories):
                            print(
                                f"    Skipping question "
                                f"(data-load failure on all rollouts, "
                                f"row_no may be invalid)."
                            )
                            continue

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
                metrics["epoch"] = epoch + 1
                metrics["elapsed_s"] = round(time.time() - t0, 1)

                # ── 3. Refresh sampling client to current LoRA weights ────────
                sampling_client = await self.training_client.save_weights_and_get_sampling_client_async()

                print(
                    f"  Step {self._step}: "
                    f"reward={metrics.get('mean_reward', 0):.4f}  "
                    f"reward_std={metrics.get('reward_std', 0):.4f}  "
                    f"zero_frac={metrics.get('zero_reward_frac', 0):.2f}  "
                    f"entropy={metrics.get('entropy_approx', float('nan')):.3f}  "
                    f"datums={metrics.get('n_datums', 0)}  "
                    f"time={metrics['elapsed_s']}s"
                )
                self._write_log(metrics)
                self._step += 1

                # ── 4. Checkpoint ─────────────────────────────────────────────
                if self._step % self.cfg.save_every == 0:
                    await self.save_checkpoint(f"step_{self._step}")

                # ── 5. Evaluation ─────────────────────────────────────────────
                if eval_fn is not None and self._step % self.cfg.eval_every == 0:
                    eval_metrics = await eval_fn(
                        self.training_client, self._step, sampling_client
                    )
                    if eval_metrics:
                        eval_metrics["type"] = "eval"
                        eval_metrics["step"] = self._step
                        self._write_log(eval_metrics)

        await self.save_checkpoint("final")
        self._log_file.close()
        print("\nTraining complete.")

    async def save_checkpoint(self, tag: str):
        """Save the current LoRA weights via Tinker and log the tinker_path."""
        name = f"checkpoint_{tag}"
        try:
            if hasattr(self.training_client, "save_state_async"):
                future = await self.training_client.save_state_async(name)
                # save_state_async returns an AwaitableConcurrentFuture — resolve it
                if hasattr(future, "result_async"):
                    result = await future.result_async()
                elif hasattr(future, "result") and callable(future.result):
                    result = future.result()
                else:
                    result = future
            else:
                result = self.training_client.save_state(name).result()

            # SaveWeightsResponse.path holds the full URI:
            # e.g. tinker://UUID:train:0/weights/checkpoint_name
            tinker_path = getattr(result, "path", None) or name
            ckpt_type   = getattr(result, "type", None)
            print(f"  Checkpoint saved: {name}  →  {tinker_path}")

            self._write_log({
                "type":        "checkpoint",
                "step":        self._step,
                "tag":         tag,
                "name":        name,
                "tinker_path": tinker_path,
                "ckpt_type":   str(ckpt_type) if ckpt_type else None,
            })

        except Exception as e:
            print(f"  [WARNING] Checkpoint save failed ({name}): {e}")
            self._write_log({
                "type":  "checkpoint",
                "step":  self._step,
                "tag":   tag,
                "name":  name,
                "error": str(e),
            })
