"""
Rollout utilities for GRPO training.

Data containers:
  Transition       — one VLM call within an episode
  Trajectory       — full episode (sequence of Transitions)
  TrajectoryGroup  — K trajectories for the same question

Core function:
  do_group_rollout(env, question, sampling_client, num_rollouts)
    Synchronous. RLSamplingVLM resolves Tinker futures with .result()
    internally, so no async machinery is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Transition:
    """
    One VLM call within an episode.

    Attributes:
        observation:      Kept for API compatibility with grpo_trainer;
                          set to None in the new XAIRLEnv design.
        action_text:      Decoded text produced by the model.
        action_token_ids: Token IDs of the model output (action only).
        action_logprobs:  Per-token log-probabilities at sampling time.
                          Used as the IS-ratio denominator in GRPO Datum.
        prompt_token_ids: Full prompt token IDs (for Datum model_input).
        reward:           Immediate reward (0.0 for intermediate turns).
        episode_done:     Whether this was the last turn.
        metrics:          Extra diagnostic info.
    """
    observation:      Optional[Any]
    action_text:      str
    action_token_ids: List[int]
    action_logprobs:  List[float]
    prompt_token_ids: List[int]
    reward:           float
    episode_done:     bool
    metrics:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    A full episode: sequence of Transitions.

    The total reward is the reward of the LAST transition (episode-level).
    Intermediate transitions have reward=0 by convention.

    ``data_load_failed`` is set to True when the pipeline raised an exception
    *before* any VLM call (e.g. invalid row_no not found in dataset).  These
    episodes produce no useful gradient signal and should be skipped entirely
    rather than consuming additional rollout slots.
    """
    transitions:       List[Transition]
    total_reward:      float   # = transitions[-1].reward if any
    data_load_failed:  bool = False


@dataclass
class TrajectoryGroup:
    """
    K trajectories for the SAME question (used for GRPO group advantage).

    ``advantages`` is populated by compute_advantages() in grpo_trainer.py.
    """
    trajectories: List[Trajectory]
    advantages:   List[float] = field(default_factory=list)

    @property
    def rewards(self) -> List[float]:
        return [t.total_reward for t in self.trajectories]


# ── Group rollout (synchronous) ───────────────────────────────────────────────

def do_group_rollout(
    env,
    question: Dict[str, Any],
    sampling_client,
    num_rollouts: int = 4,
) -> TrajectoryGroup:
    """
    Execute K rollouts for the same question and return a TrajectoryGroup.

    Synchronous: RLSamplingVLM.invoke() resolves Tinker futures with
    .result() internally, so this function can be called from within an
    async context without blocking the event loop for longer than one
    network round-trip per VLM call.

    Before each rollout, swaps the env's rl_vlm to the current policy
    snapshot via update_sampling_client(). The caller must pass in the
    sampling_client obtained from:

        await training_client.save_weights_and_get_sampling_client_async()

    Args:
        env:             XAIRLEnv instance.
        question:        Question dict from XAIRLDataset.
        sampling_client: Current-policy Tinker SamplingClient.
        num_rollouts:    K (group size; typically 4-8).

    Returns:
        TrajectoryGroup with K trajectories.
        Call compute_advantages() before assembling training datums.
    """
    trajectories: List[Trajectory] = []

    for k in range(num_rollouts):
        # Ensure rl_vlm uses the current policy for this rollout
        env.rl_vlm.update_sampling_client(sampling_client)
        traj = env.run_episode(question, rollout_id=k)
        trajectories.append(traj)

        if traj.data_load_failed:
            # The pipeline failed before making any VLM call (e.g. invalid
            # row_no not found in dataset).  Retrying with a different random
            # seed will produce the same error, so skip remaining rollouts.
            print(
                f"  [rollout] Data-load failure on rollout {k+1}. "
                f"Skipping remaining {num_rollouts - k - 1} rollout(s) for this question."
            )
            break

        print(
            f"  Rollout {k+1}/{num_rollouts}: "
            f"reward={traj.total_reward:.4f}  "
            f"n_transitions={len(traj.transitions)}"
        )

    return TrajectoryGroup(trajectories=trajectories)


# ── Group rollout (parallel) ──────────────────────────────────────────────────

def do_group_rollout_parallel(
    rollout_env_pool,
    question: Dict[str, Any],
    sampling_client,
    num_rollouts: int = 4,
    max_workers: int = 4,
) -> TrajectoryGroup:
    """
    Execute K rollouts for the same question in parallel using an env pool.

    Each rollout thread checks out an independent XAIRLEnv from
    rollout_env_pool, runs the full episode, then returns the env to the pool.
    Unlike do_group_rollout, all K rollouts start concurrently — there is no
    early exit on data_load_failed (all K are already dispatched); the
    caller's all()-check still handles skipping the group.

    Args:
        rollout_env_pool: thread-safe Queue of independent XAIRLEnv instances.
                          Must hold at least min(num_rollouts, max_workers) envs.
        question:         Question dict from XAIRLDataset.
        sampling_client:  Current-policy Tinker SamplingClient.
        num_rollouts:     K (group size).
        max_workers:      Max concurrently running rollout threads.

    Returns:
        TrajectoryGroup with up to K trajectories (fewer if threads errored).
    """
    import concurrent.futures as _cf

    def _run_one(k: int) -> Trajectory:
        env = rollout_env_pool.get()
        try:
            env.rl_vlm.update_sampling_client(sampling_client)
            return env.run_episode(question, rollout_id=k)
        finally:
            rollout_env_pool.put(env)

    n_concurrent = min(num_rollouts, max_workers)
    slot_results: List[Optional[Trajectory]] = [None] * num_rollouts

    with _cf.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        future_to_k = {ex.submit(_run_one, k): k for k in range(num_rollouts)}
        for fut in _cf.as_completed(future_to_k):
            k = future_to_k[fut]
            try:
                slot_results[k] = fut.result()
            except Exception as exc:
                print(f"  [rollout] Rollout {k} thread error: {exc}")
                slot_results[k] = Trajectory(
                    transitions=[], total_reward=0.0, data_load_failed=True
                )

    trajectories = [t for t in slot_results if t is not None]
    for k, traj in enumerate(trajectories):
        if not traj.data_load_failed:
            print(
                f"  Rollout {k+1}/{num_rollouts}: "
                f"reward={traj.total_reward:.4f}  "
                f"n_transitions={len(traj.transitions)}"
            )
    return TrajectoryGroup(trajectories=trajectories)
