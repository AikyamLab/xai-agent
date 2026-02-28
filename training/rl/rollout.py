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
    """
    transitions:  List[Transition]
    total_reward: float   # = transitions[-1].reward if any


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
        print(
            f"  Rollout {k+1}/{num_rollouts}: "
            f"reward={traj.total_reward:.4f}  "
            f"n_transitions={len(traj.transitions)}"
        )

    return TrajectoryGroup(trajectories=trajectories)
