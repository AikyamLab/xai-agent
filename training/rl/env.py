"""
XAIRLEnv — RL environment backed by XAIPipelineV2.

Thin wrapper that calls pipeline.run() for each episode and converts the
output into a Trajectory for GRPO training.

All XAI logic (prompt building, tool execution, Q-type routing for
Q4/Q9/Q10, improvement loop, SF evaluation, faithfulness critic) lives
inside XAIPipelineV2 — this class only:
  1. Calls pipeline.run() with the right flags from train.py args.
  2. Extracts the faithfulness score as the GRPO reward.
  3. Converts rl_vlm.get_transitions() into a Trajectory.

The pipeline's VLM (rl_vlm, an RLSamplingVLM instance) records every
VLM call so that GRPO can compute gradients over the full episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rollout import Trajectory, Transition


class XAIRLEnv:
    """
    RL environment backed by XAIPipelineV2.

    Usage::

        rl_vlm   = RLSamplingVLM(sampling_client, tokenizer)
        pipeline = XAIPipelineV2(vlm=rl_vlm, output_dir=..., ...)
        env      = XAIRLEnv(pipeline=pipeline, rl_vlm=rl_vlm,
                             enable_improvement=False, enable_sf=False)

        # Per rollout k = 0 .. num_rollouts-1:
        env.rl_vlm.update_sampling_client(current_sampling_client)
        traj = env.run_episode(question, rollout_id=k)

    Args:
        pipeline:               XAIPipelineV2 instance (created with vlm=rl_vlm).
        rl_vlm:                 RLSamplingVLM — the VLM injected into the pipeline.
        enable_improvement:     Forward to pipeline.run(enable_improvement=...).
                                Default False for GRPO (improvement VLM calls would
                                add extra transitions with mixed reward signal).
        enable_sf:              Forward to pipeline.run(enable_sf=...).
        sf_max_samples:         Forward to pipeline.run(sf_max_samples=...).
        faithfulness_threshold: Forward to pipeline.run(faithfulness_threshold=...).
    """

    def __init__(
        self,
        pipeline,
        rl_vlm,
        enable_improvement: bool = False,
        enable_sf: bool = False,
        sf_max_samples: Optional[int] = None,
        faithfulness_threshold: float = 0.1,
    ):
        self.pipeline               = pipeline
        self.rl_vlm                 = rl_vlm
        self.enable_improvement     = enable_improvement
        self.enable_sf              = enable_sf
        self.sf_max_samples         = sf_max_samples
        self.faithfulness_threshold = faithfulness_threshold

    def run_episode(
        self,
        question: Dict[str, Any],
        rollout_id: int = 0,
    ) -> Trajectory:
        """
        Run one full pipeline episode and return a Trajectory.

        Calls pipeline.run() which handles all Q-type routing (Q4, Q9/Q10,
        standard Q1-Q3 / Q5-Q8), then extracts the faithfulness score as
        the GRPO reward and wraps rl_vlm transitions into Transition objects.

        The question dict MUST contain ``_source_path`` and ``_question_idx``
        injected by XAIRLDataset._load_json_questions().

        Args:
            question:   Question dict from XAIRLDataset.
            rollout_id: Rollout index (0 .. K-1); logged for diagnostics.

        Returns:
            Trajectory with one Transition per VLM call; reward on last.
        """
        self.rl_vlm.reset()

        # Derive target model URL from question dict
        modality     = question.get("modality", "tabular")
        dataset_name = question.get("dataset_name", "")
        model_url    = f"{modality}/{dataset_name}.pth" if dataset_name else None

        print(f"\n[RL rollout={rollout_id}] "
              f"q_type={question.get('q_type')}  "
              f"row={question.get('row_no', '?')}  "
              f"dataset={dataset_name}")

        # Run the full XAI pipeline — handles all Q-type routing internally.
        # Two failure modes are distinguished by whether any VLM call was
        # recorded before the exception:
        #
        #   data_load_failed=True  (rl_vlm._transitions still empty when error
        #                           is raised, e.g. invalid row_no):
        #       → return a failed Trajectory immediately so do_group_rollout
        #         can skip the remaining K-1 rollouts for this question.
        #
        #   data_load_failed=False (VLM was called before the error, e.g. JSON
        #                           parse error, masking error):
        #       → reward=0.0 with the partial transitions already recorded.
        #         This IS a valid gradient signal (penalises the model output
        #         that caused the failure) and training continues normally.
        faithfulness_score = 0.0
        try:
            result = self.pipeline.run(
                question_dataset_path=question["_source_path"],
                question_id=str(question["_question_idx"]),
                target_model_url=model_url,
                evaluate_faithfulness=True,
                faithfulness_threshold=self.faithfulness_threshold,
                enable_improvement=self.enable_improvement,
                enable_sf=self.enable_sf,
                sf_max_samples=self.sf_max_samples,
            )
            faithfulness_score = _extract_faithfulness_score(result, modality=modality)
        except Exception as exc:
            recorded = self.rl_vlm.get_transitions()
            if not recorded:
                # No VLM call happened → data-load / setup failure.
                # Signal do_group_rollout to skip remaining rollouts.
                print(
                    f"  [XAIRLEnv] Data-load error ({type(exc).__name__}): {exc}\n"
                    f"  Returning data_load_failed=True — skipping remaining rollouts."
                )
                return Trajectory(
                    transitions=[], total_reward=0.0, data_load_failed=True
                )
            # VLM was called before the error → keep partial transitions,
            # assign reward=0.0, and continue training normally.
            print(
                f"  [XAIRLEnv] VLM/eval error ({type(exc).__name__}): {exc}\n"
                f"  {len(recorded)} transition(s) already recorded. "
                f"Assigning reward=0.0 — partial transitions used as gradient signal."
            )

        print(f"  → faithfulness reward: {faithfulness_score:.4f}")

        # Build Trajectory from VLM call records
        rl_transitions = self.rl_vlm.get_transitions()
        transitions: List[Transition] = [
            Transition(
                observation=None,
                action_text=t.action_text,
                action_token_ids=t.action_token_ids,
                action_logprobs=t.action_logprobs,
                prompt_token_ids=t.prompt_token_ids,
                reward=0.0,
                episode_done=False,
                metrics={},
            )
            for t in rl_transitions
        ]

        if transitions:
            transitions[-1].reward       = faithfulness_score
            transitions[-1].episode_done = True
        else:
            # No VLM calls were recorded (unexpected); create a dummy transition
            print("  [XAIRLEnv] WARNING: no VLM transitions recorded for this episode.")
            transitions.append(Transition(
                observation=None,
                action_text="",
                action_token_ids=[],
                action_logprobs=[],
                prompt_token_ids=[],
                reward=faithfulness_score,
                episode_done=True,
                metrics={"warning": "no_vlm_calls_recorded"},
            ))

        return Trajectory(transitions=transitions, total_reward=faithfulness_score)


# ── Helpers ───────────────────────────────────────────────────────────────────

# L1 tool-count penalty hyperparameter.
# Reward = soft_score + L1_loss, where L1_loss = -L1_LAMBDA * (n_tools / n_max).
# Forces GRPO to learn the minimum effective tool set per modality/dataset.
_L1_LAMBDA = 0.2
_L1_MAX_TOOLS: Dict[str, int] = {"vision": 8, "text": 6, "tabular": 6}


def _compute_l1_tool_penalty(result: Dict[str, Any], modality: str) -> float:
    """
    Compute L1 penalty = -L1_LAMBDA * (n_tools / n_max).

    n_tools = len(selected_tools) + len(autonomous_tasks) from the executed
    strategy.  Uses the improved strategy if the improvement loop ran.
    Returns 0.0 when strategy info is unavailable.
    """
    strategy = (
        (result.get("improved") or {}).get("strategy")
        or result.get("strategy")
        or {}
    )
    n_tools = (
        len(strategy.get("selected_tools") or [])
        + len(strategy.get("autonomous_tasks") or [])
    )
    n_max = _L1_MAX_TOOLS.get(modality, 8)
    return -_L1_LAMBDA * min(n_tools, n_max) / n_max


def _extract_faithfulness_score(result: Dict[str, Any], modality: str = "") -> float:
    """
    Extract the final faithfulness score from pipeline.run() output.

    Reward = soft_score + L1_loss  (clamped to [0, 1])

    soft_score is read from details["soft_score"] for all modalities — this
    is the pure quality metric without any size penalty.  Falls back to the
    top-level "score" field when soft_score is absent.

    L1_loss = -L1_LAMBDA * (n_tools / n_max) penalises tool count so that
    GRPO learns to achieve high soft_score with the fewest tools.

    Two-path reward signal:
      - First-pass passed (or enable_improvement=False): use first-pass values.
      - Improvement loop ran: use improved-pass values.
    """
    def _soft(faith: Dict[str, Any]) -> Optional[float]:
        details = faith.get("details") or {}
        raw = details.get("soft_score")
        if raw is not None:
            return float(raw)
        score = faith.get("score")
        return float(score) if score is not None else None

    # First-pass soft score
    evaluation = result.get("evaluation") or {}
    faith = evaluation.get("faithfulness") or {}
    first_pass_soft = _soft(faith) or 0.0

    # If the improvement loop ran, prefer improved soft score
    improved = result.get("improved") or {}
    improved_faith = (improved.get("evaluation") or {}).get("faithfulness") or {}
    imp_soft = _soft(improved_faith)
    soft_score = imp_soft if imp_soft is not None else first_pass_soft

    # L1 tool-count penalty
    l1_loss = _compute_l1_tool_penalty(result, modality)

    return max(0.0, min(1.0, soft_score + l1_loss))
