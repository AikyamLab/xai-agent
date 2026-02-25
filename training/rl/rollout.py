"""
Rollout utilities for GRPO training with Tinker.

Provides:
  - Transition / Trajectory / TrajectoryGroup  (data containers)
  - do_rollout(sampling_client, tokenizer, env)   (single episode)
  - do_group_rollout(...)                         (K episodes, same question)

The policy model is Tinker-hosted Qwen/Qwen3-VL-30B-A3B-Instruct.
Inference is via tinker.SamplingClient; no local GPU weights are loaded here.

Token IDs and per-token sampling log-probs are saved in each Transition
for GRPO Datum assembly in grpo_trainer.py.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .env import Observation, StepResult, XAIEnv


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Transition:
    """
    One model action within an episode.

    Attributes:
        observation:      The Observation the model saw (prompt messages + images)
        action_text:      Decoded text produced by the model
        action_token_ids: Token IDs of the model's output (action only, not prompt)
        action_logprobs:  Per-token log-probabilities at sampling time.
                          Shape: (len(action_token_ids),)
                          Used as the IS-ratio denominator in GRPO Datum.
        prompt_token_ids: Token IDs of the full prompt (for Datum model_input).
        reward:           Immediate reward (0.0 for intermediate turns)
        episode_done:     Whether this was the last turn
        metrics:          Extra diagnostic info from env.step()
    """
    observation: Observation
    action_text: str
    action_token_ids: List[int]
    action_logprobs: List[float]
    prompt_token_ids: List[int]
    reward: float
    episode_done: bool
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    A full episode: sequence of Transitions.

    The total reward is the reward of the LAST transition (episode-level).
    Intermediate transitions have reward=0 by convention.
    """
    transitions: List[Transition]
    total_reward: float   # = transitions[-1].reward


@dataclass
class TrajectoryGroup:
    """
    K trajectories for the SAME question (used to compute GRPO group advantage).

    Attributes:
        trajectories: K Trajectory objects
        advantages:   Per-trajectory GRPO advantage A_k = (r_k - mean) / std
                      Populated by compute_advantages() in grpo_trainer.py
    """
    trajectories: List[Trajectory]
    advantages: List[float] = field(default_factory=list)

    @property
    def rewards(self) -> List[float]:
        return [t.total_reward for t in self.trajectories]


# ── Tinker model I/O helpers ──────────────────────────────────────────────────

def _build_tinker_model_input(
    tokenizer,
    messages: List[Dict[str, Any]],
    images: Optional[List[Any]] = None,
) -> Tuple[Any, List[int]]:
    """
    Build a tinker.ModelInput from chat messages and optional PIL images.

    For text-only inputs (tabular, text modalities):
        Applies the HuggingFace chat template via tokenizer to get token IDs,
        then wraps them in tinker.ModelInput.from_ints().

    For multimodal inputs (vision with PIL images):
        Builds a tinker.ModelInput(chunks=[...]) with EncodedTextChunk and
        ImageChunk objects, following the Qwen3-VL interleaved format.

    Args:
        tokenizer:  HuggingFace tokenizer for Qwen3-VL-30B
        messages:   OpenAI-style chat messages list
        images:     Optional list of PIL Image objects

    Returns:
        (model_input, prompt_token_ids) – prompt_token_ids stored in Transition
        for reference (not used by Tinker directly, but useful for logging).
    """
    import tinker
    from tinker import types as tinker_types

    if not images:
        # ── Text-only path ────────────────────────────────────────────────────
        token_ids: List[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        model_input = tinker.ModelInput.from_ints(token_ids)
        return model_input, token_ids

    # ── Multimodal path (Qwen3-VL interleaved chunk format) ──────────────────
    # Build per-message text + image chunks following the Qwen3-VL chat template.
    # Images are inserted into the user message at their natural position.
    chunks = []
    image_idx = 0  # index into the `images` list

    def _enc(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def _add_text(text: str):
        ids = _enc(text)
        if ids:
            chunks.append(tinker_types.EncodedTextChunk(tokens=ids))

    def _add_image(img):
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        chunks.append(tinker_types.ImageChunk(data=buf.getvalue(), format="png"))

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        _add_text(f"<|im_start|>{role}\n")

        if isinstance(content, str):
            _add_text(content)
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type", "text")
                if ptype == "text":
                    _add_text(part.get("text", ""))
                elif ptype in ("image_url", "image") and image_idx < len(images):
                    # Vision token wrapper for Qwen3-VL
                    _add_text("<|vision_start|>")
                    _add_image(images[image_idx])
                    _add_text("<|vision_end|>")
                    image_idx += 1

        _add_text("<|im_end|>\n")

    # Append any remaining images (tool result images appended to obs.images)
    while image_idx < len(images):
        _add_text("<|vision_start|>")
        _add_image(images[image_idx])
        _add_text("<|vision_end|>")
        image_idx += 1

    # Generation prompt
    _add_text("<|im_start|>assistant\n")

    model_input = tinker.ModelInput(chunks=chunks)

    # Reconstruct approximate prompt_token_ids for logging (best-effort text only)
    approx_ids: List[int] = []
    for chunk in chunks:
        if hasattr(chunk, "tokens"):
            approx_ids.extend(chunk.tokens)

    return model_input, approx_ids


async def _sample_action(
    sampling_client,
    model_input,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    stop_tokens: Optional[List[str]] = None,
) -> Tuple[List[int], List[float], str]:
    """
    Sample one action from the Tinker sampling client (async).

    Tinker's sample() returns an APIFuture; we resolve it with result_async().

    Args:
        sampling_client:  tinker.SamplingClient (base model or current LoRA policy)
        model_input:      tinker.ModelInput (prompt)
        tokenizer:        HuggingFace tokenizer (for decoding tokens → text)
        max_new_tokens:   Maximum new tokens to generate
        temperature:      Sampling temperature
        stop_tokens:      List of stop strings (e.g. ["<|im_end|>"])

    Returns:
        (action_token_ids, action_logprobs, action_text)
        - action_token_ids: List[int] – generated token IDs
        - action_logprobs:  List[float] – per-token sampling log-probs
                            (used as IS-ratio denominator in GRPO Datum)
        - action_text:      str – decoded action text
    """
    import tinker

    stop = stop_tokens or ["<|im_end|>", "<|endoftext|>"]

    sampling_params = tinker.SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        logprobs=1,        # request per-token log-probs for IS ratio
        stop=stop,
    )

    result_or_future = sampling_client.sample(
        prompt=model_input,
        sampling_params=sampling_params,
        num_samples=1,
    )

    # Tinker sample() returns an APIFuture; resolve it
    if hasattr(result_or_future, "sequences"):
        result = result_or_future
    elif hasattr(result_or_future, "result_async"):
        result = await result_or_future.result_async()
    else:
        result = result_or_future.result()

    seq = result.sequences[0]
    action_token_ids: List[int] = list(seq.tokens)

    # Extract per-token sampling log-probs
    # Tinker returns logprobs as a list of {token_id: logprob} dicts (one per token)
    action_logprobs: List[float] = []
    if hasattr(seq, "logprobs") and seq.logprobs:
        for tok_id, lp_dict in zip(action_token_ids, seq.logprobs):
            # lp_dict maps token_id → logprob; extract logprob of the sampled token
            if isinstance(lp_dict, dict):
                action_logprobs.append(float(lp_dict.get(tok_id, 0.0)))
            elif isinstance(lp_dict, (int, float)):
                action_logprobs.append(float(lp_dict))
            else:
                action_logprobs.append(0.0)
    else:
        # Fallback: if Tinker does not return logprobs, use 0.0
        # This makes IS_ratio = exp(current - 0), i.e. no IS correction.
        # Log a warning so the user is aware.
        print(
            "  [rollout] WARNING: sampling_client did not return per-token logprobs. "
            "GRPO IS ratio will not be properly normalized."
        )
        action_logprobs = [0.0] * len(action_token_ids)

    # Decode text (strip special tokens)
    action_text = tokenizer.decode(action_token_ids, skip_special_tokens=True)

    return action_token_ids, action_logprobs, action_text


# ── Episode rollout ───────────────────────────────────────────────────────────

async def do_rollout(
    sampling_client,
    tokenizer,
    env: XAIEnv,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> Trajectory:
    """
    Execute one episode in `env` using Tinker's sampling client as the policy (async).

    Steps:
      1. env.initial_observation() → first Observation
      2. Build tinker.ModelInput from observation
      3. await _sample_action() → action tokens + logprobs
      4. Decode action text; call env.step(action_text)
      5. Record Transition; repeat until episode_done

    Args:
        sampling_client:  tinker.SamplingClient backed by the current policy
        tokenizer:        HuggingFace tokenizer for Qwen3-VL-30B
        env:              XAIEnv or XAIMultiTurnEnv instance (already reset)
        max_new_tokens:   Max tokens generated per turn
        temperature:      Sampling temperature

    Returns:
        Trajectory with all transitions (including prompt_token_ids, action_logprobs).
    """
    transitions: List[Transition] = []
    obs = env.initial_observation()

    while True:
        # ── Build Tinker prompt ────────────────────────────────────────────────
        model_input, prompt_token_ids = _build_tinker_model_input(
            tokenizer=tokenizer,
            messages=obs.messages,
            images=obs.images if obs.images else None,
        )

        # ── Sample action from current policy ─────────────────────────────────
        action_token_ids, action_logprobs, action_text = await _sample_action(
            sampling_client=sampling_client,
            model_input=model_input,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # ── Environment step (tool execution + critic evaluation) ──────────────
        step_result = env.step(action_text)

        transitions.append(Transition(
            observation=obs,
            action_text=action_text,
            action_token_ids=action_token_ids,
            action_logprobs=action_logprobs,
            prompt_token_ids=prompt_token_ids,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        ))

        if step_result.episode_done:
            break
        obs = step_result.next_observation

    total_reward = transitions[-1].reward
    return Trajectory(transitions=transitions, total_reward=total_reward)


async def do_group_rollout(
    sampling_client: Any,
    tokenizer,
    env_factory: Callable[[], XAIEnv],
    num_rollouts: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> TrajectoryGroup:
    """
    Execute K rollouts for the SAME question and collect a TrajectoryGroup (async).

    Each rollout uses a fresh env from env_factory() and the SAME sampling
    client (the current policy snapshot obtained before calling this function).

    The caller is responsible for obtaining the current-policy sampling client
    (e.g. via ``await training_client.save_weights_and_get_sampling_client_async()``
    before each batch) and passing it here.

    Args:
        sampling_client:  tinker.SamplingClient backed by the current policy
        tokenizer:        Tokenizer for the policy model
        env_factory:      Callable returning a freshly-reset XAIEnv
        num_rollouts:     K (group size, typically 4–8)
        max_new_tokens:   Max tokens per turn
        temperature:      Sampling temperature

    Returns:
        TrajectoryGroup with K trajectories.
        Call compute_advantages() from grpo_trainer.py before training.
    """
    trajectories: List[Trajectory] = []

    for k in range(num_rollouts):
        env = env_factory()   # fresh env per rollout
        traj = await do_rollout(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            env=env,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        trajectories.append(traj)
        print(
            f"  Rollout {k+1}/{num_rollouts}: "
            f"reward={traj.total_reward:.4f}, "
            f"turns={len(traj.transitions)}"
        )

    return TrajectoryGroup(trajectories=trajectories)
