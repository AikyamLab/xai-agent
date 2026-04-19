"""
RLSamplingVLM — Tinker-backed VLM for GRPO rollouts.

Wraps a Tinker SamplingClient to:
  - Serve synchronous invoke() / invoke_with_images() calls, matching the
    interface expected by BaseAgent.invoke_vlm() inside XAIPipelineV2 agents.
  - Record every VLM call as an RLTransition
    (prompt_token_ids, action_token_ids, action_logprobs, action_text).
  - Expose reset() / update_sampling_client() / get_transitions() for the
    RL training loop.

Usage pattern:
    rl_vlm   = RLSamplingVLM(sampling_client, tokenizer, max_new_tokens=512)
    pipeline = XAIPipelineV2(vlm=rl_vlm, ...)
    env      = XAIRLEnv(pipeline=pipeline, rl_vlm=rl_vlm)

    # Per rollout (k = 0 .. num_rollouts-1):
    rl_vlm.update_sampling_client(current_sampling_client)
    traj = env.run_episode(question_dict, rollout_id=k)
    # transitions and reward are inside traj.transitions / traj.total_reward
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union


# ── System prompt (matches TinkerVisionLanguageModel) ─────────────────────────

_SYSTEM_MESSAGE = (
    "You are an AI assistant that follows instructions precisely. "
    "When given a task with a specific format to follow, "
    "you MUST follow that exact format. Never respond with greetings or small talk. "
    "Always focus on the task at hand and provide structured responses as requested."
)


# ── RLTransition ──────────────────────────────────────────────────────────────

@dataclass
class RLTransition:
    """
    One VLM call within an episode.

    Attributes:
        prompt_token_ids:  Full prompt token IDs (for GRPO Datum model_input).
        action_token_ids:  Generated output token IDs.
        action_logprobs:   Per-token sampling log-probs (IS-ratio denominator).
        action_text:       Decoded action text (for logging).
    """
    prompt_token_ids: List[int]
    action_token_ids: List[int]
    action_logprobs:  List[float]
    action_text:      str


# ── RLSamplingVLM ─────────────────────────────────────────────────────────────

class RLSamplingVLM:
    """
    Tinker SamplingClient-backed VLM for GRPO training.

    Implements the same interface as VisionLanguageModel
    (invoke / invoke_with_images) so it can be injected into
    XAIPipelineV2 via the ``vlm`` constructor parameter.

    Every call to invoke() / invoke_with_images():
      1. Builds a Tinker ModelInput in Qwen3-VL chat format.
      2. Calls sampling_client.sample(...).result()  [synchronous].
      3. Decodes tokens to text.
      4. Appends an RLTransition to self._transitions.

    Training loop interface:
      - reset()                         — clear transitions before each episode
      - update_sampling_client(client)  — swap in new weights after training step
      - get_transitions()               — retrieve recorded transitions

    Args:
        sampling_client:  tinker.SamplingClient (current policy snapshot)
        tokenizer:        HuggingFace tokenizer for the Tinker model
        max_new_tokens:   Max tokens generated per VLM call (default 512)
        temperature:      Sampling temperature (default 1.0)
    """

    def __init__(
        self,
        sampling_client,
        tokenizer,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
    ):
        self.sampling_client = sampling_client
        self.tokenizer       = tokenizer
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self._transitions: List[RLTransition] = []

    # ── Episode bookkeeping ───────────────────────────────────────────────────

    def reset(self):
        """Clear recorded transitions. Call at the start of each episode."""
        self._transitions = []

    def update_sampling_client(self, sampling_client):
        """Swap in a new SamplingClient (e.g. after a weight update step)."""
        self.sampling_client = sampling_client

    def get_transitions(self) -> List[RLTransition]:
        """Return a copy of the recorded transitions for this episode."""
        return list(self._transitions)

    # ── VLM interface (BaseAgent.invoke_vlm compatible) ───────────────────────

    def invoke(
        self,
        prompt: str,
        images=None,
        **kwargs,
    ) -> str:
        """
        Text-only (or optionally multimodal) VLM call.

        Compatible with VisionLanguageModel.invoke() as called by
        BaseAgent.invoke_vlm().

        Args:
            prompt: Text prompt
            images: Optional images (PIL list or path list); delegates to
                    invoke_with_images() when provided.
            **kwargs: Ignored (API compatibility).

        Returns:
            Generated text
        """
        if images:
            imgs = images if isinstance(images, list) else [images]
            return self.invoke_with_images(prompt, imgs)
        model_input, prompt_ids = self._build_text_only_input(prompt)
        return self._sample_and_record(model_input, prompt_ids)

    def invoke_with_images(
        self,
        prompt: str,
        image_paths: List[Any],
        **kwargs,
    ) -> str:
        """
        Multimodal VLM call with images.

        Compatible with VisionLanguageModel.invoke_with_images().

        Args:
            prompt:      Text prompt
            image_paths: List of PIL Images or image path strings
            **kwargs:    Ignored (API compatibility).

        Returns:
            Generated text
        """
        model_input, prompt_ids = self._build_multimodal_input(prompt, image_paths)
        return self._sample_and_record(model_input, prompt_ids)

    # Alias used by some agents
    def invoke_multimodal(self, prompt: str, image_paths: List[Any], **kwargs) -> str:
        return self.invoke_with_images(prompt, image_paths, **kwargs)

    # ── ModelInput builders (Qwen3-VL chat format) ────────────────────────────

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_text_only_input(self, prompt: str) -> Tuple[Any, List[int]]:
        """
        Build a Tinker ModelInput for text-only prompts.

        Uses Qwen3-VL chat format:
            <|im_start|>system\\n{system}<|im_end|>\\n
            <|im_start|>user\\n{prompt}<|im_end|>\\n
            <|im_start|>assistant\\n

        Returns:
            (model_input, prompt_token_ids)
        """
        import tinker

        full_text = (
            f"<|im_start|>system\n{_SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        token_ids   = self._encode_text(full_text)
        model_input = tinker.ModelInput.from_ints(token_ids)
        return model_input, token_ids

    def _build_multimodal_input(
        self,
        prompt: str,
        images: List[Any],
    ) -> Tuple[Any, List[int]]:
        """
        Build a Tinker ModelInput with interleaved text + image chunks.

        Follows the Qwen3-VL multimodal chat format with EncodedTextChunk
        and ImageChunk objects (same as TinkerVisionLanguageModel._build_model_input).

        Returns:
            (model_input, approx_prompt_token_ids)
        """
        import tinker
        from tinker import types as tinker_types

        chunks: list = []
        approx_ids: List[int] = []

        def _add_text(text: str):
            ids = self._encode_text(text)
            if ids:
                chunks.append(tinker_types.EncodedTextChunk(tokens=ids))
                approx_ids.extend(ids)

        def _img_to_bytes(img) -> bytes:
            buf = io.BytesIO()
            if hasattr(img, "save"):          # PIL Image
                img.save(buf, format="PNG")
            else:                             # file path — always convert to PNG
                from PIL import Image as _PILImage
                _PILImage.open(img).convert("RGB").save(buf, format="PNG")
            return buf.getvalue()

        _add_text(f"<|im_start|>system\n{_SYSTEM_MESSAGE}<|im_end|>\n")
        _add_text("<|im_start|>user\n")

        for img in images:
            _add_text("<|vision_start|>")
            chunks.append(tinker_types.ImageChunk(data=_img_to_bytes(img), format="png"))
            _add_text("<|vision_end|>\n")

        _add_text(prompt)
        _add_text("<|im_end|>\n")
        _add_text("<|im_start|>assistant\n")

        model_input = tinker.ModelInput(chunks=chunks)
        return model_input, approx_ids

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _sample_and_record(
        self,
        model_input,
        prompt_token_ids: List[int],
    ) -> str:
        """
        Sample one response from Tinker, record the transition, return text.

        Resolves the Tinker future synchronously via .result() — valid inside
        an async context (blocks briefly for the network round-trip).
        """
        import tinker

        sampling_params = tinker.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            logprobs=1,   # request per-token log-probs for GRPO IS ratio
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        result_future = self.sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        )

        # Resolve synchronously
        if hasattr(result_future, "result") and callable(result_future.result):
            result = result_future.result()
        else:
            result = result_future

        seq = result.sequences[0]
        action_token_ids: List[int] = list(seq.tokens)

        # Extract per-token sampling log-probs (IS-ratio denominator)
        action_logprobs: List[float] = []
        if hasattr(seq, "logprobs") and seq.logprobs:
            for tok_id, lp in zip(action_token_ids, seq.logprobs):
                if isinstance(lp, dict):
                    action_logprobs.append(float(lp.get(tok_id, 0.0)))
                elif isinstance(lp, (int, float)):
                    action_logprobs.append(float(lp))
                else:
                    action_logprobs.append(0.0)
        else:
            print(
                "  [RLSamplingVLM] WARNING: no per-token logprobs returned. "
                "GRPO IS ratio will be un-normalised (all probs treated as 1)."
            )
            action_logprobs = [0.0] * len(action_token_ids)

        action_text = self.tokenizer.decode(action_token_ids, skip_special_tokens=True)

        self._transitions.append(RLTransition(
            prompt_token_ids=prompt_token_ids,
            action_token_ids=action_token_ids,
            action_logprobs=action_logprobs,
            action_text=action_text,
        ))

        return action_text

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "type":           "RLSamplingVLM",
            "max_new_tokens": self.max_new_tokens,
            "temperature":    self.temperature,
        }
