"""
XAI GRPO Environment

Implements the MDP for XAI explanation generation.

The Environment owns all DETERMINISTIC logic:
  - Proposer/Actor prompt construction
  - XAI tool execution (via actor_agent._execute_tools)
  - Faithfulness evaluation (via critic_agent.evaluate_faithfulness)
  - Reflection generation (template-based or LLM-based with frozen weights)

The POLICY (Qwen3-VL-8B + LoRA) generates:
  - strategy JSON  (Turn 0 action)
  - explanation    (Turn 1 action)
  - improved strategy  (Turn 2 action, only when reflection triggered)
  - improved explanation (Turn 3 action, only when reflection triggered)

All policy actions get GRPO gradients. Reflection text is injected as part of
the next observation and does NOT have gradients (it's "environment context").
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Observation:
    """
    A single-turn observation for the model (prompt + optional images).

    In Qwen3-VL format, `messages` is a list of chat turns built so far.
    The model continues from the last user message.

    Attributes:
        messages: Full conversation history as HuggingFace chat messages
                  [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        images:   PIL images from XAI tool outputs (GradCAM, LIME heatmaps, etc.)
        metadata: Extra debug info (turn number, question_id, etc.)
    """
    messages: List[Dict[str, Any]]
    images: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """
    Result of a single environment step.

    Attributes:
        reward:           Scalar reward (0.0 for intermediate turns)
        episode_done:     Whether the episode has ended
        next_observation: Next observation for the model (None if done)
        metrics:          Diagnostic info (faithfulness score, delta, etc.)
    """
    reward: float
    episode_done: bool
    next_observation: Optional[Observation]
    metrics: Dict[str, Any] = field(default_factory=dict)


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_parse_strategy(text: str) -> Dict[str, Any]:
    """
    Extract JSON strategy dict from raw model output.
    Falls back to a pure-reasoning strategy if JSON parsing fails.
    """
    # Try ```json ... ``` block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Fallback: treat as pure reasoning, no tools
    return {
        "selected_tools": [],
        "strategy_type": "pure_reasoning",
        "rationale": text[:300],
    }


def _build_template_reflection(
    score: float,
    threshold: float,
    strategy: Dict[str, Any],
    tool_results: Dict[str, Any],
    tool_importance: Optional[Dict[str, float]] = None,
) -> str:
    """
    Build a deterministic (template-based) reflection string.

    This is the default reflection builder: no LLM call needed.
    It analyses tool importance scores (if available) and gives
    concrete actionable feedback.

    Args:
        score:           Initial faithfulness score
        threshold:       Passing threshold
        strategy:        Strategy dict from the model's previous turn
        tool_results:    Results dict from tool execution
        tool_importance: Optional {tool_name: importance_score} from strategy
                         faithfulness evaluation

    Returns:
        Reflection string to be injected as context in Turn 2 observation.
    """
    lines = [
        f"[Critic Feedback]",
        f"Your previous explanation achieved a faithfulness score of {score:.3f} "
        f"(required threshold: {threshold:.3f}). Improvement is needed.",
        "",
    ]

    used_tools = [t.get("tool_name", "") for t in strategy.get("selected_tools", [])]

    if tool_importance:
        high_impact = [t for t, s in tool_importance.items() if s >= 0.5]
        low_value   = [t for t in used_tools if tool_importance.get(t, 1.0) < 0.2]
        missed      = [t for t in high_impact if t not in used_tools]

        if missed:
            lines.append(
                f"High-impact tools NOT used: {', '.join(missed)}. "
                "Consider including them in your strategy."
            )
        if low_value:
            lines.append(
                f"Low-value tools used: {', '.join(low_value)}. "
                "Consider removing or replacing them."
            )

    if not used_tools:
        lines.append(
            "No XAI tools were selected. Attribution methods (LIME, SHAP, GradCAM, "
            "Occlusion) provide concrete evidence that directly links input features "
            "to model predictions."
        )

    if score < 0.2:
        lines.append(
            "The explanation was highly unfaithful. Ensure your claims are "
            "directly supported by tool outputs and refer to specific features "
            "that change the model's output when modified."
        )
    elif score < threshold:
        lines.append(
            "The explanation partially captures the model's behaviour. "
            "Be more specific: quantify feature importance, report confidence changes, "
            "and avoid vague statements."
        )

    lines += [
        "",
        "Please revise your XAI strategy based on the feedback above.",
    ]
    return "\n".join(lines)


# ── Single-pass Environment (2 turns) ────────────────────────────────────────

class XAIEnv:
    """
    Single-pass XAI environment: Proposer → Actor (2 model turns).

    Turn 0 (Proposer):
      Observation  = question context + model info + available tools
      Action       = strategy JSON string
      Reward       = 0.0
      Next obs     = actor prompt (question + tool results + images)

    Turn 1 (Actor):
      Observation  = actor prompt with tool outputs
      Action       = explanation text
      Reward       = faithfulness_score  ∈ [0, 1]
      Done         = True

    Args:
        question:                 Question dict (q_type, modality, features, …)
        question_template:        QuestionTemplate / PromptBuilder instance
        actor_agent:              Configured ActorAgent (tools already initialized)
        critic_agent:             CriticAgent (model already set via set_model)
        proposer_prompt_fn:       fn(question, template, model_info, prediction,
                                     input_path) -> str
        actor_prompt_fn:          fn(question, template, strategy, tool_results,
                                     model_info, prediction, images) -> (str, images)
        faithfulness_threshold:   Score needed to "pass" (default 0.5)
        input_path:               Path to the input sample (image/CSV/…)
        model_info:               Dict with 'model', 'model_type', etc.
        prediction:               Model prediction dict
        input_tensor:             Raw tensor/array for faithfulness evaluation
        system_prompt:            Optional system-level instruction for the model
    """

    def __init__(
        self,
        question: Dict[str, Any],
        question_template: Any,
        actor_agent: Any,
        critic_agent: Any,
        proposer_prompt_fn: Callable,
        actor_prompt_fn: Callable,
        faithfulness_threshold: float = 0.5,
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        input_tensor: Any = None,
        system_prompt: str = "You are an expert in explainable AI (XAI).",
    ):
        self.question = question
        self.question_template = question_template
        self.actor_agent = actor_agent
        self.critic_agent = critic_agent
        self.proposer_prompt_fn = proposer_prompt_fn
        self.actor_prompt_fn = actor_prompt_fn
        self.faithfulness_threshold = faithfulness_threshold
        self.input_path = input_path
        self.model_info = model_info
        self.prediction = prediction
        self.input_tensor = input_tensor
        self.system_prompt = system_prompt

        # Episode state (reset on each call to initial_observation)
        self._turn: int = 0
        self._strategy: Optional[Dict] = None
        self._tool_results: Optional[Dict] = None
        self._conversation: List[Dict] = []   # accumulated chat history

    # ── Public MDP interface ──────────────────────────────────────────────────

    def initial_observation(self) -> Observation:
        """Reset state and return Turn-0 (Proposer) observation."""
        self._turn = 0
        self._strategy = None
        self._tool_results = None
        self._conversation = [{"role": "system", "content": self.system_prompt}]

        proposer_text = self.proposer_prompt_fn(
            question=self.question,
            question_template=self.question_template,
            model_info=self.model_info,
            prediction=self.prediction,
            input_path=self.input_path,
        )
        self._conversation.append({"role": "user", "content": proposer_text})

        return Observation(
            messages=list(self._conversation),
            metadata={"turn": 0, "question_id": self.question.get("question_id")},
        )

    def step(self, action_text: str) -> StepResult:
        """
        Process one model action and advance the episode.

        Args:
            action_text: Decoded text output from the model

        Returns:
            StepResult with reward, done flag, and next observation.
        """
        # Record the assistant's response in conversation history
        self._conversation.append({"role": "assistant", "content": action_text})

        if self._turn == 0:
            return self._proposer_step(action_text)
        elif self._turn == 1:
            return self._actor_step(action_text)
        else:
            raise RuntimeError(f"XAIEnv: unexpected turn {self._turn}")

    # ── Turn handlers ─────────────────────────────────────────────────────────

    def _proposer_step(self, strategy_text: str) -> StepResult:
        """
        Turn 0: parse strategy, execute XAI tools, build Actor observation.
        """
        self._strategy = _safe_parse_strategy(strategy_text)

        # Execute XAI tools (deterministic, no gradient)
        try:
            self._tool_results = self.actor_agent._execute_tools(
                strategy=self._strategy,
                input_path=self.input_path or "",
                prediction=self.prediction or {},
                modality=self.question.get("modality", "vision"),
                question=self.question,
            )
        except Exception as e:
            self._tool_results = {
                "tool_results": {},
                "visualization_paths": [],
                "tool_results_summary": f"Tool execution failed: {e}",
            }

        # Build Actor prompt + extract any tool images
        actor_text, images = self.actor_prompt_fn(
            question=self.question,
            question_template=self.question_template,
            strategy=self._strategy,
            tool_results=self._tool_results,
            model_info=self.model_info,
            prediction=self.prediction,
        )
        self._conversation.append({"role": "user", "content": actor_text})

        self._turn = 1
        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=Observation(
                messages=list(self._conversation),
                images=images,
                metadata={"turn": 1, "question_id": self.question.get("question_id")},
            ),
            metrics={"tools_executed": list(self._tool_results.get("tool_results", {}).keys())},
        )

    def _actor_step(self, explanation_text: str) -> StepResult:
        """
        Turn 1: evaluate faithfulness → reward.
        """
        q_type = self.question.get("q_type", 1)
        modality = self.question.get("modality", "vision")

        actor_output = {
            "explanation": explanation_text,
            "tool_results": self._tool_results.get("tool_results", {}),
            "question_id": self.question.get("question_id", "unknown"),
        }

        try:
            eval_result = self.critic_agent.evaluate_faithfulness(
                q_type=q_type,
                agent_output=actor_output,
                original_input=self.input_tensor,
                original_prediction=self.prediction or {},
                modality=modality,
            )
            score = eval_result.get("score", 0.0) or 0.0
        except Exception as e:
            score = 0.0
            eval_result = {"error": str(e)}

        passed = score >= self.faithfulness_threshold
        return StepResult(
            reward=score,
            episode_done=True,
            next_observation=None,
            metrics={
                "faithfulness": score,
                "passed": passed,
                "eval_detail": eval_result,
            },
        )


# ── Multi-pass Environment (up to 4 turns, with reflection) ──────────────────

class XAIMultiTurnEnv(XAIEnv):
    """
    Multi-pass XAI environment: adds Critic reflection + Proposer/Actor improvement.

    Inherits XAIEnv and overrides _actor_step to optionally continue the episode:

      Turn 0: Proposer → strategy JSON
      Turn 1: Actor   → explanation → faithfulness eval
              ┌ if score >= threshold:  reward=score, done=True
              └ if score < threshold:
                  reflection text is built by env (no gradient)
      Turn 2: Proposer (sees reflection) → improved strategy JSON
      Turn 3: Actor   → improved explanation → faithfulness eval
              → reward = improved_score, done = True

    GRPO gradient flows through all 4 model outputs.
    The reflection text is pure "environment context" - it has no gradient.

    Extra Args (beyond XAIEnv):
        reflection_builder_fn:  fn(score, threshold, strategy, tool_results,
                                   tool_importance) -> str
                                Defaults to _build_template_reflection.
        max_reflection_turns:   Max number of reflection rounds (default 1).
        reflection_proposer_prompt_fn:
                                fn(conversation, reflection_text, question,
                                   template, model_info, prediction) -> str
                                If None, appends reflection to existing conv.
        reflection_actor_prompt_fn:
                                fn(question, template, strategy, tool_results,
                                   model_info, prediction, reflection) -> (str, images)
                                If None, reuses actor_prompt_fn.
    """

    def __init__(
        self,
        *args,
        reflection_builder_fn: Optional[Callable] = None,
        max_reflection_turns: int = 1,
        reflection_proposer_prompt_fn: Optional[Callable] = None,
        reflection_actor_prompt_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reflection_builder_fn = reflection_builder_fn or _build_template_reflection
        self.max_reflection_turns = max_reflection_turns
        self.reflection_proposer_prompt_fn = reflection_proposer_prompt_fn
        self.reflection_actor_prompt_fn = reflection_actor_prompt_fn

        # Additional state for reflection rounds
        self._reflection_round: int = 0
        self._initial_score: Optional[float] = None
        self._reflection_text: Optional[str] = None

    def initial_observation(self) -> Observation:
        obs = super().initial_observation()
        self._reflection_round = 0
        self._initial_score = None
        self._reflection_text = None
        return obs

    def _actor_step(self, explanation_text: str) -> StepResult:
        """
        Override: after evaluating, decide whether to trigger reflection or end.
        """
        q_type = self.question.get("q_type", 1)
        modality = self.question.get("modality", "vision")

        actor_output = {
            "explanation": explanation_text,
            "tool_results": self._tool_results.get("tool_results", {}),
            "question_id": self.question.get("question_id", "unknown"),
        }

        try:
            eval_result = self.critic_agent.evaluate_faithfulness(
                q_type=q_type,
                agent_output=actor_output,
                original_input=self.input_tensor,
                original_prediction=self.prediction or {},
                modality=modality,
            )
            score = eval_result.get("score", 0.0) or 0.0
        except Exception as e:
            score = 0.0
            eval_result = {"error": str(e)}

        # ── Turn 1: initial evaluation ────────────────────────────────────────
        if self._turn == 1:
            self._initial_score = score
            passed = score >= self.faithfulness_threshold
            can_reflect = self._reflection_round < self.max_reflection_turns

            if passed or not can_reflect:
                # Episode ends cleanly
                return StepResult(
                    reward=score,
                    episode_done=True,
                    next_observation=None,
                    metrics={
                        "faithfulness": score,
                        "passed": passed,
                        "reflection_triggered": False,
                    },
                )
            else:
                # Trigger reflection loop
                return self._trigger_reflection(score, eval_result)

        # ── Turn 3+: improved evaluation ──────────────────────────────────────
        else:
            delta = score - (self._initial_score or 0.0)
            return StepResult(
                reward=score,
                episode_done=True,
                next_observation=None,
                metrics={
                    "faithfulness": score,
                    "initial_faithfulness": self._initial_score,
                    "delta": delta,
                    "reflection_triggered": True,
                    "reflection_round": self._reflection_round,
                },
            )

    def _trigger_reflection(self, score: float, eval_detail: Dict) -> StepResult:
        """
        Build the reflection observation and advance to Turn 2.

        The reflection text is deterministic (template-based by default).
        It is appended as a user message so the model can see it as context.
        """
        self._reflection_round += 1

        # Build reflection text (no LLM call by default)
        self._reflection_text = self.reflection_builder_fn(
            score=score,
            threshold=self.faithfulness_threshold,
            strategy=self._strategy,
            tool_results=self._tool_results,
            tool_importance=eval_detail.get("tool_importance"),
        )

        # Build improved-Proposer prompt
        if self.reflection_proposer_prompt_fn is not None:
            improved_proposer_text = self.reflection_proposer_prompt_fn(
                reflection_text=self._reflection_text,
                question=self.question,
                question_template=self.question_template,
                model_info=self.model_info,
                prediction=self.prediction,
                original_strategy=self._strategy,
            )
        else:
            # Default: append reflection to conversation naturally
            original_proposer_text = self.proposer_prompt_fn(
                question=self.question,
                question_template=self.question_template,
                model_info=self.model_info,
                prediction=self.prediction,
                input_path=self.input_path,
            )
            improved_proposer_text = (
                f"{original_proposer_text}\n\n"
                f"{self._reflection_text}\n\n"
                f"Based on the feedback above, revise your XAI strategy."
            )

        self._conversation.append({"role": "user", "content": improved_proposer_text})

        self._turn = 2   # next step() call goes to _proposer_step() via the turn router

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=Observation(
                messages=list(self._conversation),
                metadata={
                    "turn": 2,
                    "question_id": self.question.get("question_id"),
                    "initial_faithfulness": score,
                    "reflection_round": self._reflection_round,
                },
            ),
            metrics={"initial_faithfulness": score, "reflection_triggered": True},
        )

    def step(self, action_text: str) -> StepResult:
        """
        Route to the right handler based on current turn.

        Turn 0: Proposer (initial)
        Turn 1: Actor   (initial evaluation)
        Turn 2: Proposer (after reflection)
        Turn 3: Actor   (final evaluation)
        """
        self._conversation.append({"role": "assistant", "content": action_text})

        if self._turn == 0:
            return self._proposer_step(action_text)
        elif self._turn == 1:
            return self._actor_step(action_text)
        elif self._turn == 2:
            # Second Proposer: same logic as Turn 0, advances turn to 3
            result = self._proposer_step(action_text)
            # _proposer_step sets self._turn = 1; override to 3
            if not result.episode_done:
                self._turn = 3
            return result
        elif self._turn == 3:
            return self._actor_step(action_text)
        else:
            raise RuntimeError(f"XAIMultiTurnEnv: unexpected turn {self._turn}")
