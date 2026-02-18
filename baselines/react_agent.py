"""
ReAct (Reasoning + Acting) Baseline Agent

Implements an interleaved Thought → Action → Observation loop where
the VLM decides which XAI tool to call next, observes the result, and
repeats until it has enough information to produce a final answer.

Key differences from the 3-agent system:
- No upfront strategy planning — tools are chosen dynamically
- No separate Actor feature-extraction step — VLM sees raw results
- Single agent loop instead of Proposer→Actor→Critic
- No Critic reflection / improvement
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from baselines.base_baseline import BaseBaseline
from baselines.tool_executor import ToolExecutor


class ReActAgent(BaseBaseline):
    """
    ReAct Agent — interleaved reasoning and tool-use loop.

    Pipeline:
        1. Present question + model info to VLM
        2. Loop (max N iterations):
           a. VLM outputs Thought + Action (tool_name) or Final Answer
           b. If Action → execute tool → append Observation
           c. If Final Answer → break
        3. Parse final JSON answer
    """

    baseline_name = "react"

    MAX_ITERATIONS = 6  # Maximum Thought-Action-Observation cycles

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 6,
    ):
        super().__init__(vlm=vlm, output_dir=output_dir, agent_name="ReActAgent")
        self.tool_executor = tool_executor
        self.MAX_ITERATIONS = max_iterations

    def set_tool_executor(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor

    def run(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None,
        # Multi-instance
        input_paths: Optional[List[str]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        q_type = question.get("q_type", 1)
        modality = question.get("modality", "vision")
        is_multi = question.get("is_multi_instance", False)

        print("\n" + "=" * 70)
        print(f"REACT AGENT: Q{q_type} ({modality})")
        print("=" * 70)

        target_class = (prediction or {}).get("predicted_class_idx", 0)

        # Build initial system context
        system_context = self._build_system_context(
            question, model_info, prediction, input_path, modality
        )

        # Prepare images for VLM
        if modality == "vision" and input_path and not input_path.startswith(("text_", "tabular_")):
            images = [input_path]
        else:
            images = None

        # Available tools string
        available_tools = (
            self.tool_executor.tool_descriptions()
            if self.tool_executor
            else "No tools available."
        )

        # ReAct loop
        trajectory: List[Dict[str, str]] = []
        all_tool_results: Dict[str, Any] = {}
        final_answer = None

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n  --- Iteration {iteration}/{self.MAX_ITERATIONS} ---")

            # Build prompt with full trajectory
            prompt = self._build_react_prompt(
                system_context=system_context,
                available_tools=available_tools,
                trajectory=trajectory,
                output_fmt=self.get_output_format_instruction(modality, q_type),
                iteration=iteration,
                max_iterations=self.MAX_ITERATIONS,
            )

            # Call VLM
            response = self.invoke_vlm(prompt, images)
            print(f"  VLM response preview: {response[:200]}...")

            # Parse VLM response: either an Action or a FinalAnswer
            action = self._parse_react_response(response)

            if action["type"] == "final_answer":
                print(f"  → Final answer produced at iteration {iteration}")
                final_answer = action["content"]
                trajectory.append({
                    "role": "thought",
                    "content": action.get("thought", ""),
                })
                trajectory.append({
                    "role": "final_answer",
                    "content": json.dumps(final_answer) if isinstance(final_answer, dict) else str(final_answer),
                })
                break

            elif action["type"] == "action":
                tool_name = action["tool_name"]
                thought = action.get("thought", "")

                trajectory.append({"role": "thought", "content": thought})
                trajectory.append({
                    "role": "action",
                    "content": f"Tool: {tool_name}",
                })

                # Execute tool
                if self.tool_executor and tool_name in self.tool_executor.list_tools():
                    print(f"  → Executing tool: {tool_name}")
                    tool_result = self.tool_executor.run_tool(
                        tool_name=tool_name,
                        target_class=target_class,
                        image_path=input_path or "",
                        image_id=f"react_iter{iteration}",
                        question=question,
                    )
                    all_tool_results[tool_name] = tool_result

                    # Format observation
                    observation = self.tool_executor.format_tool_results(
                        {tool_name: tool_result}
                    )
                else:
                    observation = f"Error: Tool '{tool_name}' not available. Available tools: {self.tool_executor.list_tools() if self.tool_executor else '[]'}"

                trajectory.append({"role": "observation", "content": observation})
                print(f"  → Observation added ({len(observation)} chars)")

            else:
                # Couldn't parse — treat response as thought and continue
                trajectory.append({
                    "role": "thought",
                    "content": response[:500],
                })
                print("  → Could not parse action, added as thought")

        # If we exhausted iterations without a final answer, force one
        if final_answer is None:
            print("  → Max iterations reached, forcing final answer...")
            final_answer = self._force_final_answer(
                system_context, trajectory, modality, q_type, images
            )

        # Wrap result
        if not isinstance(final_answer, dict):
            final_answer = {"explanation": str(final_answer)[:2000]}

        result = {
            "baseline": "react",
            "question_id": question.get("question_id", "unknown"),
            "question_type": q_type,
            "output": final_answer.get("output", final_answer),
            "explanation": final_answer.get("explanation", ""),
            **final_answer,
            "tool_results": all_tool_results,
            "visualization_paths": [],
            "react_trajectory": trajectory,
            "num_iterations": len([t for t in trajectory if t["role"] == "action"]),
        }

        self.save_baseline_result(result, question)
        print(f"  ReAct agent complete for Q{q_type} ({len(all_tool_results)} tools used)")
        return result

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_context(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        input_path: Optional[str],
        modality: str,
    ) -> str:
        clean_model = self._clean_model_info(model_info)
        pred = prediction or {}

        ctx = f"""## Question
{question.get("question", "")}

## Model Information
- Model: {clean_model.get("model_name", "Unknown")}
- Architecture: {clean_model.get("architecture", "Unknown")}
- Number of classes: {clean_model.get("num_classes", "Unknown")}

## Prediction
- Predicted class: {pred.get("predicted_class_name", pred.get("predicted_class_idx", "Unknown"))}
- Confidence: {pred.get("confidence", 0.0):.4f}
- Top predictions: {pred.get("top5_predictions", "N/A")}
"""
        # Add modality-specific data
        if modality == "text":
            features = question.get("features", {})
            if isinstance(features, dict):
                if "premise" in features:
                    ctx += f"\n## Input Text\nPremise: {features['premise']}\nHypothesis: {features.get('hypothesis', '')}\n"
                else:
                    ctx += f"\n## Input Text\n{features.get('text', features.get('review_text', ''))[:500]}\n"
        elif modality == "tabular":
            features = question.get("features", {})
            if isinstance(features, dict):
                feat_str = "\n".join(f"  {k}: {v}" for k, v in list(features.items())[:15])
                ctx += f"\n## Input Features\n{feat_str}\n"

        return ctx

    def _build_react_prompt(
        self,
        system_context: str,
        available_tools: str,
        trajectory: List[Dict[str, str]],
        output_fmt: str,
        iteration: int,
        max_iterations: int,
    ) -> str:
        prompt = f"""You are an expert XAI (Explainable AI) analyst using a ReAct framework.
You have access to XAI tools to analyze model predictions. Use them to gather evidence before answering.

{system_context}

## Available XAI Tools
{available_tools}

## Instructions
At each step you must output EXACTLY ONE of two options:

**Option A — Use a tool:**
```json
{{
    "thought": "Your reasoning about what to investigate next",
    "action": "tool_name"
}}
```

**Option B — Provide your final answer (when you have enough evidence):**
```json
{{
    "thought": "Your final reasoning summarizing all evidence",
    "final_answer": {{
        "output": {{ {output_fmt} }},
        "explanation": "Your final explanation"
    }}
}}
```

IMPORTANT:
- You are on iteration {iteration}/{max_iterations}. {"Provide your final answer NOW." if iteration >= max_iterations else "Choose wisely."}
- Use tools to gather evidence BEFORE answering
- Each tool reveals different aspects of the model's decision
- Respond with ONLY valid JSON (no markdown code blocks)
"""

        # Append trajectory
        if trajectory:
            prompt += "\n## Trajectory So Far\n"
            for step in trajectory:
                role = step["role"].upper()
                content = step["content"]
                if role == "THOUGHT":
                    prompt += f"\n**Thought:** {content}\n"
                elif role == "ACTION":
                    prompt += f"\n**Action:** {content}\n"
                elif role == "OBSERVATION":
                    prompt += f"\n**Observation:**\n{content}\n"
                elif role == "FINAL_ANSWER":
                    prompt += f"\n**Final Answer:** {content}\n"

            prompt += "\n## Your Next Step\nJSON Response:"
        else:
            prompt += "\n## Your First Step\nJSON Response:"

        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_react_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response into action dict."""
        parsed = self.parse_json_response(response)

        if not parsed:
            return {"type": "unknown", "content": response[:500]}

        # Check for final_answer
        if "final_answer" in parsed:
            fa = parsed["final_answer"]
            return {
                "type": "final_answer",
                "content": fa if isinstance(fa, dict) else {"explanation": str(fa)},
                "thought": parsed.get("thought", ""),
            }

        # Check for action
        if "action" in parsed:
            return {
                "type": "action",
                "tool_name": parsed["action"],
                "thought": parsed.get("thought", ""),
            }

        # If the response itself looks like a final answer (has output key)
        if "output" in parsed:
            return {
                "type": "final_answer",
                "content": parsed,
                "thought": parsed.get("thought", parsed.get("reasoning", "")),
            }

        return {"type": "unknown", "content": parsed}

    def _force_final_answer(
        self,
        system_context: str,
        trajectory: List[Dict[str, str]],
        modality: str,
        q_type: int,
        images: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Force a final answer when max iterations reached."""
        output_fmt = self.get_output_format_instruction(modality, q_type)

        # Summarize observations
        observations = [
            step["content"]
            for step in trajectory
            if step["role"] == "observation"
        ]
        obs_summary = "\n\n".join(observations) if observations else "No observations collected."

        prompt = f"""Based on the following context and evidence, provide your FINAL answer.

{system_context}

## Evidence Collected
{obs_summary}

## Required JSON format
{{
    "output": {{ {output_fmt} }},
    "explanation": "Your explanation based on all evidence"
}}

JSON Response:"""

        response = self.invoke_vlm(prompt, images)
        parsed = self.parse_json_response(response)
        return parsed if parsed else {"explanation": response[:2000]}
