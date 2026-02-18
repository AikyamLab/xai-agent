"""
Tree of Thought (ToT) Baseline Agent

Generates multiple candidate reasoning paths (branches), scores each
using XAI tool evidence, and selects the best one.

Key ideas:
- **Propose**: VLM generates K diverse tool-selection strategies
- **Execute**: Each strategy's tools are run to produce evidence
- **Evaluate**: VLM scores each branch (0-1) based on evidence quality
- **Select**: Best-scoring branch's answer is used as the final output

Differences from the 3-agent system:
- Explores multiple strategies in parallel instead of one
- Uses a scoring heuristic instead of faithfulness evaluation
- No Critic reflection / improvement loop
- Single-pass (no iterative refinement)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from baselines.base_baseline import BaseBaseline
from baselines.tool_executor import ToolExecutor


@dataclass
class ThoughtNode:
    """Represents one branch in the Tree of Thought."""

    branch_id: int
    strategy: Dict[str, Any]  # Tool selection strategy
    tool_results: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[Dict[str, Any]] = None
    score: float = 0.0
    reasoning: str = ""


class ToTAgent(BaseBaseline):
    """
    Tree of Thought Agent — branching exploration with evaluation.

    Pipeline:
        1. **Propose**: Generate K diverse tool strategies via VLM
        2. **Execute**: Run each strategy's tools
        3. **Generate**: For each branch, VLM produces an answer
        4. **Evaluate**: VLM scores each branch's answer quality
        5. **Select**: Return the highest-scoring branch
    """

    baseline_name = "tot"

    NUM_BRANCHES = 3  # K — number of candidate strategies
    TEMPERATURE_PROPOSE = 0.8  # Higher temperature for diverse proposals

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        num_branches: int = 3,
    ):
        super().__init__(vlm=vlm, output_dir=output_dir, agent_name="ToTAgent")
        self.tool_executor = tool_executor
        self.NUM_BRANCHES = num_branches

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

        print("\n" + "=" * 70)
        print(f"TOT AGENT: Q{q_type} ({modality}) — {self.NUM_BRANCHES} branches")
        print("=" * 70)

        target_class = (prediction or {}).get("predicted_class_idx", 0)
        output_fmt = self.get_output_format_instruction(modality, q_type)

        # Prepare images for VLM
        if modality == "vision" and input_path and not input_path.startswith(("text_", "tabular_")):
            images = [input_path]
        else:
            images = None

        # ----- Step 1: Propose K diverse strategies -----
        print(f"\n  Step 1: Proposing {self.NUM_BRANCHES} strategies...")
        strategies = self._propose_strategies(
            question, model_info, prediction, modality, images
        )

        # ----- Step 2 & 3: Execute tools + Generate answer per branch -----
        branches: List[ThoughtNode] = []
        for i, strategy in enumerate(strategies):
            print(f"\n  Step 2-3: Branch {i + 1}/{self.NUM_BRANCHES}")
            node = ThoughtNode(branch_id=i, strategy=strategy)

            # Execute tools for this branch
            tool_names = strategy.get("tools", [])
            if self.tool_executor and tool_names:
                valid_tools = [
                    t for t in tool_names if t in self.tool_executor.list_tools()
                ]
                if valid_tools:
                    node.tool_results = self.tool_executor.run_tools(
                        tool_names=valid_tools,
                        target_class=target_class,
                        image_path=input_path or "",
                        image_id=f"tot_b{i}",
                        question=question,
                    )

            # Generate answer for this branch
            node.answer = self._generate_branch_answer(
                question, model_info, prediction, modality,
                node.tool_results, strategy, output_fmt, images
            )
            node.reasoning = strategy.get("rationale", "")
            branches.append(node)

        # ----- Step 4: Evaluate & Score branches -----
        print(f"\n  Step 4: Evaluating {len(branches)} branches...")
        self._score_branches(
            branches, question, model_info, prediction, modality, output_fmt, images
        )

        # ----- Step 5: Select best branch -----
        best = max(branches, key=lambda b: b.score)
        print(f"\n  Selected branch {best.branch_id + 1} (score: {best.score:.2f})")

        # Combine all tool results
        all_tool_results = {}
        for b in branches:
            for k, v in b.tool_results.items():
                if k not in all_tool_results:
                    all_tool_results[k] = v

        final_answer = best.answer or {}

        result = {
            "baseline": "tot",
            "question_id": question.get("question_id", "unknown"),
            "question_type": q_type,
            "output": final_answer.get("output", final_answer),
            "explanation": final_answer.get("explanation", ""),
            **final_answer,
            "tool_results": all_tool_results,
            "visualization_paths": [],
            "tot_branches": [
                {
                    "branch_id": b.branch_id,
                    "strategy": b.strategy,
                    "score": b.score,
                    "reasoning": b.reasoning,
                    "answer_preview": str(b.answer)[:200] if b.answer else None,
                }
                for b in branches
            ],
            "selected_branch": best.branch_id,
            "num_branches": len(branches),
        }

        self.save_baseline_result(result, question)
        print(f"  ToT agent complete for Q{q_type}")
        return result

    # ------------------------------------------------------------------
    # Step 1: Propose diverse strategies
    # ------------------------------------------------------------------

    def _propose_strategies(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        modality: str,
        images: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Generate K diverse tool-selection strategies."""
        clean_model = self._clean_model_info(model_info)
        pred = prediction or {}

        available_tools = (
            self.tool_executor.tool_descriptions()
            if self.tool_executor
            else "lime, shap, gradcam, integrated_gradients, object_detection, guided_backprop, layer_cam, sensitivity_analysis"
        )

        prompt = f"""You are an XAI strategy planner. Generate {self.NUM_BRANCHES} DIFFERENT strategies for analyzing a model's prediction.

## Question
{question.get("question", "")}

## Model Information
- Model: {clean_model.get("model_name", "Unknown")}
- Architecture: {clean_model.get("architecture", "Unknown")}
- Prediction: {pred.get("predicted_class_name", pred.get("predicted_class_idx", "Unknown"))} (confidence: {pred.get("confidence", 0.0):.4f})

## Available XAI Tools
{available_tools}

## Instructions
Generate {self.NUM_BRANCHES} diverse strategies. Each strategy should use a DIFFERENT combination of tools
and have a different analytical rationale. Strategies should be complementary, not redundant.

## Required JSON format
{{
    "strategies": [
        {{
            "strategy_id": 0,
            "rationale": "Why this combination of tools is useful",
            "tools": ["tool_name_1", "tool_name_2"]
        }},
        {{
            "strategy_id": 1,
            "rationale": "Different reasoning approach",
            "tools": ["tool_name_3", "tool_name_4"]
        }},
        ...
    ]
}}

JSON Response:"""

        response = self.invoke_vlm(prompt, images)
        parsed = self.parse_json_response(response)

        strategies = []
        if parsed and "strategies" in parsed:
            strategies = parsed["strategies"]

        # Ensure we have at least NUM_BRANCHES strategies (fill with defaults)
        default_tools_map = {
            "vision": [
                ["gradcam", "lime"],
                ["shap", "integrated_gradients"],
                ["layer_cam", "guided_backprop"],
            ],
            "text": [
                ["lime"],
                ["shap"],
                ["lime", "shap"],
            ],
            "tabular": [
                ["lime"],
                ["shap"],
                ["lime", "shap"],
            ],
        }
        defaults = default_tools_map.get(modality, [["lime"], ["shap"], ["lime", "shap"]])

        while len(strategies) < self.NUM_BRANCHES:
            idx = len(strategies) % len(defaults)
            strategies.append({
                "strategy_id": len(strategies),
                "rationale": f"Default strategy {idx}",
                "tools": defaults[idx],
            })

        return strategies[: self.NUM_BRANCHES]

    # ------------------------------------------------------------------
    # Step 3: Generate answer per branch
    # ------------------------------------------------------------------

    def _generate_branch_answer(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        modality: str,
        tool_results: Dict[str, Any],
        strategy: Dict[str, Any],
        output_fmt: str,
        images: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate an answer for a single branch using its tool results."""
        clean_model = self._clean_model_info(model_info)
        pred = prediction or {}

        # Format tool results
        tool_section = ""
        if self.tool_executor and tool_results:
            tool_section = self.tool_executor.format_tool_results(tool_results)
        else:
            tool_section = "No tool results available."

        # Modality data
        data_section = ""
        if modality == "text":
            features = question.get("features", {})
            if isinstance(features, dict):
                if "premise" in features:
                    data_section = f"\nPremise: {features['premise']}\nHypothesis: {features.get('hypothesis', '')}\n"
                else:
                    data_section = f"\nText: {features.get('text', features.get('review_text', ''))[:500]}\n"
        elif modality == "tabular":
            features = question.get("features", {})
            if isinstance(features, dict):
                data_section = "\n".join(f"  {k}: {v}" for k, v in list(features.items())[:15])

        prompt = f"""You are an XAI analyst. Based on the following evidence, answer the question.

## Question
{question.get("question", "")}

## Model Prediction
- Class: {pred.get("predicted_class_name", pred.get("predicted_class_idx", "Unknown"))}
- Confidence: {pred.get("confidence", 0.0):.4f}
{data_section}

## Analysis Strategy
{strategy.get("rationale", "N/A")}

## XAI Tool Results
{tool_section}

## Required JSON format
{{
    "output": {{ {output_fmt} }},
    "explanation": "Your explanation based on the evidence"
}}

JSON Response:"""

        response = self.invoke_vlm(prompt, images)
        parsed = self.parse_json_response(response)
        return parsed if parsed else {"explanation": response[:1000]}

    # ------------------------------------------------------------------
    # Step 4: Score branches
    # ------------------------------------------------------------------

    def _score_branches(
        self,
        branches: List[ThoughtNode],
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        modality: str,
        output_fmt: str,
        images: Optional[List[str]],
    ):
        """Score each branch by asking the VLM to evaluate answer quality."""
        # Build summaries
        branch_summaries = []
        for b in branches:
            tools_used = list(b.tool_results.keys())
            answer_preview = json.dumps(b.answer, default=str)[:300] if b.answer else "No answer"
            num_success = sum(
                1 for r in b.tool_results.values()
                if isinstance(r, dict) and r.get("success")
            )
            branch_summaries.append(
                f"### Branch {b.branch_id + 1}\n"
                f"- Strategy: {b.strategy.get('rationale', 'N/A')}\n"
                f"- Tools used: {tools_used}\n"
                f"- Successful tools: {num_success}/{len(tools_used)}\n"
                f"- Answer preview: {answer_preview}\n"
            )

        prompt = f"""You are an XAI evaluation expert. Score the quality of each candidate answer.

## Question
{question.get("question", "")}

## Candidate Answers
{chr(10).join(branch_summaries)}

## Scoring Criteria
Rate each branch from 0.0 to 1.0 based on:
1. **Evidence quality**: Did the tools produce meaningful results?
2. **Answer specificity**: Is the answer precise (specific coordinates/features)?
3. **Reasoning quality**: Does the explanation follow logically from the evidence?
4. **Answer completeness**: Does it address all parts of the question?

## Required JSON format
{{
    "scores": [
        {{"branch_id": 0, "score": 0.0-1.0, "justification": "..."}},
        {{"branch_id": 1, "score": 0.0-1.0, "justification": "..."}},
        ...
    ]
}}

JSON Response:"""

        response = self.invoke_vlm(prompt, images)
        parsed = self.parse_json_response(response)

        if parsed and "scores" in parsed:
            for score_entry in parsed["scores"]:
                bid = score_entry.get("branch_id", -1)
                score = score_entry.get("score", 0.0)
                for b in branches:
                    if b.branch_id == bid:
                        b.score = float(score)
                        print(
                            f"    Branch {bid + 1}: score={score:.2f} — "
                            f"{score_entry.get('justification', '')[:80]}"
                        )
                        break
        else:
            # Fallback: score by number of successful tools
            for b in branches:
                num_success = sum(
                    1 for r in b.tool_results.values()
                    if isinstance(r, dict) and r.get("success")
                )
                b.score = num_success / max(len(b.tool_results), 1)
                print(f"    Branch {b.branch_id + 1}: fallback score={b.score:.2f}")
