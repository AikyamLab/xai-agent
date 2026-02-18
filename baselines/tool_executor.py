"""
Tool Executor — reusable XAI tool execution logic extracted from ActorAgent.

Both ReActAgent and ToTAgent use this to run individual XAI tools from the
existing tool registry without duplicating the wiring code.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from xai_tools import set_output_dir


class ToolExecutor:
    """
    Wraps the native XAI tool registry for on-demand tool execution.

    Usage::

        executor = ToolExecutor(data_model_loader, output_dir)
        result = executor.run_tool("gradcam", target_class=3, image_path="...", image_id="run1")
        available = executor.list_tools()
    """

    def __init__(
        self,
        data_model_loader: Any,
        output_dir: str,
    ):
        self.data_model_loader = data_model_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize native tool registry (same as ActorAgent.initialize_tools)
        from xai_tools_native import create_xai_tools

        self.tool_registry = create_xai_tools(
            data_model_loader=data_model_loader,
            output_dir=str(self.output_dir / "xai_outputs"),
        )
        print(f"[ToolExecutor] Initialized with {len(self.list_tools())} tools")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_tools(self) -> List[str]:
        """Return names of all available XAI tools."""
        if self.tool_registry is None:
            return []
        return list(self.tool_registry.tools.keys())

    def tool_descriptions(self) -> str:
        """Return formatted string of tool names + descriptions for LLM prompts."""
        if self.tool_registry is None:
            return "No tools available."
        lines = []
        for name, tool in self.tool_registry.tools.items():
            desc = getattr(tool, "description", "No description")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def run_tool(
        self,
        tool_name: str,
        target_class: int,
        image_path: str = "",
        image_id: str = "baseline",
        question: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single XAI tool and return its result dict.

        Args:
            tool_name: Registered tool name (e.g. ``"gradcam"``, ``"lime"``).
            target_class: Target class index for explanation.
            image_path: Path to input data / image.
            image_id: Prefix for output file naming.
            question: Optional question dict for output-dir nesting.

        Returns:
            Tool result dict with ``success``, ``statistics``, etc.
        """
        if self.tool_registry is None:
            return {"success": False, "error": "Tools not initialized"}

        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found. Available: {self.list_tools()}",
            }

        # Set output directory per question if provided
        if question:
            xai_dir = self._resolve_output_dir(question)
            set_output_dir(str(xai_dir))

        try:
            result_str = tool.run(
                image_path=image_path,
                target_class=target_class,
                image_id=f"{image_id}_{tool_name}",
            )
            result = json.loads(result_str)

            # Enrich with suggested_bounding_box (same as ActorAgent logic)
            if result.get("success"):
                stats = result.get("statistics", {})
                top_coords = (
                    stats.get("top_attention_coords")
                    or stats.get("top_importance_coords")
                    or stats.get("top_gradient_coords")
                )
                if top_coords and len(top_coords) > 0:
                    xs = [c.get("x", 0) for c in top_coords if isinstance(c, dict)]
                    ys = [c.get("y", 0) for c in top_coords if isinstance(c, dict)]
                    if xs and ys:
                        result["suggested_bounding_box"] = [
                            min(xs), min(ys), max(xs), max(ys)
                        ]

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_tools(
        self,
        tool_names: List[str],
        target_class: int,
        image_path: str = "",
        image_id: str = "baseline",
        question: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute multiple tools and return ``{tool_name: result_dict}``."""
        results = {}
        for name in tool_names:
            print(f"  [ToolExecutor] Running {name}...")
            results[name] = self.run_tool(
                tool_name=name,
                target_class=target_class,
                image_path=image_path,
                image_id=image_id,
                question=question,
            )
        return results

    def format_tool_results(self, tool_results: Dict[str, Dict]) -> str:
        """Format tool results into a readable string for LLM prompts."""
        lines = []
        for tool_name, result in tool_results.items():
            if not isinstance(result, dict):
                continue
            success = result.get("success", False)
            if not success:
                lines.append(f"### {tool_name}: FAILED — {result.get('error', 'unknown')}")
                continue
            lines.append(f"### {tool_name}: SUCCESS")
            stats = result.get("statistics", {})
            if stats:
                for k, v in stats.items():
                    if k in ("top_attention_coords", "top_importance_coords", "top_gradient_coords"):
                        coords = v[:5] if isinstance(v, list) else v
                        lines.append(f"  {k}: {coords}")
                    elif isinstance(v, float):
                        lines.append(f"  {k}: {v:.4f}")
                    else:
                        lines.append(f"  {k}: {v}")
            bbox = result.get("suggested_bounding_box")
            if bbox:
                lines.append(f"  suggested_bounding_box: {bbox}")
            summary = result.get("summary", result.get("description"))
            if summary:
                lines.append(f"  summary: {summary}")
        return "\n".join(lines) if lines else "No tool results."

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_output_dir(self, question: Dict[str, Any]) -> Path:
        dataset_base = question.get("dataset_base_name", "unknown")
        row_no = question.get("row_no", question.get("question_id", 0))
        modality = question.get("modality", "vision")

        match = re.match(r"(.+?)_(q\d+)(?:_.*)?$", dataset_base)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base
            q_type_str = f"q{question.get('q_type', 1)}"

        instance_suffix = question.get("instance_suffix", "")
        if instance_suffix:
            xai_dir = (
                self.output_dir / "xai_outputs" / modality / dataset_name
                / q_type_str / str(row_no) / instance_suffix.strip("_")
            )
        else:
            xai_dir = (
                self.output_dir / "xai_outputs" / modality / dataset_name
                / q_type_str / str(row_no)
            )
        xai_dir.mkdir(parents=True, exist_ok=True)
        return xai_dir
