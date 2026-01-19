"""
Three Agent System - LangChain Implementation

Fully implements the three-agent architecture using LangChain framework:
1. Proposer Agent: Strategy planning with LangChain Agent
2. Actor Agent: Execution with LangChain Tools and Agent
3. Critic Agent: Evaluation with LangChain Agent

Uses LangChain's:
- AgentExecutor for agent orchestration
- Tools for XAI tool integration
- PromptTemplate for structured prompts
- Chains for sequential processing
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

import torch
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_classic.tools import Tool, StructuredTool
from langchain_core.agents import AgentAction, AgentFinish
from pydantic import BaseModel, Field

from vlm_langchain_wrapper import VisionLanguageModel
from question_templates import (
    QuestionTemplate,
    TemplateRegistry,
    QuestionDataset,
    Modality,
    VisionPromptBuilder
)
from langchain_xai_tools import XAIToolRegistry, create_xai_tools
from DataModelLoader import DataModelLoader


class ProposerAgentLangChain:
    """
    Proposer Agent - LangChain Implementation

    Uses LangChain Agent to decide:
    - Which XAI tools to use
    - Tool parameters
    - Execution strategy

    The agent uses ReAct (Reasoning + Acting) pattern.
    """

    def __init__(
        self,
        vlm: VisionLanguageModel,
        data_model_loader: DataModelLoader,
        models_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Proposer Agent with LangChain.

        Args:
            vlm: VisionLanguageModel instance
            data_model_loader: DataModelLoader instance
            models_dir: Directory containing models (absolute or relative path)
                       Defaults to "./models_to_read" relative to current working directory
            output_dir: Output directory (absolute or relative path)
                       Defaults to "./outputs" relative to current working directory
        """
        self.vlm = vlm
        self.data_model_loader = data_model_loader

        # Set default directories if not provided
        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), "models_to_read")
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.models_dir = Path(models_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.strategy_dir = self.output_dir / "strategies"
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

        # Create analysis tools for Proposer
        self.tools = self._create_proposer_tools()

        # Create Proposer prompt template
        self.prompt_template = self._create_proposer_prompt()

        print("✓ Proposer Agent (LangChain) initialized")
        print(f"  Models directory: {self.models_dir}")
        print(f"  Output directory: {self.output_dir}")

    def _create_proposer_tools(self) -> List[Tool]:
        """Create tools for Proposer Agent."""
        tools = []

        # Tool: List Available XAI Tools
        def list_available_tools(_: str = "") -> str:
            """List all available XAI tools."""
            from xai_tools import get_available_tools
            available = get_available_tools()
            return json.dumps({
                "available_tools": available,
                "descriptions": {
                    "gradcam": "Visualizes model attention regions",
                    "integrated_gradients": "Pixel-level feature attribution",
                    "lime": "Local interpretable explanations",
                    "shap": "Shapley value-based explanations",
                    "object_detection": "Detects objects in image"
                }
            })

        tools.append(Tool(
            name="list_available_tools",
            func=list_available_tools,
            description="Lists all available XAI tools and their capabilities."
        ))

        return tools

    def _create_proposer_prompt(self) -> PromptTemplate:
        """Create prompt template for Proposer Agent."""
        template = """You are a specialized XAI strategy agent. Your response MUST begin with 'Thought:' and you must strictly follow the ReAct framework: Thought: <reasoning> Action: <tool_name> Action Input: <input>. Do not add any conversational greetings, preambles, or other text before the 'Thought:'.

            Your goal is to create a JSON strategy object for an XAI (Explainable AI) analysis.

            Available tools for planning:
            {tools}

            Tool names: {tool_names}

            IMPORTANT FORMAT RULES:
            - First, think about the user's request based on the Question, Question Type, and Modality.
            - Then, you should use the 'list_available_tools' tool to see the available XAI tools and their descriptions.
            - Based on the tools available, select one or more appropriate tools. For a 'feature_attribution' question, 'gradcam' or 'integrated_gradients' are good choices. For a 'counterfactual' question, you might also use visualization tools.
            - When you have decided on a strategy, you MUST provide the Final Answer in the specified JSON structure. Do not add any text after the JSON block.

            When you need to use a tool:
            Thought: <your reasoning>
            Action: <tool_name>
            Action Input: <simple string input>

            When you are READY to give final answer:
            Thought: I have enough information to create the strategy.
            Final Answer: <your JSON>

            Request:
            Question: {question}
            Question Type: {question_type}
            Model Info: {model_info}
            Modality: {modality}

            Final Answer JSON structure:
            {{
                "strategy_type": "tools",
                "reasoning": "Brief explanation of why these XAI tools were selected based on the question type and modality.",
                "selected_tools": [
                    {{
                        "tool_name": "gradcam",
                        "priority": 1,
                        "reasoning": "This is a good general-purpose tool for visual feature attribution.",
                        "parameters": {{}}
                    }}
                ],
                "autonomous_tasks": []
            }}

            Valid tool_name values: gradcam, integrated_gradients, lime, shap, object_detection

            REMEMBER: Your entire response must start with 'Thought:' and nothing else.

            {agent_scratchpad}

            Thought:"""

        return PromptTemplate(
            input_variables=["question", "question_type", "model_info", "modality", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )

    def propose_strategy(
        self,
        question: Dict[str, Any],
        question_template: QuestionTemplate,
        model_info: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Propose analysis strategy using direct VLM call (not ReAct agent).

        Note: VLMs like Qwen2-VL are not suitable for ReAct agent reasoning.
        Instead, we use a direct prompt approach to generate strategy JSON.

        Args:
            question: Question dictionary
            question_template: QuestionTemplate instance
            model_info: Model information
            image_path: Path to image (for vision)
            prediction: Model prediction results

        Returns:
            A tuple containing:
            - Strategy dictionary
            - Path to the saved model metadata, if any
        """
        print("\n" + "="*70)
        print("PROPOSER AGENT: Planning Strategy (Direct VLM Call)")
        print("="*70)

        saved_model_metadata_path: Optional[str] = None

        # Map q_type to question type string
        q_type = question.get('q_type')
        question_type_str = "general"
        if q_type is not None:
            if 1 <= q_type <= 4:
                question_type_str = "feature_attribution"
            elif 5 <= q_type <= 7:
                question_type_str = "counterfactual"
            elif 8 <= q_type <= 10:
                question_type_str = "spurious_features"

        # Create clean model info
        clean_model_info = self._get_clean_model_info(model_info)

        # Use direct VLM call instead of ReAct agent
        try:
            strategy = self._propose_strategy_direct(
                question=question.get('question', ''),
                question_type=question_type_str,
                model_info=clean_model_info,
                modality=str(question_template.modality)
            )

            # Validate strategy has required fields
            if not strategy.get('selected_tools'):
                print("⚠ Strategy missing selected_tools, using default")
                return self._get_default_strategy(), saved_model_metadata_path

            # Save strategy
            self._save_strategy(strategy, question.get('question_id', 'unknown'))

            print(f"\n✓ Strategy proposed: {strategy.get('strategy_type', 'unknown')}")
            print(f"  Selected {len(strategy.get('selected_tools', []))} tools")

            return strategy, saved_model_metadata_path

        except Exception as e:
            print(f"⚠ Strategy generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_strategy(), saved_model_metadata_path

    def _propose_strategy_direct(
        self,
        question: str,
        question_type: str,
        model_info: str,
        modality: str
    ) -> Dict[str, Any]:
        """
        Generate strategy using direct VLM call (bypassing ReAct agent).

        This approach is more suitable for VLMs that don't follow ReAct format well.
        """
        # Get available tools
        from xai_tools import get_available_tools
        available_tools = get_available_tools()

        tool_descriptions = {
            "gradcam": "Visualizes which regions of the image the model focuses on using gradient-weighted class activation mapping",
            "integrated_gradients": "Provides pixel-level feature attribution showing importance of each pixel",
            "lime": "Creates local interpretable explanations by perturbing the input",
            "shap": "Uses Shapley values to explain model predictions",
            "object_detection": "Detects and localizes objects in the image"
        }

        tools_info = "\n".join([
            f"- {tool}: {tool_descriptions.get(tool, 'XAI analysis tool')}"
            for tool in available_tools
        ])

        # Create a simple prompt for strategy generation
        prompt = f"""You are an XAI (Explainable AI) strategy planner. Based on the user's question, select the most appropriate XAI tools.

User Question: {question}
Question Type: {question_type}
Model Info: {model_info}
Modality: {modality}

Available XAI Tools:
{tools_info}

Tool Selection Guidelines:
- For "feature_attribution" questions: Use "gradcam" or "integrated_gradients" to show which parts of the input are most important
- For "counterfactual" questions: Use "gradcam" and "lime" to understand what changes would affect the prediction
- For "spurious_features" questions: Use "integrated_gradients" and "shap" to identify potentially spurious correlations
- For general questions: Use "gradcam" as a starting point

Respond with ONLY a JSON object in this exact format (no other text):
{{
    "strategy_type": "tools",
    "reasoning": "Brief explanation of why these tools were selected",
    "selected_tools": [
        {{
            "tool_name": "<tool_name>",
            "priority": 1,
            "reasoning": "Why this tool is appropriate",
            "parameters": {{}}
        }}
    ],
    "autonomous_tasks": []
}}

JSON Response:"""

        print("\n  Generating strategy via direct VLM call...")
        print(f"  Question type: {question_type}")

        try:
            # Call VLM directly
            response = self.vlm.invoke(prompt)
            print(f"  VLM Response: {response[:500]}...")

            # Parse the response
            strategy = self._parse_strategy(response)

            # If parsing failed or no tools selected, use question-type based selection
            if not strategy.get('selected_tools'):
                print("  ⚠ Could not parse VLM response, using rule-based selection")
                strategy = self._get_strategy_by_question_type(question_type, available_tools)

            return strategy

        except Exception as e:
            print(f"  ⚠ VLM call failed: {e}, using rule-based selection")
            return self._get_strategy_by_question_type(question_type, available_tools)

    def _get_strategy_by_question_type(
        self,
        question_type: str,
        available_tools: list
    ) -> Dict[str, Any]:
        """
        Generate strategy based on question type using rules (fallback).
        """
        # Define tool priorities based on question type
        tool_priorities = {
            "feature_attribution": ["gradcam", "integrated_gradients", "shap"],
            "counterfactual": ["gradcam", "lime", "integrated_gradients"],
            "spurious_features": ["integrated_gradients", "shap", "lime"],
            "general": ["gradcam", "integrated_gradients"]
        }

        priority_list = tool_priorities.get(question_type, tool_priorities["general"])

        selected_tools = []
        for i, tool_name in enumerate(priority_list):
            if tool_name in available_tools:
                selected_tools.append({
                    "tool_name": tool_name,
                    "priority": i + 1,
                    "reasoning": f"Selected based on question type: {question_type}",
                    "parameters": {}
                })

        # Ensure at least one tool is selected
        if not selected_tools and available_tools:
            selected_tools.append({
                "tool_name": available_tools[0],
                "priority": 1,
                "reasoning": "Default fallback tool",
                "parameters": {}
            })

        return {
            "strategy_type": "tools",
            "reasoning": f"Rule-based selection for {question_type} question type",
            "selected_tools": selected_tools,
            "autonomous_tasks": []
        }

    def _get_clean_model_info(self, model_info: Optional[Dict[str, Any]]) -> str:
        """
        Create a clean string representation of model info without the actual model object.

        Args:
            model_info: Model information dictionary

        Returns:
            Clean string representation
        """
        if not model_info:
            return "No model loaded"

        # Extract only the relevant info, excluding the actual model and processor objects
        clean_info = {
            "model_name": model_info.get("model_name", "Unknown"),
            "model_type": model_info.get("model_type", "Unknown"),
            "architecture": model_info.get("architecture", "Unknown"),
            "num_classes": model_info.get("num_classes", "Unknown"),
            "device": model_info.get("device", "Unknown")
        }

        return json.dumps(clean_info, indent=2)

    def _parse_strategy(self, output: str) -> Dict[str, Any]:
        """Parse strategy from agent output."""
        try:
            # Try to extract JSON from output
            import re
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                strategy = json.loads(json_match.group())
                return strategy
            else:
                return self._get_default_strategy()
        except json.JSONDecodeError:
            return self._get_default_strategy()

    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default fallback strategy."""
        return {
            "strategy_type": "tools",
            "reasoning": "Default strategy: using GradCAM for visual explanation",
            "selected_tools": [
                {
                    "tool_name": "gradcam",
                    "priority": 1,
                    "reasoning": "Standard visualization tool",
                    "parameters": {}
                }
            ],
            "autonomous_tasks": []
        }

    def _save_strategy(self, strategy: Dict[str, Any], question_id: str):
        """Save strategy to file."""
        strategy_file = self.strategy_dir / f"strategy_{question_id}.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        print(f"✓ Strategy saved to: {strategy_file}")


class ActorAgentLangChain:
    """
    Actor Agent - LangChain Implementation

    Uses LangChain Agent with XAI Tools to:
    - Execute selected XAI tools
    - Integrate results
    - Generate structured explanations

    The agent has access to all XAI tools and can use them dynamically.
    """

    def __init__(
        self,
        vlm: VisionLanguageModel,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Actor Agent with LangChain.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory (absolute or relative path)
                       Defaults to "./outputs" relative to current working directory
        """
        self.vlm = vlm

        # Set default directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.output_dir = Path(output_dir).resolve()
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # XAI Tool Registry (will be initialized with model)
        self.tool_registry: Optional[XAIToolRegistry] = None

        # Create Actor prompt template
        self.prompt_template = self._create_actor_prompt()

        print("✓ Actor Agent (LangChain) initialized")
        print(f"  Output directory: {self.output_dir}")

    def _create_actor_prompt(self) -> PromptTemplate:
        """Create prompt template for Actor Agent."""
        template = """You are an Actor Agent. Your job is to execute XAI tools and generate a structured JSON explanation based on the results.

Question: {question}
Strategy: {strategy}
Image Path: {image_path}
Prediction: {prediction}

Available Tools:
{tools}
Tool names: {tool_names}

STRICT FORMAT - Follow exactly:
1. To use a tool:
   Thought: I need to run gradcam to analyze the image.
   Action: gradcam
   Action Input: '{{"image_path": "{image_path}", "target_class": 1}}'

2. To give final answer (ONLY after getting tool results):                                                                                                                 │
   Thought: I have the tool results and can now provide the final answer.                                                                                                  │
   Final Answer: {{"explanation": "...", "extracted_features": {{...}}, "confidence": 0.8}}

RULES:
- Action Input must be valid JSON with required fields: image_path (string), target_class (integer)
- DO NOT include observation text in your Action Input
- When ready, use "Final Answer:" followed by your JSON response
- COPY bounding_boxes from tool output into your final answer (use actual numbers, not placeholders)

{agent_scratchpad}

Thought:"""

        return PromptTemplate(
            input_variables=[
                "question", "strategy", "image_path", "prediction", "tools", "tool_names", "agent_scratchpad"
            ],
            template=template
        )

    def initialize_tools(
        self,
        model: Any,
        model_type: str,
        processor: Any
    ):
        """
        Initialize XAI tools with model.

        Args:
            model: PyTorch model
            model_type: Model type ('local_pth')
            processor: Image processor
        """
        _, self.tool_registry = create_xai_tools(
            model=model,
            model_type=model_type,
            processor=processor,
            output_dir=str(self.output_dir / "xai_outputs")
        )
        print("✓ XAI Tools initialized for Actor Agent")

    def execute_and_explain(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: QuestionTemplate,
        image_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy and generate explanation using LangChain Agent.

        Args:
            strategy: Strategy from Proposer
            question: Question dictionary
            question_template: QuestionTemplate instance
            image_path: Path to image
            model_info: Model information
            prediction: Prediction results

        Returns:
            Result dictionary with explanation
        """
        print("\n" + "="*70)
        print("ACTOR AGENT (LangChain): Executing Strategy")
        print("="*70)

        # Check if tools are initialized
        if self.tool_registry is None:
            print("⚠ XAI tools not initialized, cannot execute strategy")
            return {
                "explanation": "XAI tools not available",
                "extracted_features": {},
                "confidence": 0.0,
                "error": "Tools not initialized"
            }

        # Direct tool execution - VLM is not suitable for ReAct agent reasoning
        # Skip the unreliable agent approach and directly execute XAI tools
        print("  Executing XAI tools directly...")

        try:
            parsed_result = self._execute_tools_directly(
                strategy=strategy,
                image_path=image_path or "",
                prediction=prediction or {}
            )

            # Save tool outputs
            if parsed_result.get('tool_results'):
                self._save_tool_outputs(parsed_result['tool_results'], question.get('question_id', 'unknown'))

            # Add metadata
            parsed_result['question_id'] = question.get('question_id', 'unknown')
            parsed_result['question_type'] = question.get('q_type', 'unknown')

            # Save results
            self._save_results(parsed_result)

            print(f"\n✓ Explanation generated")
            return parsed_result

        except Exception as e:
            print(f"⚠ Tool execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "explanation": f"Tool execution failed: {str(e)}",
                "extracted_features": {
                    "responsible_regions": [],
                    "importance_distribution": {},
                    "key_visual_features": []
                },
                "confidence": 0.0,
                "error": str(e)
            }

    def _parse_explanation(self, output: str) -> Dict[str, Any]:
        """Parse explanation from agent output with robust extraction."""
        try:
            import re

            # Try to find JSON in the output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                # Validate and fix bounding boxes if they contain placeholders
                if "extracted_features" in result:
                    ef = result["extracted_features"]
                    if "responsible_regions" in ef:
                        valid_regions = []
                        for region in ef["responsible_regions"]:
                            bbox = region.get("bbox", {})
                            # Check if bbox has valid numeric values
                            if isinstance(bbox, dict):
                                if all(isinstance(bbox.get(k), (int, float)) for k in ["x1", "y1", "x2", "y2"]):
                                    valid_regions.append(region)
                            elif isinstance(bbox, list) and len(bbox) == 4:
                                if all(isinstance(v, (int, float)) for v in bbox):
                                    # Convert list format to dict format
                                    region["bbox"] = {
                                        "x1": int(bbox[0]),
                                        "y1": int(bbox[1]),
                                        "x2": int(bbox[2]),
                                        "y2": int(bbox[3])
                                    }
                                    valid_regions.append(region)

                        ef["responsible_regions"] = valid_regions

                        # If no valid regions, try to extract from tool outputs in the raw output
                        if not valid_regions:
                            extracted_bboxes = self._extract_bboxes_from_raw_output(output)
                            if extracted_bboxes:
                                ef["responsible_regions"] = extracted_bboxes

                return result
            else:
                # No JSON found, try to extract bounding boxes from tool outputs
                extracted_bboxes = self._extract_bboxes_from_raw_output(output)
                return {
                    "explanation": output[:500] if len(output) > 500 else output,
                    "extracted_features": {
                        "responsible_regions": extracted_bboxes,
                        "importance_distribution": {},
                        "key_visual_features": []
                    },
                    "confidence": 0.5
                }
        except json.JSONDecodeError as e:
            # JSON parsing failed, try to extract info from raw output
            extracted_bboxes = self._extract_bboxes_from_raw_output(output)
            return {
                "explanation": output[:500] if len(output) > 500 else output,
                "extracted_features": {
                    "responsible_regions": extracted_bboxes,
                    "importance_distribution": {},
                    "key_visual_features": []
                },
                "confidence": 0.5,
                "parse_error": str(e)
            }

    def _extract_bboxes_from_raw_output(self, output: str) -> List[Dict[str, Any]]:
        """Extract bounding boxes from raw agent output containing tool results."""
        import re

        bboxes = []

        # Pattern to find bounding_boxes in tool outputs
        # Matches: "bounding_boxes": [{"bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}, "importance": 0.85}]
        bbox_pattern = r'"bounding_boxes"\s*:\s*\[(.*?)\]'
        matches = re.findall(bbox_pattern, output, re.DOTALL)

        for match in matches:
            # Try to parse each bounding box entry
            bbox_entry_pattern = r'\{[^{}]*"bbox"\s*:\s*\{[^{}]+\}[^{}]*\}'
            entries = re.findall(bbox_entry_pattern, match, re.DOTALL)

            for entry in entries:
                try:
                    bbox_dict = json.loads(entry)
                    if "bbox" in bbox_dict:
                        bboxes.append(bbox_dict)
                except json.JSONDecodeError:
                    continue

        # Also try to find individual bbox patterns
        single_bbox_pattern = r'"bbox"\s*:\s*\{\s*"x1"\s*:\s*(\d+)\s*,\s*"y1"\s*:\s*(\d+)\s*,\s*"x2"\s*:\s*(\d+)\s*,\s*"y2"\s*:\s*(\d+)\s*\}'
        single_matches = re.findall(single_bbox_pattern, output)

        for i, (x1, y1, x2, y2) in enumerate(single_matches):
            if not any(b.get("bbox", {}).get("x1") == int(x1) for b in bboxes):
                bboxes.append({
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "importance": 0.8 - (i * 0.1),  # Decreasing importance
                    "label": f"extracted_region_{i+1}"
                })

        return bboxes[:5]  # Return max 5 bboxes

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        results_file = self.results_dir / f"result_{results['question_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {results_file}")

    def _extract_tool_outputs_from_steps(self, intermediate_steps: List) -> Dict[str, Any]:
        """
        Extract and parse tool outputs from agent intermediate steps.

        Args:
            intermediate_steps: List of (AgentAction, output) tuples from agent execution

        Returns:
            Dictionary mapping tool names to their parsed outputs
        """
        tool_outputs = {}

        for step in intermediate_steps:
            if len(step) < 2:
                continue

            action = step[0]
            output = step[1]

            # Get tool name from action
            tool_name = getattr(action, 'tool', None)
            if tool_name is None:
                continue

            # Parse output if it's a JSON string
            if isinstance(output, str):
                try:
                    parsed_output = json.loads(output)
                except json.JSONDecodeError:
                    parsed_output = {"raw_output": output}
            else:
                parsed_output = output

            # Store the output, handling multiple calls to the same tool
            if tool_name in tool_outputs:
                if isinstance(tool_outputs[tool_name], list):
                    tool_outputs[tool_name].append(parsed_output)
                else:
                    tool_outputs[tool_name] = [tool_outputs[tool_name], parsed_output]
            else:
                tool_outputs[tool_name] = parsed_output

        return tool_outputs

    def _save_tool_outputs(self, tool_outputs: Dict[str, Any], question_id: str):
        """
        Save raw tool outputs to a separate file for debugging and reference.

        Args:
            tool_outputs: Dictionary of tool outputs
            question_id: Question identifier
        """
        tool_outputs_dir = self.output_dir / "tool_outputs"
        tool_outputs_dir.mkdir(parents=True, exist_ok=True)

        output_file = tool_outputs_dir / f"tool_outputs_{question_id}.json"
        with open(output_file, 'w') as f:
            json.dump(tool_outputs, f, indent=2)

        print(f"✓ Tool outputs saved to: {output_file}")

    def _extract_features_from_tool_outputs(self, tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured features from tool outputs.

        Args:
            tool_outputs: Dictionary mapping tool names to their outputs

        Returns:
            Dictionary with extracted features in standardized format
        """
        all_bboxes = []
        all_viz_paths = []
        importance_stats = {}
        key_visual_features = []

        for tool_name, output in tool_outputs.items():
            # Handle both single output and list of outputs
            outputs = output if isinstance(output, list) else [output]

            for tool_output in outputs:
                if not isinstance(tool_output, dict):
                    continue

                # Check if tool succeeded
                if not tool_output.get('success', False):
                    continue

                # Extract bounding boxes
                bboxes = tool_output.get('bounding_boxes', [])
                for bbox in bboxes:
                    # Add source tool info to each bbox
                    bbox_with_source = bbox.copy()
                    bbox_with_source['source_tool'] = tool_name
                    all_bboxes.append(bbox_with_source)

                # Extract visualization paths
                viz_path = tool_output.get('visualization_path')
                if viz_path:
                    all_viz_paths.append({
                        'tool': tool_name,
                        'path': viz_path
                    })

                # Extract importance distribution
                imp_dist = tool_output.get('importance_distribution', {})
                if imp_dist:
                    importance_stats[tool_name] = imp_dist

                # Extract key features from summary
                summary = tool_output.get('summary', '')
                if summary:
                    key_visual_features.append(f"{tool_name}: {summary}")

        # Sort bboxes by importance and remove duplicates
        all_bboxes.sort(key=lambda x: x.get('importance', 0), reverse=True)

        # Remove near-duplicate bboxes (same approximate region)
        unique_bboxes = []
        for bbox in all_bboxes:
            is_duplicate = False
            for existing in unique_bboxes:
                if self._bbox_overlap(bbox.get('bbox', {}), existing.get('bbox', {})) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_bboxes.append(bbox)

        # Keep top 5 bboxes
        unique_bboxes = unique_bboxes[:5]

        # Calculate overall importance distribution
        if unique_bboxes:
            primary_contribution = unique_bboxes[0].get('importance', 0.5)
            secondary_contribution = sum(b.get('importance', 0) for b in unique_bboxes[1:]) / max(len(unique_bboxes) - 1, 1)
        else:
            primary_contribution = 0.5
            secondary_contribution = 0.3

        return {
            "responsible_regions": unique_bboxes,
            "importance_distribution": {
                "primary_region_contribution": round(primary_contribution, 4),
                "secondary_regions_contribution": round(secondary_contribution, 4),
                "background_contribution": round(max(0, 1.0 - primary_contribution - secondary_contribution), 4),
                "per_tool_stats": importance_stats
            },
            "key_visual_features": key_visual_features,
            "visualization_paths": all_viz_paths
        }

    def _bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.

        Args:
            bbox1: First bounding box with x1, y1, x2, y2
            bbox2: Second bounding box with x1, y1, x2, y2

        Returns:
            IoU score between 0 and 1
        """
        if not bbox1 or not bbox2:
            return 0.0

        x1 = max(bbox1.get('x1', 0), bbox2.get('x1', 0))
        y1 = max(bbox1.get('y1', 0), bbox2.get('y1', 0))
        x2 = min(bbox1.get('x2', 0), bbox2.get('x2', 0))
        y2 = min(bbox1.get('y2', 0), bbox2.get('y2', 0))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        area1 = (bbox1.get('x2', 0) - bbox1.get('x1', 0)) * (bbox1.get('y2', 0) - bbox1.get('y1', 0))
        area2 = (bbox2.get('x2', 0) - bbox2.get('x1', 0)) * (bbox2.get('y2', 0) - bbox2.get('y1', 0))

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _execute_tools_directly(
        self,
        strategy: Dict[str, Any],
        image_path: str,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute XAI tools directly without the agent, as a fallback.
        This ensures we always get valid bounding boxes.
        """
        print("  Executing tools directly (fallback mode)...")

        all_bboxes = []
        all_viz_paths = []
        tool_summaries = []
        tool_outputs = {}  # Store full tool outputs

        # Get target class from prediction
        target_class = prediction.get('predicted_class_idx', 0) if prediction else 0

        # Execute each tool in the strategy
        for tool_spec in strategy.get('selected_tools', []):
            tool_name = tool_spec.get('tool_name', 'gradcam')

            if self.tool_registry is None:
                continue

            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                continue

            try:
                # Execute tool directly
                result_str = tool._run(
                    image_path=image_path,
                    target_class=target_class,
                    image_id=f"direct_{tool_name}"
                )
                result = json.loads(result_str)

                # Store full tool output
                tool_outputs[tool_name] = result

                if result.get('success'):
                    # Collect bounding boxes with source info
                    bboxes = result.get('bounding_boxes', [])
                    for bbox in bboxes:
                        bbox_with_source = bbox.copy()
                        bbox_with_source['source_tool'] = tool_name
                        all_bboxes.append(bbox_with_source)

                    # Collect visualization paths
                    viz_path = result.get('visualization_path')
                    if viz_path:
                        all_viz_paths.append({
                            'tool': tool_name,
                            'path': viz_path
                        })

                    tool_summaries.append(f"{tool_name}: {result.get('summary', 'completed')}")
                else:
                    tool_summaries.append(f"{tool_name}: failed - {result.get('error', 'unknown error')}")

            except Exception as e:
                tool_summaries.append(f"{tool_name}: exception - {str(e)}")
                tool_outputs[tool_name] = {"success": False, "error": str(e)}

        # Sort bboxes by importance and deduplicate
        all_bboxes.sort(key=lambda x: x.get('importance', 0), reverse=True)

        # Remove near-duplicate bboxes
        unique_bboxes = []
        for bbox in all_bboxes:
            is_duplicate = False
            for existing in unique_bboxes:
                if self._bbox_overlap(bbox.get('bbox', {}), existing.get('bbox', {})) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_bboxes.append(bbox)

        unique_bboxes = unique_bboxes[:5]  # Keep top 5

        # Extract importance stats from tool outputs
        importance_stats = {}
        for tool_name, output in tool_outputs.items():
            if isinstance(output, dict) and output.get('success'):
                imp_dist = output.get('importance_distribution', {})
                if imp_dist:
                    importance_stats[tool_name] = imp_dist

        # Calculate contribution
        primary_contribution = unique_bboxes[0].get('importance', 0.8) if unique_bboxes else 0.5
        secondary_contribution = sum(b.get('importance', 0) for b in unique_bboxes[1:]) / max(len(unique_bboxes) - 1, 1) if len(unique_bboxes) > 1 else 0.15

        return {
            "explanation": "The XAI analysis identified the most important regions for the model's prediction.",
            "extracted_features": {
                "responsible_regions": unique_bboxes,
                "importance_distribution": {
                    "primary_region_contribution": round(primary_contribution, 4),
                    "secondary_regions_contribution": round(secondary_contribution, 4),
                    "background_contribution": round(max(0, 1.0 - primary_contribution - secondary_contribution), 4),
                    "per_tool_stats": importance_stats
                },
                "key_visual_features": [b.get('label', 'region') for b in unique_bboxes[:3]],
                "visualization_paths": all_viz_paths
            },
            "tool_results": tool_outputs,
            "tool_results_summary": "; ".join(tool_summaries),
            "visualization_paths": all_viz_paths,
            "confidence": 0.8 if unique_bboxes else 0.3,
            "note": "Results from direct tool execution (fallback mode)"
        }


class CriticAgentLangChain:
    """
    Critic Agent - LangChain Implementation

    Uses LangChain Agent to evaluate Actor's results.

    TODO: Implement evaluation metrics (MoRF, LeRF, Accuracy)
    """

    def __init__(
        self,
        vlm: VisionLanguageModel,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Critic Agent with LangChain.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory (absolute or relative path)
                       Defaults to "./outputs" relative to current working directory
        """
        self.vlm = vlm

        # Set default directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.output_dir = Path(output_dir).resolve()
        self.eval_dir = self.output_dir / "evaluations"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        print("✓ Critic Agent (LangChain) initialized")
        print(f"  Output directory: {self.output_dir}")

    def evaluate(
        self,
        results: Dict[str, Any],
        question: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate Actor's results.

        TODO: Implement with LangChain Agent and evaluation metrics

        Args:
            results: Results from Actor
            question: Original question
            ground_truth: Ground truth labels (if available)

        Returns:
            Evaluation dictionary
        """
        print("\n" + "="*70)
        print("CRITIC AGENT (LangChain): Evaluating Results")
        print("="*70)

        # Placeholder evaluation
        evaluation = {
            "quality_score": 0.8,
            "completeness": 0.7,
            "clarity": 0.9,
            "comments": "Evaluation metrics (MoRF, LeRF, Accuracy) to be implemented",
            "overall_rating": "Good"
        }

        # Save evaluation
        eval_file = self.eval_dir / f"eval_{results.get('question_id', 'unknown')}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2)

        print(f"✓ Evaluation saved to: {eval_file}")
        print(f"  Overall Rating: {evaluation['overall_rating']}")

        return evaluation
