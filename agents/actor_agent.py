"""
Actor Agent for XAI Agent Framework

Responsible for:
- Executing XAI tools based on strategy
- Using VLM reasoning to analyze tool outputs
- Generating structured explanations
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent


class ActorAgent(BaseAgent):
    """
    Actor Agent - XAI Tool Execution and Explanation Generation

    Implements a three-step process:
    1. Execute XAI tools to get raw results (visualizations, statistics)
    2. Use VLM reasoning to extract features from tool results
    3. Generate structured explanation output
    """

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Actor Agent.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory
        """
        super().__init__(vlm, output_dir, "ActorAgent")

        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.tool_registry: Optional[Any] = None

    def initialize_tools(
        self,
        model: Any,
        model_type: str,
        processor: Any,
        modality: str = "vision"
    ):
        """
        Initialize XAI tools with model.

        Args:
            model: PyTorch model
            model_type: Model type
            processor: Data processor
            modality: Data modality
        """
        try:
            from xai_tools_native import create_xai_tools
            self.tool_registry = create_xai_tools(
                model=model,
                model_type=model_type,
                processor=processor,
                modality=modality,
                output_dir=str(self.output_dir / "xai_outputs")
            )
            print("XAI Tools initialized for Actor Agent")
        except ImportError:
            print("Warning: xai_tools_native not available")

    def run(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - execute strategy and generate explanation.

        Args:
            strategy: Strategy from Proposer
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder
            input_path: Path to input data
            model_info: Model information
            prediction: Prediction results

        Returns:
            Result dictionary with explanation
        """
        return self.execute_and_explain(
            strategy, question, question_template,
            input_path, model_info, prediction
        )

    def execute_and_explain(
        self,
        strategy: Dict[str, Any],
        question: Dict[str, Any],
        question_template: Any,
        input_path: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy and generate explanation.

        Args:
            strategy: Strategy from Proposer
            question: Question dictionary
            question_template: QuestionTemplate or PromptBuilder
            input_path: Path to input (image, text, etc.)
            model_info: Model information
            prediction: Prediction results

        Returns:
            Result dictionary with explanation and output
        """
        print("\n" + "=" * 70)
        print("ACTOR AGENT: Executing Strategy")
        print("=" * 70)

        modality = question.get('modality', 'vision')

        try:
            # Step 1: Execute XAI tools
            print("  Step 1: Executing XAI tools...")
            tool_results = self._execute_tools(
                strategy=strategy,
                input_path=input_path or "",
                prediction=prediction or {},
                modality=modality
            )

            # Step 2: Extract features via VLM
            print("  Step 2: Extracting features via VLM reasoning...")
            extracted_features = self._extract_features_via_vlm(
                tool_results=tool_results,
                input_path=input_path,
                question=question,
                question_template=question_template,
                prediction=prediction or {}
            )

            # Step 3: Generate structured explanation
            print("  Step 3: Generating explanation...")
            prompt_builder = self._get_prompt_builder(question_template, question)

            if prompt_builder is not None:
                context = self._build_context(question, model_info, prediction, input_path)
                results_for_prompt = {
                    "tool_results": tool_results.get('tool_results', {}),
                    "extracted_features": extracted_features,
                    "autonomous_results": {}
                }

                parsed_result = self._generate_explanation_with_prompt_builder(
                    prompt_builder=prompt_builder,
                    context=context,
                    strategy=strategy,
                    results=results_for_prompt,
                    tool_results=tool_results
                )
            else:
                # Fallback
                parsed_result = {
                    "explanation": extracted_features.get('explanation', 'Analysis completed.'),
                    "output": self._format_output_from_features(extracted_features, modality),
                    "extracted_features": extracted_features,
                    "confidence": extracted_features.get('confidence', 0.7)
                }

            # Add metadata
            parsed_result['question_id'] = question.get('question_id', 'unknown')
            parsed_result['question_type'] = question.get('q_type', 'unknown')
            parsed_result['tool_results'] = tool_results.get('tool_results', {})
            parsed_result['visualization_paths'] = tool_results.get('visualization_paths', [])

            # Save results
            self._save_results(parsed_result)

            print(f"\nExplanation generated")
            return parsed_result

        except Exception as e:
            print(f"Warning: Execution failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(str(e), question)

    def _execute_tools(
        self,
        strategy: Dict[str, Any],
        input_path: str,
        prediction: Dict[str, Any],
        modality: str = "vision"
    ) -> Dict[str, Any]:
        """Execute XAI tools based on strategy"""
        all_viz_paths = []
        tool_summaries = []
        tool_outputs = {}

        target_class = prediction.get('predicted_class_idx', 0)

        for tool_spec in strategy.get('selected_tools', []):
            tool_name = tool_spec.get('tool_name', 'gradcam')

            if self.tool_registry is None:
                tool_outputs[tool_name] = {"success": False, "error": "Tools not initialized"}
                continue

            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                print(f"  Warning: Tool '{tool_name}' not found")
                continue

            try:
                print(f"  Executing {tool_name}...")
                result_str = tool.run(
                    image_path=input_path,
                    target_class=target_class,
                    image_id=f"direct_{tool_name}"
                )
                result = json.loads(result_str)
                tool_outputs[tool_name] = result

                if result.get('success'):
                    viz_path = result.get('visualization_path')
                    if viz_path:
                        all_viz_paths.append({'tool': tool_name, 'path': viz_path})
                    tool_summaries.append(f"{tool_name}: success")
                else:
                    tool_summaries.append(f"{tool_name}: failed")

            except Exception as e:
                tool_outputs[tool_name] = {"success": False, "error": str(e)}
                tool_summaries.append(f"{tool_name}: error - {str(e)}")

        return {
            "tool_results": tool_outputs,
            "tool_results_summary": "; ".join(tool_summaries),
            "visualization_paths": all_viz_paths,
        }

    def _extract_features_via_vlm(
        self,
        tool_results: Dict[str, Any],
        input_path: Optional[str],
        question: Dict[str, Any],
        question_template: Any,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use VLM to extract features from tool results"""
        try:
            modality = question.get('modality', 'vision')
            q_type = question.get('q_type', 1)

            # Extract image size from tool results (for validation)
            image_size = None
            if modality == "vision":
                for tool_name, tool_result in tool_results.get('tool_results', {}).items():
                    if isinstance(tool_result, dict) and tool_result.get('success'):
                        img_size = tool_result.get('original_image_size', {})
                        if img_size:
                            image_size = (img_size.get('width', 224), img_size.get('height', 224))
                            break

            # Build extraction prompt
            prompt = self._build_feature_extraction_prompt(
                tool_results=tool_results,
                question=question,
                prediction=prediction,
                modality=modality,
                q_type=q_type
            )

            # Collect images for VLM
            images = []
            if modality == "vision" and input_path and os.path.exists(input_path):
                images.append(input_path)
            for viz_info in tool_results.get('visualization_paths', []):
                viz_path = viz_info.get('path', '') if isinstance(viz_info, dict) else str(viz_info)
                if viz_path and os.path.exists(viz_path):
                    images.append(viz_path)

            # Call VLM
            response = self.invoke_vlm(prompt, images if images else None)

            # Parse response with validation
            return self._parse_feature_response(
                response, modality, q_type,
                tool_results=tool_results,
                image_size=image_size
            )

        except Exception as e:
            print(f"Warning: VLM feature extraction failed: {e}")
            return self._get_default_features(question.get('modality', 'vision'))

    def _build_feature_extraction_prompt(
        self,
        tool_results: Dict[str, Any],
        question: Dict[str, Any],
        prediction: Dict[str, Any],
        modality: str,
        q_type: int
    ) -> str:
        """Build prompt for feature extraction"""
        pred_class = prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))
        confidence = prediction.get('confidence', 0.0)

        # Extract image size from tool results (for vision modality)
        image_width, image_height = 224, 224  # Default assumption
        if modality == "vision":
            # Try to get actual image size from any successful tool result
            for tool_name, tool_result in tool_results.get('tool_results', {}).items():
                if isinstance(tool_result, dict) and tool_result.get('success'):
                    img_size = tool_result.get('original_image_size', {})
                    if img_size:
                        image_width = img_size.get('width', image_width)
                        image_height = img_size.get('height', image_height)
                        break

        # Get output format based on modality
        if modality == "vision":
            output_format = '"bounding_box": [x_min, y_min, x_max, y_max]'
            size_constraint = f"""
**CRITICAL IMAGE SIZE CONSTRAINT:**
- Image dimensions: {image_width} x {image_height} pixels
- ALL coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- Example valid bounding box for this image: [10, 20, 80, 70]
- Ensure x_max <= {image_width} and y_max <= {image_height}. Do not hallucinate coordinates outside this range."""
        elif modality == "text":
            output_format = '"start_index": int, "end_index": int'
            text_input = question.get('text_input', '')
            text_length = len(text_input) if text_input else 100
            size_constraint = f"""
**CRITICAL TEXT LENGTH CONSTRAINT:**
- Text length: {text_length} characters
- ALL indices MUST be within: [0, {text_length}]"""
        else:
            output_format = '"feature_key": "string"'
            size_constraint = ""

        prompt = f"""You are an expert XAI analyst. Analyze the XAI results and extract the key feature.

## Context
- Question: {question.get('question', '')}
- Question Type: Q{q_type}
- Modality: {modality}
- Model Prediction: {pred_class} (confidence: {confidence:.2%})
{size_constraint}

## Tool Results Summary
{tool_results.get('tool_results_summary', 'No tools executed')}

## Detailed Tool Statistics
{self._format_tool_statistics(tool_results)}

## Your Task
Based on the XAI analysis, identify THE SINGLE MOST IMPORTANT region/feature that answers the question.

**CRITICAL: Your response MUST be valid JSON with this exact structure:**
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this region/feature is important",
    "confidence": confidence score
}}

**Important Guidelines:**
- For vision: Provide bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates
  - MUST respect image bounds: x in [0, {image_width}], y in [0, {image_height}]
  - Use the top_attention_coords from tool results to determine the region
- For text: Provide character indices (start_index, end_index)
- For tabular: Provide the feature/column name as feature_key
- Focus on the SINGLE most important region/feature, not multiple

JSON Response:"""

        return prompt

    def _format_tool_statistics(self, tool_results: Dict[str, Any]) -> str:
        """Format detailed tool statistics for the prompt"""
        lines = []
        for tool_name, result in tool_results.get('tool_results', {}).items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            stats = result.get('statistics', {})
            if not stats:
                continue

            lines.append(f"\n### {tool_name}:")

            # Image size
            img_size = result.get('original_image_size', {})
            if img_size:
                lines.append(f"- Image size: {img_size.get('width')}x{img_size.get('height')}")

            # Top coordinates (key for determining bounding box)
            top_coords = stats.get('top_attention_coords') or stats.get('top_importance_coords') or stats.get('top_gradient_coords')
            if top_coords and len(top_coords) > 0:
                # Calculate bounding box from top coordinates
                xs = [c.get('x', 0) for c in top_coords]
                ys = [c.get('y', 0) for c in top_coords]
                if xs and ys:
                    lines.append(f"- High attention region: x=[{min(xs)}-{max(xs)}], y=[{min(ys)}-{max(ys)}]")
                    lines.append(f"- Top 3 attention points: {[(c.get('x'), c.get('y')) for c in top_coords[:3]]}")

            # Other useful stats
            if 'mean_attention' in stats:
                lines.append(f"- Mean attention: {stats['mean_attention']:.3f}")
            if 'high_attention_ratio' in stats:
                lines.append(f"- High attention ratio: {stats['high_attention_ratio']:.2%}")

        return "\n".join(lines) if lines else "No detailed statistics available."

    def _parse_feature_response(
        self,
        response: str,
        modality: str,
        q_type: int,
        tool_results: Optional[Dict[str, Any]] = None,
        image_size: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """Parse VLM feature extraction response with validation"""
        parsed = self.parse_json_response(response)

        if not parsed:
            return self._get_default_features(modality)

        # Ensure output field exists
        if 'output' not in parsed:
            parsed['output'] = self._extract_output_from_parsed(parsed, modality)

        # Validate and fix bounding box for vision modality
        if modality == "vision" and image_size:
            parsed['output'] = self._validate_and_fix_bounding_box(
                parsed.get('output', {}),
                image_size,
                tool_results
            )

        return {
            "output": parsed.get('output', {}),
            "explanation": parsed.get('explanation', ''),
            "confidence": parsed.get('confidence', 0.5),
            "raw_response": response[:500]
        }

    def _validate_and_fix_bounding_box(
        self,
        output: Dict[str, Any],
        image_size: tuple,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate bounding box and fix if out of bounds or too small"""
        width, height = image_size
        bbox = output.get('bounding_box', [])

        # Minimum size: at least 10% of image dimensions
        min_width = max(int(width * 0.1), 10)
        min_height = max(int(height * 0.1), 10)

        if not bbox or len(bbox) != 4:
            # No bounding box, try to compute from tool results
            print(f"[ActorAgent] No valid bbox provided, computing from tool results...")
            return self._compute_bbox_from_tools(tool_results, image_size)

        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Check if bbox is completely out of bounds
        if x_min >= width or y_min >= height or x_max <= 0 or y_max <= 0:
            print(f"[ActorAgent] WARNING: VLM returned out-of-bounds bbox {bbox} for {width}x{height} image")
            print(f"[ActorAgent] Computing bounding box from tool results instead...")
            return self._compute_bbox_from_tools(tool_results, image_size)

        # Check if bbox is too small
        if bbox_width < min_width or bbox_height < min_height:
            print(f"[ActorAgent] WARNING: VLM returned too-small bbox {bbox} ({bbox_width}x{bbox_height} pixels)")
            print(f"[ActorAgent] Minimum required size: {min_width}x{min_height} pixels")

            # First, try to use object detection bbox if available
            obj_det_bbox = self._get_object_detection_bbox(tool_results, image_size)
            if obj_det_bbox:
                print(f"[ActorAgent] Using object detection bbox instead: {obj_det_bbox}")
                return {"bounding_box": obj_det_bbox}

            # Otherwise, expand the small bbox around its center
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # Expand to minimum size
            half_w = max(bbox_width / 2, min_width / 2)
            half_h = max(bbox_height / 2, min_height / 2)

            x_min = max(0, int(center_x - half_w))
            x_max = min(width, int(center_x + half_w))
            y_min = max(0, int(center_y - half_h))
            y_max = min(height, int(center_y + half_h))

            print(f"[ActorAgent] Expanded bbox to: [{x_min}, {y_min}, {x_max}, {y_max}]")
            return {"bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)]}

        # Clip to image bounds
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(1, min(x_max, width))
        y_max = max(1, min(y_max, height))

        # Ensure min < max
        if x_min >= x_max:
            x_max = min(x_min + min_width, width)
        if y_min >= y_max:
            y_max = min(y_min + min_height, height)

        return {"bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)]}

    def _get_object_detection_bbox(
        self,
        tool_results: Optional[Dict[str, Any]],
        image_size: tuple
    ) -> Optional[List[int]]:
        """Get the best object detection bounding box if available"""
        if not tool_results:
            return None

        width, height = image_size
        best_detection = None
        best_confidence = 0

        for tool_name, result in tool_results.get('tool_results', {}).items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            detections = result.get('detections', [])
            for det in detections:
                conf = det.get('confidence', 0)
                bbox = det.get('bbox', {})
                if bbox and conf > best_confidence:
                    best_confidence = conf
                    best_detection = bbox

        if best_detection:
            x1 = max(0, min(best_detection.get('x1', 0), width))
            y1 = max(0, min(best_detection.get('y1', 0), height))
            x2 = max(0, min(best_detection.get('x2', width), width))
            y2 = max(0, min(best_detection.get('y2', height), height))
            if x2 > x1 and y2 > y1:
                return [int(x1), int(y1), int(x2), int(y2)]

        return None

    def _compute_bbox_from_tools(
        self,
        tool_results: Optional[Dict[str, Any]],
        image_size: tuple
    ) -> Dict[str, Any]:
        """Compute bounding box directly from tool results, prioritizing object detection"""
        width, height = image_size

        # Minimum size: at least 10% of image dimensions
        min_width = max(int(width * 0.1), 10)
        min_height = max(int(height * 0.1), 10)

        if not tool_results:
            # Default to center region (at least 50% of image)
            margin_x = width // 4
            margin_y = height // 4
            return {"bounding_box": [margin_x, margin_y, width - margin_x, height - margin_y]}

        # PRIORITY 1: Use object detection bbox if available
        obj_det_bbox = self._get_object_detection_bbox(tool_results, image_size)
        if obj_det_bbox:
            print(f"[ActorAgent] Using object detection bbox: {obj_det_bbox}")
            return {"bounding_box": obj_det_bbox}

        # PRIORITY 2: Compute from attention/importance coordinates
        all_xs = []
        all_ys = []

        for tool_name, result in tool_results.get('tool_results', {}).items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            stats = result.get('statistics', {})

            # Get top coordinates from various tool types
            top_coords = (
                stats.get('top_attention_coords') or
                stats.get('top_importance_coords') or
                stats.get('top_gradient_coords')
            )

            if top_coords:
                for coord in top_coords:
                    x, y = coord.get('x', 0), coord.get('y', 0)
                    if 0 <= x < width and 0 <= y < height:
                        all_xs.append(x)
                        all_ys.append(y)

        if all_xs and all_ys:
            # Compute center of attention region
            center_x = sum(all_xs) / len(all_xs)
            center_y = sum(all_ys) / len(all_ys)

            # Calculate spread of attention points
            x_spread = max(all_xs) - min(all_xs)
            y_spread = max(all_ys) - min(all_ys)

            # Ensure minimum size with extra padding (20% of image or spread, whichever is larger)
            half_w = max(x_spread / 2 + 5, min_width, width * 0.1)
            half_h = max(y_spread / 2 + 5, min_height, height * 0.1)

            x_min = max(0, int(center_x - half_w))
            y_min = max(0, int(center_y - half_h))
            x_max = min(width, int(center_x + half_w))
            y_max = min(height, int(center_y + half_h))

            print(f"[ActorAgent] Computed expanded bbox from attention coords: [{x_min}, {y_min}, {x_max}, {y_max}]")
            return {"bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)]}

        # Fallback to center region (50% of image)
        margin_x = width // 4
        margin_y = height // 4
        print(f"[ActorAgent] Using fallback center region bbox")
        return {"bounding_box": [margin_x, margin_y, width - margin_x, height - margin_y]}

    def _extract_output_from_parsed(self, parsed: Dict, modality: str) -> Dict:
        """Try to extract output from various parsed formats"""
        if modality == "vision":
            # Look for bounding_box in various places
            if 'bounding_box' in parsed:
                return {"bounding_box": parsed['bounding_box']}
            if 'responsible_regions' in parsed and parsed['responsible_regions']:
                region = parsed['responsible_regions'][0]
                bbox = region.get('bbox', region.get('bounding_box'))
                if bbox:
                    if isinstance(bbox, dict):
                        return {"bounding_box": [bbox.get('x1', 0), bbox.get('y1', 0),
                                                  bbox.get('x2', 100), bbox.get('y2', 100)]}
                    return {"bounding_box": bbox}
        elif modality == "text":
            if 'start_index' in parsed:
                return {"start_index": parsed['start_index'], "end_index": parsed.get('end_index', 0)}
        elif modality == "tabular":
            if 'feature_key' in parsed:
                return {"feature_key": parsed['feature_key']}

        return {}

    def _get_default_features(self, modality: str) -> Dict[str, Any]:
        """Return default features when extraction fails"""
        default_output = {
            "vision": {"bounding_box": [0, 0, 100, 100]},
            "text": {"start_index": 0, "end_index": 10},
            "tabular": {"feature_key": "unknown"}
        }
        return {
            "output": default_output.get(modality, {}),
            "explanation": "Feature extraction failed, using default.",
            "confidence": 0.1
        }

    def _format_output_from_features(
        self,
        features: Dict[str, Any],
        modality: str
    ) -> Dict[str, Any]:
        """Format output from extracted features"""
        return features.get('output', self._get_default_features(modality)['output'])

    def _get_prompt_builder(self, question_template: Any, question: Dict) -> Any:
        """Get prompt builder from template or create one"""
        if hasattr(question_template, 'build_actor_prompt'):
            return question_template

        if hasattr(question_template, 'prompt_builder') and question_template.prompt_builder:
            return question_template.prompt_builder

        try:
            from prompts import get_prompt_builder
            return get_prompt_builder(question.get('q_type', 1), question.get('modality', 'vision'))
        except ImportError:
            return None

    def _build_context(
        self,
        question: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]],
        input_path: Optional[str]
    ) -> Dict[str, Any]:
        """Build context for prompt builder"""
        modality = question.get('modality', 'vision')

        context = {
            "user_question": question.get("question", ""),
            "model_info": model_info or {},
            "prediction": prediction or {},
        }

        if modality == "vision":
            context["image_path"] = input_path
        elif modality == "text":
            context["text_input"] = question.get("text_input", "")
        elif modality == "tabular":
            context["input_data"] = question.get("features", {})

        return context

    def _generate_explanation_with_prompt_builder(
        self,
        prompt_builder: Any,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanation using prompt builder"""
        try:
            prompt = prompt_builder.build_actor_prompt(context, strategy, results)

            print("=" * 70)
            print(f"\n DEBUG, EXPLANATION PROMPT: {prompt}")

            response = self.invoke_vlm(prompt)

            print("=" * 70)
            print(f"\n DEBUG, EXPLANATION RESPONSE: {response}")

            parsed = self.parse_json_response(response)

            if parsed:
                # Merge with tool results
                parsed['tool_results'] = tool_results.get('tool_results', {})
                parsed['visualization_paths'] = tool_results.get('visualization_paths', [])
                return parsed

        except Exception as e:
            print(f"  Warning: Prompt builder failed: {e}")
            import traceback
            traceback.print_exc()

        # Fallback: use pre-extracted features directly
        return {
            "explanation": results.get('extracted_features', {}).get('explanation', ''),
            "output": results.get('extracted_features', {}).get('output', {}),
            "confidence": 0.5
        }

    def _create_error_result(self, error: str, question: Dict) -> Dict[str, Any]:
        """Create error result"""
        modality = question.get('modality', 'vision')
        return {
            "explanation": f"Execution failed: {error}",
            "output": self._get_default_features(modality)['output'],
            "confidence": 0.0,
            "error": error,
            "question_id": question.get('question_id', 'unknown')
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file"""
        question_id = results.get('question_id', 'unknown')
        filepath = self.save_json(results, f"result_{question_id}", "results")
        print(f"Results saved to: {filepath}")
