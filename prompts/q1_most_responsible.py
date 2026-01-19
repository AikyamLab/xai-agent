"""
Q1: Which part of the input was MOST responsible for the model's prediction?

Expected Output:
- Vision: bounding_box [x_min, y_min, x_max, y_max]
- Text: start_index, end_index
- Tabular: feature_key

Evaluation:
- Mask the identified part
- Metric: P_original - P_modified (higher = better, part was indeed important)
"""

from typing import Any, Dict

from .base_prompt import PromptBuilder, QuestionCategory, Modality
from .output_schemas import get_output_schema, schema_to_prompt_string


class Q1MostResponsiblePromptBuilder(PromptBuilder):
    """Prompt builder for Q1: Most responsible part identification"""

    @property
    def question_type(self) -> int:
        return 1

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.FEATURE_ATTRIBUTION

    @property
    def question_template(self) -> str:
        return "Which part of the input was most responsible for the model's prediction?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(1, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer to select XAI tools"""
        prediction = context.get('prediction', {})

        # Modality-specific configurations
        modality_config = self._get_modality_config()

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Prediction: Class {prediction.get('predicted_class_idx')}
(Confidence: {prediction.get('confidence', 0.0):.4f})
- Top-5 Predictions: {prediction.get('top5_predictions', [])}

**Input Content Description**:
{context.get(modality_config['description_key'], 'Not available')}

**Task**: Design a comprehensive strategy to identify which {modality_config['region_type']} of the input were MOST RESPONSIBLE for this prediction.

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Identify and ground important {modality_config['element_type']} in the input
   - Reason about their importance to the predicted class
   - Provide {modality_config['location_type']} and confidence scores for different {modality_config['element_type']}

2. **External XAI Tools**: Use established explainability methods:
{modality_config['tools_description']}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for finding most responsible {modality_config['element_type']} (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {modality_config['tool_list']},
        "tool_params": {{
            {modality_config['tool_params_example']}
        }},
        "reasoning": "Why these tools for {self.modality} modality"
    }}
}}

**Guidelines for {self.modality.upper()} tasks (positive attribution)**:
- {modality_config['spatial_note']}
- Target: Find {modality_config['element_type']} that drive the prediction with highest positive contribution
- Evaluation metric: P_original - P_modified (higher = better, part was indeed important)

Provide your strategy as a JSON object:
"""
        return prompt

    def _get_modality_config(self) -> Dict[str, Any]:
        """Get modality-specific configuration for prompt building"""
        if self.modality == "vision":
            return {
                'input_type': 'an IMAGE',
                'description_key': 'image_description',
                'region_type': 'SPATIAL REGIONS',
                'element_type': 'objects/regions',
                'location_type': 'bounding boxes',
                'spatial_note': 'Consider spatial nature of images - bounding boxes are crucial',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Highlights important regions using gradient-based activation
   - IntegratedGradients: Computes pixel-level importance scores
   - LIME: Segments image and tests which segments affect prediction
   - SHAP: Game-theoretic approach to feature importance
   - ObjectDetection: Detects and localizes objects in the image
   - GuidedBackprop: Visualizes gradients guided by activations
   - SensitivityAnalysis: Measures prediction sensitivity to input changes
   - LayerCAM: Layer-wise class activation mapping''',
                'tool_params_example': '"gradcam": {"layer": "layer4", "priority": 1},\n            "lime": {"num_samples": 1000, "priority": 2}'
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT',
                'description_key': 'text_description',
                'region_type': 'TEXT SPANS',
                'element_type': 'tokens/phrases',
                'location_type': 'start and end indices',
                'spatial_note': 'Consider sequential nature of text - token positions and spans are crucial',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Computes token-level importance scores
   - LIME: Tests which tokens/phrases affect prediction
   - SHAP: Game-theoretic approach to token importance
   - AttentionAnalysis: Analyzes attention weights across tokens
   - TokenImportance: Direct token-level attribution
   - SensitivityAnalysis: Measures prediction sensitivity to token changes''',
                'tool_params_example': '"integrated_gradients": {"baseline": "zero", "priority": 1},\n            "lime": {"num_samples": 500, "priority": 2}'
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA',
                'description_key': 'data_description',
                'region_type': 'FEATURES',
                'element_type': 'features/columns',
                'location_type': 'feature names',
                'spatial_note': 'Consider feature relationships and data types - feature keys are crucial',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Computes feature-level importance scores
   - LIME: Tests which features affect prediction locally
   - SHAP: Game-theoretic approach to feature importance
   - PermutationImportance: Measures importance by permuting feature values
   - SensitivityAnalysis: Measures prediction sensitivity to feature changes''',
                'tool_params_example': '"shap": {"background_samples": 100, "priority": 1},\n            "lime": {"num_samples": 1000, "priority": 2}'
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor to generate explanation"""
        prediction = context.get('prediction', {})
        output_format = self._get_modality_specific_output_format()

        # Get tool results
        tool_results = results.get('tool_results', {})
        extracted = results.get('extracted_features', {})

        # Extract image size from tool results (for vision modality)
        image_width, image_height = self._get_image_size_from_results(tool_results)

        # Format tool results with detailed statistics
        tool_summary = self._format_tool_results(tool_results)
        detailed_stats = self._format_detailed_statistics(tool_results)

        # Build size constraint for the prompt (vision only)
        size_constraint = ""
        bbox_guidance = ""
        if self.modality == "vision" and image_width and image_height:
            min_width = max(int(image_width * 0.1), 10)
            min_height = max(int(image_height * 0.1), 10)
            size_constraint = f"""
## CRITICAL IMAGE SIZE AND BOUNDING BOX CONSTRAINTS
- Image dimensions: {image_width} x {image_height} pixels
- ALL bounding box coordinates MUST be within: x in [0, {image_width}], y in [0, {image_height}]
- **MINIMUM bounding box size: {min_width} x {min_height} pixels (at least 10% of image)**
- DO NOT use tiny bounding boxes (e.g., less than 10x10 pixels)!
"""
            bbox_guidance = f"""
## BOUNDING BOX SELECTION PRIORITY (IMPORTANT!)
1. **FIRST CHOICE: Use object detection bounding box** if available in the statistics above
   - Object detection provides semantically meaningful regions (e.g., the detected airplane, car, etc.)
   - These bboxes cover the entire object, which is what we need for evaluation
2. **SECOND CHOICE: Use pre-extracted bounding box** from the Pre-extracted Features section if valid
3. **LAST RESORT: Expand attention coordinates** - if you must use GradCAM/attention coordinates:
   - Raw attention points are often just a few pixels - TOO SMALL!
   - You MUST expand them to at least {min_width}x{min_height} pixels
   - Add padding around the attention center to create a meaningful region

**WARNING**: Bounding boxes smaller than {min_width}x{min_height} pixels are INVALID and will cause evaluation failures!
"""

        prompt = f"""You are an XAI expert. Based on the analysis, identify the MOST RESPONSIBLE part.

## Question
{context.get('user_question', self.question_template)}

## Model Prediction
Class: {prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))}
Confidence: {prediction.get('confidence', 0.0):.4f}
{size_constraint}{bbox_guidance}
## XAI Analysis Results
{tool_summary}

## Detailed Tool Statistics
{detailed_stats}

## Pre-extracted Features
{self._format_extracted_features(extracted)}

## Your Task
Identify the SINGLE MOST RESPONSIBLE part that caused this prediction.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this part is most responsible",
    "confidence": 0.85
}}

**Critical Requirements:**
- Identify ONE specific region/span/feature (the most important)
- For vision: bounding_box as [x_min, y_min, x_max, y_max] in pixels
  - **PREFER object detection bbox or pre-extracted bbox** over raw attention coordinates
  - MUST be at least {min_width if self.modality == 'vision' else 10}x{min_height if self.modality == 'vision' else 10} pixels
  - MUST respect image bounds: x in [0, {image_width}], y in [0, {image_height}]
- For text: start_index and end_index as character positions
- For tabular: feature_key as the column name
- This part should have HIGH attribution to the predicted class

Respond with ONLY JSON:"""
        return prompt

    def _get_image_size_from_results(self, tool_results: Dict[str, Any]) -> tuple:
        """Extract image size from tool results"""
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get('success'):
                img_size = result.get('original_image_size', {})
                if img_size:
                    return img_size.get('width', 224), img_size.get('height', 224)
        return 224, 224  # Default fallback

    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """Format tool results for prompt"""
        if not tool_results:
            return "No tool results available."

        lines = []
        for tool_name, result in tool_results.items():
            if isinstance(result, dict):
                success = result.get('success', False)
                summary = result.get('summary', result.get('description', 'N/A'))
                lines.append(f"- {tool_name}: {'Success' if success else 'Failed'} - {summary}")
        return "\n".join(lines) if lines else "No tool results available."

    def _format_detailed_statistics(self, tool_results: Dict[str, Any]) -> str:
        """Format detailed statistics from tool results including coordinates"""
        if not tool_results:
            return "No detailed statistics available."

        lines = []
        for tool_name, result in tool_results.items():
            if not isinstance(result, dict) or not result.get('success'):
                continue

            lines.append(f"\n### {tool_name}:")

            # Image size
            img_size = result.get('original_image_size', {})
            if img_size:
                lines.append(f"- Image size: {img_size.get('width')}x{img_size.get('height')} pixels")

            # Handle object detection results
            detections = result.get('detections', [])
            if detections:
                lines.append("- Detected objects:")
                for det in detections[:5]:  # Limit to top 5
                    class_name = det.get('class_name', 'unknown')
                    conf = det.get('confidence', 0)
                    bbox = det.get('bbox', {})
                    if bbox:
                        lines.append(f"  - {class_name} (conf={conf:.2f}): bbox=[{bbox.get('x1')}, {bbox.get('y1')}, {bbox.get('x2')}, {bbox.get('y2')}]")

            # Handle statistics with attention coordinates
            stats = result.get('statistics', {})
            if stats:
                # Top attention/importance/gradient coordinates (key for determining bounding box)
                top_coords = (
                    stats.get('top_attention_coords') or
                    stats.get('top_importance_coords') or
                    stats.get('top_gradient_coords')
                )
                if top_coords and len(top_coords) > 0:
                    # Calculate bounding box from top coordinates
                    xs = [c.get('x', 0) for c in top_coords]
                    ys = [c.get('y', 0) for c in top_coords]
                    if xs and ys:
                        lines.append(f"- High attention region: x=[{min(xs)}, {max(xs)}], y=[{min(ys)}, {max(ys)}]")
                        lines.append(f"- Suggested bounding box from this tool: [{min(xs)}, {min(ys)}, {max(xs)}, {max(ys)}]")
                        # Show top 3 individual points
                        top_3 = top_coords[:3]
                        coords_str = ", ".join([f"({c.get('x')}, {c.get('y')})" for c in top_3])
                        lines.append(f"- Top 3 attention points: {coords_str}")

                # Other useful stats
                if 'mean_attention' in stats:
                    lines.append(f"- Mean attention: {stats['mean_attention']:.3f}")
                if 'max_attention' in stats:
                    lines.append(f"- Max attention: {stats['max_attention']:.3f}")
                if 'high_attention_ratio' in stats:
                    lines.append(f"- High attention ratio: {stats['high_attention_ratio']:.2%}")

        return "\n".join(lines) if lines else "No detailed statistics available."

    def _format_extracted_features(self, features: Dict[str, Any]) -> str:
        """Format extracted features for prompt"""
        if not features:
            return "No pre-extracted features available."

        lines = []

        # Handle the new format with 'output' key
        output = features.get('output', {})
        if output:
            if 'bounding_box' in output:
                bbox = output['bounding_box']
                lines.append(f"- Pre-extracted bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            if 'start_index' in output:
                lines.append(f"- Pre-extracted text span: [{output['start_index']}, {output['end_index']}]")
            if 'feature_key' in output:
                lines.append(f"- Pre-extracted feature: {output['feature_key']}")

        # Handle explanation from feature extraction
        explanation = features.get('explanation', '')
        if explanation:
            lines.append(f"- Extraction reasoning: {explanation}")

        # Handle confidence
        confidence = features.get('confidence', 0)
        if confidence:
            lines.append(f"- Extraction confidence: {confidence:.2f}")

        # Legacy format support
        if 'responsible_regions' in features:
            for i, region in enumerate(features['responsible_regions'][:3]):
                label = region.get('label', f'Region {i+1}')
                importance = region.get('importance', 0)
                bbox = region.get('bbox', region.get('bounding_box'))
                if bbox:
                    lines.append(f"- {label}: importance={importance:.2f}, bbox={bbox}")
                else:
                    lines.append(f"- {label}: importance={importance:.2f}")

        if 'key_visual_features' in features:
            lines.append(f"- Key features: {', '.join(features['key_visual_features'][:5])}")

        return "\n".join(lines) if lines else "No pre-extracted features available."
