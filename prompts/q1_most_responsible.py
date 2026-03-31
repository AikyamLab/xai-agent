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
        tools_description = self._get_available_tools_description(context, modality_config['tools_description'])
        tool_list = self._get_available_tools_list(context, modality_config['tool_list'])
        if self.modality != 'vision' and not context.get(modality_config['description_key']):
            raise ValueError(f"Input content description missing for {self.modality} modality")

        desc_key = modality_config['description_key']
        desc_section = ("**Input Content Description**:\n" + str(context.get(desc_key)) + "\n") if context.get(desc_key) else ""

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Prediction: Class {prediction.get('predicted_class_idx')}
(Confidence: {prediction.get('confidence', 0.0):.4f})
- Top-5 Predictions: {prediction.get('top5_predictions', [])}

{desc_section}**Task**: Design a comprehensive strategy to identify which {modality_config['region_type']} of the input were MOST RESPONSIBLE for this prediction.

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Identify and ground important {modality_config['element_type']} in the input
   - Reason about their importance to the predicted class
   - Provide {modality_config['location_type']} for different {modality_config['element_type']}

2. **External XAI Tools**: Use established explainability methods:
{tools_description}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for finding most responsible {modality_config['element_type']} (2-3 sentences)",
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {tool_list},
        "reasoning": "Why these tools for {self.modality} modality"
    }}
}}

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
        tool_results = results.get('tool_results', {})
        size_constraint = self._build_image_size_constraint(tool_results)
        instance_data_section = self._format_single_instance_data_section(context)
        output_size_constraint = self._build_output_size_constraint(context)

        # Build vision-specific bbox size instruction near the output format
        cfg = self.output_size_config
        vision_bbox_instruction = ""
        if self.modality == "vision" and cfg and cfg.apply_to_vision:
            image_width, image_height = self._get_image_size_from_results(tool_results)
            target_area = int(image_width * image_height * cfg.fixed_percentage)
            pct_str = f"{cfg.fixed_percentage * 100:.0f}%"
            vision_bbox_instruction = (
                f"\n- **BBOX SIZE REQUIREMENT**: The bounding box MUST cover approximately "
                f"{pct_str} of the image area. Target area: {target_area} px² "
                f"(image is {image_width}×{image_height}={image_width*image_height} px²). "
                f"Concretely: (x_max - x_min) * (y_max - y_min) MUST be close to {target_area}. "
                f"Do NOT use the full bbox if it exceeds this size — "
                f"instead, crop to the MOST IMPORTANT {pct_str} sub-region."
            )

        prompt = f"""You are an XAI expert. Based on the analysis, identify the MOST RESPONSIBLE part.

## Question
{context.get('user_question', self.question_template)}

## Model Prediction
Class: {prediction.get('predicted_class_name', prediction.get('predicted_class_idx', 'Unknown'))}
Confidence: {prediction.get('confidence', 0.0):.4f}
{size_constraint}{instance_data_section}
## XAI Analysis
{self._format_results_comprehensive(results, context)}

## Your Task
Identify the MOST RESPONSIBLE part that caused this prediction.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this part is most responsible"
}}

**Critical Requirements:**
- For vision: bounding_box as [x_min, y_min, x_max, y_max] in pixels{vision_bbox_instruction}
- For text: spans as a list of {{start_index, end_index}} character positions (one or more spans)
- For tabular: feature_keys as a list of column names{output_size_constraint}
- Base your decision on the attribution analysis

Respond with ONLY JSON:"""
        return prompt
