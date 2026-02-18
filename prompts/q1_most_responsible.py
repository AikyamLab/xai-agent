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

**Guidelines for {(self.modality or "tabular").upper()} tasks (positive attribution)**:
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

        # Format tool results with detailed statistics (use parent class methods)
        tool_summary = self._format_tool_results_summary(tool_results)
        detailed_stats = self._format_detailed_statistics(tool_results)

        # Build size constraint using parent class method
        size_constraint = self._build_image_size_constraint(tool_results)
        instance_data_section = self._format_single_instance_data_section(context)

        prompt = f"""You are an XAI expert. Based on the analysis, identify the MOST RESPONSIBLE part.

## Question
{context.get('user_question', self.question_template)}

## Model Prediction
Class: {prediction.get('predicted_class', prediction.get('predicted_class_idx', 'Unknown'))}
Confidence: {prediction.get('confidence', 0.0):.4f}
{size_constraint}{instance_data_section}
## XAI Analysis Results
{tool_summary}

## Detailed Tool Statistics
{detailed_stats}

## Your Task
Identify the SINGLE MOST RESPONSIBLE part that caused this prediction.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this part is most responsible",
    "confidence": 0.0-1.0
}}

**Critical Requirements:**
- Identify ONE specific region/span/feature (the most important)
- For vision: bounding_box as [x_min, y_min, x_max, y_max] in pixels
- For text: start_index and end_index as character positions
- For tabular: feature_key as the column name
- This part should have HIGH attribution to the predicted class

Respond with ONLY JSON:"""
        return prompt
