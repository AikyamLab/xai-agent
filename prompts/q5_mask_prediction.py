"""
Q5: If we mask a certain part, would the prediction change?

Expected Output:
- prediction_changes: 1 (Yes) or 0 (No)

Evaluation:
- Actually mask the queried part
- Metric: 1 if agent's prediction matches actual outcome, else 0
"""

from typing import Any, Dict

from .base_prompt import PromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q5MaskPredictionPromptBuilder(PromptBuilder):
    """Prompt builder for Q5: Would masking change the prediction?"""

    @property
    def question_type(self) -> int:
        return 5

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.COUNTERFACTUAL

    @property
    def question_template(self) -> str:
        return "If we mask a certain part, would the prediction change?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(5, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer"""
        prediction = context.get('prediction', {})
        queried_part = context.get('queried_part', 'a specific region')
        modality_config = self._get_modality_config()

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Prediction: Class {prediction.get('predicted_class_idx')}
(Confidence: {prediction.get('confidence', 0.0):.4f})
- Top-5 Predictions: {prediction.get('top5_predictions', [])}

**Part to be Masked**:
{queried_part}

**Input Content Description**:
{context.get(modality_config['description_key'], 'Not available')}

**Task**: Design a strategy to determine whether MASKING the specified {modality_config['element_type']} would CHANGE the model's prediction.

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Assess the importance of the queried {modality_config['element_type']} to the prediction
   - Reason about whether masking would significantly impact the output
   - Estimate the likelihood of prediction change

2. **External XAI Tools**: Use established explainability methods to assess importance:
{modality_config['tools_description']}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for assessing masking impact (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous importance analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {modality_config['tool_list']},
        "tool_params": {{
            {modality_config['tool_params_example']}
        }},
        "reasoning": "Why these tools for assessing masking impact on {self.modality}"
    }}
}}

**Guidelines for {self.modality.upper()} tasks (counterfactual prediction)**:
- {modality_config['spatial_note']}
- Target: Determine if the queried {modality_config['element_type']} has HIGH or LOW importance
- Evaluation metric: 1 if agent's prediction matches actual outcome, else 0
- HIGH attribution -> masking CHANGES prediction (output 1)
- LOW attribution -> masking does NOT change prediction (output 0)

Provide your strategy as a JSON object:
"""
        return prompt

    def _get_modality_config(self) -> Dict[str, Any]:
        """Get modality-specific configuration for prompt building"""
        if self.modality == "vision":
            return {
                'input_type': 'an IMAGE',
                'description_key': 'image_description',
                'element_type': 'region/object',
                'spatial_note': 'Check if the queried region has high activation in the model',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Check if queried region has high activation
   - IntegratedGradients: Measure pixel-level attribution of the region
   - LIME: Check local importance of the segment
   - SHAP: Compute Shapley value for the region
   - ObjectDetection: Identify if region contains important objects
   - GuidedBackprop: Visualize gradients in the region
   - SensitivityAnalysis: Measure sensitivity to changes in the region
   - LayerCAM: Check layer-wise activation in the region''',
                'tool_params_example': '"gradcam": {"layer": "layer4", "focus_region": "queried", "priority": 1},\n            "integrated_gradients": {"focus_region": "queried", "priority": 2}'
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT',
                'description_key': 'text_description',
                'element_type': 'token/phrase',
                'spatial_note': 'Check if the queried span has high attribution score',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Measure token-level attribution of the span
   - LIME: Check local importance of the tokens
   - SHAP: Compute Shapley value for the tokens
   - AttentionAnalysis: Check attention weights on the span
   - TokenImportance: Direct token-level attribution
   - SensitivityAnalysis: Measure sensitivity to token changes''',
                'tool_params_example': '"integrated_gradients": {"focus_span": "queried", "priority": 1},\n            "attention_analysis": {"focus_span": "queried", "priority": 2}'
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA',
                'description_key': 'data_description',
                'element_type': 'feature/column',
                'spatial_note': 'Check if the queried feature has high importance score',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Measure feature-level attribution
   - LIME: Check local importance of the feature
   - SHAP: Compute Shapley value for the feature
   - PermutationImportance: Measure importance by permuting the feature
   - SensitivityAnalysis: Measure sensitivity to feature changes''',
                'tool_params_example': '"shap": {"focus_feature": "queried", "priority": 1},\n            "permutation_importance": {"focus_feature": "queried", "priority": 2}'
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor"""
        prediction = context.get('prediction', {})
        queried_part = context.get('queried_part', 'the specified region')
        tool_results = results.get('tool_results', {})

        # Get image size constraint for vision modality
        size_constraint = self._build_image_size_constraint(tool_results)

        prompt = f"""You are an XAI expert. Predict if masking a part would change the prediction.

## Question
{context.get('user_question', self.question_template)}

## Current Prediction
Class: {prediction.get('predicted_class', 'Unknown')}
Confidence: {prediction.get('confidence', 0.0):.4f}

## Part to be Masked
{queried_part}
{size_constraint}
## XAI Analysis
{self._format_results_comprehensive(results)}

## Your Task
Based on the XAI analysis, predict whether masking this part would change the prediction.
- If the part has HIGH importance: masking should CHANGE the prediction (output 1)
- If the part has LOW importance: masking should NOT change the prediction (output 0)

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        "prediction_changes": 1
    }},
    "explanation": "2-3 sentences explaining your reasoning",
    "confidence": 0.85
}}

**Critical Requirements:**
- Output ONLY 1 (Yes, prediction changes) or 0 (No, prediction stays same)
- Base your decision on the attribution analysis
- High attribution part -> prediction likely changes
- Low attribution part -> prediction likely unchanged

Respond with ONLY JSON:"""
        return prompt
