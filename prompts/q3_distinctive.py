"""
Q3: Which specific parts distinguish the prediction from the next-best alternative?

Expected Output:
- Vision: bounding_box [x_min, y_min, x_max, y_max]
- Text: start_index, end_index
- Tabular: feature_key

Evaluation:
- Mask the identified parts
- Metric: 1 if (P_top2_modified - P_top1_modified) > 0, else 0
  (Success if masking flips the ranking between top-1 and top-2)
"""

from typing import Any, Dict

from .base_prompt import PromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q3DistinctivePromptBuilder(PromptBuilder):
    """Prompt builder for Q3: Parts distinguishing from next-best alternative"""

    @property
    def question_type(self) -> int:
        return 3

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.FEATURE_ATTRIBUTION

    @property
    def question_template(self) -> str:
        return "Which specific parts distinguish the prediction from the next-best alternative?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(3, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer"""
        prediction = context.get('prediction', {})
        top5 = prediction.get('top5_predictions', [])
        modality_config = self._get_modality_config()
        tools_description = self._get_available_tools_description(context, modality_config['tools_description'])
        tool_list = self._get_available_tools_list(context, modality_config['tool_list'])

        top1_class = prediction.get('predicted_class_name', prediction.get('predicted_class_idx', 'Unknown'))
        if len(top5) > 1:
            top2_class = top5[1]
        else:
            probs = prediction.get('probabilities', {})
            if isinstance(probs, dict) and len(probs) > 1:
                top2_class = sorted(probs, key=probs.get, reverse=True)[1]
            else:
                top2_class = 'Unknown'

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Top-1 Prediction: {top1_class} (Confidence: {prediction.get('confidence', 0.0):.4f})
- Top-2 Prediction (Alternative): {top2_class}
- Top-5 Predictions: {top5}

**Input Content Description**:
{context.get(modality_config['description_key'], 'Not available')}

**Task**: Design a comprehensive strategy to identify which {modality_config['region_type']} DISTINGUISH the top-1 prediction ({top1_class}) from the top-2 alternative ({top2_class}).

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Identify {modality_config['element_type']} that are distinctive for {top1_class} vs {top2_class}
   - Reason about class-specific features that differentiate the two predictions
   - Provide {modality_config['location_type']} for decisive {modality_config['element_type']}

2. **External XAI Tools**: Use established explainability methods for contrastive analysis:
{tools_description}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for finding distinctive {modality_config['element_type']} between top-1 and top-2 (2-3 sentences)",
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous contrastive analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {tool_list},
        "reasoning": "Why these tools for contrastive {self.modality} analysis"
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
                'spatial_note': 'Consider class-specific visual features - find regions unique to top-1 class',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Compare class-specific activation maps for top-1 vs top-2
   - IntegratedGradients: Compute contrastive attribution between classes
   - LIME: Compare local explanations for each class
   - SHAP: Compare feature importance across classes
   - ObjectDetection: Identify class-specific objects
   - GuidedBackprop: Visualize class-specific gradients
   - SensitivityAnalysis: Compare sensitivity for different class targets
   - LayerCAM: Class-wise activation comparison''',
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT',
                'description_key': 'text_description',
                'region_type': 'TEXT SPANS',
                'element_type': 'tokens/phrases',
                'location_type': 'start and end indices',
                'spatial_note': 'Consider class-specific tokens - find phrases unique to top-1 prediction',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Compute contrastive token attribution
   - LIME: Compare token importance for each class
   - SHAP: Compare token-level Shapley values across classes
   - AttentionAnalysis: Compare attention patterns for different predictions
   - TokenImportance: Class-specific token attribution
   - SensitivityAnalysis: Compare sensitivity for different class targets''',
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA',
                'description_key': 'data_description',
                'region_type': 'FEATURES',
                'element_type': 'features/columns',
                'location_type': 'feature names',
                'spatial_note': 'Consider class-specific feature patterns - find features unique to top-1 prediction',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Compute contrastive feature attribution
   - LIME: Compare feature importance for each class
   - SHAP: Compare Shapley values across classes
   - PermutationImportance: Class-specific feature importance
   - SensitivityAnalysis: Compare sensitivity for different class targets''',
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor"""
        prediction = context.get('prediction', {})
        top5 = prediction.get('top5_predictions', [])
        output_format = self._get_modality_specific_output_format()
        tool_results = results.get('tool_results', {})

        top1 = prediction.get('predicted_class_name', 'Top-1')
        if len(top5) > 1:
            top2 = top5[1]
        else:
            probs = prediction.get('probabilities', {})
            if isinstance(probs, dict) and len(probs) > 1:
                top2 = sorted(probs, key=probs.get, reverse=True)[1]
            else:
                top2 = 'Top-2'

        # Get image size constraint for vision modality
        image_width, image_height = self._get_image_size_from_results(tool_results)
        size_constraint = self._build_image_size_constraint(tool_results)
        instance_data_section = self._format_single_instance_data_section(context)

        prompt = f"""You are an XAI expert. Identify the DISTINCTIVE part that separates top-1 from top-2.

## Question
{context.get('user_question', self.question_template)}

## Predictions
- Top-1: {top1} ({prediction.get('confidence', 0):.2%})
- Top-2: {top2}
{size_constraint}{instance_data_section}
## XAI Analysis
{self._format_results_comprehensive(results)}

## Your Task
Identify the part that DISTINGUISHES {top1} from {top2}.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this part distinguishes the two classes"
}}

**Critical Requirements:**
- For vision: bounding_box MUST be within image bounds (x in [0, {image_width}], y in [0, {image_height}])
- For text: spans as a list of {{start_index, end_index}} character positions (one or more spans)
- For tabular: feature_keys as a list of column names (one or more features)
- Base your decision on the attribution analysis

Respond with ONLY JSON:"""
        return prompt
