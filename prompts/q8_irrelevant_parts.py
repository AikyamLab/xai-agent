"""
Q8: Is there any irrelevant part causing the model's wrong prediction?

Expected Output:
- Vision: bounding_box of irrelevant/spurious region
- Text: start_index, end_index of irrelevant span
- Tabular: feature_key of irrelevant feature

Evaluation:
- Mask the identified irrelevant parts
- Metric: 1 if P_correct_modified > P_correct_original, else 0
  (Success if masking the spurious part improves correct class probability)
"""

from typing import Any, Dict

from .base_prompt import PromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q8IrrelevantPartsPromptBuilder(PromptBuilder):
    """Prompt builder for Q8: Irrelevant parts causing wrong prediction"""

    @property
    def question_type(self) -> int:
        return 8

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.SPURIOUS_FEATURES

    @property
    def question_template(self) -> str:
        return "Is there any irrelevant part causing the model's wrong prediction?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(8, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer"""
        prediction = context.get('prediction', {})
        ground_truth = context.get('ground_truth', 'Unknown')
        modality_config = self._get_modality_config()

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's MISCLASSIFICATION on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Model Predicted: Class {prediction.get('predicted_class_idx')} ({prediction.get('predicted_class', 'Unknown')})
(Confidence: {prediction.get('confidence', 0.0):.4f})
- **Ground Truth**: {ground_truth}
- **Status**: MISCLASSIFIED

**Input Content Description**:
{context.get(modality_config['description_key'], 'Not available')}

**Task**: Design a strategy to identify SPURIOUS/IRRELEVANT {modality_config['element_type']} that are causing the model's WRONG prediction. These are {modality_config['element_type']} that:
- Have HIGH attribution from the model
- But are NOT semantically related to the correct class ({ground_truth})

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Compare model's attention regions with semantically relevant {modality_config['element_type']}
   - Identify {modality_config['element_type']} that appear irrelevant to the true class
   - Reason about spurious correlations the model may have learned

2. **External XAI Tools**: Use established explainability methods to find spurious features:
{modality_config['tools_description']}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for finding spurious {modality_config['element_type']} (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous spurious feature analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {modality_config['tool_list']},
        "tool_params": {{
            {modality_config['tool_params_example']}
        }},
        "reasoning": "Why these tools for finding spurious {self.modality} features"
    }}
}}

**Guidelines for {self.modality.upper()} tasks (spurious feature detection)**:
- {modality_config['spatial_note']}
- Target: Find {modality_config['element_type']} with HIGH attribution but LOW semantic relevance to ground truth
- Evaluation metric: 1 if masking spurious part improves correct class probability, else 0
- Examples: background patterns, artifacts, watermarks, confounding correlations

Provide your strategy as a JSON object:
"""
        return prompt

    def _get_modality_config(self) -> Dict[str, Any]:
        """Get modality-specific configuration for prompt building"""
        if self.modality == "vision":
            return {
                'input_type': 'an IMAGE',
                'description_key': 'image_description',
                'element_type': 'regions/objects',
                'spatial_note': 'Compare high-activation regions with semantically relevant objects for the true class',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Identify high-activation regions (potential spurious areas)
   - IntegratedGradients: Find pixels with high attribution to wrong class
   - LIME: Find segments contributing to wrong prediction
   - SHAP: Identify features that shouldn't matter for true class
   - ObjectDetection: Compare detected objects with true class expectations
   - GuidedBackprop: Visualize what the model wrongly focuses on
   - SensitivityAnalysis: Find regions sensitive but irrelevant
   - LayerCAM: Identify spurious activations at different layers''',
                'tool_params_example': '"gradcam": {"layer": "layer4", "target_class": "predicted", "priority": 1},\n            "object_detection": {"compare_with_ground_truth": true, "priority": 2}'
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT',
                'description_key': 'text_description',
                'element_type': 'tokens/phrases',
                'spatial_note': 'Compare high-attribution tokens with semantically relevant words for the true class',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Find tokens with high attribution to wrong class
   - LIME: Find tokens contributing to wrong prediction
   - SHAP: Identify tokens that shouldn't matter for true class
   - AttentionAnalysis: See tokens the model wrongly attends to
   - TokenImportance: Find high-importance but irrelevant tokens
   - SensitivityAnalysis: Find tokens sensitive but semantically irrelevant''',
                'tool_params_example': '"integrated_gradients": {"target_class": "predicted", "priority": 1},\n            "attention_analysis": {"compare_semantic_relevance": true, "priority": 2}'
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA',
                'description_key': 'data_description',
                'element_type': 'features/columns',
                'spatial_note': 'Compare high-importance features with domain-relevant features for the true class',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Find features with high attribution to wrong class
   - LIME: Find features contributing to wrong prediction
   - SHAP: Identify features that shouldn't matter for true class
   - PermutationImportance: Find high-importance but irrelevant features
   - SensitivityAnalysis: Find features sensitive but domain-irrelevant''',
                'tool_params_example': '"shap": {"target_class": "predicted", "priority": 1},\n            "permutation_importance": {"compare_domain_relevance": true, "priority": 2}'
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor"""
        prediction = context.get('prediction', {})
        ground_truth = context.get('ground_truth', 'Unknown')
        output_format = self._get_modality_specific_output_format()
        tool_results = results.get('tool_results', {})

        # Get image size constraint for vision modality
        image_width, image_height = self._get_image_size_from_results(tool_results)
        size_constraint = self._build_image_size_constraint(tool_results)
        instance_data_section = self._format_single_instance_data_section(context)

        prompt = f"""You are an XAI expert. Identify SPURIOUS/IRRELEVANT parts causing the wrong prediction.

## Question
{context.get('user_question', self.question_template)}

## Misclassification
- Model Predicted: {prediction.get('predicted_class', 'Unknown')} ({prediction.get('confidence', 0):.2%})
- Ground Truth: {ground_truth}
{size_constraint}{instance_data_section}
## XAI Analysis
{self._format_results_comprehensive(results)}

## Your Task
Identify the IRRELEVANT/SPURIOUS part that is causing the model to make the wrong prediction.
This is a part that:
- Has HIGH attribution from the model
- But is NOT semantically related to the correct class
- Examples: background, watermarks, artifacts, spurious correlations

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this part is spurious and causing the error",
    "confidence": 0.0-1.0
}}

**Critical Requirements:**
- Identify a part that is BOTH high-attribution AND semantically irrelevant
- Masking this part should IMPROVE the probability of the correct class
- This is about finding what the model WRONGLY relies on
- For vision: bounding_box MUST be within image bounds (x in [0, {image_width}], y in [0, {image_height}])

Respond with ONLY JSON:"""
        return prompt
