"""
Q7: If we remove/change one important part, how would the prediction change?

Expected Output:
- changed_class: predicted class after the modification
Evaluation:
- Apply the modification (remove/change the specified part)
- Metric: 1 if R1_modified == agent_predicted_class, else 0
"""

from typing import Any, Dict

from .base_prompt import PromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q7ChangePredictionPromptBuilder(PromptBuilder):
    """Prompt builder for Q7: Predict outcome of changing important part"""

    @property
    def question_type(self) -> int:
        return 7

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.COUNTERFACTUAL

    @property
    def question_template(self) -> str:
        return "If we remove/change one important part, how would the prediction change?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(7, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer"""
        prediction = context.get('prediction', {})
        part_to_change = context.get('part_to_change', 'the most important part')
        modality_config = self._get_modality_config()
        tools_description = self._get_available_tools_description(context, modality_config['tools_description'])
        tool_list = self._get_available_tools_list(context, modality_config['tool_list'])
        if self.modality != 'vision' and not context.get(modality_config['description_key']):
            raise ValueError(f"Input content description missing for {self.modality} modality")

        desc_key = modality_config['description_key']
        desc_section = ("**Input Content Description**:\n" + str(context.get(desc_key)) + "\n") if context.get(desc_key) else ""

        autonomous_bullets = (
            f"   - Assess the importance of the specified {modality_config['element_type']}\n"
            f"   - Reason about what alternative prediction the model would make\n"
            f"   - Consider the distribution of remaining {modality_config['element_type']} importance"
        )
        methods_section = self._build_methods_section(
            context, autonomous_bullets=autonomous_bullets, tools_description=tools_description
        )
        strategy_schema = self._build_strategy_schema_block(
            context,
            tool_list=tool_list,
            reasoning_hint="Explain why you chose this strategy for predicting outcome (2-3 sentences)",
        )

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Current Prediction: Class {prediction.get('predicted_class_idx')} ({prediction.get('predicted_class_name', 'Unknown')})
(Confidence: {prediction.get('confidence', 0.0):.4f})
- Top-5 Predictions: {prediction.get('top5_predictions', [])}

**Part to be Changed/Removed**:
{part_to_change}

{desc_section}**Task**: Design a strategy to PREDICT how the model's prediction would change if we REMOVE or MODIFY the specified {modality_config['element_type']}.

{methods_section}

**Your Response Must Be Valid JSON** with the following structure:
{strategy_schema}

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
                'spatial_note': 'Analyze remaining visual features after modification to predict new class',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Assess importance of the region and see alternative activation
   - IntegratedGradients: Understand attribution distribution across the image
   - LIME: Analyze how removing segment affects local prediction
   - SHAP: Understand contribution distribution to predict new outcome
   - ObjectDetection: Identify remaining objects after modification
   - GuidedBackprop: Visualize what else the model relies on
   - SensitivityAnalysis: Measure sensitivity to predict change magnitude
   - LayerCAM: Understand layer-wise dependence on the region''',
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT',
                'description_key': 'text_description',
                'element_type': 'token/phrase',
                'spatial_note': 'Analyze remaining token importance to predict new class',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Understand attribution distribution across tokens
   - LIME: Analyze how removing tokens affects local prediction
   - SHAP: Understand contribution distribution to predict new outcome
   - AttentionAnalysis: See what else the model attends to
   - TokenImportance: Understand token-level importance distribution
   - SensitivityAnalysis: Measure sensitivity to predict change magnitude''',
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA',
                'description_key': 'data_description',
                'element_type': 'feature/column',
                'spatial_note': 'Analyze remaining feature importance to predict new class',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Understand attribution distribution across features
   - LIME: Analyze how changing feature affects local prediction
   - SHAP: Understand contribution distribution to predict new outcome
   - PermutationImportance: Understand importance of remaining features
   - SensitivityAnalysis: Measure sensitivity to predict change magnitude''',
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor"""
        if self.modality == "vision":
            return self._build_actor_prompt_vision(context, strategy, results)
        return self._build_actor_prompt_default(context, strategy, results)

    def _build_actor_prompt_vision(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build vision-specific actor prompt that also identifies the region to mask."""
        prediction = context.get('prediction', {})
        top5 = prediction.get('top5_predictions', [])
        tool_results = results.get('tool_results', {})
        size_constraint = self._build_image_size_constraint(tool_results)
        output_size_constraint = self._build_output_size_constraint(context)

        available_classes = context.get('available_classes')
        if available_classes:
            classes_section = (
                "\n## Available Classes\n"
                "The model classifies images into these classes — your `changed_class` "
                "MUST be one of them exactly:\n"
                + ", ".join(available_classes) + "\n"
            )
        else:
            classes_section = ""

        prompt = f"""You are an XAI expert. Identify one specific region in the image and predict how the model's prediction would change after masking it.

## Question
{context.get('user_question', self.question_template)}

## Current State
- Current Prediction: {prediction.get('predicted_class_name', 'Unknown')} ({prediction.get('confidence', 0):.2%})
- Top-5 Predictions: {top5}
{size_constraint}{classes_section}
## XAI Analysis
{self._format_results_comprehensive(results, context)}

## Your Task
1. Based on the XAI tool results (e.g. GradCAM heatmaps, attribution maps), identify one specific region in the image as a bounding box [x_min, y_min, x_max, y_max].
   - The bounding box must be within the image dimensions.
2. Predict what the NEW prediction class would be after masking/removing that region.
   - Consider the remaining visual features and the top-5 predictions.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        "changed_class": "predicted class name after modification",
        "masked_region": {{
            "bounding_box": [x_min, y_min, x_max, y_max]
        }}
    }},
    "explanation": "2-3 sentences: which region you chose, why, and what the new prediction would be"
}}

**Critical Requirements:**
- changed_class: The class the model would predict AFTER masking the region
- masked_region.bounding_box: [x_min, y_min, x_max, y_max] integers within image bounds{output_size_constraint}
- Base your decision on the attribution analysis

Respond with ONLY JSON:"""
        return prompt

    def _build_actor_prompt_default(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build actor prompt for text/tabular where the region is already specified."""
        prediction = context.get('prediction', {})
        part_to_change = context.get('part_to_change', 'the most important part')
        top5 = prediction.get('top5_predictions', [])
        tool_results = results.get('tool_results', {})
        size_constraint = self._build_image_size_constraint(tool_results)
        instance_data_section = self._format_single_instance_data_section(context)

        prompt = f"""You are an XAI expert. Predict the new class after modifying the specified part in the question.

## Question
{context.get('user_question', self.question_template)}

## Current State
- Current Prediction: {prediction.get('predicted_class_name', 'Unknown')} ({prediction.get('confidence', 0):.2%})
- Top-5 Predictions: {top5}
- Part to Change: {part_to_change}
{size_constraint}{instance_data_section}
## XAI Analysis
{self._format_results_comprehensive(results, context)}

## Your Task
Predict what the NEW prediction would be after removing/changing the specified part.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        "changed_class": "predicted class name after modification"
    }},
    "explanation": "2-3 sentences explaining why the prediction would change to this class"
}}

**Critical Requirements:**
- changed_class: The class the model would predict AFTER the modification
- This is often the second-most-likely class (top-2) but could be different
- Consider what features remain after the modification
- Base your decision on the attribution analysis

Respond with ONLY JSON:"""
        return prompt
