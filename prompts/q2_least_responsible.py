"""
Q2: Which part of the input was LEAST responsible for the model's prediction?

Expected Output:
- Vision: bounding_box [x_min, y_min, x_max, y_max]
- Text: start_index, end_index
- Tabular: feature_key

Evaluation:
- Mask the identified part
- Metric: -(P_original - P_modified) (smaller difference = better, part was indeed unimportant)
"""

from typing import Any, Dict

from .base_prompt import PromptBuilder, QuestionCategory, Modality
from .output_schemas import get_output_schema


class Q2LeastResponsiblePromptBuilder(PromptBuilder):
    """Prompt builder for Q2: Least responsible part identification"""

    @property
    def question_type(self) -> int:
        return 2

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.FEATURE_ATTRIBUTION

    @property
    def question_template(self) -> str:
        return "Which part of the input was least responsible for the model's prediction?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(2, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer"""
        prediction = context.get('prediction', {})
        modality_config = self._get_modality_config()
        tools_description = self._get_available_tools_description(context, modality_config['tools_description'])
        tool_list = self._get_available_tools_list(context, modality_config['tool_list'])
        if self.modality != 'vision' and not context.get(modality_config['description_key']):
            raise ValueError(f"Input content description missing for {self.modality} modality")

        desc_key = modality_config['description_key']
        desc_section = ("**Input Content Description**:\n" + str(context.get(desc_key)) + "\n") if context.get(desc_key) else ""

        autonomous_bullets = (
            f"   - Identify {modality_config['element_type']} that appear irrelevant to the predicted class\n"
            f"   - Reason about which parts have minimal contribution\n"
            f"   - Provide {modality_config['location_type']} for unimportant {modality_config['element_type']}"
        )
        methods_section = self._build_methods_section(
            context, autonomous_bullets=autonomous_bullets, tools_description=tools_description
        )
        strategy_schema = self._build_strategy_schema_block(
            context,
            tool_list=tool_list,
            reasoning_hint=f"Explain why you chose this strategy for finding least responsible {modality_config['element_type']} (2-3 sentences)",
        )

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Prediction: Class {prediction.get('predicted_class_idx')}
(Confidence: {prediction.get('confidence', 0.0):.4f})
- Top-5 Predictions: {prediction.get('top5_predictions', [])}

{desc_section}**Task**: Design a comprehensive strategy to identify which {modality_config['region_type']} of the input were LEAST RESPONSIBLE (had minimal impact) for this prediction.

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
                'region_type': 'SPATIAL REGIONS',
                'element_type': 'objects/regions',
                'location_type': 'bounding boxes',
                'spatial_note': 'Consider spatial nature of images - look for background or irrelevant regions',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Low activation regions indicate less importance
   - IntegratedGradients: Near-zero scores indicate minimal contribution
   - LIME: Low weights indicate unimportant segments
   - SHAP: Near-zero Shapley values indicate irrelevance
   - ObjectDetection: Helps identify background vs foreground
   - GuidedBackprop: Low gradient regions are less important
   - SensitivityAnalysis: Low sensitivity indicates minimal impact
   - LayerCAM: Low activation areas are less responsible''',
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT',
                'description_key': 'text_description',
                'region_type': 'TEXT SPANS',
                'element_type': 'tokens/phrases',
                'location_type': 'start and end indices',
                'spatial_note': 'Consider sequential nature of text - look for filler words or irrelevant phrases',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Near-zero token scores indicate minimal contribution
   - LIME: Low weights indicate unimportant tokens
   - SHAP: Near-zero values indicate irrelevant tokens
   - AttentionAnalysis: Low attention weights suggest less importance
   - TokenImportance: Identifies tokens with minimal attribution
   - SensitivityAnalysis: Low sensitivity indicates minimal impact''',
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA',
                'description_key': 'data_description',
                'region_type': 'FEATURES',
                'element_type': 'features/columns',
                'location_type': 'feature names',
                'spatial_note': 'Consider feature relationships - look for features with near-zero contribution',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Near-zero scores indicate minimal contribution
   - LIME: Low weights indicate unimportant features
   - SHAP: Near-zero Shapley values indicate irrelevance
   - PermutationImportance: Low importance indicates minimal impact
   - SensitivityAnalysis: Low sensitivity indicates feature is irrelevant''',
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor"""
        prediction = context.get('prediction', {})
        output_format = self._get_modality_specific_output_format()
        tool_results = results.get('tool_results', {})

        # Get image size constraint for vision modality
        image_width, image_height = self._get_image_size_from_results(tool_results)
        size_constraint = self._build_image_size_constraint(tool_results)
        instance_data_section = self._format_single_instance_data_section(context)

        # Q2: tool summary should show LEAST important features (ascending sort)
        context['feature_sort_direction'] = 'ascending'
        output_size_constraint = self._build_output_size_constraint(context)
        # Override constraint text for Q2 (bottom %, not top %)
        cfg = self.output_size_config
        if cfg:
            pct_str = f"{cfg.fixed_percentage * 100:.0f}%"
            if self.modality == 'tabular' and cfg.apply_to_tabular:
                n_target = context.get('n_target_features')
                total = context.get('total_features')
                if n_target and total:
                    output_size_constraint = (
                        f"\n- **OUTPUT SIZE CONSTRAINT**: Return EXACTLY {n_target} feature_keys "
                        f"(bottom {pct_str} least responsible of {total} total features)"
                    )
            elif self.modality == 'text' and cfg.apply_to_text:
                target_chars = context.get('target_chars')
                text_length = context.get('text_length')
                if target_chars and text_length:
                    output_size_constraint = (
                        f"\n- **OUTPUT SIZE CONSTRAINT**: Spans must cover approximately "
                        f"{target_chars} characters total (bottom {pct_str} least responsible of {text_length} chars)"
                    )

        prompt = f"""You are an XAI expert. Identify the LEAST RESPONSIBLE part of the input.

## Question
{context.get('user_question', self.question_template)}

## Model Prediction
Class: {prediction.get('predicted_class_name', 'Unknown')}
Confidence: {prediction.get('confidence', 0.0):.4f}
{size_constraint}{instance_data_section}
## XAI Analysis
{self._format_results_comprehensive(results, context)}

## Your Task
Identify the part that had LEAST impact on the prediction.
This should be part that, if masked, would NOT significantly change the prediction.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this part is least responsible"
}}

**Critical Requirements:**
- For vision: bounding_box MUST be within image bounds (x in [0, {image_width}], y in [0, {image_height}])
- For text: text_spans as a list of exact phrase strings copied verbatim from the input text (one or more phrases)
- For tabular: feature_keys as a list of column names {output_size_constraint}
- Base your decision on the attribution analysis

Respond with ONLY JSON:"""
        return prompt
