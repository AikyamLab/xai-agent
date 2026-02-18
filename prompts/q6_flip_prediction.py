"""
Q6: How should the instance change to flip the model prediction to [expected_class]?

Expected Output:
- change_plan: {region, action, new_value}

Evaluation:
- Apply the change plan to generate modified instance
- Metric: 1 if R1_modified == expected_class, else 0
"""

from typing import Any, Dict

from .base_prompt import PromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q6FlipPredictionPromptBuilder(PromptBuilder):
    """Prompt builder for Q6: How to flip prediction to target class"""

    @property
    def question_type(self) -> int:
        return 6

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.COUNTERFACTUAL

    @property
    def question_template(self) -> str:
        return "How should the instance change to flip the model prediction to [expected_class]?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(6, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer"""
        prediction = context.get('prediction', {})
        target_class = context.get('target_class', 'a different class')
        modality_config = self._get_modality_config()

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Current Prediction: Class {prediction.get('predicted_class_idx')} ({prediction.get('predicted_class', 'Unknown')})
(Confidence: {prediction.get('confidence', 0.0):.4f})
- Target Prediction: {target_class}
- Top-5 Predictions: {prediction.get('top5_predictions', [])}

**Input Content Description**:
{context.get(modality_config['description_key'], 'Not available')}

**Task**: Design a comprehensive strategy to identify what CHANGES to the input would FLIP the prediction from "{prediction.get('predicted_class', 'current')}" to "{target_class}".

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Identify {modality_config['element_type']} that are critical for the current prediction
   - Reason about what modifications would shift the prediction to the target class
   - Propose specific change plans with {modality_config['location_type']}

2. **External XAI Tools**: Use established explainability methods to identify modification targets:
{modality_config['tools_description']}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for finding counterfactual changes (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous counterfactual analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {modality_config['tool_list']},
        "tool_params": {{
            {modality_config['tool_params_example']}
        }},
        "reasoning": "Why these tools for finding {self.modality} counterfactuals"
    }}
}}

**Guidelines for {self.modality.upper()} tasks (counterfactual generation)**:
- {modality_config['spatial_note']}
- Target: Find {modality_config['element_type']} to modify that would flip prediction to target class
- Evaluation metric: 1 if modified input predicts target class, else 0
- Changes should be MINIMAL but sufficient to flip prediction

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
                'location_type': 'bounding boxes and change descriptions',
                'spatial_note': 'Consider what visual features distinguish current vs target class',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Find regions important for current class (candidates for modification)
   - IntegratedGradients: Identify pixels with highest attribution to modify
   - LIME: Find segments that most affect current prediction
   - SHAP: Identify features contributing to current vs target class
   - ObjectDetection: Identify objects to add/remove/modify
   - GuidedBackprop: Visualize what the model focuses on
   - SensitivityAnalysis: Find regions most sensitive to changes
   - LayerCAM: Identify layer-specific activation to target''',
                'tool_params_example': '"gradcam": {"layer": "layer4", "target_class": "current", "priority": 1},\n            "integrated_gradients": {"target_class": "contrastive", "priority": 2}'
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT',
                'description_key': 'text_description',
                'element_type': 'tokens/phrases',
                'location_type': 'indices and replacement text',
                'spatial_note': 'Consider what words/phrases distinguish current vs target class',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Identify tokens with highest attribution to modify
   - LIME: Find tokens that most affect current prediction
   - SHAP: Identify tokens contributing to current vs target class
   - AttentionAnalysis: Find tokens the model attends to most
   - TokenImportance: Direct token attribution for modification targets
   - SensitivityAnalysis: Find tokens most sensitive to changes''',
                'tool_params_example': '"integrated_gradients": {"target_class": "contrastive", "priority": 1},\n            "lime": {"target_class": "current", "priority": 2}'
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA',
                'description_key': 'data_description',
                'element_type': 'features/columns',
                'location_type': 'feature names and new values',
                'spatial_note': 'Consider what feature values distinguish current vs target class',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Identify features with highest attribution to modify
   - LIME: Find features that most affect current prediction
   - SHAP: Identify features contributing to current vs target class
   - PermutationImportance: Find features most important for prediction
   - SensitivityAnalysis: Find features most sensitive to changes''',
                'tool_params_example': '"shap": {"target_class": "contrastive", "priority": 1},\n            "sensitivity_analysis": {"target_class": "target", "priority": 2}'
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor"""
        prediction = context.get('prediction', {})
        target_class = context.get('target_class', 'target')
        tool_results = results.get('tool_results', {})

        # Get image size constraint for vision modality
        image_width, image_height = self._get_image_size_from_results(tool_results)
        size_constraint = self._build_image_size_constraint(tool_results)
        instance_data_section = self._format_single_instance_data_section(context)

        # Build modality-specific output format
        if self.modality == "vision":
            output_format = '''"change_plan": {
            "bounding_box": [x_min, y_min, x_max, y_max],
            "action": "change",
            "new_value": "Stable Diffusion inpainting prompt describing ONLY the desired appearance of the modified region (e.g. 'a deer head with elongated snout, large pointed ears, brown fur, side-facing eyes'). Do NOT mention the original class or use phrases like 'replace X with Y'. Just describe what should appear in this region as if painting it from scratch."
        }'''
        elif self.modality == "text":
            output_format = '''"change_plan": {
            "start_index": int,
            "end_index": int,
            "action": "change",
            "new_value": "replacement text"
        }'''
        else:
            # Show original pre-encoding feature names and values so the agent can reason
            # naturally. The evaluator handles one-hot translation internally.
            orig_features = context.get('instance_data_single', {}).get('features', {})
            if isinstance(orig_features, dict) and orig_features:
                feat_list = list(orig_features.keys())
                feat_preview = ", ".join(
                    f"{k}={repr(v)}" for k, v in list(orig_features.items())[:15]
                )
            else:
                feat_list = []
                feat_preview = "N/A"
            output_format = f'''"change_plan": [
            {{
                "feature_key": "feature_name",
                "action": "change",
                "new_value": <value>
            }}
        ]
        // Available features: {feat_list}
        // Current values: {feat_preview}
        // Use original feature names (e.g. "occupation") and original value formats (e.g. "Exec-managerial" or 40)
        // Include multiple entries if changing several features is needed to flip the prediction'''

        prompt = f"""You are an XAI expert. Propose a change plan to FLIP the prediction.

## Question
{context.get('user_question', self.question_template)}

## Current State
- Current Prediction: {prediction.get('predicted_class', 'Unknown')} ({prediction.get('confidence', 0):.2%})
- Target Prediction: {target_class}
{size_constraint}{instance_data_section}
## XAI Analysis
{self._format_results_comprehensive(results)}

## Your Task
Propose a SPECIFIC change plan that would flip the prediction to "{target_class}".

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining why this change would flip the prediction",
    "confidence": 0.0-1.0
}}

**Critical Requirements:**
- action must be one of: "change", "delete", "swap", "add"
- For vision: identify region to modify and provide new_value as a Stable Diffusion inpainting prompt
  - if the action is "delete", new_value automatically sets to null
  - bounding_box MUST be within image bounds (x in [0, {image_width}], y in [0, {image_height}])
- For text: identify span to replace and provide new text
- For tabular: feature_key must be an exact name from the available features list; new_value should match the original data format
- The change should be MINIMAL but sufficient to flip prediction

Respond with ONLY JSON:"""
        return prompt
