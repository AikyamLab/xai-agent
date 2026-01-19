"""
Q10: Why are two similar instances given different predictions (one correct, one wrong)?

Expected Output:
- Vision: Natural language descriptions of features
- Text/Tabular: Specific regions/features for correct and wrong predictions

Evaluation:
- Compare feature sets from correct vs wrong predictions
- Metric: -Sim(F_correct, F_wrong) (larger overlap = worse)
  - Vision: word similarity on descriptions
  - Text: word overlap
  - Tabular: rank correlation
"""

from typing import Any, Dict, List

from .base_prompt import MultiInstancePromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q10SimilarDifferentPromptBuilder(MultiInstancePromptBuilder):
    """Prompt builder for Q10: Similar instances with different predictions"""

    @property
    def question_type(self) -> int:
        return 10

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.SPURIOUS_FEATURES

    @property
    def question_template(self) -> str:
        return "Why are two similar instances given different predictions (one correct, one wrong)?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(10, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build proposer prompt"""
        return self.build_proposer_prompt_multi(context, [])

    def build_proposer_prompt_multi(
        self,
        context: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Proposer"""
        modality_config = self._get_modality_config()

        # Build instance info
        instance_info = ""
        if instances and len(instances) >= 2:
            inst_a = instances[0]
            inst_b = instances[1]
            pred_a = inst_a.get('prediction', {})
            pred_b = inst_b.get('prediction', {})
            instance_info = f"""- Instance A (CORRECT): Predicted {pred_a.get('predicted_class', 'Unknown')} (Ground Truth: {inst_a.get('ground_truth', 'Same')})
- Instance B (WRONG): Predicted {pred_b.get('predicted_class', 'Unknown')} (Ground Truth: {inst_b.get('ground_truth', 'Different')})"""
        else:
            instance_info = """- Instance A: Correctly classified
- Instance B: Incorrectly classified (similar to A but wrong prediction)"""

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's predictions on SIMILAR {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}

**Instance Predictions**:
{instance_info}
- **Note**: Both instances are SIMILAR but have DIFFERENT prediction outcomes

**Input Content Descriptions**:
- Instance A: {context.get(modality_config['description_key'] + '_A', context.get(modality_config['description_key'], 'Not available'))}
- Instance B: {context.get(modality_config['description_key'] + '_B', 'Not available')}

**Task**: Design a strategy to identify DISTINCT {modality_config['element_type']} that explain why:
- Instance A is classified CORRECTLY
- Instance B is classified INCORRECTLY
The goal is to find features that are AS DIFFERENT AS POSSIBLE between the correct and wrong prediction.

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Compare {modality_config['element_type']} between the correctly and incorrectly classified instances
   - Identify what makes Instance A succeed where Instance B fails
   - Find distinguishing characteristics between the two outcomes

2. **External XAI Tools**: Use established explainability methods for contrastive analysis:
{modality_config['tools_description']}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for comparing correct vs incorrect predictions (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous correct/wrong comparison",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {modality_config['tool_list']},
        "tool_params": {{
            {modality_config['tool_params_example']}
        }},
        "reasoning": "Why these tools for comparing {self.modality} success vs failure"
    }}
}}

**Guidelines for {self.modality.upper()} tasks (correct vs incorrect comparison)**:
- {modality_config['spatial_note']}
- Target: Find DISTINCT {modality_config['element_type']} - minimize overlap between correct and wrong feature sets
- Evaluation metric: -Sim(F_correct, F_wrong) - larger difference = better explanation
- Focus on what makes one succeed and the other fail

Provide your strategy as a JSON object:
"""
        return prompt

    def _get_modality_config(self) -> Dict[str, Any]:
        """Get modality-specific configuration for prompt building"""
        if self.modality == "vision":
            return {
                'input_type': 'IMAGES',
                'description_key': 'image_description',
                'element_type': 'regions/objects',
                'spatial_note': 'Compare visual features between correctly and incorrectly classified similar images',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Compare activation patterns between correct and wrong instance
   - IntegratedGradients: Compare pixel-level attributions
   - LIME: Compare segment importance between instances
   - SHAP: Compare feature contributions
   - ObjectDetection: Compare detected objects between instances
   - GuidedBackprop: Visualize focus differences
   - SensitivityAnalysis: Compare sensitivity patterns
   - LayerCAM: Compare layer-wise activations''',
                'tool_params_example': '"gradcam": {"layer": "layer4", "apply_to": "both", "comparison_mode": "difference", "priority": 1},\n            "object_detection": {"compare_instances": true, "priority": 2}'
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT INPUTS',
                'description_key': 'text_description',
                'element_type': 'tokens/phrases',
                'spatial_note': 'Compare token importance between correctly and incorrectly classified similar texts',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Compare token-level attributions
   - LIME: Compare token importance between instances
   - SHAP: Compare token contributions
   - AttentionAnalysis: Compare attention patterns
   - TokenImportance: Compare token-level attribution
   - SensitivityAnalysis: Compare sensitivity patterns''',
                'tool_params_example': '"integrated_gradients": {"apply_to": "both", "comparison_mode": "difference", "priority": 1},\n            "attention_analysis": {"compare_instances": true, "priority": 2}'
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA INSTANCES',
                'description_key': 'data_description',
                'element_type': 'features/columns',
                'spatial_note': 'Compare feature importance between correctly and incorrectly classified similar records',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Compare feature-level attributions
   - LIME: Compare feature importance between instances
   - SHAP: Compare Shapley values
   - PermutationImportance: Compare feature importance
   - SensitivityAnalysis: Compare sensitivity patterns''',
                'tool_params_example': '"shap": {"apply_to": "both", "comparison_mode": "difference", "priority": 1},\n            "permutation_importance": {"compare_instances": true, "priority": 2}'
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build actor prompt"""
        return self.build_actor_prompt_multi(context, strategy, results, [])

    def build_actor_prompt_multi(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Actor"""
        tool_results = results.get('tool_results', {})

        # Get image size constraint for vision modality
        image_width, image_height = self._get_image_size_from_results(tool_results)
        size_constraint = self._build_image_size_constraint(tool_results)

        # Different output format for vision vs other modalities
        if self.modality == "vision":
            output_format = '''"correct_instance_features": "natural language description of features leading to correct prediction",
        "wrong_instance_features": "natural language description of features leading to wrong prediction"'''
        elif self.modality == "text":
            output_format = '''"correct_instance_features": {
            "start_index": int,
            "end_index": int
        },
        "wrong_instance_features": {
            "start_index": int,
            "end_index": int
        }'''
        else:
            output_format = '''"correct_instance_features": {
            "feature_key": "feature name"
        },
        "wrong_instance_features": {
            "feature_key": "feature name"
        }'''

        prompt = f"""You are an XAI expert. Explain why similar instances have different prediction outcomes.

## Question
{context.get('user_question', self.question_template)}

## Context
- Instance A: Correctly classified
- Instance B: Incorrectly classified (similar to A but wrong prediction)
{size_constraint}
## XAI Analysis
{self._format_results_comprehensive(results)}

## Your Task
Identify and CONTRAST the features:
1. Features in the CORRECT instance that led to correct prediction
2. Features in the WRONG instance that led to wrong prediction

The goal is to find DISTINCT features - less overlap = better explanation.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format}
    }},
    "explanation": "2-3 sentences explaining the key difference between correct and wrong prediction",
    "confidence": 0.85
}}

**Critical Requirements:**
- For vision: Provide DISTINCT natural language descriptions (minimize word overlap)
- For text: Identify DIFFERENT spans in each instance
- For tabular: Identify DIFFERENT features
- The two feature sets should be as DISTINCT as possible
- Explain what makes one succeed and the other fail
- For vision with bounding boxes: MUST be within image bounds (x in [0, {image_width}], y in [0, {image_height}])

Respond with ONLY JSON:"""
        return prompt
