"""
Q4: Why are instances A and B given different predictions?

Expected Output:
- input_A: region in A that causes its prediction
- input_B: region in B that causes its prediction

Evaluation:
- Mask both identified parts in both inputs
- Metric: 1 if (P_A_a1_orig - P_B_a1_orig) - (P_A_a1_mod - P_B_a1_mod) > 0, else 0
  (Success if the probability gap reduces after masking)
"""

from typing import Any, Dict, List

from .base_prompt import MultiInstancePromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q4ContrastiveInstancesPromptBuilder(MultiInstancePromptBuilder):
    """Prompt builder for Q4: Why different predictions for A and B"""

    @property
    def question_type(self) -> int:
        return 4

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.FEATURE_ATTRIBUTION

    @property
    def question_template(self) -> str:
        return "Why are instances A and B given different predictions?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(4, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build proposer prompt for comparing two instances"""
        return self.build_proposer_prompt_multi(context, [])

    def build_proposer_prompt_multi(
        self,
        context: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Proposer with multiple instances"""
        modality_config = self._get_modality_config()
        tools_description = self._get_available_tools_description(context, modality_config['tools_description'])
        tool_list = self._get_available_tools_list(context, modality_config['tool_list'])

        if self.modality != 'vision' and not (context.get(modality_config['description_key'] + '_A') or context.get(modality_config['description_key'])):
            raise ValueError(f"Input content description missing for {self.modality} modality")

        # Extract instance predictions if available
        instance_info = ""
        if instances:
            for i, inst in enumerate(instances):
                pred = inst.get('prediction', {})
                instance_info += f"- Instance {chr(65+i)}: Predicted {pred.get('predicted_class_name', 'Unknown')} (Confidence: {pred.get('confidence', 0):.4f})\n"
        else:
            instance_info = "- Instance A: Different prediction from B\n- Instance B: Different prediction from A\n"

        desc_key = modality_config['description_key']
        has_desc = context.get(desc_key + '_A') or context.get(desc_key)
        desc_section = (
            "**Input Content Descriptions**:\n- Instance A: "
            + str(context.get(desc_key + '_A', context.get(desc_key)))
            + "\n- Instance B: "
            + str(context.get(desc_key + '_B', ''))
            + "\n"
        ) if has_desc else ""

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's predictions on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}

**Instance Predictions**:
{instance_info}

{desc_section}**Task**: Design a comprehensive strategy to identify which {modality_config['region_type']} in EACH instance cause their DIFFERENT predictions. Find:
- What {modality_config['element_type']} in Instance A cause prediction A
- What {modality_config['element_type']} in Instance B cause prediction B

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Compare {modality_config['element_type']} between the two instances
   - Reason about what causes each instance's different prediction
   - Provide {modality_config['location_type']} for decisive {modality_config['element_type']} in each instance

2. **External XAI Tools**: Use established explainability methods for comparative analysis:
{tools_description}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for comparing {modality_config['element_type']} between instances (2-3 sentences)",
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous comparative analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {tool_list},
        "reasoning": "Why these tools for comparing {self.modality} instances"
    }}
}}

Provide your strategy as a JSON object:
"""
        return prompt

    def _get_modality_config(self) -> Dict[str, Any]:
        """Get modality-specific configuration for prompt building"""
        if self.modality == "vision":
            return {
                'input_type': 'IMAGES',
                'description_key': 'image_description',
                'region_type': 'SPATIAL REGIONS',
                'element_type': 'objects/regions',
                'location_type': 'bounding boxes',
                'spatial_note': 'Compare visual features between instances - identify distinctive regions in each image',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Generate activation maps for each instance
   - IntegratedGradients: Compute pixel-level attribution for each instance
   - LIME: Generate local explanations for each instance
   - SHAP: Compute feature importance for each instance
   - ObjectDetection: Detect objects in each instance for comparison
   - GuidedBackprop: Visualize gradients for each instance
   - SensitivityAnalysis: Measure sensitivity in each instance
   - LayerCAM: Layer-wise activation for each instance''',
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT INPUTS',
                'description_key': 'text_description',
                'region_type': 'TEXT SPANS',
                'element_type': 'tokens/phrases',
                'location_type': 'start and end indices',
                'spatial_note': 'Compare textual features between instances - identify distinctive tokens in each text',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Compute token attribution for each instance
   - LIME: Generate local explanations for each instance
   - SHAP: Compute token importance for each instance
   - AttentionAnalysis: Analyze attention patterns in each instance
   - TokenImportance: Compute token-level attribution for each instance
   - SensitivityAnalysis: Measure sensitivity in each instance''',
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA INSTANCES',
                'description_key': 'data_description',
                'region_type': 'FEATURES',
                'element_type': 'features/columns',
                'location_type': 'feature names',
                'spatial_note': 'Compare feature values between instances - identify distinctive features in each record',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Compute feature attribution for each instance
   - LIME: Generate local explanations for each instance
   - SHAP: Compute Shapley values for each instance
   - PermutationImportance: Measure feature importance for each instance
   - SensitivityAnalysis: Measure sensitivity in each instance''',
            }

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build actor prompt"""
        return self.build_actor_prompt_multi(context, strategy, results, [])

    def _get_image_size_from_multi_results(self, tool_results: Dict[str, Any]) -> tuple:
        """Extract image size from multi-instance tool results"""
        # Try instance_A first, then instance_B
        for instance_key in ['instance_A', 'instance_B']:
            instance_results = tool_results.get(instance_key, {})
            for tool_name, result in instance_results.items():
                if isinstance(result, dict) and result.get('success'):
                    img_size = result.get('original_image_size', {})
                    if img_size:
                        return img_size.get('width', 224), img_size.get('height', 224)
        return 224, 224

    def build_actor_prompt_multi(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Actor with multiple instances"""
        tool_results = results.get('tool_results', {})

        # Different output format for vision vs other modalities
        if self.modality == "vision":
            output_format_block = '''"input_A": "concise feature phrase for prediction A (e.g. 'fur texture and primate facial features')",
        "input_B": "concise feature phrase for prediction B (e.g. 'wings and fuselage body')"'''
            critical_reqs = """- For vision: Output a SHORT feature phrase — only name the concrete visual features/objects/patterns
- Do NOT write full sentences
"""
        elif self.modality == "text":
            output_format_block = '''"input_A": {
            "text_spans": ["exact phrase from instance A text"]
        },
        "input_B": {
            "text_spans": ["exact phrase from instance B text"]
        }'''
            critical_reqs = """- For text: text_spans as a list of exact phrase strings copied verbatim from each instance's text (one or more phrases per instance)
"""
        else:
            output_format_block = '''"input_A": {
            "feature_keys": ["feature_name_1", "feature_name_2"]
        },
        "input_B": {
            "feature_keys": ["feature_name_1", "feature_name_2"]
        }'''
            critical_reqs = """- For tabular: feature_keys as a list of column names for each instance
- List the most decisive features, using exact column names
"""

        # Include text/tabular instance data if available
        instance_data_section = self._format_instance_data_section(context)
        output_size_constraint = self._build_output_size_constraint(context) if self.modality != 'vision' else ""

        prompt = f"""You are an XAI expert. Explain why instances A and B have DIFFERENT predictions.

## Question
{context.get('user_question', self.question_template)}
{instance_data_section}
## XAI Analysis
{self._format_results_comprehensive(results, context)}

## Your Task
Identify the DECISIVE parts in BOTH instances:
- Part in A that causes A's prediction
- Part in B that causes B's prediction

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {output_format_block}
    }},
    "explanation": "2-3 sentences explaining the key differences"
}}

**Critical Requirements:**
{critical_reqs}{output_size_constraint}

Respond with ONLY JSON:"""
        return prompt
