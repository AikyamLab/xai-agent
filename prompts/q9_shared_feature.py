"""
Q9: What shared feature makes multiple misclassified inputs difficult?

Expected Output:
- input_A, input_B, ...: regions in each misclassified input
- shared_feature_description: description of the common feature

Evaluation:
- Remove the shared feature from each input
- Metric: 1 if ALL inputs improve (P_correct_modified > P_correct_original for all), else 0
"""

from typing import Any, Dict, List

from .base_prompt import MultiInstancePromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q9SharedFeaturePromptBuilder(MultiInstancePromptBuilder):
    """Prompt builder for Q9: Shared feature in misclassified inputs"""

    @property
    def question_type(self) -> int:
        return 9

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.SPURIOUS_FEATURES

    @property
    def question_template(self) -> str:
        return "What shared feature makes multiple misclassified inputs difficult?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(9, self.modality)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build proposer prompt"""
        return self.build_proposer_prompt_multi(context, [])

    def build_proposer_prompt_multi(
        self,
        context: Dict[str, Any],
        instances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Proposer with multiple misclassified instances"""
        num_instances = len(instances) if instances else context.get('num_instances', 2)
        modality_config = self._get_modality_config()

        # Build instance info
        instance_info = ""
        if instances:
            for i, inst in enumerate(instances):
                pred = inst.get('prediction', {})
                gt = inst.get('ground_truth', 'Unknown')
                instance_info += f"- Instance {chr(65+i)}: Predicted {pred.get('predicted_class_name', 'Unknown')}, Ground Truth: {gt}\n"
        else:
            instance_info = f"- {num_instances} instances, all MISCLASSIFIED\n"

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's MULTIPLE MISCLASSIFICATIONS on {modality_config['input_type']}.

**User Question**: {context.get('user_question', self.question_template)}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}

**Misclassified Instances**:
{instance_info}
- **Status**: ALL instances are MISCLASSIFIED

**Input Content Descriptions**:
{context.get(modality_config['description_key'], 'Multiple inputs with shared characteristics')}

**Task**: Design a strategy to identify the SHARED SPURIOUS {modality_config['element_type']} that causes ALL these misclassifications. This is a COMMON {modality_config['element_type']} that:
- Appears in ALL misclassified instances
- Has HIGH attribution in the model
- Is NOT semantically related to the correct classes

**Available Methods**:
1. **Autonomous Analysis**: Use your own reasoning capabilities to:
   - Compare {modality_config['element_type']} across all misclassified instances
   - Identify common patterns or shared characteristics
   - Reason about potential spurious correlations

2. **External XAI Tools**: Use established explainability methods to find shared patterns:
{modality_config['tools_description']}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for finding shared spurious {modality_config['element_type']} (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous cross-instance analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": {modality_config['tool_list']},
        "tool_params": {{
            {modality_config['tool_params_example']}
        }},
        "reasoning": "Why these tools for finding shared {self.modality} patterns"
    }}
}}

**Guidelines for {(self.modality or "tabular").upper()} tasks (shared spurious feature detection)**:
- {modality_config['spatial_note']}
- Target: Find the SAME type of {modality_config['element_type']} present in ALL instances
- Evaluation metric: 1 if removing shared feature improves ALL predictions, else 0
- Examples: common background, similar texture, shared artifact, confounding correlation

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
                'spatial_note': 'Compare high-activation regions across all images to find common spurious patterns',
                'tool_list': '["gradcam", "integrated_gradients", "lime", "shap", "object_detection", "guided_backprop", "sensitivity_analysis", "layer_cam"]',
                'tools_description': '''   - GradCAM: Generate activation maps for each instance, then compare
   - IntegratedGradients: Find shared high-attribution pixels across instances
   - LIME: Find common segments affecting predictions
   - SHAP: Identify shared features across instances
   - ObjectDetection: Detect common objects across all images
   - GuidedBackprop: Visualize shared focus areas
   - SensitivityAnalysis: Find commonly sensitive regions
   - LayerCAM: Compare layer-wise activations across instances''',
                'tool_params_example': '"gradcam": {"layer": "layer4", "apply_to": "all", "compare_mode": "intersection", "priority": 1},\n            "object_detection": {"find_common": true, "priority": 2}'
            }
        elif self.modality == "text":
            return {
                'input_type': 'TEXT INPUTS',
                'description_key': 'text_description',
                'element_type': 'tokens/phrases',
                'spatial_note': 'Compare high-attribution tokens across all texts to find common spurious patterns',
                'tool_list': '["integrated_gradients", "lime", "shap", "attention_analysis", "token_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Find shared high-attribution tokens across instances
   - LIME: Find common tokens affecting predictions
   - SHAP: Identify shared token patterns across instances
   - AttentionAnalysis: Compare attention patterns across texts
   - TokenImportance: Find common high-importance tokens
   - SensitivityAnalysis: Find commonly sensitive tokens''',
                'tool_params_example': '"integrated_gradients": {"apply_to": "all", "compare_mode": "intersection", "priority": 1},\n            "attention_analysis": {"find_common_patterns": true, "priority": 2}'
            }
        else:  # tabular
            return {
                'input_type': 'TABULAR DATA INSTANCES',
                'description_key': 'data_description',
                'element_type': 'features/columns',
                'spatial_note': 'Compare high-importance features across all records to find common spurious patterns',
                'tool_list': '["integrated_gradients", "lime", "shap", "permutation_importance", "sensitivity_analysis"]',
                'tools_description': '''   - IntegratedGradients: Find shared high-attribution features across instances
   - LIME: Find common features affecting predictions
   - SHAP: Identify shared feature patterns across instances
   - PermutationImportance: Find consistently important spurious features
   - SensitivityAnalysis: Find commonly sensitive features''',
                'tool_params_example': '"shap": {"apply_to": "all", "compare_mode": "intersection", "priority": 1},\n            "permutation_importance": {"find_common": true, "priority": 2}'
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
        """Build prompt for Actor with multiple instances"""
        output_format = self._get_modality_specific_output_format()
        num_instances = len(instances) if instances else 2
        tool_results = results.get('tool_results', {})

        # Get image size constraint for vision modality
        image_width, image_height = self._get_image_size_from_results(tool_results)
        size_constraint = self._build_image_size_constraint(tool_results)

        # Include text/tabular instance data if available
        instance_data_section = self._format_instance_data_section(context)

        # Build output slots dynamically for all instances
        instance_slots = ",\n        ".join([
            f'"input_{chr(65 + i)}": {{\n            {output_format}\n        }}'
            for i in range(num_instances)
        ])

        prompt = f"""You are an XAI expert. Find the SHARED feature causing all misclassifications.

## Question
{context.get('user_question', self.question_template)}

## Context
Multiple instances ({num_instances}) are all misclassified.
Find what COMMON feature they share that confuses the model.
{size_constraint}{instance_data_section}

## XAI Analysis
{self._format_results_comprehensive(results)}

## Your Task
Identify the SHARED spurious feature and its location in EACH of the {num_instances} instances.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        {instance_slots}
    }},
    "shared_feature_description": "Description of the common feature across all inputs",
    "explanation": "2-3 sentences explaining how this shared feature confuses the model",
    "confidence": 0.0-1.0
}}

**Critical Requirements:**
- Provide a region for EVERY instance (input_A through input_{chr(65 + num_instances - 1)})
- Identify the SAME type of feature in EACH instance
- This shared feature should be SPURIOUS (not truly relevant to classification)
- Removing this feature from ALL instances should improve ALL predictions
- Examples: common background, similar texture, shared artifact
- For vision: bounding_box MUST be within image bounds (x in [0, {image_width}], y in [0, {image_height}])

Respond with ONLY JSON:"""
        return prompt
