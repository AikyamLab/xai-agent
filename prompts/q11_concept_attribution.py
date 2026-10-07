"""
Q11: Which concept was most responsible for the model's prediction? (CUB-only, vision)

Mirrors Q1 exactly on the Proposer/tool-execution side: same 8 vision
attribution tools (GradCAM, IntegratedGradients, LIME, SHAP, ObjectDetection,
GuidedBackprop, SensitivityAnalysis, LayerCAM), same ablation_mode handling,
no bespoke Q11 tool. The difference is entirely in what the Actor is asked to
report and how faithfulness is evaluated:

- Q1's Actor reports a spatial bounding_box; Q11's Actor reports a
  *concept name* (a human-annotated CUB attribute, e.g. "wing color: blue"),
  reasoning from the same tool output plus a grounding list of concepts
  actually present in this image (from concept_level/cav_quality.csv +
  ground-truth attribute labels -- see concept_level/build_q11_dataset.py).
  The grounding list keeps answers verifiable (same principle as Q1-Q10's
  hallucination checks: text spans must be exact substrings, tabular feature
  keys must exist in the dataset) without turning this into a multiple-choice
  task -- the Actor still has to reason about *which* of the grounded
  concepts the tool evidence points to.
- Q1's Evaluator masks the reported bbox in pixel space; Q11's Evaluator
  ablates the reported concept's CAV direction in representation space (see
  evaluation/explanation_faithfulness/q11_evaluator.py). No masking_utils
  involved at all.

Expected Output:
- Vision: concept_name (string, must match one of the grounded candidates)

Evaluation:
- Orthogonally project out the matched concept's CAV direction from the
  penultimate activation, re-run the classifier head, measure probability
  drop: max(0, P_original - P_modified)
"""

from typing import Any, Dict, List

from .base_prompt import PromptBuilder, QuestionCategory
from .output_schemas import get_output_schema


class Q11ConceptAttributionPromptBuilder(PromptBuilder):
    """Prompt builder for Q11: which concept was most responsible (CUB-only)"""

    @property
    def question_type(self) -> int:
        return 11

    @property
    def question_category(self) -> QuestionCategory:
        return QuestionCategory.CONCEPT_ATTRIBUTION

    @property
    def question_template(self) -> str:
        return "Which concept was most responsible for the model's prediction?"

    def get_output_schema(self) -> Dict[str, Any]:
        return get_output_schema(11, self.modality)

    def _get_modality_config(self) -> Dict[str, Any]:
        # Identical to Q1's vision config -- same tool menu, same tool
        # descriptions. Q11 only differs in what the Actor is asked to
        # extract from the tool evidence (a concept, not a region).
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
        }

    def _format_candidate_concepts(self, candidate_concepts: List[Dict[str, Any]]) -> str:
        if not candidate_concepts:
            return "(no grounded concept candidates available for this image)"
        lines = [f'   - "{c["concept_name"]}"' for c in candidate_concepts]
        return "\n".join(lines)

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer to select XAI tools -- same structure as Q1."""
        prediction = context.get('prediction', {})

        modality_config = self._get_modality_config()
        tools_description = self._get_available_tools_description(context, modality_config['tools_description'])
        tool_list = self._get_available_tools_list(context, modality_config['tool_list'])

        autonomous_bullets = (
            f"   - Identify and ground important {modality_config['element_type']} in the input\n"
            f"   - Reason about their importance to the predicted class\n"
            f"   - Provide {modality_config['location_type']} for different {modality_config['element_type']}"
        )
        methods_section = self._build_methods_section(
            context, autonomous_bullets=autonomous_bullets, tools_description=tools_description
        )
        strategy_schema = self._build_strategy_schema_block(
            context,
            tool_list=tool_list,
            reasoning_hint="Explain why you chose this strategy for finding the most responsible concept (2-3 sentences)",
        )

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on {modality_config['input_type']}.

**User Question**: {context.get('user_question') or self.question_template}

**Model Information**:
- Model: {context.get('model_info', {}).get('model_name', 'Unknown')}
- Architecture: {context.get('model_info', {}).get('architecture', 'Unknown')}
- Prediction: Class {prediction.get('predicted_class_idx')}
(Confidence: {prediction.get('confidence', 0.0):.4f})
- Top-5 Predictions: {prediction.get('top5_predictions', [])}

**Task**: Design a comprehensive strategy to identify which {modality_config['region_type']} of the input were MOST RESPONSIBLE for this prediction. This evidence will then be used to name the specific visual CONCEPT (a semantic attribute such as the color, shape, or pattern of a particular body part) that region corresponds to.

{methods_section}

**Your Response Must Be Valid JSON** with the following structure:
{strategy_schema}

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor to name the most-responsible concept."""
        prediction = context.get('prediction', {})
        size_constraint = self._build_image_size_constraint(results.get('tool_results', {}))
        candidate_concepts = context.get('candidate_concepts', [])

        prompt = f"""You are an XAI expert. Based on the analysis below, identify which CONCEPT was MOST RESPONSIBLE for this prediction.

## Question
{context.get('user_question') or self.question_template}

## Model Prediction
Class: {prediction.get('predicted_class_name', prediction.get('predicted_class_idx', 'Unknown'))}
Confidence: {prediction.get('confidence', 0.0):.4f}
{size_constraint}
## XAI Analysis
{self._format_results_comprehensive(results, context)}

## Candidate Concepts Present In This Image
A concept is a specific visual attribute (the color/shape/pattern of a body
part, e.g. "wing color: blue", "bill shape: hooked"). The following concepts
are confirmed present in this image:
{self._format_candidate_concepts(candidate_concepts)}

## Your Task
Using the XAI analysis above (which region/pixels the model relied on), decide
which ONE of the candidate concepts listed above best explains that evidence
-- i.e. which concept was most responsible for the prediction.

## REQUIRED OUTPUT FORMAT (JSON only)
{{
    "output": {{
        "concept_name": "string, copied EXACTLY from the candidate list above"
    }},
    "explanation": "2-3 sentences connecting the XAI evidence to this concept"
}}

**Critical Requirements:**
- concept_name MUST be copied verbatim (exact string) from the candidate list above
- Do NOT invent a concept that is not in the candidate list
- Base your decision on the attribution analysis, not just the concept list alone

Respond with ONLY JSON:"""
        return prompt
