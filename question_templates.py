"""
Question Templates for XAI Agent Framework

Defines question categories, templates, and extraction specifications for different
explanation types across vision, text, and tabular modalities.

Each modality has its own PromptBuilder classes for Proposer and Actor agents.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json


class QuestionCategory(Enum):
    """Categories of explanation questions"""
    FEATURE_ATTRIBUTION = "feature_attribution"
    COUNTERFACTUAL = "counterfactual"
    SPURIOUS_FEATURES = "spurious_features"


class Modality(Enum):
    """Data modality types"""
    VISION = "vision"
    TEXT = "text"
    TABULAR = "tabular"


class AttributionType(Enum):
    """Types of feature attribution"""
    MOST = "most"  # Most responsible features
    LEAST = "least"  # Least responsible features
    TOP_K = "top_k"  # Top K important features
    DISTINCTIVE = "distinctive"  # Features that distinguish from other classes


@dataclass
class ExtractionField:
    """Defines a field to be extracted from agent output"""
    name: str
    field_type: str  # e.g., "List[BoundingBox]", "str", "Dict[str, float]"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates"""
    x1: int  # top-left x
    y1: int  # top-left y
    x2: int  # bottom-right x
    y2: int  # bottom-right y
    confidence: float = 1.0
    label: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "label": self.label
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BoundingBox':
        return cls(
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
            confidence=data.get("confidence", 1.0),
            label=data.get("label")
        )


@dataclass
class ImageModification:
    """Describes a specific modification to an image for counterfactuals"""
    region: Optional[BoundingBox] = None  # Region to modify, None means global
    modification_type: str = ""  # e.g., "color_change", "add_object", "remove_object"
    parameters: Dict[str, Any] = field(default_factory=dict)  # Specific parameters
    description: str = ""  # Natural language description
    
    def to_dict(self) -> Dict:
        return {
            "region": self.region.to_dict() if self.region else None,
            "modification_type": self.modification_type,
            "parameters": self.parameters,
            "description": self.description
        }


# ============================================================================
# ABSTRACT PROMPT BUILDER BASE CLASSES
# ============================================================================

class PromptBuilder(ABC):
    """Abstract base class for building prompts for different question types"""
    
    @abstractmethod
    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Proposer Agent"""
        pass
    
    @abstractmethod
    def build_actor_prompt(
        self, 
        context: Dict[str, Any], 
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """Build prompt for Actor Agent"""
        pass


# ============================================================================
# VISION MODALITY PROMPT BUILDERS
# ============================================================================

class VisionPromptBuilder(PromptBuilder):
    """Base prompt builder for Vision modality with Vision-specific features"""
    
    @staticmethod
    def build_image_analysis_prompt(image_path: Optional[str] = None) -> str:
        """
        Build prompt for analyzing image content (Vision-specific)

        This provides the Proposer with understanding of what's in the image
        before deciding on analysis strategy.

        Args:
            image_path: Optional path to the image being analyzed. If provided,
                       it will be included in the prompt for context.

        Returns:
            Prompt string for image analysis
        """
        image_context = ""
        if image_path:
            image_context = f"\n        Image being analyzed: {image_path}\n"

        prompt = f"""Please provide a detailed description of this image, focusing on:
        1. Main objects and subjects
        2. Spatial layout and composition
        3. Colors, textures, and visual patterns
        4. Any notable features or characteristics
{image_context}
        Keep the description objective and comprehensive. This will help in planning the explainability analysis.
        """
        return prompt


# TODO: Modify and finish the proposer prompt and actor prompt builder based on the 10 question templates
class VisionFeatureAttributionPromptBuilder(VisionPromptBuilder):
    """
    Prompt builder for Feature Attribution questions in Vision modality

    Supports different attribution types:
    - "most": Find most responsible regions
    - "least": Find least responsible regions
    - "distinctive": Find regions that distinguish from alternatives
    - "contrastive": Compare different instances
    """

    def __init__(self, attribution_type: str = "most"):
        """
        Initialize Vision Feature Attribution Prompt Builder

        Args:
            attribution_type: Type of attribution ("most", "least", "distinctive", "contrastive")
        """
        super().__init__()
        self.attribution_type = attribution_type

        # Define attribution-specific terminology
        self._attribution_config = {
            "most": {
                "target": "most responsible",
                "adjective": "important",
                "verb": "contributed to",
                "noun": "contribution",
                "extraction_focus": "highly salient regions",
                "tool_priority": "high importance",
                "metric": "MoRF"
            },
            "least": {
                "target": "least responsible",
                "adjective": "irrelevant",
                "verb": "had minimal impact on",
                "noun": "irrelevance",
                "extraction_focus": "regions with minimal impact",
                "tool_priority": "low importance",
                "metric": "LeRF"
            },
            "distinctive": {
                "target": "most distinctive",
                "adjective": "distinguishing",
                "verb": "distinguished from alternatives",
                "noun": "distinctiveness",
                "extraction_focus": "discriminative regions",
                "tool_priority": "contrastive importance",
                "metric": "Acc/MoRF"
            },
            "contrastive": {
                "target": "differentiating",
                "adjective": "contrasting",
                "verb": "differentiated between",
                "noun": "difference",
                "extraction_focus": "instance-specific regions",
                "tool_priority": "comparative importance",
                "metric": "MoRF"
            }
        }

        if attribution_type not in self._attribution_config:
            raise ValueError(f"Invalid attribution_type: {attribution_type}. "
                           f"Must be one of {list(self._attribution_config.keys())}")

    def build_proposer_prompt(self, context: Dict[str, Any], for_react_agent: bool = True) -> str:
        """
        Build prompt for Proposer to decide strategy for Vision Feature Attribution

        Args:
            context: Dict containing:
                - user_question: str
                - image_path: str
                - model_info: Dict
                - prediction: Dict
                - image_description: str (from analyze_image_content)
            for_react_agent: If True, format prompt for ReAct agent with Thought/Action/Final Answer
        """
        config = self._attribution_config[self.attribution_type]

        if for_react_agent:
            # ReAct-compatible prompt format
            prompt = f"""You are a specialized XAI strategy agent. You MUST follow the ReAct format strictly.

**Context - Vision Feature Attribution ({self.attribution_type})**:
- User Question: {context['user_question']}
- Model: {context['model_info'].get('model_name', 'Unknown')} ({context['model_info'].get('architecture', 'Unknown')})
- Prediction: Class {context['prediction'].get('predicted_class_idx')} (Confidence: {context['prediction'].get('confidence', 0.0):.4f})
- Task: Identify which SPATIAL REGIONS were {config['target']} for this prediction

**Valid XAI tool_name values**: gradcam, integrated_gradients, lime, shap, object_detection

**Final Answer JSON Structure** (you MUST use this exact format):
{{
    "strategy_type": "tools",
    "reasoning": "Brief explanation of tool selection (1-2 sentences)",
    "selected_tools": [
        {{
            "tool_name": "gradcam",
            "priority": 1,
            "reasoning": "Why this tool for {config['extraction_focus']}",
            "parameters": {{}}
        }}
    ],
    "autonomous_tasks": []
}}

**ReAct Format Rules**:
1. First, use 'list_available_tools' to check available tools
2. Then provide your Final Answer with the JSON strategy

When using a tool:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input string>

When ready to answer:
Thought: I have the information needed to create a strategy.
Final Answer: <your JSON object>

Begin with Thought:"""
        else:
            # Original prompt format for direct VLM calls
            prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a machine learning model's prediction on an IMAGE.

**User Question**: {context['user_question']}

**Model Information**:
- Model: {context['model_info'].get('model_name', 'Unknown')}
- Architecture: {context['model_info'].get('architecture', 'Unknown')}
- Prediction: Class {context['prediction'].get('predicted_class_idx')}
(Confidence: {context['prediction'].get('confidence', 0.0):.4f})
- Top-5 Predictions: {context['prediction'].get('top5_predictions', [])}

**Image Content Description**:
{context.get('image_description', 'Not available')}

**Task**: Design a comprehensive strategy to identify which SPATIAL REGIONS of the input image were {config['target']} for this prediction.

**Available Methods**:
1. **Autonomous Visual Analysis**: Use your own vision-language reasoning capabilities to:
   - Identify and ground {config['adjective']} objects/regions in the image
   - Reason about their {config['noun']} to the predicted class
   - Provide bounding boxes and confidence scores for different regions

2. **External XAI Tools**: Use established explainability methods:
   - GradCAM: Highlights {config['adjective']} regions using gradient-based activation
   - LIME: Segments image and tests which segments affect prediction
   - IntegratedGradients: Computes pixel-level importance scores
   - SHAP: Game-theoretic approach to feature {config['noun']}

**Your Response Must Be Valid JSON** with the following structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Explain why you chose this strategy for finding {config['extraction_focus']} (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning" | "comparison",
            "query": "Specific query for autonomous VISUAL analysis",
            "expected_output": "What should be extracted from this task"
        }}
    ],
    "tool_selection": {{
        "selected_tools": ["gradcam", "lime", ...],
        "tool_params": {{
            "gradcam": {{"layer": "layer4", "priority": 1}},
            "lime": {{"num_samples": 1000, "priority": 2}}
        }},
        "reasoning": "Why these VISION-specific tools"
    }}
}}

**Guidelines for VISION tasks ({self.attribution_type} attribution)**:
- Consider spatial nature of images - bounding boxes are crucial
- Target: Find regions that {config['verb']} the prediction
- Evaluation metric: {config['metric']}

Provide your strategy as a JSON object:
"""
        return prompt
    
    def build_actor_prompt(
        self, 
        context: Dict[str, Any], 
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Build prompt for Actor to generate explanation with structured extraction for VISION
        
        Args:
            context: Dict containing user_question, model_info, prediction, image_path
            strategy: Dict from Proposer Agent
            results: Dict containing autonomous_results and/or tool_results
        """
        config = self._attribution_config[self.attribution_type]

        prompt = f"""You are an AI explainability expert providing a comprehensive explanation based on user questions and the model predictions.

        **User Question**: {context['user_question']}

        **Model Prediction**:
        - Predicted Class: {context['prediction'].get('predicted_class_idx')}
        - Class Name: {context['prediction'].get('predicted_class_name', 'Unknown')}
        - Confidence: {context['prediction'].get('confidence', 0.0):.4f}

        **Analysis Results Available**:
        """

        # Add autonomous results if available
        if results.get('autonomous_results'):
            prompt += "\n**Autonomous Visual Analysis**:\n"
            for task_name, task_result in results['autonomous_results'].items():
                result_data = task_result.get('result', {})
                prompt += f"""
        - Task: {task_result.get('task_type', 'unknown')}
        - Query: {task_result.get('query', 'N/A')}
        - Answer: {result_data.get('answer', 'N/A')}
        - Regions Identified: {result_data.get('num_regions', 0)}
        - Bounding Boxes: {result_data.get('bboxes', [])}
        """

        # Add tool results if available
        if results.get('tool_results'):
            prompt += "\n**External XAI Tool Results**:\n"
            for tool_name, tool_info in results['tool_results'].items():
                prompt += f"""
        - {tool_name.replace('_', ' ').title()}:
        - Result: {tool_info.get('result', 'N/A')}
        - Key Statistics: {tool_info.get('statistics', {})}
        - Visualization: {tool_info.get('visualization_path', 'Not generated')}
        """

        # Add VLM-extracted features if available (from feature extraction step)
        if results.get('extracted_features'):
            ef = results['extracted_features']
            prompt += "\n**VLM Feature Extraction Results** (pre-analyzed from visualizations):\n"

            # Add responsible regions if available
            regions = ef.get('responsible_regions', [])
            if regions:
                prompt += f"        - Identified Regions: {len(regions)} region(s) found\n"
                for i, region in enumerate(regions[:3]):  # Show top 3
                    label = region.get('label', f'Region {i+1}')
                    importance = region.get('importance', 0.0)
                    bbox = region.get('bbox', {})
                    prompt += f"          * {label}: importance={importance:.2f}, bbox={bbox}\n"

            # Add importance distribution
            imp_dist = ef.get('importance_distribution', {})
            if imp_dist:
                prompt += f"        - Importance Distribution:\n"
                prompt += f"          * Primary: {imp_dist.get('primary_region_contribution', 0):.2f}\n"
                prompt += f"          * Secondary: {imp_dist.get('secondary_regions_contribution', 0):.2f}\n"
                prompt += f"          * Background: {imp_dist.get('background_contribution', 0):.2f}\n"

            # Add key visual features
            key_features = ef.get('key_visual_features', [])
            if key_features:
                prompt += f"        - Key Visual Features: {', '.join(key_features[:5])}\n"

            # Add VLM explanation if available
            vlm_explanation = ef.get('explanation', '')
            if vlm_explanation:
                prompt += f"        - VLM Analysis: {vlm_explanation[:200]}...\n"

        prompt += f"""

        **Your Task**: Generate a comprehensive explanation for this VISION task ({self.attribution_type} attribution) that:
        1. Directly answers which SPATIAL REGIONS were {config['target']}
        2. Provides specific bounding boxes for these regions (pixel coordinates)
        3. Assigns {config['noun']} scores to different regions
        4. Explains WHY these regions {config['verb']} this specific prediction
        5. Synthesizes findings from all analysis methods

        **CRITICAL: Your response MUST be valid JSON** with this exact structure:
        {{
            "explanation": "Natural language explanation (3-5 sentences) directly answering which regions were {config['target']}",
            "extracted_features": {{
                "responsible_regions": [
                    {{
                        "bbox": {{
                            "x1": int (pixel coordinates),
                            "y1": int,
                            "x2": int,
                            "y2": int
                        }},
                        "label": "Description of what's in this SPATIAL region",
                        "{config['noun']}_score": float (0.0-1.0),
                        "reasoning": "Why this region {config['verb']} the prediction"
                    }}
                ],
                "importance_distribution": {{
                    "primary_region_contribution": float (0.0-1.0),
                    "secondary_regions_contribution": float (0.0-1.0),
                    "background_contribution": float (0.0-1.0)
                }},
                "key_visual_features": [
                    "List of specific VISUAL features that {config['verb']} the prediction"
                ]
            }},
            "confidence": float (0.0-1.0, how confident are you in this explanation),
            "evidence_summary": {{
                "autonomous_analysis": "Summary of what autonomous VISUAL analysis revealed",
                "tool_analysis": "Summary of what XAI tools revealed",
                "agreement_level": "high|medium|low - do different methods agree?"
            }}
        }}

        **Guidelines for Bounding Boxes (VISION-specific)**:
        - Use pixel coordinates (integers) based on the original image dimensions
        - x1, y1 is the top-left corner
        - x2, y2 is the bottom-right corner
        - Ensure x2 > x1 and y2 > y1
        - Multiple regions are allowed and encouraged if {config['noun']} is distributed
        - Order by {config['noun']} ({self.attribution_type} first)

        **Guidelines for {config['noun'].capitalize()} Scores**:
        - For "{self.attribution_type}" attribution: Higher scores indicate {config['target']} regions
        - Scores should be meaningful and comparable across regions
        - Consider both autonomous and tool-based evidence
        - Be honest if {config['noun']} is uncertain (reflect in confidence score)

        Provide your explanation as a JSON object:
        """
        return prompt


class VisionCounterfactualPromptBuilder(VisionPromptBuilder):
    """
    Prompt builder for Counterfactual questions in Vision modality
    Example: "How should the IMAGE change to flip the model into {target}?"
    """

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build prompt for Proposer to decide strategy for Vision Counterfactual Analysis

        Args:
            context: Dict containing:
                - user_question: str
                - image_path: str
                - model_info: Dict
                - prediction: Dict
                - image_description: str (optional)
                - target_class: str (optional, for specific counterfactual)
        """
        target_info = context.get('target_class', 'a different prediction')

        prompt = f"""You are an AI explainability expert designing a strategy for COUNTERFACTUAL ANALYSIS on an image classification model.

        **User Question**: {context['user_question']}

        **Model Information**:
        - Model: {context['model_info'].get('model_name', 'Unknown')}
        - Architecture: {context['model_info'].get('architecture', 'Unknown')}
        - Current Prediction: Class {context['prediction'].get('predicted_class_idx')}
        (Confidence: {context['prediction'].get('confidence', 0.0):.4f})
        - Top-5 Predictions: {context['prediction'].get('top5_predictions', [])}

        **Target**: {target_info}

        **Image Content Description**:
        {context.get('image_description', 'Not available')}

        **Task**: Design a strategy to identify how the IMAGE should be MODIFIED to change the prediction to the target class.

        **Available Methods**:
        1. **Autonomous Visual Analysis**: Use your vision capabilities to:
        - Analyze differences between current and target class visual patterns
        - Identify regions that should be modified
        - Reason about what changes would flip the prediction
        - Provide modification descriptions (color, texture, object changes)

        2. **External XAI Tools**: Use gradient-based analysis:
        - GradCAM: Identify currently important regions
        - IntegratedGradients: Compute pixel-level importance
        - Counterfactual generation tools (if available)

        3. **Image Generation Tools**: For vision counterfactuals:
        - Stable Diffusion inpainting: Modify specific regions
        - Object removal/addition tools
        - Style transfer methods

        **Your Response Must Be Valid JSON** with this structure:
        {{
            "strategy_type": "autonomous" | "tools" | "hybrid",
            "reasoning": "Why you chose this strategy for COUNTERFACTUAL generation (2-3 sentences)",
            "confidence": 0.0-1.0,
            "autonomous_tasks": [
                {{
                    "task_type": "counterfactual_analysis" | "reasoning" | "comparison",
                    "query": "Specific query for autonomous visual analysis",
                    "expected_output": "What should be extracted from this task"
                }}
            ],
            "tool_selection": {{
                "selected_tools": ["gradcam", "stable_diffusion", ...],
                "tool_params": {{
                    "gradcam": {{"layer": "layer4", "priority": 1}},
                    "stable_diffusion": {{"prompt_guidance": "auto", "priority": 2}}
                }},
                "reasoning": "Why these tools for COUNTERFACTUAL generation"
            }}
        }}

        **Guidelines for VISION Counterfactuals**:
        - Consider what visual features distinguish current class from target class
        - For object-based changes, use image generation tools
        - For saliency-based changes, use gradient methods first
        - Hybrid approach: gradient analysis + image generation is often most effective
        - Consider feasibility: some changes may be impossible without breaking realism

        Provide your strategy as a JSON object:
        """
        return prompt

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Build prompt for Actor to generate counterfactual explanation for VISION

        Args:
            context: Dict containing user_question, model_info, prediction, image_path
            strategy: Dict from Proposer Agent
            results: Dict containing autonomous_results and/or tool_results
        """
        target_info = context.get('target_class', 'a different prediction')

        prompt = f"""You are an AI explainability expert providing a COUNTERFACTUAL explanation for an image classification model.

**User Question**: {context['user_question']}

**Current Model Prediction**:
- Predicted Class: {context['prediction'].get('predicted_class_idx')}
- Class Name: {context['prediction'].get('predicted_class_name', 'Unknown')}
- Confidence: {context['prediction'].get('confidence', 0.0):.4f}

**Target Prediction**: {target_info}

**Analysis Results Available**:
"""

        # Add autonomous results
        if results.get('autonomous_results'):
            prompt += "\n**Autonomous Counterfactual Analysis**:\n"
            for task_name, task_result in results['autonomous_results'].items():
                result_data = task_result.get('result', {})
                prompt += f"""
- Task: {task_result.get('task_type', 'unknown')}
- Query: {task_result.get('query', 'N/A')}
- Answer: {result_data.get('answer', 'N/A')}
- Identified Changes: {result_data.get('suggested_changes', [])}
"""

        # Add tool results
        if results.get('tool_results'):
            prompt += "\n**XAI Tool Results**:\n"
            for tool_name, tool_info in results['tool_results'].items():
                prompt += f"""
- {tool_name.replace('_', ' ').title()}:
  - Result: {tool_info.get('result', 'N/A')}
  - Statistics: {tool_info.get('statistics', {})}
"""

        # Add VLM-extracted features if available
        if results.get('extracted_features'):
            ef = results['extracted_features']
            prompt += "\n**VLM Feature Extraction Results** (pre-analyzed from visualizations):\n"

            regions = ef.get('responsible_regions', [])
            if regions:
                prompt += f"- Identified Regions for Modification: {len(regions)} region(s)\n"
                for i, region in enumerate(regions[:3]):
                    label = region.get('label', f'Region {i+1}')
                    importance = region.get('importance', 0.0)
                    bbox = region.get('bbox', {})
                    prompt += f"  * {label}: importance={importance:.2f}, bbox={bbox}\n"

            key_features = ef.get('key_visual_features', [])
            if key_features:
                prompt += f"- Key Visual Features: {', '.join(key_features[:5])}\n"

            vlm_explanation = ef.get('explanation', '')
            if vlm_explanation:
                prompt += f"- VLM Analysis: {vlm_explanation[:200]}...\n"

        prompt += f"""

**Your Task**: Generate a comprehensive COUNTERFACTUAL explanation that:
1. Specifies EXACTLY what IMAGE modifications are needed
2. Provides specific regions (bounding boxes) that need to change
3. Describes the type of modification (color, texture, object addition/removal, etc.)
4. Explains WHY these changes would flip the prediction
5. Assesses the feasibility and realism of these changes

**CRITICAL: Your response MUST be valid JSON** with this structure:
{{
    "explanation": "Natural language explanation of how to change the IMAGE (3-5 sentences)",
    "extracted_features": {{
        "modification_plan": [
            {{
                "region": {{
                    "x1": int, "y1": int, "x2": int, "y2": int
                }} or null (for global changes),
                "modification_type": "color_change" | "texture_change" | "add_object" | "remove_object" | "style_transfer",
                "parameters": {{
                    "specific parameters for this modification type"
                }},
                "description": "Natural language description of this modification",
                "importance": float (0.0-1.0, how critical is this change),
                "reasoning": "Why this change helps flip to target class"
            }}
        ],
        "target_prediction": "{target_info}",
        "modification_difficulty": float (0.0-1.0, estimated difficulty),
        "feasibility_assessment": "Assessment of how realistic/achievable these changes are"
    }},
    "confidence": float (0.0-1.0, confidence in this counterfactual),
    "evidence_summary": {{
        "autonomous_analysis": "Summary of autonomous analysis",
        "tool_analysis": "Summary of XAI tool findings",
        "synthesis": "How different methods agree on the counterfactual"
    }}
}}

**Guidelines for Modification Types (VISION-specific)**:
- color_change: {{target_color: [R, G, B], intensity: float}}
- texture_change: {{target_texture: str, strength: float}}
- add_object: {{object_type: str, position: "center"|"left"|"right", size: float}}
- remove_object: {{object_id: str}}
- style_transfer: {{target_style: str}}

**Guidelines for Region Specifications**:
- Use pixel coordinates (integers)
- null region means global/whole-image modification
- Multiple modifications can be specified for different regions
- Order by importance (most critical change first)

Provide your counterfactual explanation as a JSON object:
"""
        return prompt


class VisionSpuriousFeaturesPromptBuilder(VisionPromptBuilder):
    """
    Prompt builder for Spurious Features questions in Vision modality
    Example: "Are there irrelevant VISUAL parts that influenced the prediction?"
    """

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build prompt for Proposer to decide strategy for Vision Spurious Feature Detection

        Args:
            context: Dict containing:
                - user_question: str
                - image_path: str or List[str] (multiple images for Q9)
                - model_info: Dict
                - prediction: Dict or List[Dict] (for multiple images)
                - image_description: str or List[str]
                - ground_truth: Optional[str] (for wrong prediction analysis)
        """
        is_multi_image = isinstance(context.get('image_path'), list)

        prompt = f"""You are an AI explainability expert designing a strategy to detect SPURIOUS FEATURES that influenced the model's prediction.

**User Question**: {context['user_question']}

**Model Information**:
- Model: {context['model_info'].get('model_name', 'Unknown')}
- Architecture: {context['model_info'].get('architecture', 'Unknown')}
"""

        if is_multi_image:
            prompt += f"""
**Multiple Images Analysis**:
- Number of images: {len(context['image_path'])}
- All images were misclassified
- Task: Find SHARED spurious features across all images
"""
            predictions = context.get('prediction', [])
            for i, pred in enumerate(predictions[:3]):  # Show first 3
                prompt += f"\n- Image {i+1}: Predicted as {pred.get('predicted_class_name', 'Unknown')} (Should be: {context.get('ground_truth', ['Unknown'])[i] if isinstance(context.get('ground_truth'), list) else 'Unknown'})"
        else:
            prompt += f"""
**Single Image Analysis**:
- Prediction: Class {context['prediction'].get('predicted_class_idx')}
  (Confidence: {context['prediction'].get('confidence', 0.0):.4f})
- Ground Truth: {context.get('ground_truth', 'Unknown')}
- Status: {'Misclassified' if context.get('ground_truth') else 'Unknown'}
"""

        prompt += f"""

**Task**: Design a strategy to identify SPURIOUS (irrelevant) visual features that influenced the {'wrong' if context.get('ground_truth') else ''} prediction.

**Available Methods**:
1. **Autonomous Visual Analysis**: Use your vision capabilities to:
   - Identify background elements, textures, or patterns
   - Detect features NOT related to the predicted/ground-truth class
   - Compare foreground (relevant) vs background (potentially spurious)
   - For multiple images: find common visual elements across failures

2. **External XAI Tools**: Use attribution methods:
   - GradCAM: Highlight what the model focuses on
   - RISE: Region-based importance estimation
   - Occlusion: Test impact of removing different regions
   - Compare attribution with semantic relevance

3. **Statistical Analysis** (for multiple images):
   - Cluster analysis of visual features
   - Correlation analysis between features and misclassifications
   - Common pattern detection

**Your Response Must Be Valid JSON** with this structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Why you chose this strategy for SPURIOUS feature detection (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "spurious_detection" | "comparison" | "reasoning",
            "query": "Specific query for detecting spurious features",
            "expected_output": "What should be extracted"
        }}
    ],
    "tool_selection": {{
        "selected_tools": ["gradcam", "rise", "occlusion", ...],
        "tool_params": {{
            "gradcam": {{"layer": "layer4", "priority": 1}},
            "occlusion": {{"patch_size": 16, "stride": 8, "priority": 2}}
        }},
        "reasoning": "Why these tools for SPURIOUS detection"
    }}
}}

**Guidelines for SPURIOUS Feature Detection**:
- Spurious features are visually present but semantically irrelevant
- Common examples: watermarks, backgrounds, dataset artifacts
- For misclassifications: look for features of the WRONG predicted class
- For multiple images: find SHARED patterns across all failures
- Attribution tools help but may not distinguish spurious from relevant

{'- Focus on finding COMMON spurious features across all provided images' if is_multi_image else '- Focus on background and contextual elements'}

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Build prompt for Actor to generate spurious feature explanation for VISION

        Args:
            context: Dict containing user_question, model_info, prediction
            strategy: Dict from Proposer Agent
            results: Dict containing autonomous_results and/or tool_results
        """
        is_multi_image = isinstance(context.get('image_path'), list)

        prompt = f"""You are an AI explainability expert identifying SPURIOUS FEATURES in an image classification model.

**User Question**: {context['user_question']}

"""

        if is_multi_image:
            prompt += f"""**Multiple Images Analysis**:
- Analyzing {len(context['image_path'])} misclassified images
- Task: Identify SHARED spurious features
"""
        else:
            prompt += f"""**Model Prediction**:
- Predicted: {context['prediction'].get('predicted_class_name', 'Unknown')}
- Confidence: {context['prediction'].get('confidence', 0.0):.4f}
- Ground Truth: {context.get('ground_truth', 'Unknown')}
"""

        prompt += """
**Analysis Results Available**:
"""

        # Add autonomous results
        if results.get('autonomous_results'):
            prompt += "\n**Autonomous Spurious Detection**:\n"
            for task_name, task_result in results['autonomous_results'].items():
                result_data = task_result.get('result', {})
                prompt += f"""
- Task: {task_result.get('task_type', 'unknown')}
- Query: {task_result.get('query', 'N/A')}
- Answer: {result_data.get('answer', 'N/A')}
- Detected Features: {result_data.get('spurious_features', [])}
"""

        # Add tool results
        if results.get('tool_results'):
            prompt += "\n**XAI Tool Results**:\n"
            for tool_name, tool_info in results['tool_results'].items():
                prompt += f"""
- {tool_name.replace('_', ' ').title()}:
  - Important Regions: {tool_info.get('important_regions', [])}
  - Attribution Map: {tool_info.get('visualization_path', 'N/A')}
"""

        # Add VLM-extracted features if available
        if results.get('extracted_features'):
            ef = results['extracted_features']
            prompt += "\n**VLM Feature Extraction Results** (pre-analyzed from visualizations):\n"

            regions = ef.get('responsible_regions', [])
            if regions:
                prompt += f"- Identified Regions (potential spurious): {len(regions)} region(s)\n"
                for i, region in enumerate(regions[:3]):
                    label = region.get('label', f'Region {i+1}')
                    importance = region.get('importance', 0.0)
                    bbox = region.get('bbox', {})
                    prompt += f"  * {label}: importance={importance:.2f}, bbox={bbox}\n"

            imp_dist = ef.get('importance_distribution', {})
            if imp_dist:
                prompt += f"- Importance Distribution:\n"
                prompt += f"  * Primary: {imp_dist.get('primary_region_contribution', 0):.2f}\n"
                prompt += f"  * Background (potential spurious): {imp_dist.get('background_contribution', 0):.2f}\n"

            key_features = ef.get('key_visual_features', [])
            if key_features:
                prompt += f"- Key Visual Features (may include spurious): {', '.join(key_features[:5])}\n"

            vlm_explanation = ef.get('explanation', '')
            if vlm_explanation:
                prompt += f"- VLM Analysis: {vlm_explanation[:200]}...\n"

        prompt += """

**Your Task**: Generate a comprehensive analysis identifying SPURIOUS (irrelevant) features:
1. Identify specific VISUAL regions that are spurious
2. Explain WHY these features are spurious (not semantically relevant)
3. Assess their IMPACT on the (wrong) prediction
4. For multiple images: describe the SHARED spurious pattern
5. Provide evidence from both autonomous and tool-based analysis

**CRITICAL: Your response MUST be valid JSON** with this structure:
"""

        if is_multi_image:
            # Multi-image format
            prompt += """{
    "explanation": "Natural language explanation of the spurious features found (3-5 sentences)",
    "extracted_features": {
        "shared_feature_description": "Description of shared spurious feature across all images",
        "feature_locations": {
            "image_1": [{"bbox": {"x1": int, "y1": int, "x2": int, "y2": int}, "description": "..."}],
            "image_2": [{"bbox": {"x1": int, "y1": int, "x2": int, "y2": int}, "description": "..."}]
        },
        "confusion_mechanism": "How this shared feature confuses the model",
        "impact_assessment": float (0.0-1.0, overall impact of spurious features)
    },
    "confidence": float (0.0-1.0, confidence in spurious detection),
    "evidence_summary": {
        "autonomous_analysis": "Summary of autonomous detection",
        "tool_analysis": "Summary of attribution tool findings",
        "agreement": "Do different methods agree on spurious features?"
    }
}"""
        else:
            # Single-image format
            prompt += """{
    "explanation": "Natural language explanation of the spurious features found (3-5 sentences)",
    "extracted_features": {
        "spurious_regions": [
            {
                "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                "label": "What spurious feature is in this region",
                "spuriousness_score": float (0.0-1.0, how spurious it is),
                "reasoning": "Why this is spurious, not relevant to true class",
                "impact": float (0.0-1.0, estimated impact on wrong prediction)
            }
        ],
        "spurious_analysis": {
            "total_spurious_regions": int,
            "dominant_spurious_type": "background" | "texture" | "artifact" | "object",
            "semantic_mismatch": "Why these features don't match the true class"
        },
        "impact_assessment": float (0.0-1.0, overall impact of spurious features)
    },
    "confidence": float (0.0-1.0, confidence in spurious detection),
    "evidence_summary": {
        "autonomous_analysis": "Summary of autonomous detection",
        "tool_analysis": "Summary of attribution tool findings",
        "agreement": "Do different methods agree on spurious features?"
    }
}"""

        prompt += """

**Guidelines for Spurious Feature Identification (VISION-specific)**:
- Spurious features are visually salient but semantically irrelevant
- Common types: backgrounds, watermarks, dataset bias, spurious correlations
- Use bounding boxes to precisely locate spurious regions
- Distinguish between "not important" and "spurious" (spurious = misleading)
- For multiple images: focus on COMMON patterns, not individual image features
- Impact assessment should consider: would removing this feature fix the prediction?

**Examples of Spurious Features**:
- Sky background correlated with "airplane" in training data
- Grass texture correlated with "cow"
- Copyright watermarks in specific positions
- Lighting conditions specific to certain classes

Provide your spurious feature analysis as a JSON object:
"""
        return prompt


# ============================================================================
# TEXT MODALITY PROMPT BUILDERS (TODO)
# ============================================================================

class TextPromptBuilder(PromptBuilder):
    """Base prompt builder for Text modality"""
    pass


class TextFeatureAttributionPromptBuilder(TextPromptBuilder):
    """
    Prompt builder for Feature Attribution in Text modality

    Supports different attribution types:
    - "most": Find most responsible words/phrases
    - "least": Find least responsible words/phrases
    - "distinctive": Find words/phrases that distinguish from alternatives
    - "contrastive": Compare different instances
    """

    def __init__(self, attribution_type: str = "most"):
        """
        Initialize Text Feature Attribution Prompt Builder

        Args:
            attribution_type: Type of attribution ("most", "least", "distinctive", "contrastive")
        """
        super().__init__()
        self.attribution_type = attribution_type

        # Define attribution-specific terminology for TEXT
        self._attribution_config = {
            "most": {
                "target": "most responsible",
                "adjective": "important",
                "verb": "contributed to",
                "noun": "contribution",
                "extraction_focus": "highly influential words/phrases",
                "tool_priority": "high importance",
                "metric": "MoRF"
            },
            "least": {
                "target": "least responsible",
                "adjective": "irrelevant",
                "verb": "had minimal impact on",
                "noun": "irrelevance",
                "extraction_focus": "words/phrases with minimal impact",
                "tool_priority": "low importance",
                "metric": "LeRF"
            },
            "distinctive": {
                "target": "most distinctive",
                "adjective": "distinguishing",
                "verb": "distinguished from alternatives",
                "noun": "distinctiveness",
                "extraction_focus": "discriminative words/phrases",
                "tool_priority": "contrastive importance",
                "metric": "Acc/MoRF"
            },
            "contrastive": {
                "target": "differentiating",
                "adjective": "contrasting",
                "verb": "differentiated between",
                "noun": "difference",
                "extraction_focus": "instance-specific words/phrases",
                "tool_priority": "comparative importance",
                "metric": "MoRF"
            }
        }

        if attribution_type not in self._attribution_config:
            raise ValueError(f"Invalid attribution_type: {attribution_type}. "
                           f"Must be one of {list(self._attribution_config.keys())}")

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build prompt for Proposer to decide strategy for Text Feature Attribution

        Args:
            context: Dict containing:
                - user_question: str
                - text_input: str
                - model_info: Dict
                - prediction: Dict
                - text_length: int (optional)
        """
        config = self._attribution_config[self.attribution_type]

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a TEXT classification model's prediction.

**User Question**: {context['user_question']}

**Model Information**:
- Model: {context['model_info'].get('model_name', 'Unknown')}
- Architecture: {context['model_info'].get('architecture', 'Unknown')}
- Prediction: Class {context['prediction'].get('predicted_class_idx')}
  (Confidence: {context['prediction'].get('confidence', 0.0):.4f})
- Top-5 Predictions: {context['prediction'].get('top5_predictions', [])}

**Text Input** (first 200 chars):
{context.get('text_input', '')[:200]}...

**Task**: Design a comprehensive strategy to identify which WORDS and PHRASES in the text were {config['target']} for this prediction.

**Available Methods**:
1. **Autonomous Language Analysis**: Use your language understanding to:
   - Identify semantically {config['adjective']} words/phrases
   - Reason about their {config['noun']} to the predicted class
   - Analyze sentiment, entities, and key concepts
   - Highlight words that {config['verb']} the prediction

2. **External XAI Tools**: Use text attribution methods:
   - LIME: Tests {config['noun']} by masking word combinations
   - SHAP: Game-theoretic word {config['noun']}
   - Integrated Gradients: Gradient-based token attribution
   - Attention Visualization: For transformer models
   - Layer-wise Relevance Propagation: For deep models

**Your Response Must Be Valid JSON** with this structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Why you chose this strategy for finding {config['extraction_focus']} (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "grounding" | "reasoning",
            "query": "Specific query for autonomous TEXT analysis to find {config['extraction_focus']}",
            "expected_output": "What should be extracted (word indices, phrases, etc.)"
        }}
    ],
    "tool_selection": {{
        "selected_tools": ["lime", "shap", "integrated_gradients", "attention", ...],
        "tool_params": {{
            "lime": {{"num_samples": 1000, "priority": 1}},
            "shap": {{"nsamples": 100, "priority": 2}}
        }},
        "reasoning": "Why these TEXT-specific tools for finding {config['extraction_focus']}"
    }}
}}

**Guidelines for TEXT attribution ({self.attribution_type})**:
- Focus on WORD and PHRASE level {config['noun']}, not character-level
- Target: Find words/phrases that {config['verb']} the prediction
- Consider both individual words and multi-word phrases
- For simple sentiment/topic classification, autonomous analysis may suffice
- For complex models (BERT, GPT), use gradient-based tools
- Attention scores (if available) provide model's internal focus
- Evaluation metric: {config['metric']}

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Build prompt for Actor to generate explanation for TEXT attribution

        Args:
            context: Dict containing user_question, text_input, model_info, prediction
            strategy: Dict from Proposer Agent
            results: Dict containing autonomous_results and/or tool_results
        """
        prompt = f"""You are an AI explainability expert providing a comprehensive explanation for a TEXT classification model.

**User Question**: {context['user_question']}

**Model Prediction**:
- Predicted Class: {context['prediction'].get('predicted_class_idx')}
- Class Name: {context['prediction'].get('predicted_class_name', 'Unknown')}
- Confidence: {context['prediction'].get('confidence', 0.0):.4f}

**Text Input** (first 200 chars):
{context.get('text_input', '')[:200]}...

**Analysis Results Available**:
"""

        # Add autonomous results
        if results.get('autonomous_results'):
            prompt += "\n**Autonomous Language Analysis**:\n"
            for task_name, task_result in results['autonomous_results'].items():
                result_data = task_result.get('result', {})
                prompt += f"""
- Task: {task_result.get('task_type', 'unknown')}
- Query: {task_result.get('query', 'N/A')}
- Answer: {result_data.get('answer', 'N/A')}
- Identified Words/Phrases: {result_data.get('important_spans', [])}
"""

        # Add tool results
        if results.get('tool_results'):
            prompt += "\n**XAI Tool Results**:\n"
            for tool_name, tool_info in results['tool_results'].items():
                prompt += f"""
- {tool_name.replace('_', ' ').title()}:
  - Result: {tool_info.get('result', 'N/A')}
  - Top Words: {tool_info.get('top_words', [])}
  - Statistics: {tool_info.get('statistics', {})}
"""

        # Add VLM-extracted features if available
        if results.get('extracted_features'):
            ef = results['extracted_features']
            prompt += "\n**VLM Feature Extraction Results**:\n"

            important_words = ef.get('important_words', ef.get('responsible_regions', []))
            if important_words:
                prompt += f"- Identified Words/Phrases: {len(important_words)} element(s)\n"
                for i, word in enumerate(important_words[:5]):
                    text = word.get('text', word.get('label', f'Element {i+1}'))
                    importance = word.get('importance_score', word.get('importance', 0.0))
                    prompt += f"  * '{text}': importance={importance:.2f}\n"

            key_features = ef.get('key_visual_features', ef.get('key_text_features', []))
            if key_features:
                prompt += f"- Key Features: {', '.join(str(f) for f in key_features[:5])}\n"

            vlm_explanation = ef.get('explanation', '')
            if vlm_explanation:
                prompt += f"- VLM Analysis: {vlm_explanation[:200]}...\n"

        prompt += f"""

**Your Task**: Generate a comprehensive explanation for this TEXT classification that:
1. Identifies specific WORDS and PHRASES most responsible
2. Provides character-level positions (start_idx, end_idx) for each
3. Assigns importance scores (0.0-1.0) to each word/phrase
4. Explains WHY these words/phrases led to this prediction
5. Synthesizes findings from all analysis methods

**CRITICAL: Your response MUST be valid JSON** with this structure:
{{
    "explanation": "Natural language explanation (3-5 sentences) answering which WORDS/PHRASES were most responsible",
    "extracted_features": {{
        "important_words": [
            {{
                "text": "the actual word",
                "start_idx": int (character position in original text),
                "end_idx": int (character position),
                "importance_score": float (0.0-1.0),
                "reasoning": "Why this word is important for the prediction"
            }}
        ],
        "important_phrases": [
            {{
                "text": "multi-word phrase",
                "start_idx": int,
                "end_idx": int,
                "importance_score": float (0.0-1.0),
                "reasoning": "Why this phrase is important"
            }}
        ],
        "importance_distribution": {{
            "top_3_contribution": float (0.0-1.0, % of importance from top 3 words),
            "rare_words_impact": float (0.0-1.0, impact of uncommon words),
            "context_dependency": "high|medium|low - how much context matters"
        }}
    }},
    "confidence": float (0.0-1.0, confidence in this explanation),
    "evidence_summary": {{
        "autonomous_analysis": "Summary of language understanding findings",
        "tool_analysis": "Summary of XAI tool findings",
        "agreement_level": "high|medium|low - do different methods agree?"
    }}
}}

**Guidelines for TEXT Attribution (Text-specific)**:
- Use character-level indices (start_idx, end_idx) for precise location
- Words can overlap with phrases (e.g., "not good" as phrase, "good" as word)
- Importance scores should be relative to the prediction, not absolute
- Consider negations, modifiers, and context
- For transformer models, consider subword tokens vs whole words
- Order by importance (highest first)

**Examples**:
- Single word: {{"text": "excellent", "start_idx": 45, "end_idx": 54, "importance_score": 0.9}}
- Phrase: {{"text": "not bad", "start_idx": 10, "end_idx": 17, "importance_score": 0.7}}
- Context word: {{"text": "however", "start_idx": 30, "end_idx": 37, "importance_score": 0.6}}

Provide your explanation as a JSON object:
"""
        return prompt


class TextCounterfactualPromptBuilder(TextPromptBuilder):
    """Prompt builder for Text counterfactual questions"""

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build proposer prompt for TEXT counterfactual analysis"""
        target_info = context.get('target_class', 'a different prediction')

        prompt = f"""You are an AI explainability expert designing a strategy for COUNTERFACTUAL ANALYSIS on a TEXT classification model.

**User Question**: {context['user_question']}

**Model Information**:
- Model: {context['model_info'].get('model_name', 'Unknown')}
- Current Prediction: {context['prediction'].get('predicted_class_name', 'Unknown')}
  (Confidence: {context['prediction'].get('confidence', 0.0):.4f})
- Target: {target_info}

**Text Input** (first 200 chars):
{context.get('text_input', '')[:200]}...

**Task**: Design a strategy to identify how the TEXT should be MODIFIED to change the prediction to the target class.

**Available Methods**:
1. **Autonomous Language Analysis**:
   - Identify words/phrases characteristic of target class
   - Suggest specific word replacements, additions, or deletions
   - Reason about minimal changes needed

2. **External Tools**:
   - LIME/SHAP: Identify current important words
   - Text generation: Suggest alternative phrasings
   - Semantic similarity: Find minimal semantic changes

**Your Response Must Be Valid JSON**:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Why this strategy for TEXT counterfactual (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [...],
    "tool_selection": {{...}}
}}

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(self, context: Dict[str, Any], strategy: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Build actor prompt for TEXT counterfactual explanation"""
        target_info = context.get('target_class', 'a different prediction')

        prompt = f"""Generate a COUNTERFACTUAL explanation for TEXT classification.

**Current Prediction**: {context['prediction'].get('predicted_class_name', 'Unknown')}
**Target**: {target_info}

**Your response MUST be valid JSON**:
{{
    "explanation": "How to modify the TEXT (3-5 sentences)",
    "extracted_features": {{
        "text_modifications": [
            {{
                "action": "replace" | "insert" | "delete",
                "start_idx": int,
                "end_idx": int,
                "original_text": "...",
                "modified_text": "...",
                "reasoning": "Why this change helps"
            }}
        ],
        "target_prediction": "{target_info}",
        "modification_difficulty": float (0.0-1.0)
    }},
    "confidence": float
}}

Provide your explanation as a JSON object:
"""
        return prompt


class TextSpuriousFeaturesPromptBuilder(TextPromptBuilder):
    """Prompt builder for Text spurious features questions"""

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build proposer prompt for TEXT spurious feature detection"""
        prompt = f"""Design a strategy to detect SPURIOUS words/phrases in TEXT that influenced the (wrong) prediction.

**User Question**: {context['user_question']}

**Model Information**:
- Prediction: {context['prediction'].get('predicted_class_name', 'Unknown')}
- Ground Truth: {context.get('ground_truth', 'Unknown')}

**Text Input** (first 200 chars):
{context.get('text_input', '')[:200]}...

**Available Methods**:
1. **Autonomous Analysis**: Identify semantically irrelevant words
2. **Attribution Tools**: Find what model focuses on vs what's semantically relevant
3. **Comparison**: Compare model attribution with semantic importance

**Your Response Must Be Valid JSON**:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Strategy for TEXT spurious detection",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [...],
    "tool_selection": {{...}}
}}

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(self, context: Dict[str, Any], strategy: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Build actor prompt for TEXT spurious explanation"""
        prompt = f"""Identify SPURIOUS words/phrases in TEXT classification.

**Prediction**: {context['prediction'].get('predicted_class_name', 'Unknown')}
**Ground Truth**: {context.get('ground_truth', 'Unknown')}

**Your response MUST be valid JSON**:
{{
    "explanation": "Description of spurious features (3-5 sentences)",
    "extracted_features": {{
        "spurious_words": [
            {{
                "text": "word",
                "start_idx": int,
                "end_idx": int,
                "spuriousness_score": float (0.0-1.0),
                "reasoning": "Why this is spurious"
            }}
        ],
        "spurious_analysis": "Overall analysis",
        "impact_assessment": float (0.0-1.0)
    }},
    "confidence": float
}}

Provide your analysis as a JSON object:
"""
        return prompt


# ============================================================================
# TABULAR MODALITY PROMPT BUILDERS (TODO)
# ============================================================================

class TabularPromptBuilder(PromptBuilder):
    """Base prompt builder for Tabular modality"""
    pass


class TabularFeatureAttributionPromptBuilder(TabularPromptBuilder):
    """
    Prompt builder for Feature Attribution in Tabular modality

    Supports different attribution types:
    - "most": Find most responsible features
    - "least": Find least responsible features
    - "distinctive": Find features that distinguish from alternatives
    - "contrastive": Compare different instances
    """

    def __init__(self, attribution_type: str = "most"):
        """
        Initialize Tabular Feature Attribution Prompt Builder

        Args:
            attribution_type: Type of attribution ("most", "least", "distinctive", "contrastive")
        """
        super().__init__()
        self.attribution_type = attribution_type

        # Define attribution-specific terminology for TABULAR
        self._attribution_config = {
            "most": {
                "target": "most responsible",
                "adjective": "important",
                "verb": "contributed to",
                "noun": "contribution",
                "extraction_focus": "highly influential features",
                "tool_priority": "high importance",
                "metric": "MoRF"
            },
            "least": {
                "target": "least responsible",
                "adjective": "irrelevant",
                "verb": "had minimal impact on",
                "noun": "irrelevance",
                "extraction_focus": "features with minimal impact",
                "tool_priority": "low importance",
                "metric": "LeRF"
            },
            "distinctive": {
                "target": "most distinctive",
                "adjective": "distinguishing",
                "verb": "distinguished from alternatives",
                "noun": "distinctiveness",
                "extraction_focus": "discriminative features",
                "tool_priority": "contrastive importance",
                "metric": "Acc/MoRF"
            },
            "contrastive": {
                "target": "differentiating",
                "adjective": "contrasting",
                "verb": "differentiated between",
                "noun": "difference",
                "extraction_focus": "instance-specific features",
                "tool_priority": "comparative importance",
                "metric": "MoRF"
            }
        }

        if attribution_type not in self._attribution_config:
            raise ValueError(f"Invalid attribution_type: {attribution_type}. "
                           f"Must be one of {list(self._attribution_config.keys())}")

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build prompt for Proposer to decide strategy for Tabular Feature Attribution

        Args:
            context: Dict containing:
                - user_question: str
                - input_data: Dict[str, Any] (feature values)
                - model_info: Dict
                - prediction: Dict
                - feature_names: List[str]
                - feature_types: Dict[str, str] (numerical/categorical)
        """
        config = self._attribution_config[self.attribution_type]

        feature_info = ""
        if 'input_data' in context and 'feature_names' in context:
            feature_info = "\n".join([
                f"- {name}: {context['input_data'].get(name, 'N/A')}"
                for name in context['feature_names'][:10]  # Show first 10
            ])

        prompt = f"""You are an AI explainability expert designing a strategy to answer the following question about a TABULAR data model's prediction.

**User Question**: {context['user_question']}

**Model Information**:
- Model: {context['model_info'].get('model_name', 'Unknown')}
- Model Type: {context['model_info'].get('model_type', 'Unknown')}
- Prediction: {context['prediction'].get('predicted_class_name', context['prediction'].get('predicted_value', 'Unknown'))}
  (Confidence/Score: {context['prediction'].get('confidence', context['prediction'].get('score', 0.0)):.4f})

**Input Features** (first 10):
{feature_info}

**Total Features**: {len(context.get('feature_names', []))}
**Feature Types**: {context.get('feature_types', {})}

**Task**: Design a comprehensive strategy to identify which FEATURES (columns) were most responsible for this prediction.

**Available Methods**:
1. **Autonomous Tabular Analysis**: Use your reasoning to:
   - Identify obviously important features (e.g., extreme values)
   - Reason about domain relevance of features
   - Detect potential feature interactions
   - Consider typical importance for this prediction task

2. **External XAI Tools**: Use tabular attribution methods:
   - SHAP: Game-theoretic feature importance (best for tabular)
   - LIME: Local linear approximation
   - Permutation Importance: Feature shuffle impact
   - Partial Dependence: Feature value effect
   - Feature Interaction Detection

**Your Response Must Be Valid JSON** with this structure:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Why you chose this strategy for TABULAR attribution (2-3 sentences)",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [
        {{
            "task_type": "reasoning" | "grounding",
            "query": "Specific query for autonomous TABULAR analysis",
            "expected_output": "What should be extracted (feature names, values, etc.)"
        }}
    ],
    "tool_selection": {{
        "selected_tools": ["shap", "lime", "permutation", ...],
        "tool_params": {{
            "shap": {{"nsamples": 100, "priority": 1}},
            "lime": {{"num_samples": 1000, "priority": 2}}
        }},
        "reasoning": "Why these TABULAR-specific tools were selected"
    }}
}}

**Guidelines for TABULAR attribution**:
- Focus on COLUMN-level importance (not row-level)
- Consider both individual features and interactions
- For tree-based models, SHAP is highly effective
- For simple linear models, autonomous analysis may suffice
- Extreme values often indicate high importance
- Hybrid: combine domain reasoning with model-based attribution

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(
        self,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Build prompt for Actor to generate explanation for TABULAR attribution

        Args:
            context: Dict containing user_question, input_data, model_info, prediction
            strategy: Dict from Proposer Agent
            results: Dict containing autonomous_results and/or tool_results
        """
        prompt = f"""You are an AI explainability expert providing a comprehensive explanation for a TABULAR data model.

**User Question**: {context['user_question']}

**Model Prediction**:
- Prediction: {context['prediction'].get('predicted_class_name', context['prediction'].get('predicted_value', 'Unknown'))}
- Confidence/Score: {context['prediction'].get('confidence', context['prediction'].get('score', 0.0)):.4f}

**Analysis Results Available**:
"""

        # Add autonomous results
        if results.get('autonomous_results'):
            prompt += "\n**Autonomous Tabular Analysis**:\n"
            for task_name, task_result in results['autonomous_results'].items():
                result_data = task_result.get('result', {})
                prompt += f"""
- Task: {task_result.get('task_type', 'unknown')}
- Query: {task_result.get('query', 'N/A')}
- Answer: {result_data.get('answer', 'N/A')}
- Identified Features: {result_data.get('important_features', [])}
"""

        # Add tool results
        if results.get('tool_results'):
            prompt += "\n**XAI Tool Results**:\n"
            for tool_name, tool_info in results['tool_results'].items():
                prompt += f"""
- {tool_name.replace('_', ' ').title()}:
  - Feature Importances: {tool_info.get('feature_importances', {})}
  - Statistics: {tool_info.get('statistics', {})}
"""

        # Add VLM-extracted features if available
        if results.get('extracted_features'):
            ef = results['extracted_features']
            prompt += "\n**VLM Feature Extraction Results**:\n"

            important_features = ef.get('important_features', ef.get('responsible_regions', {}))
            if important_features:
                if isinstance(important_features, dict):
                    prompt += f"- Identified Features: {len(important_features)} feature(s)\n"
                    for feat_name, feat_info in list(important_features.items())[:5]:
                        if isinstance(feat_info, dict):
                            importance = feat_info.get('importance_score', feat_info.get('importance', 0.0))
                            prompt += f"  * {feat_name}: importance={importance:.2f}\n"
                elif isinstance(important_features, list):
                    prompt += f"- Identified Features: {len(important_features)} element(s)\n"
                    for i, feat in enumerate(important_features[:5]):
                        label = feat.get('label', f'Feature {i+1}')
                        importance = feat.get('importance', 0.0)
                        prompt += f"  * {label}: importance={importance:.2f}\n"

            vlm_explanation = ef.get('explanation', '')
            if vlm_explanation:
                prompt += f"- VLM Analysis: {vlm_explanation[:200]}...\n"

        prompt += f"""

**Your Task**: Generate a comprehensive explanation for this TABULAR prediction that:
1. Identifies specific FEATURES (columns) most responsible
2. Reports their actual VALUES in this instance
3. Assigns importance scores (0.0-1.0) to each feature
4. Identifies any important FEATURE INTERACTIONS
5. Explains WHY these features/values led to this prediction
6. Synthesizes findings from all analysis methods

**CRITICAL: Your response MUST be valid JSON** with this structure:
{{
    "explanation": "Natural language explanation (3-5 sentences) answering which FEATURES were most responsible",
    "extracted_features": {{
        "important_features": {{
            "feature_name_1": {{
                "value": actual_value,
                "importance_score": float (0.0-1.0),
                "reasoning": "Why this feature/value is important"
            }},
            "feature_name_2": {{...}}
        }},
        "critical_values": {{
            "feature_name_1": {{
                "actual_value": value,
                "typical_range": [min, max],
                "is_extreme": true|false,
                "impact_direction": "increases|decreases prediction"
            }}
        }},
        "feature_interactions": [
            {{
                "features": ["feature1", "feature2"],
                "interaction_strength": float (0.0-1.0),
                "description": "How these features interact"
            }}
        ]
    }},
    "confidence": float (0.0-1.0, confidence in this explanation),
    "evidence_summary": {{
        "autonomous_analysis": "Summary of reasoning-based findings",
        "tool_analysis": "Summary of XAI tool findings",
        "agreement_level": "high|medium|low - do different methods agree?"
    }}
}}

**Guidelines for TABULAR Attribution (Tabular-specific)**:
- Feature names should match exactly with input column names
- Report ACTUAL values, not just importance scores
- For numerical features: include value range context
- For categorical features: mention the specific category value
- Feature interactions are important for non-linear models
- Order by importance (highest first)
- Consider domain knowledge (e.g., Age=5 for loan approval is unusual)

**Examples**:
- Numerical: {{"Credit_Score": {{"value": 780, "importance_score": 0.9, "reasoning": "High credit score strongly indicates approval"}}}}
- Categorical: {{"Loan_Purpose": {{"value": "education", "importance_score": 0.7, "reasoning": "Education loans have lower default rates"}}}}
- Interaction: {{"features": ["Age", "Income"], "interaction_strength": 0.6, "description": "Young age with high income indicates high approval"}}

Provide your explanation as a JSON object:
"""
        return prompt


class TabularCounterfactualPromptBuilder(TabularPromptBuilder):
    """Prompt builder for Tabular counterfactual questions"""

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build proposer prompt for TABULAR counterfactual analysis"""
        target_info = context.get('target_class', 'a different prediction')

        prompt = f"""Design a strategy for COUNTERFACTUAL ANALYSIS on a TABULAR data model.

**User Question**: {context['user_question']}

**Current Prediction**: {context['prediction'].get('predicted_class_name', 'Unknown')}
**Target**: {target_info}

**Available Methods**:
1. **Autonomous Analysis**: Reason about which feature changes would flip prediction
2. **Counterfactual Explainers**: DiCE, Counterfactual Explanations, Actionable Recourse
3. **Optimization**: Find minimal changes to flip prediction

**Your Response Must Be Valid JSON**:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Strategy for TABULAR counterfactual",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [...],
    "tool_selection": {{...}}
}}

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(self, context: Dict[str, Any], strategy: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Build actor prompt for TABULAR counterfactual explanation"""
        target_info = context.get('target_class', 'a different prediction')

        prompt = f"""Generate a COUNTERFACTUAL explanation for TABULAR data.

**Current Prediction**: {context['prediction'].get('predicted_class_name', 'Unknown')}
**Target**: {target_info}

**Your response MUST be valid JSON**:
{{
    "explanation": "How to change FEATURE VALUES (3-5 sentences)",
    "extracted_features": {{
        "feature_changes": {{
            "feature_name": {{
                "current_value": value,
                "suggested_value": value,
                "change_magnitude": float,
                "feasibility": "easy|moderate|difficult",
                "reasoning": "Why this change helps"
            }}
        }},
        "target_prediction": "{target_info}",
        "modification_difficulty": float (0.0-1.0),
        "feasibility": float (0.0-1.0)
    }},
    "confidence": float
}}

Provide your explanation as a JSON object:
"""
        return prompt


class TabularSpuriousFeaturesPromptBuilder(TabularPromptBuilder):
    """Prompt builder for Tabular spurious features questions"""

    def build_proposer_prompt(self, context: Dict[str, Any]) -> str:
        """Build proposer prompt for TABULAR spurious feature detection"""
        prompt = f"""Design a strategy to detect SPURIOUS features in TABULAR data that influenced the (wrong) prediction.

**User Question**: {context['user_question']}

**Prediction**: {context['prediction'].get('predicted_class_name', 'Unknown')}
**Ground Truth**: {context.get('ground_truth', 'Unknown')}

**Available Methods**:
1. **Autonomous Analysis**: Reason about domain relevance of features
2. **Attribution Tools**: Compare model focus vs domain knowledge
3. **Statistical Analysis**: Detect spurious correlations

**Your Response Must Be Valid JSON**:
{{
    "strategy_type": "autonomous" | "tools" | "hybrid",
    "reasoning": "Strategy for TABULAR spurious detection",
    "confidence": 0.0-1.0,
    "autonomous_tasks": [...],
    "tool_selection": {{...}}
}}

Provide your strategy as a JSON object:
"""
        return prompt

    def build_actor_prompt(self, context: Dict[str, Any], strategy: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Build actor prompt for TABULAR spurious explanation"""
        prompt = f"""Identify SPURIOUS features in TABULAR data classification.

**Prediction**: {context['prediction'].get('predicted_class_name', 'Unknown')}
**Ground Truth**: {context.get('ground_truth', 'Unknown')}

**Your response MUST be valid JSON**:
{{
    "explanation": "Description of spurious features (3-5 sentences)",
    "extracted_features": {{
        "spurious_features": [
            {{
                "feature_name": "...",
                "feature_value": value,
                "spuriousness_score": float (0.0-1.0),
                "reasoning": "Why this feature is spurious (not domain-relevant)"
            }}
        ],
        "spurious_analysis": {{
            "domain_mismatch": "Why these features shouldn't matter",
            "correlation_type": "spurious|confounding|irrelevant"
        }},
        "impact_assessment": float (0.0-1.0)
    }},
    "confidence": float
}}

Provide your analysis as a JSON object:
"""
        return prompt


# ============================================================================
# QUESTION TEMPLATE DEFINITION
# ============================================================================

@dataclass
class QuestionTemplate:
    """Template for a specific XAI question"""
    template_id: str  # Unique identifier for the template
    category: QuestionCategory
    modality: Modality
    template: str
    extraction_fields: List[ExtractionField]
    prompt_builder: Optional[PromptBuilder] = None
    examples: List[str] = field(default_factory=list)
    attribution_type: Optional[str] = None  # For feature attribution: "most", "least", "distinctive", "contrastive"

    def get_extraction_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for extraction"""
        schema = {}
        for field in self.extraction_fields:
            schema[field.name] = {
                "type": field.field_type,
                "description": field.description,
                "required": field.required
            }
        return schema


# ============================================================================
# PROMPT BUILDER MAPPING
# ============================================================================

# Mapping of template types to their prompt builder classes
PROMPT_BUILDER_MAP = {
    # Vision
    "vision_feature_attribution": VisionFeatureAttributionPromptBuilder,
    "vision_counterfactual": VisionCounterfactualPromptBuilder,
    "vision_spurious_features": VisionSpuriousFeaturesPromptBuilder,

    # Text
    "text_feature_attribution": TextFeatureAttributionPromptBuilder,
    "text_counterfactual": TextCounterfactualPromptBuilder,
    "text_spurious_features": TextSpuriousFeaturesPromptBuilder,

    # Tabular
    "tabular_feature_attribution": TabularFeatureAttributionPromptBuilder,
    "tabular_counterfactual": TabularCounterfactualPromptBuilder,
    "tabular_spurious_features": TabularSpuriousFeaturesPromptBuilder,
}


def _get_prompt_builder_class(modality: str, category: str) -> type:
    """
    Get the prompt builder class for a given modality and category

    Args:
        modality: "vision", "text", or "tabular"
        category: "feature_attribution", "counterfactual", or "spurious_features"

    Returns:
        Prompt builder class
    """
    key = f"{modality}_{category}"
    return PROMPT_BUILDER_MAP.get(key)


def _load_templates_from_config(config_path: str = None) -> Dict[str, QuestionTemplate]:
    """
    Load question templates from configuration file

    Args:
        config_path: Path to templates configuration JSON file

    Returns:
        Dict of template_id (q_type) -> QuestionTemplate
    """
    if config_path is None:
        # Look for templates_config.json in current working directory
        import os
        config_path = os.path.join(os.getcwd(), "templates_config.json")

        # If not found, try relative to this file (fallback)
        if not os.path.exists(config_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "templates_config.json")

    if not os.path.exists(config_path):
        print(f"Warning: Template config file not found at {config_path}")
        return {}

    with open(config_path, 'r') as f:
        config = json.load(f)

    templates = {}

    for modality, categories in config['templates'].items():
        for category, template_list in categories.items():
            for template_data in template_list:
                q_type = template_data['q_type']

                # Get prompt builder class
                builder_class = _get_prompt_builder_class(modality, category)
                if builder_class is None:
                    print(f"Warning: No prompt builder found for {modality}_{category}")
                    continue

                # Create extraction fields
                extraction_fields = [
                    ExtractionField(
                        name=field['name'],
                        field_type=field['field_type'],
                        description=field['description'],
                        required=field.get('required', True)
                    )
                    for field in template_data['extraction_fields']
                ]

                # Create prompt builder instance with attribution_type if applicable
                attribution_type = template_data.get('attribution_type')
                if attribution_type and category == 'feature_attribution':
                    # For feature attribution, pass attribution_type to builder
                    prompt_builder = builder_class(attribution_type=attribution_type)
                else:
                    # For other categories, use default initialization
                    prompt_builder = builder_class()

                # Create template
                templates[q_type] = QuestionTemplate(
                    template_id=q_type,
                    category=QuestionCategory(category),
                    modality=Modality(modality),
                    template=template_data['template'],
                    extraction_fields=extraction_fields,
                    prompt_builder=prompt_builder,
                    examples=[],  # Examples are not stored in templates
                    attribution_type=attribution_type  # Store attribution_type in template
                )

    return templates


# ============================================================================
# TEMPLATE REGISTRY AND UTILITIES
# ============================================================================

# Load templates from configuration file
try:
    ALL_TEMPLATES = _load_templates_from_config()
    print(f"✓ Loaded {len(ALL_TEMPLATES)} templates from configuration")
except Exception as e:
    print(f"Warning: Could not load templates from config: {e}")
    print("Using empty template registry. Make sure templates_config.json exists.")
    ALL_TEMPLATES = {}


# Mapping from integer q_type to template information
# Based on questions_dataset_category.md
Q_TYPE_TO_TEMPLATE = {
    1: {
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "modality_prefix": "vision",  # Default, will be inferred from data
        "attribution_type": AttributionType.MOST,
        "template": "Which part of the input was most responsible for the model's prediction?"
    },
    2: {
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "modality_prefix": "vision",
        "attribution_type": AttributionType.LEAST,
        "template": "Which part of the input was least responsible for the model's prediction?"
    },
    3: {
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "modality_prefix": "vision",
        "attribution_type": AttributionType.MOST,
        "template": "Which specific parts of the input distinguish its prediction from the next-best alternative?"
    },
    4: {
        "category": QuestionCategory.FEATURE_ATTRIBUTION,
        "modality_prefix": "vision",
        "attribution_type": AttributionType.DISTINCTIVE,
        "template": "Why are instances A and B given different predictions?"
    },
    5: {
        "category": QuestionCategory.COUNTERFACTUAL,
        "modality_prefix": "vision",
        "attribution_type": None,
        "template": "If we mask the certain part of this input, would the prediction change?"
    },
    6: {
        "category": QuestionCategory.COUNTERFACTUAL,
        "modality_prefix": "vision",
        "attribution_type": None,
        "template": "How should the instance change to flip the model into a different prediction?"
    },
    7: {
        "category": QuestionCategory.COUNTERFACTUAL,
        "modality_prefix": "vision",
        "attribution_type": AttributionType.MOST,
        "template": "If we remove/change one important part of the instance, how would the prediction change?"
    },
    8: {
        "category": QuestionCategory.SPURIOUS_FEATURES,
        "modality_prefix": "vision",
        "attribution_type": None,
        "template": "Is there any irrelevant part in this input that causes the model's wrong prediction?"
    },
    9: {
        "category": QuestionCategory.SPURIOUS_FEATURES,
        "modality_prefix": "vision",
        "attribution_type": None,
        "template": "What shared feature makes the misclassified input A and B and C… difficult for the model?"
    },
    10: {
        "category": QuestionCategory.SPURIOUS_FEATURES,
        "modality_prefix": "vision",
        "attribution_type": None,
        "template": "Why are instances A and B given different predictions?"
    }
}


class TemplateRegistry:
    """Registry for managing question templates"""

    @staticmethod
    def get_template(template_id: str) -> Optional[QuestionTemplate]:
        """Get template by ID"""
        return ALL_TEMPLATES.get(template_id)

    @staticmethod
    def get_template_by_qtype(q_type: int, modality: Optional[Modality] = None) -> Optional[QuestionTemplate]:
        """
        Get template by integer q_type (1-10)

        Args:
            q_type: Integer question type (1-10)
            modality: Modality to use (if None, will be inferred)

        Returns:
            QuestionTemplate or None
        """
        if q_type not in Q_TYPE_TO_TEMPLATE:
            return None

        template_info = Q_TYPE_TO_TEMPLATE[q_type]

        # Determine modality - default to vision if not provided
        if modality is None:
            modality = Modality.VISION

        # Create a template on the fly
        # Try to find a matching template in ALL_TEMPLATES first
        template_key = f"{modality.value}_{template_info['category'].value}"
        if template_key in ALL_TEMPLATES:
            return ALL_TEMPLATES[template_key]

        # Otherwise, create a simple template
        attribution_type = template_info.get('attribution_type')
        # Convert AttributionType enum to string value
        if attribution_type is not None and isinstance(attribution_type, AttributionType):
            attribution_type = attribution_type.value

        return QuestionTemplate(
            template_id=f"q_type_{q_type}",
            modality=modality,
            category=template_info['category'],
            template=template_info['template'],
            extraction_fields=[],
            prompt_builder=None,
            examples=[],
            attribution_type=attribution_type
        )
    
    @staticmethod
    def match_question(question: str, modality: Modality = Modality.VISION) -> Optional[QuestionTemplate]:
        """
        Match user question to a template
        
        For now, uses simple substring matching.
        TODO: Implement more sophisticated matching (e.g., semantic similarity)
        """
        question_lower = question.lower()
        
        # Filter templates by modality
        modality_templates = {
            k: v for k, v in ALL_TEMPLATES.items() 
            if v.modality == modality
        }
        
        # Try to find best match
        best_match = None
        best_score = 0
        
        for template_id, template in modality_templates.items():
            # Check if template keywords are in question
            template_keywords = template.template.lower().split()
            matches = sum(1 for keyword in template_keywords if keyword in question_lower)
            score = matches / len(template_keywords)
            
            if score > best_score:
                best_score = score
                best_match = template
        
        return best_match if best_score > 0.3 else None
    
    @staticmethod
    def list_templates(
        category: Optional[QuestionCategory] = None,
        modality: Optional[Modality] = None
    ) -> Dict[str, QuestionTemplate]:
        """List templates filtered by category and/or modality"""
        templates = ALL_TEMPLATES
        
        if category:
            templates = {k: v for k, v in templates.items() if v.category == category}
        if modality:
            templates = {k: v for k, v in templates.items() if v.modality == modality}
        
        return templates


# ============================================================================
# QUESTION DATASET INTERFACE (TODO)
# ============================================================================

class QuestionDataset:
    """
    Interface for loading questions from a dataset file

    Dataset format can be JSON, JSONL, or CSV with the following fields:
    - question_id: Unique identifier for the question
    - q_type: Template type (e.g., "vision_feature_attribution_1")
    - question: The actual question text
    - modality: "vision", "text", or "tabular"
    - data_path: Path to the input data (image, text file, csv)
    - ground_truth: Optional ground truth label
    - metadata: Optional additional metadata

    Example JSON format:
    {
        "questions": [
            {
                "question_id": "q001",
                "q_type": "vision_feature_attribution_1",
                "question": "Which part of the image was most responsible?",
                "modality": "vision",
                "data_path": "/path/to/image.jpg",
                "ground_truth": "cat",
                "metadata": {"dataset": "imagenet"}
            },
            ...
        ]
    }
    """

    def __init__(self, dataset_path: str):
        """
        Initialize QuestionDataset

        Args:
            dataset_path: Path to dataset file (JSON, JSONL, or CSV)
        """
        self.dataset_path = dataset_path
        self.questions = []
        self.questions_by_id = {}
        self.load()

    def load(self) -> List[Dict[str, Any]]:
        """
        Load all questions from dataset

        Returns:
            List of question dictionaries
        """
        import os

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        file_ext = os.path.splitext(self.dataset_path)[1].lower()

        if file_ext == '.json':
            self._load_json()
        elif file_ext == '.jsonl':
            self._load_jsonl()
        elif file_ext == '.csv':
            self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use JSON, JSONL, or CSV.")

        # Build index by question_id
        self.questions_by_id = {q['question_id']: q for q in self.questions}

        print(f"✓ Loaded {len(self.questions)} questions from {self.dataset_path}")
        return self.questions

    def _load_json(self):
        """Load questions from JSON file"""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        # Support both {"questions": [...]} and [...]
        if isinstance(data, dict) and 'questions' in data:
            self.questions = data['questions']
        elif isinstance(data, list):
            # Check if this is the new format (has 'features', 'target', 'q_type')
            if len(data) > 0 and 'features' in data[0] and 'target' in data[0]:
                # Convert new format to standard format
                self.questions = self._convert_new_format_to_standard(data)
            else:
                self.questions = data
        else:
            raise ValueError("JSON file must contain 'questions' array or be an array itself")

    def _convert_new_format_to_standard(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert new dataset format to standard question format

        New format has:
        - dataset: str (e.g., "adult_census", "imdb", "stl10")
        - row_no: int
        - features: Dict (varies by dataset type)
        - target: Dict with value, label
        - predicted: Dict with value, label
        - q_type: int (1-10)
        - q: str (question template)
        - example: str (filled question)

        Standard format needs:
        - question_id: str
        - q_type: int (template identifier)
        - question: str
        - modality: str
        - data_path: str (path to data file)
        - features: Dict
        - target: Dict
        - predicted: Dict
        - metadata: Dict
        """
        converted = []

        for item in data:
            # Determine modality and dataset name based on features and dataset field
            features = item.get('features', {})
            dataset_name = item.get('dataset', '')

            # Determine modality
            if 'modality' in features and features['modality'] == 'image':
                modality = 'vision'
            elif 'review_text' in features:
                modality = 'text'
            else:
                modality = 'tabular'

            # Create standard question dict
            question_dict = {
                'question_id': f"{dataset_name}_q{item.get('q_type')}_{item.get('row_no', len(converted))}",
                'q_type': item.get('q_type', 0),  # Keep as int (1-10)
                'question': item.get('example', item.get('q', '')),
                'q_template': item.get('q', ''),
                'modality': modality,
                'dataset': dataset_name,
                'features': features,
                'target': item.get('target', {}),
                'predicted': item.get('predicted', {}),
                'metadata': {
                    'row_no': item.get('row_no'),
                    'dataset': dataset_name
                }
            }

            # Add modality-specific fields and data paths
            if modality == 'vision':
                # For vision, we need to construct the image path
                # Assuming images are stored in a directory based on dataset name
                image_index = features.get('image_index', item.get('row_no', 0))
                question_dict['data_path'] = None  # To be set by user or loaded from dataset directory
                question_dict['metadata']['image_index'] = image_index
                question_dict['metadata']['image_shape'] = features.get('image_shape')
            elif modality == 'text':
                # For text, store the review text
                question_dict['text_input'] = features.get('review_text', '')
                question_dict['data_path'] = None
            else:  # tabular
                # For tabular, features are already in the dict
                question_dict['data_path'] = None

            converted.append(question_dict)

        print(f"  ✓ Converted {len(converted)} items from new format to standard format")
        print(f"    - Modalities: {sum(1 for q in converted if q['modality']=='tabular')} tabular, "
              f"{sum(1 for q in converted if q['modality']=='text')} text, "
              f"{sum(1 for q in converted if q['modality']=='vision')} vision")
        print(f"    - Datasets: {', '.join(set(q['dataset'] for q in converted))}")

        return converted

    def _load_jsonl(self):
        """Load questions from JSONL file (one JSON object per line)"""
        self.questions = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.questions.append(json.loads(line))

    def _load_csv(self):
        """Load questions from CSV file"""
        import csv

        self.questions = []
        with open(self.dataset_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert metadata if it's a JSON string
                if 'metadata' in row and isinstance(row['metadata'], str):
                    try:
                        row['metadata'] = json.loads(row['metadata'])
                    except:
                        pass
                self.questions.append(row)

    def get_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific question by ID

        Args:
            question_id: The question ID to retrieve

        Returns:
            Question dictionary or None if not found
        """
        return self.questions_by_id.get(question_id)

    def get_template_for_question(self, question: Dict[str, Any]) -> Optional[QuestionTemplate]:
        """
        Get the template for a given question

        Args:
            question: Question dictionary with 'q_type' field

        Returns:
            QuestionTemplate or None if not found
        """
        q_type = question.get('q_type')
        if q_type is None:
            return None

        # Determine modality from question
        modality_str = question.get('modality', 'vision')
        try:
            modality = Modality(modality_str)
        except ValueError:
            modality = Modality.VISION

        # Handle integer q_type (1-10)
        if isinstance(q_type, int):
            return TemplateRegistry.get_template_by_qtype(q_type, modality)

        # Handle string template IDs
        elif isinstance(q_type, str):
            return TemplateRegistry.get_template(q_type)

        return None

    def iterate(self):
        """
        Iterator over all questions

        Yields:
            Question dictionaries
        """
        for question in self.questions:
            yield question

    def filter_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """
        Filter questions by modality

        Args:
            modality: "vision", "text", or "tabular"

        Returns:
            List of questions matching the modality
        """
        return [q for q in self.questions if q.get('modality') == modality]

    def filter_by_q_type(self, q_type: str) -> List[Dict[str, Any]]:
        """
        Filter questions by template type

        Args:
            q_type: Template type (e.g., "vision_feature_attribution_1")

        Returns:
            List of questions matching the type
        """
        return [q for q in self.questions if q.get('q_type') == q_type]

    def __len__(self) -> int:
        """Return number of questions"""
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get question by index"""
        return self.questions[idx]


if __name__ == "__main__":
    # Test template matching
    test_question = "The model predicted the image to be a plane. Which part of the input was most responsible for the model's prediction?"
    
    matched_template = TemplateRegistry.match_question(test_question, Modality.VISION)
    
    if matched_template:
        print(f"✓ Matched template: {matched_template.template}")
        print(f"  Category: {matched_template.category.value}")
        print(f"  Modality: {matched_template.modality.value}")
        print(f"  Prompt Builder: {matched_template.prompt_builder.__class__.__name__}")
        print(f"  Extraction fields: {[f.name for f in matched_template.extraction_fields]}")
        
        # Test Vision-specific methods
        if isinstance(matched_template.prompt_builder, VisionPromptBuilder):
            print("\n✓ Vision-specific features available:")
            print("  - build_image_analysis_prompt()")
    else:
        print("✗ No matching template found")