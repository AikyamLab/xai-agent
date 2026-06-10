"""
XAI Tools Native Implementation (No LangChain Dependency)

Provides XAI tool wrappers and registry without LangChain.
Uses simple Python ABC for tool base class.

IMPORTANT: Tools return RAW results (visualizations, statistics, descriptions).
Feature extraction (bounding boxes, importance regions) is done by the Actor Agent
through VLM reasoning on the visualization outputs.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import json
import torch
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from xai_tools import (
    get_available_tools,
    execute_gradcam,
    execute_integrated_gradients,
    execute_lime,
    execute_shap,
    execute_guided_backprop,
    execute_guided_backprop_text,
    execute_guided_backprop_tabular,
    execute_smoothgrad,
    execute_smoothgrad_text,
    execute_smoothgrad_tabular,
    set_output_dir,
    get_output_dir
)


# ===================================================================
# Utility: detect whether a model applies sigmoid/softmax internally
# ===================================================================

def _model_has_final_sigmoid(model) -> bool:
    """Return True if the model's last activation is Sigmoid or Softmax.

    Mirrors the logic in evaluation/base_evaluator.py so that tools and
    evaluator treat single-output binary models consistently.  When a model
    already applies sigmoid in forward(), tools must NOT apply it again.
    """
    modules = list(model.modules())
    for m in reversed(modules):
        if isinstance(m, (torch.nn.Sigmoid, torch.nn.Softmax)):
            return True
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            return False
    return False


# ===================================================================
# Base Tool Class (Replaces LangChain BaseTool)
# ===================================================================

class BaseTool(ABC):
    """
    Base class for XAI tools.

    Replaces langchain_classic.tools.BaseTool with a simple abstract base class.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        Execute the tool and return results as JSON string.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            JSON string with results
        """
        pass

    def __call__(self, **kwargs) -> str:
        """Allow tool to be called directly."""
        return self.run(**kwargs)


# ===================================================================
# Input Schemas for Tools (Using Pydantic)
# ===================================================================

def _parse_nested_json_input(data: Any, field_names: List[str]) -> Dict:
    """
    Common helper to parse nested JSON input for XAI tools.

    Handles cases where the input is malformed - e.g., when the entire JSON string
    is incorrectly passed as the first field's value.
    """
    if not isinstance(data, dict):
        return data

    first_field = field_names[0] if field_names else 'image_path'
    first_value = data.get(first_field, '')

    if isinstance(first_value, str) and first_value.strip().startswith('{'):
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', first_value)
            if json_match:
                nested_data = json.loads(json_match.group())
                for field in field_names:
                    if field in nested_data:
                        data[field] = nested_data[field]
        except (json.JSONDecodeError, AttributeError):
            pass

    return data


class GradCAMInput(BaseModel):
    """Input schema for GradCAM tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'image_id'])


class IntegratedGradientsInput(BaseModel):
    """Input schema for Integrated Gradients tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    n_steps: int = Field(default=200, description="Number of integration steps")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'n_steps', 'image_id'])


class LIMEInput(BaseModel):
    """Input schema for LIME tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    num_samples: int = Field(default=1000, description="Number of samples for LIME")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'num_samples', 'image_id'])


class SHAPInput(BaseModel):
    """Input schema for SHAP tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    num_samples: int = Field(default=1000, description="Number of samples for SHAP")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'num_samples', 'image_id'])


class GuidedBackpropInput(BaseModel):
    """Input schema for Guided Backpropagation tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'image_id'])



# ===================================================================
# XAI Tool Implementations
# ===================================================================

class GradCAMTool(BaseTool):
    """Tool for executing GradCAM analysis."""

    name = "gradcam"
    description = (
        "Executes GradCAM to visualize which image regions the model focuses on. "
        "Returns a heatmap visualization and statistics. "
        "The Actor Agent will analyze the visualization to identify important regions."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """
        Initialize GradCAM tool.

        Args:
            data_model_loader: DataModelLoader instance for accessing the model and data.
        """
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        target_class: Optional[int] = None,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute GradCAM analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device
        model_type = self.data_model_loader.model_name
        
        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded in DataModelLoader"})

        if target_class is None:
            return json.dumps({
                "success": False,
                "error": "target_class is required"
            })

        try:
            target_class = int(target_class)
            image = self.data_model_loader.get_display_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })

            # Get pre-processed tensor if available
            input_tensor = self.data_model_loader.get_current_tensor()

            result = execute_gradcam(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=target_class,
                device=device,
                image_id=image_id,
                input_tensor=input_tensor
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


def _get_image_from_source(data_model_loader: Optional[Any], image_path: Optional[str]) -> Optional[Image.Image]:
    """
    Helper function to get PIL Image from available sources.

    Priority:
    1. data_model_loader.current_image (if available)
    2. File path (if exists)

    Args:
        data_model_loader: DataModelLoader instance
        image_path: Path to image file

    Returns:
        PIL Image or None
    """
    import os

    # 1. Try data_model_loader.current_image first
    if data_model_loader is not None and hasattr(data_model_loader, 'current_image'):
        if data_model_loader.current_image is not None:
            img = data_model_loader.current_image
            if isinstance(img, Image.Image):
                return img

    # 2. Try to open from file path
    if image_path is not None:
        image_path = str(image_path).strip().strip('"').strip("'")
        if os.path.exists(image_path):
            return Image.open(image_path).convert('RGB')

    return None


class IntegratedGradientsTool(BaseTool):
    """Tool for executing Integrated Gradients analysis."""

    name = "integrated_gradients"
    description = (
        "Executes Integrated Gradients to compute pixel-level importance. "
        "Returns an attribution map showing which pixels contributed to the prediction. "
        "The Actor Agent will analyze the visualization to identify important regions."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """
        Initialize Integrated Gradients tool.

        Args:
            data_model_loader: DataModelLoader instance for accessing the model and data.
        """
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        target_class: Optional[int] = None,
        n_steps: int = 200,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute Integrated Gradients analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device
        model_type = self.data_model_loader.model_name
        
        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded in DataModelLoader"})

        if target_class is None:
            return json.dumps({
                "success": False,
                "error": "target_class is required"
            })

        try:
            target_class = int(target_class)
            image = self.data_model_loader.get_display_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })

            # Get pre-processed tensor if available
            input_tensor = self.data_model_loader.get_current_tensor()

            result = execute_integrated_gradients(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=target_class,
                device=device,
                image_id=image_id,
                n_steps=n_steps,
                input_tensor=input_tensor
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class LIMETool(BaseTool):
    """Tool for executing LIME analysis."""

    name = "lime"
    description = (
        "Executes LIME (Local Interpretable Model-agnostic Explanations) to identify important image regions. "
        "Returns a visualization with superpixel-level importance. "
        "The Actor Agent will analyze the visualization to identify important regions."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """
        Initialize LIME tool.

        Args:
            data_model_loader: DataModelLoader instance for accessing the model and data.
        """
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        target_class: Optional[int] = None,
        num_samples: int = 1000,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute LIME analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device
        model_type = self.data_model_loader.model_name

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded in DataModelLoader"})

        if target_class is None:
            return json.dumps({
                "success": False,
                "error": "target_class is required"
            })

        try:
            target_class = int(target_class)
            image = self.data_model_loader.get_display_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })
            result = execute_lime(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=target_class,
                device=device,
                image_id=image_id,
                num_samples=num_samples
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class SHAPTool(BaseTool):
    """Tool for executing SHAP analysis."""

    name = "shap"
    description = (
        "Executes SHAP (SHapley Additive exPlanations) to compute feature importance. "
        "Returns an attribution map based on game-theoretic feature importance. "
        "The Actor Agent will analyze the visualization to identify important regions."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """
        Initialize SHAP tool.

        Args:
            data_model_loader: DataModelLoader instance for accessing the model and data.
        """
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        target_class: Optional[int] = None,
        num_samples: int = 1000,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute SHAP analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device
        model_type = self.data_model_loader.model_name

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded in DataModelLoader"})

        if target_class is None:
            return json.dumps({
                "success": False,
                "error": "target_class is required"
            })

        try:
            target_class = int(target_class)
            # Use raw image (not preprocessed) so SHAP operates at original resolution.
            # get_display_image() may return the processed (upscaled) image for small inputs
            # like STL10 (96×96), causing SHAP to run on 224×224 with too few evaluations.
            sample_data = self.data_model_loader.current_sample_data
            image = sample_data.get("image") if sample_data else None
            if image is None:
                image = self.data_model_loader.get_display_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })

            result = execute_shap(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=target_class,
                device=device,
                image_id=image_id,
                num_samples=num_samples
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class GuidedBackpropTool(BaseTool):
    """Tool for executing Guided Backpropagation analysis."""

    name = "guided_backprop"
    description = (
        "Executes Guided Backpropagation to visualize pixel-level importance. "
        "Returns a saliency map highlighting pixels that contribute to the prediction. "
        "The Actor Agent will analyze the visualization to identify important regions."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """
        Initialize Guided Backpropagation tool.

        Args:
            data_model_loader: DataModelLoader instance for accessing the model and data.
        """
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        target_class: Optional[int] = None,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute Guided Backpropagation analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device
        model_type = self.data_model_loader.model_name

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded in DataModelLoader"})

        if target_class is None:
            return json.dumps({
                "success": False,
                "error": "target_class is required"
            })

        try:
            target_class = int(target_class)
            image = self.data_model_loader.get_display_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })

            # Get pre-processed tensor if available
            input_tensor = self.data_model_loader.get_current_tensor()

            result = execute_guided_backprop(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=target_class,
                device=device,
                image_id=image_id,
                input_tensor=input_tensor
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})



# ===================================================================
# Text-specific XAI Tool Implementations
# ===================================================================

class LIMETextTool(BaseTool):
    """Tool for executing LIME analysis on text data."""

    name = "lime"
    description = (
        "Executes LIME (Local Interpretable Model-agnostic Explanations) for text. "
        "Returns word-level importance scores showing which words contributed to the prediction."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, num_samples: int = 1000,
            image_id: str = "temp", **kwargs) -> str:
        """Execute LIME text analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        # Detect dual-input NLI model (e.g., SNLI: premise + hypothesis)
        sample_data = self.data_model_loader.current_sample_data or {}
        is_nli = 'hypothesis_tensor' in sample_data and 'hypothesis' in sample_data
        is_dual_input = is_nli and len(inspect.signature(model.forward).parameters) >= 2

        if is_nli:
            # For NLI, LIME perturbs only the premise; hypothesis stays fixed
            text = sample_data.get('premise', '')
        else:
            text = self.data_model_loader.get_current_text()

        if text is None or text == '':
            return json.dumps({"success": False, "error": "No text available. Load a sample first."})

        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        try:
            from lime import lime_text
            import numpy as np
            target_class = int(target_class)

            # Detect HuggingFace tokenizers: calling them returns a BatchEncoding
            # (dict-like but NOT a subclass of dict) rather than a plain list of ints.
            # We probe with a short string and check for an 'input_ids' key.
            _probe = processor(text[:64]) if processor is not None else None
            _is_hf_tokenizer = _probe is not None and hasattr(_probe, 'get') and 'input_ids' in _probe

            def _encode(t):
                """Return (input_ids_tensor, attention_mask_tensor_or_None) on device."""
                enc = processor(t, truncation=True, max_length=512) if _is_hf_tokenizer else processor(t)
                if _is_hf_tokenizer:
                    ids  = enc.get('input_ids')
                    mask = enc.get('attention_mask')
                    if not isinstance(ids, torch.Tensor):
                        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                    if mask is not None and not isinstance(mask, torch.Tensor):
                        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
                    return ids.to(device), (mask.to(device) if mask is not None else None)
                else:
                    # Legacy: plain list of int token IDs
                    return torch.tensor([enc], dtype=torch.long).to(device), None

            if is_nli:
                hypothesis_ids = sample_data.get('hypothesis_ids')
                if hypothesis_ids is None and processor is not None:
                    hyp_ids, _ = _encode(sample_data.get('hypothesis', ''))
                else:
                    hyp_ids = torch.tensor([hypothesis_ids], dtype=torch.long).to(device)
                hyp_tensor = hyp_ids

            def predict_fn(texts):
                """Generic predict function for any PyTorch text model."""
                model.eval()
                results = []
                with torch.no_grad():
                    for t in texts:
                        input_ids, attention_mask = _encode(t)

                        if is_dual_input:
                            logits = model(input_ids, hyp_tensor)
                        elif is_nli:
                            logits = model(torch.cat([input_ids, hyp_tensor], dim=1))
                        elif _is_hf_tokenizer and attention_mask is not None:
                            # HuggingFace model: must pass attention_mask via keyword args
                            out = model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = out.logits if hasattr(out, 'logits') else out
                        else:
                            logits = model(input_ids)

                        if logits.shape[-1] == 1:
                            # Binary classification: sigmoid
                            prob_pos = torch.sigmoid(logits).item()
                            results.append([1.0 - prob_pos, prob_pos])
                        else:
                            # Multi-class: softmax
                            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                            results.append(probs)
                return np.array(results)

            num_classes = predict_fn([text]).shape[1]
            explainer = lime_text.LimeTextExplainer(
                class_names=[str(i) for i in range(num_classes)]
            )

            explanation = explainer.explain_instance(
                text,
                predict_fn,
                num_samples=num_samples,
                labels=(target_class,)
            )

            explanation_list = explanation.as_list(label=target_class)

            word_importance = []
            for word, weight in explanation_list[:20]:
                word_importance.append({
                    "word": word,
                    "weight": round(float(weight), 4),
                    "direction": "positive" if weight > 0 else "negative"
                })

            # Save visualization
            output_dir = get_output_dir()
            viz_path = str(output_dir / f"lime_text_{image_id}.html")
            explanation.save_to_file(viz_path)

            result = {
                "success": True,
                "method": "LIME (Text)",
                "target_class": target_class,
                "num_samples": num_samples,
                "visualization_path": viz_path,
                "text_length": len(text),
                "statistics": {
                    "num_important_words": len(word_importance),
                    "word_importance": word_importance,
                    "max_positive_weight": max([w["weight"] for w in word_importance if w["weight"] > 0], default=0),
                    "max_negative_weight": min([w["weight"] for w in word_importance if w["weight"] < 0], default=0)
                },
                "description": (
                    f"LIME text analysis for class {target_class}: "
                    f"Found {len(word_importance)} important words. "
                    f"Top: {word_importance[0]['word'] if word_importance else 'N/A'}."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


class IntegratedGradientsTextTool(BaseTool):
    """Tool for executing Integrated Gradients analysis on text data."""

    name = "integrated_gradients"
    description = (
        "Executes Integrated Gradients on text to compute per-token importance. "
        "Returns attribution scores for each token in the input text."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, n_steps: int = 50,
            image_id: str = "temp", **kwargs) -> str:
        """Execute Integrated Gradients for text."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        # Detect dual-input NLI model (e.g., SNLI: premise + hypothesis)
        sample_data = self.data_model_loader.current_sample_data or {}
        is_nli = 'hypothesis_tensor' in sample_data and 'hypothesis' in sample_data
        is_dual_input = is_nli and len(inspect.signature(model.forward).parameters) >= 2

        if is_nli:
            text = sample_data.get('premise', '')
        else:
            text = self.data_model_loader.get_current_text()

        if text is None or text == '':
            return json.dumps({"success": False, "error": "No text available. Load a sample first."})

        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        try:
            from captum.attr import LayerIntegratedGradients, IntegratedGradients
            import numpy as np
            import re
            target_class = int(target_class)

            # Detect HuggingFace tokenizers (BatchEncoding, not plain list of ints)
            _probe = processor(text[:64]) if processor is not None else None
            _is_hf_tokenizer = _probe is not None and hasattr(_probe, 'get') and 'input_ids' in _probe

            def _encode(t):
                enc = processor(t, truncation=True, max_length=512) if _is_hf_tokenizer else processor(t)
                if _is_hf_tokenizer:
                    ids = enc.get('input_ids')
                    mask = enc.get('attention_mask')
                    if not isinstance(ids, torch.Tensor):
                        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                    if mask is not None and not isinstance(mask, torch.Tensor):
                        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
                    return ids.to(device), (mask.to(device) if mask is not None else None)
                else:
                    return torch.tensor([enc], dtype=torch.long).to(device), None

            # Build input_tensor and attention_mask
            if _is_hf_tokenizer:
                input_tensor, attention_mask_tensor = _encode(text)
            else:
                if is_nli:
                    token_ids = sample_data.get('premise_ids')
                    if token_ids is None:
                        token_ids = processor(text)
                else:
                    token_ids = self.data_model_loader.get_current_token_ids()
                    if token_ids is None:
                        token_ids = processor(text)
                input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
                attention_mask_tensor = None

            if is_nli:
                hypothesis_ids = sample_data.get('hypothesis_ids')
                if hypothesis_ids is None and processor is not None:
                    hyp_ids, _ = _encode(sample_data.get('hypothesis', ''))
                else:
                    hyp_ids = torch.tensor([hypothesis_ids], dtype=torch.long).to(device)
                hyp_tensor = hyp_ids

            # Find embedding layer
            embedding_layer = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    embedding_layer = module
                    break

            if embedding_layer is None:
                return json.dumps({"success": False, "error": "No embedding layer found in model"})

            if is_dual_input:
                # Dual-input NLI (e.g. CNN_SNLI): use IntegratedGradients on premise
                # embeddings with a hook that only replaces the first embedding call.
                premise_embeds = embedding_layer(input_tensor).detach().clone().requires_grad_(True)
                baseline_embeds = embedding_layer(torch.zeros_like(input_tensor).to(device)).detach().clone()

                def forward_from_embeds(prem_embeds):
                    batch_size = prem_embeds.shape[0]
                    call_count = [0]
                    def hook_fn(module, inp, out):
                        call_count[0] += 1
                        if call_count[0] == 1:  # First call = premise
                            return prem_embeds
                        return out  # Second call = hypothesis, keep original
                    handle = embedding_layer.register_forward_hook(hook_fn)
                    expanded_input = input_tensor.expand(batch_size, -1)
                    expanded_hyp = hyp_tensor.expand(batch_size, -1)
                    out = model(expanded_input, expanded_hyp)
                    logits = out.logits if hasattr(out, 'logits') else out
                    handle.remove()
                    if logits.shape[-1] == 1:
                        prob_pos = torch.sigmoid(logits)
                        return torch.cat([1.0 - prob_pos, prob_pos], dim=-1)
                    return logits

                ig = IntegratedGradients(forward_from_embeds)
                attributions = ig.attribute(
                    premise_embeds,
                    baselines=baseline_embeds,
                    target=target_class,
                    n_steps=n_steps
                )
            else:
                # Standard single-input (IMDB or single-input SNLI): LayerIntegratedGradients
                def forward_func(input_ids):
                    if _is_hf_tokenizer and attention_mask_tensor is not None:
                        out = model(input_ids=input_ids, attention_mask=attention_mask_tensor)
                    else:
                        out = model(input_ids)
                    logits = out.logits if hasattr(out, 'logits') else out
                    if logits.shape[-1] == 1:
                        prob_pos = torch.sigmoid(logits)
                        probs = torch.cat([1.0 - prob_pos, prob_pos], dim=-1)
                        return probs
                    return logits

                lig = LayerIntegratedGradients(forward_func, embedding_layer)
                baseline = torch.zeros_like(input_tensor).to(device)
                attributions = lig.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target_class,
                    n_steps=n_steps
                )

            # Sum attributions across embedding dim to get per-token scores
            attr_scores = attributions.sum(dim=-1).squeeze().cpu().detach().numpy()

            # Map scores to words
            words = text.lower()
            words = re.sub(r'<br\s*/?>', ' ', words)
            words = re.sub(r'[^a-z0-9\s]', ' ', words)
            word_list = words.split()

            token_importance = []
            for i, word in enumerate(word_list[:len(attr_scores)]):
                score = float(attr_scores[i]) if i < len(attr_scores) else 0.0
                token_importance.append({
                    "token": word,
                    "token_index": i,
                    "attribution_score": round(score, 6),
                    "direction": "positive" if score > 0 else "negative"
                })

            # Sort by absolute attribution
            token_importance.sort(key=lambda x: abs(x['attribution_score']), reverse=True)

            result = {
                "success": True,
                "method": "Integrated Gradients (Text)",
                "target_class": target_class,
                "n_steps": n_steps,
                "statistics": {
                    "num_tokens": len(token_importance),
                    "token_importance": token_importance[:20],
                    "max_positive": max([t["attribution_score"] for t in token_importance if t["attribution_score"] > 0], default=0),
                    "max_negative": min([t["attribution_score"] for t in token_importance if t["attribution_score"] < 0], default=0)
                },
                "description": (
                    f"Integrated Gradients text analysis for class {target_class}: "
                    f"Analyzed {len(token_importance)} tokens. "
                    f"Most influential: {token_importance[0]['token'] if token_importance else 'N/A'}."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


class SHAPTextTool(BaseTool):
    """Tool for executing SHAP analysis on text data."""

    name = "shap"
    description = (
        "Executes SHAP (SHapley Additive exPlanations) for text using KernelExplainer. "
        "Returns word-level SHAP values showing each word's contribution to the prediction."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, image_id: str = "temp", **kwargs) -> str:
        """Execute SHAP text analysis using word-level perturbation."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        sample_data = self.data_model_loader.current_sample_data or {}
        is_nli = 'hypothesis_tensor' in sample_data and 'hypothesis' in sample_data
        is_dual_input = is_nli and len(inspect.signature(model.forward).parameters) >= 2

        if is_nli:
            text = sample_data.get('premise', '')
        else:
            text = self.data_model_loader.get_current_text()

        if text is None or text == '':
            return json.dumps({"success": False, "error": "No text available. Load a sample first."})

        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        try:
            import shap
            import numpy as np
            import re
            target_class = int(target_class)

            # Detect HuggingFace tokenizers (BatchEncoding, not plain list of ints)
            _probe = processor(text[:64]) if processor is not None else None
            _is_hf_tokenizer = _probe is not None and hasattr(_probe, 'get') and 'input_ids' in _probe

            def _encode(t):
                enc = processor(t, truncation=True, max_length=512) if _is_hf_tokenizer else processor(t)
                if _is_hf_tokenizer:
                    ids = enc.get('input_ids')
                    mask = enc.get('attention_mask')
                    if not isinstance(ids, torch.Tensor):
                        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                    if mask is not None and not isinstance(mask, torch.Tensor):
                        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
                    return ids.to(device), (mask.to(device) if mask is not None else None)
                else:
                    return torch.tensor([enc], dtype=torch.long).to(device), None

            # Tokenize text into words
            words = text.split()
            n_words = len(words)

            if n_words == 0:
                return json.dumps({"success": False, "error": "Text has no words after splitting"})

            # For NLI, prepare hypothesis tensor
            if is_nli:
                hypothesis_ids = sample_data.get('hypothesis_ids')
                if hypothesis_ids is None and processor is not None:
                    hyp_ids, _ = _encode(sample_data.get('hypothesis', ''))
                else:
                    hyp_ids = torch.tensor([hypothesis_ids], dtype=torch.long).to(device)
                hyp_tensor = hyp_ids

            def predict_fn(masks):
                """Predict from binary word masks. masks: (n_samples, n_words)"""
                model.eval()
                results = []
                with torch.no_grad():
                    for mask in masks:
                        masked_words = [w for w, m in zip(words, mask) if m == 1]
                        masked_text = ' '.join(masked_words) if masked_words else '.'

                        input_ids, attention_mask = _encode(masked_text)

                        if is_dual_input:
                            out = model(input_ids, hyp_tensor)
                        elif is_nli:
                            out = model(torch.cat([input_ids, hyp_tensor], dim=1))
                        elif _is_hf_tokenizer and attention_mask is not None:
                            out = model(input_ids=input_ids, attention_mask=attention_mask)
                        else:
                            out = model(input_ids)

                        logits = out.logits if hasattr(out, 'logits') else out

                        if logits.shape[-1] == 1:
                            prob_pos = torch.sigmoid(logits).item()
                            results.append([1.0 - prob_pos, prob_pos])
                        else:
                            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                            results.append(probs)
                return np.array(results)

            # Background: no words present (empty text baseline).
            # Sample: all words present.
            # SHAP values measure each word's contribution relative to the empty baseline.
            background = np.zeros((1, n_words))
            explainer = shap.KernelExplainer(predict_fn, background)

            # Explain: all words present
            sample = np.ones((1, n_words))
            shap_values = explainer.shap_values(sample, nsamples=min(2 * n_words + 2048, 5000), silent=True)

            # Get SHAP values for target class
            if isinstance(shap_values, list):
                vals = shap_values[target_class].flatten()
            else:
                vals = shap_values.flatten()

            # Build word importance list
            word_importance = []
            for i, (word, val) in enumerate(zip(words, vals)):
                word_importance.append({
                    "word": word,
                    "shap_value": round(float(val), 6),
                    "direction": "positive" if val > 0 else "negative"
                })

            word_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)

            result = {
                "success": True,
                "method": "SHAP (Text)",
                "target_class": target_class,
                "statistics": {
                    "num_words": len(word_importance),
                    "word_importance": word_importance[:20],
                    "top_positive": max([w["shap_value"] for w in word_importance if w["shap_value"] > 0], default=0),
                    "top_negative": min([w["shap_value"] for w in word_importance if w["shap_value"] < 0], default=0)
                },
                "description": (
                    f"SHAP text analysis for class {target_class}: "
                    f"Analyzed {len(word_importance)} words. "
                    f"Most influential: {word_importance[0]['word'] if word_importance else 'N/A'}."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


class SensitivityAnalysisTextTool(BaseTool):
    """Tool for executing leave-one-out sensitivity analysis on text data."""

    name = "sensitivity_analysis"
    description = (
        "Executes leave-one-out word ablation to measure each word's impact on prediction. "
        "Returns probability changes when each word is removed from the text."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, image_id: str = "temp", **kwargs) -> str:
        """Execute leave-one-out sensitivity analysis for text."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        sample_data = self.data_model_loader.current_sample_data or {}
        is_nli = 'hypothesis_tensor' in sample_data and 'hypothesis' in sample_data
        is_dual_input = is_nli and len(inspect.signature(model.forward).parameters) >= 2

        if is_nli:
            text = sample_data.get('premise', '')
        else:
            text = self.data_model_loader.get_current_text()

        if text is None or text == '':
            return json.dumps({"success": False, "error": "No text available. Load a sample first."})

        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        try:
            import numpy as np
            target_class = int(target_class)

            # Detect HuggingFace tokenizers (BatchEncoding, not plain list of ints)
            _probe = processor(text[:64]) if processor is not None else None
            _is_hf_tokenizer = _probe is not None and hasattr(_probe, 'get') and 'input_ids' in _probe

            def _encode(t):
                enc = processor(t, truncation=True, max_length=512) if _is_hf_tokenizer else processor(t)
                if _is_hf_tokenizer:
                    ids = enc.get('input_ids')
                    mask = enc.get('attention_mask')
                    if not isinstance(ids, torch.Tensor):
                        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                    if mask is not None and not isinstance(mask, torch.Tensor):
                        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
                    return ids.to(device), (mask.to(device) if mask is not None else None)
                else:
                    return torch.tensor([enc], dtype=torch.long).to(device), None

            words = text.split()
            n_words = len(words)

            if n_words == 0:
                return json.dumps({"success": False, "error": "Text has no words after splitting"})

            # For NLI, prepare hypothesis tensor
            if is_nli:
                hypothesis_ids = sample_data.get('hypothesis_ids')
                if hypothesis_ids is None and processor is not None:
                    hyp_ids, _ = _encode(sample_data.get('hypothesis', ''))
                else:
                    hyp_ids = torch.tensor([hypothesis_ids], dtype=torch.long).to(device)
                hyp_tensor = hyp_ids

            def get_prob(input_text):
                """Get target class probability for a text."""
                model.eval()
                with torch.no_grad():
                    input_ids, attention_mask = _encode(input_text)

                    if is_dual_input:
                        out = model(input_ids, hyp_tensor)
                    elif is_nli:
                        out = model(torch.cat([input_ids, hyp_tensor], dim=1))
                    elif _is_hf_tokenizer and attention_mask is not None:
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        out = model(input_ids)

                    logits = out.logits if hasattr(out, 'logits') else out

                    if logits.shape[-1] == 1:
                        prob_pos = torch.sigmoid(logits).item()
                        probs = [1.0 - prob_pos, prob_pos]
                    else:
                        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
                    return probs[target_class]

            # Get original probability
            original_prob = get_prob(text)

            # Leave-one-out: remove each word and measure probability change
            word_sensitivity = []
            for i, word in enumerate(words):
                masked_words = words[:i] + words[i+1:]
                masked_text = ' '.join(masked_words) if masked_words else '.'
                masked_prob = get_prob(masked_text)
                prob_drop = original_prob - masked_prob

                word_sensitivity.append({
                    "word": word,
                    "word_index": i,
                    "original_prob": round(float(original_prob), 6),
                    "masked_prob": round(float(masked_prob), 6),
                    "prob_drop": round(float(prob_drop), 6)
                })

            # Sort by absolute prob_drop (most sensitive first)
            word_sensitivity.sort(key=lambda x: abs(x['prob_drop']), reverse=True)

            result = {
                "success": True,
                "method": "Sensitivity Analysis (Text)",
                "target_class": target_class,
                "original_prob": round(float(original_prob), 6),
                "statistics": {
                    "num_words": len(word_sensitivity),
                    "word_sensitivity": word_sensitivity[:20],
                    "max_prob_drop": max([w["prob_drop"] for w in word_sensitivity], default=0),
                    "max_prob_increase": min([w["prob_drop"] for w in word_sensitivity], default=0)
                },
                "description": (
                    f"Sensitivity analysis for class {target_class}: "
                    f"Ablated {len(word_sensitivity)} words. "
                    f"Most sensitive: {word_sensitivity[0]['word'] if word_sensitivity else 'N/A'} "
                    f"(prob drop: {word_sensitivity[0]['prob_drop'] if word_sensitivity else 0})."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ===================================================================
# Tabular-specific XAI Tool Implementations
# ===================================================================

def _aggregate_encoded_to_original(items, score_key, encoded_to_original, raw_features_dict):
    """
    Aggregate one-hot encoded feature attributions back to original feature names.
    Sums attribution scores for all encoded features belonging to the same original column.
    Returns list sorted by absolute score descending, with 'value' showing the raw feature value.
    """
    agg = {}
    for item in items:
        enc_name = item.get("feature", "")
        orig_col = encoded_to_original.get(enc_name, enc_name)
        score = item.get(score_key, 0.0)
        if orig_col not in agg:
            raw_val = raw_features_dict.get(orig_col, "?") if raw_features_dict else "?"
            agg[orig_col] = {"feature": orig_col, "value": raw_val, score_key: 0.0}
        agg[orig_col][score_key] += score
    result = list(agg.values())
    for item in result:
        item["direction"] = "positive" if item[score_key] > 0 else "negative"
    result.sort(key=lambda x: abs(x[score_key]), reverse=True)
    return result


def _extract_feature_name_from_lime_condition(condition, feature_names_sorted_by_len):
    """Extract the base feature name from a LIME condition string (e.g. '0.50 < feat <= 1.00' -> 'feat')."""
    for name in feature_names_sorted_by_len:
        if name in condition:
            return name
    return condition

class SHAPTabularTool(BaseTool):
    """Tool for executing SHAP analysis on tabular data."""

    name = "shap"
    description = (
        "Executes SHAP (SHapley Additive exPlanations) for tabular data. "
        "Returns feature-level importance scores based on game-theoretic values."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, image_id: str = "temp", **kwargs) -> str:
        """Execute SHAP tabular analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        features = self.data_model_loader.get_current_features()
        if features is None:
            return json.dumps({"success": False, "error": "No tabular data available. Load a sample first."})


        feature_names = self.data_model_loader.get_feature_names()

        try:
            import shap
            import numpy as np
            target_class = int(target_class) if target_class is not None else 0

            def predict_fn(X):
                """Predict function for SHAP."""
                model.eval()
                with torch.no_grad():
                    if isinstance(X, np.ndarray):
                        input_t = torch.tensor(X, dtype=torch.float32).to(device)
                    else:
                        input_t = X.float().to(device)
                    if input_t.dim() == 1:
                        input_t = input_t.unsqueeze(0)
                    outputs = model(input_t)
                    if outputs.shape[-1] == 1:
                        prob_pos = outputs.cpu().numpy().flatten()
                        return np.column_stack([1.0 - prob_pos, prob_pos])
                    else:
                        return torch.softmax(outputs, dim=-1).cpu().numpy()

            # Use training data as background if available
            training_data = self.data_model_loader.get_training_data()
            if training_data is not None:
                # Subsample for efficiency
                n_bg = min(100, len(training_data))
                indices = np.random.choice(len(training_data), n_bg, replace=False)
                background = training_data[indices].numpy()
            else:
                background = features.unsqueeze(0).cpu().numpy()

            explainer = shap.KernelExplainer(predict_fn, background)

            sample = features.unsqueeze(0).cpu().numpy() if features.dim() == 1 else features.cpu().numpy()
            n_encoded = sample.shape[-1]
            nsamples = max(2048, 2 * n_encoded + 2048)
            shap_values = explainer.shap_values(sample, nsamples=nsamples, silent=True)

            # Get SHAP values for target class
            if isinstance(shap_values, list):
                vals = shap_values[target_class].flatten()
            else:
                vals = shap_values.flatten()

            # Build feature importance list
            names = feature_names if feature_names else [f"feature_{i}" for i in range(len(vals))]
            feature_importance = []
            for i, (name, val) in enumerate(zip(names, vals)):
                feature_importance.append({
                    "feature": name,
                    "feature_index": i,
                    "shap_value": round(float(val), 6),
                    "direction": "positive" if val > 0 else "negative"
                })

            # Aggregate one-hot features back to original feature names (adult datasets)
            encoded_to_original = self.data_model_loader.get_encoded_to_original()
            raw_features_dict = self.data_model_loader.get_raw_features_dict()
            if encoded_to_original is not None:
                feature_importance = _aggregate_encoded_to_original(
                    feature_importance, "shap_value", encoded_to_original, raw_features_dict
                )
            else:
                feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)

            result = {
                "success": True,
                "method": "SHAP (Tabular)",
                "target_class": target_class,
                "statistics": {
                    "num_features": len(feature_importance),
                    "feature_importance": feature_importance[:20],
                    "top_positive": max([f["shap_value"] for f in feature_importance if f["shap_value"] > 0], default=0),
                    "top_negative": min([f["shap_value"] for f in feature_importance if f["shap_value"] < 0], default=0)
                },
                "description": (
                    f"SHAP tabular analysis for class {target_class}: "
                    f"Analyzed {len(feature_importance)} features. "
                    f"Most influential: {feature_importance[0]['feature'] if feature_importance else 'N/A'}."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


class LIMETabularTool(BaseTool):
    """Tool for executing LIME analysis on tabular data."""

    name = "lime"
    description = (
        "Executes LIME (Local Interpretable Model-agnostic Explanations) for tabular data. "
        "Returns feature-level importance scores for the prediction."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, num_samples: int = 1000,
            image_id: str = "temp", **kwargs) -> str:
        """Execute LIME tabular analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        features = self.data_model_loader.get_current_features()
        if features is None:
            return json.dumps({"success": False, "error": "No tabular data available. Load a sample first."})

        feature_names = self.data_model_loader.get_feature_names()

        try:
            from lime import lime_tabular
            import numpy as np
            target_class = int(target_class) if target_class is not None else 0

            def predict_fn(X):
                """Predict function for LIME."""
                model.eval()
                with torch.no_grad():
                    input_t = torch.tensor(X, dtype=torch.float32).to(device)
                    if input_t.dim() == 1:
                        input_t = input_t.unsqueeze(0)
                    outputs = model(input_t)
                    if outputs.shape[-1] == 1:
                        prob_pos = outputs.cpu().numpy().flatten()
                        return np.column_stack([1.0 - prob_pos, prob_pos])
                    else:
                        return torch.softmax(outputs, dim=-1).cpu().numpy()

            # Use training data for LIME explainer
            training_data = self.data_model_loader.get_training_data()
            if training_data is not None:
                train_np = training_data.numpy()
            else:
                train_np = features.unsqueeze(0).cpu().numpy()

            names = feature_names if feature_names else [f"feature_{i}" for i in range(features.shape[0])]

            explainer = lime_tabular.LimeTabularExplainer(
                training_data=train_np,
                feature_names=names,
                mode='classification'
            )

            sample = features.cpu().numpy()
            explanation = explainer.explain_instance(
                sample,
                predict_fn,
                num_samples=num_samples,
                labels=(target_class,)
            )

            explanation_list = explanation.as_list(label=target_class)

            # Aggregate one-hot features back to original feature names (adult datasets)
            encoded_to_original = self.data_model_loader.get_encoded_to_original()
            raw_features_dict = self.data_model_loader.get_raw_features_dict()
            if encoded_to_original is not None:
                names_by_len = sorted(names, key=len, reverse=True)
                converted = []
                for feat_desc, weight in explanation_list:
                    base_name = _extract_feature_name_from_lime_condition(feat_desc, names_by_len)
                    converted.append({"feature": base_name, "weight": round(float(weight), 6)})
                feature_importance = _aggregate_encoded_to_original(
                    converted, "weight", encoded_to_original, raw_features_dict
                )
            else:
                names_by_len = sorted(names, key=len, reverse=True)
                feature_importance = []
                for feat_desc, weight in explanation_list[:20]:
                    clean_name = _extract_feature_name_from_lime_condition(feat_desc, names_by_len)
                    feature_importance.append({
                        "feature": clean_name,
                        "weight": round(float(weight), 6),
                        "direction": "positive" if weight > 0 else "negative"
                    })

            result = {
                "success": True,
                "method": "LIME (Tabular)",
                "target_class": target_class,
                "num_samples": num_samples,
                "statistics": {
                    "num_features": len(feature_importance),
                    "feature_importance": feature_importance,
                    "max_positive_weight": max([f["weight"] for f in feature_importance if f["weight"] > 0], default=0),
                    "max_negative_weight": min([f["weight"] for f in feature_importance if f["weight"] < 0], default=0)
                },
                "description": (
                    f"LIME tabular analysis for class {target_class}: "
                    f"Found {len(feature_importance)} important features. "
                    f"Top: {feature_importance[0]['feature'] if feature_importance else 'N/A'}."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


class IntegratedGradientsTabularTool(BaseTool):
    """Tool for executing Integrated Gradients analysis on tabular data."""

    name = "integrated_gradients"
    description = (
        "Executes Integrated Gradients for tabular data to compute per-feature attribution scores. "
        "Returns feature-level importance based on gradient integration from a zero baseline."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, n_steps: int = 50,
            image_id: str = "temp", **kwargs) -> str:
        """Execute Integrated Gradients for tabular data."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        features = self.data_model_loader.get_current_features()
        if features is None:
            return json.dumps({"success": False, "error": "No tabular data available. Load a sample first."})

        feature_names = self.data_model_loader.get_feature_names()

        try:
            from captum.attr import IntegratedGradients
            import numpy as np
            target_class = int(target_class) if target_class is not None else 0

            has_sigmoid = _model_has_final_sigmoid(model)

            def forward_func(input_tensor):
                logits = model(input_tensor)
                if logits.shape[-1] == 1:
                    prob_pos = logits if has_sigmoid else torch.sigmoid(logits)
                    return torch.cat([1.0 - prob_pos, prob_pos], dim=-1)
                return logits

            ig = IntegratedGradients(forward_func)

            input_tensor = features.unsqueeze(0).float().to(device).requires_grad_(True)
            baseline = torch.zeros_like(input_tensor).to(device)

            attributions = ig.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class,
                n_steps=n_steps
            )

            attr_scores = attributions.squeeze().cpu().detach().numpy()

            names = feature_names if feature_names else [f"feature_{i}" for i in range(len(attr_scores))]

            feature_importance = []
            for i, (name, score) in enumerate(zip(names, attr_scores)):
                feature_importance.append({
                    "feature": name,
                    "feature_index": i,
                    "attribution_score": round(float(score), 6),
                    "direction": "positive" if score > 0 else "negative"
                })

            # Aggregate one-hot features back to original feature names (adult datasets)
            encoded_to_original = self.data_model_loader.get_encoded_to_original()
            raw_features_dict = self.data_model_loader.get_raw_features_dict()
            if encoded_to_original is not None:
                feature_importance = _aggregate_encoded_to_original(
                    feature_importance, "attribution_score", encoded_to_original, raw_features_dict
                )
            else:
                feature_importance.sort(key=lambda x: abs(x['attribution_score']), reverse=True)

            result = {
                "success": True,
                "method": "Integrated Gradients (Tabular)",
                "target_class": target_class,
                "n_steps": n_steps,
                "statistics": {
                    "num_features": len(feature_importance),
                    "feature_importance": feature_importance[:20],
                    "max_positive": max([f["attribution_score"] for f in feature_importance if f["attribution_score"] > 0], default=0),
                    "max_negative": min([f["attribution_score"] for f in feature_importance if f["attribution_score"] < 0], default=0)
                },
                "description": (
                    f"Integrated Gradients tabular analysis for class {target_class}: "
                    f"Analyzed {len(feature_importance)} features. "
                    f"Most influential: {feature_importance[0]['feature'] if feature_importance else 'N/A'}."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


class SensitivityAnalysisTabularTool(BaseTool):
    """Tool for executing leave-one-out sensitivity analysis on tabular data."""

    name = "sensitivity_analysis"
    description = (
        "Executes leave-one-out feature ablation to measure each feature's impact on prediction. "
        "Returns probability changes when each feature is set to zero."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, image_id: str = "temp", **kwargs) -> str:
        """Execute leave-one-out sensitivity analysis for tabular data."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        features = self.data_model_loader.get_current_features()
        if features is None:
            return json.dumps({"success": False, "error": "No tabular data available. Load a sample first."})

        feature_names = self.data_model_loader.get_feature_names()

        try:
            import numpy as np
            target_class = int(target_class) if target_class is not None else 0

            has_sigmoid = _model_has_final_sigmoid(model)

            def get_prob(input_tensor):
                """Get target class probability."""
                model.eval()
                with torch.no_grad():
                    if input_tensor.dim() == 1:
                        input_tensor = input_tensor.unsqueeze(0)
                    logits = model(input_tensor.float().to(device))
                    if logits.shape[-1] == 1:
                        prob_pos = logits.item() if has_sigmoid else torch.sigmoid(logits).item()
                        probs = [1.0 - prob_pos, prob_pos]
                    else:
                        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
                    return probs[target_class]

            # Get original probability
            original_prob = get_prob(features)

            names = feature_names if feature_names else [f"feature_{i}" for i in range(features.shape[0])]

            encoded_to_original = self.data_model_loader.get_encoded_to_original()
            raw_features_dict = self.data_model_loader.get_raw_features_dict()

            feature_sensitivity = []
            if encoded_to_original is not None:
                # Group encoded feature indices by original feature and ablate together
                from collections import defaultdict
                orig_to_indices = defaultdict(list)
                for i, name in enumerate(names):
                    orig_col = encoded_to_original.get(name, name)
                    orig_to_indices[orig_col].append(i)

                for orig_col, indices in orig_to_indices.items():
                    modified = features.clone()
                    for idx in indices:
                        modified[idx] = 0.0
                    modified_prob = get_prob(modified)
                    prob_drop = original_prob - modified_prob
                    raw_val = raw_features_dict.get(orig_col, "?") if raw_features_dict else "?"
                    feature_sensitivity.append({
                        "feature": orig_col,
                        "value": raw_val,
                        "original_prob": round(float(original_prob), 6),
                        "modified_prob": round(float(modified_prob), 6),
                        "prob_drop": round(float(prob_drop), 6)
                    })
            else:
                # Leave-one-out: zero out each feature individually
                for i in range(features.shape[0]):
                    modified = features.clone()
                    modified[i] = 0.0
                    modified_prob = get_prob(modified)
                    prob_drop = original_prob - modified_prob
                    feature_sensitivity.append({
                        "feature": names[i],
                        "feature_index": i,
                        "original_prob": round(float(original_prob), 6),
                        "modified_prob": round(float(modified_prob), 6),
                        "prob_drop": round(float(prob_drop), 6)
                    })

            # Sort by absolute prob_drop
            feature_sensitivity.sort(key=lambda x: abs(x['prob_drop']), reverse=True)

            result = {
                "success": True,
                "method": "Sensitivity Analysis (Tabular)",
                "target_class": target_class,
                "original_prob": round(float(original_prob), 6),
                "statistics": {
                    "num_features": len(feature_sensitivity),
                    "feature_sensitivity": feature_sensitivity[:20],
                    "max_prob_drop": max([f["prob_drop"] for f in feature_sensitivity], default=0),
                    "max_prob_increase": min([f["prob_drop"] for f in feature_sensitivity], default=0)
                },
                "description": (
                    f"Sensitivity analysis for class {target_class}: "
                    f"Ablated {len(feature_sensitivity)} features. "
                    f"Most sensitive: {feature_sensitivity[0]['feature'] if feature_sensitivity else 'N/A'} "
                    f"(prob drop: {feature_sensitivity[0]['prob_drop'] if feature_sensitivity else 0})."
                )
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ===================================================================
# GuidedBackprop for Text
# ===================================================================

class GuidedBackpropTextTool(BaseTool):
    """Tool for executing Guided Backpropagation analysis on text data."""

    name = "guided_backprop"
    description = (
        "Executes Guided Backpropagation on text to compute per-token importance using modified "
        "ReLU gradient rules. Returns attribution scores for each token without path integration. "
        "Requires the model to have ReLU activations."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, image_id: str = "temp", **kwargs) -> str:
        """Execute Guided Backpropagation for text."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        sample_data = self.data_model_loader.current_sample_data or {}
        is_nli = 'hypothesis_tensor' in sample_data and 'hypothesis' in sample_data
        text = sample_data.get('premise', '') if is_nli else self.data_model_loader.get_current_text()

        if not text:
            return json.dumps({"success": False, "error": "No text available. Load a sample first."})
        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        try:
            result = execute_guided_backprop_text(
                text=text,
                model=model,
                processor=processor,
                target_class=int(target_class),
                device=device,
                instance_id=image_id,
                sample_data=sample_data
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ===================================================================
# GuidedBackprop for Tabular
# ===================================================================

class GuidedBackpropTabularTool(BaseTool):
    """Tool for executing Guided Backpropagation analysis on tabular data."""

    name = "guided_backprop"
    description = (
        "Executes Guided Backpropagation on tabular features using modified ReLU gradient rules. "
        "Returns per-feature attribution scores. Requires the model to have ReLU activations."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(self, target_class: Optional[int] = None, image_id: str = "temp", **kwargs) -> str:
        """Execute Guided Backpropagation for tabular data."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})

        features = self.data_model_loader.get_current_features()
        if features is None:
            return json.dumps({"success": False, "error": "No tabular data available. Load a sample first."})
        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        try:
            result = execute_guided_backprop_tabular(
                features=features,
                model=model,
                feature_names=self.data_model_loader.get_feature_names(),
                target_class=int(target_class),
                device=device,
                encoded_to_original=self.data_model_loader.get_encoded_to_original(),
                raw_features_dict=self.data_model_loader.get_raw_features_dict()
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ===================================================================
# SmoothGrad Tool — Vision
# ===================================================================

class SmoothGradTool(BaseTool):
    """Tool for executing SmoothGrad analysis on images."""

    name = "smoothgrad"
    description = (
        "Executes SmoothGrad (NoiseTunnel + Saliency) to produce noise-reduced saliency maps. "
        "Averages gradients over multiple noisy copies of the input image, yielding smoother "
        "and more stable importance maps than standard gradient methods. "
        "The Actor Agent will analyze the visualization to identify important regions."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(
        self,
        target_class: Optional[int] = None,
        image_id: str = "temp",
        n_samples: int = 100,
        stdevs: float = 0.1,
        **kwargs
    ) -> str:
        """Execute SmoothGrad analysis."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device
        model_type = self.data_model_loader.model_name

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})
        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        image = self.data_model_loader.get_display_image()
        if image is None:
            return json.dumps({"success": False, "error": "No image available. Load a sample first."})

        try:
            input_tensor = self.data_model_loader.get_current_tensor()
            result = execute_smoothgrad(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=int(target_class),
                device=device,
                image_id=image_id,
                n_samples=n_samples,
                stdevs=stdevs,
                input_tensor=input_tensor
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


# ===================================================================
# SmoothGrad for Text
# ===================================================================

class SmoothGradTextTool(BaseTool):
    """Tool for executing SmoothGrad analysis on text data."""

    name = "smoothgrad"
    description = (
        "Executes SmoothGrad on text by averaging saliency gradients over multiple noisy "
        "embedding perturbations. Produces stable, noise-reduced per-token importance scores."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(
        self,
        target_class: Optional[int] = None,
        image_id: str = "temp",
        n_samples: int = 100,
        stdevs: float = 0.1,
        **kwargs
    ) -> str:
        """Execute SmoothGrad for text (CNN-compatible, supports single-input and NLI dual-input)."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        processor = self.data_model_loader.get_processor()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})
        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        sample_data = self.data_model_loader.current_sample_data or {}
        is_nli = 'hypothesis_tensor' in sample_data and 'hypothesis' in sample_data
        is_dual_input = is_nli and len(inspect.signature(model.forward).parameters) >= 2
        text = sample_data.get('premise', '') if is_nli else self.data_model_loader.get_current_text()

        if not text:
            return json.dumps({"success": False, "error": "No text available. Load a sample first."})

        try:
            import numpy as np
            import re
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from xai_tools import get_output_dir

            target_class = int(target_class)

            # --- tokenize ---
            if is_nli:
                token_ids = sample_data.get('premise_ids') or processor(text)
            else:
                token_ids = self.data_model_loader.get_current_token_ids() or processor(text)

            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

            if is_nli:
                hypothesis_ids = sample_data.get('hypothesis_ids')
                if hypothesis_ids is None and processor is not None:
                    hypothesis_ids = processor(sample_data.get('hypothesis', ''))
                hyp_tensor = torch.tensor([hypothesis_ids], dtype=torch.long).to(device)

            # --- find embedding layer ---
            embedding_layer = None
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    embedding_layer = module
                    break

            if embedding_layer is None:
                return json.dumps({"success": False, "error": "No embedding layer found in model"})

            # --- SmoothGrad: average |grad| over n_samples noisy copies ---
            model.eval()
            base_embeddings = embedding_layer(input_tensor).detach()  # (1, seq_len, embed_dim)

            all_grads = []
            for _ in range(n_samples):
                noise = torch.randn_like(base_embeddings) * stdevs
                noisy_embeddings = (base_embeddings + noise).requires_grad_(True)

                # Hook replaces embedding output with noisy_embeddings
                if is_dual_input:
                    call_count = [0]
                    def _hook(module, inp, out, _ne=noisy_embeddings, _cc=call_count):
                        _cc[0] += 1
                        return _ne if _cc[0] == 1 else out
                    handle = embedding_layer.register_forward_hook(_hook)
                    logits = model(input_tensor, hyp_tensor)
                    handle.remove()
                else:
                    def _hook(module, inp, out, _ne=noisy_embeddings):
                        return _ne
                    handle = embedding_layer.register_forward_hook(_hook)
                    logits = model(input_tensor)
                    handle.remove()

                if logits.shape[-1] == 1:
                    score = logits[0, 0] if target_class == 1 else -logits[0, 0]
                else:
                    score = logits[0, target_class]

                score.backward()
                # grad×input: position-specific even for mean-pool models where plain |grad| is uniform
                all_grads.append(
                    (noisy_embeddings.grad * noisy_embeddings.detach()).squeeze(0).abs().mean(dim=-1).cpu().detach().numpy()
                )

            importance = np.mean(all_grads, axis=0)  # (seq_len,)

            if importance.max() == importance.min():
                raise RuntimeError(
                    f"SmoothGrad (Text) attribution is uniform (all={importance.max():.6f}) "
                    f"for class {target_class}. Gradients may be zero."
                )
            importance = (importance - importance.min()) / (importance.max() - importance.min())

            # --- map scores to words ---
            words = re.sub(r'<br\s*/?>', ' ', text.lower())
            words = re.sub(r'[^a-z0-9\s]', ' ', words)
            word_list = words.split()

            token_importance = []
            for i, word in enumerate(word_list[:len(importance)]):
                score = float(importance[i]) if i < len(importance) else 0.0
                token_importance.append({"token": word, "importance": round(score, 4)})
            token_importance_sorted = sorted(token_importance, key=lambda x: x["importance"], reverse=True)

            # --- visualization ---
            output_dir = get_output_dir()
            viz_path = str(output_dir / f"smoothgrad_text_{image_id}.png")
            display_tokens = [t["token"] for t in token_importance]
            display_scores = [t["importance"] for t in token_importance]
            fig, ax = plt.subplots(figsize=(max(8, len(display_tokens) * 0.4), 3))
            ax.bar(range(len(display_tokens)), display_scores)
            ax.set_xticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=90, fontsize=8)
            ax.set_title(f"SmoothGrad (Text) - Class {target_class} ({n_samples} samples)")
            ax.set_ylabel("Importance")
            plt.tight_layout()
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            return json.dumps({
                "success": True,
                "method": "SmoothGrad (Text)",
                "target_class": target_class,
                "n_samples": n_samples,
                "stdevs": stdevs,
                "visualization_path": viz_path,
                "statistics": {
                    "num_tokens": len(token_importance),
                    "top_tokens": token_importance_sorted[:10],
                    "mean_importance": round(float(np.mean(display_scores)), 4),
                    "max_importance": round(float(max(display_scores)) if display_scores else 0.0, 4),
                },
                "description": (
                    f"SmoothGrad text analysis ({n_samples} samples) for class {target_class}: "
                    f"Analyzed {len(token_importance)} tokens. "
                    f"Most important: {token_importance_sorted[0]['token'] if token_importance_sorted else 'N/A'}."
                ),
            }, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ===================================================================
# SmoothGrad for Tabular
# ===================================================================

class SmoothGradTabularTool(BaseTool):
    """Tool for executing SmoothGrad analysis on tabular data."""

    name = "smoothgrad"
    description = (
        "Executes SmoothGrad on tabular data by averaging saliency gradients over multiple "
        "noisy feature perturbations. Produces stable, noise-reduced per-feature importance scores."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        self.data_model_loader = data_model_loader

    def run(
        self,
        target_class: Optional[int] = None,
        image_id: str = "temp",
        n_samples: int = 100,
        stdevs: float = 0.1,
        **kwargs
    ) -> str:
        """Execute SmoothGrad for tabular data."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        model = self.data_model_loader.get_model()
        device = self.data_model_loader.device

        if model is None:
            return json.dumps({"success": False, "error": "Model not loaded"})
        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required"})

        features = self.data_model_loader.get_current_features()
        if features is None:
            return json.dumps({"success": False, "error": "No tabular data available. Load a sample first."})

        try:
            result = execute_smoothgrad_tabular(
                features=features,
                model=model,
                feature_names=self.data_model_loader.get_feature_names(),
                target_class=int(target_class),
                device=device,
                n_samples=n_samples,
                stdevs=stdevs,
                encoded_to_original=self.data_model_loader.get_encoded_to_original(),
                raw_features_dict=self.data_model_loader.get_raw_features_dict()
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            import traceback
            return json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})


# ===================================================================
# Tool Registry (Replaces LangChain Tool Management)
# ===================================================================

class XAIToolRegistry:
    """
    Registry for XAI Tools.

    Manages initialization and access to all XAI tools.
    Tools return raw results - feature extraction is done by Actor Agent.
    """

    def __init__(
        self,
        modality: Optional[str] = None,
        data_model_loader: Optional[Any] = None
    ):
        """
        Initialize tool registry.

        Args:
            model: PyTorch model
            model_type: Model type ('local_pth', 'timm', etc.)
            processor: Image preprocessor
            modality: Data modality ('vision', 'text', 'tabular')
            device: Torch device
            data_model_loader: DataModelLoader instance for accessing loaded images
        """
        self.modality = modality
        self.data_model_loader = data_model_loader
        self.device = self.data_model_loader.device if self.data_model_loader else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tools: Dict[str, BaseTool] = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize all available tools."""
        available_tools = get_available_tools()
        tool_context = {
            "data_model_loader": self.data_model_loader
        }

        if self.modality == 'text':
            # Text-specific tools
            self._tools['lime'] = LIMETextTool(**tool_context)
            self._tools['integrated_gradients'] = IntegratedGradientsTextTool(**tool_context)
            self._tools['shap'] = SHAPTextTool(**tool_context)
            self._tools['sensitivity_analysis'] = SensitivityAnalysisTextTool(**tool_context)
            # Modality-agnostic tools
            self._tools['guided_backprop'] = GuidedBackpropTextTool(**tool_context)
            self._tools['smoothgrad'] = SmoothGradTextTool(**tool_context)
            return

        if self.modality == 'tabular':
            # Tabular-specific tools
            self._tools['shap'] = SHAPTabularTool(**tool_context)
            self._tools['lime'] = LIMETabularTool(**tool_context)
            self._tools['integrated_gradients'] = IntegratedGradientsTabularTool(**tool_context)
            self._tools['sensitivity_analysis'] = SensitivityAnalysisTabularTool(**tool_context)
            # Modality-agnostic tools
            self._tools['guided_backprop'] = GuidedBackpropTabularTool(**tool_context)
            self._tools['smoothgrad'] = SmoothGradTabularTool(**tool_context)
            return

        # Vision-specific tools
        if self.modality == 'vision' or self.modality is None:
            if 'gradcam' in available_tools:
                self._tools['gradcam'] = GradCAMTool(**tool_context)
            if 'guided_backprop' in available_tools:
                self._tools['guided_backprop'] = GuidedBackpropTool(**tool_context)
            if 'smoothgrad' in available_tools:
                self._tools['smoothgrad'] = SmoothGradTool(**tool_context)

        # Vision multi-modal tools
        if 'integrated_gradients' in available_tools:
            self._tools['integrated_gradients'] = IntegratedGradientsTool(**tool_context)
        if 'lime' in available_tools:
            self._tools['lime'] = LIMETool(**tool_context)
        if 'shap' in available_tools:
            self._tools['shap'] = SHAPTool(**tool_context)



    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.

        Handles modality-suffixed aliases the proposer may generate, e.g.
        'shap_tabular' → 'shap', 'lime_vision' → 'lime'.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not available
        """
        tool = self._tools.get(tool_name)
        if tool is not None:
            return tool
        # Strip known modality suffixes and retry
        for suffix in ("_tabular", "_vision", "_text", "_image"):
            if tool_name.endswith(suffix):
                canonical = tool_name[: -len(suffix)]
                tool = self._tools.get(canonical)
                if tool is not None:
                    return tool
        return None

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all available tools.

        Returns:
            List of tool instances
        """
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """
        Get names of all available tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tools_by_names(self, tool_names: List[str]) -> List[BaseTool]:
        """
        Get specific tools by their names.

        Args:
            tool_names: List of tool names to retrieve

        Returns:
            List of available tool instances
        """
        return [tool for name in tool_names if (tool := self.get_tool(name)) is not None]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all available tools.

        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {name: tool.description for name, tool in self._tools.items()}

    def execute_tool(
        self,
        tool_name: str,
        image_path: Optional[str] = None, # Make image_path optional here
        target_class: Optional[int] = None,
        image_id: str = "temp",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name and return parsed results.

        Args:
            tool_name: Name of the tool
            image_path: Path to input image
            target_class: Target class index
            image_id: Identifier for output files
            **kwargs: Additional tool-specific arguments

        Returns:
            Parsed result dictionary
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}

        result_str = tool.run(
            image_path=image_path,
            target_class=target_class,
            image_id=image_id,
            **kwargs
        )

        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse tool result"}


def create_xai_tools(
    data_model_loader: Any,
    output_dir: str = "./outputs"
) -> XAIToolRegistry:
    """
    Create XAI tool registry with a data model loader.

    Args:
        data_model_loader: An instance of DataModelLoader.
        output_dir: Output directory for visualizations

    Returns:
        XAIToolRegistry instance
    """
    # Set the output directory for XAI visualizations
    set_output_dir(output_dir)

    # Create tool registry from the loader
    registry = XAIToolRegistry(
        modality=data_model_loader.modality,
        data_model_loader=data_model_loader
    )

    return registry


# Test code
if __name__ == "__main__":
    print("Testing XAI Tools Native...")

    # Test tool registry without model
    registry = XAIToolRegistry()
    print(f"Available tools: {registry.get_tool_names()}")
    print(f"Tool descriptions: {registry.get_tool_descriptions()}")

    print("\nXAI Tools Native test complete!")
