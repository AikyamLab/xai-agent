"""
XAI Tools Native Implementation (No LangChain Dependency)

Provides XAI tool wrappers and registry without LangChain.
Uses simple Python ABC for tool base class.

IMPORTANT: Tools return RAW results (visualizations, statistics, descriptions).
Feature extraction (bounding boxes, importance regions) is done by the Actor Agent
through VLM reasoning on the visualization outputs.
"""

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
    execute_object_detection,
    execute_guided_backprop,
    execute_layer_cam,
    execute_sensitivity_analysis,
    set_output_dir,
    get_output_dir
)


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
    n_steps: int = Field(default=50, description="Number of integration steps")
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
    num_samples: int = Field(default=100, description="Number of samples for SHAP")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'num_samples', 'image_id'])


class ObjectDetectionInput(BaseModel):
    """Input schema for Object Detection tool."""
    image_path: str = Field(description="Path to the image file")
    confidence_threshold: float = Field(default=0.25, description="Confidence threshold")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'confidence_threshold', 'image_id'])


class GuidedBackpropInput(BaseModel):
    """Input schema for Guided Backpropagation tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'image_id'])


class LayerCAMInput(BaseModel):
    """Input schema for Layer CAM tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    layer_name: Optional[str] = Field(default=None, description="Specific layer name (optional)")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'layer_name', 'image_id'])


class SensitivityAnalysisInput(BaseModel):
    """Input schema for Sensitivity Analysis tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    perturbation_type: str = Field(default="noise", description="Type of perturbation: 'noise' or 'blur'")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'perturbation_type', 'image_id'])


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
            image = self.data_model_loader.get_current_image()

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
        n_steps: int = 50,
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
            image = self.data_model_loader.get_current_image()

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
            image = self.data_model_loader.get_current_image()

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
        num_samples: int = 100,
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
            image = self.data_model_loader.get_current_image()

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


class ObjectDetectionTool(BaseTool):
    """Tool for executing object detection."""

    name = "object_detection"
    description = (
        "Executes YOLO object detection to find objects in the image. "
        "Returns detected objects with their bounding boxes and confidence scores. "
        "Useful for understanding what objects are present in the image."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """Initialize Object Detection tool (no model context needed)."""
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute Object Detection."""
        if not self.data_model_loader:
            return json.dumps({"success": False, "error": "DataModelLoader not initialized"})

        try:
            image = self.data_model_loader.get_current_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })

            result = execute_object_detection(
                image=image,
                image_id=image_id,
                confidence_threshold=confidence_threshold
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class SensitivityAnalysisTool(BaseTool):
    """Tool for executing Sensitivity Analysis."""

    name = "sensitivity_analysis"
    description = (
        "Executes Sensitivity Analysis to test model robustness to perturbations. "
        "Returns probability changes under different perturbation levels. "
        "Useful for understanding model stability and reliability."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """
        Initialize Sensitivity Analysis tool.

        Args:
            data_model_loader: DataModelLoader instance for accessing the model and data.
        """
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        target_class: Optional[int] = None,
        perturbation_type: str = "noise",
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute Sensitivity Analysis."""
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
            image = self.data_model_loader.get_current_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })

            # Get pre-processed tensor if available
            input_tensor = self.data_model_loader.get_current_tensor()

            result = execute_sensitivity_analysis(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=target_class,
                device=device,
                image_id=image_id,
                perturbation_type=perturbation_type,
                input_tensor=input_tensor
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
            image = self.data_model_loader.get_current_image()

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


class LayerCAMTool(BaseTool):
    """Tool for executing Layer CAM analysis."""

    name = "layer_cam"
    description = (
        "Executes Layer CAM to visualize class-specific activation maps at different layers. "
        "Returns a heatmap showing which regions activate for the target class. "
        "The Actor Agent will analyze the visualization to identify important regions."
    )

    def __init__(self, data_model_loader: Optional[Any] = None):
        """
        Initialize Layer CAM tool.

        Args:
            data_model_loader: DataModelLoader instance for accessing the model and data.
        """
        self.data_model_loader = data_model_loader

    def run(
        self,
        image_path: Optional[str] = None,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute Layer CAM analysis."""
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
            image = self.data_model_loader.get_current_image()

            if image is None:
                return json.dumps({
                    "success": False,
                    "error": "No image available. Load a sample with data_model_loader.load_sample() first."
                })

            # Get pre-processed tensor if available
            input_tensor = self.data_model_loader.get_current_tensor()

            result = execute_layer_cam(
                image=image,
                model=model,
                model_type=model_type,
                processor=processor,
                target_class=target_class,
                device=device,
                image_id=image_id,
                layer_name=layer_name,
                input_tensor=input_tensor
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


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

        # Vision-specific tools
        if self.modality == 'vision' or self.modality is None:
            if 'gradcam' in available_tools:
                self._tools['gradcam'] = GradCAMTool(**tool_context)
            if 'object_detection' in available_tools:
                self._tools['object_detection'] = ObjectDetectionTool(**tool_context)
            if 'guided_backprop' in available_tools:
                self._tools['guided_backprop'] = GuidedBackpropTool(**tool_context)
            if 'layer_cam' in available_tools:
                self._tools['layer_cam'] = LayerCAMTool(**tool_context)
            if 'sensitivity_analysis' in available_tools:
                self._tools['sensitivity_analysis'] = SensitivityAnalysisTool(**tool_context)

        # Multi-modal tools (work for vision)
        if 'integrated_gradients' in available_tools:
            self._tools['integrated_gradients'] = IntegratedGradientsTool(**tool_context)
        if 'lime' in available_tools:
            self._tools['lime'] = LIMETool(**tool_context)
        if 'shap' in available_tools:
            self._tools['shap'] = SHAPTool(**tool_context)



    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not available
        """
        return self._tools.get(tool_name)

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
