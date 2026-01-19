"""
LangChain Tool Wrappers for XAI Tools

Wraps XAI tools (GradCAM, LIME, SHAP, etc.) as LangChain Tools
for use in LangChain Agent framework.
"""

from typing import Any, Dict, Optional, Type
from pathlib import Path
import json
import torch

from typing import Any, Dict, Optional, Type, Union
from pathlib import Path
import json
import torch
import pandas as pd

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from xai_tools import (
    get_available_tools,
    execute_gradcam,
    execute_integrated_gradients,
    execute_lime,
    # TODO: Add execute_lime_text, execute_lime_tabular
    execute_shap,
    # TODO: Add execute_shap_text, execute_shap_tabular
    execute_object_detection,
    set_output_dir,
    get_output_dir
)


# ===================================================================
# Input Schemas for Tools
# ===================================================================

class VisionXAIToolInput(BaseModel):
    """Base input schema for vision-based XAI tools."""
    image_path: str = Field(description="Path to the image file for explanation.")
    target_class: int = Field(description="Target class index for explanation.")
    instance_id: str = Field(default="temp", description="Identifier for output files.")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        """
        Handle cases where the input is malformed - e.g., when the entire JSON string
        is incorrectly passed as the image_path value.

        This happens when LangChain's output parser incorrectly parses the Action Input.
        For example, input might come as:
        {'image_path': '{"image_path": "/path/to/image.png", "target_class": 1}'}
        instead of:
        {'image_path': '/path/to/image.png', 'target_class': 1}
        """
        if not isinstance(data, dict):
            return data

        image_path = data.get('image_path', '')

        # Check if image_path contains a nested JSON string
        if isinstance(image_path, str) and image_path.strip().startswith('{'):
            try:
                # Try to parse the nested JSON
                import re
                json_match = re.search(r'\{[^{}]*\}', image_path)
                if json_match:
                    nested_data = json.loads(json_match.group())
                    # Extract fields from the nested JSON
                    if 'image_path' in nested_data:
                        data['image_path'] = nested_data['image_path']
                    if 'target_class' in nested_data:
                        data['target_class'] = nested_data['target_class']
                    if 'instance_id' in nested_data:
                        data['instance_id'] = nested_data['instance_id']
            except (json.JSONDecodeError, AttributeError):
                # If parsing fails, keep original data
                pass

        return data

class GradCAMInput(VisionXAIToolInput):
    """Input for GradCAM tool."""
    pass

class IntegratedGradientsInput(VisionXAIToolInput):
    """Input for Integrated Gradients tool."""
    n_steps: int = Field(default=50, description="Number of integration steps.")

class LIMEInput(VisionXAIToolInput):
    """Input for LIME tool."""
    num_samples: int = Field(default=1000, description="Number of samples for LIME.")

class SHAPInput(VisionXAIToolInput):
    """Input for SHAP tool."""
    num_samples: int = Field(default=100, description="Number of samples for SHAP.")

class ObjectDetectionInput(BaseModel):
    """Input for Object Detection tool (vision-only)."""
    image_path: str = Field(description="Path to the image file.")
    confidence_threshold: float = Field(default=0.25, description="Confidence threshold.")
    image_id: str = Field(default="temp", description="Identifier for output files.")


# ===================================================================
# LangChain Tool Implementations
# ===================================================================

class GradCAMTool(BaseTool):
    """Tool for executing GradCAM analysis. VISION-ONLY."""

    name: str = "gradcam"
    description: str = (
        "Executes GradCAM to visualize where a VISION model focuses. "
        "REQUIRED INPUT: {'image_path': '/path/to/image.jpg', 'target_class': 1} "
        "Returns bounding_boxes with importance scores."
    )
    args_schema: Type[BaseModel] = GradCAMInput
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(self, image_path: str, target_class: int, instance_id: str = "temp") -> str:
        """Execute GradCAM."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized."})

        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            result = execute_gradcam(
                image=image, model=self.model, model_type=self.model_type,
                processor=self.processor, target_class=target_class,
                device=self.device, image_id=instance_id
            )
            return json.dumps(result, indent=2)
        except FileNotFoundError:
            return json.dumps({"success": False, "error": f"Image file not found: {image_path}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

class IntegratedGradientsTool(BaseTool):
    """Tool for executing Integrated Gradients analysis."""

    name: str = "integrated_gradients"
    description: str = (
        "Executes Integrated Gradients for pixel-level attribution. "
        "REQUIRED INPUT: {'image_path': '/path/to/image.jpg', 'target_class': 1, 'n_steps': 50}"
    )
    args_schema: Type[BaseModel] = IntegratedGradientsInput
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(self, image_path: str, target_class: int, n_steps: int = 50, instance_id: str = "temp") -> str:
        """Execute Integrated Gradients."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized."})

        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            result = execute_integrated_gradients(
                image=image, model=self.model, model_type=self.model_type,
                processor=self.processor, target_class=target_class,
                device=self.device, image_id=instance_id, n_steps=n_steps
            )
            return json.dumps(result, indent=2)
        except FileNotFoundError:
            return json.dumps({"success": False, "error": f"Image file not found: {image_path}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

class LIMETool(BaseTool):
    """Tool for executing LIME analysis."""

    name: str = "lime"
    description: str = (
        "Executes LIME for local interpretable explanations. "
        "REQUIRED INPUT: {'image_path': '/path/to/image.jpg', 'target_class': 1, 'num_samples': 1000}"
    )
    args_schema: Type[BaseModel] = LIMEInput
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(self, image_path: str, target_class: int, num_samples: int = 1000, instance_id: str = "temp") -> str:
        """Execute LIME."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized."})

        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            result = execute_lime(
                image=image, model=self.model, model_type=self.model_type,
                processor=self.processor, target_class=target_class,
                device=self.device, image_id=instance_id, num_samples=num_samples
            )
            return json.dumps(result, indent=2)
        except FileNotFoundError:
            return json.dumps({"success": False, "error": f"Image file not found: {image_path}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

class SHAPTool(BaseTool):
    """Tool for executing SHAP analysis."""

    name: str = "shap"
    description: str = (
        "Executes SHAP for game-theoretic feature importance. "
        "REQUIRED INPUT: {'image_path': '/path/to/image.jpg', 'target_class': 1, 'num_samples': 100}"
    )
    args_schema: Type[BaseModel] = SHAPInput
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(self, image_path: str, target_class: int, num_samples: int = 100, instance_id: str = "temp") -> str:
        """Execute SHAP."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized."})

        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            result = execute_shap(
                image=image, model=self.model, model_type=self.model_type,
                processor=self.processor, target_class=target_class,
                device=self.device, image_id=instance_id, num_samples=num_samples
            )
            return json.dumps(result, indent=2)
        except FileNotFoundError:
            return json.dumps({"success": False, "error": f"Image file not found: {image_path}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

class ObjectDetectionTool(BaseTool):
    """Tool for executing object detection. VISION-ONLY."""

    name: str = "object_detection"
    description: str = (
        "Executes YOLO object detection on an image. "
        "REQUIRED INPUT: {'image_path': '/path/to/image.jpg', 'confidence_threshold': 0.25} "
        "Returns bounding_boxes with detected objects and confidence."
    )
    args_schema: Type[BaseModel] = ObjectDetectionInput

    def _run(self, image_path: str, confidence_threshold: float = 0.25, image_id: str = "temp") -> str:
        """Execute Object Detection."""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            result = execute_object_detection(
                image=image, image_id=image_id, confidence_threshold=confidence_threshold
            )
            return json.dumps(result, indent=2)
        except FileNotFoundError:
            return json.dumps({"success": False, "error": f"Image file not found: {image_path}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


# ===================================================================
# Tool Registry
# ===================================================================

class XAIToolRegistry:
    """Registry for XAI Tools, managing initialization and context."""

    def __init__(
        self,
        model: Optional[Any] = None,
        model_type: Optional[str] = None,
        processor: Optional[Any] = None,
        modality: Optional[str] = None, # Added modality
        device: Optional[torch.device] = None
    ):
        """Initialize tool registry."""
        self.model = model
        self.model_type = model_type
        self.processor = processor
        self.modality = modality
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tools = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize all available tools and set their context."""
        tool_context = {
            "model": self.model,
            "model_type": self.model_type,
            "processor": self.processor,
            "device": self.device
        }

        # Vision-specific tools
        if self.modality == 'vision':
            if 'gradcam' in get_available_tools():
                self._tools['gradcam'] = GradCAMTool(**tool_context)
            if 'object_detection' in get_available_tools():
                self._tools['object_detection'] = ObjectDetectionTool()

        # Multi-modal tools
        if 'integrated_gradients' in get_available_tools():
            self._tools['integrated_gradients'] = IntegratedGradientsTool(**tool_context)
        if 'lime' in get_available_tools():
            self._tools['lime'] = LIMETool(**tool_context)
        if 'shap' in get_available_tools():
            self._tools['shap'] = SHAPTool(**tool_context)

    def set_model_context(
        self,
        model: Any,
        model_type: str,
        processor: Any,
        modality: str,
        device: Optional[torch.device] = None
    ):
        """Update the model context for all registered tools."""
        self.model = model
        self.model_type = model_type
        self.processor = processor
        self.modality = modality
        self.device = device or self.device

        # Re-initialize tools with the new context
        self._initialize_tools()

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        return self._tools.get(tool_name)

    def get_all_tools(self) -> list[BaseTool]:
        """Get all available tools for the current modality."""
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Get names of all available tools for the current modality."""
        return list(self._tools.keys())

    def get_tools_by_names(self, tool_names: list[str]) -> list[BaseTool]:
        """Get specific tools by their names if available."""
        return [tool for name in tool_names if (tool := self.get_tool(name)) is not None]


def create_xai_tools(
    model: Any,
    model_type: str,
    processor: Any,
    modality: str,
    output_dir: str = "./outputs"
) -> tuple[None, XAIToolRegistry]:
    """
    Create XAI tool registry with model and modality context.
    """
    # Use model's device if available, otherwise default
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_output_dir(output_dir)

    registry = XAIToolRegistry(
        model=model,
        model_type=model_type,
        processor=processor,
        modality=modality,
        device=device
    )

    return None, registry
from pydantic import BaseModel, Field, model_validator

from xai_tools import (
    get_available_tools,
    execute_gradcam,
    execute_integrated_gradients,
    execute_lime,
    execute_shap,
    execute_object_detection,
    set_output_dir,
    get_output_dir
)


def _parse_nested_json_input(data, field_names):
    """
    Common helper to parse nested JSON input for XAI tools.

    Handles cases where the input is malformed - e.g., when the entire JSON string
    is incorrectly passed as the first field's value.
    """
    if not isinstance(data, dict):
        return data

    # Check the first field (usually image_path) for nested JSON
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


# Input schemas for each tool
class GradCAMInput(BaseModel):
    """Input for GradCAM tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'image_id'])


class IntegratedGradientsInput(BaseModel):
    """Input for Integrated Gradients tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    n_steps: int = Field(default=50, description="Number of integration steps")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'n_steps', 'image_id'])


class LIMEInput(BaseModel):
    """Input for LIME tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    num_samples: int = Field(default=1000, description="Number of samples for LIME")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'num_samples', 'image_id'])


class SHAPInput(BaseModel):
    """Input for SHAP tool."""
    image_path: str = Field(description="Path to the image file")
    target_class: int = Field(description="Target class index for explanation")
    num_samples: int = Field(default=100, description="Number of samples for SHAP")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'target_class', 'num_samples', 'image_id'])


class ObjectDetectionInput(BaseModel):
    """Input for Object Detection tool."""
    image_path: str = Field(description="Path to the image file")
    confidence_threshold: float = Field(default=0.25, description="Confidence threshold")
    image_id: str = Field(default="temp", description="Identifier for output files")

    @model_validator(mode='before')
    @classmethod
    def parse_nested_json(cls, data):
        return _parse_nested_json_input(data, ['image_path', 'confidence_threshold', 'image_id'])


def _clean_tool_input(input_data: Any) -> dict:
    """Clean and validate tool input, handling malformed inputs."""
    if isinstance(input_data, str):
        # Try to parse as JSON
        try:
            # Remove any non-JSON prefix/suffix
            import re
            json_match = re.search(r'\{[^{}]*\}', input_data)
            if json_match:
                return json.loads(json_match.group())
            return {"raw_input": input_data}
        except json.JSONDecodeError:
            return {"raw_input": input_data}
    elif isinstance(input_data, dict):
        return input_data
    else:
        return {"raw_input": str(input_data)}


# LangChain Tool implementations
class GradCAMTool(BaseTool):
    """Tool for executing GradCAM analysis."""

    name: str = "gradcam"
    description: str = (
        "Executes GradCAM to visualize which image regions the model focuses on. "
        "REQUIRED INPUT: {\"image_path\": \"/path/to/image.jpg\", \"target_class\": 1} "
        "Returns bounding_boxes with actual coordinates."
    )
    args_schema: Type[BaseModel] = GradCAMInput

    # Model context
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(self, image_path: str = None, target_class: int = None, image_id: str = "temp", **kwargs) -> str:
        """Execute GradCAM with robust input handling."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized"})

        # Handle case where input is passed as a single string
        if image_path is None and kwargs:
            cleaned = _clean_tool_input(kwargs)
            image_path = cleaned.get("image_path")
            target_class = cleaned.get("target_class")
            image_id = cleaned.get("image_id", "temp")

        # Validate required fields
        if image_path is None:
            return json.dumps({"success": False, "error": "image_path is required. Use format: {\"image_path\": \"/path/to/image.jpg\", \"target_class\": 1}"})
        if target_class is None:
            return json.dumps({"success": False, "error": "target_class is required. Use format: {\"image_path\": \"/path/to/image.jpg\", \"target_class\": 1}"})

        try:
            from PIL import Image
            # Clean image path (remove any extra quotes or whitespace)
            image_path = str(image_path).strip().strip('"').strip("'")
            target_class = int(target_class)

            image = Image.open(image_path).convert('RGB')
            result = execute_gradcam(
                image=image,
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                target_class=target_class,
                device=self.device,
                image_id=image_id
            )
            return json.dumps(result, indent=2)
        except FileNotFoundError:
            return json.dumps({"success": False, "error": f"Image file not found: {image_path}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _arun(self, *args, **kwargs) -> str:
        """Async version (not implemented)."""
        raise NotImplementedError("GradCAM does not support async execution")


class IntegratedGradientsTool(BaseTool):
    """Tool for executing Integrated Gradients analysis."""

    name: str = "integrated_gradients"
    description: str = (
        "Executes Integrated Gradients for pixel-level attribution. "
        "REQUIRED INPUT: {\"image_path\": \"/path/to/image.jpg\", \"target_class\": 1} "
        "Returns bounding_boxes with actual coordinates."
    )
    args_schema: Type[BaseModel] = IntegratedGradientsInput

    # Model context
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(
        self,
        image_path: str = None,
        target_class: int = None,
        n_steps: int = 50,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute Integrated Gradients with robust input handling."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized"})

        # Handle malformed input
        if image_path is None and kwargs:
            cleaned = _clean_tool_input(kwargs)
            image_path = cleaned.get("image_path")
            target_class = cleaned.get("target_class")
            n_steps = cleaned.get("n_steps", 50)
            image_id = cleaned.get("image_id", "temp")

        if image_path is None or target_class is None:
            return json.dumps({"success": False, "error": "image_path and target_class are required"})

        try:
            from PIL import Image
            image_path = str(image_path).strip().strip('"').strip("'")
            target_class = int(target_class)

            image = Image.open(image_path).convert('RGB')
            result = execute_integrated_gradients(
                image=image,
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                target_class=target_class,
                device=self.device,
                image_id=image_id,
                n_steps=n_steps
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _arun(self, *args, **kwargs) -> str:
        """Async version (not implemented)."""
        raise NotImplementedError("Integrated Gradients does not support async execution")


class LIMETool(BaseTool):
    """Tool for executing LIME analysis."""

    name: str = "lime"
    description: str = (
        "Executes LIME for local interpretable explanations. "
        "REQUIRED INPUT: {\"image_path\": \"/path/to/image.jpg\", \"target_class\": 1} "
        "Returns bounding_boxes with actual coordinates."
    )
    args_schema: Type[BaseModel] = LIMEInput

    # Model context
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(
        self,
        image_path: str = None,
        target_class: int = None,
        num_samples: int = 1000,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute LIME with robust input handling."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized"})

        # Handle malformed input
        if image_path is None and kwargs:
            cleaned = _clean_tool_input(kwargs)
            image_path = cleaned.get("image_path")
            target_class = cleaned.get("target_class")
            num_samples = cleaned.get("num_samples", 1000)
            image_id = cleaned.get("image_id", "temp")

        if image_path is None or target_class is None:
            return json.dumps({"success": False, "error": "image_path and target_class are required"})

        try:
            from PIL import Image
            image_path = str(image_path).strip().strip('"').strip("'")
            target_class = int(target_class)

            image = Image.open(image_path).convert('RGB')
            result = execute_lime(
                image=image,
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                target_class=target_class,
                device=self.device,
                image_id=image_id,
                num_samples=num_samples
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _arun(self, *args, **kwargs) -> str:
        """Async version (not implemented)."""
        raise NotImplementedError("LIME does not support async execution")


class SHAPTool(BaseTool):
    """Tool for executing SHAP analysis."""

    name: str = "shap"
    description: str = (
        "Executes SHAP for game-theoretic feature importance. "
        "REQUIRED INPUT: {\"image_path\": \"/path/to/image.jpg\", \"target_class\": 1} "
        "Returns bounding_boxes with actual coordinates."
    )
    args_schema: Type[BaseModel] = SHAPInput

    # Model context
    model: Optional[Any] = None
    model_type: Optional[str] = None
    processor: Optional[Any] = None
    device: Optional[torch.device] = None

    def _run(
        self,
        image_path: str = None,
        target_class: int = None,
        num_samples: int = 100,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute SHAP with robust input handling."""
        if self.model is None:
            return json.dumps({"success": False, "error": "Model not initialized"})

        # Handle malformed input
        if image_path is None and kwargs:
            cleaned = _clean_tool_input(kwargs)
            image_path = cleaned.get("image_path")
            target_class = cleaned.get("target_class")
            num_samples = cleaned.get("num_samples", 100)
            image_id = cleaned.get("image_id", "temp")

        if image_path is None or target_class is None:
            return json.dumps({"success": False, "error": "image_path and target_class are required"})

        try:
            from PIL import Image
            image_path = str(image_path).strip().strip('"').strip("'")
            target_class = int(target_class)

            image = Image.open(image_path).convert('RGB')
            result = execute_shap(
                image=image,
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                target_class=target_class,
                device=self.device,
                image_id=image_id,
                num_samples=num_samples
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _arun(self, *args, **kwargs) -> str:
        """Async version (not implemented)."""
        raise NotImplementedError("SHAP does not support async execution")


class ObjectDetectionTool(BaseTool):
    """Tool for executing object detection."""

    name: str = "object_detection"
    description: str = (
        "Executes YOLO object detection. "
        "REQUIRED INPUT: {\"image_path\": \"/path/to/image.jpg\"} "
        "Returns bounding_boxes with actual coordinates for detected objects."
    )
    args_schema: Type[BaseModel] = ObjectDetectionInput

    def _run(
        self,
        image_path: str = None,
        confidence_threshold: float = 0.25,
        image_id: str = "temp",
        **kwargs
    ) -> str:
        """Execute Object Detection with robust input handling."""
        # Handle malformed input
        if image_path is None and kwargs:
            cleaned = _clean_tool_input(kwargs)
            image_path = cleaned.get("image_path")
            confidence_threshold = cleaned.get("confidence_threshold", 0.25)
            image_id = cleaned.get("image_id", "temp")

        if image_path is None:
            return json.dumps({"success": False, "error": "image_path is required"})

        try:
            from PIL import Image
            image_path = str(image_path).strip().strip('"').strip("'")

            image = Image.open(image_path).convert('RGB')
            result = execute_object_detection(
                image=image,
                image_id=image_id,
                confidence_threshold=confidence_threshold
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _arun(self, *args, **kwargs) -> str:
        """Async version (not implemented)."""
        raise NotImplementedError("Object Detection does not support async execution")


class XAIToolRegistry:
    """
    Registry for XAI Tools in LangChain format.

    Manages initialization and access to all XAI tools.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_type: Optional[str] = None,
        processor: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize tool registry.

        Args:
            model: PyTorch model
            model_type: Model type ('local_pth')
            processor: Image preprocessor
            device: Torch device
        """
        self.model = model
        self.model_type = model_type
        self.processor = processor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tools = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize all available tools."""
        available_tools = get_available_tools()

        if 'gradcam' in available_tools:
            self._tools['gradcam'] = GradCAMTool(
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                device=self.device
            )

        if 'integrated_gradients' in available_tools:
            self._tools['integrated_gradients'] = IntegratedGradientsTool(
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                device=self.device
            )

        if 'lime' in available_tools:
            self._tools['lime'] = LIMETool(
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                device=self.device
            )

        if 'shap' in available_tools:
            self._tools['shap'] = SHAPTool(
                model=self.model,
                model_type=self.model_type,
                processor=self.processor,
                device=self.device
            )

        if 'object_detection' in available_tools:
            self._tools['object_detection'] = ObjectDetectionTool()

    def set_model_context(
        self,
        model: Any,
        model_type: str,
        processor: Any,
        device: Optional[torch.device] = None
    ):
        """
        Set model context for all tools.

        Args:
            model: PyTorch model
            model_type: Model type ('local_pth')
            processor: Image preprocessor
            device: Torch device
        """
        self.model = model
        self.model_type = model_type
        self.processor = processor
        self.device = device or self.device

        # Update all tools (except object_detection which doesn't need model)
        for tool_name, tool in self._tools.items():
            if tool_name != 'object_detection':
                tool.model = model
                tool.model_type = model_type
                tool.processor = processor
                tool.device = self.device

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not available
        """
        return self._tools.get(tool_name)

    def get_all_tools(self) -> list[BaseTool]:
        """
        Get all available tools.

        Returns:
            List of tool instances
        """
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """
        Get names of all available tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tools_by_names(self, tool_names: list[str]) -> list[BaseTool]:
        """
        Get specific tools by their names.

        Args:
            tool_names: List of tool names to retrieve

        Returns:
            List of available tool instances
        """
        tools = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool is not None:
                tools.append(tool)
        return tools


def create_xai_tools(
    model: Any,
    model_type: str,
    processor: Any,
    output_dir: str = "./outputs"
) -> tuple[None, XAIToolRegistry]:
    """
    Create XAI tool registry with model context.

    Args:
        model: PyTorch model
        model_type: Model type ('local_pth')
        processor: Image preprocessor
        output_dir: Output directory for visualizations

    Returns:
        Tuple of (None, XAIToolRegistry)
        Note: First element is None for backwards compatibility
    """
    device = next(model.parameters()).device

    # Set the output directory for XAI visualizations
    set_output_dir(output_dir)

    # Create tool registry
    registry = XAIToolRegistry(
        model=model,
        model_type=model_type,
        processor=processor,
        device=device
    )

    return None, registry
