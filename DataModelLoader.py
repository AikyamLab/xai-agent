import importlib
import torch
from pathlib import Path
from typing import Any, Dict, Optional

class DataModelLoader:
    """
    A generic loader that wraps model-specific loading scripts
    from the `models_to_read` directory.
    """
    def __init__(self, model_name: str, modality: str, data_path: Optional[str] = None):
        """
        Initializes the loader by dynamically importing the specified model's loader module.

        Args:
            model_name (str): The name of the model to load (e.g., 'stl10_resnet').
            modality (str): The data modality (e.g., 'vision', 'text', 'tabular').
            data_path (str, optional): Path to the dataset if different from default.
        """
        self.model_name = model_name
        self.modality = modality
        self.loader_module = self._import_loader_module()
        
        # Determine model path
        model_path_str = f"models_to_read/{self.modality}/{self.model_name}.pth"
        model_path = Path(model_path_str)
        if not model_path.exists():
             model_path_str = f"models_to_read/{self.modality}/{self.model_name}.h5"
             model_path = Path(model_path_str)
             if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path_str} or with .h5 extension")

        self.model, self.processor = self.loader_module.load_model(str(model_path))

        self.current_sample_data: Optional[Dict[str, Any]] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DataModelLoader initialized for '{self.model_name}' on device '{self.device}'.")
        print(f"Model and processor loaded successfully.")

    def _import_loader_module(self):
        """Dynamically imports the loader module for the given model."""
        module_name = f"models_to_read.{self.modality}.load_{self.model_name}"
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
            raise

    def load_sample(self, index: int, split: str = "test") -> Dict[str, Any]:
        """
        Loads a specific data sample using the model's loader module.
        The loaded sample is stored in `self.current_sample_data`.

        Args:
            index (int): The index of the sample to load.
            split (str): The dataset split (e.g., 'test', 'train').

        Returns:
            Dict[str, Any]: A dictionary containing the sample data.
        """
        print(f"Loading sample {index} from '{split}' split...")
        self.current_sample_data = self.loader_module.load_data(index, split)
        print("Sample loaded successfully.")
        return self.current_sample_data

    def get_model(self) -> Any:
        """Returns the loaded model."""
        return self.model

    def get_processor(self) -> Any:
        """Returns the loaded data processor/transform."""
        return self.processor

    def predict(self, image: Any) -> Dict[str, Any]:
        """
        Performs prediction using the loaded model's predict function.

        Args:
            image (Any): The input image (e.g., PIL Image or Tensor) for prediction.

        Returns:
            Dict[str, Any]: The prediction result.
        """
        return self.loader_module.predict(self.model, image, self.processor)

    def get_info(self) -> Dict[str, Any]:
        """Returns information about the model."""
        if hasattr(self.loader_module, 'get_model_info'):
            return self.loader_module.get_model_info(self.model)
        return {"error": "get_model_info function not found in loader module."}

    def get_label_map(self) -> Dict[int, str]:
        """Returns the label map from the loader module."""
        if hasattr(self.loader_module, 'LABEL_MAP'):
            return self.loader_module.LABEL_MAP
        return {"error": "LABEL_MAP not found in loader module."}
        
    def get_current_image(self) -> Any:
        """
        Returns the raw image from the currently loaded sample.

        Returns:
            Any: The raw image (e.g., PIL Image), or None if no sample is loaded.
        """
        if self.current_sample_data and "image" in self.current_sample_data:
            return self.current_sample_data["image"]
        return None

    def get_current_tensor(self) -> Optional[torch.Tensor]:
        """
        Returns the preprocessed image tensor from the currently loaded sample.

        Returns:
            torch.Tensor: The preprocessed tensor ready for model input, or None if no sample is loaded.
        """
        if self.current_sample_data and "image_tensor" in self.current_sample_data:
            tensor = self.current_sample_data["image_tensor"]
            # Ensure batch dimension
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)
        return None
