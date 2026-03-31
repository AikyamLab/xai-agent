import importlib
import torch
import torchvision.transforms as transforms
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

        if self.modality == 'vision' and "image" in self.current_sample_data:
            self.current_sample_data["processed_image"] = self._compute_processed_image(
                self.current_sample_data["image"]
            )

        print("Sample loaded successfully.")
        return self.current_sample_data

    def _compute_processed_image(self, raw_image: Any) -> Any:
        """
        Apply spatial-only transforms (Resize + CenterCrop if any) from the loader's
        get_transform(), stripping ToTensor and Normalize, to produce a PIL Image
        at the same spatial resolution the model sees.
        """
        full_transform = self.loader_module.get_transform()
        spatial_steps = [
            t for t in full_transform.transforms
            if not isinstance(t, (transforms.ToTensor, transforms.Normalize))
        ]
        viz_transform = transforms.Compose(spatial_steps)
        return viz_transform(raw_image)

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
        Returns the processed image (after Resize+CenterCrop, before ToTensor/Normalize)
        from the currently loaded sample.  This is the image the model spatially sees,
        so XAI heatmaps and bounding-box coordinates are in this coordinate space.

        Raises KeyError if no sample is loaded or processed_image is missing.
        """
        return self.current_sample_data["processed_image"]

    def get_display_image(self) -> Any:
        """
        Returns the best PIL image for XAI heatmap display:
        - Raw image  when raw_size < processed_size (pure Resize, no CenterCrop).
          The heatmap is then downsampled from model-input resolution → natural
          spatial smoothing (e.g. STL-10: 96×96 raw vs 224×224 processed).
        - Processed image otherwise (e.g. CUB: raw images are larger than 224×224
          and have been center-cropped, so processed_image is the correct canvas).
        Gradient computation always uses the pre-computed input_tensor (unaffected).
        """
        processed = self.current_sample_data["processed_image"]
        raw = self.current_sample_data.get("image")
        if raw is not None and hasattr(raw, "size"):
            raw_px = raw.size[0] * raw.size[1]
            proc_px = processed.size[0] * processed.size[1]
            if raw_px < proc_px:
                return raw
        return processed

    def get_current_text(self) -> Optional[str]:
        """
        Returns the raw text from the currently loaded sample.

        Returns:
            str or None if no text sample is loaded.
        """
        if self.current_sample_data and "text" in self.current_sample_data:
            return self.current_sample_data["text"]
        if hasattr(self, 'current_text_data') and self.current_text_data is not None:
            if isinstance(self.current_text_data, str):
                return self.current_text_data
            if isinstance(self.current_text_data, dict):
                return self.current_text_data.get('text', str(self.current_text_data))
        return None

    def get_current_token_ids(self) -> Optional[list]:
        """
        Returns the token IDs from the currently loaded sample.
        """
        if self.current_sample_data and "token_ids" in self.current_sample_data:
            return self.current_sample_data["token_ids"]
        return None

    def get_current_input(self) -> Any:
        """
        Returns the current input data based on modality.
        """
        if self.modality == 'vision':
            return self.get_current_image()
        elif self.modality == 'text':
            return self.get_current_text()
        elif self.modality == 'tabular':
            if hasattr(self, 'current_tabular_data'):
                return self.current_tabular_data
            if self.current_sample_data:
                return self.current_sample_data.get('features')
        return None

    def get_current_features(self) -> Optional[torch.Tensor]:
        """Returns the preprocessed feature tensor for tabular data."""
        if self.current_sample_data and "features" in self.current_sample_data:
            return self.current_sample_data["features"]
        return None

    def get_current_features_dict(self) -> Optional[Dict[str, float]]:
        """Returns the feature name->value dict for tabular data."""
        if self.current_sample_data and "features_dict" in self.current_sample_data:
            return self.current_sample_data["features_dict"]
        return None

    def get_feature_names(self) -> Optional[list]:
        """Returns feature names for tabular data."""
        if self.current_sample_data and "feature_names" in self.current_sample_data:
            return self.current_sample_data["feature_names"]
        if hasattr(self.loader_module, '_cache') and self.loader_module._cache.get("feature_names"):
            return self.loader_module._cache["feature_names"]
        return None

    def get_encoded_to_original(self) -> Optional[dict]:
        """Returns mapping from encoded feature name to original feature name (adult/one-hot datasets only)."""
        if self.current_sample_data and "encoded_to_original" in self.current_sample_data:
            return self.current_sample_data["encoded_to_original"]
        if hasattr(self.loader_module, '_cache') and self.loader_module._cache.get("encoded_to_original"):
            return self.loader_module._cache["encoded_to_original"]
        return None

    def get_raw_features_dict(self) -> Optional[dict]:
        """Returns original (pre-encoding) feature values for the current sample."""
        if self.current_sample_data and "raw_features_dict" in self.current_sample_data:
            return self.current_sample_data["raw_features_dict"]
        return None

    def get_training_data(self) -> Optional[torch.Tensor]:
        """Returns training data tensor (for SHAP/LIME background data)."""
        if not hasattr(self.loader_module, '_cache'):
            return None
        cache = self.loader_module._cache

        # Direct X_train key (not currently used by any loader, but forward-compatible)
        if cache.get("X_train") is not None:
            return cache["X_train"]

        # Adult Census: full dataset + train_indices set → reconstruct train split
        if cache.get("X_all") is not None and cache.get("train_indices") is not None:
            X_all = cache["X_all"]
            train_indices = cache["train_indices"]  # set of original DataFrame indices
            orig_to_pos = cache.get("orig_to_pos", {})
            if orig_to_pos:
                positions = [orig_to_pos[i] for i in train_indices if i in orig_to_pos]
                if positions:
                    return X_all[positions]

        # Breast Cancer: X_full + X_test → derive train rows by exclusion
        if cache.get("X_full") is not None and cache.get("X_test") is not None:
            return cache["X_full"]

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
