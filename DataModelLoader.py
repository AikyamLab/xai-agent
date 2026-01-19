import torch
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import json
import importlib.util
from pathlib import Path
import timm


class DataModelLoader:
    """
    Loader for machine learning models and data from local paths.
    """
    
    def __init__(self, cache_dir: str = "./model_cache"):
        """
        Initialize DataModelLoader
        
        Args:
            cache_dir: Directory to cache downloaded models and save weights
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model = None
        self.model_name = None
        self.model_type = None
        self.processor = None
        self.config = None
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Store loaded data
        self.current_image = None
        self.current_image_path = None
        self.current_tabular_data = None
        self.current_text_data = None
        self.current_data_type = None  # 'image', 'tabular', or 'text'
        
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from local path

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path).convert('RGB')
            self.current_image = image
            self.current_image_path = image_path
            self.current_data_type = 'image'
            print(f"Successfully loaded image from: {image_path}")
            print(f"Image size: {image.size}")
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {image_path}: {e}")

    def load_tabular_data(self, data: Union[Dict[str, Any], pd.DataFrame, str]) -> Dict[str, Any]:
        """
        Load tabular data from dictionary, DataFrame, or CSV file

        Args:
            data: Dictionary of features, DataFrame, or path to CSV file

        Returns:
            Dict containing tabular data
        """
        try:
            if isinstance(data, str):
                # Load from CSV file
                df = pd.read_csv(data)
                data_dict = df.to_dict('records')[0]  # Get first row as dict
                print(f"Successfully loaded tabular data from CSV: {data}")
            elif isinstance(data, pd.DataFrame):
                data_dict = data.to_dict('records')[0]
                print(f"Successfully loaded tabular data from DataFrame")
            elif isinstance(data, dict):
                data_dict = data
                print(f"Successfully loaded tabular data from dictionary")
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            self.current_tabular_data = data_dict
            self.current_data_type = 'tabular'
            print(f"Tabular data features: {list(data_dict.keys())}")
            return data_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load tabular data: {e}")

    def load_text_data(self, text: str) -> str:
        """
        Load text data

        Args:
            text: Text string or path to text file

        Returns:
            Text string
        """
        try:
            # Check if it's a file path
            if os.path.isfile(text):
                with open(text, 'r', encoding='utf-8') as f:
                    text_data = f.read()
                print(f"Successfully loaded text from file: {text}")
            else:
                text_data = text
                print(f"Successfully loaded text data (length: {len(text_data)})")

            self.current_text_data = text_data
            self.current_data_type = 'text'
            return text_data
        except Exception as e:
            raise RuntimeError(f"Failed to load text data: {e}")


    def load_model_from_url(self, model_url: str) -> Dict[str, Any]:
        """
        Load model from HuggingFace URL or timm model name
        Supports both torch and HuggingFace interfaces

        Args:
            model_url: Model URL or identifier
                      e.g., "https://huggingface.co/timm/resnet18.a1_in1k"
                      or "resnet18"

        Returns:
            Dict containing model info and loading status
        """
        try:
            # Parse model identifier from URL
            if "huggingface.co" in model_url:
                # Extract model name from URL
                # e.g., "https://huggingface.co/timm/resnet18.a1_in1k" -> "timm/resnet18.a1_in1k"
                parts = model_url.split("huggingface.co/")[-1].strip("/")
                model_identifier = parts
            else:
                model_identifier = model_url

            print(f"Attempting to load model: {model_identifier}")

            # Clear CUDA cache before loading to prevent NVML issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Try loading as timm model first
            if self._try_load_timm_model(model_identifier):
                return {
                    "success": True,
                    "model": self.model,  # Include model object
                    "processor": self.processor,  # Include processor
                    "model_name": model_identifier,
                    "model_type": "timm",
                    "architecture": self.model.__class__.__name__,
                    "num_classes": self._get_num_classes(),
                    "num_parameters": sum(p.numel() for p in self.model.parameters()),
                    "device": str(self.device)
                }
            else:
                # timm loading failed
                return {
                    "success": False,
                    "error": f"Failed to load model: {model_identifier}",
                    "model_name": model_identifier
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": model_url
            }
    
    def _try_load_timm_model(self, model_name: str) -> bool:
        """
        Try to load model using timm library

        Args:
            model_name: Model identifier for timm

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear CUDA cache before model creation to prevent NVML issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Synchronize to ensure cache is cleared
                torch.cuda.synchronize()

            self.model = timm.create_model(model_name, pretrained=True)
            self.model.eval()

            # CRITICAL: Move model to device with error handling
            try:
                self.model = self.model.to(self.device)
            except RuntimeError as cuda_err:
                # If CUDA fails, try CPU as fallback
                if "CUDA" in str(cuda_err) or "nvml" in str(cuda_err).lower():
                    print(f"Warning: CUDA transfer failed, using CPU: {cuda_err}")
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
                else:
                    raise

            self.model_name = model_name
            self.model_type = "timm"

            # Get data config for preprocessing
            data_config = timm.data.resolve_model_data_config(self.model)
            self.processor = timm.data.create_transform(**data_config, is_training=False)

            print(f"Successfully loaded timm model: {model_name} on {self.device}")
            return True
        except Exception as e:
            print(f"Failed to load as timm model: {e}")
            return False
            

    def _load_model_from_pth(self, model_path: str) -> Dict[str, Any]:
        """
        Load a model from a .pth file using a dynamic loader script.
        The loader script is expected to be in the same directory as the model file,
        with a name like 'load_MODELNAME.py'.
        """
        try:
            model_path = Path(model_path)
            model_name_stem = model_path.stem
            # Construct loader module path, e.g., .../text/load_imdb_2layernn.py
            loader_module_name = f"load_{model_name_stem}.py"
            loader_path = model_path.parent / loader_module_name

            if not loader_path.exists():
                raise FileNotFoundError(f"Loader module not found: {loader_path}")

            # Dynamically import the loader module
            spec = importlib.util.spec_from_file_location(loader_path.stem, loader_path)
            loader_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader_module)

            if not hasattr(loader_module, 'load_model'):
                raise AttributeError(f"Loader module {loader_path} does not have a 'load_model' function.")

            # Load the model and processor using the loader function
            model, processor = loader_module.load_model(str(model_path))
            
            self.model = model
            self.processor = processor
            
            self.model.eval()
            self.model = self.model.to(self.device)
            self.model_name = model_path.name
            self.model_type = "local_pth"
            
            print(f"Successfully loaded model from {model_path} using {loader_path}")

            return {
                "success": True,
                "model": self.model,
                "processor": self.processor,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "model_path": str(model_path),
                "device": str(self.device),
                "architecture": self.model.__class__.__name__,
                "num_classes": self._get_num_classes(),
                "num_parameters": sum(p.numel() for p in self.model.parameters())
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_path": str(model_path)
            }

    def load_model_from_path(self, model_path: str) -> Dict[str, Any]:
        """
        Load model from a local .pth or .pt file.

        Args:
            model_path: Path to a local .pth or .pt file.

        Returns:
            Dict containing model info and loading status
        """
        try:
            if os.path.isfile(model_path) and (model_path.endswith('.pth') or model_path.endswith('.pt')):
                print(f"Detected local model file: {model_path}")
                return self._load_model_from_pth(model_path)
            else:
                raise ValueError(f"Invalid model path: {model_path}. Only local .pth/.pt files are supported.")
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": model_path
            }
    
    def _get_num_classes(self) -> Optional[int]:
        """
        Get number of output classes from the model
        
        Returns:
            Number of classes or None if not available
        """
        try:
            if hasattr(self.model, 'num_classes'):
                return self.model.num_classes
            elif hasattr(self.model, 'get_classifier'):
                classifier = self.model.get_classifier()
                if hasattr(classifier, 'out_features'):
                    return classifier.out_features
            elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'out_features'):
                 return self.model.fc.out_features
            elif hasattr(self.config, 'num_labels'):
                return self.config.num_labels
            elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'out_features'):
                return self.model.classifier.out_features
        except:
            pass
        return None
    
    def predict(self, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Make prediction using the loaded model
        
        Args:
            image: PIL Image object (uses self.current_image if None)
            
        Returns:
            Dict containing prediction results
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model_from_path first.")
        
        if image is None:
            image = self.current_image
        
        if image is None:
            raise RuntimeError("No image provided. Call load_image first or pass image parameter.")
        
        try:
            # Preprocess image
            if self.processor is not None:
                input_tensor = self.processor(image).unsqueeze(0)
            else:
                # Fallback: basic preprocessing
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(image).unsqueeze(0)
            
            # CRITICAL FIX: Move input tensor to the same device as model
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top predictions
            top5_prob, top5_idx = torch.topk(probabilities, min(5, len(probabilities)))
            
            predictions = []
            for i in range(len(top5_idx)):
                predictions.append({
                    "class_idx": int(top5_idx[i].item()),
                    "probability": float(top5_prob[i].item())
                })
            
            return {
                "success": True,
                "predicted_class_idx": int(top5_idx[0].item()),
                "confidence": float(top5_prob[0].item()),
                "top5_predictions": predictions,
                "device": str(self.device)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the loaded model
        
        Returns:
            Dict containing model architecture details
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        summary = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "architecture": self.model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "num_classes": self._get_num_classes(),
            "device": str(self.device)
        }
        
        # Get layer information
        layers = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layers.append({
                    "name": name,
                    "type": module.__class__.__name__
                })
        
        summary["total_layers"] = len(layers)
        summary["sample_layers"] = layers[:10]  # First 10 layers as sample
        
        return summary

# Test code
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DataModelLoader")
    parser.add_argument("--image_path", type=str, default="./dataset/train_image_png_1.png", help="Path to test image")
    parser.add_argument("--model_path", type=str, default="./models_to_read/vision/stl10_resnet.pth", help="Local path to model")
    args = parser.parse_args()

    # Initialize loader
    loader = DataModelLoader()
    
    # Load image
    loader.load_image(args.image_path)
    
    # Load model
    model_info = loader.load_model_from_path(args.model_path)
    summary = loader.get_model_summary()
    print("\n"+"="*70)
    print(summary)
    print("\n"+"="*70)
    if not model_info.get("success"):
        print(f"Failed to load model: {model_info.get('error')}")
    else:
        print(f"Model Info: {model_info}")
        # Predict
        prediction = loader.predict()
        print(f"Prediction: {prediction}")