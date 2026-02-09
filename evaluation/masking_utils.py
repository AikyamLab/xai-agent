"""
Masking Utilities for XAI Agent Framework

Provides modality-specific masking strategies for evaluation:
- Vision: Gray-fill bounding box regions (neutral color, less bias than black)
- Text: DELETE text spans (removes content without introducing new tokens)
- Tabular: Mean-fill with cached statistics from dataset

Updated: Automatic saving and optimized caching for tabular mean values.
"""

import os
import json
import time
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class MaskingStrategy(Enum):
    """Available masking strategies"""
    ZERO = "zero"           # Replace with zeros
    GRAY = "gray"           # Replace with gray (128,128,128) - for vision
    MEAN = "mean"           # Replace with mean value
    BLUR = "blur"           # Apply Gaussian blur (vision only)
    DELETE = "delete"       # Delete entirely (text/tabular)
    MASK_TOKEN = "mask_token"  # Replace with [MASK] (text only)
    RANDOM = "random"       # Replace with random values


class FeatureMeanCache:
    """
    Cache for storing computed feature means from datasets.
    Persists to disk to avoid recomputation across sessions.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path("/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/outputs/feature_mean_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "feature_means.json"
        self._cache: Dict[str, Dict[str, float]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk if exists"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
                print(f"[FeatureMeanCache] Loaded {len(self._cache)} dataset entries from cache")
            except Exception as e:
                print(f"[FeatureMeanCache] Warning: Could not load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"[FeatureMeanCache] Warning: Could not save cache: {e}")

    def get_mean(self, dataset_path: str, feature_key: str) -> Optional[float]:
        """Get cached mean value for a feature in a dataset"""
        dataset_key = str(Path(dataset_path).resolve())
        if dataset_key in self._cache and feature_key in self._cache[dataset_key]:
            return self._cache[dataset_key][feature_key]
        return None

    def set_mean(self, dataset_path: str, feature_key: str, mean_value: float):
        """Store mean value in cache"""
        dataset_key = str(Path(dataset_path).resolve())
        if dataset_key not in self._cache:
            self._cache[dataset_key] = {}
        self._cache[dataset_key][feature_key] = mean_value
        self._save_cache()
        print(f"[FeatureMeanCache] Cached mean for {feature_key}: {mean_value:.4f}")

    def compute_and_cache_mean(self, dataset_path: str, feature_key: str) -> float:
        """
        Compute mean for a feature from dataset file and cache it.
        Supports CSV and JSON formats.
        """
        # Check cache first
        cached = self.get_mean(dataset_path, feature_key)
        if cached is not None:
            print(f"[FeatureMeanCache] Using cached mean for {feature_key}: {cached:.4f}")
            return cached

        # Compute from dataset
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"[FeatureMeanCache] Warning: Dataset not found: {dataset_path}, using 0.0")
            return 0.0

        try:
            if dataset_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(dataset_path)
                if feature_key in df.columns:
                    mean_value = float(df[feature_key].mean())
                else:
                    print(f"[FeatureMeanCache] Warning: Feature {feature_key} not in dataset, using 0.0")
                    mean_value = 0.0
            elif dataset_path.suffix.lower() == '.json':
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    values = [item.get(feature_key, 0) for item in data if feature_key in item]
                    mean_value = float(np.mean(values)) if values else 0.0
                else:
                    mean_value = 0.0
            else:
                print(f"[FeatureMeanCache] Warning: Unsupported format {dataset_path.suffix}, using 0.0")
                mean_value = 0.0

            # Cache the result
            self.set_mean(str(dataset_path), feature_key, mean_value)
            return mean_value

        except Exception as e:
            print(f"[FeatureMeanCache] Error computing mean: {e}, using 0.0")
            return 0.0


# Global cache instance
_feature_mean_cache: Optional[FeatureMeanCache] = None

def get_feature_mean_cache() -> FeatureMeanCache:
    """Get or create the global feature mean cache"""
    global _feature_mean_cache
    if _feature_mean_cache is None:
        _feature_mean_cache = FeatureMeanCache()
    return _feature_mean_cache


class BaseMasker(ABC):
    """Abstract base class for modality-specific maskers"""

    def __init__(self, strategy: MaskingStrategy = MaskingStrategy.ZERO):
        """
        Initialize masker.

        Args:
            strategy: Masking strategy to use
        """
        self.strategy = strategy
        self.output_root = Path("/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/outputs/masked_inputs")

    def mask(
        self,
        input_data: Any,
        region: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Apply masking and AUTOMATICALLY save the output.

        Args:
            input_data: Original input data
            region: Region specification (modality-specific)
            **kwargs: Additional parameters for masking
                - dataset_base_name: Dataset base name from JSON file (e.g., 'stl10_resnet_q1_test')
                - row_no: Row number / question_id for filename (e.g., 12)
                - tool_name: Tool name for tool attribution masked inputs (optional)
                - instance_suffix: Instance identifier for multi-instance questions (e.g., '_A', '_B')

        Returns:
            Masked input data
        """
        import re

        # 1. Apply the mask logic
        masked_data = self._apply_mask(input_data, region, **kwargs)

        # 2. Extract dataset_name and q_type from dataset_base_name
        dataset_base_name = kwargs.get('dataset_base_name')
        row_no = kwargs.get('row_no')
        tool_name = kwargs.get('tool_name')  # For tool attribution
        instance_suffix = kwargs.get('instance_suffix', '')  # For multi-instance (Q4, Q9, Q10)
        mask_suffix = kwargs.get('mask_suffix', '')  # For _improved file naming

        dataset_name = None
        q_type_str = None

        if dataset_base_name:
            match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
            if match:
                dataset_name = match.group(1)  # e.g., "stl10_resnet"
                q_type_str = match.group(2)    # e.g., "q1"

        # 3. Generate filename with instance suffix
        instance_str = instance_suffix.strip('_') if instance_suffix else ''
        if dataset_name and row_no is not None:
            if tool_name:
                # New format: {tool_name}_mask_{dataset_name}_{question_id}_{instance}
                if instance_str:
                    auto_filename = f"{tool_name}_mask_{dataset_name}_{row_no}_{instance_str}"
                else:
                    auto_filename = f"{tool_name}_mask_{dataset_name}_{row_no}"
            else:
                # Regular format: mask_{dataset_name}_{question_id}_{instance}
                if instance_str:
                    auto_filename = f"mask_{dataset_name}_{row_no}_{instance_str}"
                else:
                    auto_filename = f"mask_{dataset_name}_{row_no}"
        else:
            # Fallback to timestamp-based naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if instance_str:
                auto_filename = f"mask_{timestamp}_{instance_str}"
            else:
                auto_filename = f"mask_{timestamp}"

        # 3.5. Append mask_suffix (e.g., "_improved") to filename
        if mask_suffix:
            auto_filename = f"{auto_filename}{mask_suffix}"

        # 4. Store path info for save method (include instance in question_id path if applicable)
        self._current_dataset_name = dataset_name
        self._current_q_type = q_type_str
        # For multi-instance, create subfolder for each instance
        if instance_str:
            self._current_question_id = f"{row_no}/{instance_str}"
        else:
            self._current_question_id = row_no

        # 5. Automatically save the result
        self.save(masked_data, auto_filename)

        return masked_data

    @abstractmethod
    def _apply_mask(self, input_data: Any, region: Dict[str, Any], **kwargs) -> Any:
        """Modality-specific masking implementation"""
        pass

    @abstractmethod
    def save(self, masked_data: Any, filename: str):
        """Modality-specific save implementation"""
        pass

    @abstractmethod
    def validate_region(self, region: Dict[str, Any]) -> bool:
        """Validate that region specification is valid"""
        pass

    def _get_target_path(self, modality: str, extension: str, filename: str,
                         dataset_name: str = None, q_type: str = None, question_id: str = None) -> Path:
        """Helper to prepare directory and return full save path

        Directory structure: /{modality}/{dataset_name}/{q_type}/{question_id}/
        """
        if dataset_name and q_type and question_id is not None:
            target_dir = self.output_root / modality / dataset_name / q_type / str(question_id)
        else:
            target_dir = self.output_root / modality
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{filename}.{extension}"


class VisionMasker(BaseMasker):
    """
    Masker for vision/image data.

    Default strategy: GRAY (neutral color, less bias than black/zero)
    """

    def __init__(self, strategy: MaskingStrategy = MaskingStrategy.GRAY):
        super().__init__(strategy)

    def _apply_mask(self, input_data: Any, region: Dict[str, Any], **kwargs) -> Any:
        if not self.validate_region(region):
            raise ValueError(f"Invalid region specification: {region}")

        bbox = region["bounding_box"]
        x_min, y_min, x_max, y_max = bbox

        if PIL_AVAILABLE and isinstance(input_data, Image.Image):
            return self._mask_pil(input_data, x_min, y_min, x_max, y_max, **kwargs)
        if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor):
            return self._mask_tensor(input_data, x_min, y_min, x_max, y_max, **kwargs)
        if isinstance(input_data, np.ndarray):
            return self._mask_numpy(input_data, x_min, y_min, x_max, y_max, **kwargs)
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    def save(self, masked_data: Any, filename: str):
        """Save image as PNG"""
        save_path = self._get_target_path(
            "vision", "png", filename,
            dataset_name=getattr(self, '_current_dataset_name', None),
            q_type=getattr(self, '_current_q_type', None),
            question_id=getattr(self, '_current_question_id', None)
        )

        if TORCH_AVAILABLE and isinstance(masked_data, torch.Tensor):
            from torchvision.utils import save_image
            save_image(masked_data, save_path)
        elif PIL_AVAILABLE and isinstance(masked_data, Image.Image):
            masked_data.save(save_path)
        elif isinstance(masked_data, np.ndarray):
            if PIL_AVAILABLE:
                Image.fromarray(masked_data.astype('uint8')).save(save_path)
            else:
                import matplotlib.pyplot as plt
                plt.imsave(save_path, masked_data)

        print(f"[Auto-Save] Vision output: {save_path}")

    def _mask_pil(self, img: "Image.Image", x_min: int, y_min: int, x_max: int, y_max: int, **kwargs) -> "Image.Image":
        """Mask PIL Image with GRAY fill by default"""
        img = img.copy()
        w, h = img.size

        # Check if bounding box is completely out of bounds
        original_bbox = (x_min, y_min, x_max, y_max)
        if x_min >= w or y_min >= h or x_max <= 0 or y_max <= 0:
            print(f"[VisionMasker] WARNING: Bounding box {original_bbox} is COMPLETELY outside image bounds ({w}x{h})!")
            print(f"[VisionMasker] No masking will be applied. Please check the bounding box coordinates.")
            return img

        # Check if bounding box needs clipping
        if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
            print(f"[VisionMasker] WARNING: Bounding box {original_bbox} exceeds image bounds ({w}x{h}), clipping...")

        # Clip coordinates to image bounds
        x_min, x_max = max(0, min(x_min, w)), max(0, min(x_max, w))
        y_min, y_max = max(0, min(y_min, h)), max(0, min(y_max, h))

        # Check if clipped region is valid
        if x_min >= x_max or y_min >= y_max:
            print(f"[VisionMasker] WARNING: After clipping, region is empty! Original: {original_bbox}, Image: {w}x{h}")
            return img

        print(f"[VisionMasker] Masking region: ({x_min}, {y_min}) to ({x_max}, {y_max}) in {w}x{h} image")

        pixels = img.load()

        if self.strategy == MaskingStrategy.ZERO:
            # Black fill
            fill_color = (0, 0, 0) if img.mode == 'RGB' else 0
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    pixels[x, y] = fill_color

        elif self.strategy == MaskingStrategy.GRAY:
            # Gray fill (128, 128, 128) - neutral color
            fill_color = (128, 128, 128) if img.mode == 'RGB' else 128
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    pixels[x, y] = fill_color

        elif self.strategy == MaskingStrategy.MEAN:
            region = img.crop((x_min, y_min, x_max, y_max))
            mean_color = tuple(int(c) for c in np.array(region).mean(axis=(0, 1)))
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    pixels[x, y] = mean_color

        elif self.strategy == MaskingStrategy.BLUR:
            blur_radius = kwargs.get("blur_radius", 15)
            region = img.crop((x_min, y_min, x_max, y_max))
            blurred = region.filter(ImageFilter.GaussianBlur(blur_radius))
            img.paste(blurred, (x_min, y_min))

        elif self.strategy == MaskingStrategy.RANDOM:
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    pixels[x, y] = tuple(np.random.randint(0, 256, 3)) if img.mode == 'RGB' else np.random.randint(0, 256)

        return img

    def _mask_tensor(self, tensor: "torch.Tensor", x_min: int, y_min: int, x_max: int, y_max: int, **kwargs) -> "torch.Tensor":
        """Mask torch tensor with GRAY fill by default"""
        tensor = tensor.clone()

        # Handle different tensor shapes
        if tensor.dim() == 4:
            c, h, w = tensor.shape[1:]
            region_slice = (slice(None), slice(None), slice(y_min, y_max), slice(x_min, x_max))
        else:
            c, h, w = tensor.shape
            region_slice = (slice(None), slice(y_min, y_max), slice(x_min, x_max))

        x_min, x_max = max(0, min(x_min, w)), max(0, min(x_max, w))
        y_min, y_max = max(0, min(y_min, h)), max(0, min(y_max, h))

        if self.strategy == MaskingStrategy.ZERO:
            tensor[region_slice] = 0

        elif self.strategy == MaskingStrategy.GRAY:
            # Gray value: 0.5 for normalized tensors (typical range 0-1)
            # or 128 for 0-255 range
            max_val = tensor.max().item()
            gray_value = 0.5 if max_val <= 1.0 else 128.0
            tensor[region_slice] = gray_value

        elif self.strategy == MaskingStrategy.MEAN:
            tensor[region_slice] = tensor[region_slice].mean()

        elif self.strategy == MaskingStrategy.BLUR:
            blur_kernel_size = kwargs.get("blur_kernel_size", 15)
            sigma = kwargs.get("blur_sigma", 5.0)

            was_3d = tensor.dim() == 3
            if was_3d:
                tensor = tensor.unsqueeze(0)

            x = torch.arange(blur_kernel_size) - blur_kernel_size // 2
            gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            kernel = (gauss.outer(gauss) / gauss.outer(gauss).sum())
            kernel = kernel.view(1, 1, blur_kernel_size, blur_kernel_size).expand(c, -1, -1, -1)
            kernel = kernel.to(tensor.device, tensor.dtype)

            padding = blur_kernel_size // 2
            region = tensor[:, :, y_min:y_max, x_min:x_max]
            blurred = F.conv2d(region, kernel, padding=padding, groups=c)
            tensor[:, :, y_min:y_max, x_min:x_max] = blurred[:, :, :y_max-y_min, :x_max-x_min]

            if was_3d:
                tensor = tensor.squeeze(0)

        elif self.strategy == MaskingStrategy.RANDOM:
            tensor[region_slice] = torch.rand_like(tensor[region_slice])

        return tensor

    def _mask_numpy(self, arr: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int, **kwargs) -> np.ndarray:
        """Mask numpy array with GRAY fill by default"""
        arr = arr.copy()
        h, w = arr.shape[:2]
        x_min, x_max = max(0, min(x_min, w)), max(0, min(x_max, w))
        y_min, y_max = max(0, min(y_min, h)), max(0, min(y_max, h))

        if self.strategy == MaskingStrategy.ZERO:
            arr[y_min:y_max, x_min:x_max] = 0

        elif self.strategy == MaskingStrategy.GRAY:
            # Gray value: 128 for typical 0-255 images
            arr[y_min:y_max, x_min:x_max] = 128

        elif self.strategy == MaskingStrategy.MEAN:
            mean_axis = (0, 1) if arr.ndim == 3 else None
            arr[y_min:y_max, x_min:x_max] = arr[y_min:y_max, x_min:x_max].mean(axis=mean_axis)

        elif self.strategy == MaskingStrategy.RANDOM:
            shape = (y_max - y_min, x_max - x_min) + ((arr.shape[2],) if arr.ndim == 3 else ())
            arr[y_min:y_max, x_min:x_max] = np.random.rand(*shape) * 255

        return arr

    def validate_region(self, region: Dict[str, Any]) -> bool:
        if "bounding_box" not in region:
            return False
        bbox = region["bounding_box"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        x_min, y_min, x_max, y_max = bbox
        return x_max > x_min and y_max > y_min


class TextMasker(BaseMasker):
    """
    Masker for text data.

    Default strategy: DELETE (removes text without introducing new tokens)

    Rationale: In XAI evaluation, we want to test "what happens if this text is removed",
    not "what does [MASK] mean". DELETE is more appropriate than MASK_TOKEN because:
    1. It doesn't introduce new semantics (BERT-style [MASK] has learned meaning)
    2. It truly removes the information we want to test
    3. It's model-agnostic (works with any text model, not just MLM)
    """

    def __init__(self, strategy: MaskingStrategy = MaskingStrategy.DELETE, mask_token: str = "[MASK]"):
        super().__init__(strategy)
        self.mask_token = mask_token

    def _apply_mask(self, input_data: str, region: Dict[str, Any], **kwargs) -> str:
        if not self.validate_region(region):
            raise ValueError(f"Invalid region specification: {region}")

        start, end = region["start_index"], region["end_index"]
        start = max(0, min(start, len(input_data)))
        end = max(0, min(end, len(input_data)))

        if self.strategy == MaskingStrategy.DELETE:
            # Simply remove the text span
            replacement = ""
        elif self.strategy == MaskingStrategy.MASK_TOKEN:
            # Replace with [MASK] token (less recommended for XAI eval)
            replacement = self.mask_token
        elif self.strategy == MaskingStrategy.RANDOM:
            # Random characters
            replacement = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), end - start))
        elif self.strategy == MaskingStrategy.ZERO:
            # Spaces (blank)
            replacement = " " * (end - start)
        else:
            replacement = kwargs.get("replacement", "")

        return input_data[:start] + replacement + input_data[end:]

    def save(self, masked_data: str, filename: str):
        """Save text as .txt"""
        save_path = self._get_target_path(
            "text", "txt", filename,
            dataset_name=getattr(self, '_current_dataset_name', None),
            q_type=getattr(self, '_current_q_type', None),
            question_id=getattr(self, '_current_question_id', None)
        )
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(masked_data)
        print(f"[Auto-Save] Text output: {save_path}")

    def validate_region(self, region: Dict[str, Any]) -> bool:
        if "start_index" not in region or "end_index" not in region:
            return False
        start = region["start_index"]
        end = region["end_index"]
        return isinstance(start, int) and isinstance(end, int) and end > start and start >= 0


class TabularMasker(BaseMasker):
    """
    Masker for tabular data.

    Default strategy: MEAN (fill with dataset feature mean)

    Features:
    - Computes mean from provided dataset file
    - Caches computed means for reuse across sessions
    - Falls back to 0 if dataset not available
    """

    def __init__(
        self,
        strategy: MaskingStrategy = MaskingStrategy.MEAN,
        feature_means: Optional[Dict[str, float]] = None,
        dataset_path: Optional[str] = None
    ):
        super().__init__(strategy)
        self.feature_means = feature_means or {}
        self.dataset_path = dataset_path
        self._mean_cache = get_feature_mean_cache()

    def _apply_mask(self, input_data: Any, region: Dict[str, Any], **kwargs) -> Any:
        if not self.validate_region(region):
            raise ValueError(f"Invalid region specification: {region}")

        feature_key = region["feature_key"]

        # Get dataset path from kwargs or instance
        dataset_path = kwargs.get("dataset_path", self.dataset_path)

        if isinstance(input_data, dict):
            data = input_data.copy()
            if feature_key not in data:
                raise ValueError(f"Feature '{feature_key}' not found in input data")

            if self.strategy == MaskingStrategy.ZERO:
                data[feature_key] = 0

            elif self.strategy == MaskingStrategy.MEAN:
                # Try to get mean from cache or compute from dataset
                mean_value = self._get_feature_mean(feature_key, dataset_path, kwargs)
                data[feature_key] = mean_value

            elif self.strategy == MaskingStrategy.DELETE:
                del data[feature_key]

            elif self.strategy == MaskingStrategy.RANDOM:
                original = data[feature_key]
                if isinstance(original, (int, float)):
                    data[feature_key] = np.random.uniform(original * 0.5, original * 1.5)
                else:
                    data[feature_key] = None

            return data

        # Handle numpy array or torch tensor
        if isinstance(input_data, (np.ndarray,)) or (TORCH_AVAILABLE and isinstance(input_data, torch.Tensor)):
            is_tensor = TORCH_AVAILABLE and isinstance(input_data, torch.Tensor)
            data = input_data.clone() if is_tensor else input_data.copy()

            try:
                col_idx = int(feature_key)
            except ValueError:
                raise ValueError(f"For array/tensor input, feature_key must be column index, got: {feature_key}")

            if self.strategy == MaskingStrategy.ZERO:
                if data.ndim == 2:
                    data[:, col_idx] = 0
                else:
                    data[col_idx] = 0

            elif self.strategy == MaskingStrategy.MEAN:
                # Compute mean from the column itself if no dataset provided
                if data.ndim == 2:
                    col_data = data[:, col_idx]
                    mean_val = col_data.mean().item() if is_tensor else col_data.mean()
                    data[:, col_idx] = mean_val
                else:
                    mean_val = data.mean().item() if is_tensor else data.mean()
                    data[col_idx] = mean_val

            elif self.strategy == MaskingStrategy.DELETE:
                if data.ndim == 2:
                    if is_tensor:
                        indices = [i for i in range(data.shape[1]) if i != col_idx]
                        data = data[:, indices]
                    else:
                        data = np.delete(data, col_idx, axis=1)
                else:
                    if is_tensor:
                        indices = [i for i in range(data.shape[0]) if i != col_idx]
                        data = data[indices]
                    else:
                        data = np.delete(data, col_idx)

            return data

        raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _get_feature_mean(self, feature_key: str, dataset_path: Optional[str], kwargs: Dict) -> float:
        """Get feature mean from cache, kwargs, or compute from dataset"""
        # Priority 1: Explicit mean_value in kwargs
        if "mean_value" in kwargs:
            return float(kwargs["mean_value"])

        # Priority 2: Pre-provided feature_means dict
        if feature_key in self.feature_means:
            return self.feature_means[feature_key]

        # Priority 3: Compute from dataset and cache
        if dataset_path:
            return self._mean_cache.compute_and_cache_mean(dataset_path, feature_key)

        # Fallback: 0
        print(f"[TabularMasker] Warning: No mean available for {feature_key}, using 0.0")
        return 0.0

    def save(self, masked_data: Any, filename: str):
        """Save tabular as JSON or CSV"""
        path_kwargs = {
            'dataset_name': getattr(self, '_current_dataset_name', None),
            'q_type': getattr(self, '_current_q_type', None),
            'question_id': getattr(self, '_current_question_id', None)
        }
        if isinstance(masked_data, dict):
            save_path = self._get_target_path("tabular", "json", filename, **path_kwargs)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(masked_data, f, indent=4)
        else:
            save_path = self._get_target_path("tabular", "csv", filename, **path_kwargs)
            if TORCH_AVAILABLE and isinstance(masked_data, torch.Tensor):
                arr = masked_data.cpu().numpy()
            else:
                arr = masked_data
            np.savetxt(save_path, arr, delimiter=",")
        print(f"[Auto-Save] Tabular output: {save_path}")

    def validate_region(self, region: Dict[str, Any]) -> bool:
        return "feature_key" in region and region["feature_key"] is not None


def get_masker(
    modality: str,
    strategy: Optional[MaskingStrategy] = None,
    **kwargs
) -> BaseMasker:
    """
    Factory function to get appropriate masker for modality.

    Default strategies:
    - vision: GRAY (neutral color, less biased than black)
    - text: DELETE (removes content, no new tokens introduced)
    - tabular: MEAN (fills with dataset mean, cached for efficiency)

    Args:
        modality: "vision", "text", or "tabular"
        strategy: Masking strategy (uses modality-appropriate default if None)
        **kwargs: Additional arguments for masker
            - mask_token: Token for text MASK_TOKEN strategy
            - feature_means: Pre-computed means for tabular
            - dataset_path: Path to dataset for tabular mean computation

    Returns:
        BaseMasker subclass instance
    """
    if modality == "vision":
        # Default: GRAY fill (neutral, less biased)
        if strategy is None:
            strategy = MaskingStrategy.GRAY
        return VisionMasker(strategy)

    elif modality == "text":
        # Default: DELETE (removes text, no new tokens)
        if strategy is None:
            strategy = MaskingStrategy.DELETE
        mask_token = kwargs.get("mask_token", "[MASK]")
        return TextMasker(strategy, mask_token)

    elif modality == "tabular":
        # Default: MEAN fill (uses cached dataset means)
        if strategy is None:
            strategy = MaskingStrategy.MEAN
        feature_means = kwargs.get("feature_means")
        dataset_path = kwargs.get("dataset_path")
        return TabularMasker(strategy, feature_means, dataset_path)

    else:
        raise ValueError(f"Unsupported modality: {modality}")
