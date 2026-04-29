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
import re
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


# ============================================================================
# Module-level output directory configuration
# ============================================================================

_masking_output_dir: Optional[Path] = None


def set_masking_output_dir(output_dir: str):
    """Set the root output directory for masking utilities (feature_mean_cache, masked_inputs)."""
    global _masking_output_dir, _feature_mean_cache
    _masking_output_dir = Path(output_dir)
    # Reset the cache so it picks up the new directory
    _feature_mean_cache = None
    print(f"Masking output directory set to: {_masking_output_dir}")


def get_masking_output_dir() -> Path:
    """Get the root output directory. Falls back to ./outputs if not explicitly set."""
    global _masking_output_dir
    if _masking_output_dir is None:
        _masking_output_dir = Path(os.getcwd()) / "outputs"
    return _masking_output_dir


class FeatureMeanCache:
    """
    Cache for storing computed feature means/modes from datasets.
    Persists to disk to avoid recomputation across sessions.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = get_masking_output_dir() / "feature_mean_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "feature_means.json"
        self._mode_cache_file = self.cache_dir / "feature_modes.json"
        self._cache: Dict[str, Dict[str, float]] = {}
        self._mode_cache: Dict[str, Dict[str, Any]] = {}
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
        if self._mode_cache_file.exists():
            try:
                with open(self._mode_cache_file, 'r') as f:
                    self._mode_cache = json.load(f)
            except Exception as e:
                print(f"[FeatureMeanCache] Warning: Could not load mode cache: {e}")
                self._mode_cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"[FeatureMeanCache] Warning: Could not save cache: {e}")

    def _save_mode_cache(self):
        """Save mode cache to disk"""
        try:
            with open(self._mode_cache_file, 'w') as f:
                json.dump(self._mode_cache, f, indent=2)
        except Exception as e:
            print(f"[FeatureMeanCache] Warning: Could not save mode cache: {e}")

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

    def compute_and_cache_mode(self, dataset_path: str, feature_key: str) -> Any:
        """
        Compute mode (most frequent value) for a feature from dataset file and cache it.
        Returns the raw value (string for categorical, float for numeric).
        Supports CSV and JSON formats.
        """
        dataset_key = str(Path(dataset_path).resolve())
        if dataset_key in self._mode_cache and feature_key in self._mode_cache[dataset_key]:
            mode_val = self._mode_cache[dataset_key][feature_key]
            print(f"[FeatureMeanCache] Using cached mode for {feature_key}: {mode_val!r}")
            return mode_val

        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            print(f"[FeatureMeanCache] Warning: Dataset not found: {dataset_path_obj}, using None for mode")
            return None

        try:
            if dataset_path_obj.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(dataset_path_obj)
                if feature_key in df.columns:
                    mode_series = df[feature_key].mode()
                    mode_val = mode_series.iloc[0] if len(mode_series) > 0 else None
                    # Convert numpy types to native Python for JSON serialisation
                    if hasattr(mode_val, 'item'):
                        mode_val = mode_val.item()
                else:
                    print(f"[FeatureMeanCache] Warning: Feature {feature_key} not in dataset, mode=None")
                    mode_val = None
            elif dataset_path_obj.suffix.lower() == '.json':
                with open(dataset_path_obj, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    values = [item[feature_key] for item in data if feature_key in item]
                    if values:
                        from collections import Counter
                        mode_val = Counter(values).most_common(1)[0][0]
                    else:
                        mode_val = None
                else:
                    mode_val = None
            else:
                print(f"[FeatureMeanCache] Warning: Unsupported format {dataset_path_obj.suffix}, mode=None")
                mode_val = None

            if dataset_key not in self._mode_cache:
                self._mode_cache[dataset_key] = {}
            self._mode_cache[dataset_key][feature_key] = mode_val
            self._save_mode_cache()
            print(f"[FeatureMeanCache] Cached mode for {feature_key}: {mode_val!r}")
            return mode_val

        except Exception as e:
            print(f"[FeatureMeanCache] Error computing mode: {e}, using None")
            return None


# Global cache instance
_feature_mean_cache: Optional[FeatureMeanCache] = None

def get_feature_mean_cache() -> FeatureMeanCache:
    """Get or create the global feature mean cache"""
    global _feature_mean_cache
    if _feature_mean_cache is None:
        cache_dir = get_masking_output_dir() / "feature_mean_cache"
        _feature_mean_cache = FeatureMeanCache(cache_dir=cache_dir)
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
        self.output_root = get_masking_output_dir() / "masked_inputs"

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

    def save_from_kwargs(self, data: Any, **kwargs):
        """Save pre-modified data using the same naming/path logic as mask().

        Use this when the input modification is done outside of mask() (e.g., SD
        inpainting, text replacement, tabular direct-set) so the result still gets
        persisted to the masked_inputs directory.

        Args:
            data: Already-modified data to save.
            **kwargs: Same kwargs accepted by mask():
                - dataset_base_name, row_no, tool_name, instance_suffix, mask_suffix
        """
        import re

        dataset_base_name = kwargs.get('dataset_base_name')
        row_no = kwargs.get('row_no')
        tool_name = kwargs.get('tool_name')
        instance_suffix = kwargs.get('instance_suffix', '')
        mask_suffix = kwargs.get('mask_suffix', '')

        dataset_name = None
        q_type_str = None
        if dataset_base_name:
            match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
            if match:
                dataset_name = match.group(1)
                q_type_str = match.group(2)

        instance_str = instance_suffix.strip('_') if instance_suffix else ''
        if dataset_name and row_no is not None:
            if tool_name:
                auto_filename = (f"{tool_name}_mask_{dataset_name}_{row_no}_{instance_str}"
                                 if instance_str else f"{tool_name}_mask_{dataset_name}_{row_no}")
            else:
                auto_filename = (f"mask_{dataset_name}_{row_no}_{instance_str}"
                                 if instance_str else f"mask_{dataset_name}_{row_no}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            auto_filename = f"mask_{timestamp}_{instance_str}" if instance_str else f"mask_{timestamp}"

        if mask_suffix:
            auto_filename = f"{auto_filename}{mask_suffix}"

        self._current_dataset_name = dataset_name
        self._current_q_type = q_type_str
        self._current_question_id = f"{row_no}/{instance_str}" if instance_str else row_no
        self._current_feature_names = kwargs.get('feature_names', [])
        self._current_original_features = kwargs.get('original_features', {})
        self._current_masked_keys = kwargs.get('masked_keys', [])
        self._current_changed_features = kwargs.get('changed_features', {})

        self.save(data, auto_filename)

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

    def _apply_mask(self, input_data: Any, region: Dict[str, Any], **kwargs) -> Any:
        # Handle new text_spans format: {"text_spans": ["phrase1", "phrase2"]}
        if "text_spans" in region:
            spans = region["text_spans"]
            if isinstance(input_data, dict) and 'premise' in input_data:
                text = input_data['premise']
                text = self._mask_text_spans(text, spans, **kwargs)
                result = input_data.copy()
                result['premise'] = text
                return result
            return self._mask_text_spans(input_data, spans, **kwargs)

        # Handle Q6 single-span format: {"span_text": "phrase"}
        if "span_text" in region:
            span_text = region["span_text"]
            spans = [span_text] if span_text else []
            if isinstance(input_data, dict) and 'premise' in input_data:
                result = input_data.copy()
                # Try premise first; if the span isn't there, fall back to hypothesis.
                if span_text and span_text.lower() in input_data['premise'].lower():
                    result['premise'] = self._mask_text_spans(input_data['premise'], spans, **kwargs)
                elif span_text and 'hypothesis' in input_data and span_text.lower() in input_data['hypothesis'].lower():
                    result['hypothesis'] = self._mask_text_spans(input_data['hypothesis'], spans, **kwargs)
                else:
                    # Span not found exactly; apply to premise (case-insensitive regex will be a no-op)
                    result['premise'] = self._mask_text_spans(input_data['premise'], spans, **kwargs)
                return result
            return self._mask_text_spans(input_data, spans, **kwargs)

        # Handle legacy spans format: {"spans": [{start_index, end_index}, ...]}
        if "spans" in region:
            spans = region["spans"]
            # Sort descending by start_index so deletions don't shift subsequent indices
            sorted_spans = sorted(spans, key=lambda s: s.get("start_index", 0), reverse=True)
            if isinstance(input_data, dict) and 'premise' in input_data:
                text = input_data['premise']
                for span in sorted_spans:
                    text = self._mask_text(text, span, **kwargs)
                result = input_data.copy()
                result['premise'] = text
                return result
            text = input_data
            for span in sorted_spans:
                text = self._mask_text(text, span, **kwargs)
            return text

        if not self.validate_region(region):
            raise ValueError(f"Invalid region specification: {region}")

        # Handle NLI dict input: {'premise': '...', 'hypothesis': '...'}
        # Region indices refer to the premise text; hypothesis stays unchanged
        if isinstance(input_data, dict) and 'premise' in input_data:
            text = input_data['premise']
            masked_premise = self._mask_text(text, region, **kwargs)
            result = input_data.copy()
            result['premise'] = masked_premise
            return result

        return self._mask_text(input_data, region, **kwargs)

    def _mask_text_spans(self, text: str, spans: List[str], **kwargs) -> str:
        """Mask all occurrences of each span phrase (case-insensitive)."""
        if self.strategy == MaskingStrategy.DELETE:
            replacement = ""
        elif self.strategy == MaskingStrategy.MASK_TOKEN:
            replacement = self.mask_token
        elif self.strategy == MaskingStrategy.ZERO:
            replacement = None  # computed per span below
        else:
            replacement = kwargs.get("replacement", "")

        result = text
        for span in spans:
            if not isinstance(span, str) or not span:
                continue
            if replacement is None:
                repl = " " * len(span)
            else:
                repl = replacement
            result = re.sub(re.escape(span), repl, result, flags=re.IGNORECASE)
        return result

    def _mask_text(self, text: str, region: Dict[str, Any], **kwargs) -> str:
        """Apply masking strategy to a text string using index-based span."""
        start, end = region["start_index"], region["end_index"]
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))

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

        return text[:start] + replacement + text[end:]

    def save(self, masked_data: Any, filename: str):
        """Save text as .txt (or .json for NLI dict)"""
        if isinstance(masked_data, dict):
            save_path = self._get_target_path(
                "text", "json", filename,
                dataset_name=getattr(self, '_current_dataset_name', None),
                q_type=getattr(self, '_current_q_type', None),
                question_id=getattr(self, '_current_question_id', None)
            )
            import json as _json
            with open(save_path, "w", encoding="utf-8") as f:
                _json.dump(masked_data, f, indent=2, ensure_ascii=False)
        else:
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
        # Accept new text_spans format
        if "text_spans" in region:
            spans = region["text_spans"]
            return isinstance(spans, list) and len(spans) > 0 and any(isinstance(s, str) and s for s in spans)
        # Accept Q6 single-span format
        if "span_text" in region:
            return isinstance(region["span_text"], str) and bool(region["span_text"])
        # Accept legacy spans format
        if "spans" in region:
            spans = region["spans"]
            return isinstance(spans, list) and len(spans) > 0
        # Accept legacy single-span format
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
        dataset_path: Optional[str] = None,
        preprocessor: Optional[Any] = None,
        feature_modes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(strategy)
        self.feature_means = feature_means or {}
        self.feature_modes = feature_modes or {}
        self.dataset_path = dataset_path
        self.preprocessor = preprocessor
        self._mean_cache = get_feature_mean_cache()

    def _apply_mask(self, input_data: Any, region: Dict[str, Any], **kwargs) -> Any:
        # Handle multi-key format: {"feature_keys": ["f1", "f2", ...]}
        if "feature_keys" in region:
            feature_keys = region["feature_keys"]
            # Track for save()
            self._current_masked_keys = list(feature_keys)
            self._current_original_features = kwargs.get('original_features', {})
            self._current_feature_names = kwargs.get('feature_names', [])
            result = input_data
            for key in feature_keys:
                result = self._apply_mask(result, {"feature_key": key}, **kwargs)
            return result

        # Single feature_key — only initialize if not already set by the multi-key caller above
        if not getattr(self, '_current_masked_keys', None):
            self._current_masked_keys = [region.get("feature_key")]
            self._current_original_features = kwargs.get('original_features', {})
            self._current_feature_names = kwargs.get('feature_names', [])

        if not self.validate_region(region):
            raise ValueError(f"Invalid region specification: {region}")

        feature_key = region["feature_key"]
        # Strip trailing comparison conditions from feature key
        # e.g. 'worst concave points > 0.71' → 'worst concave points'
        feature_key = re.sub(r'\s*[><=!]+[\s\d.]+$', '', str(feature_key)).strip()

        # Get dataset path from kwargs or instance
        dataset_path = kwargs.get("dataset_path", self.dataset_path)

        # For tabular, GRAY is not meaningful — treat as MEAN (dataset feature mean)
        effective_strategy = self.strategy
        if effective_strategy == MaskingStrategy.GRAY:
            effective_strategy = MaskingStrategy.MEAN

        if isinstance(input_data, dict):
            data = input_data.copy()
            # Handle 'feature=value' format: try replacing '=' with '_' if the
            # exact key is absent (VLMs often use 'occupation=Craft-repair').
            if feature_key not in data and '=' in feature_key:
                feature_key = feature_key.replace('=', '_')
            if feature_key not in data:
                raise ValueError(f"Feature '{feature_key}' not found in input data")

            if effective_strategy == MaskingStrategy.ZERO:
                data[feature_key] = 0

            elif effective_strategy == MaskingStrategy.MEAN:
                # For categorical (string) features use mode; for numeric use mean
                if isinstance(data[feature_key], str):
                    mode_value = self._get_feature_mode(feature_key, dataset_path, kwargs)
                    data[feature_key] = mode_value if mode_value is not None else data[feature_key]
                else:
                    mean_value = self._get_feature_mean(feature_key, dataset_path, kwargs)
                    data[feature_key] = mean_value

            elif effective_strategy == MaskingStrategy.DELETE:
                del data[feature_key]

            elif effective_strategy == MaskingStrategy.RANDOM:
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
                col_indices = [col_idx]  # single column
            except ValueError:
                # feature_key is a string name — look up in feature_names if provided
                feature_names = kwargs.get('feature_names', [])
                if feature_names and feature_key in feature_names:
                    col_idx = feature_names.index(feature_key)
                    col_indices = [col_idx]
                elif feature_names:
                    # VLMs sometimes use '=' as a separator for categorical features,
                    # e.g. 'occupation=Craft-repair' instead of 'occupation_Craft-repair'.
                    # Normalise that variant up-front so all subsequent lookups work.
                    fk_eq = feature_key.replace('=', '_') if '=' in feature_key else feature_key

                    # Normalise the key: agents may use underscores where feature_names
                    # use spaces (e.g. 'mean_compactness' vs 'mean compactness'), or
                    # vice-versa.  Try both directions before falling back to prefix search.
                    # Also try the '='-replaced variant.
                    alt_key = fk_eq.replace('_', ' ') if '_' in fk_eq else fk_eq.replace(' ', '_')
                    matched_key = next(
                        (k for k in (fk_eq, alt_key) if k in feature_names),
                        None
                    )
                    if matched_key is not None:
                        col_idx = feature_names.index(matched_key)
                        col_indices = [col_idx]
                    else:
                        # Categorical one-hot fallback: the agent used a pre-encoding column
                        # name (e.g. 'relationship') whose post-encoding representation is a
                        # set of '{feature_key}_*' one-hot columns.  Mask all siblings together.
                        # Try both the original key and the '='-normalised variant.
                        col_indices = []
                        for fk_try in dict.fromkeys([feature_key, fk_eq]):  # unique, order-preserving
                            prefix = f"{fk_try}_"
                            col_indices = [i for i, n in enumerate(feature_names) if n.startswith(prefix)]
                            if col_indices:
                                break
                        if not col_indices:
                            # Reverse-prefix fallback: agent used an already-encoded column
                            # name (e.g. 'occupation_Craft-repair') but feature_names has the
                            # base column ('occupation') or vice-versa with hyphens.
                            # Find which base column the key belongs to, then mask all siblings.
                            base_matches = []
                            for fk_try in dict.fromkeys([feature_key, fk_eq]):
                                base_matches = [
                                    (i, n) for i, n in enumerate(feature_names)
                                    if fk_try.startswith(n + '_') or fk_try.startswith(n + '-')
                                    or fk_try.startswith(n + ' ')
                                ]
                                if base_matches:
                                    break
                            if base_matches:
                                # Use the longest matching base name
                                base_idx, base_name = max(base_matches, key=lambda x: len(x[1]))
                                sib_prefix = f"{base_name}_"
                                col_indices = [i for i, n in enumerate(feature_names)
                                               if n.startswith(sib_prefix)]
                                if not col_indices:
                                    col_indices = [base_idx]
                        if not col_indices:
                            raise ValueError(
                                f"For array/tensor input, feature_key must be a column index or a name "
                                f"in feature_names. Got: '{feature_key}'. "
                                f"feature_names provided: {bool(feature_names)}"
                            )
                        col_idx = col_indices[0]  # used only for MEAN/single-col fallbacks
                else:
                    raise ValueError(
                        f"For array/tensor input, feature_key must be a column index or a name "
                        f"in feature_names. Got: '{feature_key}'. "
                        f"feature_names provided: {bool(feature_names)}"
                    )

            if effective_strategy == MaskingStrategy.ZERO:
                # For a single numeric column: set to the standardized equivalent of
                # original-scale 0, i.e. (0 - mean) / scale = -mean / scale.
                # For one-hot siblings (categorical): 0 in encoded space is already correct.
                if len(col_indices) == 1:
                    zero_val = self._get_standardized_zero(feature_key)
                else:
                    zero_val = 0.0
                for ci in col_indices:
                    if data.ndim == 2:
                        data[:, ci] = zero_val
                    else:
                        if is_tensor:
                            data[ci] = torch.tensor(zero_val, dtype=data.dtype)
                        else:
                            data[ci] = zero_val

            elif effective_strategy == MaskingStrategy.MEAN:
                dataset_path = kwargs.get("dataset_path", self.dataset_path)
                for ci in col_indices:
                    if data.ndim == 2:
                        col_data = data[:, ci]
                        mean_val = col_data.mean().item() if is_tensor else col_data.mean()
                        data[:, ci] = mean_val
                    else:
                        # For one-hot siblings use 0 as neutral; for numeric use dataset mean
                        if len(col_indices) > 1:
                            mean_val = 0.0
                        else:
                            mean_val = self._get_feature_mean(feature_key, dataset_path, kwargs)
                        if is_tensor:
                            data[ci] = torch.tensor(mean_val, dtype=data.dtype)
                        else:
                            data[ci] = mean_val

            elif effective_strategy == MaskingStrategy.DELETE:
                col_set = set(col_indices)
                if data.ndim == 2:
                    if is_tensor:
                        indices = [i for i in range(data.shape[1]) if i not in col_set]
                        data = data[:, indices]
                    else:
                        data = np.delete(data, col_indices, axis=1)
                else:
                    if is_tensor:
                        indices = [i for i in range(data.shape[0]) if i not in col_set]
                        data = data[indices]
                    else:
                        data = np.delete(data, col_indices)

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

        # Priority 4: If the feature is scaled by StandardScaler (plain or inside a
        # ColumnTransformer), the scaled mean is exactly 0.0 — no warning needed.
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            preprocessor = self.preprocessor
            if isinstance(preprocessor, StandardScaler):
                return 0.0
            if isinstance(preprocessor, ColumnTransformer):
                for _, transformer, cols in preprocessor.transformers_:
                    if isinstance(transformer, StandardScaler) and feature_key in list(cols):
                        return 0.0
        except ImportError:
            pass

        # Fallback: 0 (warn only when we truly have no information)
        print(f"[TabularMasker] Warning: No mean available for {feature_key}, using 0.0")
        return 0.0

    def _get_standardized_zero(self, feature_key: str) -> float:
        """Return the standardized value corresponding to original-scale 0 for a numeric feature.
        For StandardScaler: standardized_zero = (0 - mean) / scale = -mean / scale.
        Returns 0.0 if preprocessor is unavailable (falls back to encoded-space zero)."""
        preprocessor = self.preprocessor
        if preprocessor is None:
            return 0.0
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            if isinstance(preprocessor, ColumnTransformer):
                for _, transformer, cols in preprocessor.transformers_:
                    cols_list = list(cols)
                    if isinstance(transformer, StandardScaler) and feature_key in cols_list:
                        idx = cols_list.index(feature_key)
                        mean = transformer.mean_[idx]
                        scale = transformer.scale_[idx]
                        return float(-mean / scale)
            elif isinstance(preprocessor, StandardScaler):
                if feature_key in getattr(preprocessor, 'feature_names_in_', []):
                    idx = list(preprocessor.feature_names_in_).index(feature_key)
                    return float(-preprocessor.mean_[idx] / preprocessor.scale_[idx])
        except Exception:
            pass
        return 0.0

    def _get_feature_mode(self, feature_key: str, dataset_path: Optional[str], kwargs: Dict) -> Any:
        """Get feature mode (most frequent value) from pre-computed dict, cache, or dataset file"""
        if feature_key in self.feature_modes:
            return self.feature_modes[feature_key]
        if dataset_path:
            return self._mean_cache.compute_and_cache_mode(dataset_path, feature_key)
        return None

    def _get_original_scale_mean(self, feature_key: str, feature_names: list) -> Optional[float]:
        """Get the training mean for a feature in original (pre-standardization) scale."""
        preprocessor = self.preprocessor
        if preprocessor is None:
            return None
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            if isinstance(preprocessor, ColumnTransformer):
                for _, transformer, cols in preprocessor.transformers_:
                    cols_list = list(cols)
                    if isinstance(transformer, StandardScaler) and feature_key in cols_list:
                        idx = cols_list.index(feature_key)
                        return float(transformer.mean_[idx])
            elif isinstance(preprocessor, StandardScaler):
                if feature_key in feature_names:
                    idx = feature_names.index(feature_key)
                    return float(preprocessor.mean_[idx])
        except Exception:
            pass
        return None

    def save(self, masked_data: Any, filename: str):
        """Save tabular masked input as human-readable JSON in original feature scale."""
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
            if TORCH_AVAILABLE and isinstance(masked_data, torch.Tensor):
                arr = masked_data.cpu().numpy().flatten()
            else:
                arr = np.asarray(masked_data).flatten()

            original_features = getattr(self, '_current_original_features', {})
            masked_keys = getattr(self, '_current_masked_keys', [])
            feature_names = getattr(self, '_current_feature_names', [])

            if original_features:
                # Build human-readable dict from original-scale features, replacing
                # masked features with their training mean (numeric) or mode (categorical).
                readable_dict = dict(original_features)
                for key in masked_keys:
                    orig_val = original_features.get(key)
                    if isinstance(orig_val, str):
                        # Categorical feature: use pre-computed mode dict
                        if key not in self.feature_modes:
                            raise RuntimeError(
                                f"[TabularMasker] Cannot compute mode for categorical feature '{key}': "
                                f"feature_modes dict not set on masker. Pass feature_modes to get_masker()."
                            )
                        readable_dict[key] = self.feature_modes[key]
                    else:
                        if self.strategy == MaskingStrategy.ZERO:
                            readable_dict[key] = 0
                        else:
                            mean_val = self._get_original_scale_mean(key, feature_names)
                            if mean_val is None:
                                raise RuntimeError(
                                    f"[TabularMasker] Cannot get original-scale mean for numeric feature '{key}'. "
                                    f"Ensure preprocessor is set on the masker."
                                )
                            readable_dict[key] = round(mean_val, 4)
                # Apply Q6-style direct changes (raw new values, not mean-fill).
                changed_features = getattr(self, '_current_changed_features', {})
                for key, val in changed_features.items():
                    if key in readable_dict:
                        readable_dict[key] = round(float(val), 6) if isinstance(val, (int, float)) else val
                save_path = self._get_target_path("tabular", "json", filename, **path_kwargs)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(readable_dict, f, indent=4)
            elif feature_names and len(feature_names) == len(arr):
                # Fallback: standardized tensor values with feature names
                feat_dict = {
                    name: round(float(arr[i]), 6)
                    for i, name in enumerate(feature_names)
                }
                save_path = self._get_target_path("tabular", "json", filename, **path_kwargs)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(feat_dict, f, indent=4)
            else:
                # Last resort: index-keyed JSON (no CSV)
                feat_dict = {f"feature_{i}": round(float(v), 6) for i, v in enumerate(arr)}
                save_path = self._get_target_path("tabular", "json", filename, **path_kwargs)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(feat_dict, f, indent=4)

        print(f"[Auto-Save] Tabular output: {save_path}")

    def validate_region(self, region: Dict[str, Any]) -> bool:
        # Accept multi-key format
        if "feature_keys" in region:
            keys = region["feature_keys"]
            return isinstance(keys, list) and len(keys) > 0
        # Accept legacy single-key format
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
        preprocessor = kwargs.get("preprocessor")
        feature_modes = kwargs.get("feature_modes")
        return TabularMasker(strategy, feature_means, dataset_path, preprocessor, feature_modes)

    else:
        raise ValueError(f"Unsupported modality: {modality}")
