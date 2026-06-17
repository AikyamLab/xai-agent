"""
XAI Tools Core Implementation

Provides core implementations for XAI methods (GradCAM, LIME, SHAP, etc.)
These are low-level implementations that return raw XAI results.

IMPORTANT: Feature extraction (bounding boxes, importance regions) is NOT done here.
That task is delegated to the Actor Agent's VLM reasoning capability.

All tools:
- Execute the XAI method
- Save visualizations
- Return raw statistics and descriptions
- Let the Agent extract structured features through reasoning
"""

import threading
import torch
import numpy as np
from typing import Any, Dict, Optional, List
from pathlib import Path
from PIL import Image
import json
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import gaussian_filter

# XAI Libraries
from captum.attr import (
    LayerGradCam,
    IntegratedGradients,
    GuidedBackprop,
    Saliency,
    NoiseTunnel
)
from lime import lime_image
import shap
from skimage.segmentation import mark_boundaries

# Computer Vision Libraries
from ultralytics import YOLO


# Thread-local output directory for XAI visualizations.
# Using threading.local() instead of a module-level global so that parallel
# rollout workers (ThreadPoolExecutor) each maintain their own output path
# without overwriting each other.
_tls = threading.local()

# GPU concurrency is controlled by limiting max_rollout_workers (=40) at the
# trainer level instead of a per-tool semaphore.  40 concurrent workers × 1.5 GB
# (LIME peak) + 3.7 GB model weights + 3 GB overhead ≈ 67 GB, safely within
# the 79 GB GPU budget.  A semaphore caused severe queuing latency (164 workers
# × 32 sequential semaphore acquisitions per LIME call → step time 66-93 min).
_LIME_SHAP_GPU_SEM = threading.Semaphore(1000)  # effectively disabled


def set_output_dir(output_dir: str):
    """Set the per-thread output directory for XAI visualizations."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    _tls.output_dir = p
    print(f"XAI output directory set to: {p}")


def get_output_dir() -> Path:
    """Get the per-thread output directory, creating a default if not set."""
    d = getattr(_tls, 'output_dir', None)
    if d is None:
        d = Path(os.getcwd()) / "outputs" / "xai_visualizations"
        d.mkdir(parents=True, exist_ok=True)
        _tls.output_dir = d
    return d


def _save_heatmap_visualization(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    title: str = "Heatmap",
    bbox: Optional[List[int]] = None,
) -> str:
    """
    Save a heatmap visualization overlaid on the original image.

    Uses matplotlib OO API (Figure/FigureCanvasAgg) instead of pyplot globals
    so this function is safe to call from multiple threads simultaneously.

    Args:
        original_image: Original image as numpy array (H, W, 3)
        heatmap: 2D numpy array with activation values (normalized 0-1)
        output_path: Path to save the visualization
        title: Title for the visualization
        bbox: Optional [x0, y0, x1, y1] box (e.g. `suggested_bounding_box`) to
            draw on the heatmap and overlay panels

    Returns:
        Path to saved visualization
    """
    fig = Figure(figsize=(15, 5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(title)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])

    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        for ax in (axes[1], axes[2]):
            ax.add_patch(patches.Rectangle(
                (x0, y0), w, h, linewidth=2.5, edgecolor='white', facecolor='none'
            ))
            ax.add_patch(patches.Rectangle(
                (x0, y0), w, h, linewidth=1.2, edgecolor='black', facecolor='none', linestyle='--'
            ))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return output_path


def _save_raw_heatmap(heatmap: np.ndarray, output_path: str) -> str:
    """
    Save raw heatmap as numpy file for later analysis by Agent.

    Args:
        heatmap: 2D numpy array
        output_path: Path to save (without extension)

    Returns:
        Path to saved file
    """
    npy_path = output_path.replace('.png', '_heatmap.npy')
    np.save(npy_path, heatmap)
    return npy_path


def get_available_tools() -> list[str]:
    """
    Get list of available XAI tools.

    Returns:
        List of tool names
    """
    return [
        'gradcam',
        'integrated_gradients',
        'lime',
        'shap',
        'object_detection',
        'guided_backprop',
        'smoothgrad',
        'sensitivity_analysis',
        'layer_cam'
    ]


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Get the target layer for gradient-based methods.

    For ResNet/similar architectures, targets the full last residual block
    (e.g. model.layer4[-1]) so GradCAM sees activations after the residual
    add + ReLU.  Targeting a raw Conv2d inside a bottleneck (before the
    residual path) causes all-negative pre-ReLU attributions that become
    all-zero after the GradCAM ReLU step.

    Returns:
        Target layer — last residual/dense block, or last Conv2d as fallback
    """
    # ResNet / wide-resnet style: layer4 contains the last residual blocks
    for attr_name in ('layer4', 'layer3'):
        if hasattr(model, attr_name):
            block = getattr(model, attr_name)
            if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                return block[-1]

    # DenseNet style: features is a Sequential ending with denseblock + norm
    if hasattr(model, 'features') and isinstance(model.features, torch.nn.Sequential):
        # Return the last sub-module that contains Conv2d layers
        for child in reversed(list(model.features.children())):
            if any(isinstance(m, torch.nn.Conv2d) for m in child.modules()):
                return child

    # Generic fallback: last Conv2d in the model
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        modules = list(model.children())
        target_layer = modules[-2] if len(modules) > 1 else modules[-1]

    return target_layer


def _preprocess_image(
    image: Image.Image,
    model_type: str,
    processor: Any,
    device: torch.device
) -> torch.Tensor:
    """
    Preprocess image for model input.

    Args:
        image: PIL Image
        model_type: Type of model ('timm', 'local_pth', 'huggingface')
        processor: Image preprocessor (can be torchvision.transforms.Compose or HuggingFace processor)
        device: Torch device

    Returns:
        Preprocessed tensor
    """
    import torchvision.transforms as transforms

    # Check if processor is a torchvision Compose transform
    if processor is not None and isinstance(processor, transforms.Compose):
        input_tensor = processor(image)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return input_tensor.to(device).float()

    if model_type in ["timm", "local_pth"]:
        # Standard ImageNet preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        input_tensor = preprocess(image).unsqueeze(0)
    else:
        # HuggingFace processor
        if processor is not None:
            inputs = processor(images=image, return_tensors="pt")
            input_tensor = inputs['pixel_values']
        else:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            input_tensor = preprocess(image).unsqueeze(0)

    # Explicitly cast to float32: guards against global dtype contamination
    # (e.g. from diffusers fp16 pipeline or concurrent timm pretrained loading).
    return input_tensor.to(device).float()


# ===================================================================
# GradCAM Implementation
# ===================================================================

def execute_gradcam(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute GradCAM analysis with visualization.

    Returns raw results - feature extraction is done by Actor Agent.

    Args:
        image: PIL Image
        model: PyTorch model
        model_type: Model type ('local_pth', 'timm', etc.)
        processor: Image preprocessor
        target_class: Target class index
        device: Torch device
        image_id: Identifier for output files
        input_tensor: Optional pre-processed tensor. If provided, skips preprocessing.

    Returns:
        Dictionary with raw results:
        - visualization_path: Path to saved visualization
        - heatmap_path: Path to raw heatmap numpy file
        - statistics: Raw statistics from the analysis
        - description: Human-readable description
    """
    # Store original image size
    original_size = image.size  # (width, height)
    img_np = np.array(image)

    model.eval()

    # Use provided tensor or preprocess
    if input_tensor is None:
        input_tensor = _preprocess_image(image, model_type, processor, device)
    else:
        input_tensor = input_tensor.to(device).float()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
    target_layer = _get_target_layer(model)

    # Initialize LayerGradCam
    gradcam = LayerGradCam(model, target_layer)

    # Compute attribution
    attribution = gradcam.attribute(
        input_tensor,
        target=target_class
    )

    # Convert to numpy and normalize
    attr_np = attribution.squeeze().cpu().detach().numpy()
    attr_np = np.maximum(attr_np, 0)  # ReLU
    attr_np = attr_np.mean(axis=0) if len(attr_np.shape) == 3 else attr_np

    # Resize to original image size
    heatmap = cv2.resize(attr_np, (original_size[0], original_size[1]))
    if heatmap.max() == heatmap.min():
        raise RuntimeError(
            f"GradCAM attribution is uniform (all values = {heatmap.max():.6f}) for class {target_class}. "
            f"The target layer may have zero gradients. Check model, target layer, and target_class."
        )
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Save visualization
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"gradcam_{image_id}_class{target_class}.png")
    _save_heatmap_visualization(img_np, heatmap, viz_path, "GradCAM")

    # Save raw heatmap
    heatmap_path = _save_raw_heatmap(heatmap, viz_path)

    # Compute statistics
    high_attention_ratio = float((heatmap > 0.7).sum() / heatmap.size)
    mean_attention = float(heatmap.mean())
    max_attention = float(heatmap.max())

    # Find top attention regions (coordinates for Agent reference)
    top_k = 10
    flat_idx = np.argsort(heatmap.flatten())[-top_k:]
    top_coords = [
        {"y": int(np.unravel_index(idx, heatmap.shape)[0]),
         "x": int(np.unravel_index(idx, heatmap.shape)[1]),
         "value": float(heatmap.flatten()[idx])}
        for idx in flat_idx
    ]

    # Compute suggested bounding box from top 1% of heatmap pixels
    threshold = np.percentile(heatmap, 99)
    ys, xs = np.where(heatmap >= threshold)
    suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

    result = {
        "success": True,
        "method": "GradCAM",
        "target_class": target_class,
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "heatmap_shape": list(heatmap.shape),
        "image_id": image_id,
        "visualization_path": viz_path,
        "heatmap_path": heatmap_path,
        "suggested_bounding_box": suggested_bbox,
        "statistics": {
            "high_attention_ratio": round(high_attention_ratio, 4),
            "mean_attention": round(mean_attention, 4),
            "max_attention": round(max_attention, 4),
            "std_attention": round(float(np.std(heatmap)), 4),
            "top_attention_coords": top_coords
        },
        "description": (
            f"GradCAM analysis for class {target_class}: "
            f"{high_attention_ratio*100:.1f}% of regions show high attention (>0.7 threshold). "
            f"Mean attention: {mean_attention:.3f}, Max: {max_attention:.3f}. "
            f"The model focuses on specific regions - see visualization for details."
        )
    }

    return result


# ===================================================================
# Integrated Gradients Implementation
# ===================================================================

def execute_integrated_gradients(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    n_steps: int = 200,
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute Integrated Gradients with High-Precision post-processing.
    """
    try:
        original_size = image.size
        img_np = np.array(image)
        model.eval()

        if input_tensor is None:
            input_tensor = _preprocess_image(image, model_type, processor, device)
        else:
            input_tensor = input_tensor.to(device).float()
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

        # Baseline: True black in normalized space to prevent background leakage.
        means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        baseline = (torch.zeros_like(input_tensor) - means) / stds

        inplace_states = {}
        for name, module in model.named_modules():
            if hasattr(module, 'inplace') and module.inplace:
                inplace_states[name] = True
                module.inplace = False

        try:
            ig = IntegratedGradients(model)
            attribution = ig.attribute(input_tensor, baselines=baseline, target=target_class, n_steps=n_steps, internal_batch_size=4)
        finally:
            for name, module in model.named_modules():
                if name in inplace_states:
                    module.inplace = True

        # Process: [C, H, W]
        attr_np = attribution.squeeze(0).cpu().detach().numpy()
        
        # 1. Aggregation: Sum signed gradients across channels
        if len(attr_np.shape) == 3:
            attr_np = attr_np.sum(axis=0)
        
        # 2. Noise Reduction: Percentile clipping (99.5th)
        vmax = np.percentile(np.abs(attr_np), 99.5)
        attr_np = np.clip(attr_np, -vmax, vmax)
        
        # 3. Absolute importance
        heatmap = np.abs(attr_np)

        # 4. Resize and Visual Smoothing
        heatmap = cv2.resize(heatmap, (original_size[0], original_size[1]))
        sigma = max(0.5, original_size[0] * 0.005)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        viz_path = str(get_output_dir() / f"ig_{image_id}_class{target_class}.png")
        _save_heatmap_visualization(img_np, heatmap, viz_path, "Integrated Gradients")
        heatmap_path = _save_raw_heatmap(heatmap, viz_path)

        p90 = float(np.percentile(heatmap, 90))
        mean_val = float(heatmap.mean())
        flat_idx = np.argsort(heatmap.flatten())[-10:]
        top_coords = [{"y": int(np.unravel_index(idx, heatmap.shape)[0]), "x": int(np.unravel_index(idx, heatmap.shape)[1]), "value": float(heatmap.flatten()[idx])} for idx in flat_idx]

        # Compute suggested bounding box from top 1% of heatmap pixels
        threshold = np.percentile(heatmap, 99)
        ys, xs = np.where(heatmap >= threshold)
        suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

        return {
            "success": True, "method": "IntegratedGradients", "target_class": target_class,
            "original_image_size": {"width": original_size[0], "height": original_size[1]},
            "visualization_path": viz_path, "heatmap_path": heatmap_path,
            "suggested_bounding_box": suggested_bbox,
            "statistics": {"high_importance_ratio": round(float((heatmap > 0.6).sum() / heatmap.size), 4), "mean_importance": round(mean_val, 4), "top_importance_coords": top_coords},
            "description": f"High-precision IG with True-Black baseline and outlier clipping."
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc(), "method": "IntegratedGradients"}


# ===================================================================
# LIME Implementation
# ===================================================================

def execute_lime(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    num_samples: int = 1000
) -> Dict[str, Any]:
    """
    Execute LIME analysis with visualization.

    Returns raw results - feature extraction is done by Actor Agent.
    """
    original_size = image.size
    img_np = np.array(image)

    # Define prediction function for LIME
    # Batched: stack _LIME_BATCH images into one GPU call instead of 1000 individual
    # calls, reducing Python GIL holding time ~32× and GPU launch overhead.
    _LIME_BATCH = 32
    def predict_fn(images):
        # images shape: (N, H, W, C) uint8
        results = []
        for i in range(0, len(images), _LIME_BATCH):
            batch_imgs = images[i:i + _LIME_BATCH]
            tensors = torch.cat([
                _preprocess_image(Image.fromarray(img.astype('uint8')), model_type, processor, device)
                for img in batch_imgs
            ], dim=0)  # [B, C, H, W]
            with torch.no_grad():
                output = model(tensors)
                probs = torch.nn.functional.softmax(output, dim=1)
            results.append(probs.cpu().numpy())
        return np.concatenate(results, axis=0)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples
    )

    # Get mask for target class
    label_to_explain = int(target_class if target_class in explanation.top_labels else explanation.top_labels[0])
    temp, mask = explanation.get_image_and_mask(
        label_to_explain,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Get segments and their weights
    segments = explanation.segments
    local_exp = explanation.local_exp[label_to_explain]

    # Calculate segment statistics
    segment_weights = {seg_id: weight for seg_id, weight in local_exp}
    positive_segments = [(seg_id, w) for seg_id, w in local_exp if w > 0]
    negative_segments = [(seg_id, w) for seg_id, w in local_exp if w < 0]

    positive_segments.sort(key=lambda x: x[1], reverse=True)
    negative_segments.sort(key=lambda x: x[1])

    # Top segment info
    top_positive = [
        {"segment_id": int(seg_id), "weight": round(float(w), 4)}
        for seg_id, w in positive_segments[:5]
    ]
    top_negative = [
        {"segment_id": int(seg_id), "weight": round(float(w), 4)}
        for seg_id, w in negative_segments[:5]
    ]

    # Save visualization
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"lime_{image_id}_class{target_class}.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mark_boundaries(temp / 255.0, mask))
    axes[1].set_title(f'LIME - Top Positive Features (Class {label_to_explain})')
    axes[1].axis('off')

    # Create importance heatmap from segments
    importance_map = np.zeros(segments.shape)
    for seg_id, weight in local_exp:
        importance_map[segments == seg_id] = weight

    max_abs = np.abs(importance_map).max() if np.abs(importance_map).max() > 0 else 1
    axes[2].imshow(importance_map, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title('LIME Importance Map')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Statistics
    important_regions = float(mask.sum() / mask.size)

    # Compute bounding boxes for top positive segments
    def get_segment_bbox(segments, seg_id):
        """Get bounding box [x_min, y_min, x_max, y_max] for a segment"""
        mask = (segments == seg_id)
        if not mask.any():
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    # Add bounding box info to top segments
    top_positive_with_bbox = []
    for seg_id, weight in positive_segments[:5]:
        bbox = get_segment_bbox(segments, seg_id)
        top_positive_with_bbox.append({
            "segment_id": int(seg_id),
            "weight": round(float(weight), 4),
            "bbox": bbox
        })

    top_negative_with_bbox = []
    for seg_id, weight in negative_segments[:5]:
        bbox = get_segment_bbox(segments, seg_id)
        top_negative_with_bbox.append({
            "segment_id": int(seg_id),
            "weight": round(float(weight), 4),
            "bbox": bbox
        })

    # Compute suggested bounding box from top 1% of importance_map pixels
    threshold = np.percentile(importance_map, 99)
    ys, xs = np.where(importance_map >= threshold)
    suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

    result = {
        "success": True,
        "method": "LIME",
        "target_class": target_class,
        "explained_class": label_to_explain,
        "num_samples": num_samples,
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "top_labels": [int(x) for x in explanation.top_labels],
        "image_id": image_id,
        "visualization_path": viz_path,
        "num_segments": int(segments.max()) + 1,
        "suggested_bounding_box": suggested_bbox,
        "statistics": {
            "num_positive_segments": len(positive_segments),
            "num_negative_segments": len(negative_segments),
            "important_region_ratio": round(important_regions, 4),
            "max_positive_weight": round(float(max([w for _, w in positive_segments], default=0)), 4),
            "max_negative_weight": round(float(min([w for _, w in negative_segments], default=0)), 4),
            "top_positive_segments": top_positive_with_bbox,
            "top_negative_segments": top_negative_with_bbox
        },
        "description": (
            f"LIME analysis ({num_samples} samples) for class {label_to_explain}: "
            f"Found {len(positive_segments)} positive and {len(negative_segments)} negative segments. "
            f"{important_regions*100:.1f}% of image marked as important. "
            f"Superpixel-based explanation shows which regions support/oppose the prediction."
        )
    }

    return result


# ===================================================================
# SHAP Implementation
# ===================================================================

def execute_shap(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    num_samples: int = 1000
) -> Dict[str, Any]:
    """
    Execute SHAP analysis with visualization.

    Returns raw results - feature extraction is done by Actor Agent.
    """
    original_size = image.size
    img_np = np.array(image).astype(np.float32) / 255.0

    # Define prediction function
    # Batched: same GIL-reduction strategy as LIME's predict_fn.
    # SHAP already passes batch_size=10 to the explainer, but model_predict was still
    # processing one image at a time; now we batch the GPU forward pass.
    _SHAP_BATCH = 32
    def model_predict(imgs):
        # imgs shape: (N, H, W, C) float32 in [0, 1]
        results = []
        for i in range(0, len(imgs), _SHAP_BATCH):
            batch_imgs = imgs[i:i + _SHAP_BATCH]
            tensors = torch.cat([
                _preprocess_image(Image.fromarray((img * 255).astype(np.uint8)), model_type, processor, device)
                for img in batch_imgs
            ], dim=0)  # [B, C, H, W]
            with torch.no_grad():
                output = model(tensors)
                probs = torch.nn.functional.softmax(output, dim=1)
            results.append(probs.cpu().numpy())
        return np.concatenate(results, axis=0)

    # Create SHAP explainer with masker
    # Detect actual number of model outputs to avoid dimension mismatch
    _dummy = model_predict(np.expand_dims(img_np, axis=0))
    num_classes = _dummy.shape[1]
    # blur masker: masked regions are blurred rather than inpainted.
    # inpaint_telea often fails on natural images because it reconstructs
    # plausible content, leaving model confidence unchanged.
    # Gaussian blur consistently degrades local features and produces non-zero SHAP values.
    masker = shap.maskers.Image("blur(8,8)", img_np.shape)
    explainer = shap.Explainer(model_predict, masker, output_names=list(range(num_classes)))

    # Compute SHAP values
    shap_values = explainer(
        np.expand_dims(img_np, axis=0),
        max_evals=num_samples,
        batch_size=10
    )

    # Extract values for target class
    shap_vals = shap_values.values[0, :, :, :, target_class]

    # Create heatmap
    shap_heatmap = np.abs(shap_vals).mean(axis=-1)
    if shap_heatmap.max() == shap_heatmap.min():
        raise RuntimeError(
            f"SHAP heatmap is uniform (all values = {shap_heatmap.max():.6f}) for class {target_class}. "
            f"This means the model output did not change across {num_samples} perturbations. "
            f"Check target_class index and model output shape."
        )
    normalized_shap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min())

    # Statistics
    high_impact = float((normalized_shap > 0.6).sum() / normalized_shap.size)
    mean_impact = float(normalized_shap.mean())

    # Top coordinates
    top_k = 10
    flat_idx = np.argsort(normalized_shap.flatten())[-top_k:]
    top_coords = [
        {"y": int(np.unravel_index(idx, normalized_shap.shape)[0]),
         "x": int(np.unravel_index(idx, normalized_shap.shape)[1]),
         "value": float(normalized_shap.flatten()[idx])}
        for idx in flat_idx
    ]

    # Positive/negative SHAP statistics
    positive_shap = shap_vals[shap_vals > 0]
    negative_shap = shap_vals[shap_vals < 0]

    # Save visualization
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"shap_{image_id}_class{target_class}.png")
    img_uint8 = (img_np * 255).astype(np.uint8)
    _save_heatmap_visualization(img_uint8, normalized_shap, viz_path, "SHAP Values")

    heatmap_path = _save_raw_heatmap(normalized_shap, viz_path)

    # Compute suggested bounding box from top 1% of heatmap pixels
    shap_threshold = np.percentile(normalized_shap, 99)
    ys, xs = np.where(normalized_shap >= shap_threshold)
    shap_suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

    result = {
        "success": True,
        "method": "SHAP",
        "target_class": target_class,
        "num_samples": num_samples,
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "shap_values_shape": list(shap_values.values.shape),
        "image_id": image_id,
        "visualization_path": viz_path,
        "heatmap_path": heatmap_path,
        "suggested_bounding_box": shap_suggested_bbox,
        "statistics": {
            "high_impact_ratio": round(high_impact, 4),
            "mean_impact": round(mean_impact, 4),
            "max_impact": round(float(np.max(normalized_shap)), 4),
            "total_positive_shap": round(float(positive_shap.sum()), 4) if len(positive_shap) > 0 else 0,
            "total_negative_shap": round(float(negative_shap.sum()), 4) if len(negative_shap) > 0 else 0,
            "top_impact_coords": top_coords
        },
        "description": (
            f"SHAP analysis ({num_samples} evaluations) for class {target_class}: "
            f"{high_impact*100:.1f}% of regions show high impact (>0.6 threshold). "
            f"Mean SHAP impact: {mean_impact:.3f}. "
            f"Game-theoretic feature importance shows pixel contributions to prediction."
        )
    }

    return result


# ===================================================================
# Object Detection Implementation
# ===================================================================

def execute_object_detection(
    image: Image.Image,
    image_id: str = "temp",
    confidence_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Execute object detection using YOLO with visualization.

    This tool naturally returns bounding boxes as that's its purpose.
    """
    original_size = image.size

    # Load YOLO model; force float32 to prevent auto-fp16 on CUDA
    yolo_model = YOLO('yolov8n.pt')
    yolo_model.model.float()

    # Run detection (half=False: suppress ultralytics' fp16 auto-detection)
    results = yolo_model(image, conf=confidence_threshold, half=False)

    # Extract detection info
    detections = []
    img_np = np.array(image)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = yolo_model.names[cls]

            detections.append({
                "class_name": class_name,
                "class_id": cls,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

            # Draw on image
            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img_np, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sort by confidence
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # Save visualization
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"detection_{image_id}.png")
    cv2.imwrite(viz_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Group by class
    class_counts = {}
    for det in detections:
        cls = det["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    result = {
        "success": True,
        "method": "ObjectDetection",
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "num_detections": len(detections),
        "detections": detections,
        "confidence_threshold": confidence_threshold,
        "image_id": image_id,
        "visualization_path": viz_path,
        "statistics": {
            "total_detections": len(detections),
            "unique_classes": len(class_counts),
            "class_counts": class_counts,
            "avg_confidence": round(float(np.mean([d["confidence"] for d in detections])), 4) if detections else 0.0
        },
        "description": (
            f"Object detection found {len(detections)} objects "
            f"({len(class_counts)} unique classes). "
            f"Detected: {', '.join([f'{c}({n})' for c, n in list(class_counts.items())[:5]])}."
        )
    }

    return result


# ===================================================================
# Guided Backpropagation Implementation
# ===================================================================

def execute_guided_backprop(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute Guided Backpropagation with spatial alignment and noise reduction.
    """
    try:
        original_size = image.size
        img_np = np.array(image)
        model.eval()

        if input_tensor is None:
            input_tensor = _preprocess_image(image, model_type, processor, device)
        else:
            input_tensor = input_tensor.to(device).float()
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

        inplace_states = {}
        for name, module in model.named_modules():
            if hasattr(module, 'inplace') and module.inplace:
                inplace_states[name] = True
                module.inplace = False

        try:
            gbp = GuidedBackprop(model)
            attribution = gbp.attribute(input_tensor, target=target_class)
        finally:
            for name, module in model.named_modules():
                if name in inplace_states:
                    module.inplace = True

        # Process: [C, H, W]
        attr_np = attribution.squeeze(0).cpu().detach().numpy()
        
        # 1. Aggregation: Sum signed gradients across channels
        if len(attr_np.shape) == 3:
            attr_np = attr_np.sum(axis=0)
        
        # 2. Noise Reduction: Percentile clipping (99.5th)
        vmax = np.percentile(np.abs(attr_np), 99.5)
        attr_np = np.clip(attr_np, -vmax, vmax)
        
        # 3. Absolute importance
        heatmap = np.abs(attr_np)

        # 4. Resize and Visual Smoothing
        heatmap = cv2.resize(heatmap, (original_size[0], original_size[1]))
        sigma = max(0.5, original_size[0] * 0.005)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        # Save visualization
        output_dir = get_output_dir()
        viz_path = str(output_dir / f"guided_bp_{image_id}_class{target_class}.png")
        _save_heatmap_visualization(img_np, heatmap, viz_path, "Guided Backprop")
        heatmap_path = _save_raw_heatmap(heatmap, viz_path)

        mean_val = float(heatmap.mean())
        flat_idx = np.argsort(heatmap.flatten())[-10:]
        top_coords = [{"y": int(np.unravel_index(idx, heatmap.shape)[0]), "x": int(np.unravel_index(idx, heatmap.shape)[1]), "value": float(heatmap.flatten()[idx])} for idx in flat_idx]

        # Compute suggested bounding box from top 1% of heatmap pixels
        threshold = np.percentile(heatmap, 99)
        ys, xs = np.where(heatmap >= threshold)
        suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

        return {
            "success": True, "method": "GuidedBackprop", "target_class": target_class,
            "original_image_size": {"width": original_size[0], "height": original_size[1]},
            "image_id": image_id,
            "visualization_path": viz_path, "heatmap_path": heatmap_path,
            "suggested_bounding_box": suggested_bbox,
            "statistics": {"mean_importance": round(mean_val, 4), "top_importance_coords": top_coords},
            "description": f"Guided Backprop highlights high-resolution features for class {target_class}."
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc(), "method": "GuidedBackprop"}


# ===================================================================
# Layer CAM Implementation
# ===================================================================

def execute_layer_cam(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    layer_name: Optional[str] = None,
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute Layer CAM analysis.

    Returns raw results - feature extraction is done by Actor Agent.
    """
    original_size = image.size
    img_np = np.array(image)

    # Use provided tensor or preprocess
    if input_tensor is None:
        input_tensor = _preprocess_image(image, model_type, processor, device)
    else:
        input_tensor = input_tensor.to(device).float()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

    # Use specified layer or get target layer
    if layer_name:
        target_layer = dict(model.named_modules())[layer_name]
    else:
        target_layer = _get_target_layer(model)

    # Similar to GradCAM but for specific layer
    layer_cam = LayerGradCam(model, target_layer)
    attribution = layer_cam.attribute(input_tensor, target=target_class)

    # Process attribution
    attr_np = attribution.squeeze().cpu().detach().numpy()
    attr_np = np.maximum(attr_np, 0)
    attr_np = attr_np.mean(axis=0) if len(attr_np.shape) == 3 else attr_np

    # Resize and normalize
    heatmap = cv2.resize(attr_np, (original_size[0], original_size[1]))
    if heatmap.max() == heatmap.min():
        raise RuntimeError(
            f"LayerCAM attribution is uniform (all values = {heatmap.max():.6f}) for class {target_class}. "
            f"The target layer may have zero gradients. Check layer_name='{layer_name}' and target_class."
        )
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Save visualization
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"layercam_{image_id}_class{target_class}.png")
    _save_heatmap_visualization(img_np, heatmap, viz_path, "Layer CAM")

    heatmap_path = _save_raw_heatmap(heatmap, viz_path)

    # Compute statistics (same as GradCAM)
    high_attention_ratio = float((heatmap > 0.7).sum() / heatmap.size)
    mean_attention = float(heatmap.mean())
    max_attention = float(heatmap.max())

    # Find top attention coordinates (same as GradCAM)
    top_k = 10
    flat_idx = np.argsort(heatmap.flatten())[-top_k:]
    top_coords = [
        {"y": int(np.unravel_index(idx, heatmap.shape)[0]),
         "x": int(np.unravel_index(idx, heatmap.shape)[1]),
         "value": float(heatmap.flatten()[idx])}
        for idx in flat_idx
    ]

    # Compute suggested bounding box from top 1% of heatmap pixels
    threshold = np.percentile(heatmap, 99)
    ys, xs = np.where(heatmap >= threshold)
    suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

    result = {
        "success": True,
        "method": "LayerCAM",
        "target_class": target_class,
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "heatmap_shape": list(heatmap.shape),
        "image_id": image_id,
        "visualization_path": viz_path,
        "heatmap_path": heatmap_path,
        "suggested_bounding_box": suggested_bbox,
        "statistics": {
            "high_attention_ratio": round(high_attention_ratio, 4),
            "mean_attention": round(mean_attention, 4),
            "max_attention": round(max_attention, 4),
            "std_attention": round(float(np.std(heatmap)), 4),
            "layer_name": layer_name or "auto-detected",
            "top_attention_coords": top_coords
        },
        "description": (
            f"Layer CAM analysis for class {target_class}: "
            f"{high_attention_ratio*100:.1f}% of regions show high attention (>0.7 threshold). "
            f"Mean attention: {mean_attention:.3f}, Max: {max_attention:.3f}. "
            f"The model focuses on specific regions - see visualization for details."
        )
    }

    return result


# ===================================================================
# Sensitivity Analysis Implementation
# ===================================================================

def execute_sensitivity_analysis(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    perturbation_type: str = "noise",
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute sensitivity analysis to test model robustness.

    Returns raw results - feature extraction is done by Actor Agent.
    """
    original_size = image.size
    img_np = np.array(image)

    # Get original prediction - use provided tensor or preprocess
    if input_tensor is None:
        input_tensor = _preprocess_image(image, model_type, processor, device)
    else:
        input_tensor = input_tensor.to(device).float()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        original_output = model(input_tensor)
        original_prob = torch.nn.functional.softmax(original_output, dim=1)[0, target_class].item()

    # Apply perturbations
    perturbation_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    probabilities = [original_prob]

    for level in perturbation_levels[1:]:
        if perturbation_type == "noise":
            noise = np.random.normal(0, level * 255, img_np.shape)
            perturbed = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        else:  # blur
            ksize = int(level * 20) * 2 + 1
            perturbed = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

        perturbed_pil = Image.fromarray(perturbed)
        perturbed_tensor = _preprocess_image(perturbed_pil, model_type, processor, device)

        with torch.no_grad():
            output = model(perturbed_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)[0, target_class].item()
            probabilities.append(prob)

    # Visualize
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"sensitivity_{image_id}_class{target_class}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(perturbation_levels, probabilities, marker='o', linewidth=2)
    plt.xlabel('Perturbation Level')
    plt.ylabel('Prediction Probability')
    plt.title(f'Sensitivity Analysis ({perturbation_type})')
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate sensitivity metric
    prob_drop = original_prob - probabilities[-1]
    sensitivity = prob_drop / perturbation_levels[-1] if perturbation_levels[-1] > 0 else 0

    result = {
        "success": True,
        "method": "SensitivityAnalysis",
        "target_class": target_class,
        "perturbation_type": perturbation_type,
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "image_id": image_id,
        "visualization_path": viz_path,
        "statistics": {
            "original_probability": round(float(original_prob), 4),
            "final_probability": round(float(probabilities[-1]), 4),
            "probability_drop": round(float(prob_drop), 4),
            "sensitivity_score": round(float(sensitivity), 4),
            "perturbation_levels": perturbation_levels,
            "probabilities": [round(p, 4) for p in probabilities]
        },
        "description": (
            f"Model sensitivity to {perturbation_type}: "
            f"Probability dropped from {original_prob:.3f} to {probabilities[-1]:.3f} "
            f"({prob_drop*100:.1f}% decrease). "
            f"Sensitivity score: {sensitivity:.3f}."
        )
    }

    return result


# ===================================================================
# LIME for Text Implementation
# ===================================================================

def execute_lime_text(
    text_instance: str,
    model: Any,
    processor: Any,
    target_class: int,
    device: torch.device,
    instance_id: str = "temp",
    num_samples: int = 1000,
    num_features: int = 10
) -> Dict[str, Any]:
    """
    Execute LIME analysis for a text instance.

    Returns raw results - feature extraction is done by Actor Agent.
    """
    from lime import lime_text

    def predict_fn(texts: List[str]) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            inputs = processor(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            return probs.cpu().numpy()

    explainer = lime_text.LimeTextExplainer(class_names=list(range(predict_fn([text_instance]).shape[1])))

    explanation = explainer.explain_instance(
        text_instance,
        predict_fn,
        num_samples=num_samples,
        labels=(target_class,)
    )

    explanation_list = explanation.as_list(label=target_class)

    # Extract word importance (raw, for Agent to process)
    word_importance = []
    for word, weight in explanation_list[:num_features]:
        word_importance.append({
            "word": word,
            "weight": round(float(weight), 4),
            "direction": "positive" if weight > 0 else "negative"
        })

    # Save visualization
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"lime_text_{instance_id}.html")
    explanation.save_to_file(viz_path)

    result = {
        "success": True,
        "method": "LIME (Text)",
        "target_class": target_class,
        "num_samples": num_samples,
        "instance_id": instance_id,
        "visualization_path": viz_path,
        "text_length": len(text_instance),
        "statistics": {
            "num_important_words": len(word_importance),
            "word_importance": word_importance,
            "max_positive_weight": max([w["weight"] for w in word_importance if w["weight"] > 0], default=0),
            "max_negative_weight": min([w["weight"] for w in word_importance if w["weight"] < 0], default=0)
        },
        "description": (
            f"LIME text analysis for class {target_class}: "
            f"Found {len(word_importance)} important words. "
            f"Top positive: {word_importance[0]['word'] if word_importance else 'N/A'}. "
            f"See visualization for full word importance."
        )
    }

    return result


# ===================================================================
# SHAP for Text Implementation
# ===================================================================

def execute_shap_text(
    text_instance: str,
    model: Any,
    processor: Any,
    target_class: int,
    device: torch.device,
    instance_id: str = "temp"
) -> Dict[str, Any]:
    """
    Execute SHAP analysis for a text instance.

    Returns raw results - feature extraction is done by Actor Agent.
    """
    from transformers import pipeline

    model_device = 0 if device.type == 'cuda' else -1
    text_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=processor,
        device=model_device,
        return_all_scores=True
    )

    explainer = shap.Explainer(text_classifier)
    shap_values = explainer([text_instance])

    # Extract word importance
    output_names = shap_values.output_names
    output_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in output_names]
    target_class_name = model.config.id2label[target_class]
    target_index = output_names.index(target_class_name)

    words = shap_values.data[0]
    values = shap_values.values[0][:, target_index]

    word_importance = []
    for word, value in zip(words, values):
        word = str(word).strip()
        if word:
            word_importance.append({
                "word": word,
                "shap_value": round(float(value), 4),
                "direction": "positive" if value > 0 else "negative"
            })

    word_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    # Save visualization
    output_dir = get_output_dir()
    viz_path = str(output_dir / f"shap_text_{instance_id}.png")

    shap.plots.text(shap_values[0, :, target_index], display=False)
    plt.title(f"SHAP Explanation for Class '{model.config.id2label.get(target_class, 'Unknown')}'")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    result = {
        "success": True,
        "method": "SHAP (Text)",
        "target_class": target_class,
        "instance_id": instance_id,
        "visualization_path": viz_path,
        "statistics": {
            "num_tokens": len(word_importance),
            "word_importance": word_importance[:20],  # Top 20
            "total_positive_shap": sum(w["shap_value"] for w in word_importance if w["shap_value"] > 0),
            "total_negative_shap": sum(w["shap_value"] for w in word_importance if w["shap_value"] < 0)
        },
        "description": (
            f"SHAP text analysis for class {target_class}: "
            f"Analyzed {len(word_importance)} tokens. "
            f"Most influential: {word_importance[0]['word'] if word_importance else 'N/A'}. "
            f"Game-theoretic word importance calculated."
        )
    }

    return result


# ===================================================================
# SmoothGrad Implementation (Vision)
# ===================================================================

def execute_smoothgrad(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    n_samples: int = 100,
    stdevs: float = 0.1,
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute SmoothGrad analysis (NoiseTunnel + Saliency) with visualization.

    Returns raw results - feature extraction is done by Actor Agent.
    """
    original_size = image.size
    img_np = np.array(image)

    if input_tensor is None:
        input_tensor = _preprocess_image(image, model_type, processor, device)
    else:
        input_tensor = input_tensor.to(device).float()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)

    attribution = nt.attribute(
        input_tensor,
        target=target_class,
        nt_type='smoothgrad',
        nt_samples=n_samples,
        nt_samples_batch_size=4,  # process 4 noise samples at a time to avoid OOM
        stdevs=stdevs
    )

    attr_np = attribution.squeeze().cpu().detach().numpy()
    attr_np = np.abs(attr_np).mean(axis=0) if len(attr_np.shape) == 3 else np.abs(attr_np)

    heatmap = cv2.resize(attr_np, (original_size[0], original_size[1]))
    if heatmap.max() == heatmap.min():
        raise RuntimeError(
            f"SmoothGrad attribution is uniform (all values = {heatmap.max():.6f}) for class {target_class}. "
            f"Gradients may be zero. Check target_class and n_samples={n_samples}."
        )
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    output_dir = get_output_dir()
    viz_path = str(output_dir / f"smoothgrad_{image_id}_class{target_class}.png")
    _save_heatmap_visualization(img_np, heatmap, viz_path, "SmoothGrad")
    heatmap_path = _save_raw_heatmap(heatmap, viz_path)

    mean_attention = float(heatmap.mean())
    max_attention = float(heatmap.max())

    top_k = 10
    flat_idx = np.argsort(heatmap.flatten())[-top_k:]
    top_coords = [
        {"y": int(np.unravel_index(idx, heatmap.shape)[0]),
         "x": int(np.unravel_index(idx, heatmap.shape)[1]),
         "value": float(heatmap.flatten()[idx])}
        for idx in flat_idx
    ]

    # Compute suggested bounding box from top 1% of heatmap pixels
    threshold = np.percentile(heatmap, 99)
    ys, xs = np.where(heatmap >= threshold)
    suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

    return {
        "success": True,
        "method": "SmoothGrad",
        "target_class": target_class,
        "n_samples": n_samples,
        "stdevs": stdevs,
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "heatmap_shape": list(heatmap.shape),
        "image_id": image_id,
        "visualization_path": viz_path,
        "heatmap_path": heatmap_path,
        "suggested_bounding_box": suggested_bbox,
        "statistics": {
            "mean_attention": round(mean_attention, 4),
            "max_attention": round(max_attention, 4),
            "std_attention": round(float(np.std(heatmap)), 4),
            "top_attention_coords": top_coords
        },
        "description": (
            f"SmoothGrad analysis ({n_samples} samples, stdev={stdevs}) for class {target_class}: "
            f"Mean attention: {mean_attention:.3f}, Max: {max_attention:.3f}. "
            f"Noise-averaged gradients reduce artifacts and highlight robust features."
        )
    }


# ===================================================================
# Guided Backpropagation for Text
# ===================================================================

def execute_guided_backprop_text(
    text: str,
    model: Any,
    processor: Any,
    target_class: int,
    device: torch.device,
    instance_id: str = "temp",
    sample_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute Guided Backpropagation for text (gradient w.r.t. embeddings).

    Returns raw results - feature extraction is done by Actor Agent.
    """
    model.eval()

    # HuggingFace tokenizers expose convert_ids_to_tokens; simple callables don't
    is_hf_tokenizer = hasattr(processor, 'convert_ids_to_tokens')

    if is_hf_tokenizer:
        inputs = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)
        embedding_layer = model.get_input_embeddings()
        input_ids = inputs['input_ids']
        embeddings = embedding_layer(input_ids).detach().requires_grad_(True)
        outputs = model(inputs_embeds=embeddings, attention_mask=inputs.get('attention_mask'))
        score = outputs.logits[0, target_class]
        score.backward()
        grad = embeddings.grad.squeeze(0)  # (seq_len, hidden_dim)
        importance = grad.abs().mean(dim=-1).cpu().detach().numpy()
        tokens = processor.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    else:
        # Custom PyTorch model with model.embedding and simple tokenizer callable.
        # Use a forward hook to capture embedding gradients without model-specific forwarding.
        is_nli = bool(sample_data and 'hypothesis' in sample_data)
        token_ids = processor(text)  # List[int]
        token_ids_tensor = torch.tensor([token_ids], device=device)

        captured_embs = []

        def _hook(module, inp, out):
            out.retain_grad()
            captured_embs.append(out)

        handle = model.embedding.register_forward_hook(_hook)

        if is_nli:
            hyp_ids = processor(sample_data['hypothesis'])
            hyp_ids_tensor = torch.tensor([hyp_ids], device=device)
            output = model(token_ids_tensor, hyp_ids_tensor)
        else:
            output = model(token_ids_tensor)

        logits = output.logits if hasattr(output, 'logits') else output
        # Binary models output shape (B, 1); use sign flip for class 0
        if logits.shape[-1] == 1:
            score = logits[0, 0] if target_class == 1 else -logits[0, 0]
        else:
            score = logits[0, target_class]
        score.backward()
        handle.remove()

        # captured_embs[0] = premise (or single-input) embeddings; shape (1, seq_len, embed_dim)
        prem_emb = captured_embs[0].squeeze(0).detach()   # (seq_len, embed_dim)
        prem_grad = captured_embs[0].grad.squeeze(0)      # (seq_len, embed_dim)
        # grad×input: position-specific even for mean-pool models where plain |grad| is uniform
        importance = (prem_grad * prem_emb).abs().mean(dim=-1).cpu().detach().numpy()

        # Trim to actual words (padding positions carry negligible info)
        words = text.lower().split()
        n_words = min(len(words), len(importance))
        tokens = words[:n_words]
        importance = importance[:n_words]

    if importance.max() == importance.min():
        raise RuntimeError(
            f"GuidedBackprop (Text) attribution is uniform (all values = {importance.max():.6f}) for class {target_class}. "
            f"Gradients may be zero. Check target_class and model compatibility."
        )
    importance = (importance - importance.min()) / (importance.max() - importance.min())

    token_importance = [
        {"token": tok, "importance": round(float(imp), 4)}
        for tok, imp in zip(tokens, importance)
    ]
    token_importance_sorted = sorted(token_importance, key=lambda x: x["importance"], reverse=True)

    output_dir = get_output_dir()
    viz_path = str(output_dir / f"guided_bp_text_{instance_id}.png")

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.4), 3))
    ax.bar(range(len(tokens)), importance)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_title(f"Guided Backprop (Text) - Class {target_class}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "success": True,
        "method": "GuidedBackprop (Text)",
        "target_class": target_class,
        "instance_id": instance_id,
        "visualization_path": viz_path,
        "statistics": {
            "num_tokens": len(tokens),
            "top_tokens": token_importance_sorted[:10],
            "mean_importance": round(float(importance.mean()), 4),
            "max_importance": round(float(importance.max()), 4)
        },
        "description": (
            f"Guided Backprop text analysis for class {target_class}: "
            f"Analyzed {len(tokens)} tokens. "
            f"Most important: {token_importance_sorted[0]['token'] if token_importance_sorted else 'N/A'}."
        )
    }


# ===================================================================
# Guided Backpropagation for Tabular
# ===================================================================

def execute_guided_backprop_tabular(
    features: torch.Tensor,
    model: Any,
    feature_names: List[str],
    target_class: int,
    device: torch.device,
    encoded_to_original: Optional[Dict] = None,
    raw_features_dict: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute Guided Backpropagation for tabular data (gradient w.r.t. input features).

    Returns raw results - feature extraction is done by Actor Agent.
    """
    model.eval()
    x = features.to(device).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.detach().requires_grad_(True)

    output = model(x)
    if output.shape[1] == 1:
        score = output[0, 0]
    else:
        score = output[0, target_class]
    score.backward()

    grad = x.grad.squeeze(0).cpu().detach().numpy()
    importance = np.abs(grad)

    if importance.max() == importance.min():
        raise RuntimeError(
            f"GuidedBackprop (Tabular) attribution is uniform (all values = {importance.max():.6f}) for class {target_class}. "
            f"Gradients may be zero. Check target_class and model output shape."
        )
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min())

    feature_importance = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "importance": round(float(importance_norm[i]), 4),
            "gradient": round(float(grad[i]), 6)
        }
        for i in range(len(importance))
    ]
    feature_importance_sorted = sorted(feature_importance, key=lambda x: x["importance"], reverse=True)

    output_dir = get_output_dir()
    viz_path = str(output_dir / f"guided_bp_tabular_class{target_class}.png")

    top_n = min(20, len(feature_importance_sorted))
    top_feats = feature_importance_sorted[:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    ax.barh([f["feature"] for f in reversed(top_feats)], [f["importance"] for f in reversed(top_feats)])
    ax.set_title(f"Guided Backprop (Tabular) - Class {target_class}")
    ax.set_xlabel("Normalized Importance")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "success": True,
        "method": "GuidedBackprop (Tabular)",
        "target_class": target_class,
        "visualization_path": viz_path,
        "statistics": {
            "num_features": len(feature_importance),
            "top_features": feature_importance_sorted[:10],
            "mean_importance": round(float(importance_norm.mean()), 4),
            "max_importance": round(float(importance_norm.max()), 4)
        },
        "description": (
            f"Guided Backprop tabular analysis for class {target_class}: "
            f"Analyzed {len(feature_importance)} features. "
            f"Most important: {feature_importance_sorted[0]['feature'] if feature_importance_sorted else 'N/A'}."
        )
    }


# ===================================================================
# SmoothGrad for Text
# ===================================================================

def execute_smoothgrad_text(
    text: str,
    model: Any,
    processor: Any,
    target_class: int,
    device: torch.device,
    instance_id: str = "temp",
    n_samples: int = 100,
    stdevs: float = 0.1,
    sample_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute SmoothGrad for text (noise-averaged gradients w.r.t. embeddings).

    Returns raw results - feature extraction is done by Actor Agent.
    """
    model.eval()
    inputs = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)

    embedding_layer = model.get_input_embeddings()
    input_ids = inputs['input_ids']
    base_embeddings = embedding_layer(input_ids).detach()

    # SmoothGrad: average gradients over n_samples noisy embeddings
    all_grads = []
    for _ in range(n_samples):
        noise = torch.randn_like(base_embeddings) * stdevs
        noisy_embeddings = (base_embeddings + noise).requires_grad_(True)

        outputs = model(inputs_embeds=noisy_embeddings, attention_mask=inputs.get('attention_mask'))
        score = outputs.logits[0, target_class]
        score.backward()

        all_grads.append(noisy_embeddings.grad.squeeze(0).abs().mean(dim=-1).cpu().detach().numpy())

    importance = np.mean(all_grads, axis=0)  # (seq_len,)

    if importance.max() == importance.min():
        raise RuntimeError(
            f"SmoothGrad (Text) attribution is uniform (all values = {importance.max():.6f}) for class {target_class}. "
            f"Gradients may be zero. Check target_class and n_samples={n_samples}."
        )
    importance = (importance - importance.min()) / (importance.max() - importance.min())

    tokens = processor.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    token_importance = [
        {"token": tok, "importance": round(float(imp), 4)}
        for tok, imp in zip(tokens, importance)
    ]
    token_importance_sorted = sorted(token_importance, key=lambda x: x["importance"], reverse=True)

    output_dir = get_output_dir()
    viz_path = str(output_dir / f"smoothgrad_text_{instance_id}.png")

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.4), 3))
    ax.bar(range(len(tokens)), importance)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_title(f"SmoothGrad (Text) - Class {target_class} ({n_samples} samples)")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "success": True,
        "method": "SmoothGrad (Text)",
        "target_class": target_class,
        "instance_id": instance_id,
        "n_samples": n_samples,
        "stdevs": stdevs,
        "visualization_path": viz_path,
        "statistics": {
            "num_tokens": len(tokens),
            "top_tokens": token_importance_sorted[:10],
            "mean_importance": round(float(importance.mean()), 4),
            "max_importance": round(float(importance.max()), 4)
        },
        "description": (
            f"SmoothGrad text analysis ({n_samples} samples) for class {target_class}: "
            f"Analyzed {len(tokens)} tokens. "
            f"Most important: {token_importance_sorted[0]['token'] if token_importance_sorted else 'N/A'}."
        )
    }


# ===================================================================
# SmoothGrad for Tabular
# ===================================================================

def execute_smoothgrad_tabular(
    features: torch.Tensor,
    model: Any,
    feature_names: List[str],
    target_class: int,
    device: torch.device,
    n_samples: int = 100,
    stdevs: float = 0.1,
    encoded_to_original: Optional[Dict] = None,
    raw_features_dict: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute SmoothGrad for tabular data (noise-averaged gradients w.r.t. input features).

    Returns raw results - feature extraction is done by Actor Agent.
    """
    model.eval()
    x_base = features.to(device).float()
    if x_base.dim() == 1:
        x_base = x_base.unsqueeze(0)
    x_base = x_base.detach()

    all_grads = []
    for _ in range(n_samples):
        noise = torch.randn_like(x_base) * stdevs
        x_noisy = (x_base + noise).requires_grad_(True)

        output = model(x_noisy)
        if output.shape[1] == 1:
            score = output[0, 0]
        else:
            score = output[0, target_class]
        score.backward()

        all_grads.append(x_noisy.grad.squeeze(0).abs().cpu().detach().numpy())

    importance = np.mean(all_grads, axis=0)

    if importance.max() == importance.min():
        raise RuntimeError(
            f"SmoothGrad (Tabular) attribution is uniform (all values = {importance.max():.6f}) for class {target_class}. "
            f"Gradients may be zero. Check target_class and n_samples={n_samples}."
        )
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min())

    feature_importance = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "importance": round(float(importance_norm[i]), 4),
            "raw_gradient": round(float(importance[i]), 6)
        }
        for i in range(len(importance))
    ]
    feature_importance_sorted = sorted(feature_importance, key=lambda x: x["importance"], reverse=True)

    output_dir = get_output_dir()
    viz_path = str(output_dir / f"smoothgrad_tabular_class{target_class}.png")

    top_n = min(20, len(feature_importance_sorted))
    top_feats = feature_importance_sorted[:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    ax.barh([f["feature"] for f in reversed(top_feats)], [f["importance"] for f in reversed(top_feats)])
    ax.set_title(f"SmoothGrad (Tabular) - Class {target_class} ({n_samples} samples)")
    ax.set_xlabel("Normalized Importance")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "success": True,
        "method": "SmoothGrad (Tabular)",
        "target_class": target_class,
        "n_samples": n_samples,
        "stdevs": stdevs,
        "visualization_path": viz_path,
        "statistics": {
            "num_features": len(feature_importance),
            "top_features": feature_importance_sorted[:10],
            "mean_importance": round(float(importance_norm.mean()), 4),
            "max_importance": round(float(importance_norm.max()), 4)
        },
        "description": (
            f"SmoothGrad tabular analysis ({n_samples} samples) for class {target_class}: "
            f"Analyzed {len(feature_importance)} features. "
            f"Most important: {feature_importance_sorted[0]['feature'] if feature_importance_sorted else 'N/A'}."
        )
    }
