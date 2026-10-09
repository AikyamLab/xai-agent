import torch
import timm # Added
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from PIL import Image
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATASET_ROOT = "dataset/image/stl-10"
DATASET_ROOT = DEFAULT_DATASET_ROOT
NUM_CLASSES = 10

LABEL_MAP = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",
    8: "ship",
    9: "truck"
}


def get_transform():
    """Returns the image transform for STL-10 DenseNet model."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# Module-level dataset cache.
_dataset_cache: dict = {}


def _resolve_dataset_root(dataset_dir: str) -> str:
    base = Path(dataset_dir).expanduser()
    candidates = [
        base / "image" / "stl-10",
        base / "stl-10",
        base,
    ]
    for candidate in candidates:
        if (candidate / "stl10_binary").exists():
            return str(candidate)
    return str(candidates[0])


def set_dataset_root(dataset_dir: str) -> None:
    """Configure STL-10 root dynamically (supports dataset and dataset_ood layouts)."""
    global DATASET_ROOT
    DATASET_ROOT = _resolve_dataset_root(dataset_dir)
    _dataset_cache.clear()


def load_model(model_path: str):
    """
    Loads the STL-10 Densenet model.

    Args:
        model_path (str): The path to the .pth model file.

    Returns:
        tuple: A tuple containing the loaded model and the image transform.
    """
    model = timm.create_model("densenet121", pretrained=False, num_classes=NUM_CLASSES)
    # Load to CPU first; use assign=True so meta-device parameters (newer timm) are
    # replaced rather than copied-into, avoiding "Cannot copy out of meta tensor".
    model = model.to_empty(device=DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    transform = get_transform()

    return model, transform


def load_data(index: int, split: str = "test") -> Dict[str, Any]:
    """
    Load a specific sample from the STL-10 dataset.

    Args:
        index: Index of the sample to load
        split: Dataset split ('train' or 'test')

    Returns:
        Dict containing:
            - image: PIL Image (raw, before transform)
            - image_tensor: Transformed tensor ready for model
            - label: Ground truth label index
            - label_name: Human-readable label name
            - index: Sample index
    """
    transform = get_transform()

    key_t = f"{split}_with_transform"
    key_r = f"{split}_raw"
    if key_t not in _dataset_cache:
        _dataset_cache[key_t] = STL10(root=DATASET_ROOT, split=split, download=False, transform=transform)
    if key_r not in _dataset_cache:
        _dataset_cache[key_r] = STL10(root=DATASET_ROOT, split=split, download=False, transform=None)

    dataset_with_transform = _dataset_cache[key_t]
    dataset_raw             = _dataset_cache[key_r]

    if index < 0 or index >= len(dataset_raw):
        raise IndexError(f"Index {index} out of range. Dataset has {len(dataset_raw)} samples.")

    # Get raw image and label
    raw_image, label = dataset_raw[index]
    label = int(label)

    # Get transformed tensor
    image_tensor, _ = dataset_with_transform[index]

    return {
        "image": raw_image,  # PIL Image
        "image_tensor": image_tensor,  # Transformed tensor [C, H, W]
        "label": label,
        "label_name": LABEL_MAP.get(label, f"class_{label}"),
        "index": index,
        "split": split
    }


def predict(
    model: torch.nn.Module,
    image: Union[Image.Image, torch.Tensor],
    transform: Optional[transforms.Compose] = None
) -> Dict[str, Any]:
    """
    Make prediction on an image using the loaded model.

    Args:
        model: Loaded PyTorch model
        image: Either a PIL Image or a pre-transformed tensor
        transform: Transform to apply if image is PIL Image (optional)

    Returns:
        Dict containing prediction results
    """
    model.eval()

    # Prepare input tensor
    if isinstance(image, Image.Image):
        if transform is None:
            transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE).float()
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            input_tensor = image.unsqueeze(0).to(DEVICE).float()
        else:
            input_tensor = image.to(DEVICE).float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Make prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

        # Get top-5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, min(5, NUM_CLASSES))

        predictions = []
        for i in range(len(top5_idx)):
            predictions.append({
                "class_idx": int(top5_idx[i].item()),
                "class_name": LABEL_MAP.get(int(top5_idx[i].item()), f"class_{top5_idx[i].item()}"),
                "probability": float(top5_prob[i].item())
            })

    predicted_class = int(top5_idx[0].item())

    return {
        "success": True,
        "predicted_class_idx": predicted_class,
        "predicted_class_name": LABEL_MAP.get(predicted_class, f"class_{predicted_class}"),
        "confidence": float(top5_prob[0].item()),
        "top5_predictions": predictions,
        "probabilities": probabilities.cpu().numpy(),
        "device": str(DEVICE)
    }


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Args:
        model: Loaded PyTorch model

    Returns:
        Dict containing model information
    """
    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(DEVICE),
        "label_map": LABEL_MAP
    }


# Main function for testing
if __name__ == "__main__":
    model_path = "models_to_read/vision/stl10_densenet.pth"

    # Load model
    model, transform = load_model(model_path)
    print(f"Model loaded on {DEVICE}")
    print(get_model_info(model))

    # Test on first 5 samples
    for idx in range(5):
        data = load_data(idx, split="test")
        print(f"\nSample {idx}:")
        print(f"  Ground truth: {data['label_name']} (class {data['label']})")

        # Predict using raw image
        result = predict(model, data['image'], transform)
        print(f"  Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")
        print(f"  Correct: {result['predicted_class_idx'] == data['label']}")
