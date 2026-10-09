import torch
import timm
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Union

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATASET_ROOT = "dataset_ood/image/cifar-10"
DATASET_ROOT = DEFAULT_DATASET_ROOT
NUM_CLASSES = 10

LABEL_MAP = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

_dataset_cache: dict = {}


def _resolve_dataset_root(dataset_dir: str) -> str:
    base = Path(dataset_dir).expanduser()
    candidates = [
        base / "image" / "cifar-10",
        base / "cifar-10",
        base,
    ]
    for candidate in candidates:
        if (candidate / "cifar-10-batches-py").exists():
            return str(candidate)
    return str(candidates[0])


def set_dataset_root(dataset_dir: str) -> None:
    """Configure CIFAR root dynamically (supports dataset and dataset_ood layouts)."""
    global DATASET_ROOT
    DATASET_ROOT = _resolve_dataset_root(dataset_dir)
    _dataset_cache.clear()


def get_transform():
    """Returns the image transform for CIFAR-10 ResNet model."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


def load_model(model_path: str):
    """
    Loads the CIFAR-10 ResNet model.

    Args:
        model_path (str): Kept for interface compatibility. Not used because this
            loader uses pretrained timm weights via detectors registration.

    Returns:
        tuple: (model, transform)
    """
    # Required for timm model registration on your environment.
    import detectors  # noqa: F401

    model = timm.create_model("resnet18_cifar10", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model.eval()
    return model, get_transform()


def load_data(index: int, split: str = "test") -> Dict[str, Any]:
    """
    Load a specific sample from CIFAR-10.

    Args:
        index: Index of the sample to load
        split: Dataset split ('train' or 'test')

    Returns:
        Dict containing image, transformed tensor, labels, and metadata.
    """
    transform = get_transform()
    train_split = (split == "train")

    key_t = f"{split}_with_transform"
    key_r = f"{split}_raw"
    if key_t not in _dataset_cache:
        _dataset_cache[key_t] = torchvision.datasets.CIFAR10(
            root=DATASET_ROOT,
            train=train_split,
            download=True,
            transform=transform,
        )
    if key_r not in _dataset_cache:
        _dataset_cache[key_r] = torchvision.datasets.CIFAR10(
            root=DATASET_ROOT,
            train=train_split,
            download=True,
            transform=None,
        )

    dataset_with_transform = _dataset_cache[key_t]
    dataset_raw = _dataset_cache[key_r]

    if index < 0 or index >= len(dataset_raw):
        raise IndexError(f"Index {index} out of range. Dataset has {len(dataset_raw)} samples.")

    raw_image, label = dataset_raw[index]
    label = int(label)
    image_tensor, _ = dataset_with_transform[index]

    return {
        "image": raw_image,
        "image_tensor": image_tensor,
        "label": label,
        "label_name": LABEL_MAP.get(label, f"class_{label}"),
        "index": index,
        "split": split,
    }


def predict(
    model: torch.nn.Module,
    image: Union[Image.Image, torch.Tensor],
    transform: Optional[transforms.Compose] = None
) -> Dict[str, Any]:
    """Make prediction on an image using the loaded model."""
    if isinstance(image, Image.Image):
        if transform is None:
            transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            input_tensor = image.unsqueeze(0).to(DEVICE)
        else:
            input_tensor = image.to(DEVICE)
    else:
        raise TypeError("Image must be PIL Image or torch.Tensor")

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class_idx = int(pred_idx.item())
    confidence_val = float(confidence.item())

    return {
        "success": True,
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_name": LABEL_MAP.get(predicted_class_idx, f"class_{predicted_class_idx}"),
        "confidence": confidence_val,
        "probabilities": probs.cpu().numpy().flatten().tolist(),
    }


def get_model_info(model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """Returns metadata for the CIFAR-10 ResNet model."""
    info = {
        "model_name": "resnet18_cifar10",
        "model_type": "timm",
        "num_classes": NUM_CLASSES,
        "label_map": LABEL_MAP,
        "dataset_root": DATASET_ROOT,
    }

    if model is not None:
        info.update({
            "architecture": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        })

    return info
