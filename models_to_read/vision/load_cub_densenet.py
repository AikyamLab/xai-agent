import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Any, Optional, Union
import os
import pandas as pd
from collections import OrderedDict
from pathlib import Path

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATASET_ROOT = "dataset/image/CUB_200_2011"
DATASET_ROOT = DEFAULT_DATASET_ROOT
NUM_CLASSES = 200

def get_label_map(root_path):
    """Parses classes.txt to create a label map."""
    classes_path = os.path.join(root_path, 'classes.txt')
    if not os.path.exists(classes_path):
        return {}
    label_map = {}
    with open(classes_path, 'r') as f:
        for line in f:
            idx, class_name = line.strip().split()
            # The CUB dataset labels are 1-based in the files, but 0-based in the loader
            label_map[int(idx) - 1] = class_name
    return label_map

LABEL_MAP = get_label_map(DATASET_ROOT)

# Module-level dataset cache.
_dataset_cache: dict = {}


def _resolve_dataset_root(dataset_dir: str) -> str:
    base = Path(dataset_dir).expanduser()
    candidates = [
        base / "image" / "CUB_200_2011",
        base / "CUB_200_2011",
        base,
    ]
    for candidate in candidates:
        if (candidate / "classes.txt").exists():
            return str(candidate)
    return str(candidates[0])


def set_dataset_root(dataset_dir: str) -> None:
    """Configure CUB root dynamically (supports dataset and dataset_ood layouts)."""
    global DATASET_ROOT, LABEL_MAP
    DATASET_ROOT = _resolve_dataset_root(dataset_dir)
    LABEL_MAP = get_label_map(DATASET_ROOT)
    _dataset_cache.clear()


class CUB_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_type='test', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        df_img = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
        df_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
        df_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
        df = pd.concat([df_img, df_label, df_split], axis=1)
        # relabel
        df['Label'] = df['Label'] - 1

        if dataset_type == 'test':
            df = df[df['Train'] == 0]
        elif dataset_type == 'train':
            df = df[df['Train'] == 1]
        else:
            raise ValueError('Unsupported dataset_type!')

        self.img_name_list = df['Image'].tolist()
        self.label_list = df['Label'].tolist()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.img_name_list[idx])
        image = Image.open(img_path).convert('RGB')
        target = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


def get_transform():
    """Returns the image transform for CUBS DenseNet model."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path: str):
    """
    Loads the CUBS DenseNet model.

    Args:
        model_path (str): The path to the .pth model file.

    Returns:
        tuple: A tuple containing the loaded model and the image transform.
    """
    model = models.densenet201(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(1920, 256)),
        ('norm0', nn.BatchNorm1d(256)),
        ('relu0', nn.ReLU(inplace=True)),
        ('fc1', nn.Linear(256, NUM_CLASSES))
    ]))

    model.classifier = classifier
    model = model.to_empty(device=DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    transform = get_transform()

    return model, transform


def load_data(index: int, split: str = "test") -> Dict[str, Any]:
    """
    Load a specific sample from the CUBS dataset.

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
        _dataset_cache[key_t] = CUB_Dataset(root=DATASET_ROOT, dataset_type=split, transform=transform)
    if key_r not in _dataset_cache:
        _dataset_cache[key_r] = CUB_Dataset(root=DATASET_ROOT, dataset_type=split, transform=None)

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
    try:
        model.eval()

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

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

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
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
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
