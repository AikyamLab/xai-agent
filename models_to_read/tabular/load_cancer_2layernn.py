import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 32
RANDOM_SEED = 42
NUM_CLASSES = 2

LABEL_MAP = {
    0: "malignant",
    1: "benign"
}

# Cache for preprocessor and data to avoid reloading
_cache = {
    "scaler": None,
    "X_full": None,
    "y_full": None,
    "X_test": None,
    "y_test": None,
    "feature_names": None,
}


class TwoLayerNN(nn.Module):
    """Cancer 2-layer NN — NO BatchNorm (matches cancer_2layernn.pth weights)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def _load_and_preprocess_data():
    """Load and preprocess the Breast Cancer Wisconsin dataset. Results are cached."""
    if _cache["scaler"] is not None:
        return

    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()

    X = data.data          # shape: (569, 30)
    y = data.target        # 0 = malignant, 1 = benign
    feature_names = list(data.feature_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    _cache["scaler"] = scaler
    _cache["X_full"] = torch.tensor(X_scaled, dtype=torch.float32)
    _cache["y_full"] = torch.tensor(y, dtype=torch.long)
    _cache["X_test"] = torch.tensor(X_test, dtype=torch.float32)
    _cache["y_test"] = torch.tensor(y_test, dtype=torch.long)
    _cache["feature_names"] = feature_names


def load_model(model_path: str):
    """
    Load the Breast Cancer 2-layer NN model.

    Args:
        model_path: Path to the .pth model file.

    Returns:
        tuple: (model, scaler)
    """
    _load_and_preprocess_data()
    scaler = _cache["scaler"]

    input_size = _cache["X_full"].shape[1]  # 30
    model = TwoLayerNN(input_size, HIDDEN_SIZE).to(DEVICE)

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler


def load_data(index: int, split: str = "test") -> Dict[str, Any]:
    """
    Load a specific sample from the Breast Cancer dataset.

    Args:
        index: Positional index. Meaning depends on split:
            - split='test': positional index into the test set (Q1-Q3, Q5-Q7)
            - split='full': positional index into the full dataset (Q4, Q8-Q10)
            - split='train': positional index into the train set
        split: 'test', 'train', or 'full'.

    Returns:
        Dict containing features, label, feature_names, etc.
    """
    _load_and_preprocess_data()

    if split == 'full':
        X = _cache["X_full"]
        y = _cache["y_full"]
    elif split == "train":
        # Not cached separately; re-derive if needed — but unlikely path
        X = _cache["X_full"]
        y = _cache["y_full"]
    else:
        X = _cache["X_test"]
        y = _cache["y_test"]

    if index < 0 or index >= len(X):
        raise IndexError(f"Index {index} out of range for split='{split}'. Dataset has {len(X)} samples.")

    features = X[index]
    label = int(y[index].item())
    feature_names = _cache["feature_names"]

    # Build feature dict
    features_dict = {}
    for i, name in enumerate(feature_names):
        features_dict[name] = float(features[i].item())

    return {
        "features": features,           # Tensor [30]
        "features_dict": features_dict,  # {feature_name: value}
        "label": label,
        "label_name": LABEL_MAP.get(label, f"class_{label}"),
        "index": index,
        "split": split,
        "feature_names": feature_names
    }


def predict(
    model: nn.Module,
    input_data: Union[torch.Tensor, np.ndarray],
    preprocessor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Make prediction on tabular input using the loaded model.

    Args:
        model: Loaded PyTorch model.
        input_data: Feature tensor (1D or 2D) or numpy array.
        preprocessor: Unused for pre-processed data, kept for interface compatibility.

    Returns:
        Dict containing prediction results.
    """
    model.eval()

    # Prepare input tensor
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
    elif isinstance(input_data, torch.Tensor):
        input_tensor = input_data.float()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        prob_positive = model(input_tensor).squeeze()
        prob_positive = float(prob_positive.item())
        prob_negative = 1.0 - prob_positive

        probabilities = np.array([prob_negative, prob_positive])
        predicted_class = 1 if prob_positive >= 0.5 else 0

        predictions = []
        for cls_idx in range(NUM_CLASSES):
            predictions.append({
                "class_idx": cls_idx,
                "class_name": LABEL_MAP.get(cls_idx, f"class_{cls_idx}"),
                "probability": float(probabilities[cls_idx])
            })
        # Sort by probability descending
        predictions.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "success": True,
        "predicted_class_idx": predicted_class,
        "predicted_class_name": LABEL_MAP.get(predicted_class, f"class_{predicted_class}"),
        "confidence": float(probabilities[predicted_class]),
        "top5_predictions": predictions,
        "probabilities": probabilities,
        "device": str(DEVICE)
    }


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Args:
        model: Loaded PyTorch model.

    Returns:
        Dict containing model information.
    """
    _load_and_preprocess_data()

    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(DEVICE),
        "label_map": LABEL_MAP,
        "feature_names": _cache["feature_names"],
        "num_features": len(_cache["feature_names"]) if _cache["feature_names"] else 0
    }


# Main function for testing
if __name__ == "__main__":
    model_path = "/sfs/ceph/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/models_to_read/tabular/cancer_2layernn.pth"

    # Load model
    model, scaler = load_model(model_path)
    print(f"Model loaded on {DEVICE}")
    print(get_model_info(model))

    # Test with test-set positional indices (Q1-Q7 style)
    print("\n--- Test set samples (Q1-Q7 style) ---")
    for idx in range(3):
        data = load_data(idx, split="test")
        print(f"\nTest[{idx}]:")
        print(f"  Ground truth: {data['label_name']} (class {data['label']})")
        result = predict(model, data['features'], scaler)
        print(f"  Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")

    # Test with full-dataset positional indices (Q4/Q8-Q10 style)
    print("\n--- Full dataset samples (Q4/Q8-Q10 style) ---")
    for idx in [541, 64, 73]:
        data = load_data(idx, split="full")
        print(f"\nFull[{idx}]:")
        print(f"  Ground truth: {data['label_name']} (class {data['label']})")
        result = predict(model, data['features'], scaler)
        print(f"  Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")
