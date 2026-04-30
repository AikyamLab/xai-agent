import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
NUM_CLASSES = 2

LABEL_MAP = {
    0: "bad",
    1: "good",
}

_cache = {
    "preprocessor": None,
    "df": None,
    "y_series": None,
    "X_all": None,
    "train_indices": None,
    "test_indices": None,
    "feature_names": None,
    "encoded_to_original": None,
    "orig_to_pos": None,
}


class ThreeLayerNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)  # logits


def _load_and_preprocess_data():
    if _cache["preprocessor"] is not None:
        return

    print("Loading German Credit dataset...")
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = data.frame.copy()

    df["class"] = df["class"].map({"good": 1, "bad": 0})
    X = df.drop(columns=["class"])
    y = df["class"].astype(int)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    X_processed = preprocessor.fit_transform(X)

    if hasattr(X_processed, "toarray"):
        X_array = X_processed.toarray()
    else:
        X_array = np.asarray(X_processed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    # Build feature names
    feature_names = num_cols.copy()
    cat_encoder = preprocessor.named_transformers_["cat"]
    for col, categories in zip(cat_cols, cat_encoder.categories_):
        for category in categories:
            feature_names.append(f"{col}_{category}")

    encoded_to_original = {}
    for name in num_cols:
        encoded_to_original[name] = name
    for col, categories in zip(cat_cols, cat_encoder.categories_):
        for category in categories:
            encoded_to_original[f"{col}_{category}"] = col

    _cache["df"] = df
    _cache["y_series"] = y
    _cache["X_all"] = torch.tensor(X_array, dtype=torch.float32)
    _cache["train_indices"] = set(y_train.index.tolist())
    _cache["test_indices"] = set(y_test.index.tolist())
    _cache["feature_names"] = feature_names
    _cache["encoded_to_original"] = encoded_to_original
    _cache["orig_to_pos"] = {orig_idx: pos for pos, orig_idx in enumerate(df.index.tolist())}
    _cache["preprocessor"] = preprocessor


def load_model(model_path: str):
    """
    Load German Credit 3-layer neural network.

    Args:
        model_path: Path to checkpoint (.pth).

    Returns:
        tuple: (model, preprocessor)
    """
    _load_and_preprocess_data()
    preprocessor = _cache["preprocessor"]

    input_dim = _cache["X_all"].shape[1]
    model = ThreeLayerNN(input_dim).to(DEVICE)

    print(f"Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return model, preprocessor


def load_data(index: int, split: str = "test") -> Dict[str, Any]:
    """
    Load a specific sample by original dataframe index.

    Args:
        index: Original row index from dataset JSON.
        split: train/test/full (kept for compatibility).

    Returns:
        Dict with features, labels and metadata.
    """
    _load_and_preprocess_data()

    orig_to_pos = _cache["orig_to_pos"]
    if index not in orig_to_pos:
        raise IndexError(f"Index {index} not found in German Credit dataset.")

    pos = orig_to_pos[index]
    features = _cache["X_all"][pos]
    label = int(_cache["y_series"].iloc[pos])
    feature_names = _cache["feature_names"]

    features_dict = {name: float(features[i].item()) for i, name in enumerate(feature_names)}

    raw_features_dict = {}
    for col in _cache["df"].columns:
        if col == "class":
            continue
        value = _cache["df"].loc[index, col]
        try:
            raw_features_dict[col] = value.item()
        except AttributeError:
            raw_features_dict[col] = str(value)

    return {
        "features": features,
        "features_dict": features_dict,
        "raw_features_dict": raw_features_dict,
        "encoded_to_original": _cache["encoded_to_original"],
        "label": label,
        "label_name": LABEL_MAP.get(label, f"class_{label}"),
        "index": index,
        "split": split,
        "feature_names": feature_names,
    }


def predict(
    model: nn.Module,
    input_data: Union[torch.Tensor, np.ndarray],
    preprocessor: Optional[Any] = None
) -> Dict[str, Any]:
    """Make prediction on tabular input using BCE-with-logits output."""
    model.eval()

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
        logits = model(input_tensor).squeeze()
        prob_positive = float(torch.sigmoid(logits).item())
        prob_negative = 1.0 - prob_positive

    probabilities = np.array([prob_negative, prob_positive])
    predicted_class = 1 if prob_positive >= 0.5 else 0
    confidence = prob_positive if predicted_class == 1 else prob_negative

    predictions = []
    for cls_idx in range(NUM_CLASSES):
        predictions.append({
            "class_idx": cls_idx,
            "class_name": LABEL_MAP.get(cls_idx, f"class_{cls_idx}"),
            "probability": float(probabilities[cls_idx]),
        })
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "success": True,
        "predicted_class_idx": predicted_class,
        "predicted_class_name": LABEL_MAP[predicted_class],
        "confidence": float(confidence),
        "probabilities": probabilities.tolist(),
        "top5_predictions": predictions[:5],
        "raw_logits": float(logits.item()),
        "device": str(DEVICE),
    }


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(DEVICE),
        "label_map": LABEL_MAP,
        "feature_names": _cache["feature_names"],
        "num_features": len(_cache["feature_names"]) if _cache["feature_names"] else 0,
    }
