import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
RANDOM_SEED = 42
NUM_CLASSES = 2

LABEL_MAP = {
    0: "<=50K",
    1: ">50K"
}

# Cache for preprocessor and data to avoid reloading
_cache = {
    "preprocessor": None,
    "df": None,
    "y_series": None,
    "X_all": None,
    "train_indices": None,
    "test_indices": None,
    "feature_names": None,
}


class TabNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_rate=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),    # network.0
            nn.BatchNorm1d(hidden_size_1),           # network.1
            nn.ReLU(),                               # network.2
            nn.Dropout(dropout_rate),                # network.3
            nn.Linear(hidden_size_1, hidden_size_2), # network.4
            nn.BatchNorm1d(hidden_size_2),           # network.5
            nn.ReLU(),                               # network.6
            nn.Dropout(dropout_rate),                # network.7
            nn.Linear(hidden_size_2, 1),             # network.8
            nn.Sigmoid()                             # network.9
        )

    def forward(self, x):
        return self.network(x)


def _load_and_preprocess_data():
    """Load and preprocess the Adult dataset. Results are cached."""
    if _cache["preprocessor"] is not None:
        return

    print("Loading Adult dataset...")
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame

    # Clean missing values
    df = df.replace("?", pd.NA).dropna()

    X = df.drop("class", axis=1)
    y = (df["class"] == ">50K").astype(int)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # Fit on ALL data then transform (matches original training script)
    X_transformed = preprocessor.fit_transform(X)

    # Build feature names
    feature_names = numerical_cols.copy()
    cat_encoder = preprocessor.named_transformers_["cat"]
    for col, categories in zip(categorical_cols, cat_encoder.categories_):
        for cat in categories:
            feature_names.append(f"{col}_{cat}")

    # Split WITHOUT stratify — dataset row_no values match this split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    # Convert to dense arrays
    X_all_arr = X_transformed.toarray() if hasattr(X_transformed, 'toarray') else np.array(X_transformed)

    # Store original DataFrame indices for train/test sets
    # Build encoded→original mapping for aggregating attributions back to original features
    encoded_to_original = {}
    for name in numerical_cols:
        encoded_to_original[name] = name
    for col, categories in zip(categorical_cols, cat_encoder.categories_):
        for cat in categories:
            encoded_to_original[f"{col}_{cat}"] = col

    _cache["preprocessor"] = preprocessor
    _cache["df"] = df
    _cache["y_series"] = y
    _cache["X_all"] = torch.tensor(X_all_arr, dtype=torch.float32)
    _cache["train_indices"] = set(y_train.index.tolist())
    _cache["test_indices"] = set(y_test.index.tolist())
    _cache["feature_names"] = feature_names
    _cache["encoded_to_original"] = encoded_to_original
    # Map from original DataFrame index to positional index in X_all
    _cache["orig_to_pos"] = {orig_idx: pos for pos, orig_idx in enumerate(df.index.tolist())}


def load_model(model_path: str):
    """
    Load the Adult Census TabNN model.

    Args:
        model_path: Path to the .pth model file.

    Returns:
        tuple: (model, preprocessor)
    """
    _load_and_preprocess_data()
    preprocessor = _cache["preprocessor"]

    input_size = _cache["X_all"].shape[1]
    model = TabNN(input_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2).to(DEVICE)

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, preprocessor


def load_data(index: int, split: str = "test") -> Dict[str, Any]:
    """
    Load a specific sample from the Adult dataset by original DataFrame index.

    Args:
        index: Original DataFrame index (row_no from dataset files).
        split: Dataset split ('train' or 'test'), used for validation.

    Returns:
        Dict containing:
            - features: Feature tensor (1D)
            - features_dict: Dict mapping feature names to values
            - label: Ground truth label index
            - label_name: Human-readable label name
            - index: Sample index
            - feature_names: List of feature names
    """
    _load_and_preprocess_data()

    orig_to_pos = _cache["orig_to_pos"]
    if index not in orig_to_pos:
        raise IndexError(
            f"Index {index} not found in dataset. "
            f"Valid range: original DataFrame indices after cleaning."
        )

    pos = orig_to_pos[index]
    features = _cache["X_all"][pos]
    label = int(_cache["y_series"].iloc[pos])
    feature_names = _cache["feature_names"]

    # Build encoded feature dict
    features_dict = {}
    for i, name in enumerate(feature_names):
        features_dict[name] = float(features[i].item())

    # Raw (pre-encoding) feature values from original DataFrame
    raw_features_dict = {}
    for col in _cache["df"].columns:
        if col == "class":
            continue
        val = _cache["df"].loc[index, col]
        try:
            raw_features_dict[col] = val.item()
        except AttributeError:
            raw_features_dict[col] = str(val)

    return {
        "features": features,                               # Tensor [num_features]
        "features_dict": features_dict,                    # {encoded_name: value}
        "raw_features_dict": raw_features_dict,            # {original_col: raw_value}
        "encoded_to_original": _cache["encoded_to_original"],
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
    model_path = "/sfs/ceph/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/models_to_read/tabular/adult_tabnn.pth"

    # Load model
    model, preprocessor = load_model(model_path)
    print(f"Model loaded on {DEVICE}")
    print(get_model_info(model))

    # Test with row_no values from dataset files
    test_row_nos = [21762, 21701, 42663]
    for row_no in test_row_nos:
        data = load_data(row_no, split="test")
        print(f"\nSample row_no={row_no}:")
        print(f"  Ground truth: {data['label_name']} (class {data['label']})")

        result = predict(model, data['features'], preprocessor)
        print(f"  Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")
        print(f"  Correct: {result['predicted_class_idx'] == data['label']}")
