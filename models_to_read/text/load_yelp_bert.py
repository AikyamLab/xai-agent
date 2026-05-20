import torch
from datasets import load_dataset
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "rttl-ai/bert-base-uncased-yelp-polarity"
MAX_LENGTH = 256
NUM_CLASSES = 2

LABEL_MAP = {
    0: "negative",
    1: "positive",
}

_tokenizer = None
_dataset_cache: Dict[str, Any] = {}


def _get_dataset_split(split: str):
    split_name = "train" if split == "train" else "test"
    if split_name not in _dataset_cache:
        _dataset_cache[split_name] = load_dataset("yelp_polarity", split=split_name)
    return _dataset_cache[split_name]


def _get_tokenizer(tokenizer: Optional[AutoTokenizer] = None):
    global _tokenizer
    if tokenizer is not None:
        return tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def load_model(model_path: str):
    """
    Loads Yelp polarity BERT model and tokenizer.

    Args:
        model_path (str): Kept for interface compatibility. Not used because this
            loader uses pretrained Hugging Face weights.

    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = _get_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer


def load_data(data_input: Union[str, Dict[str, Any], int], index: Optional[int] = None) -> Dict[str, Any]:
    """
    Load and preprocess Yelp text input.

    Args:
        data_input: Raw text, dict containing text, or integer row index
        index: Optional index when data_input is text/dict

    Returns:
        Dict with tokenized tensors and metadata.
    """
    tokenizer = _get_tokenizer()

    if isinstance(data_input, int):
        row = _get_dataset_split("test")[data_input]
        text = row["text"]
        row_index = data_input
        label = int(row["label"])
    elif isinstance(data_input, dict):
        text = data_input.get("review_text", data_input.get("text", ""))
        row_index = index
        label = data_input.get("label")
    else:
        text = str(data_input)
        row_index = index
        label = None

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {
        "text": text,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "input_tensor": encoded["input_ids"],
        "token_ids": encoded["input_ids"][0].tolist(),
        "index": row_index,
        "label": int(label) if label is not None else None,
        "label_name": LABEL_MAP.get(int(label), None) if label is not None else None,
    }


def predict(
    model: torch.nn.Module,
    text_input: Union[str, Dict[str, Any], torch.Tensor],
    tokenizer: Optional[AutoTokenizer] = None
) -> Dict[str, Any]:
    """Make prediction on Yelp text using the loaded model."""
    tok = _get_tokenizer(tokenizer)
    model.eval()

    if isinstance(text_input, dict):
        if "input_ids" in text_input and "attention_mask" in text_input:
            input_ids = text_input["input_ids"]
            attention_mask = text_input["attention_mask"]
        else:
            text = text_input.get("review_text", text_input.get("text", ""))
            encoded = tok(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
    elif isinstance(text_input, torch.Tensor):
        input_ids = text_input
        attention_mask = (text_input != tok.pad_token_id).long()
    else:
        encoded = tok(
            str(text_input),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class_idx = int(pred_idx.item())
    confidence_val = float(confidence.item())
    probabilities = probs.cpu().numpy().flatten().tolist()

    return {
        "success": True,
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_name": LABEL_MAP.get(predicted_class_idx, f"class_{predicted_class_idx}"),
        "confidence": confidence_val,
        "probabilities": probabilities,
        "class_probabilities": {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1]),
        },
        "top5_predictions": [
            {"class_idx": i, "class_name": LABEL_MAP.get(i, f"class_{i}"), "probability": float(probabilities[i])}
            for i in range(len(probabilities))
        ],
        "device": str(DEVICE),
    }


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "model_name": MODEL_NAME,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(DEVICE),
        "label_map": LABEL_MAP,
    }
