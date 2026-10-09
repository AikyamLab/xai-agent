import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import re
from datasets import load_dataset # Added
from collections import Counter # Added

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# VOCAB_SIZE will be determined dynamically

_vocab_lock = threading.Lock()
_vocab_built = False
EMBED_DIM = 300
NUM_FILTERS = 100
KERNEL_SIZES = (3, 4, 5)
MAX_LENGTH = 256
NUM_CLASSES = 2

# Global vocabulary variable, will be populated by load_model
global_vocab = None

LABEL_MAP = {
    0: "negative",
    1: "positive"
}


class CNN_IMDB(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        num_filters=100,
        kernel_sizes=(3, 4, 5)
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.relu = nn.ReLU()

        self.fc = nn.Linear(
            num_filters * len(kernel_sizes),
            1
        )

    def forward(self, x):
        emb = self.embedding(x)       # [B, L, D]
        emb = emb.transpose(1, 2)     # [B, D, L]

        conv_outs = []
        for conv in self.convs:
            c = self.relu(conv(emb))
            c = torch.max(c, dim=2)[0]
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=1)
        return self.fc(x)


def simple_tokenize(text: str, max_length: int = MAX_LENGTH) -> List[int]: # Removed vocab_size parameter
    """
    Simple tokenization for text: convert words to token IDs using the global vocabulary.

    Args:
        text: Input text string
        max_length: Maximum sequence length

    Returns:
        List of token IDs
    """
    global global_vocab
    if global_vocab is None:
        raise ValueError("Vocabulary not initialized. Call load_model first.")

    # Clean and lowercase
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML breaks
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Keep only alphanumeric
    words = text.split()

    # Convert to token IDs using the global vocabulary
    token_ids = []
    for word in words[:max_length]:
        token_id = global_vocab.get(word, global_vocab["<unk>"]) # Use <unk> for unknown words
        token_ids.append(token_id)

    # Pad or truncate to max_length
    if len(token_ids) < max_length:
        token_ids = token_ids + [global_vocab["<pad>"]] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    return token_ids


def load_model(model_path: str, embed_dim: int = EMBED_DIM,
               num_filters: int = NUM_FILTERS, kernel_sizes: tuple = KERNEL_SIZES):
    """
    Loads the IMDB CNN model.

    Args:
        model_path (str): The path to the .pth model file.
        embed_dim (int): Embedding dimension.
        num_filters (int): Number of filters per kernel size.
        kernel_sizes (tuple): Tuple of kernel sizes.

    Returns:
        tuple: A tuple containing the loaded model and a processor (tokenizer function).
    """
    global global_vocab, _vocab_built

    # Read vocab_size from checkpoint to avoid thread-race size mismatch
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    actual_vocab_size = state_dict["embedding.weight"].shape[0]

    # Build vocabulary once (thread-safe); other threads wait and reuse it
    with _vocab_lock:
        if not _vocab_built:
            print("Building vocabulary...")
            dataset_imdb = load_dataset("imdb")

            def _tokenize(text):
                return re.findall(r"\b\w+\b", text.lower())

            counter = Counter()
            for ex in dataset_imdb["train"]:
                counter.update(_tokenize(ex["text"]))

            global_vocab = {"<pad>": 0, "<unk>": 1}
            for word, _ in counter.most_common(actual_vocab_size - 2):
                global_vocab[word] = len(global_vocab)

            _vocab_built = True
            print(f"Vocabulary built with size: {len(global_vocab)}")

    model = CNN_IMDB(actual_vocab_size, embed_dim, num_filters, kernel_sizes)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, simple_tokenize


def load_data(text_input: Union[str, Dict[str, Any]], index: Optional[int] = None) -> Dict[str, Any]:
    """
    Load and preprocess text data for IMDB sentiment classification.

    Args:
        text_input: Either a raw text string or a dict containing 'review_text'
        index: Optional index (for compatibility, not used for text)

    Returns:
        Dict containing:
            - text: Original text
            - token_ids: Tokenized input
            - input_tensor: Tensor ready for model
    """
    # Extract text from input
    if isinstance(text_input, dict):
        text = text_input.get('review_text', text_input.get('text', ''))
    else:
        text = str(text_input)

    # Tokenize
    token_ids = simple_tokenize(text)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    return {
        "text": text,
        "token_ids": token_ids,
        "input_tensor": input_tensor,
        "index": index
    }


def predict(
    model: nn.Module,
    text_input: Union[str, Dict[str, Any], torch.Tensor],
    tokenizer: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Make prediction on text input using the loaded model.

    Args:
        model: Loaded PyTorch model
        text_input: Either raw text, dict with 'review_text', or pre-tokenized tensor
        tokenizer: Tokenizer function (optional, uses simple_tokenize if None)

    Returns:
        Dict containing prediction results
    """
    model.eval()

    # Prepare input tensor
    if isinstance(text_input, torch.Tensor):
        input_tensor = text_input.to(DEVICE)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
    elif isinstance(text_input, dict):
        if 'input_tensor' in text_input:
            input_tensor = text_input['input_tensor'].to(DEVICE)
        elif 'token_ids' in text_input:
            input_tensor = torch.tensor([text_input['token_ids']], dtype=torch.long).to(DEVICE)
        else:
            text = text_input.get('review_text', text_input.get('text', ''))
            if tokenizer is None:
                tokenizer = simple_tokenize
            token_ids = tokenizer(text)
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)
    else:
        # Raw text string
        if tokenizer is None:
            tokenizer = simple_tokenize
        token_ids = tokenizer(str(text_input))
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

    # Make prediction
    with torch.no_grad():
        logits = model(input_tensor)

        # Binary classification: apply sigmoid
        prob_positive = torch.sigmoid(logits).item()
        prob_negative = 1.0 - prob_positive

        predicted_class = 1 if prob_positive >= 0.5 else 0
        confidence = prob_positive if predicted_class == 1 else prob_negative

    top5_predictions = sorted([
        {"class_idx": 0, "class_name": LABEL_MAP[0], "probability": float(prob_negative)},
        {"class_idx": 1, "class_name": LABEL_MAP[1], "probability": float(prob_positive)},
    ], key=lambda x: x["probability"], reverse=True)

    return {
        "success": True,
        "predicted_class_idx": predicted_class,
        "predicted_class_name": LABEL_MAP[predicted_class],
        "confidence": float(confidence),
        "probabilities": {
            "negative": float(prob_negative),
            "positive": float(prob_positive)
        },
        "top5_predictions": top5_predictions,
        "raw_logit": float(logits.item()),
        "device": str(DEVICE)
    }


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Args:
        model: Loaded PyTorch model

    Returns:
        Dict containing model information
    """
    global global_vocab # Access the global vocabulary
    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "vocab_size": len(global_vocab) if global_vocab else "Not Initialized", # Use actual vocab size
        "embed_dim": EMBED_DIM,
        "num_filters": NUM_FILTERS,
        "kernel_sizes": KERNEL_SIZES,
        "max_length": MAX_LENGTH,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(DEVICE),
        "label_map": LABEL_MAP
    }


# Main function for testing
if __name__ == "__main__":
    model_path = "models_to_read/text/imdb_cnn.pth"

    # Load model
    model, tokenizer = load_model(model_path)
    print(f"Model loaded on {DEVICE}")
    print(get_model_info(model))

    # Test with sample text
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Complete waste of time and money.",
        "An average movie with some good moments but also many flaws."
    ]

    for text in test_texts:
        data = load_data(text)
        result = predict(model, data)
        print(f"\nText: {text[:50]}...")
        print(f"  Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")
