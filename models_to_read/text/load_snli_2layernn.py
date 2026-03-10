import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from transformers import BertTokenizer

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 30522       # BertTokenizer vocab size (matches checkpoint)
EMBED_DIM = 128          # Matches checkpoint embedding.weight [30522, 128]
HIDDEN_DIM = 256         # Matches checkpoint fc1.weight [256, 128]
MAX_LENGTH = 128
NUM_CLASSES = 3

# Global tokenizer, populated by load_model
global_tokenizer = None

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}


class TwoLayerNN_SNLI(nn.Module):
    """SNLI 2-layer NN — single-input architecture matching snli_2layernn.pth.

    Checkpoint weights:
        embedding.weight: [30522, 128]
        fc1.weight:       [256, 128]   → Linear(128, 256)
        fc1.bias:         [256]
        fc2.weight:       [3, 256]     → Linear(256, 3)
        fc2.bias:         [3]
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Single-input forward: x is token IDs [B, L]."""
        emb = self.embedding(x)       # [B, L, D]
        vec = emb.mean(dim=1)         # [B, D]
        out = self.relu(self.fc1(vec))
        return self.fc2(out)


def simple_tokenize(text: str, max_length: int = MAX_LENGTH) -> List[int]:
    """
    Tokenize text using BertTokenizer (vocab_size=30522, matching checkpoint).

    Args:
        text: Input text string
        max_length: Maximum sequence length

    Returns:
        List of token IDs
    """
    global global_tokenizer
    if global_tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call load_model first.")

    tokens = global_tokenizer.encode(
        text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True
    )

    # Pad to max_length
    if len(tokens) < max_length:
        tokens = tokens + [0] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return tokens


def load_model(model_path: str, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM,
               num_classes: int = NUM_CLASSES):
    """
    Loads the SNLI 2-layer NN model.

    Args:
        model_path (str): The path to the .pth model file.
        embed_dim (int): Embedding dimension.
        hidden_dim (int): Hidden layer dimension.
        num_classes (int): Number of output classes.

    Returns:
        tuple: A tuple containing the loaded model and a processor (tokenizer function).
    """
    global global_tokenizer

    print("Loading BertTokenizer (vocab_size=30522)...")
    global_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = TwoLayerNN_SNLI(VOCAB_SIZE, embed_dim, hidden_dim, num_classes)

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, simple_tokenize


def load_data(data_input: Union[str, Dict[str, Any]], index: Optional[int] = None) -> Dict[str, Any]:
    """
    Load and preprocess text data for SNLI NLI classification.

    The model uses single-input architecture: premise and hypothesis are
    concatenated into one sequence, tokenized together.

    Args:
        data_input: Dict containing 'premise' and 'hypothesis', or a string
        index: Optional index (for compatibility)

    Returns:
        Dict containing:
            - text: Combined premise + hypothesis for display/tools
            - premise: Original premise text
            - hypothesis: Original hypothesis text
            - token_ids: Tokenized combined text
            - premise_ids: Tokenized premise (for XAI tools)
            - hypothesis_ids: Tokenized hypothesis (for XAI tools)
            - input_tensor: Combined tensor for model input
            - premise_tensor: Premise tensor (for XAI tools)
            - hypothesis_tensor: Hypothesis tensor (for XAI tools)
    """
    if isinstance(data_input, dict):
        premise = data_input.get('premise', '')
        hypothesis = data_input.get('hypothesis', '')
    else:
        premise = str(data_input)
        hypothesis = ''

    # Combined text for the single-input model
    combined_text = f"Premise: {premise} Hypothesis: {hypothesis}"
    combined_ids = simple_tokenize(combined_text)
    combined_tensor = torch.tensor([combined_ids], dtype=torch.long)

    # Separate tokenizations for XAI tools (LIME, SHAP, etc.)
    premise_ids = simple_tokenize(premise)
    hypothesis_ids = simple_tokenize(hypothesis)
    premise_tensor = torch.tensor([premise_ids], dtype=torch.long)
    hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long)

    return {
        "text": combined_text,
        "premise": premise,
        "hypothesis": hypothesis,
        "token_ids": combined_ids,
        "premise_ids": premise_ids,
        "hypothesis_ids": hypothesis_ids,
        "input_tensor": combined_tensor,
        "premise_tensor": premise_tensor,
        "hypothesis_tensor": hypothesis_tensor,
        "index": index
    }


def predict(
    model: nn.Module,
    text_input: Union[Dict[str, Any], tuple, torch.Tensor],
    tokenizer: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Make prediction on SNLI input using the loaded model.

    The model is single-input: premise + hypothesis concatenated and mean-pooled.

    Args:
        model: Loaded PyTorch model
        text_input: Dict with tensors/ids, a tuple (premise, hypothesis), or a raw tensor
        tokenizer: Tokenizer function (optional, uses simple_tokenize if None)

    Returns:
        Dict containing prediction results
    """
    model.eval()

    if isinstance(text_input, torch.Tensor):
        # Raw tensor input (e.g., from XAI tools)
        input_tensor = text_input.to(DEVICE)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
    elif isinstance(text_input, dict):
        if 'input_tensor' in text_input:
            input_tensor = text_input['input_tensor'].to(DEVICE)
        elif 'premise_tensor' in text_input and 'hypothesis_tensor' in text_input:
            # Legacy dual-input: concatenate premise + hypothesis tokens
            p = text_input['premise_tensor']
            h = text_input['hypothesis_tensor']
            # Combine by concatenating along sequence dimension
            input_tensor = torch.cat([p, h], dim=1).to(DEVICE)
        elif 'premise' in text_input and 'hypothesis' in text_input:
            if tokenizer is None:
                tokenizer = simple_tokenize
            combined = f"Premise: {text_input['premise']} Hypothesis: {text_input['hypothesis']}"
            ids = tokenizer(combined)
            input_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        else:
            raise ValueError("text_input dict missing required keys")
    elif isinstance(text_input, tuple) and len(text_input) == 2:
        # Legacy tuple (premise_tensor, hypothesis_tensor): concatenate
        p, h = text_input
        input_tensor = torch.cat([p.to(DEVICE), h.to(DEVICE)], dim=1)
    else:
        raise TypeError(f"Unsupported text_input type: {type(text_input)}")

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

    top5_predictions = sorted([
        {"class_idx": 0, "class_name": LABEL_MAP[0], "probability": float(probabilities[0].item())},
        {"class_idx": 1, "class_name": LABEL_MAP[1], "probability": float(probabilities[1].item())},
        {"class_idx": 2, "class_name": LABEL_MAP[2], "probability": float(probabilities[2].item())},
    ], key=lambda x: x["probability"], reverse=True)

    return {
        "success": True,
        "predicted_class_idx": predicted_class,
        "predicted_class_name": LABEL_MAP[predicted_class],
        "confidence": float(confidence),
        "probabilities": {
            "entailment": float(probabilities[0].item()),
            "neutral": float(probabilities[1].item()),
            "contradiction": float(probabilities[2].item())
        },
        "top5_predictions": top5_predictions,
        "raw_logits": logits[0].cpu().numpy().tolist(),
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
    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_length": MAX_LENGTH,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(DEVICE),
        "label_map": LABEL_MAP
    }


# Main function for testing
if __name__ == "__main__":
    model_path = "/sfs/ceph/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/models_to_read/text/snli_2layernn.pth"

    # Load model
    model, tokenizer = load_model(model_path)
    print(f"Model loaded on {DEVICE}")
    print(get_model_info(model))

    # Test with sample data
    test_pairs = [
        {"premise": "A man is playing guitar.", "hypothesis": "A person is making music."},
        {"premise": "A dog is running in the park.", "hypothesis": "A cat is sleeping."},
        {"premise": "The children are playing soccer.", "hypothesis": "The kids are not playing any sport."}
    ]

    for pair in test_pairs:
        data = load_data(pair)
        result = predict(model, data)
        print(f"\nPremise: {pair['premise']}")
        print(f"Hypothesis: {pair['hypothesis']}")
        print(f"  Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")
