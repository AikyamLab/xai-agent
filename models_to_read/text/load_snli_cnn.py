import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import re

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 30000
EMBED_DIM = 300
NUM_FILTERS = 100
KERNEL_SIZES = (3, 4, 5)
MAX_LENGTH = 128
NUM_CLASSES = 3

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}


class CNN_SNLI(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        num_classes=3,
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
            num_filters * len(kernel_sizes) * 2,
            num_classes
        )

    def encode_sentence(self, x):
        emb = self.embedding(x)       # [B, L, D]
        emb = emb.transpose(1, 2)     # [B, D, L]

        conv_outs = []
        for conv in self.convs:
            c = self.relu(conv(emb))
            c = torch.max(c, dim=2)[0]
            conv_outs.append(c)

        return torch.cat(conv_outs, dim=1)

    def forward(self, premise, hypothesis):
        prem_vec = self.encode_sentence(premise)
        hyp_vec = self.encode_sentence(hypothesis)
        x = torch.cat([prem_vec, hyp_vec], dim=1)
        return self.fc(x)


def simple_tokenize(text: str, max_length: int = MAX_LENGTH, vocab_size: int = VOCAB_SIZE) -> List[int]:
    """
    Simple tokenization for text: convert words to hash-based token IDs.

    Args:
        text: Input text string
        max_length: Maximum sequence length
        vocab_size: Size of vocabulary (for hash modulo)

    Returns:
        List of token IDs
    """
    # Clean and lowercase
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()

    # Convert to token IDs using hash
    token_ids = []
    for word in words[:max_length]:
        token_id = (hash(word) % (vocab_size - 2)) + 2
        token_ids.append(token_id)

    # Pad or truncate to max_length
    if len(token_ids) < max_length:
        token_ids = token_ids + [0] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    return token_ids


def load_model(model_path: str, vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM,
               num_filters: int = NUM_FILTERS, kernel_sizes: tuple = KERNEL_SIZES,
               num_classes: int = NUM_CLASSES):
    """
    Loads the SNLI CNN model.

    Args:
        model_path (str): The path to the .pth model file.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        num_filters (int): Number of filters per kernel size.
        kernel_sizes (tuple): Tuple of kernel sizes.
        num_classes (int): Number of output classes.

    Returns:
        tuple: A tuple containing the loaded model and a processor (tokenizer function).
    """
    model = CNN_SNLI(vocab_size, embed_dim, num_classes, num_filters, kernel_sizes)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, simple_tokenize


def load_data(data_input: Dict[str, Any], index: Optional[int] = None) -> Dict[str, Any]:
    """
    Load and preprocess text data for SNLI NLI classification.

    Args:
        data_input: Dict containing 'premise' and 'hypothesis'
        index: Optional index (for compatibility)

    Returns:
        Dict containing:
            - premise: Original premise text
            - hypothesis: Original hypothesis text
            - premise_ids: Tokenized premise
            - hypothesis_ids: Tokenized hypothesis
            - premise_tensor: Tensor ready for model
            - hypothesis_tensor: Tensor ready for model
    """
    premise = data_input.get('premise', '')
    hypothesis = data_input.get('hypothesis', '')

    # Tokenize
    premise_ids = simple_tokenize(premise)
    hypothesis_ids = simple_tokenize(hypothesis)

    premise_tensor = torch.tensor([premise_ids], dtype=torch.long)
    hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long)

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "premise_ids": premise_ids,
        "hypothesis_ids": hypothesis_ids,
        "premise_tensor": premise_tensor,
        "hypothesis_tensor": hypothesis_tensor,
        "index": index
    }


def predict(
    model: nn.Module,
    data_input: Union[Dict[str, Any], tuple],
    tokenizer: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Make prediction on SNLI input using the loaded model.

    Args:
        model: Loaded PyTorch model
        data_input: Either a dict with 'premise'/'hypothesis' or preprocessed tensors
        tokenizer: Tokenizer function (optional, uses simple_tokenize if None)

    Returns:
        Dict containing prediction results
    """
    model.eval()

    # Prepare input tensors
    if isinstance(data_input, dict):
        if 'premise_tensor' in data_input and 'hypothesis_tensor' in data_input:
            premise_tensor = data_input['premise_tensor'].to(DEVICE)
            hypothesis_tensor = data_input['hypothesis_tensor'].to(DEVICE)
        elif 'premise_ids' in data_input and 'hypothesis_ids' in data_input:
            premise_tensor = torch.tensor([data_input['premise_ids']], dtype=torch.long).to(DEVICE)
            hypothesis_tensor = torch.tensor([data_input['hypothesis_ids']], dtype=torch.long).to(DEVICE)
        else:
            # Raw text
            if tokenizer is None:
                tokenizer = simple_tokenize
            premise_ids = tokenizer(data_input.get('premise', ''))
            hypothesis_ids = tokenizer(data_input.get('hypothesis', ''))
            premise_tensor = torch.tensor([premise_ids], dtype=torch.long).to(DEVICE)
            hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long).to(DEVICE)
    elif isinstance(data_input, tuple) and len(data_input) == 2:
        premise_tensor, hypothesis_tensor = data_input
        premise_tensor = premise_tensor.to(DEVICE)
        hypothesis_tensor = hypothesis_tensor.to(DEVICE)
    else:
        raise TypeError(f"Unsupported data_input type: {type(data_input)}")

    # Make prediction
    with torch.no_grad():
        logits = model(premise_tensor, hypothesis_tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

    return {
        "success": True,
        "predicted_class_idx": predicted_class,
        "predicted_class_name": LABEL_MAP[predicted_class],
        "confidence": confidence,
        "probabilities": {
            "entailment": float(probabilities[0].item()),
            "neutral": float(probabilities[1].item()),
            "contradiction": float(probabilities[2].item())
        },
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
        "num_filters": NUM_FILTERS,
        "kernel_sizes": KERNEL_SIZES,
        "max_length": MAX_LENGTH,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(DEVICE),
        "label_map": LABEL_MAP
    }


# Main function for testing
if __name__ == "__main__":
    model_path = "/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/models_to_read/text/snli_cnn.pth"

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
