import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import re
from datasets import load_dataset
from collections import Counter

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_vocab_lock = threading.Lock()
_vocab_built = False
VOCAB_SIZE = 20002
EMBED_DIM = 300
NUM_FILTERS = 100
KERNEL_SIZES = (3, 4, 5)
MAX_LENGTH = 128
NUM_CLASSES = 3

# Global vocabulary variable, will be populated by load_model
global_vocab = None

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


def tokenize(text):
    """Tokenize text for vocabulary building."""
    return re.findall(r"\b\w+\b", text.lower())


def simple_tokenize(text: str, max_length: int = MAX_LENGTH) -> List[int]:
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

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()

    token_ids = []
    for word in words[:max_length]:
        token_id = global_vocab.get(word, global_vocab["<unk>"])
        token_ids.append(token_id)

    if len(token_ids) < max_length:
        token_ids = token_ids + [global_vocab["<pad>"]] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    return token_ids


def load_model(model_path: str, embed_dim: int = EMBED_DIM,
               num_filters: int = NUM_FILTERS, kernel_sizes: tuple = KERNEL_SIZES,
               num_classes: int = NUM_CLASSES):
    """
    Loads the SNLI CNN model.

    Args:
        model_path (str): The path to the .pth model file.
        embed_dim (int): Embedding dimension.
        num_filters (int): Number of filters per kernel size.
        kernel_sizes (tuple): Tuple of kernel sizes.
        num_classes (int): Number of output classes.

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
            print("Building vocabulary from SNLI training data...")
            dataset_snli = load_dataset("snli")
            # Filter using manual loop to avoid TensorFlow tensor validation issue
            # (datasets library tries to use tf.Tensor which may not be available)
            
            counter = Counter()
            for ex in dataset_snli["train"]:
                # Skip invalid labels manually instead of using .filter()
                if ex.get("label") != -1:
                    counter.update(tokenize(ex["premise"]))
                    counter.update(tokenize(ex["hypothesis"]))

            global_vocab = {"<pad>": 0, "<unk>": 1}
            for word, _ in counter.most_common(actual_vocab_size - 2):
                global_vocab[word] = len(global_vocab)

            _vocab_built = True
            print(f"Vocabulary built with size: {len(global_vocab)}")

    model = CNN_SNLI(actual_vocab_size, embed_dim, num_classes, num_filters, kernel_sizes)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, simple_tokenize


def load_data(data_input: Union[str, Dict[str, Any]], index: Optional[int] = None) -> Dict[str, Any]:
    """
    Load and preprocess text data for SNLI NLI classification.

    Args:
        data_input: Dict containing 'premise' and 'hypothesis', or a string
        index: Optional index (for compatibility)

    Returns:
        Dict containing:
            - text: Combined premise + hypothesis for display/tools
            - premise: Original premise text
            - hypothesis: Original hypothesis text
            - token_ids: Tokenized premise (for compatibility with text tools)
            - premise_ids: Tokenized premise
            - hypothesis_ids: Tokenized hypothesis
            - input_tensor: Premise tensor (for compatibility)
            - premise_tensor: Tensor ready for model
            - hypothesis_tensor: Tensor ready for model
    """
    if isinstance(data_input, dict):
        premise = data_input.get('premise', '')
        hypothesis = data_input.get('hypothesis', '')
    else:
        # Fallback: treat as premise with empty hypothesis
        premise = str(data_input)
        hypothesis = ''

    premise_ids = simple_tokenize(premise)
    hypothesis_ids = simple_tokenize(hypothesis)

    premise_tensor = torch.tensor([premise_ids], dtype=torch.long)
    hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long)

    # Combined text for display and text-based XAI tools
    combined_text = f"Premise: {premise} Hypothesis: {hypothesis}"

    return {
        "text": combined_text,
        "premise": premise,
        "hypothesis": hypothesis,
        "token_ids": premise_ids,
        "premise_ids": premise_ids,
        "hypothesis_ids": hypothesis_ids,
        "input_tensor": premise_tensor,
        "premise_tensor": premise_tensor,
        "hypothesis_tensor": hypothesis_tensor,
        "index": index
    }


def predict(
    model: nn.Module,
    text_input: Union[Dict[str, Any], tuple],
    tokenizer: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Make prediction on SNLI input using the loaded model.

    Args:
        model: Loaded PyTorch model
        text_input: Either a dict with 'premise'/'hypothesis' or preprocessed tensors
        tokenizer: Tokenizer function (optional, uses simple_tokenize if None)

    Returns:
        Dict containing prediction results
    """
    model.eval()

    if isinstance(text_input, dict):
        if 'premise_tensor' in text_input and 'hypothesis_tensor' in text_input:
            premise_tensor = text_input['premise_tensor'].to(DEVICE)
            hypothesis_tensor = text_input['hypothesis_tensor'].to(DEVICE)
        elif 'premise_ids' in text_input and 'hypothesis_ids' in text_input:
            premise_tensor = torch.tensor([text_input['premise_ids']], dtype=torch.long).to(DEVICE)
            hypothesis_tensor = torch.tensor([text_input['hypothesis_ids']], dtype=torch.long).to(DEVICE)
        else:
            if tokenizer is None:
                tokenizer = simple_tokenize
            premise_ids = tokenizer(text_input.get('premise', ''))
            hypothesis_ids = tokenizer(text_input.get('hypothesis', ''))
            premise_tensor = torch.tensor([premise_ids], dtype=torch.long).to(DEVICE)
            hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long).to(DEVICE)
    elif isinstance(text_input, tuple) and len(text_input) == 2:
        premise_tensor, hypothesis_tensor = text_input
        premise_tensor = premise_tensor.to(DEVICE)
        hypothesis_tensor = hypothesis_tensor.to(DEVICE)
    else:
        raise TypeError(f"Unsupported text_input type: {type(text_input)}")

    with torch.no_grad():
        logits = model(premise_tensor, hypothesis_tensor)
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
        "probabilities": probabilities.cpu().numpy(),  # array indexed by class_idx (0=entailment, 1=neutral, 2=contradiction)
        "class_probabilities": {
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
    global global_vocab
    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "vocab_size": len(global_vocab) if global_vocab else "Not Initialized",
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
