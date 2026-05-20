import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import re
from datasets import load_dataset
from collections import Counter

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 20002 # This will be the max vocab size, actual size might be smaller
EMBED_DIM = 300
HIDDEN_DIM = 256
MAX_LENGTH = 128
NUM_CLASSES = 3

# Global vocabulary variable, will be populated by load_model
global_vocab = None

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}


class TwoLayerNN_SNLI(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, premise, hypothesis):
        prem_emb = self.embedding(premise)   # [B, L, D]
        hyp_emb = self.embedding(hypothesis)

        prem_vec = prem_emb.mean(dim=1)      # [B, D]
        hyp_vec = hyp_emb.mean(dim=1)

        x = torch.cat([prem_vec, hyp_vec], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def tokenize(text): # This is the original tokenize from the reference, used for vocab building
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


def load_model(model_path: str, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES):
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
    global global_vocab

    print("Building vocabulary...")
    dataset_snli = load_dataset("snli")
    dataset_snli = dataset_snli.filter(lambda x: x["label"] != -1)

    counter = Counter()
    for ex in dataset_snli["train"]:
        counter.update(tokenize(ex["premise"]))
        counter.update(tokenize(ex["hypothesis"]))

    global_vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        global_vocab[word] = len(global_vocab)
    
    actual_vocab_size = len(global_vocab)
    print(f"Vocabulary built with size: {actual_vocab_size}")

    model = TwoLayerNN_SNLI(actual_vocab_size, embed_dim, hidden_dim, num_classes)

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
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
    global global_vocab
    return {
        "architecture": model.__class__.__name__,
        "num_classes": NUM_CLASSES,
        "vocab_size": len(global_vocab) if global_vocab else "Not Initialized",
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_length": MAX_LENGTH,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(DEVICE),
        "label_map": LABEL_MAP
    }

