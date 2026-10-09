import pickle
import re
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Constants matching checkpoint weights:
#   embedding.weight: [20002, 300]
#   fc1.weight:       [256, 1200]  (1200 = 4 * 300 from [u,v,u*v,u-v] concat)
#   fc2.weight:       [3, 256]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 20002
EMBED_DIM = 300
HIDDEN_DIM = 256
MAX_LENGTH = 128
NUM_CLASSES = 3

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

# Global vocab loaded by load_model
global_vocab: Optional[Dict[str, int]] = None


class TwoLayerNN_SNLI(nn.Module):
    """SNLI 2-layer NN with dual-input InferSent-style encoding.

    Premise and hypothesis are embedded and mean-pooled separately, then
    combined as [u, v, u*v, u-v] (4 * EMBED_DIM = 1200) before fc1.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(4 * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _masked_mean(emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool embeddings over non-padding positions only."""
        mask = (ids != 0).float().unsqueeze(-1)          # [B, L, 1]
        return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, premise_ids, hypothesis_ids):
        u = self._masked_mean(self.embedding(premise_ids), premise_ids)     # [B, D]
        v = self._masked_mean(self.embedding(hypothesis_ids), hypothesis_ids)  # [B, D]
        combined = torch.cat([u, v, (u - v).abs(), u * v], dim=1)  # [B, 4D]
        out = self.relu(self.fc1(combined))
        return self.fc2(out)


def _load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def tokenize(text: str, max_length: int = MAX_LENGTH) -> List[int]:
    global global_vocab
    if global_vocab is None:
        raise ValueError("Vocabulary not loaded. Call load_model first.")

    # Strip punctuation (vocab has no punctuation tokens) then split on whitespace
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = cleaned.split()
    ids = [global_vocab.get(t, global_vocab.get("<unk>", 1)) for t in tokens]

    if len(ids) < max_length:
        ids = ids + [0] * (max_length - len(ids))
    else:
        ids = ids[:max_length]

    return ids


def load_model(model_path: str, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM,
               num_classes: int = NUM_CLASSES):
    global global_vocab

    vocab_path = Path("models_to_read/text/vocab.pkl")
    print(f"Loading vocabulary from {vocab_path}...")
    global_vocab = _load_vocab(str(vocab_path))

    model = TwoLayerNN_SNLI(VOCAB_SIZE, embed_dim, hidden_dim, num_classes)

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, tokenize


def load_data(data_input: Union[str, Dict[str, Any]], index: Optional[int] = None) -> Dict[str, Any]:
    if isinstance(data_input, dict):
        premise = data_input.get('premise', '')
        hypothesis = data_input.get('hypothesis', '')
    else:
        premise = str(data_input)
        hypothesis = ''

    premise_ids = tokenize(premise)
    hypothesis_ids = tokenize(hypothesis)
    premise_tensor = torch.tensor([premise_ids], dtype=torch.long)
    hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long)

    return {
        "text": f"{premise} {hypothesis}",
        "premise": premise,
        "hypothesis": hypothesis,
        "premise_ids": premise_ids,
        "hypothesis_ids": hypothesis_ids,
        "premise_tensor": premise_tensor,
        "hypothesis_tensor": hypothesis_tensor,
        # Combined tensor not meaningful for dual-input; include for API compatibility
        "input_tensor": premise_tensor,
        "token_ids": premise_ids,
        "index": index
    }


def predict(
    model: nn.Module,
    text_input: Union[Dict[str, Any], tuple, torch.Tensor],
    tokenizer: Optional[callable] = None
) -> Dict[str, Any]:
    model.eval()

    if isinstance(text_input, dict):
        if 'premise_tensor' in text_input and 'hypothesis_tensor' in text_input:
            p = text_input['premise_tensor'].to(DEVICE)
            h = text_input['hypothesis_tensor'].to(DEVICE)
        elif 'premise' in text_input and 'hypothesis' in text_input:
            _tok = tokenizer or tokenize
            p = torch.tensor([_tok(text_input['premise'])], dtype=torch.long).to(DEVICE)
            h = torch.tensor([_tok(text_input['hypothesis'])], dtype=torch.long).to(DEVICE)
        else:
            raise ValueError("text_input dict missing 'premise'/'hypothesis' keys")
    elif isinstance(text_input, tuple) and len(text_input) == 2:
        p, h = text_input
        p, h = p.to(DEVICE), h.to(DEVICE)
    else:
        raise TypeError(f"Unsupported text_input type: {type(text_input)}")

    with torch.no_grad():
        logits = model(p, h)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

    top5_predictions = sorted([
        {"class_idx": i, "class_name": LABEL_MAP[i], "probability": float(probabilities[i].item())}
        for i in range(NUM_CLASSES)
    ], key=lambda x: x["probability"], reverse=True)

    return {
        "success": True,
        "predicted_class_idx": predicted_class,
        "predicted_class_name": LABEL_MAP[predicted_class],
        "confidence": float(confidence),
        "probabilities": probabilities.cpu().numpy(),
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


if __name__ == "__main__":
    model_path = "models_to_read/text/snli_2layernn.pth"

    model, tok = load_model(model_path)
    print(f"Model loaded on {DEVICE}")
    print(get_model_info(model))

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
