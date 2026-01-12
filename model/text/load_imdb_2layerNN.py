import torch
import torch.nn as nn
from datasets import load_dataset
from collections import Counter
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

dataset = load_dataset("imdb")

# Rebuild Vocabulary 
counter = Counter()
for ex in dataset["train"]:
    counter.update(tokenize(ex["text"]))

vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(20000):
    vocab[word] = len(vocab)

vocab_size = len(vocab)
print("Vocab size:", vocab_size)

class TwoLayerNN_IMDB(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)        # [B, L, D]
        sent_vec = emb.mean(dim=1)     # [B, D]
        h = self.relu(self.fc1(sent_vec))
        logits = self.fc2(h)           # [B, 1]
        return logits

model = TwoLayerNN_IMDB(
    vocab_size=vocab_size,
    embed_dim=300,
    hidden_dim=256
).to(device)

# Load Weights
state_dict = torch.load('weights.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")
