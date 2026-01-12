import torch
import torch.nn as nn
from datasets import load_dataset
from collections import Counter
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


dataset = load_dataset("snli")
dataset = dataset.filter(lambda x: x["label"] != -1)


counter = Counter()
for ex in dataset["train"]:
    counter.update(tokenize(ex["premise"]))
    counter.update(tokenize(ex["hypothesis"]))

vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(20000):
    vocab[word] = len(vocab)

vocab_size = len(vocab)
print("Vocab size:", vocab_size)


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

model = TwoLayerNN_SNLI(
    vocab_size=vocab_size,
    embed_dim=300,
    hidden_dim=256
).to(device)

state_dict = torch.load("weights.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("SNLI 2-layer model loaded successfully.")
