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

counter = Counter()
for ex in dataset["train"]:
    counter.update(tokenize(ex["text"]))

vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(20000):
    vocab[word] = len(vocab)

vocab_size = len(vocab)
print("Vocab size:", vocab_size)

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

model = CNN_IMDB(
    vocab_size=vocab_size,
    embed_dim=300
).to(device)

state_dict = torch.load("weights.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("IMDB CNN model loaded successfully.")
