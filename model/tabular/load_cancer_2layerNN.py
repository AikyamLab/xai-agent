import os
import torch
import torch.nn as nn
import gdown
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

GDRIVE_FILE_ID = "PUT_YOUR_FILE_ID_HERE"
WEIGHTS_PATH = "/content/2_layer_cancer (1).pth"
HIDDEN_SIZE = 32 # Changed from 64 to 32 to match pretrained weights
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# 1. Download pretrained weights
# =============================
if not os.path.exists(WEIGHTS_PATH):
    print("Downloading pretrained weights from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        WEIGHTS_PATH,
        quiet=False
    )
else:
    print("Pretrained weights already exist.")


print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()

X = data.data          # shape: (n_samples, 30)
y = data.target        # 0 = malignant, 1 = benign

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Removed BatchNorm1d layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Adjusted forward pass to remove BatchNorm1d
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

input_size = X_train.shape[1]  # should be 30

model = TwoLayerNN(input_size, HIDDEN_SIZE).to(DEVICE)

print("Loading model weights...")
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
