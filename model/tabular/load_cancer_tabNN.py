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
WEIGHTS_PATH = "/content/tabnn_cancer.pth" # Set back to tabnn_cancer.pth as requested
HIDDEN_SIZE_1 = 64 # Size of the first hidden layer to match checkpoint
HIDDEN_SIZE_2 = 32 # Size of the second hidden layer to match checkpoint
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# 1. Download pretrained weights
# =============================
if not os.path.exists(WEIGHTS_PATH):
    print("Downloading pretrained weights from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", # Fixed incomplete f-string
        WEIGHTS_PATH,
        quiet=False
    )
else:
    print("Pretrained weights already exist.")


print("Loading Breast Cancer Wisconsin dataset...")
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer

RANDOM_SEED = 42

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
# # Use the preprocessed X_test and y_test from the breast cancer dataset
# X_test_torch = X_test.to(DEVICE)
# y_test_torch = y_test.to(DEVICE)

# # with torch.no_grad():
# #     y_pred_proba = model(X_test_torch)

# # # Convert probabilities to binary predictions
# # y_pred = (y_pred_proba >= 0.5).int()

# # # Compare predictions with true target values
# # correct_predictions = (y_pred == y_test_torch)

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test_torch.cpu().numpy(), y_pred.cpu().numpy())
# # print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# Modified TwoLayerNN architecture to suit TabNN model
class TabNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_rate=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),   # network.0
            nn.BatchNorm1d(hidden_size_1),          # network.1
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2), # network.4
            nn.BatchNorm1d(hidden_size_2),          # network.5
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_2, 1),           # network.8
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]  # should be 30

model = TabNN(input_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2).to(DEVICE)

print("Loading model weights...")
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
