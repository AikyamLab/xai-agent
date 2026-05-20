import os
import torch
import torch.nn as nn
import gdown
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

GDRIVE_FILE_ID = "PUT_YOUR_FILE_ID_HERE"
WEIGHTS_PATH = "./two_layer_adult.pth"
HIDDEN_SIZE = 64
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if not os.path.exists(WEIGHTS_PATH):
    print("Downloading pretrained weights from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        WEIGHTS_PATH,
        quiet=False
    )
else:
    print("Pretrained weights already exist.")

print("Loading Adult dataset...")
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame

# Clean missing values
df = df.replace("?", pd.NA).dropna()

X = df.drop("class", axis=1)
y = (df["class"] == ">50K").astype(int)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test  = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

input_size = X_train.shape[1]

model = TwoLayerNN(input_size, HIDDEN_SIZE).to(DEVICE)

print("Loading model weights...")
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    probs = model(X_test)
    preds = (probs >= 0.5).float()

    acc = accuracy_score(
        y_test.cpu().numpy(),
        preds.cpu().numpy()
    )

print(f"Test Accuracy (pretrained model): {acc:.4f}")
