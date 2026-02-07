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

# with torch.no_grad():
#     y_pred_proba = model(X_test_torch)

# # Convert probabilities to binary predictions
# y_pred = (y_pred_proba >= 0.5).int()

# # Compare predictions with true target values
# correct_predictions = (y_pred == y_test_torch)

# # Calculate accuracy
# accuracy = accuracy_score(y_test_torch.cpu().numpy(), y_pred.cpu().numpy())
# print(f"Model Accuracy on Test Set: {accuracy:.4f}")

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
