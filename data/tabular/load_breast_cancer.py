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
# Use the preprocessed X_test and y_test from the breast cancer dataset
X_test_torch = X_test.to(DEVICE)
y_test_torch = y_test.to(DEVICE)

with torch.no_grad():
    y_pred_proba = model(X_test_torch)

# Convert probabilities to binary predictions
y_pred = (y_pred_proba >= 0.5).int()

# Compare predictions with true target values
correct_predictions = (y_pred == y_test_torch)

# Calculate accuracy
accuracy = accuracy_score(y_test_torch.cpu().numpy(), y_pred.cpu().numpy())
print(f"Model Accuracy on Test Set: {accuracy:.4f}")
