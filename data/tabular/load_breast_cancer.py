import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Assuming RANDOM_SEED is defined (e.g., from earlier cells)
# If not, define it here:
# RANDOM_SEED = 42

# 1. Load the full breast cancer dataset as a DataFrame to get original features and target
data_full = load_breast_cancer(as_frame=True)
X_full = data_full.data
y_full = data_full.target

# 2. Re-create the train-test split for the *original* data
# This is crucial to get the same X_test_original used in the JSON generation
_, X_test_original, _, y_test_original = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_full
)

# Reset index to ensure direct indexing using row_idx from JSON
X_test_original = X_test_original.reset_index(drop=True)
y_test_original = y_test_original.reset_index(drop=True)

# 3. Initialize and fit the StandardScaler on the FULL dataset (or X_train if you want to be precise)
# For consistency with how X was scaled in the initial setup, we fit on the full X data.
scaler = StandardScaler()
scaler.fit(X_full)
