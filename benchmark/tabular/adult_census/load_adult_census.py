import os
import torch
import torch.nn as nn
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Define default values for consistency, can be overwritten by args
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def use_benchmark_adult(row_idx: int, random_seed: int = RANDOM_SEED, device: str = DEVICE):
  """
  Prepares a specific instance from the Adult Census dataset for model inference,
  mimicking the preprocessing pipeline.

  Args:
    row_idx (int): The original index of the row to retrieve from the full dataset `df_raw`.
                          This instance must also be part of the test set split.
    random_seed (int): The random seed used for the train-test split.
    device (str): The torch device ('cpu' or 'cuda') for the output tensor.

  Returns:
    torch.Tensor: The preprocessed PyTorch tensor of the specified instance.
  """
  # 1. Load the Adult dataset
  adult = fetch_openml(name="adult", version=2, as_frame=True)
  df_raw = adult.frame

  # 2. Clean the data (as typically done for this dataset)
  df_raw = df_raw.replace('?', pd.NA).dropna()

  X_full = df_raw.drop('class', axis=1)
  y_full = (df_raw['class'] == '>50K').astype(int) # Convert target to 0/1

  # 3. Define columns for preprocessing
  categorical_cols = X_full.select_dtypes(include=['object', 'category']).columns.tolist()
  numerical_cols = X_full.select_dtypes(exclude=['object', 'category']).columns.tolist()

  # 4. Create preprocessor
  preprocessor = ColumnTransformer([
      ('num', StandardScaler(), numerical_cols),
      ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
  ])

  # 5. Perform train-test split on raw data, retaining original indices
  X_train_raw, X_test_raw, _, _ = train_test_split(
      X_full, y_full, test_size=0.2, random_state=random_seed
  )

  # 6. Find the instance corresponding to row_idx within the test set
  # Check if the row_idx actually made it into the test set
  if row_idx not in X_test_raw.index:
      raise ValueError(
          f"Original index {row_idx} was not found in the test set "
          f"(after train_test_split and cleaning). It might be in the training set or was dropped during cleaning."
      )

  # Retrieve the specific raw instance from the test set using its original index
  raw_instance = X_test_raw.loc[[row_idx]] # Keep it as a DataFrame for preprocessor

  # 7. Fit preprocessor on training data (simulating fitting on the actual training set)
  # We fit on X_train_raw for a more realistic scenario.
  preprocessor.fit(X_train_raw.replace('?', pd.NA).dropna())

  # 8. Preprocess the selected instance
  processed_instance = preprocessor.transform(raw_instance)

  # 9. Convert to PyTorch tensor and return (handle sparse output from OneHotEncoder)
  input_tensor = torch.tensor(processed_instance.toarray(), dtype=torch.float32).to(device)

  return raw_instance, input_tensor

# --- Example Usage (will be handled by the original call, now that the function is fixed) ---
