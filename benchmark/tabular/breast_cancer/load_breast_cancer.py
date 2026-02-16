import os
import torch
import torch.nn as nn
import pandas as pd

from sklearn.datasets import load_breast_cancer # Changed from fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

def use_benchmark_breast_cancer(row_idx: int, is_spurious = False, random_seed: int = 42, device: str = "cpu"):
  """
  Prepares a specific instance from the Breast Cancer dataset for model inference,
  mimicking the preprocessing pipeline.

  Args:
    row_idx (int): The index of the row to retrieve from the test set (after splitting).
    is_spurious (bool): Set this to True when accessing the rows of Q8,9, and 10.
    random_seed (int): The random seed used for the train-test split.
    device (str): The torch device ('cpu' or 'cuda') for the output tensor.

  Returns:
    torch.Tensor: The preprocessed PyTorch tensor of the specified instance.
  """
  # 1. Load the Breast Cancer dataset
  data_full = load_breast_cancer(as_frame=True)
  df_raw = data_full.frame
  
  X_full = df_raw.drop('target', axis=1)
  y_full = df_raw['target'].astype(int) # Target is already 0/1 for Breast Cancer
  
  # 2. Define columns (all are numerical for Breast Cancer)
  categorical_cols = []
  numerical_cols = X_full.columns.tolist()
  
  # 3. Create preprocessor
  preprocessor = ColumnTransformer([
      ('num', StandardScaler(), numerical_cols)
  ])
  
  # 4. Perform train-test split on raw data to get correct indices
  X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
      X_full, y_full, test_size=0.2, random_state=random_seed, stratify=y_full
  )
  
  # 5. Fit preprocessor on training data
  preprocessor.fit(X_train_raw)

  # 6. Retrieve the specific raw instance from the test set
  # We use .iloc[row_idx] as the JSON row_idx corresponds to this index
  if not is_spurious:
    raw_instance = X_test_raw.iloc[[row_idx]] # Keep it as a DataFrame for preprocessor

  raw_instance = df_raw.iloc[[row_idx]]
  
  
  # 7. Preprocess the selected instance
  processed_instance = preprocessor.transform(raw_instance)
  
  # 8. Convert to PyTorch tensor and return
  input_tensor = torch.tensor(processed_instance, dtype=torch.float32).to(device)
  
  return raw_instance, input_tensor
