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
    row_idx (int): The index of the row to retrieve from the test set (after splitting).
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

  # 5. Perform train-test split on raw data to get correct indices
  _, X_test_raw, _, _ = train_test_split(
      X_full, y_full, test_size=0.2, random_state=random_seed
  )

  # Reset index for consistent .iloc indexing with row_idx
  X_test_raw = X_test_raw.reset_index(drop=True)

  # 6. Fit preprocessor on training data (simulating fitting on the actual training set)
  # Note: A full X_train_raw would ideally be used for fitting, but for this example,
  # we're just setting up the preprocessor's scale and categories based on the full dataset or a hypothetical train set.
  # To ensure a robust fit, you might want to perform the full train_test_split and fit on X_train_raw.
  # For demonstration, we'll fit on X_full (cleaned) for simplicity, assuming a similar distribution.
  preprocessor.fit(X_full.drop('class', axis=1).replace('?', pd.NA).dropna())

  # 7. Retrieve the specific raw instance from the test set
  raw_instance = X_test_raw.iloc[[row_idx]] # Keep it as a DataFrame for preprocessor

  # 8. Preprocess the selected instance
  processed_instance = preprocessor.transform(raw_instance)

  # 9. Convert to PyTorch tensor and return (handle sparse output from OneHotEncoder)
  input_tensor = torch.tensor(processed_instance.toarray(), dtype=torch.float32).to(device)

  return input_tensor

# --- Example Usage ---
# Let's say you want to get the tensor for the 5th instance in the Adult test set (index 4)
# example_row_idx = 4
# preprocessed_adult_tensor = use_benchmark_adult(example_row_idx)
# print(f"Preprocessed Adult tensor for row_idx {example_row_idx}:\n{preprocessed_adult_tensor}")
# print(f"Shape: {preprocessed_adult_tensor.shape}")

# You can then pass this to your model:
# with torch.no_grad():
#     # Assuming 'model' is your trained PyTorch model
#     adult_prediction = model(preprocessed_adult_tensor)
#     print(f"Model prediction for Adult row_idx {example_row_idx}: {adult_prediction.item()}")
