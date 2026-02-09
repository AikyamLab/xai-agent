import os
import torch
import torch.nn as nn
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

adult = fetch_openml(name="adult", version=2, as_frame=True)
df_raw = adult.frame

df_raw = df_raw.replace('?', pd.NA).dropna()

df_cleaned_original_indices = df_raw.copy()

X_full = df_raw.drop('class', axis=1)
y_full = (df_raw['class'] == '>50K').astype(int)

categorical_cols = X_full.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_full.select_dtypes(exclude=['object', 'category']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

preprocessor.fit(X_train_raw)
X_train_processed = preprocessor.transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

X_test_torch = torch.tensor(X_test_processed.toarray(), dtype=torch.float32)
y_test_torch = torch.tensor(y_test_raw.values, dtype=torch.float32).view(-1, 1)
