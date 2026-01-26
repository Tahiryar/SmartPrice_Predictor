# -------------------------------
# 02_Preprocessing.py
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("../data/used_device_data.csv")
print(df.shape)

# Target column
TARGET = "normalized_used_price"

# -------------------------------
# Handle Missing Values
# -------------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include="object").columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# Encode Categorical Columns
# -------------------------------
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Feature / Target split
# -------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------------------
# Train-Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# -------------------------------
# Save processed data
# -------------------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(X_train, "../models/X_train.pkl")
joblib.dump(X_test, "../models/X_test.pkl")
joblib.dump(y_train, "../models/y_train.pkl")
joblib.dump(y_test, "../models/y_test.pkl")

print("Preprocessing completed & data saved ✅")
