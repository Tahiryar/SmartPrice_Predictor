import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Load processed data
data = pd.read_csv("../data/processed_data.csv")

print("Dataset shape:", data.shape)

# Target column (price)
target_column = "Price"
  # ⚠️ make sure yahi naam ho

X = data.drop(columns=[target_column])
y = data[target_column]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n====== Linear Regression Baseline ======")
print("R2 Score :", r2)
print("MAE      :", mae)
print("RMSE     :", rmse)

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/linear_model.pkl")

print("\nModel saved as models/linear_model.pkl")
