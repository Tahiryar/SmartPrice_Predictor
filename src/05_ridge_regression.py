# 05_ridge_regression.py
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np

# 🔹 Step 1: Load preprocessed data from pickle files
models_path = os.path.join(os.path.dirname(__file__), "../models")
X_train = joblib.load(os.path.join(models_path, "X_train.pkl"))
X_test  = joblib.load(os.path.join(models_path, "X_test.pkl"))
y_train = joblib.load(os.path.join(models_path, "y_train.pkl"))
y_test  = joblib.load(os.path.join(models_path, "y_test.pkl"))

# 🔹 Step 2: Scale features (Ridge Regression is sensitive to scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 🔹 Step 3: Initialize and train Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# 🔹 Step 4: Predict on test set
y_pred = ridge.predict(X_test_scaled)

# 🔹 Step 5: Evaluate performance
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

# 🔹 Step 6: Display results
print("Ridge Regression Results:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 🔹 Step 7: Save trained model
joblib.dump(ridge, os.path.join(models_path, "ridge_model.pkl"))
print("\n✅ Model saved as 'ridge_model.pkl' in the models folder.")
