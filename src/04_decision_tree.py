# 04_decision_tree.py
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np

# 🔹 Step 1: Load preprocessed data from pickle files
models_path = os.path.join(os.path.dirname(__file__), "../models")
X_train = joblib.load(os.path.join(models_path, "X_train.pkl"))
X_test  = joblib.load(os.path.join(models_path, "X_test.pkl"))
y_train = joblib.load(os.path.join(models_path, "y_train.pkl"))
y_test  = joblib.load(os.path.join(models_path, "y_test.pkl"))

# 🔹 Step 2: Initialize Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# 🔹 Step 3: Predict on test set
y_pred = dt.predict(X_test)

# 🔹 Step 4: Evaluate performance
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

# 🔹 Step 5: Display results
print("Decision Tree Regression Results:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 🔹 Step 6: Save trained model
joblib.dump(dt, os.path.join(models_path, "decision_tree_model.pkl"))
print("\n✅ Model saved as 'decision_tree_model.pkl' in the models folder.")
