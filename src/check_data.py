# check_data.py
import joblib
import os

# ✅ Paths to saved files
models_path = os.path.join(os.path.dirname(__file__), "../models")
X_train_path = os.path.join(models_path, "X_train.pkl")
y_train_path = os.path.join(models_path, "y_train.pkl")
X_test_path  = os.path.join(models_path, "X_test.pkl")
y_test_path  = os.path.join(models_path, "y_test.pkl")

# Load the preprocessed data
X_train = joblib.load(X_train_path)
y_train = joblib.load(y_train_path)
X_test  = joblib.load(X_test_path)
y_test  = joblib.load(y_test_path)

# ✅ Print shapes
print("Shapes of preprocessed data:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")

# ✅ Show first 5 rows of each
print("\nX_train sample:")
print(X_train.head())

print("\ny_train sample:")
print(y_train.head())

print("\nX_test sample:")
print(X_test.head())

print("\ny_test sample:")
print(y_test.head())
