# 09_compare_models.py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Models folder
models_path = os.path.join(os.path.dirname(__file__), "../models")

# List of saved models
model_files = {
    "Linear Regression": "linear_model.pkl",
    "Polynomial Regression (deg2)": "polynomial_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Ridge Regression": "ridge_model.pkl",
    "Lasso Regression": "lasso_model.pkl",
    "ElasticNet": "elasticnet_model.pkl",
    "SVR (RBF)": "svr_rbf_model.pkl"
}

# Load preprocessed data
X_train = joblib.load(os.path.join(models_path, "X_train.pkl"))
X_test  = joblib.load(os.path.join(models_path, "X_test.pkl"))
y_train = joblib.load(os.path.join(models_path, "y_train.pkl"))
y_test  = joblib.load(os.path.join(models_path, "y_test.pkl"))

# For scaled features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Prepare results dictionary
results = {"Model": [], "MAE": [], "RMSE": [], "R²": []}

for name, file in model_files.items():
    model = joblib.load(os.path.join(models_path, file))
    
    # Decide whether to use scaled features
    if name in ["Ridge Regression","Lasso Regression","ElasticNet","SVR (RBF)"]:
        X_test_input = X_test_scaled
    elif name == "Polynomial Regression (deg2)":
        # For polynomial, we need numeric features and PolynomialFeatures transform
        numeric_cols = ['screen_size','rear_camera_mp','front_camera_mp',
                        'internal_memory','ram','battery','weight',
                        'release_year','days_used']
        from sklearn.preprocessing import PolynomialFeatures
        df_numeric = pd.DataFrame(X_test, columns=X_train.columns)[numeric_cols]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_test_input = poly.fit_transform(df_numeric)
    else:
        X_test_input = X_test

    y_pred = model.predict(X_test_input)
    
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    results["Model"].append(name)
    results["MAE"].append(mae)
    results["RMSE"].append(rmse)
    results["R²"].append(r2)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot comparison
metrics = ["MAE", "RMSE", "R²"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

plt.figure(figsize=(12,6))
x = np.arange(len(results_df["Model"]))
width = 0.25

plt.bar(x - width, results_df["MAE"], width, color=colors[0], label="MAE")
plt.bar(x, results_df["RMSE"], width, color=colors[1], label="RMSE")
plt.bar(x + width, results_df["R²"], width, color=colors[2], label="R²")

plt.xticks(x, results_df["Model"], rotation=45, ha="right")
plt.ylabel("Metric Value")
plt.title("Regression Model Comparison: MAE, RMSE, R²")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
