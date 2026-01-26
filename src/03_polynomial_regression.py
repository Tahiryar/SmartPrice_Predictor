# 03_polynomial_regression.py
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np

# 🔹 Step 1: Load preprocessed data from pickle files
models_path = os.path.join(os.path.dirname(__file__), "../models")
X_train = joblib.load(os.path.join(models_path, "X_train.pkl"))
X_test  = joblib.load(os.path.join(models_path, "X_test.pkl"))
y_train = joblib.load(os.path.join(models_path, "y_train.pkl"))
y_test  = joblib.load(os.path.join(models_path, "y_test.pkl"))

# 🔹 Step 2: Select numeric features only
numeric_cols = ['screen_size','rear_camera_mp','front_camera_mp',
                'internal_memory','ram','battery','weight',
                'release_year','days_used']
X_train_num = X_train[numeric_cols]
X_test_num  = X_test[numeric_cols]

# 🔹 Step 3: Transform features to polynomial (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_num)
X_test_poly  = poly.transform(X_test_num)

# 🔹 Step 4: Train Linear Regression on polynomial features
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)

# 🔹 Step 5: Predict and evaluate
y_pred = poly_lr.predict(X_test_poly)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

# 🔹 Step 6: Display results
print("Polynomial Regression (degree=2) Results:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 🔹 Step 7: Save trained model
joblib.dump(poly_lr, os.path.join(models_path, "polynomial_model.pkl"))
print("\n✅ Model saved as 'polynomial_model.pkl' in the models folder.")
