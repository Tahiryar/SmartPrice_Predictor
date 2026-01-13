import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load dataset safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "mobile_sales_data.csv")

df = pd.read_csv(data_path)

print("Dataset Loaded:", df.shape)

# -----------------------------
# 2. Drop useless columns
# -----------------------------
drop_cols = [
    "Product Code",
    "Customer Name",
    "Customer Location",
    "Inward Date",
    "Dispatch Date",
    "Product Specification"
]

df.drop(columns=drop_cols, inplace=True)

print("After dropping useless columns:", df.shape)

# -----------------------------
# 3. Convert RAM, ROM, SSD to numbers
# -----------------------------

def convert_storage(val):
    if pd.isna(val):
        return 0
    val = val.upper().replace("GB","").replace("TB","")
    try:
        return int(val)
    except:
        return 0

df["RAM"] = df["RAM"].apply(convert_storage)
df["ROM"] = df["ROM"].apply(convert_storage)

# SSD has TB sometimes
def convert_ssd(val):
    if pd.isna(val):
        return 0
    val = val.upper()
    if "TB" in val:
        return int(val.replace("TB","")) * 1024
    if "GB" in val:
        return int(val.replace("GB",""))
    return 0

df["SSD"] = df["SSD"].apply(convert_ssd)

# -----------------------------
# 4. Handle Missing Values
# -----------------------------
df["Core Specification"] = df["Core Specification"].fillna("Unknown")

# -----------------------------
# 5. Encode categorical columns
# -----------------------------
cat_cols = df.select_dtypes(include="object").columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Categorical columns encoded")

# -----------------------------
# 6. Separate target
# -----------------------------
y = df["Price"]
X = df.drop("Price", axis=1)

print("Features:", X.shape)
print("Target:", y.shape)

# -----------------------------
# 7. Save ML-ready dataset
# -----------------------------
processed_path = os.path.join(BASE_DIR, "data", "processed_data.csv")
final_df = pd.concat([X, y], axis=1)
final_df.to_csv(processed_path, index=False)

print("\n✅ Preprocessing Complete")
print("ML-ready file saved at:")
print(processed_path)
