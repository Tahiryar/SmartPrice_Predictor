# -------------------------------
# 01_EDA_and_Preprocessing.py
# -------------------------------

# 1️⃣ Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 2️⃣ Load Dataset
df = pd.read_csv("../data/mobile_sales_data.csv")

# 3️⃣ Quick Overview
print("Top 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 4️⃣ Basic Statistics
print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Columns:")
print(df.select_dtypes(include='object').nunique())

# 5️⃣ Distribution of Target (Price)
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# 6️⃣ Check Price by Brand
plt.figure(figsize=(12,6))
sns.boxplot(x='Brand', y='Price', data=df)
plt.xticks(rotation=45)
plt.title("Price Distribution by Brand")
plt.show()

# 7️⃣ Check Price by Product Type
plt.figure(figsize=(6,4))
sns.boxplot(x='Product_Type', y='Price', data=df)
plt.title("Price by Product Type")
plt.show()

# 8️⃣ Missing Values Handling (Basic)
# Fill numerical missing values with median
numerical_cols = df.select_dtypes(include=['int64','float64']).columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 9️⃣ Encode Categorical Columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nData after encoding:")
print(df.head())

# 10️⃣ Save Preprocessed Data (Optional)
df.to_csv("data/processed_data.csv", index=False)
print("\nPreprocessed dataset saved as 'processed_data.csv'")
