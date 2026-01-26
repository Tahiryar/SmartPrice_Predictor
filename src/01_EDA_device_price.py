# -----------------------------------
# 01_EDA_device_price.py
# -----------------------------------

# 1️⃣ Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 2️⃣ Load Dataset
df = pd.read_csv("../data/used_device_data.csv")

# 3️⃣ Basic Overview
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# 4️⃣ Missing Values
print("\nMissing Values per column:")
print(df.isnull().sum())

# 5️⃣ Separate Column Types
categorical_cols = df.select_dtypes(include="object").columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# 6️⃣ Statistical Summary
print("\nNumerical Summary:")
print(df[numerical_cols].describe())

# 7️⃣ Target Variable Analysis
TARGET = "normalized_used_price"

plt.figure(figsize=(8,5))
sns.histplot(df[TARGET], bins=40, kde=True)
plt.title("Distribution of Used Device Price")
plt.xlabel("Normalized Used Price")
plt.ylabel("Count")
plt.show()

# 8️⃣ Price vs Important Numerical Features
important_numeric = [
    "ram",
    "internal_memory",
    "battery",
    "screen_size",
    "days_used",
    "release_year",
    "normalized_new_price"
]

for col in important_numeric:
    if col in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df[TARGET])
        plt.title(f"{col} vs Used Price")
        plt.xlabel(col)
        plt.ylabel("Used Price")
        plt.show()

# 9️⃣ Price vs Categorical Features
for col in categorical_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(x=df[col], y=df[TARGET])
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Used Price")
    plt.show()

# 🔟 Correlation Analysis
plt.figure(figsize=(10,6))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

print("\nCorrelation with Target:")
print(
    corr[TARGET].sort_values(ascending=False)
)

print("\nEDA Completed Successfully ✅")
