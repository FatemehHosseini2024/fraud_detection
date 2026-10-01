import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = "creditcard.csv"
OUTPUT_PATH = "creditcard_preprocessed.csv"

print("=" * 70)
print("Data Preprocessing")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Original dataset: {df.shape[0]} rows x {df.shape[1]} columns")

dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")

df = df.drop_duplicates().reset_index(drop=True)
print(f"After removing duplicates: {df.shape[0]} rows")

df = df.drop(columns=["Time"])
print("Dropped 'Time' column after deduplication")
print(f"Current dataset: {df.shape[0]} rows x {df.shape[1]} columns")

print("\nApplying log1p transformation to 'Amount' column...")
df["Amount"] = np.log1p(df["Amount"])

print(f"Amount statistics after log1p:")
print(df["Amount"].describe().round(4))

print("\nStandardizing 'Amount' column using StandardScaler...")
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

print(f"Amount statistics after standardization:")
print(df["Amount"].describe().round(4))

os.makedirs("preprocessed_data", exist_ok=True)
df.to_csv(f"preprocessed_data/{OUTPUT_PATH}", index=False)
print(f"\nSaved preprocessed dataset to: preprocessed_data/{OUTPUT_PATH}")
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
