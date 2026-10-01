import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = "creditcard.csv"
OUTPUT_PATH = "creditcard_preprocessed.csv"

print("=" * 70)
print("Data Preprocessing")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Original dataset: {df.shape[0]} rows x {df.shape[1]} columns")

df.insert(0, "transaction_id", range(len(df)))
print("Added 'transaction_id' column (unique values)")

dup_cols = [col for col in df.columns if col != "transaction_id"]
dup_count = df.duplicated(subset=dup_cols).sum()
print(f"Duplicate rows (excluding transaction_id): {dup_count}")

df = df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
print(f"After removing duplicates: {df.shape[0]} rows")

print("\nStandardizing 'Amount' column using StandardScaler...")
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

print(f"Amount statistics after standardization:")
print(df["Amount"].describe().round(4))

os.makedirs("preprocessed_data", exist_ok=True)
df.to_csv(f"preprocessed_data/{OUTPUT_PATH}", index=False)
print(f"\nSaved preprocessed dataset to: preprocessed_data/{OUTPUT_PATH}")
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
