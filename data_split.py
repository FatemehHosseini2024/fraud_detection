import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

PREPROCESSED_PATH = "preprocessed_data/creditcard_preprocessed.csv"
OUTPUT_DIR = "data_splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Data Splitting")
print("=" * 70)

df = pd.read_csv(PREPROCESSED_PATH)
print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

X = df.drop(columns=["Class"])
y = df["Class"]

print(f"\nTotal fraud cases: {y.sum()} ({y.sum()/len(y)*100:.4f}%)")
print(f"Total legit cases: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.4f}%)")

print("\n" + "-" * 40)
print("Train/Test Split (80/20, stratified)")
print("-" * 40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape[0]} rows")
print(f"  Fraud:      {y_train.sum()} ({y_train.sum()/len(y_train)*100:.4f}%)")
print(f"  Legit:      {(1-y_train).sum()} ({(1-y_train).sum()/len(y_train)*100:.4f}%)")
print(f"Test set:  {X_test.shape[0]} rows")
print(f"  Fraud:      {y_test.sum()} ({y_test.sum()/len(y_test)*100:.4f}%)")
print(f"  Legit:      {(1-y_test).sum()} ({(1-y_test).sum()/len(y_test)*100:.4f}%)")

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}/train.csv, {OUTPUT_DIR}/test.csv")

print("\n" + "-" * 40)
print("5-Fold Stratified Cross-Validation (on train)")
print("-" * 40)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_num = 0
for train_idx, val_idx in skf.split(X_train, y_train):
    fold_num += 1
    cv_train_fraud = y_train.iloc[train_idx].sum()
    cv_val_fraud = y_train.iloc[val_idx].sum()
    print(f"\nFold {fold_num}:")
    print(f"  Train: {len(train_idx)} rows | Fraud: {cv_train_fraud} ({cv_train_fraud/len(train_idx)*100:.4f}%)")
    print(f"  Val:   {len(val_idx)} rows | Fraud: {cv_val_fraud} ({cv_val_fraud/len(val_idx)*100:.4f}%)")
    fold_train = pd.concat([X_train.iloc[train_idx], y_train.iloc[train_idx]], axis=1)
    fold_val = pd.concat([X_train.iloc[val_idx], y_train.iloc[val_idx]], axis=1)
    fold_train.to_csv(f"{OUTPUT_DIR}/train_fold_{fold_num}.csv", index=False)
    fold_val.to_csv(f"{OUTPUT_DIR}/val_fold_{fold_num}.csv", index=False)

print(f"\nSaved: {OUTPUT_DIR}/train_fold_*.csv, {OUTPUT_DIR}/val_fold_*.csv")
print("\nData splitting complete.")
