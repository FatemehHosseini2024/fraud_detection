import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

DATA_PATH = "creditcard.csv"


def main():
    print("=" * 70)
    print("PIPELINE: Data Loading, Preprocessing & Splitting")
    print("=" * 70)

    # Step 1: Read data once
    print("\n--- Step 1: Load Data ---")
    df = pd.read_csv(DATA_PATH)
    print(f"Original dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Fraud: {df['Class'].sum()} | Legit: {(1-df['Class']).sum()}")

    # Step 2: Preprocessing
    print("\n--- Step 2: Preprocessing ---")
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {df.shape[0]} rows")

    df = df.drop(columns=["Time"])
    print("Dropped 'Time' column")
    print(f"Columns now: {list(df.columns)}")

    print("\nApplying log1p to 'Amount'...")
    df["Amount"] = np.log1p(df["Amount"])

    print("Standardizing 'Amount'...")
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    print("\nAmount after log1p + StandardScaler:")
    print(df["Amount"].describe().round(4))

    # Step 3: Train/Test Split (80/20, stratified)
    print("\n--- Step 3: Train/Test Split (80/20, stratified) ---")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    fraud_rate_orig = y.sum() / len(y) * 100
    fraud_rate_train = y_train.sum() / len(y_train) * 100
    fraud_rate_test = y_test.sum() / len(y_test) * 100

    print(f"Train: {train.shape[0]} rows | Fraud rate: {fraud_rate_train:.4f}%")
    print(f"Test:  {test.shape[0]} rows | Fraud rate: {fraud_rate_test:.4f}%")
    print(f"Original fraud rate: {fraud_rate_orig:.4f}%")

    # Step 4: 5-Fold Stratified CV on train
    print("\n--- Step 4: 5-Fold Stratified CV (on train) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        fold_train = pd.concat([X_train.iloc[train_idx], y_train.iloc[train_idx]], axis=1)
        fold_val = pd.concat([X_train.iloc[val_idx], y_train.iloc[val_idx]], axis=1)

        fraud_tr = y_train.iloc[train_idx].sum()
        fraud_va = y_train.iloc[val_idx].sum()
        rate_tr = fraud_tr / len(train_idx) * 100
        rate_va = fraud_va / len(val_idx) * 100

        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_idx)} rows | Fraud: {fraud_tr} ({rate_tr:.4f}%)")
        print(f"  Val:   {len(val_idx)} rows | Fraud: {fraud_va} ({rate_va:.4f}%)")

        folds.append((fold_train, fold_val))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original rows:      {df.shape[0]}")
    print(f"Train rows:         {train.shape[0]}  (fraud rate: {fraud_rate_train:.4f}%)")
    print(f"Test rows:          {test.shape[0]}  (fraud rate: {fraud_rate_test:.4f}%)")
    print(f"CV folds:           5")
    print(f"  Each fold train:  {folds[0][0].shape[0]} rows")
    print(f"  Each fold val:    {folds[0][1].shape[0]} rows")

    print("\nDataframes available in memory:")
    print(f"  df      - full preprocessed ({df.shape})")
    print(f"  train   - train split ({train.shape})")
    print(f"  test    - test split ({test.shape})")
    print(f"  folds   - list of (train, val) tuples (5 folds)")


if __name__ == "__main__":
    main()
