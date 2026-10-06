import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from assemble_data import assemble

DATA_PATH = "creditcard.csv"


def prepare_data():
    if not os.path.exists(DATA_PATH):
        assemble()
    df = pd.read_csv(DATA_PATH)

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=["Time"])

    df["Amount"] = np.log1p(df["Amount"])
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_train = pd.concat([X_train.iloc[train_idx], y_train.iloc[train_idx]], axis=1)
        fold_val = pd.concat([X_train.iloc[val_idx], y_train.iloc[val_idx]], axis=1)
        folds.append((fold_train, fold_val))

    return df, train, test, folds


def main():
    print("=" * 70)
    print("PIPELINE: Data Loading, Preprocessing & Splitting")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"Original dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Fraud: {df['Class'].sum()} | Legit: {(1-df['Class']).sum()}")

    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")

    df, train, test, folds = prepare_data()

    print(f"After removing duplicates: {df.shape[0]} rows")
    print(f"Dropped 'Time' column")
    print(f"\nTrain: {train.shape[0]} rows | Fraud rate: {train['Class'].sum()/len(train)*100:.4f}%")
    print(f"Test:  {test.shape[0]} rows | Fraud rate: {test['Class'].sum()/len(test)*100:.4f}%")

    print("\n--- 5-Fold Stratified CV (on train) ---")
    for i, (f_train, f_val) in enumerate(folds, 1):
        fraud_tr = f_train["Class"].sum()
        fraud_va = f_val["Class"].sum()
        print(f"Fold {i}: Train={len(f_train)} (fraud={fraud_tr}, {fraud_tr/len(f_train)*100:.4f}%) | "
              f"Val={len(f_val)} (fraud={fraud_va}, {fraud_va/len(f_val)*100:.4f}%)")

    print("\n" + "=" * 70)
    print("Dataframes available via prepare_data():")
    print(f"  df      - full preprocessed ({df.shape})")
    print(f"  train   - train split ({train.shape})")
    print(f"  test    - test split ({test.shape})")
    print(f"  folds   - list of (train, val) tuples (5 folds, same splits for all models)")


if __name__ == "__main__":
    main()
