import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from collections import defaultdict

from pipeline import prepare_data

RANDOM_STATE = 42
N_SPLITS = 5

print("=" * 70)
print("PROBABILITY CALIBRATION: Nested CV with RF + Sigmoid/Isotonic")
print("=" * 70)

# Load data (uses same preprocessing as pipeline.py)
df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Outer CV on train set: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Fraud rate: {y.sum()/len(y)*100:.4f}%")
print(f"Folds: {N_SPLITS} outer x {N_SPLITS} inner")

# Reference Brier (constant prediction = fraud rate)
fraud_rate = y.sum() / len(y)
reference_brier = fraud_rate * (1 - fraud_rate)
print(f"\nReference Brier (constant pred = fraud rate): {reference_brier:.6f}")

outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

brier_raw = []
brier_sigmoid = []
brier_isotonic = []

for fold_num, (outer_train_idx, outer_val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Outer Fold {fold_num}/{N_SPLITS} ---")
    X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
    y_outer_train = y.iloc[outer_train_idx].reset_index(drop=True)
    X_outer_val = X.iloc[outer_val_idx]
    y_outer_val = y.iloc[outer_val_idx]

    print(f"  Outer train: {len(X_outer_train)} | Outer val: {len(X_outer_val)}")

    # Step 2: Inner OOF predictions
    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y_outer_train))

    for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
        X_inner_train = X_outer_train.iloc[inner_train_idx]
        y_inner_train = y_outer_train.iloc[inner_train_idx]
        X_inner_val = X_outer_train.iloc[inner_val_idx]

        rf_inner = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        rf_inner.fit(X_inner_train, y_inner_train)
        oof_preds[inner_val_idx] = rf_inner.predict_proba(X_inner_val)[:, 1]

    print(f"  OOF predictions collected: {len(oof_preds)} samples")

    # Step 3: Fit calibrators on OOF predictions
    sigmoid_cal = LogisticRegression(random_state=RANDOM_STATE)
    sigmoid_cal.fit(oof_preds.reshape(-1, 1), y_outer_train)

    isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    isotonic_cal.fit(oof_preds, y_outer_train)

    # Step 4: Train final RF on entire outer training set
    rf_final = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_outer_train, y_outer_train)

    # Step 5: Predict on outer validation fold
    raw_proba = rf_final.predict_proba(X_outer_val)[:, 1]
    sigmoid_proba = sigmoid_cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    isotonic_proba = isotonic_cal.predict(raw_proba)

    # Compute Brier scores
    b_raw = brier_score_loss(y_outer_val, raw_proba)
    b_sig = brier_score_loss(y_outer_val, sigmoid_proba)
    b_iso = brier_score_loss(y_outer_val, isotonic_proba)

    brier_raw.append(b_raw)
    brier_sigmoid.append(b_sig)
    brier_isotonic.append(b_iso)

    print(f"  Brier raw:       {b_raw:.6f}")
    print(f"  Brier sigmoid:   {b_sig:.6f}")
    print(f"  Brier isotonic:  {b_iso:.6f}")

print("\n" + "=" * 70)
print("RESULTS: Mean +/- Std across 5 outer folds")
print("=" * 70)
print(f"\n{'Method':<20} {'Mean Brier':>12} {'Std Brier':>12}")
print("-" * 44)
print(f"{'Raw RF':<20} {np.mean(brier_raw):12.6f} {np.std(brier_raw):12.6f}")
print(f"{'Sigmoid':<20} {np.mean(brier_sigmoid):12.6f} {np.std(brier_sigmoid):12.6f}")
print(f"{'Isotonic':<20} {np.mean(brier_isotonic):12.6f} {np.std(brier_isotonic):12.6f}")
print("-" * 44)
print(f"{'Reference':<20} {reference_brier:12.6f} {'':>12}")

print(f"\nLower Brier = better (closer to 0).")
print(f"Reference Brier (constant fraud-rate prediction): {reference_brier:.6f}")
print(f"\nBest method: {min(np.mean(brier_raw), np.mean(brier_sigmoid), np.mean(brier_isotonic))}")
print("Calibration analysis complete.")
