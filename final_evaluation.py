import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, fbeta_score, confusion_matrix,
    average_precision_score, brier_score_loss, roc_auc_score
)

from pipeline import prepare_data

RANDOM_STATE = 42
N_FOLDS = 5
LOCKED_THRESHOLD = 0.214286

warnings.filterwarnings("ignore")

print("=" * 70)
print("FINAL EVALUATION: Default RF + Isotonic on Full Train -> Test")
print("=" * 70)

# Load data
df, train, test, folds = prepare_data()
X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

print(f"Train: {X_train.shape[0]} rows, Fraud: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.4f}%)")
print(f"Test:  {X_test.shape[0]} rows, Fraud: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.4f}%)")

# Step 1: Get OOF predictions on Train for calibrator (nested CV)
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_raw_proba = []
all_y_val = []

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
    X_outer_train = X_train.iloc[train_idx].reset_index(drop=True)
    y_outer_train = y_train.iloc[train_idx].reset_index(drop=True)
    X_outer_val = X_train.iloc[val_idx]
    y_outer_val = y_train.iloc[val_idx]
    
    # Inner CV for calibrator
    inner_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_oof_preds = np.zeros(len(y_outer_train))
    
    for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
        rf_inner = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        rf_inner.fit(X_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_train_idx])
        inner_oof_preds[inner_val_idx] = rf_inner.predict_proba(X_outer_train.iloc[inner_val_idx])[:, 1]
    
    # Fit calibrator on inner OOF
    isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    isotonic_cal.fit(inner_oof_preds, y_outer_train)
    
    # Train final RF on full outer train
    rf_final = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_outer_train, y_outer_train)
    
    # Predict on outer validation
    raw_proba = rf_final.predict_proba(X_outer_val)[:, 1]
    
    all_raw_proba.append(raw_proba)
    all_y_val.append(y_outer_val)

# Aggregate OOF predictions
oof_raw = np.concatenate(all_raw_proba)
oof_y = np.concatenate(all_y_val)

# Fit final isotonic calibrator on ALL OOF
final_iso = IsotonicRegression(out_of_bounds="clip")
final_iso.fit(oof_raw, oof_y)

print(f"\nCalibrator fitted on {len(oof_raw)} OOF samples")

# Step 2: Train final RF on FULL Train set
print("\nTraining final RF on full Train set...")
rf_full = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_full.fit(X_train, y_train)

# Step 3: Predict on Test set
print("Predicting on Test set...")
test_raw_proba = rf_full.predict_proba(X_test)[:, 1]
test_iso_proba = final_iso.predict(test_raw_proba)

# Step 4: Apply locked threshold
test_pred = (test_iso_proba >= LOCKED_THRESHOLD).astype(int)

# Step 5: Evaluate
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

tp = ((test_pred == 1) & (y_test == 1)).sum()
fn = ((test_pred == 0) & (y_test == 1)).sum()
fp = ((test_pred == 1) & (y_test == 0)).sum()
tn = ((test_pred == 0) & (y_test == 0)).sum()

print(f"\nConfusion Matrix (threshold={LOCKED_THRESHOLD}):")
print(f"  TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"  Precision: {tp/(tp+fp):.4f}")
print(f"  Recall:    {tp/(tp+fn):.4f}")
print(f"  F2 Score:  {2*tp/(2*tp+fn+fp):.4f}")
print(f"  F1 Score:  {2*tp/(2*tp+fn+fp)*tp/(tp+fn)/tp*2 if tp>0 else 0:.4f}")

# F1 manually
prec = tp/(tp+fp) if (tp+fp) > 0 else 0
rec = tp/(tp+fn) if (tp+fn) > 0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
print(f"  F1 Score:  {f1:.4f}")

# Additional metrics
pr_auc = average_precision_score(y_test, test_iso_proba)
roc_auc = roc_auc_score(y_test, test_iso_proba)
brier = brier_score_loss(y_test, test_iso_proba)

print(f"\nPR-AUC:    {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"Brier:     {brier:.6f}")

# Calibration: reliability bins
print("\nCalibration bins (10 quantile bins):")
bins = np.quantile(test_iso_proba, np.linspace(0, 1, 11))
bins[0] = 0
bins[-1] = 1
for i in range(len(bins)-1):
    mask = (test_iso_proba >= bins[i]) & (test_iso_proba <= bins[i+1]) if i == len(bins)-2 else \
           (test_iso_proba >= bins[i]) & (test_iso_proba < bins[i+1])
    if mask.sum() > 0:
        bin_pred = test_pred[mask]
        bin_true = y_test[mask]
        bin_proba = test_iso_proba[mask]
        emp_rate = bin_true.mean()
        mean_proba = bin_proba.mean()
        print(f"  [{bins[i]:.4f}, {bins[i+1]:.4f}): n={mask.sum()}, mean_p={mean_proba:.4f}, emp_rate={emp_rate:.4f}")

# Compare with OOF performance
print("\n" + "=" * 70)
print("COMPARISON: OOF vs TEST")
print("=" * 70)

# OOF performance at locked threshold
oof_iso = final_iso.predict(oof_raw)
oof_pred = (oof_iso >= LOCKED_THRESHOLD).astype(int)
oof_tp = ((oof_pred == 1) & (oof_y == 1)).sum()
oof_fn = ((oof_pred == 0) & (oof_y == 1)).sum()
oof_fp = ((oof_pred == 1) & (oof_y == 0)).sum()

print(f"OOF  - Precision: {oof_tp/(oof_tp+oof_fp):.4f}, Recall: {oof_tp/(oof_tp+oof_fn):.4f}, F2: {2*oof_tp/(2*oof_tp+oof_fn+oof_fp):.4f}")
print(f"Test - Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}, F2: {2*tp/(2*tp+fn+fp):.4f}")

print("\nFinal evaluation complete.")

# Save results
results = {
    "threshold": LOCKED_THRESHOLD,
    "test": {
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "precision": float(tp/(tp+fp)),
        "recall": float(tp/(tp+fn)),
        "f2": float(2*tp/(2*tp+fn+fp)),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier": float(brier)
    },
    "oof": {
        "tp": int(oof_tp), "fn": int(oof_fn), "fp": int(oof_fp),
        "precision": float(oof_tp/(oof_tp+oof_fp)),
        "recall": float(oof_tp/(oof_tp+oof_fn)),
        "f2": float(2*oof_tp/(2*oof_tp+oof_fn+oof_fp))
    }
}

import json
with open("final_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to final_evaluation_results.json")