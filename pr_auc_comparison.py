import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve

from pipeline import prepare_data

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5

print("=" * 70)
print("PR-AUC COMPARISON: Raw RF vs RF+Isotonic vs RF+Sigmoid")
print("=" * 70)

df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train set: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Fraud rate: {y.sum()/len(y)*100:.4f}%")
print(f"Structure: {N_OUTER} outer x {N_INNER} inner folds")

outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

results = {
    "Raw RF": [],
    "Sigmoid": [],
    "Isotonic": [],
}

for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Outer Fold {outer_fold}/{N_OUTER} ---")
    
    X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
    y_outer_train = y.iloc[outer_train_idx].reset_index(drop=True)
    X_outer_val = X.iloc[outer_val_idx]
    y_outer_val = y.iloc[outer_val_idx]
    
    # Inner CV for calibrator training
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    inner_oof_preds = np.zeros(len(y_outer_train))
    
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
        rf_inner = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        rf_inner.fit(X_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_train_idx])
        inner_oof_preds[inner_val_idx] = rf_inner.predict_proba(X_outer_train.iloc[inner_val_idx])[:, 1]
    
    # Fit calibrators on INNER OOF
    sigmoid_cal = LogisticRegression(random_state=RANDOM_STATE)
    sigmoid_cal.fit(inner_oof_preds.reshape(-1, 1), y_outer_train)
    
    isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    isotonic_cal.fit(inner_oof_preds, y_outer_train)
    
    # Train final RF on FULL outer training set
    rf_final = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_outer_train, y_outer_train)
    
    # Predict on OUTER validation fold
    raw_proba = rf_final.predict_proba(X_outer_val)[:, 1]
    sigmoid_proba = sigmoid_cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    isotonic_proba = isotonic_cal.predict(raw_proba)
    
    # Compute PR-AUC
    pr_auc_raw = average_precision_score(y_outer_val, raw_proba)
    pr_auc_sig = average_precision_score(y_outer_val, sigmoid_proba)
    pr_auc_iso = average_precision_score(y_outer_val, isotonic_proba)
    
    results["Raw RF"].append(pr_auc_raw)
    results["Sigmoid"].append(pr_auc_sig)
    results["Isotonic"].append(pr_auc_iso)
    
    print(f"  PR-AUC Raw:      {pr_auc_raw:.4f}")
    print(f"  PR-AUC Sigmoid:  {pr_auc_sig:.4f}")
    print(f"  PR-AUC Isotonic: {pr_auc_iso:.4f}")

print("\n" + "=" * 70)
print("RESULTS: Mean +/- Std across 5 outer folds")
print("=" * 70)
print(f"\n{'Method':<12} {'PR-AUC Mean':>12} {'Std':>10}")
print("-" * 36)
for name in ["Raw RF", "Sigmoid", "Isotonic"]:
    vals = results[name]
    print(f"{name:<12} {np.mean(vals):12.4f} {np.std(vals):10.4f}")

print("-" * 36)
best = max(results, key=lambda k: np.mean(results[k]))
print(f"\nBest PR-AUC: {best} ({np.mean(results[best]):.4f} ± {np.std(results[best]):.4f})")

# Also compute on aggregated (optional)
all_raw = np.concatenate([r for r in results["Raw RF"]])  # Not meaningful, just per-fold
# Actually aggregate all outer predictions for a single PR-AUC
print("\nNote: These are per-fold PR-AUCs averaged. For a single aggregate PR-AUC,")
print("you would concatenate all outer predictions and compute once.")

print("\nPR-AUC comparison complete.")