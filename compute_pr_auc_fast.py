import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score

from pipeline import prepare_data

RANDOM_STATE = 42
N_FOLDS = 5

warnings.filterwarnings("ignore")

print("=" * 70)
print("PR-AUC: Default RF + Isotonic (5-fold OOF, fast - exact error_analysis_fast model)")
print("=" * 70)

# Load data
df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train: {X.shape[0]} rows, Fraud: {y.sum()} ({y.sum()/len(y)*100:.4f}%)")

# Single 5-fold CV for OOF predictions (same as error_analysis_fast.py)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_raw_proba = []
all_y_val = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"  Fold {fold}/{N_FOLDS}...")
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    raw_proba = rf.predict_proba(X_val)[:, 1]
    
    all_raw_proba.append(raw_proba)
    all_y_val.append(y_val)

# Aggregate OOF
oof_raw = np.concatenate(all_raw_proba)
oof_y = np.concatenate(all_y_val)

print(f"\nOOF collected: {len(oof_raw)} samples")

# Fit isotonic on ALL OOF (same as error_analysis_fast.py)
final_iso = IsotonicRegression(out_of_bounds="clip")
final_iso.fit(oof_raw, oof_y)
oof_iso = final_iso.predict(oof_raw)

# PR-AUC on calibrated OOF probabilities
pr_auc_iso = average_precision_score(oof_y, oof_iso)
pr_auc_raw = average_precision_score(oof_y, oof_raw)

print(f"\nPR-AUC (Raw RF OOF):      {pr_auc_raw:.4f}")
print(f"PR-AUC (Isotonic OOF):    {pr_auc_iso:.4f}")

# Find locked threshold: Max F2 on unique isotonic outputs
unique_iso = np.unique(oof_iso)
unique_iso = unique_iso[unique_iso > 0]
best_f2 = -1
LOCKED_THRESHOLD = None
for thresh in unique_iso:
    y_pred = (oof_iso >= thresh).astype(int)
    prec = precision_score(oof_y, y_pred, zero_division=0)
    rec = recall_score(oof_y, y_pred, zero_division=0)
    f2 = fbeta_score(oof_y, y_pred, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2 = f2
        LOCKED_THRESHOLD = thresh

print(f"\nLocked threshold (Max F2): {LOCKED_THRESHOLD:.6f}")

# Metrics at locked threshold
oof_pred = (oof_iso >= LOCKED_THRESHOLD).astype(int)
tp = ((oof_pred == 1) & (oof_y == 1)).sum()
fn = ((oof_pred == 0) & (oof_y == 1)).sum()
fp = ((oof_pred == 1) & (oof_y == 0)).sum()
tn = ((oof_pred == 0) & (oof_y == 0)).sum()

print(f"\nConfusion Matrix (threshold={LOCKED_THRESHOLD:.6f}):")
print(f"  TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"  Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}")
print(f"  F1: {fbeta_score(oof_y, oof_pred, beta=1, zero_division=0):.4f}, "
      f"F2: {fbeta_score(oof_y, oof_pred, beta=2, zero_division=0):.4f}")