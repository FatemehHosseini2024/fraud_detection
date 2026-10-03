import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from pipeline import prepare_data

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5
N_BINS = 10

print("=" * 70)
print("DETAILED BIN STATISTICS (Nested CV)")
print("=" * 70)

df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train: {X.shape[0]} rows")
print(f"Fraud rate: {y.sum()/len(y)*100:.4f}%")

# Get OOF predictions via outer CV only (faster - no nested inner loop for this analysis)
# We'll use the outer CV to get outer validation predictions, 
# and for calibrators, we use the OOF from the outer CV directly (as in the previous scripts)
# This is slightly different from full nested but gives us the data quickly

outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

all_raw_proba = []
all_sigmoid_proba = []
all_isotonic_proba = []
all_y_val = []

for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"  Fold {outer_fold}...")
    
    X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
    y_outer_train = y.iloc[outer_train_idx].reset_index(drop=True)
    X_outer_val = X.iloc[outer_val_idx]
    y_outer_val = y.iloc[outer_val_idx]
    
    # Inner OOF on outer training set for calibrators
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    inner_oof_preds = np.zeros(len(y_outer_train))
    
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
        rf_inner = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        rf_inner.fit(X_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_train_idx])
        inner_oof_preds[inner_val_idx] = rf_inner.predict_proba(X_outer_train.iloc[inner_val_idx])[:, 1]
    
    # Fit calibrators
    sigmoid_cal = LogisticRegression(random_state=RANDOM_STATE)
    sigmoid_cal.fit(inner_oof_preds.reshape(-1, 1), y_outer_train)
    
    isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    isotonic_cal.fit(inner_oof_preds, y_outer_train)
    
    # Final RF
    rf_final = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_outer_train, y_outer_train)
    
    # Predictions on outer validation
    raw_proba = rf_final.predict_proba(X_outer_val)[:, 1]
    sigmoid_proba = sigmoid_cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    isotonic_proba = isotonic_cal.predict(raw_proba)
    
    all_raw_proba.append(raw_proba)
    all_sigmoid_proba.append(sigmoid_proba)
    all_isotonic_proba.append(isotonic_proba)
    all_y_val.append(y_outer_val)

all_raw = np.concatenate(all_raw_proba)
all_sig = np.concatenate(all_sigmoid_proba)
all_iso = np.concatenate(all_isotonic_proba)
all_y = np.concatenate(all_y_val)

print(f"\nAggregated: {len(all_y)} samples, {all_y.sum()} frauds")

# Quantile bins based on raw
sorted_idx = np.argsort(all_raw)
bin_indices = np.zeros(len(all_raw), dtype=int)
bin_size = len(all_raw) // N_BINS
for i in range(N_BINS):
    start = i * bin_size
    end = (i + 1) * bin_size if i < N_BINS - 1 else len(all_raw)
    bin_indices[sorted_idx[start:end]] = i

curves = {"Raw RF": all_raw, "Sigmoid": all_sig, "Isotonic": all_iso}

# Detailed table
print(f"\n{'Bin':<4} {'Total':>8} {'Frauds':>8} | "
      f"{'Raw Pred':>12} {'Raw Actual':>12} {'Raw Err':>10} | "
      f"{'Sig Pred':>12} {'Sig Actual':>12} {'Sig Err':>10} | "
      f"{'Iso Pred':>12} {'Iso Actual':>12} {'Iso Err':>10}")
print("-" * 130)

for i in range(N_BINS):
    mask = (bin_indices == i)
    if mask.sum() == 0:
        continue
    total = mask.sum()
    frauds = all_y[mask].sum()
    
    row = f"{i:<4} {total:>8} {frauds:>8} | "
    for method_name, proba in curves.items():
        mean_pred = proba[mask].mean()
        actual_rate = all_y[mask].mean()
        error = mean_pred - actual_rate
        row += f"{mean_pred:>12.6f} {actual_rate:>12.6f} {error:>+10.6f} | "
    print(row)

print(f"\nTotal: {len(all_y)} samples, {all_y.sum()} frauds")

# Save to CSV
import csv
with open("eda_outputs/bin_details.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Bin", "Total", "Frauds", 
                     "Raw_Mean_Pred", "Raw_Actual_Rate", "Raw_Error",
                     "Sigmoid_Mean_Pred", "Sigmoid_Actual_Rate", "Sigmoid_Error",
                     "Isotonic_Mean_Pred", "Isotonic_Actual_Rate", "Isotonic_Error"])
    for i in range(N_BINS):
        mask = (bin_indices == i)
        if mask.sum() == 0:
            continue
        total = mask.sum()
        frauds = all_y[mask].sum()
        row = [i, total, frauds]
        for method_name, proba in curves.items():
            mean_pred = proba[mask].mean()
            actual_rate = all_y[mask].mean()
            error = mean_pred - actual_rate
            row.extend([mean_pred, actual_rate, error])
        writer.writerow(row)

print("Saved: eda_outputs/bin_details.csv")
print("Done.")