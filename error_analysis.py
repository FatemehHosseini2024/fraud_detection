import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score

from pipeline import prepare_data

RANDOM_STATE = 42
N_FOLDS = 5

warnings.filterwarnings("ignore")

print("=" * 70)
print("ERROR ANALYSIS: Default RF + Isotonic (OOF only)")
print("=" * 70)

# 1. Load data and get OOF predictions (EXACT same as threshold_optimization.py)
df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train: {X.shape[0]} rows, Fraud: {y.sum()} ({y.sum()/len(y)*100:.4f}%)")

# Nested CV to get OOF predictions (same as threshold_optimization.py)
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_raw_proba = []
all_y_val = []

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"  Outer fold {fold}/{N_FOLDS}...")
    X_outer_train = X.iloc[train_idx].reset_index(drop=True)
    y_outer_train = y.iloc[train_idx].reset_index(drop=True)
    X_outer_val = X.iloc[val_idx]
    y_outer_val = y.iloc[val_idx]
    
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
    iso_proba = isotonic_cal.predict(raw_proba)
    
    all_raw_proba.append(raw_proba)
    all_y_val.append(y_outer_val)

# Aggregate OOF predictions
oof_raw = np.concatenate(all_raw_proba)
oof_y = np.concatenate(all_y_val)

# Fit isotonic on ALL OOF (same as threshold_optimization.py)
final_iso = IsotonicRegression(out_of_bounds="clip")
final_iso.fit(oof_raw, oof_y)
oof_iso = final_iso.predict(oof_raw)

# Find locked threshold: Max F2 on unique isotonic outputs (same as threshold_optimization.py)
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

print(f"\nLocked threshold (Max F2 on unique isotonic outputs): {LOCKED_THRESHOLD:.6f}")

# Apply locked threshold
oof_pred = (oof_iso >= LOCKED_THRESHOLD).astype(int)

# Confusion matrix
tp = ((oof_pred == 1) & (oof_y == 1)).sum()
fn = ((oof_pred == 0) & (oof_y == 1)).sum()
fp = ((oof_pred == 1) & (oof_y == 0)).sum()
tn = ((oof_pred == 0) & (oof_y == 0)).sum()

print(f"\nConfusion Matrix (threshold={LOCKED_THRESHOLD:.6f}):")
print(f"  TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"  Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}")
print(f"  F1: {fbeta_score(oof_y, oof_pred, beta=1, zero_division=0):.4f}, "
      f"F2: {fbeta_score(oof_y, oof_pred, beta=2, zero_division=0):.4f}")

# Create analysis dataframe with original indices
all_val_indices = []
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
    all_val_indices.append(X.index[val_idx])
oof_indices = np.concatenate(all_val_indices)

analysis_df = pd.DataFrame({
    "orig_idx": oof_indices,
    "y_true": oof_y,
    "y_pred": oof_pred,
    "raw_proba": oof_raw,
    "iso_proba": oof_iso,
    "fold": np.repeat(np.arange(1, N_FOLDS+1), [len(x) for x in all_y_val]),
})

# Add original features from train
analysis_df = analysis_df.join(train.loc[analysis_df["orig_idx"]].reset_index(drop=True))

# Get original amounts from creditcard.csv (reverse transform) - BEFORE creating caught/missed
original_df = pd.read_csv("creditcard.csv")
original_df = original_df.drop_duplicates().reset_index(drop=True)
original_df = original_df.drop(columns=["Time"])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(np.log1p(original_df[["Amount"]]))

# Reverse transform for analysis
analysis_df["amount_original"] = scaler.inverse_transform(analysis_df[["Amount"]]).flatten()
analysis_df["amount_log"] = np.log1p(analysis_df["amount_original"])

# ============================================================
# 1. MISSED FRAUDS: Near-misses vs Invisible
# ============================================================
print("\n" + "=" * 70)
print("1. MISSED FRAUDS: Near-misses vs Invisible")
print("=" * 70)

missed = analysis_df[(analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 0)]
caught = analysis_df[(analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 1)]

print(f"Total frauds: {len(analysis_df[analysis_df['y_true']==1])}")
print(f"Caught (TP): {len(caught)}")
print(f"Missed (FN): {len(missed)}")

print(f"\nMissed frauds calibrated probability distribution:")
print(f"  Min: {missed['iso_proba'].min():.6f}")
print(f"  Median: {missed['iso_proba'].median():.6f}")
print(f"  Mean: {missed['iso_proba'].mean():.6f}")
print(f"  Max: {missed['iso_proba'].max():.6f}")
print(f"  Std: {missed['iso_proba'].std():.6f}")

print(f"\nCaught frauds calibrated probability distribution:")
print(f"  Min: {caught['iso_proba'].min():.6f}")
print(f"  Median: {caught['iso_proba'].median():.6f}")
print(f"  Mean: {caught['iso_proba'].mean():.6f}")
print(f"  Max: {caught['iso_proba'].max():.6f}")
print(f"  Std: {caught['iso_proba'].std():.6f}")

near_miss_low = LOCKED_THRESHOLD - 0.2
near_miss_high = LOCKED_THRESHOLD
near_misses = missed[(missed["iso_proba"] >= near_miss_low) & (missed["iso_proba"] < near_miss_high)]
invisible = missed[missed["iso_proba"] < near_miss_low]

print(f"\nNear-misses ({near_miss_low:.3f} <= p < {near_miss_high:.3f}): {len(near_misses)} ({len(near_misses)/len(missed)*100:.1f}%)")
print(f"Invisible (p < {near_miss_low:.3f}): {len(invisible)} ({len(invisible)/len(missed)*100:.1f}%)")

bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
missed_hist, _ = np.histogram(missed["iso_proba"], bins=bins)
print(f"\nMissed frauds by probability bin:")
for i in range(len(bins)-1):
    print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {missed_hist[i]}")

# ============================================================
# 2. TRANSACTION AMOUNT
# ============================================================
print("\n" + "=" * 70)
print("2. TRANSACTION AMOUNT ANALYSIS")
print("=" * 70)

print("\nAmount statistics (original scale):")
for group_name, group_df in [("Caught Fraud (TP)", caught), 
                              ("Missed Fraud (FN)", missed), 
                              ("Normal (TN)", analysis_df[(analysis_df["y_true"]==0) & (analysis_df["y_pred"]==0)]),
                              ("False Alarm (FP)", analysis_df[(analysis_df["y_true"]==0) & (analysis_df["y_pred"]==1)])]:
    amt = group_df["amount_original"]
    print(f"\n  {group_name} (n={len(group_df)}):")
    print(f"    Median: {amt.median():.2f}")
    print(f"    Mean: {amt.mean():.2f}")
    print(f"    Q25: {amt.quantile(0.25):.2f}, Q75: {amt.quantile(0.75):.2f}")
    print(f"    Min: {amt.min():.2f}, Max: {amt.max():.2f}")

# ============================================================
# 3. VALUE-WEIGHTED RECALL
# ============================================================
print("\n" + "=" * 70)
print("3. VALUE-WEIGHTED RECALL")
print("=" * 70)

total_fraud_amount = analysis_df[analysis_df["y_true"] == 1]["amount_original"].sum()
caught_fraud_amount = caught["amount_original"].sum()
missed_fraud_amount = missed["amount_original"].sum()

print(f"Total fraud amount: ${total_fraud_amount:,.2f}")
print(f"Caught fraud amount: ${caught_fraud_amount:,.2f} ({caught_fraud_amount/total_fraud_amount*100:.2f}%)")
print(f"Missed fraud amount: ${missed_fraud_amount:,.2f} ({missed_fraud_amount/total_fraud_amount*100:.2f}%)")
print(f"\nCount-weighted Recall: {len(caught)/(len(caught)+len(missed))*100:.2f}%")
print(f"Amount-weighted Recall: {caught_fraud_amount/total_fraud_amount*100:.2f}%")

amount_bins = [0, 10, 50, 100, 500, 1000, 5000, np.inf]
labels = ["0-10", "10-50", "50-100", "100-500", "500-1000", "1000-5000", "5000+"]
analysis_df["amount_bin"] = pd.cut(analysis_df["amount_original"], bins=amount_bins, labels=labels)

print("\nRecall by amount bin:")
for label in labels:
    bin_df = analysis_df[analysis_df["amount_bin"] == label]
    if len(bin_df) == 0:
        continue
    frauds = bin_df[bin_df["y_true"] == 1]
    caught_bin = frauds[frauds["y_pred"] == 1]
    if len(frauds) > 0:
        print(f"  {label}: {len(caught_bin)}/{len(frauds)} = {len(caught_bin)/len(frauds)*100:.1f}% | "
              f"Amount share: {caught_bin['amount_original'].sum()/frauds['amount_original'].sum()*100:.1f}%")

# ============================================================
# 4. V FEATURES
# ============================================================
print("\n" + "=" * 70)
print("4. V FEATURES ANALYSIS")
print("=" * 70)

rf_full = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_full.fit(X, y)
importances = pd.Series(rf_full.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(10).index.tolist()

print("Top 10 features by importance:")
for i, feat in enumerate(top_features, 1):
    print(f"  {i}. {feat}: {importances[feat]:.6f}")

print("\nMean feature values by group:")
for feat in top_features[:6]:
    c_mean = caught[feat].mean()
    m_mean = missed[feat].mean()
    n_mean = analysis_df[analysis_df["y_true"]==0][feat].mean()
    print(f"\n  {feat}:")
    print(f"    Caught: {c_mean:.4f}, Missed: {m_mean:.4f}, Normal: {n_mean:.4f}")
    print(f"    Diff (Missed-Caught): {m_mean - c_mean:.4f}")
    print(f"    Diff (Missed-Normal): {m_mean - n_mean:.4f}")

# ============================================================
# 5. FALSE ALARMS
# ============================================================
print("\n" + "=" * 70)
print("5. FALSE ALARMS ANALYSIS")
print("=" * 70)

false_alarms = analysis_df[(analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 1)]
print(f"Total false alarms: {len(false_alarms)}")

print(f"\nFalse alarm calibrated probabilities:")
print(f"  Min: {false_alarms['iso_proba'].min():.6f}")
print(f"  Median: {false_alarms['iso_proba'].median():.6f}")
print(f"  Mean: {false_alarms['iso_proba'].mean():.6f}")
print(f"  Max: {false_alarms['iso_proba'].max():.6f}")

high_conf_fp = false_alarms[false_alarms["iso_proba"] >= 0.8]
print(f"\nHigh confidence false alarms (p >= 0.8): {len(high_conf_fp)}")
if len(high_conf_fp) > 0:
    print(f"  Amounts: median=${high_conf_fp['amount_original'].median():.2f}, "
          f"mean=${high_conf_fp['amount_original'].mean():.2f}")

print(f"\nAmount comparison:")
print(f"  Caught fraud median: ${caught['amount_original'].median():.2f}")
print(f"  False alarm median: ${false_alarms['amount_original'].median():.2f}")
print(f"  Normal median: ${analysis_df[(analysis_df['y_true']==0) & (analysis_df['y_pred']==0)]['amount_original'].median():.2f}")

# ============================================================
# 6. SPREAD ACROSS FOLDS
# ============================================================
print("\n" + "=" * 70)
print("6. ERROR SPREAD ACROSS FOLDS")
print("=" * 70)

for fold in range(1, N_FOLDS+1):
    fold_df = analysis_df[analysis_df["fold"] == fold]
    tp_f = ((fold_df["y_pred"] == 1) & (fold_df["y_true"] == 1)).sum()
    fn_f = ((fold_df["y_pred"] == 0) & (fold_df["y_true"] == 1)).sum()
    fp_f = ((fold_df["y_pred"] == 1) & (fold_df["y_true"] == 0)).sum()
    tn_f = ((fold_df["y_pred"] == 0) & (fold_df["y_true"] == 0)).sum()
    fraud_f = (fold_df["y_true"] == 1).sum()
    normal_f = (fold_df["y_true"] == 0).sum()
    
    print(f"\nFold {fold} (fraud={fraud_f}, normal={normal_f}):")
    print(f"  TP: {tp_f}, FN: {fn_f}, FP: {fp_f}, TN: {tn_f}")
    print(f"  Recall: {tp_f/(tp_f+fn_f)*100:.1f}%, Precision: {tp_f/(tp_f+fp_f)*100:.1f}%")
    print(f"  FP rate: {fp_f/normal_f*100:.3f}%")

# Overall summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Locked threshold: {LOCKED_THRESHOLD:.6f}")
print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}")
print(f"Amount-weighted Recall: {caught_fraud_amount/total_fraud_amount*100:.2f}%")
print(f"Near-misses (p in [{near_miss_low:.2f}, {near_miss_high:.2f})): {len(near_misses)} ({len(near_misses)/len(missed)*100:.1f}%)")
print(f"Invisible (p < {near_miss_low:.2f}): {len(invisible)} ({len(invisible)/len(missed)*100:.1f}%)")
print(f"High-conf false alarms (p>=0.8): {len(high_conf_fp)}")

print("\nError analysis complete (OOF only, Test untouched).")