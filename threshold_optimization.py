import pandas as pd
import numpy as np
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

from pipeline import prepare_data
from mlflow_helpers import start_run, log_json_artifact
import mlflow

RANDOM_STATE = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000
PRECISION_FLOORS = [0.80, 0.90, 0.95]

warnings.filterwarnings("ignore")

print("=" * 70)
print("THRESHOLD OPTIMIZATION: Default RF + Isotonic (OOF only)")
print("=" * 70)

run = start_run(
    "threshold_optimization",
    tags={"stage": "threshold", "data": "train OOF only", "test": "not evaluated"},
)

# 1. Load data and get OOF predictions
df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train: {X.shape[0]} rows, Fraud: {y.sum()} ({y.sum()/len(y)*100:.4f}%)")

# Nested CV to get OOF predictions
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_raw_proba = []
all_y_val = []

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
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

# Fit isotonic on ALL OOF
print("\nFitting final isotonic calibrator on all OOF...")
final_iso = IsotonicRegression(out_of_bounds="clip")
final_iso.fit(oof_raw, oof_y)

# Calibrated OOF probabilities
oof_iso = final_iso.predict(oof_raw)

# Step 2: Candidate thresholds = unique isotonic output values
unique_iso = np.unique(oof_iso)
unique_iso = unique_iso[unique_iso > 0]
candidate_thresholds = unique_iso
print(f"\nCandidate thresholds (unique isotonic outputs): {len(candidate_thresholds)}")
print(f"Range: [{candidate_thresholds.min():.6f}, {candidate_thresholds.max():.6f}]")

# Step 3: Compute Precision, Recall, F1, F2 at each threshold
print("\nComputing metrics at each threshold...")
results = []

for thresh in candidate_thresholds:
    y_pred = (oof_iso >= thresh).astype(int)
    
    prec = precision_score(oof_y, y_pred, zero_division=0)
    rec = recall_score(oof_y, y_pred, zero_division=0)
    f1 = fbeta_score(oof_y, y_pred, beta=1, zero_division=0)
    f2 = fbeta_score(oof_y, y_pred, beta=2, zero_division=0)
    
    results.append({
        "threshold": float(thresh),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f2": float(f2),
        "n_pred_pos": int(y_pred.sum()),
        "n_true_pos": int((y_pred & oof_y).sum()),
    })

results_df = pd.DataFrame(results).sort_values("threshold").reset_index(drop=True)

# Step 4: Apply criteria
print("\n--- Criterion 1: Max F2 ---")
best_f2_idx = results_df["f2"].idxmax()
best_f2 = results_df.loc[best_f2_idx]
print(f"Threshold: {best_f2['threshold']:.6f}")
print(f"Precision: {best_f2['precision']:.4f}, Recall: {best_f2['recall']:.4f}")
print(f"F1: {best_f2['f1']:.4f}, F2: {best_f2['f2']:.4f}")

print("\n--- Criterion 2: Max Recall with Precision floor ---")
floor_results = {}
for floor in PRECISION_FLOORS:
    valid = results_df[results_df["precision"] >= floor]
    if len(valid) > 0:
        best = valid.loc[valid["recall"].idxmax()]
        floor_results[floor] = best
        print(f"\nFloor {floor:.2f}:")
        print(f"  Threshold: {best['threshold']:.6f}")
        print(f"  Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f}")
        print(f"  F1: {best['f1']:.4f}, F2: {best['f2']:.4f}")
    else:
        floor_results[floor] = None
        print(f"\nFloor {floor:.2f}: No threshold meets this floor")

# Step 5: Bootstrap stability
print(f"\n--- Bootstrap Stability (N={N_BOOTSTRAP}) ---")
rng = np.random.default_rng(RANDOM_STATE)
n_samples = len(oof_y)

bootstrap_stats = {c: [] for c in ["f2"] + [f"floor_{f:.2f}" for f in PRECISION_FLOORS]}

for b in range(N_BOOTSTRAP):
    idx = rng.choice(n_samples, size=n_samples, replace=True)
    y_b = oof_y[idx]
    iso_b = oof_iso[idx]
    
    uniq = np.unique(iso_b[iso_b > 0])
    if len(uniq) < 5:
        continue
    
    precs = []
    recs = []
    f2s = []
    for thresh in uniq:
        y_pred_b = (iso_b >= thresh).astype(int)
        p = precision_score(y_b, y_pred_b, zero_division=0)
        r = recall_score(y_b, y_pred_b, zero_division=0)
        f2 = fbeta_score(y_b, y_pred_b, beta=2, zero_division=0)
        precs.append(p)
        recs.append(r)
        f2s.append(f2)
    
    precs = np.array(precs)
    recs = np.array(recs)
    f2s = np.array(f2s)
    
    if len(f2s) > 0:
        bootstrap_stats["f2"].append(uniq[f2s.argmax()])
    
    for floor in PRECISION_FLOORS:
        valid_mask = precs >= floor
        if valid_mask.any():
            best_rec_idx = recs[valid_mask].argmax()
            bootstrap_stats[f"floor_{floor:.2f}"].append(uniq[valid_mask][best_rec_idx])

print("\nBootstrap threshold distribution (median, 2.5%, 97.5%):")
for key, vals in bootstrap_stats.items():
    if len(vals) > 0:
        vals = np.array(vals)
        median = np.median(vals)
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        print(f"  {key}: median={median:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={len(vals)}")
    else:
        print(f"  {key}: no valid bootstrap samples")

# Step 6: Comparison table
print("\n" + "=" * 90)
print("COMPARISON TABLE: All Criteria on OOF")
print("=" * 90)
print(f"\n{'Criterion':<20} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'F2':>10}")
print("-" * 70)

best_f1_idx = results_df["f1"].idxmax()
best_f1 = results_df.loc[best_f1_idx]
print(f"{'Max F1':<20} {best_f1['threshold']:>10.4f} {best_f1['precision']:>10.4f} {best_f1['recall']:>10.4f} {best_f1['f1']:>10.4f} {best_f1['f2']:>10.4f}")

print(f"{'Max F2':<20} {best_f2['threshold']:>10.4f} {best_f2['precision']:>10.4f} {best_f2['recall']:>10.4f} {best_f2['f1']:>10.4f} {best_f2['f2']:>10.4f}")

for floor in PRECISION_FLOORS:
    if floor_results[floor] is not None:
        r = floor_results[floor]
        print(f"{f'Max Recall (P>={floor:.2f})':<20} {r['threshold']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['f2']:>10.4f}")
    else:
        print(f"{f'Max Recall (P>={floor:.2f})':<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# Step 7: Lock threshold - choose Max F2 as primary
locked_threshold = best_f2["threshold"]
locked_prec = best_f2["precision"]
locked_rec = best_f2["recall"]
locked_f1 = best_f2["f1"]
locked_f2 = best_f2["f2"]

print(f"\n{'='*70}")
print(f"LOCKED THRESHOLD (Max F2): {locked_threshold:.6f}")
print(f"  OOF Precision: {locked_prec:.4f}, Recall: {locked_rec:.4f}, F1: {locked_f1:.4f}, F2: {locked_f2:.4f}")
print(f"{'='*70}")

# Save results
output = {
    "locked_threshold": float(locked_threshold),
    "locked_criterion": "max_f2",
    "oof_metrics": {
        "precision": float(locked_prec),
        "recall": float(locked_rec),
        "f1": float(locked_f1),
        "f2": float(locked_f2),
    },
    "test_bootstrap_ci": {k: {"median": float(np.median(v)), "ci_95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} 
                          for k, v in bootstrap_stats.items() if len(v) > 0},
    "candidate_thresholds_count": len(candidate_thresholds),
    "all_criteria": {
        "max_f1": {k: float(v) for k, v in best_f1.items()},
        "max_f2": {k: float(v) for k, v in best_f2.items()},
        "precision_floors": {str(k): {kk: float(vv) for kk, vv in v.items()} if v is not None else None
                            for k, v in floor_results.items()},
    },
    "threshold_grid": results_df.to_dict(orient="records"),
}

# --- MLflow: log params, metrics and the threshold artifact ---
mlflow.log_param("random_state", RANDOM_STATE)
mlflow.log_param("n_folds", N_FOLDS)
mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
mlflow.log_param("precision_floors", PRECISION_FLOORS)
mlflow.log_param("estimator", "RandomForestClassifier(random_state=42, n_jobs=-1)")
mlflow.log_param("calibrator", "IsotonicRegression(out_of_bounds='clip')")
mlflow.log_param("criterion", "max_f2")

mlflow.log_metric("locked_threshold", float(locked_threshold))
mlflow.log_metric("oof_precision", float(locked_prec))
mlflow.log_metric("oof_recall", float(locked_rec))
mlflow.log_metric("oof_f1", float(locked_f1))
mlflow.log_metric("oof_f2", float(locked_f2))
mlflow.log_metric("oof_pr_auc", float(average_precision_score(oof_y, oof_iso)))
mlflow.log_metric("oof_brier", float(brier_score_loss(oof_y, oof_iso)))
mlflow.log_metric("candidate_thresholds_count", len(candidate_thresholds))
mlflow.log_metric("max_f1_threshold", float(best_f1["threshold"]))
mlflow.log_metric("max_f1_f1", float(best_f1["f1"]))

for floor in PRECISION_FLOORS:
    if floor_results[floor] is not None:
        mlflow.log_metric(f"floor_{floor:.2f}_threshold", float(floor_results[floor]["threshold"]))
        mlflow.log_metric(f"floor_{floor:.2f}_precision", float(floor_results[floor]["precision"]))
        mlflow.log_metric(f"floor_{floor:.2f}_recall", float(floor_results[floor]["recall"]))
        mlflow.log_metric(f"floor_{floor:.2f}_f2", float(floor_results[floor]["f2"]))

for key, vals in bootstrap_stats.items():
    if len(vals) > 0:
        vals = np.array(vals)
        mlflow.log_metric(f"bootstrap_{key}_median", float(np.median(vals)))
        mlflow.log_metric(f"bootstrap_{key}_ci_low", float(np.percentile(vals, 2.5)))
        mlflow.log_metric(f"bootstrap_{key}_ci_high", float(np.percentile(vals, 97.5)))
        mlflow.log_metric(f"bootstrap_{key}_n", len(vals))

log_json_artifact(output, "threshold_optimization.json")
mlflow.log_artifact("eda_outputs/threshold_optimization.json", artifact_path="metrics")

import os
os.makedirs("eda_outputs", exist_ok=True)
with open("eda_outputs/threshold_optimization.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to eda_outputs/threshold_optimization.json")
print("\nThreshold optimization complete (OOF only, Test not evaluated).")

mlflow.end_run()