import pandas as pd
import numpy as np
import json
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from pipeline import prepare_data

RANDOM_STATE = 42
N_FOLDS = 5
RESULTS_FILE = "eda_outputs/stage2_results.json"

warnings.filterwarnings("ignore")

# Top 3 from Stage 1 + default
TOP_CONFIGS = [
    # Trial 8 (rank 1)
    {
        "name": "Trial_1_best",
        "min_samples_leaf": 2,
        "n_estimators": 271,
        "max_features": 0.2567800847856748,
        "max_samples": 0.7365894999736293,
        "max_depth": 28,
    },
    # Trial 15 (rank 2)
    {
        "name": "Trial_2_best",
        "min_samples_leaf": 1,
        "n_estimators": 264,
        "max_features": 0.26230829112845244,
        "max_samples": 0.9784287637010953,
        "max_depth": None,
    },
    # Trial 7 (rank 3)
    {
        "name": "Trial_3_best",
        "min_samples_leaf": 3,
        "n_estimators": 200,
        "max_features": 0.305520218021604,
        "max_samples": 0.7861443328051817,
        "max_depth": 28,
    },
    # Default
    {
        "name": "Default_RF",
        "min_samples_leaf": 1,
        "n_estimators": 100,
        "max_features": "sqrt",
        "max_samples": None,
        "max_depth": None,
    },
]

print("=" * 70)
print("STAGE 2: Evaluate Top 3 + Default on Full Data (5 Outer Folds)")
print("=" * 70)

df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train set: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Fraud rate: {y.sum()/len(y)*100:.4f}%")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_results = {}

for config in TOP_CONFIGS:
    name = config.pop("name")
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}")
    print(f"{'='*70}")
    print(f"Params: {config}")
    
    # Base params
    base_params = {
        "criterion": "gini",
        "bootstrap": True,
        "class_weight": None,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    base_params.update(config)
    
    fold_results = []
    all_raw_proba = []
    all_y_val = []
    fold_times = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        print(f"  Train: {len(X_train)} | Val: {len(X_val)} (fraud: {y_val.sum()})")
        
        # Train RF on full outer training set (NO subsampling)
        rf = RandomForestClassifier(**base_params)
        start = time.perf_counter()
        rf.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        fold_times.append(elapsed)
        
        # Raw predictions on outer validation
        raw_proba = rf.predict_proba(X_val)[:, 1]
        all_raw_proba.append(raw_proba)
        all_y_val.append(y_val)
        
        pr_auc = average_precision_score(y_val, raw_proba)
        brier = brier_score_loss(y_val, raw_proba)
        
        fold_results.append({
            "fold": fold,
            "pr_auc": float(pr_auc),
            "brier": float(brier),
            "time": float(elapsed),
        })
        
        print(f"  PR-AUC: {pr_auc:.4f} | Brier: {brier:.6f} | Time: {elapsed:.1f}s")
    
    # Aggregate
    all_raw = np.concatenate(all_raw_proba)
    all_y = np.concatenate(all_y_val)
    
    # Overall PR-AUC and Brier on aggregated outer validation
    overall_pr_auc = average_precision_score(all_y, all_raw)
    overall_brier = brier_score_loss(all_y, all_raw)
    
    # Calibration: fit isotonic on OOF predictions
    print(f"\n  Fitting Isotonic calibrator on {len(all_raw)} OOF samples...")
    isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    isotonic_cal.fit(all_raw, all_y)
    iso_proba = isotonic_cal.predict(all_raw)
    
    iso_pr_auc = average_precision_score(all_y, iso_proba)
    iso_brier = brier_score_loss(all_y, iso_proba)
    
    mean_pr_auc = np.mean([r["pr_auc"] for r in fold_results])
    std_pr_auc = np.std([r["pr_auc"] for r in fold_results])
    mean_brier = np.mean([r["brier"] for r in fold_results])
    std_brier = np.std([r["brier"] for r in fold_results])
    mean_time = np.mean(fold_times)
    
    result = {
        "name": name,
        "params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in config.items()},
        "folds": fold_results,
        "mean_pr_auc": float(mean_pr_auc),
        "std_pr_auc": float(std_pr_auc),
        "mean_brier": float(mean_brier),
        "std_brier": float(std_brier),
        "overall_pr_auc": float(overall_pr_auc),
        "overall_brier": float(overall_brier),
        "isotonic_pr_auc": float(iso_pr_auc),
        "isotonic_brier": float(iso_brier),
        "mean_time": float(np.mean(fold_times)),
    }
    
    all_results[name] = result
    
    print(f"\n  Mean PR-AUC (per-fold): {mean_pr_auc:.4f} ± {std_pr_auc:.4f}")
    print(f"  Overall PR-AUC (aggregated): {overall_pr_auc:.4f}")
    print(f"  Mean Brier (per-fold): {mean_brier:.6f} ± {std_brier:.6f}")
    print(f"  Overall Brier: {overall_brier:.6f}")
    print(f"  Isotonic PR-AUC: {iso_pr_auc:.4f}")
    print(f"  Isotonic Brier:  {iso_brier:.6f}")
    print(f"  Mean time: {np.mean(fold_times):.1f}s")
    
    # Save OOF predictions for calibration plots
    np.save(f"eda_outputs/stage2_{name}_raw_proba.npy", all_raw)
    np.save(f"eda_outputs/stage2_{name}_iso_proba.npy", iso_proba)
    np.save(f"eda_outputs/stage2_{name}_y_val.npy", all_y)

# Save results
with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 70)
print("STAGE 2 COMPLETE - SUMMARY")
print("=" * 70)
print(f"\n{'Config':<20} {'Mean PR-AUC':>12} {'Std PR-AUC':>12} {'Overall PR':>12} {'Mean Brier':>12} {'Iso Brier':>12}")
print("-" * 90)
for name, r in all_results.items():
    print(f"{name:<20} {r['mean_pr_auc']:>12.4f} {r['std_pr_auc']:>12.4f} {r['overall_pr_auc']:>12.4f} {r['mean_brier']:>12.6f} {r['isotonic_brier']:>12.6f}")

# Compare with default
default_pr = all_results["Default_RF"]["mean_pr_auc"]
print(f"\nDefault RF PR-AUC: {default_pr:.4f}")
for name in ["Trial_1_best", "Trial_2_best", "Trial_3_best"]:
    diff = all_results[name]["mean_pr_auc"] - default_pr
    print(f"  {name}: {all_results[name]['mean_pr_auc']:.4f} (diff: {diff:+.4f})")

print("\nStage 2 complete. Results saved to eda_outputs/stage2_results.json")