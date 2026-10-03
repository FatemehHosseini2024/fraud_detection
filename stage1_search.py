import pandas as pd
import numpy as np
import json
import time
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from pipeline import prepare_data

RANDOM_STATE = 42
N_TRIALS = 30
N_FOLDS_STAGE1 = 3
N_FOLDS_STAGE2 = 5
RESULTS_FILE = "eda_outputs/stage1_results.jsonl"

warnings.filterwarnings("ignore")

# Search space
PARAM_RANGES = {
    "min_samples_leaf": (1, 20),          # log scale
    "n_estimators": (100, 300),
    "max_features": (0.1, 0.5),           # fraction of features
    "max_samples": (0.3, 1.0),
    "max_depth": (10, 40),                # None also allowed
}

# Load data
print("=" * 70)
print("STAGE 1: Random Search with Subsampling")
print("=" * 70)

df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train set: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Fraud rate: {y.sum()/len(y)*100:.4f}%")

# Fraud and normal indices
fraud_idx = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]
print(f"Fraud: {len(fraud_idx)}, Normal: {len(normal_idx)}")

# 3-fold stratified CV
skf = StratifiedKFold(n_splits=N_FOLDS_STAGE1, shuffle=True, random_state=RANDOM_STATE)

# Load existing results if any
completed_trials = []
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            completed_trials.append(json.loads(line))
    print(f"Resuming: {len(completed_trials)} trials already completed")

def sample_params(trial_id):
    rng = np.random.default_rng(RANDOM_STATE + trial_id)
    
    # min_samples_leaf: log-uniform
    min_samples_leaf = int(np.exp(rng.uniform(np.log(1), np.log(20))))
    
    # n_estimators: uniform
    n_estimators = rng.integers(100, 301)
    
    # max_features: uniform fraction
    max_features = rng.uniform(0.1, 0.5)
    
    # max_samples: uniform
    max_samples = rng.uniform(0.3, 1.0)
    
    # max_depth: 50% chance None, 50% uniform 10-40
    if rng.random() < 0.5:
        max_depth = None
    else:
        max_depth = rng.integers(10, 41)
    
    return {
        "min_samples_leaf": min_samples_leaf,
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_samples": max_samples,
        "max_depth": max_depth,
        "criterion": "gini",
        "bootstrap": True,
        "class_weight": None,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

# Run trials
for trial in range(len(completed_trials), N_TRIALS):
    print(f"\n{'='*70}")
    print(f"TRIAL {trial+1}/{N_TRIALS}")
    print(f"{'='*70}")
    
    params = sample_params(trial)
    print(f"Params: {params}")
    
    fold_scores = []
    fold_times = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train = X.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y.iloc[train_idx].reset_index(drop=True)
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]
        
        # SUBSAMPLING: keep all frauds, 20% of normals in TRAIN only
        fold_fraud_idx = np.where(y_fold_train == 1)[0]
        fold_normal_idx = np.where(y_fold_train == 0)[0]
        
        n_normal_keep = int(len(fold_normal_idx) * 0.2)
        normal_keep_idx = np.random.default_rng(RANDOM_STATE + trial * 100 + fold).choice(
            fold_normal_idx, size=n_normal_keep, replace=False
        )
        
        subsample_idx = np.concatenate([fold_fraud_idx, normal_keep_idx])
        subsample_idx.sort()
        
        X_sub = X_fold_train.iloc[subsample_idx]
        y_sub = y_fold_train.iloc[subsample_idx]
        
        print(f"  Fold {fold}: train={len(X_sub)} (fraud={y_sub.sum()}, normal={len(y_sub)-y_sub.sum()}), "
              f"val={len(X_fold_val)} (fraud={y_fold_val.sum()})")
        
        # Train
        rf = RandomForestClassifier(**params)
        start = time.perf_counter()
        rf.fit(X_sub, y_sub)
        elapsed = time.perf_counter() - start
        
        # Predict on VALIDATION (full distribution)
        val_proba = rf.predict_proba(X_fold_val)[:, 1]
        pr_auc = average_precision_score(y_fold_val, val_proba)
        
        fold_scores.append(pr_auc)
        fold_times.append(elapsed)
        print(f"    PR-AUC: {pr_auc:.4f}, time: {elapsed:.1f}s")
    
    mean_pr_auc = np.mean(fold_scores)
    std_pr_auc = np.std(fold_scores)
    mean_time = np.mean(fold_times)
    
    result = {
        "trial": trial + 1,
        "params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in params.items()},
        "fold_pr_aucs": [float(x) for x in fold_scores],
        "mean_pr_auc": float(mean_pr_auc),
        "std_pr_auc": float(std_pr_auc),
        "fold_times": [float(x) for x in fold_times],
        "mean_time": float(mean_time),
    }
    
    completed_trials.append(result)
    
    # Save immediately
    with open(RESULTS_FILE, "w") as f:
        for r in completed_trials:
            f.write(json.dumps(r) + "\n")
    
    print(f"  Mean PR-AUC: {mean_pr_auc:.4f} ± {std_pr_auc:.4f}")
    print(f"  Mean time:   {mean_time:.1f}s")

print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)

# Sort by mean PR-AUC descending
completed_trials.sort(key=lambda x: x["mean_pr_auc"], reverse=True)

print(f"\nTop 5 trials:")
for i, r in enumerate(completed_trials[:5]):
    print(f"  {i+1}. PR-AUC: {r['mean_pr_auc']:.4f} ± {r['std_pr_auc']:.4f} | "
          f"time: {r['mean_time']:.1f}s | params: {r['params']}")

# Save final sorted results
with open("eda_outputs/stage1_results_sorted.json", "w") as f:
    json.dump(completed_trials, f, indent=2)

print(f"\nResults saved to {RESULTS_FILE} and eda_outputs/stage1_results_sorted.json")
print("Stage 1 complete. Ready for Stage 2.")