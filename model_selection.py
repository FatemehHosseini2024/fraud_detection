import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

from pipeline import prepare_data

RANDOM_STATE = 42

print("=" * 70)
print("BASELINE MODEL SELECTION")
print("=" * 70)

# Load preprocessed data and folds (same folds used for all models)
_, train, _, folds = prepare_data()
print(f"Train set: {train.shape[0]} rows")
print(f"Fraud rate: {train['Class'].sum()/len(train)*100:.4f}%")
print(f"Number of CV folds: {len(folds)}")

# Define models with default parameters
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=1),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
}

# Collect results: {model_name: {metric: [fold1, fold2, ..., fold5]}}
results = defaultdict(lambda: defaultdict(list))

for fold_idx, (fold_train, fold_val) in enumerate(folds, 1):
    X_train_fold = fold_train.drop(columns=["Class"])
    y_train_fold = fold_train["Class"]
    X_val_fold = fold_val.drop(columns=["Class"])
    y_val_fold = fold_val["Class"]

    print(f"\n--- Fold {fold_idx}/{len(folds)} ---")

    for name, model in models.items():
        model.fit(X_train_fold, y_train_fold)
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        pr_auc = average_precision_score(y_val_fold, y_proba)
        precision = precision_score(y_val_fold, y_pred, zero_division=0)
        recall = recall_score(y_val_fold, y_pred, zero_division=0)

        results[name]["PR-AUC"].append(pr_auc)
        results[name]["Precision"].append(precision)
        results[name]["Recall"].append(recall)

        print(f"  {name:20s} | PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

print("\n" + "=" * 70)
print("RESULTS: Mean ± Std across 5 folds")
print("=" * 70)
print(f"\n{'Model':<22} {'Metric':<12} {'Mean':>8} {'Std':>8}")
print("-" * 52)
for name in models:
    for metric in ["PR-AUC", "Precision", "Recall"]:
        vals = results[name][metric]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"{name:<22} {metric:<12} {mean_val:8.4f} {std_val:8.4f}")
    print("-" * 52)

print("\nModel selection complete.")
