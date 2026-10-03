import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

from pipeline import prepare_data

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5
N_BINS = 10

print("=" * 70)
print("NESTED CV RELIABILITY DIAGRAMS")
print("=" * 70)

df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train set: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Fraud rate: {y.sum()/len(y)*100:.4f}%")
print(f"Structure: {N_OUTER} outer x {N_INNER} inner folds")

outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

all_raw_proba = []
all_sigmoid_proba = []
all_isotonic_proba = []
all_y_val = []

for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Outer Fold {outer_fold}/{N_OUTER} ---")
    
    X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
    y_outer_train = y.iloc[outer_train_idx].reset_index(drop=True)
    X_outer_val = X.iloc[outer_val_idx]
    y_outer_val = y.iloc[outer_val_idx]
    
    print(f"  Outer train: {len(X_outer_train)} | Outer val: {len(X_outer_val)}")
    
    # Inner CV for calibrator training
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    inner_oof_preds = np.zeros(len(y_outer_train))
    
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
        rf_inner = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        rf_inner.fit(X_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_train_idx])
        inner_oof_preds[inner_val_idx] = rf_inner.predict_proba(X_outer_train.iloc[inner_val_idx])[:, 1]
    
    print(f"  Inner OOF collected: {len(inner_oof_preds)} samples")
    
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
    
    all_raw_proba.append(raw_proba)
    all_sigmoid_proba.append(sigmoid_proba)
    all_isotonic_proba.append(isotonic_proba)
    all_y_val.append(y_outer_val)
    
    b_raw = brier_score_loss(y_outer_val, raw_proba)
    b_sig = brier_score_loss(y_outer_val, sigmoid_proba)
    b_iso = brier_score_loss(y_outer_val, isotonic_proba)
    print(f"  Brier raw: {b_raw:.6f} | sigmoid: {b_sig:.6f} | isotonic: {b_iso:.6f}")

all_raw = np.concatenate(all_raw_proba)
all_sig = np.concatenate(all_sigmoid_proba)
all_iso = np.concatenate(all_isotonic_proba)
all_y = np.concatenate(all_y_val)

print(f"\nAggregated outer validation: {len(all_y)} samples, {all_y.sum()} frauds")
print(f"Overall Brier: Raw={brier_score_loss(all_y, all_raw):.6f}, "
      f"Sigmoid={brier_score_loss(all_y, all_sig):.6f}, "
      f"Isotonic={brier_score_loss(all_y, all_iso):.6f}")

# Quantile binning on aggregated raw predictions
print(f"\nCreating {N_BINS} quantile bins...")
sorted_idx = np.argsort(all_raw)
bin_indices = np.zeros(len(all_raw), dtype=int)
bin_size = len(all_raw) // N_BINS
for i in range(N_BINS):
    start = i * bin_size
    end = (i + 1) * bin_size if i < N_BINS - 1 else len(all_raw)
    bin_indices[sorted_idx[start:end]] = i

curves = {"Raw RF": all_raw, "Sigmoid": all_sig, "Isotonic": all_iso}
colors = {"Raw RF": "steelblue", "Sigmoid": "darkorange", "Isotonic": "green"}
markers = {"Raw RF": "o", "Sigmoid": "s", "Isotonic": "D"}

# FIGURE 1: Zoomed (0-0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.5)
for curve_name, proba in curves.items():
    mean_pred, frac_pos, n_samples, n_frauds = [], [], [], []
    for i in range(N_BINS):
        mask = (bin_indices == i)
        if mask.sum() == 0: continue
        mean_p = proba[mask].mean()
        frac = all_y[mask].mean()
        n_s = mask.sum()
        n_f = all_y[mask].sum()
        mean_pred.append(mean_p)
        frac_pos.append(frac)
        n_samples.append(n_s)
        n_frauds.append(n_f)
    mean_pred = np.array(mean_pred)
    frac_pos = np.array(frac_pos)
    ax1.plot(mean_pred, frac_pos, marker=markers[curve_name], color=colors[curve_name],
             linewidth=2, markersize=7, label=curve_name)
    for j in range(len(mean_pred)):
        ax1.annotate(f"{n_frauds[j]}/{n_samples[j]}", (mean_pred[j], frac_pos[j]),
                     textcoords="offset points", xytext=(8, 5), fontsize=7,
                     color=colors[curve_name], alpha=0.8)
ax1.set_xlabel("Mean predicted probability", fontsize=12)
ax1.set_ylabel("Fraction of actual positives", fontsize=12)
ax1.set_title("Nested CV Reliability Diagram: Zoomed (0–0.1)\n"
              "Calibrators from inner OOF, evaluated on outer validation", fontsize=13)
ax1.set_xlim(-0.005, 0.12); ax1.set_ylim(0, 0.12)
ax1.legend(loc="upper left", fontsize=10); ax1.grid(True, alpha=0.3)

# FIGURE 2: Calibration Error
fig2, ax2 = plt.subplots(figsize=(10, 6))
for curve_name, proba in curves.items():
    mean_pred, diff = [], []
    for i in range(N_BINS):
        mask = (bin_indices == i)
        if mask.sum() == 0: continue
        mean_p = proba[mask].mean()
        frac = all_y[mask].mean()
        mean_pred.append(mean_p); diff.append(mean_p - frac)
    mean_pred = np.array(mean_pred); diff = np.array(diff)
    ax2.plot(mean_pred, diff, marker=markers[curve_name], color=colors[curve_name],
             linewidth=2, markersize=7, label=curve_name)
    for j in range(len(mean_pred)):
        ax2.annotate(f"{diff[j]:+.4f}", (mean_pred[j], diff[j]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7,
                     color=colors[curve_name], alpha=0.8)
ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel("Mean predicted probability", fontsize=12)
ax2.set_ylabel("Predicted − Actual (Calibration Error)", fontsize=12)
ax2.set_title("Nested CV Calibration Error\nPositive=Overconfident, Negative=Underconfident", fontsize=13)
ax2.set_xlim(-0.005, 0.12); ax2.set_ylim(-0.05, 0.05)
ax2.legend(loc="upper left", fontsize=10); ax2.grid(True, alpha=0.3)

# FIGURE 3: Full symlog
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.5)
for curve_name, proba in curves.items():
    mean_pred, frac_pos = [], []
    for i in range(N_BINS):
        mask = (bin_indices == i)
        if mask.sum() == 0: continue
        mean_pred.append(proba[mask].mean()); frac_pos.append(all_y[mask].mean())
    ax3.plot(np.array(mean_pred), np.array(frac_pos),
             marker=markers[curve_name], color=colors[curve_name],
             linewidth=2, markersize=7, label=curve_name)
ax3.set_xlabel("Mean predicted probability (symlog)", fontsize=12)
ax3.set_ylabel("Fraction of actual positives", fontsize=12)
ax3.set_title("Nested CV Reliability: Full Range (symlog)", fontsize=13)
ax3.set_xscale("symlog", linthresh=0.01); ax3.set_xlim(-0.01, 0.5); ax3.set_ylim(0, 0.5)
ax3.legend(loc="upper left", fontsize=10); ax3.grid(True, alpha=0.3)

# Save
import os
os.makedirs("eda_outputs", exist_ok=True)

fig1.tight_layout()
fig1.savefig("eda_outputs/reliability_diagram_nested_zoomed.png", dpi=150)
plt.close(fig1)

fig2.tight_layout()
fig2.savefig("eda_outputs/calibration_error_nested.png", dpi=150)
plt.close(fig2)

fig3.tight_layout()
fig3.savefig("eda_outputs/reliability_diagram_nested_full.png", dpi=150)
plt.close(fig3)

print("\nSaved:")
print("  eda_outputs/reliability_diagram_nested_zoomed.png")
print("  eda_outputs/calibration_error_nested.png")
print("  eda_outputs/reliability_diagram_nested_full.png")
print("\nNested CV reliability diagrams complete.")