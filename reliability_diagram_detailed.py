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
N_FOLDS = 5
N_BINS = 10

print("=" * 70)
print("ZOOMED RELIABILITY DIAGRAM + CALIBRATION ERROR PLOT")
print("=" * 70)

# Load data and get OOF predictions
df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Train set: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Fraud rate: {y.sum()/len(y)*100:.4f}%")

# Step 1: Get OOF predictions via 5-fold stratified CV
print("\nCollecting OOF predictions (5-fold CV)...")
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(y))

for fold_num, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"  Fold {fold_num}...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_preds[val_idx] = rf.predict_proba(X.iloc[val_idx])[:, 1]

print(f"OOF predictions collected: {len(oof_preds)} samples")
print(f"Fraud in OOF: {y.sum()} ({y.sum()/len(y)*100:.4f}%)")
print(f"OOF prob range: [{oof_preds.min():.6f}, {oof_preds.max():.6f}]")

# Step 2: Fit calibrators on OOF predictions
print("\nFitting calibrators on OOF predictions...")
sigmoid_cal = LogisticRegression(random_state=RANDOM_STATE)
sigmoid_cal.fit(oof_preds.reshape(-1, 1), y)

isotonic_cal = IsotonicRegression(out_of_bounds="clip")
isotonic_cal.fit(oof_preds, y)

# Step 3: Apply calibrators
oob_sigmoid = sigmoid_cal.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
oob_isotonic = isotonic_cal.predict(oof_preds)

# Step 4: Create quantile bins
sorted_idx = np.argsort(oof_preds)
bin_indices = np.zeros(len(oof_preds), dtype=int)
bin_size = len(oof_preds) // N_BINS
for i in range(N_BINS):
    start = i * bin_size
    end = (i + 1) * bin_size if i < N_BINS - 1 else len(oof_preds)
    bin_indices[sorted_idx[start:end]] = i

curves = {
    "Raw RF": oof_preds,
    "Sigmoid": oob_sigmoid,
    "Isotonic": oob_isotonic,
}

# ---- FIGURE 1: Zoomed Reliability Diagram (low prob region) ----
fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.5)

colors = {"Raw RF": "steelblue", "Sigmoid": "darkorange", "Isotonic": "green"}
markers = {"Raw RF": "o", "Sigmoid": "s", "Isotonic": "D"}

for curve_name, proba in curves.items():
    mean_pred = []
    frac_pos = []
    n_samples = []
    n_frauds = []

    for i in range(N_BINS):
        mask = (bin_indices == i)
        if mask.sum() == 0:
            continue
        mean_p = proba[mask].mean()
        frac = y.values[mask].mean()
        n_s = mask.sum()
        n_f = y.values[mask].sum()

        status = "UNSTABLE" if n_f < 3 else "OK"
        print(f"  {curve_name} bin {i}: {n_s} samples, {n_f} frauds ({status})")

        mean_pred.append(mean_p)
        frac_pos.append(frac)
        n_samples.append(n_s)
        n_frauds.append(n_f)

    mean_pred = np.array(mean_pred)
    frac_pos = np.array(frac_pos)

    ax1.plot(mean_pred, frac_pos, marker=markers[curve_name], color=colors[curve_name],
             linewidth=2, markersize=7, label=curve_name)

    for j in range(len(mean_pred)):
        label = f"{n_frauds[j]}/{n_samples[j]}"
        ax1.annotate(label, (mean_pred[j], frac_pos[j]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=7, color=colors[curve_name], alpha=0.8)

ax1.set_xlabel("Mean predicted probability", fontsize=12)
ax1.set_ylabel("Fraction of actual positives (fraud rate)", fontsize=12)
ax1.set_title("Reliability Diagram: Zoomed View (0–0.1)\n"
              "5-fold OOF predictions, quantile bins", fontsize=13)
ax1.set_xlim(-0.005, 0.12)
ax1.set_ylim(0, 0.12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# ---- FIGURE 2: Calibration Error (Difference Plot) ----
fig2, ax2 = plt.subplots(figsize=(10, 6))

for curve_name, proba in curves.items():
    mean_pred = []
    diff = []
    n_samples = []

    for i in range(N_BINS):
        mask = (bin_indices == i)
        if mask.sum() == 0:
            continue
        mean_p = proba[mask].mean()
        frac = y.values[mask].mean()
        n_s = mask.sum()

        mean_pred.append(mean_p)
        diff.append(mean_p - frac)  # positive = overconfident, negative = underconfident
        n_samples.append(n_s)

    mean_pred = np.array(mean_pred)
    diff = np.array(diff)

    ax2.plot(mean_pred, diff, marker=markers[curve_name], color=colors[curve_name],
             linewidth=2, markersize=7, label=curve_name)

    for j in range(len(mean_pred)):
        label = f"{diff[j]:+.4f}"
        ax2.annotate(label, (mean_pred[j], diff[j]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color=colors[curve_name], alpha=0.8)

ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel("Mean predicted probability", fontsize=12)
ax2.set_ylabel("Predicted - Actual fraud rate (Calibration Error)", fontsize=12)
ax2.set_title("Calibration Error: Predicted vs Actual\n"
              "Positive = Overconfident, Negative = Underconfident", fontsize=13)
ax2.set_xlim(-0.005, 0.12)
ax2.set_ylim(-0.05, 0.05)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)

# ---- FIGURE 3: Full-range reliability diagram with symlog ----
fig3, ax3 = plt.subplots(figsize=(10, 8))

ax3.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.5)

for curve_name, proba in curves.items():
    mean_pred = []
    frac_pos = []

    for i in range(N_BINS):
        mask = (bin_indices == i)
        if mask.sum() == 0:
            continue
        mean_p = proba[mask].mean()
        frac = y.values[mask].mean()
        mean_pred.append(mean_p)
        frac_pos.append(frac)

    mean_pred = np.array(mean_pred)
    frac_pos = np.array(frac_pos)

    ax3.plot(mean_pred, frac_pos, marker=markers[curve_name], color=colors[curve_name],
             linewidth=2, markersize=7, label=curve_name)

ax3.set_xlabel("Mean predicted probability (symlog scale)", fontsize=12)
ax3.set_ylabel("Fraction of actual positives", fontsize=12)
ax3.set_title("Reliability Diagram: Full Range (symlog)\n"
              "5-fold OOF predictions, quantile bins", fontsize=13)
ax3.set_xscale("symlog", linthresh=0.01)
ax3.set_xlim(-0.01, 0.5)
ax3.set_ylim(0, 0.5)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(True, alpha=0.3)

# Save all figures
import os
os.makedirs("eda_outputs", exist_ok=True)

fig1.tight_layout()
fig1.savefig("eda_outputs/reliability_diagram_zoomed.png", dpi=150)
plt.close(fig1)

fig2.tight_layout()
fig2.savefig("eda_outputs/calibration_error.png", dpi=150)
plt.close(fig2)

fig3.tight_layout()
fig3.savefig("eda_outputs/reliability_diagram_full.png", dpi=150)
plt.close(fig3)

print("\nSaved:")
print("  eda_outputs/reliability_diagram_zoomed.png  (0–0.1 zoom)")
print("  eda_outputs/calibration_error.png          (difference plot)")
print("  eda_outputs/reliability_diagram_full.png   (full range symlog)")
print("\nReliability diagrams complete.")