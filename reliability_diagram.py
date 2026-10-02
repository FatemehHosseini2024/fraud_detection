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
print("RELIABILITY DIAGRAM: RF Calibration Curves")
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
print("  Sigmoid calibrator fitted")

isotonic_cal = IsotonicRegression(out_of_bounds="clip")
isotonic_cal.fit(oof_preds, y)
print("  Isotonic calibrator fitted")

# Step 3: Apply calibrators to get calibrated OOF probabilities
oob_sigmoid = sigmoid_cal.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
oob_isotonic = isotonic_cal.predict(oof_preds)

print("\nBrier Scores on OOF predictions:")
print(f"  Raw RF:     {brier_score_loss(y, oof_preds):.6f}")
print(f"  Sigmoid:    {brier_score_loss(y, oob_sigmoid):.6f}")
print(f"  Isotonic:   {brier_score_loss(y, oob_isotonic):.6f}")

# Step 4: Create quantile bins based on Raw RF predictions
# Use equal-count (quantile) bins via manual sorting for robustness
# against degenerate distributions (many 0 probabilities)
print(f"\nCreating {N_BINS} quantile bins from Raw RF predictions...")
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

fig, ax = plt.subplots(figsize=(10, 8))

# Perfect calibration diagonal
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration (y=x)", alpha=0.5)

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

    # Plot curve
    ax.plot(mean_pred, frac_pos, marker=markers[curve_name], color=colors[curve_name],
            linewidth=2, markersize=7, label=curve_name)

    # Annotate with fraud/sample counts
    for j in range(len(mean_pred)):
        label = f"{n_frauds[j]}/{n_samples[j]}"
        ax.annotate(label, (mean_pred[j], frac_pos[j]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=7, color=colors[curve_name], alpha=0.8)

# Axis settings
ax.set_xlabel("Mean predicted probability (log scale)", fontsize=12)
ax.set_ylabel("Fraction of actual positives (fraud rate in bin)", fontsize=12)
ax.set_title("Reliability Diagram: RF + Calibration Curves\n"
             "5-fold OOF predictions, quantile bins", fontsize=13)
ax.set_xscale("symlog", linthresh=0.01)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

# Annotation note
ax.text(0.02, 0.98,
        "Annotation: frauds/samples per bin\n"
        "symlog x-axis (linear near 0, log elsewhere)\n"
        "Diagonal = perfect calibration",
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig("eda_outputs/reliability_diagram.png", dpi=150)
plt.close()
print("\nSaved: eda_outputs/reliability_diagram.png")
print("\nReliability diagram complete.")
