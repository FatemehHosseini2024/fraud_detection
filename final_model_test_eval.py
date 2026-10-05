"""
FINAL MODEL: Default RF + Isotonic -> trained on FULL TRAIN, evaluated ONCE on TEST.

Methodology is identical to error_analysis_fast.py:
  1. 5-fold StratifiedKFold OOF on train with RandomForestClassifier(random_state=42, n_jobs=-1)
  2. IsotonicRegression(out_of_bounds="clip") fitted on ALL OOF train probabilities (calibrator)
  3. LOCKED_THRESHOLD = Max F2 over unique isotonic outputs (> 0) on OOF
  4. RF retrained on the FULL train frame, calibrator + threshold frozen from train only
  5. Test scored exactly once

The ONLY difference vs error_analysis_fast.py: steps 4-5 (final model uses 100% of
train instead of 80%, and is scored on test instead of OOF).

NOTHING is fitted, tuned, scaled or selected on test: scaler, RF, isotonic calibrator
and threshold are all inherited from the train-only artifacts above.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from pipeline import prepare_data

RANDOM_STATE = 42
N_FOLDS = 5
N_BOOTSTRAP = 2000
N_BINS = 10

OUTPUT_DIR = "eda_outputs"
CALIB_CSV = os.path.join(OUTPUT_DIR, "final_test_calibration_bins.csv")
CALIB_PNG = os.path.join(OUTPUT_DIR, "final_test_calibration_chart.png")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "final_model_test_eval.json")

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)


def make_rf():
    """Exact estimator used by error_analysis_fast.py - all sklearn defaults."""
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)


def metrics_from(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def bootstrap_ci(y_true, scores, threshold, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Plain nonparametric bootstrap over test rows. Fixed predictions, fixed threshold."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)

    keys = ["pr_auc", "brier", "precision", "recall", "f1", "f2"]
    draws = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        s_b = scores[idx]
        if y_b.sum() == 0:
            continue
        p_b = (s_b >= threshold).astype(int)
        draws["pr_auc"].append(average_precision_score(y_b, s_b))
        draws["brier"].append(brier_score_loss(y_b, s_b))
        draws["precision"].append(precision_score(y_b, p_b, zero_division=0))
        draws["recall"].append(recall_score(y_b, p_b, zero_division=0))
        draws["f1"].append(fbeta_score(y_b, p_b, beta=1, zero_division=0))
        draws["f2"].append(fbeta_score(y_b, p_b, beta=2, zero_division=0))

    out = {}
    for k in keys:
        v = np.asarray(draws[k])
        out[k] = {
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "ci_95_low": float(np.percentile(v, 2.5)),
            "ci_95_high": float(np.percentile(v, 97.5)),
            "n_valid": int(len(v)),
        }
    return out


def calibration_table(y_true, raw, iso, edges):
    """Reliability bins with FIXED edges (derived from train, never from test)."""
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (raw >= lo) & (raw <= hi)
        else:
            mask = (raw >= lo) & (raw < hi)
        n = int(mask.sum())
        emp_rate = float(y_true[mask].mean()) if n > 0 else np.nan
        row = {
            "bin": i,
            "bin_low": float(lo),
            "bin_high": float(hi),
            "n": n,
            "frauds": int(y_true[mask].sum()) if n > 0 else 0,
            "mean_raw": float(raw[mask].mean()) if n > 0 else np.nan,
            "mean_iso": float(iso[mask].mean()) if n > 0 else np.nan,
            "emp_rate": emp_rate,
        }
        row["gap_raw"] = row["mean_raw"] - emp_rate
        row["gap_iso"] = row["mean_iso"] - emp_rate
        rows.append(row)
    return rows


def print_ci_block(name, m, ci):
    print(f"\n  {name}")
    print(f"    {'Metric':<10} {'Value':>10} {'95% CI':>26}")
    print(f"    {'-'*10} {'-'*10} {'-'*26}")
    for k in ["pr_auc", "brier", "precision", "recall", "f1", "f2"]:
        c = ci[k]
        print(f"    {k.upper():<10} {m[k]:>10.4f} "
              f"{'[' + format(c['ci_95_low'], '.4f') + ', ' + format(c['ci_95_high'], '.4f') + ']':>26}")


def print_cm(m):
    print(f"    Confusion matrix at threshold={m['threshold']:.6f}")
    print(f"    (rows = actual, cols = predicted)")
    print(f"      {'':<10} {'pred=0':>10} {'pred=1':>10}")
    print(f"      {'actual=0':<10} {m['tn']:>10} {m['fp']:>10}")
    print(f"      {'actual=1':<10} {m['fn']:>10} {m['tp']:>10}")


def make_calibration_chart(bin_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axz, axn) = plt.subplots(
        3, 1, figsize=(12, 16), gridspec_kw={"height_ratios": [3, 3, 1.4]}
    )

    ax.plot([0, 1], [0, 1], ls="--", lw=2, color="black", label="Perfect calibration")
    ax.plot(bin_df["mean_raw"], bin_df["emp_rate"], marker="s", ms=7, lw=2,
            color="#1f77b4", label="TEST - raw RF (uncalibrated)")
    ax.plot(bin_df["mean_iso"], bin_df["emp_rate"], marker="o", ms=8, lw=2,
            color="#d62728", label="TEST - isotonic calibrated (deployed score)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability in bin")
    ax.set_ylabel("Empirical fraud rate in bin")
    ax.set_title("Calibration on TEST - full range\n"
                 "10 bins, edges frozen from TRAIN OOF before test was scored")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    z = bin_df[bin_df["bin"] > 0].dropna(subset=["mean_iso"])
    floor = 1e-4
    axz.plot([floor, 1], [floor, 1], ls="--", lw=2, color="black", label="Perfect calibration")
    axz.plot(z["mean_raw"].clip(lower=floor), z["emp_rate"].clip(lower=floor),
             marker="s", ms=7, lw=2, color="#1f77b4", label="TEST - raw RF")
    axz.plot(z["mean_iso"].clip(lower=floor), z["emp_rate"].clip(lower=floor),
             marker="o", ms=8, lw=2, color="#d62728", label="TEST - isotonic calibrated")
    axz.set_xscale("log")
    axz.set_yscale("log")
    axz.set_xlim(floor, 1.05)
    axz.set_ylim(floor, 1.05)
    axz.set_xlabel("Mean predicted probability in bin (log)")
    axz.set_ylabel("Empirical fraud rate in bin (log)")
    axz.set_title("Calibration on TEST - log zoom, p == 0 mass excluded")
    axz.grid(alpha=0.3, which="both")
    axz.legend(loc="upper left")

    axn.bar(range(len(bin_df)), bin_df["n"], color="#d62728")
    axn.set_yscale("log")
    axn.set_xticks(range(len(bin_df)))
    axn.set_xlabel("Bin index (0 = p == 0 mass)")
    axn.set_ylabel("Test rows in bin (log)")
    axn.set_title("Bin occupancy on TEST")
    axn.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(CALIB_PNG, dpi=130)
    plt.close(fig)


# =============================================================================
print("=" * 78)
print("FINAL MODEL: Default RF + Isotonic -> Full Train -> Test (single evaluation)")
print("=" * 78)

df, train, test, _ = prepare_data()
X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"].to_numpy()

print(f"\nTrain: {X_train.shape[0]} rows, Fraud: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.4f}%)")
print(f"Test:  {X_test.shape[0]} rows, Fraud: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.4f}%)")
print(f"Features: {X_train.shape[1]}")

# -----------------------------------------------------------------------------
# STAGE 1 (train-only, identical to error_analysis_fast.py): OOF -> calibrator -> threshold
# -----------------------------------------------------------------------------
print("\n" + "-" * 78)
print("STAGE 1 - TRAIN-ONLY ARTIFACTS (identical to error_analysis_fast.py)")
print("-" * 78)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_raw_proba = []
all_y_val = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"  OOF fold {fold}/{N_FOLDS}...")
    rf = make_rf()
    rf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    all_raw_proba.append(rf.predict_proba(X_train.iloc[val_idx])[:, 1])
    all_y_val.append(y_train.iloc[val_idx].to_numpy())

oof_raw = np.concatenate(all_raw_proba)
oof_y = np.concatenate(all_y_val)
print(f"\n  OOF collected: {len(oof_raw)} samples, {oof_y.sum()} frauds")

# Calibrator: isotonic fitted on ALL OOF train probabilities
final_iso = IsotonicRegression(out_of_bounds="clip")
final_iso.fit(oof_raw, oof_y)
oof_iso = final_iso.predict(oof_raw)
print(f"  Isotonic calibrator fitted on {len(oof_raw)} train OOF samples")

# Locked threshold: Max F2 on unique isotonic outputs (>0), strict '>' so the
# smallest threshold wins ties - identical tie-breaking to error_analysis_fast.py
unique_iso = np.unique(oof_iso)
unique_iso = unique_iso[unique_iso > 0]
best_f2 = -1
LOCKED_THRESHOLD = None
for thresh in unique_iso:
    y_pred = (oof_iso >= thresh).astype(int)
    f2 = fbeta_score(oof_y, y_pred, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2 = f2
        LOCKED_THRESHOLD = thresh
LOCKED_THRESHOLD = float(LOCKED_THRESHOLD)
print(f"  LOCKED_THRESHOLD (Max F2 on train OOF): {LOCKED_THRESHOLD:.6f}")

# Train-side metrics for the error_analysis_fast comparison (recomputed, same config)
oof_metrics = metrics_from(oof_y, oof_iso, LOCKED_THRESHOLD)
print(f"  [reference] error_analysis_fast config on train OOF:")
print(f"    Precision={oof_metrics['precision']:.4f} Recall={oof_metrics['recall']:.4f} "
      f"F1={oof_metrics['f1']:.4f} F2={oof_metrics['f2']:.4f} "
      f"PR-AUC={oof_metrics['pr_auc']:.4f} Brier={oof_metrics['brier']:.6f}")

# Calibration bin edges: FROZEN from train, then applied unchanged to test.
#
# RandomForest probabilities here are discrete (k/100 for integer k), so plain deciles of
# the raw score collapse onto a handful of values and produce far fewer than N_BINS real
# bins. Fix: bin 0 takes the p == 0 mass, and the nonzero train OOF probabilities are cut
# with the finest quantile grid that still yields at least N_BINS bins. All edges come from
# train only.
oof_nonzero = oof_raw[oof_raw > 0]
n_zero = len(oof_raw) - len(oof_nonzero)
print(f"  Train OOF raw prob: {n_zero}/{len(oof_raw)} ({n_zero/len(oof_raw)*100:.2f}%) are exactly 0")

bin_edges = np.array([0.0])
for k in range(2, 102):
    cand = np.unique(np.concatenate([[0.0], np.quantile(oof_nonzero, np.linspace(0, 1, k))]))
    if len(cand) - 1 >= N_BINS:
        break
bin_edges = cand
bin_edges[-1] = 1.0
print(f"  Calibration bin edges (train OOF nonzero raw probabilities, frozen): "
      f"{len(bin_edges)-1} bins")
print(f"    edges: {[round(float(e), 6) for e in bin_edges]}")
print(f"  Distinct calibrated levels produced by isotonic on train OOF: {len(np.unique(oof_iso))}")

# -----------------------------------------------------------------------------
# STAGE 2: retrain on FULL train, freeze everything, score test ONCE
# -----------------------------------------------------------------------------
print("\n" + "-" * 78)
print("STAGE 2 - FINAL MODEL: fit on 100% of train, evaluate test once")
print("-" * 78)

rf_full = make_rf()
rf_full.fit(X_train, y_train)
print(f"  Final RF trained on all {X_train.shape[0]} train rows")

test_raw = rf_full.predict_proba(X_test)[:, 1]
test_iso = final_iso.predict(test_raw)          # frozen train calibrator
test_pred = (test_iso >= LOCKED_THRESHOLD).astype(int)  # frozen train threshold

test_metrics = metrics_from(y_test, test_iso, LOCKED_THRESHOLD)
test_ci = bootstrap_ci(y_test, test_iso, LOCKED_THRESHOLD)
oof_ci = bootstrap_ci(oof_y, oof_iso, LOCKED_THRESHOLD)

n_bootstrap_used = test_ci["pr_auc"]["n_valid"]
print(f"  Bootstrap: {N_BOOTSTRAP} resamples, {n_bootstrap_used} valid, seed={RANDOM_STATE}")

# -----------------------------------------------------------------------------
# TEST RESULTS
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("TEST RESULTS (single evaluation, nothing fitted on test)")
print("=" * 78)
print(f"\nThreshold applied: {LOCKED_THRESHOLD:.6f}  (locked on train OOF, never revisited)")
print(f"Calibrator: IsotonicRegression(out_of_bounds='clip') fitted on train OOF only")
print(f"Classifier: RandomForestClassifier(random_state=42, n_jobs=-1), sklearn defaults, "
      f"trained on 100% of train")

print_cm(test_metrics)

print("\n  Confusion matrix, counts as fractions")
tot = len(y_test)
print(f"    Precision = TP/(TP+FP) = {test_metrics['tp']}/{test_metrics['tp']+test_metrics['fp']}"
      f" = {test_metrics['precision']:.6f}")
print(f"    Recall    = TP/(TP+FN) = {test_metrics['tp']}/{test_metrics['tp']+test_metrics['fn']}"
      f" = {test_metrics['recall']:.6f}")
print(f"    F1        = {test_metrics['f1']:.6f}")
print(f"    F2        = {test_metrics['f2']:.6f}")
print(f"    Fraud base rate = {y_test.sum()}/{tot} = {y_test.sum()/tot:.6f}")
print(f"    Flagged rate    = {test_pred.sum()}/{tot} = {test_pred.sum()/tot:.6f}")

print_ci_block("TEST METRICS + BOOTSTRAP 95% CI", test_metrics, test_ci)

print("\n  PR-AUC uses isotonic-calibrated test probabilities (the deployed score).")
print(f"  For reference, ROC-AUC = {test_metrics['roc_auc']:.6f} and "
      f"PR-AUC on raw RF (uncalibrated) = {average_precision_score(y_test, test_raw):.6f}")

# -----------------------------------------------------------------------------
# CALIBRATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("CALIBRATION ON TEST (bin edges frozen from train OOF)")
print("=" * 78)

bin_df = pd.DataFrame(calibration_table(y_test, test_raw, test_iso, bin_edges))
bin_df.to_csv(CALIB_CSV, index=False)

print(f"\n  Bin 0 is the p == 0 mass; bins 1+ come from the frozen train-only quantile grid.")
print(f"\n  {'Bin':<4} {'range (raw)':<24} {'n':>9} {'frauds':>8} "
      f"{'mean_raw':>10} {'mean_iso':>10} {'emp_rate':>10} {'gap_raw':>11} {'gap_iso':>11}")
print(f"  {'-'*4} {'-'*24} {'-'*9} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*11}")
for _, r in bin_df.iterrows():
    rng = f"[{r['bin_low']:.6f}, {r['bin_high']:.6f})"
    print(f"  {int(r['bin']):<4} {rng:<24} {int(r['n']):>9} {int(r['frauds']):>8} "
          f"{r['mean_raw']:>10.6f} {r['mean_iso']:>10.6f} {r['emp_rate']:>10.6f} "
          f"{r['gap_raw']:>+11.6f} {r['gap_iso']:>+11.6f}")

w = bin_df["n"] / tot
ece_raw = float((w * bin_df["gap_raw"].abs()).sum())
ece_iso = float((w * bin_df["gap_iso"].abs()).sum())
mse_raw = float((w * bin_df["gap_raw"].pow(2)).sum())
mse_iso = float((w * bin_df["gap_iso"].pow(2)).sum())
print(f"\n  ECE  raw={ece_raw:.6f}   isotonic={ece_iso:.6f}")
print(f"  MSE  raw={mse_raw:.8f}   isotonic={mse_iso:.8f}")
print(f"  Brier (pointwise, whole test set): raw={brier_score_loss(y_test, test_raw):.8f}  "
      f"isotonic={test_metrics['brier']:.8f}")

# Calibration restricted to the flagged region (iso >= threshold) and to the whole set.
flag = test_iso >= LOCKED_THRESHOLD
print(f"\n  Within the flagged region only (iso >= {LOCKED_THRESHOLD:.6f}, n={int(flag.sum())}):")
print(f"    mean calibrated p = {test_iso[flag].mean():.6f}, "
      f"empirical fraud rate = {y_test[flag].mean():.6f}, "
      f"gap = {test_iso[flag].mean() - y_test[flag].mean():+.6f}")

# distinct calibrated levels actually emitted by the frozen calibrator on test
levels = pd.Series(test_iso).value_counts().sort_index()
print(f"\n  Distinct calibrated levels emitted by the frozen isotonic on test: {len(levels)}")
print(f"  {'level p':>10} {'n':>9} {'frauds':>8} {'emp_rate':>11} {'gap':>11}")
print(f"  {'-'*10} {'-'*9} {'-'*8} {'-'*11} {'-'*11}")
for lv, cnt in levels.items():
    m = test_iso == lv
    emp = y_test[m].mean()
    print(f"  {lv:>10.6f} {cnt:>9} {int(y_test[m].sum()):>8} {emp:>11.6f} {lv - emp:>+11.6f}")

make_calibration_chart(bin_df)
print(f"\n  Calibration chart saved: {CALIB_PNG}")
print(f"  Calibration bins CSV:  {CALIB_CSV}")

# -----------------------------------------------------------------------------
# COMPARISON vs error_analysis_fast
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("COMPARISON: error_analysis_fast.py vs FINAL MODEL")
print("=" * 78)
print(f"\n  {'Metric':<10} {'error_analysis_fast':>26} {'FINAL (test)':>26} {'Delta':>12}")
print(f"  {'-'*10} {'-'*26} {'-'*26} {'-'*12}")

rows_cmp = []
for k in ["pr_auc", "brier", "precision", "recall", "f1", "f2"]:
    o, t = oof_metrics[k], test_metrics[k]
    lo_o, hi_o = oof_ci[k]["ci_95_low"], oof_ci[k]["ci_95_high"]
    lo_t, hi_t = test_ci[k]["ci_95_low"], test_ci[k]["ci_95_high"]
    left = f"{o:.4f} [{lo_o:.4f}, {hi_o:.4f}]"
    right = f"{t:.4f} [{lo_t:.4f}, {hi_t:.4f}]"
    print(f"  {k.upper():<10} {left:>26} {right:>26} {t - o:>+12.4f}")
    rows_cmp.append({"metric": k, "fast_oof": o, "fast_ci_low": lo_o, "fast_ci_high": hi_o,
                     "final_test": t, "final_ci_low": lo_t, "final_ci_high": hi_t,
                     "delta": t - o})

print(f"\n  {'Count':<10} {'error_analysis_fast':>26} {'FINAL (test)':>26}")
print(f"  {'-'*10} {'-'*26} {'-'*26}")
for k in ["tp", "fn", "fp", "tn"]:
    print(f"  {k.upper():<10} {oof_metrics[k]:>26} {test_metrics[k]:>26}")

print(f"\n  {'Other':<10} {'error_analysis_fast':>26} {'FINAL (test)':>26}")
print(f"  {'-'*10} {'-'*26} {'-'*26}")
print(f"  {'threshold':<10} {LOCKED_THRESHOLD:>26.6f} {LOCKED_THRESHOLD:>26.6f}")
print(f"  {'roc_auc':<10} {oof_metrics['roc_auc']:>26.4f} {test_metrics['roc_auc']:>26.4f}")

print("\n  Reading the comparison:")
print("   - error_analysis_fast numbers are 5-fold OOF on TRAIN: each score comes from an")
print("     RF trained on 80% of train. FINAL numbers are on TEST: one RF trained on 100%")
print("     of train. Two things changed at once (evaluation split and training size), so a")
print("     delta is not attributable to either alone.")
print("   - Both use the identical locked threshold, so threshold effects are removed.")
print("   - error_analysis_fast's threshold was fitted on the same OOF it reports, so its")
print("     precision/recall/F1/F2 are optimistically biased; the FINAL test metrics are not.")

# -----------------------------------------------------------------------------
results = {
    "config": {
        "estimator": "RandomForestClassifier(random_state=42, n_jobs=-1), sklearn defaults",
        "calibrator": "IsotonicRegression(out_of_bounds='clip') fitted on train OOF",
        "cv": f"StratifiedKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_STATE})",
        "locked_threshold": LOCKED_THRESHOLD,
        "trained_on": "100% of train",
        "evaluated_on": "test (single evaluation, nothing fitted on test)",
        "bootstrap": {"n": N_BOOTSTRAP, "seed": RANDOM_STATE, "method": "nonparametric, fixed predictions/threshold"},
    },
    "train": {"n": int(X_train.shape[0]), "frauds": int(y_train.sum())},
    "test": {"n": int(tot), "frauds": int(y_test.sum()), "metrics": test_metrics, "bootstrap_ci": test_ci},
    "error_analysis_fast_reference": {
        "evaluated_on": "train OOF (5-fold)",
        "metrics": oof_metrics,
        "bootstrap_ci": oof_ci,
    },
    "comparison": rows_cmp,
    "calibration": {
        "bin_edges": [float(e) for e in bin_edges],
        "bin_edges_source": "train OOF raw RF probability: bin 0 = p==0 mass, bins 1+ = finest train-only quantile grid of the nonzero mass yielding >=10 bins (frozen before test)",
        "bins": bin_df.to_dict(orient="records"),
        "ece_raw": ece_raw,
        "ece_iso": ece_iso,
        "mse_raw": mse_raw,
        "mse_iso": mse_iso,
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Full results JSON: {RESULTS_JSON}")

print("\n" + "=" * 78)
print("DONE. Test was scored once; scaler, RF, calibrator and threshold all came from train.")
print("=" * 78)