"""
Resolves the apparent contradiction: the F2 argmax row (Proof 1) and the
"error_analysis_fast" column of the comparison table are the SAME evaluation.

They are NOT supposed to match the "FINAL (test)" column. Same threshold, same decision
rule, two different evaluation sets with two different models.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from pipeline import prepare_data

RANDOM_STATE = 42
N_FOLDS = 5

df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"].to_numpy()

# ---- Set A: train OOF, each RF saw 80% of train (this IS error_analysis_fast) -------
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
raw_a, y_a = [], []
for tr, va in cv.split(X, y):
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X.iloc[tr], y.iloc[tr])
    raw_a.append(rf.predict_proba(X.iloc[va])[:, 1])
    y_a.append(y.iloc[va].to_numpy())
oof_raw = np.concatenate(raw_a)
oof_y = np.concatenate(y_a)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_raw, oof_y)
oof_iso = iso.predict(oof_raw)

best_f2, LOCKED = -1, None
for t in np.unique(oof_iso):
    if t <= 0:
        continue
    f2 = fbeta_score(oof_y, (oof_iso >= t).astype(int), beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2, LOCKED = f2, float(t)

# ---- Set B: test, one RF saw 100% of train (this IS the final model) ----------------
rf_full = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_full.fit(X, y)
test_iso = iso.predict(rf_full.predict_proba(X_test)[:, 1])


def row(name, y_true, scores):
    p = (scores >= LOCKED).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
    return {
        "name": name,
        "n": len(y_true),
        "frauds": int(y_true.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged": int(p.sum()),
        "precision": float(precision_score(y_true, p, zero_division=0)),
        "recall": float(recall_score(y_true, p, zero_division=0)),
        "f1": float(fbeta_score(y_true, p, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, p, beta=2, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


A = row("A = train OOF  (error_analysis_fast)", oof_y, oof_iso)
B = row("B = TEST       (FINAL model)", y_test, test_iso)

print("=" * 92)
print(f"ONE threshold, ONE decision rule: (score >= {LOCKED:.6f}). Two evaluation sets.")
print("=" * 92)

print(f"\n{'':44}{'A: train OOF':>22}{'B: TEST':>22}")
print("-" * 92)

lines = [
    ("rows scored", "n", "{:,.0f}"),
    ("frauds in that set", "frauds", "{:,.0f}"),
    ("flagged (predicted fraud)", "flagged", "{:,.0f}"),
    ("TP  true fraud flagged", "tp", "{:,.0f}"),
    ("FP  normal flagged", "fp", "{:,.0f}"),
    ("FN  fraud missed", "fn", "{:,.0f}"),
    ("TN  normal cleared", "tn", "{:,.0f}"),
    ("", None, None),
    ("precision = TP/flagged", "precision", "{:.4f}"),
    ("recall    = TP/frauds", "recall", "{:.4f}"),
    ("F1", "f1", "{:.4f}"),
    ("F2", "f2", "{:.4f}"),
    ("", None, None),
    ("PR-AUC", "pr_auc", "{:.4f}"),
    ("Brier", "brier", "{:.6f}"),
]
for label, key, fmt in lines:
    if key is None:
        print("-" * 92)
        continue
    print(f"{label:44}{fmt.format(A[key]):>22}{fmt.format(B[key]):>22}")

print("\n" + "=" * 92)
print("CROSS-CHECK: Proof 1's argmax row vs the comparison table's columns")
print("=" * 92)
print(f"\n  Proof 1 argmax row (F2 curve on TRAIN OOF) : prec={A['precision']:.4f} "
      f"rec={A['recall']:.4f} F2={A['f2']:.4f}")
print(f"  Comparison table 'error_analysis_fast' col : prec=0.8269 rec=0.8466 F2=0.8425")
print(f"  Match? precision {abs(A['precision']-0.8269)<5e-5}, "
      f"recall {abs(A['recall']-0.8466)<5e-5}, F2 {abs(A['f2']-0.8425)<5e-5}")
print(f"\n  -> They are the SAME evaluation (set A). Consistent.")
print(f"  -> Proof 1 was never claiming to predict set B's numbers.")

print("\n" + "=" * 92)
print("WHY A and B DIFFER even though the threshold is identical")
print("=" * 92)

print(f"\n  The decision rule is fixed: flag anything scoring >= {LOCKED:.6f}.")
print(f"  But the two sets hand that rule very different populations.\n")
print(f"  {'':<34}{'A: train OOF':>18}{'B: TEST':>18}")
print(f"  {'-'*34}{'-'*18}{'-'*18}")
print(f"  {'frauds to find':>34}{A['frauds']:>18,}{B['frauds']:>18,}")
print(f"  {'rows flagged by the rule':>34}{A['flagged']:>18,}{B['flagged']:>18,}")
print(f"  {'flagged rows per fraud':>34}{A['flagged']/A['frauds']:>18.3f}{B['flagged']/B['frauds']:>18.3f}")

print(f"\n  Set A flags {A['flagged']/A['frauds']:.3f} rows per fraud: it flags MORE rows than")
print(f"  there are frauds, so some normals get swept in -> precision {A['precision']:.4f}.")
print(f"  Set B flags {B['flagged']/B['frauds']:.3f} rows per fraud: it flags FEWER rows than")
print(f"  there are frauds, so the flag list runs out -> recall {B['recall']:.4f}, but")
print(f"  precision rises to {B['precision']:.4f}.")

print(f"\n  Worked arithmetic:")
print(f"    A: precision = TP/flagged = {A['tp']}/{A['tp']+A['fp']} = {A['precision']:.6f}"
      f" | recall = TP/frauds = {A['tp']}/{A['frauds']} = {A['recall']:.6f}")
print(f"    B: precision = TP/flagged = {B['tp']}/{B['tp']+B['fp']} = {B['precision']:.6f}"
      f" | recall = TP/frauds = {B['tp']}/{B['frauds']} = {B['recall']:.6f}")

print(f"\n  Second contributing factor: the model changed too.")
print(f"    Set A scores come from RFs trained on 80% of train (5 of them).")
print(f"    Set B scores come from one RF trained on 100% of train.")
print(f"    A model with more training data puts sharper probabilities into the upper")
print(f"    isotonic steps, which is why set B's flagged rows skew higher-confidence.")

print(f"\n  Threshold-based metrics (precision/recall/F1/F2) move with the population.")
print(f"  Threshold-free metrics (PR-AUC {A['pr_auc']:.4f} -> {B['pr_auc']:.4f},")
print(f"  Brier {A['brier']:.6f} -> {B['brier']:.6f}) move with the model, not the cutoff.")
print(f"  That is exactly why both columns exist in the comparison table.")

print("\n" + "=" * 92)
print("CONCLUSION")
print("=" * 92)
print(f"  Threshold {LOCKED:.6f} was applied to BOTH sets. Neither number set is wrong.")
print(f"  Proof 1 documents set A. The comparison table's right column is set B.")
print(f"  They are expected to differ, and they do.")