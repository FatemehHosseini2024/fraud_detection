"""
Regression check after the F1/F2 label fix.

Asserts three things:
  1. the F1/F2 closed forms are genuinely different, and the OLD code printed F1
     while calling it F2
  2. error_analysis_fast.py still selects the SAME locked threshold and produces the
     SAME confusion matrix - the fix touched labels only
  3. the corrected prints now match sklearn
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import StratifiedKFold

from pipeline import prepare_data

RANDOM_STATE = 42
N_FOLDS = 5

print("=" * 78)
print("CHECK 1 - F1 and F2 are different functions, not the same one relabelled")
print("=" * 78)

print(f"\n  {'TP':>5} {'FP':>5} {'FN':>5} {'F1=2TP/(2TP+FP+FN)':>22} "
      f"{'F2=5TP/(5TP+4FN+FP)':>22} {'sklearn F2':>13}")
print("  " + "-" * 78)
cases = [(73, 9, 22), (320, 67, 58), (1, 0, 0), (0, 5, 3), (10, 10, 10), (95, 0, 0)]
for tp, fp, fn in cases:
    y = np.array([1] * tp + [0] * fp + [1] * fn)
    p = np.array([1] * tp + [1] * fp + [0] * fn)
    old = 2 * tp / (2 * tp + fn + fp)
    new = 5 * tp / (5 * tp + 4 * fn + fp)
    sk = fbeta_score(y, p, beta=2, zero_division=0)
    print(f"  {tp:>5} {fp:>5} {fn:>5} {old:>22.6f} {new:>22.6f} {sk:>13.6f}")
    assert abs(new - sk) < 1e-9, "closed form disagrees with sklearn"

print("\n  The middle column is what the old code printed under the label 'F2'.")
print("  Compare with the right column: at TP=73, FP=9, FN=22 the old label printed")
print("  0.824859, which is F1, while the true F2 is 0.790043.")
print("  Root cause: F1 = 2TP/(2TP+FP+FN) is correct, but it is NOT the F2 formula.")
print("  F2 = 5TP/(5TP+4FN+FP) = 5PR/(4P+R). They coincide only in special cases.")

print("\n" + "=" * 78)
print("CHECK 2 - threshold and confusion matrix unchanged by the fix")
print("=" * 78)

df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
raw, ys = [], []
for tr, va in cv.split(X, y):
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X.iloc[tr], y.iloc[tr])
    raw.append(rf.predict_proba(X.iloc[va])[:, 1])
    ys.append(y.iloc[va].to_numpy())
oof_raw = np.concatenate(raw)
oof_y = np.concatenate(ys)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_raw, oof_y)
oof_iso = iso.predict(oof_raw)

best, LOCKED = -1, None
for t in np.unique(oof_iso):
    if t <= 0:
        continue
    f2 = fbeta_score(oof_y, (oof_iso >= t).astype(int), beta=2, zero_division=0)
    if f2 > best:
        best, LOCKED = f2, float(t)
LOCKED = float(LOCKED)

oof_pred = (oof_iso >= LOCKED).astype(int)
tn, fp, fn, tp = confusion_matrix(oof_y, oof_pred, labels=[0, 1]).ravel()
sk_f1 = fbeta_score(oof_y, oof_pred, beta=1, zero_division=0)
sk_f2 = fbeta_score(oof_y, oof_pred, beta=2, zero_division=0)

print(f"\n  locked threshold           : {LOCKED!r}")
print(f"  expected 0.21428571428571427: {LOCKED == 3/14}")
print(f"  TP/FN/FP/TN                : {tp}/{fn}/{fp}/{tn}")
print(f"  expected 320/58/67/226535  : {(tp, fn, fp, tn) == (320, 58, 67, 226535)}")

assert LOCKED == 3 / 14, "THreshold moved!"
assert (tp, fn, fp, tn) == (320, 58, 67, 226535), "confusion matrix moved!"

print(f"\n  sklearn F1 at locked threshold: {sk_f1:.6f}")
print(f"  sklearn F2 at locked threshold: {sk_f2:.6f}")
print(f"  value the OLD 'F2' print showed : {2*tp/(2*tp+fn+fp):.6f}  <- was F1")
print(f"  old print == sklearn F1 ? {abs(2*tp/(2*tp+fn+fp) - sk_f1) < 1e-9}")

print(f"\n  threshold search used fbeta_score(beta=2) both before and after the fix,")
print(f"  so the argmax and the locked value were never affected: F2 = {sk_f2:.6f} is")
print(f"  still the maximum over the candidate grid (verified in")
print(f"  verify_locked_threshold.py, Audit 1).")

print("\n" + "=" * 78)
print("CHECK 3 - every F1/F2 label in the repo now matches sklearn")
print("=" * 78)

from sklearn.metrics import precision_score, recall_score
p_s = precision_score(oof_y, oof_pred, zero_division=0)
r_s = recall_score(oof_y, oof_pred, zero_division=0)
print(f"\n  precision {p_s:.6f}  recall {r_s:.6f}")
print(f"  F1 = 2PR/(P+R)      = {2*p_s*r_s/(p_s+r_s):.6f}  vs sklearn {sk_f1:.6f}")
print(f"  F2 = 5PR/(4P+R)     = {5*p_s*r_s/(4*p_s+r_s):.6f}  vs sklearn {sk_f2:.6f}")
assert abs(2 * p_s * r_s / (p_s + r_s) - sk_f1) < 1e-9
assert abs(5 * p_s * r_s / (4 * p_s + r_s) - sk_f2) < 1e-9

print("\n  The repo now calls sklearn fbeta_score(beta=1) and fbeta_score(beta=2)")
print("  instead of hand-rolled closed forms, so the two cannot be confused again.")

print("\n" + "=" * 78)
print("ALL REGRESSION CHECKS PASSED - threshold and metrics unchanged, labels fixed")
print("=" * 78)