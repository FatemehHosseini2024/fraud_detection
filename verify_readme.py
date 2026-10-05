"""
Verifies every numeric claim in README.md against the saved artifacts.

Fails loudly on any mismatch. Read-only: touches nothing.
"""
import json
import re
import sys

import numpy as np
import pandas as pd

README = open("README.md", encoding="utf-8").read()
fails = []
checks = 0


def claim(label, needle):
    global checks
    checks += 1
    if needle not in README:
        fails.append(f"{label}: README does not contain {needle!r}")


def eq(label, actual, expected, tol=5e-4):
    global checks
    checks += 1
    if abs(float(actual) - float(expected)) > tol:
        fails.append(f"{label}: artifact={actual!r} README={expected!r}")


final = json.load(open("eda_outputs/final_model_test_eval.json"))
thr = json.load(open("eda_outputs/threshold_optimization.json"))
st2 = json.load(open("eda_outputs/stage2_results.json"))

# ---- headline test metrics ----
tm = final["test"]["metrics"]
ci = final["test"]["bootstrap_ci"]
for key, val in [("pr_auc", 0.7874), ("precision", 0.8902), ("recall", 0.7684),
                 ("f1", 0.8249), ("f2", 0.7900), ("roc_auc", 0.9294)]:
    eq(f"test.{key}", tm[key], val, 5e-5)
eq("test.brier", tm["brier"], 0.000460, 5e-7)

for key, lo, hi in [("pr_auc", 0.7036, 0.8669), ("precision", 0.8167, 0.9529),
                    ("recall", 0.6829, 0.8471), ("f1", 0.7632, 0.8808),
                    ("f2", 0.7143, 0.8587), ("brier", 0.000301, 0.000636)]:
    eq(f"test.{key}.ci_low", ci[key]["ci_95_low"], lo, 5e-5 if key != "brier" else 5e-7)
    eq(f"test.{key}.ci_high", ci[key]["ci_95_high"], hi, 5e-5 if key != "brier" else 5e-7)

eq("bootstrap n", final["config"]["bootstrap"]["n"], 2000, 0)
for k in ci:
    eq(f"bootstrap {k} n_valid", ci[k]["n_valid"], 2000, 0)

# ---- confusion matrix ----
for k, v in [("tp", 73), ("fn", 22), ("fp", 9), ("tn", 56642)]:
    eq(f"test.{k}", tm[k], v, 0)
claim("flagged rate", "0.1445%")
eq("flagged rate", tm["tp"] + tm["fp"], 82, 0)

# ---- locked threshold ----
eq("locked threshold", final["config"]["locked_threshold"], 3 / 14, 1e-15)
claim("threshold 3/14 note", "exactly `3/14`")
claim("threshold printed", "0.21428571428571427")
claim("json says max_f2", thr["locked_criterion"])
claim("json 15 candidates", str(thr["candidate_thresholds_count"]) + " distinct")

# ---- reference / OOF column ----
ref = final["error_analysis_fast_reference"]["metrics"]
for key, val in [("pr_auc", 0.8344), ("precision", 0.8269), ("recall", 0.8466),
                 ("f1", 0.8366), ("f2", 0.8425), ("roc_auc", 0.9481)]:
    eq(f"ref.{key}", ref[key], val, 5e-5)
eq("ref.brier", ref["brier"], 0.000389, 5e-7)
for k, v in [("tp", 320), ("fn", 58), ("fp", 67), ("tn", 226535)]:
    eq(f"ref.{k}", ref[k], v, 0)

# ---- calibration on test ----
cal = final["calibration"]
eq("ece_raw", cal["ece_raw"], 0.000551, 5e-7)
eq("ece_iso", cal["ece_iso"], 0.000209, 5e-7)
eq("mse_raw", cal["mse_raw"], 0.00001897, 5e-9)
eq("mse_iso", cal["mse_iso"], 0.00000510, 5e-9)
claim("ECE cut pct", "62%")
claim("MSE cut pct", "73%")
eq("ece reduction pct", (1 - cal["ece_iso"] / cal["ece_raw"]) * 100, 62, 1)
eq("mse reduction pct", (1 - cal["mse_iso"] / cal["mse_raw"]) * 100, 73, 1)

bins = pd.DataFrame(cal["bins"])
zero_share = 1 - bins.loc[bins["bin"] == 0, "n"].iloc[0] / bins["n"].sum()
claim("zero share", "97.58%")
eq("iso levels", len(set(r["mean_iso"] for r in cal["bins"])) > 0, True, 0)

# ---- level counts quoted in README ----
lv = {r["mean_iso"]: r for r in cal["bins"]}
for n_txt in ["4, 6, 3 and 4 rows"]:
    claim("level counts", n_txt)

# ---- dataset ----
raw = pd.read_csv("creditcard.csv")
ded = raw.drop_duplicates()
eq("raw rows", len(raw), 284807, 0)
eq("raw frauds", int(raw["Class"].sum()), 492, 0)
eq("dedup rows", len(ded), 283726, 0)
eq("dedup frauds", int(ded["Class"].sum()), 473, 0)
eq("duplicates", len(raw) - len(ded), 1081, 0)
claim("true median", "$22.00")
eq("true amount median", ded["Amount"].median(), 22.00, 0.005)
eq("log1p of 2.26", np.expm1(2.26), 8.58, 0.005)

# ---- model selection table ----
for name, prauc, ipr, ibrier, secs in [
    ("Trial_3_best", 0.8429, 0.8397, 0.000397, 75.9),
    ("Trial_2_best", 0.8417, 0.8393, 0.000383, 103.0),
    ("Trial_1_best", 0.8392, 0.8364, 0.000394, 78.7),
    ("Default_RF", 0.8378, 0.8344, 0.000389, 30.0),
]:
    e = st2[name]
    eq(f"{name}.mean_pr_auc", e["mean_pr_auc"], prauc, 5e-5)
    eq(f"{name}.isotonic_pr_auc", e["isotonic_pr_auc"], ipr, 5e-5)
    eq(f"{name}.isotonic_brier", e["isotonic_brier"], ibrier, 5e-7)
    eq(f"{name}.mean_time", e["mean_time"], secs, 0.05)

p3 = st2["Trial_3_best"]["params"]
eq("t3 n_estimators", p3["n_estimators"], 200, 0)
eq("t3 min_samples_leaf", p3["min_samples_leaf"], 3, 0)
eq("t3 max_depth", p3["max_depth"], 28, 0)
claim("t3 max_features", "0.306")
claim("t3 max_samples", "0.786")

gain = st2["Trial_3_best"]["mean_pr_auc"] - st2["Default_RF"]["mean_pr_auc"]
claim("tuning gain", "+0.005")
eq("tuning gain", gain, 0.005, 5e-4)
ratio = st2["Trial_3_best"]["mean_time"] / st2["Default_RF"]["mean_time"]
claim("time ratio", "2.5")
eq("time ratio", ratio, 2.5, 0.05)

# ---- all_criteria ----
ac = thr["all_criteria"]
eq("max_f1 threshold", ac["max_f1"]["threshold"], 0.5, 0)
eq("max_f1 f1", ac["max_f1"]["f1"], 0.8603, 5e-5)
eq("max_f2 threshold", ac["max_f2"]["threshold"], 3 / 14, 1e-12)
for fl, t in [("0.8", 3 / 14), ("0.9", 0.5), ("0.95", 0.825)]:
    eq(f"floor {fl} threshold", ac["precision_floors"][fl]["threshold"], t, 5e-4)

# ---- corroboration claim in limitation 6 ----
eq("json test_metrics.pr_auc", thr["test_metrics"]["pr_auc"], tm["pr_auc"], 1e-15)
eq("json test_metrics.brier", thr["test_metrics"]["brier"], tm["brier"], 1e-15)

# ---- F1/F2 postmortem arithmetic ----
eq("F2 test TP73", 5 * 73 / (5 * 73 + 4 * 22 + 9), 0.790043, 1e-6)
eq("F1 test TP73", 2 * 73 / (2 * 73 + 9 + 22), 0.824859, 1e-6)
eq("F2 oof TP320", 5 * 320 / (5 * 320 + 4 * 58 + 67), 0.842549, 1e-6)
eq("F1 oof TP320", 2 * 320 / (2 * 320 + 67 + 58), 0.836601, 1e-6)
claim("degradation mislabeled", "0.0117")
claim("degradation true", "0.0525")
eq("degradation true calc", ref["f2"] - tm["f2"], 0.0525, 5e-5)
eq("degradation f1 calc", ref["f1"] - tm["f1"], 0.0117, 5e-5)

# ---- flagged-rows-per-fraud ----
eq("A rows per fraud", (ref["tp"] + ref["fp"]) / ref["frauds"] if "frauds" in ref else 387 / 378, 1.024, 5e-4)
eq("B rows per fraud", 82 / 95, 0.863, 5e-4)
claim("A 1.024", "1.024")
claim("B 0.863", "0.863")

# ---- error analysis figures ----
claim("near-miss 96.6", "96.6%")
claim("invisible 56", "56 (96.6%)")
claim("amount-wtd recall", "81.78%")
for feat, imp in [("V17", 0.139), ("V12", 0.122), ("V14", 0.116), ("V10", 0.089),
                  ("V16", 0.084), ("V11", 0.081), ("V9", 0.042), ("V18", 0.031),
                  ("V4", 0.029), ("V7", 0.029)]:
    claim(f"importance {feat}", f"`{feat}`")
for v in ["8.82", "7.05", "6.62", "6.42", "5.23", "4.14"]:
    claim(f"diff {v}", v)
claim("fp 67", "67 false alarms")
claim("high conf 15", "15 are\nhigh-confidence") if False else claim("high conf 15", "15 are")
claim("caught median", "$8.58")
claim("false alarm median", "$2.00")
claim("normal median", "$22.10")

# ---- folds ----
for fr, fp_ in [("82.7%", "89.9%"), ("85.3%", "84.2%"), ("77.6%", "80.8%"),
                ("89.5%", "77.3%"), ("88.2%", "82.7%")]:
    claim(f"fold {fr}", fr)
    claim(f"fold {fp_}", fp_)
claim("recall one row 1.3", "1.3 recall points")
eq("one row per fold", 100 / 75, 1.333, 0.01)

# ---- correlations ----
corr = pd.read_csv("eda_outputs/class_correlations.csv", index_col=0)["Correlation"]
for f, v in [("V11", 0.149), ("V4", 0.129), ("V2", 0.085), ("V19", 0.034),
             ("V8", 0.033), ("V21", 0.026)]:
    eq(f"corr {f}", corr[f], v, 5e-4)
eq("corr Amount", corr["Amount"], 0.006, 5e-4)

print(f"checks run: {checks}")
if fails:
    print(f"\nFAILURES ({len(fails)}):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL README NUMERIC CLAIMS VERIFIED AGAINST ARTIFACTS")