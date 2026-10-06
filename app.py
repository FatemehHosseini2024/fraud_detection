"""
Streamlit UI for Fraud Detection - Transaction Analysis

Allows user to select a transaction from test set and see:
- Model prediction (fraud/legit)
- Calibrated probability of fraud
- Actual label
- Feature analysis for the transaction
"""

import json
import os

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import StratifiedKFold

from pipeline import prepare_data
from mlflow_helpers import start_run

RANDOM_STATE = 42
N_FOLDS = 5
N_BOOTSTRAP = 2000
N_BINS = 10

MODEL_NAME = "fraud_rf_default"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

mlflow.set_tracking_uri(TRACKING_URI)


def make_rf():
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)


@st.cache_resource
def load_model_artifacts():
    """Train/fetch the final model artifacts (RF, isotonic calibrator, threshold)."""
    df, train, test, _ = prepare_data()
    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]
    X_test = test.drop(columns=["Class"])
    y_test = test["Class"].to_numpy()

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_raw_proba = []
    all_y_val = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        rf = make_rf()
        rf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        all_raw_proba.append(rf.predict_proba(X_train.iloc[val_idx])[:, 1])
        all_y_val.append(y_train.iloc[val_idx].to_numpy())

    oof_raw = np.concatenate(all_raw_proba)
    oof_y = np.concatenate(all_y_val)

    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(oof_raw, oof_y)
    oof_iso = final_iso.predict(oof_raw)

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

    rf_full = make_rf()
    rf_full.fit(X_train, y_train)

    test_raw = rf_full.predict_proba(X_test)[:, 1]
    test_iso = final_iso.predict(test_raw)
    test_pred = (test_iso >= LOCKED_THRESHOLD).astype(int)

    return {
        "rf_full": rf_full,
        "final_iso": final_iso,
        "LOCKED_THRESHOLD": LOCKED_THRESHOLD,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_raw": test_raw,
        "test_iso": test_iso,
        "test_pred": test_pred,
        "feature_names": X_train.columns.tolist(),
    }


def get_transaction_analysis(artifacts, idx):
    """Get detailed analysis for a single transaction."""
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    test_raw = artifacts["test_raw"]
    test_iso = artifacts["test_iso"]
    test_pred = artifacts["test_pred"]
    feature_names = artifacts["feature_names"]
    threshold = artifacts["LOCKED_THRESHOLD"]

    row = X_test.iloc[idx]
    actual = int(y_test[idx])
    raw_prob = float(test_raw[idx])
    iso_prob = float(test_iso[idx])
    pred = int(test_pred[idx])

    is_fraud = actual == 1
    is_flagged = pred == 1

    feature_vals = row.to_dict()

    feature_importance = pd.Series(
        artifacts["rf_full"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    top_features = feature_importance.head(10)

    return {
        "index": idx,
        "actual": actual,
        "predicted": pred,
        "raw_probability": raw_prob,
        "calibrated_probability": iso_prob,
        "threshold": threshold,
        "is_fraud": is_fraud,
        "is_flagged": is_flagged,
        "feature_values": feature_vals,
        "top_features": top_features.to_dict(),
    }


def main():
    st.set_page_config(
        page_title="Fraud Detection - Transaction Analysis",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Fraud Detection - Transaction Analysis")
    st.markdown(
        "Select a transaction from the test set to view the model's prediction, "
        "calibrated fraud probability, and feature-level analysis."
    )

    with st.spinner("Loading model artifacts..."):
        artifacts = load_model_artifacts()

    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    test_iso = artifacts["test_iso"]
    test_pred = artifacts["test_pred"]
    threshold = artifacts["LOCKED_THRESHOLD"]

    st.sidebar.header("📊 Model Summary")
    st.sidebar.metric("Test Set Size", len(X_test))
    st.sidebar.metric("Fraud Rate", f"{y_test.mean()*100:.4f}%")
    st.sidebar.metric("Locked Threshold", f"{threshold:.6f}")
    st.sidebar.metric("Flagged Transactions", int(test_pred.sum()))
    st.sidebar.metric("Actual Frauds in Test", int(y_test.sum()))

    st.sidebar.divider()
    st.sidebar.header("🎯 Select Transaction")

    fraud_indices = np.where(y_test == 1)[0]
    legit_indices = np.where(y_test == 0)[0]

    view_mode = st.sidebar.radio(
        "Filter by:",
        ["All Transactions", "Fraud Only", "Legit Only", "Flagged by Model", "Correctly Classified", "Misclassified"],
    )

    if view_mode == "All Transactions":
        available_indices = np.arange(len(X_test))
    elif view_mode == "Fraud Only":
        available_indices = fraud_indices
    elif view_mode == "Legit Only":
        available_indices = legit_indices
    elif view_mode == "Flagged by Model":
        available_indices = np.where(test_pred == 1)[0]
    elif view_mode == "Correctly Classified":
        available_indices = np.where(test_pred == y_test)[0]
    else:
        available_indices = np.where(test_pred != y_test)[0]

    if len(available_indices) == 0:
        st.sidebar.warning("No transactions match this filter.")
        return

    selected_idx = st.sidebar.selectbox(
        "Transaction Index",
        options=available_indices,
        format_func=lambda i: f"#{i} | Actual: {'Fraud' if y_test[i]==1 else 'Legit'} | Pred: {'Fraud' if test_pred[i]==1 else 'Legit'} | p={test_iso[i]:.4f}",
    )

    analysis = get_transaction_analysis(artifacts, selected_idx)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Actual Label", "🔴 Fraud" if analysis["actual"] == 1 else "🟢 Legitimate")
    with col2:
        st.metric("Model Prediction", "🔴 Fraud" if analysis["predicted"] == 1 else "🟢 Legitimate")
    with col3:
        status = "✅ Correct" if analysis["actual"] == analysis["predicted"] else "❌ Incorrect"
        st.metric("Classification", status)

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📈 Probability Scores")
        st.metric("Raw RF Probability", f"{analysis['raw_probability']:.6f}")
        st.metric("Calibrated (Isotonic) Probability", f"{analysis['calibrated_probability']:.6f}")
        st.metric("Decision Threshold", f"{analysis['threshold']:.6f}")

        fig_data = pd.DataFrame({
            "Type": ["Raw RF", "Calibrated", "Threshold"],
            "Value": [
                analysis["raw_probability"],
                analysis["calibrated_probability"],
                analysis["threshold"],
            ],
        })
        st.bar_chart(fig_data.set_index("Type"))

    with col_right:
        st.subheader("🎯 Prediction Logic")
        if analysis["calibrated_probability"] >= analysis["threshold"]:
            st.success(f"✅ **FLAGGED AS FRAUD** (p = {analysis['calibrated_probability']:.6f} ≥ {analysis['threshold']:.6f})")
        else:
            st.info(f"✅ **NOT FLAGGED** (p = {analysis['calibrated_probability']:.6f} < {analysis['threshold']:.6f})")

        st.markdown("---")
        st.markdown("**Confusion Matrix Context**")
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Legit", "Actual: Fraud"],
            columns=["Pred: Legit", "Pred: Fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.divider()
    st.subheader("🔬 Feature Analysis for This Transaction")

    feature_df = pd.DataFrame({
        "Feature": list(analysis["feature_values"].keys()),
        "Value": list(analysis["feature_values"].values()),
    })
    feature_df["Abs_Value"] = feature_df["Value"].abs()
    feature_df = feature_df.sort_values("Abs_Value", ascending=False).drop("Abs_Value", axis=1)

    st.dataframe(feature_df, use_container_width=True, height=400)

    st.subheader("📊 Top 10 Most Important Features (Global)")
    importance_df = pd.DataFrame({
        "Feature": list(analysis["top_features"].keys()),
        "Importance": list(analysis["top_features"].values()),
    })
    st.bar_chart(importance_df.set_index("Feature"))

    with st.expander("📋 Full Transaction Data (JSON)"):
        st.json({
            "index": analysis["index"],
            "actual": int(analysis["actual"]),
            "predicted": int(analysis["predicted"]),
            "raw_probability": analysis["raw_probability"],
            "calibrated_probability": analysis["calibrated_probability"],
            "threshold": analysis["threshold"],
            "feature_values": {k: float(v) for k, v in analysis["feature_values"].items()},
        })


if __name__ == "__main__":
    main()