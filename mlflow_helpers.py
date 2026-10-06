"""Shared MLflow helpers for the fraud-detection project.

Every script imports `start_run` from here instead of calling `mlflow.start_run()`
directly, so the tracking URI, experiment name and run naming convention live in one
place. Run with:

    MLFLOW_TRACKING_URI=file:./mlruns python <script>.py

or simply `python <script>.py` — the default is a local `mlruns/` directory, which is
what Streamlit Cloud needs (no server, no credentials, no network).
"""

import os

import mlflow

# Local file backend by default. Streamlit Cloud cannot reach a tracking server, and
# the file backend needs no credentials. Override with MLFLOW_TRACKING_URI.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "fraud_detection"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(run_name=None, tags=None):
    """Open an MLflow run. Returns the run context manager.

    Usage::

        with start_run("threshold_optimization"):
            ...
    """
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def log_json_artifact(obj, filename):
    """Dump a dict to a temp file and log it as an artifact."""
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(obj, f, indent=2)
        path = f.name
    mlflow.log_artifact(path, artifact_path="metrics")
    os.remove(path)
    return path


def log_dataframe(df, filename, artifact_path="metrics"):
    """Log a DataFrame as a CSV artifact."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as f:
        df.to_csv(f, index=False)
        path = f.name
    mlflow.log_artifact(path, artifact_path=artifact_path)
    os.remove(path)
    return path