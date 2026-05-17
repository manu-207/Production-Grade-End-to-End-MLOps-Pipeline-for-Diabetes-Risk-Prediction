import pandas as pd
import pickle
import os
import json
import yaml
import mlflow
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

# ── Evidently 0.7.x correct imports ───────────────────────────────────────────
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ── MLflow setup ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise EnvironmentError("MLFLOW_TRACKING_URI env var is not set")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

params = yaml.safe_load(open("params.yaml"))["train"]

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def run_monitoring(data_path: str, model_path: str, report_dir: str = "reports"):
    """
    1. Loads the dataset, splits into reference (70%) / current (30%).
    2. Adds model predictions to both splits (to track prediction drift).
    3. Runs Evidently report with DriftedColumnsCount + ValueDrift + DataDriftPreset.
    4. Saves HTML report  -> reports/data_drift_report.html
    5. Saves JSON summary -> reports/evidently_summary.json
    6. Logs all metrics + artifacts to MLflow under run 'evidently-monitoring'.
    """

    # ── Load data & model ─────────────────────────────────────────────────────
    data  = pd.read_csv(data_path)
    X     = data[FEATURES]
    y     = data["Outcome"]
    model = pickle.load(open(model_path, "rb"))

    X_ref, X_cur, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)

    X_ref = X_ref.copy()
    X_cur = X_cur.copy()

    X_ref["prediction"] = model.predict(X_ref)
    X_cur["prediction"] = model.predict(X_cur)

    # ── Build Evidently Report (0.7.x API) ────────────────────────────────────
    report = Report(metrics=[
        DriftedColumnsCount(),
        ValueDrift(column="Glucose"),
        ValueDrift(column="BMI"),
        ValueDrift(column="Age"),
        ValueDrift(column="prediction"),
        DataDriftPreset(),
        DataSummaryPreset(),
    ])
    snapshot = report.run(reference_data=X_ref, current_data=X_cur)

    # ── Save HTML report ──────────────────────────────────────────────────────
    os.makedirs(report_dir, exist_ok=True)
    html_path = os.path.join(report_dir, "data_drift_report.html")
    snapshot.save_html(html_path)
    print(f"[Evidently] HTML report saved -> {html_path}")

    # ── Extract metrics from Snapshot using .as_dict() ───────────────────────
    # evidently 0.7.x: snapshot.as_dict() returns the full results as a dict
    drift_share      = 0.0
    glucose_drift    = 0
    bmi_drift        = 0
    age_drift        = 0
    prediction_drift = 0

    try:
        result_dict = snapshot.as_dict()
        metrics_list = result_dict.get("metrics", [])

        for metric in metrics_list:
            metric_id = str(metric.get("metric", ""))
            result    = metric.get("result", {})

            # DriftedColumnsCount → share_drifted_features
            if "DriftedColumnsCount" in metric_id:
                share = result.get("share_drifted_features", None)
                if share is not None:
                    drift_share = float(share)

            # ValueDrift per column → drift_detected (bool)
            elif "ValueDrift" in metric_id:
                col      = result.get("column_name", "")
                detected = int(result.get("drift_detected", False))
                if col == "Glucose":    glucose_drift    = detected
                elif col == "BMI":      bmi_drift        = detected
                elif col == "Age":      age_drift        = detected
                elif col == "prediction": prediction_drift = detected

    except Exception as e:
        print(f"[Evidently] WARNING: Could not extract detailed metrics: {e}")
        print("[Evidently] Using default values (drift_share=0)")

    summary = {
        "drift_share":      round(drift_share, 4),
        "glucose_drift":    glucose_drift,
        "bmi_drift":        bmi_drift,
        "age_drift":        age_drift,
        "prediction_drift": prediction_drift,
    }

    # ── Save JSON summary ─────────────────────────────────────────────────────
    json_path = os.path.join(report_dir, "evidently_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Evidently] JSON summary  saved -> {json_path}")
    print(f"[Evidently] Metrics: {summary}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="evidently-monitoring"):
        mlflow.log_metric("evidently_drift_share",      summary["drift_share"])
        mlflow.log_metric("evidently_glucose_drift",    summary["glucose_drift"])
        mlflow.log_metric("evidently_bmi_drift",        summary["bmi_drift"])
        mlflow.log_metric("evidently_age_drift",        summary["age_drift"])
        mlflow.log_metric("evidently_prediction_drift", summary["prediction_drift"])
        mlflow.log_artifact(html_path, artifact_path="evidently")
        mlflow.log_artifact(json_path, artifact_path="evidently")
        print("[Evidently] Metrics + artifacts logged to MLflow")

    return summary


if __name__ == "__main__":
    summary = run_monitoring(
        data_path  = params["data"],
        model_path = params["model"],
    )
