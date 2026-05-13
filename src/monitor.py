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

# ── MLflow setup (same pattern as train.py) ───────────────────────────────────
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
    # NOTE: report.run() returns a Snapshot object in evidently >= 0.6
    report = Report(metrics=[
        DriftedColumnsCount(),          # overall drift share across all columns
        ValueDrift(column="Glucose"),   # KS p-value for Glucose
        ValueDrift(column="BMI"),
        ValueDrift(column="Age"),
        ValueDrift(column="prediction"),
        DataDriftPreset(),              # per-column drift table
        DataSummaryPreset(),            # data quality summary
    ])
    snapshot = report.run(reference_data=X_ref, current_data=X_cur)

    # ── Save HTML report ──────────────────────────────────────────────────────
    os.makedirs(report_dir, exist_ok=True)
    html_path = os.path.join(report_dir, "data_drift_report.html")
    snapshot.save_html(html_path)
    print(f"[Evidently] HTML report saved -> {html_path}")

    # ── Extract metrics from Snapshot.metric_results ──────────────────────────
    # DriftedColumnsCount: result.share.value = fraction of drifted columns
    # ValueDrift:          result.value = KS p-value; drift if p < 0.05
    drift_share      = 0.0
    glucose_drift    = 0
    bmi_drift        = 0
    age_drift        = 0
    prediction_drift = 0

    for _, result in snapshot.metric_results.items():
        name = result.display_name

        if hasattr(result, "share") and "Count of Drifted" in name:
            drift_share = float(result.share.value)

        elif hasattr(result, "value") and "Value drift for" in name:
            p_value  = float(result.value)
            detected = int(p_value < 0.05)
            if "Glucose"     in name: glucose_drift    = detected
            elif "BMI"       in name: bmi_drift        = detected
            elif "Age"       in name: age_drift        = detected
            elif "prediction" in name: prediction_drift = detected

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
