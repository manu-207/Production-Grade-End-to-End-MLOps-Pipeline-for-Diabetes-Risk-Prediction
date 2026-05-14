"""
model_retrain_dag.py
=====================
Main weekly retraining pipeline for the Diabetes Risk Prediction model.
Runs every Monday 00:00 UTC — 30 minutes AFTER data_quality_dag completes.

Tasks:
  1. dvc_pull        — fetch latest data + cached artifacts from S3
  2. preprocess      — raw CSV → processed CSV
  3. train           — GridSearchCV + MLflow logging + model.pkl
  4. evaluate        — accuracy score logged to MLflow
  5. monitor         — Evidently drift report + evidently_summary.json
  6. drift_gate      — block deploy if drift_share > 50%
  7. dvc_push        — push new model + dvc.lock back to S3
  8. deploy_to_ecs   — force new ECS deployment (model reloaded, no Docker build)
"""

from __future__ import annotations

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json
import os

PROJECT_DIR = "/opt/airflow/project"

# ── Shared environment for all BashOperators ──────────────────────────────────
# Values come from docker-compose .env → container environment
# Using os.environ.get with fallback so DAG parses cleanly even if vars missing
SHARED_ENV = {
    "AWS_ACCESS_KEY_ID":      "{{ var.value.get('AWS_ACCESS_KEY_ID', '') }}",
    "AWS_SECRET_ACCESS_KEY":  "{{ var.value.get('AWS_SECRET_ACCESS_KEY', '') }}",
    "AWS_DEFAULT_REGION":     os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
    "MLFLOW_TRACKING_URI":    "{{ var.value.get('MLFLOW_TRACKING_URI', '') }}",
    "MLFLOW_EXPERIMENT_NAME": os.environ.get("MLFLOW_EXPERIMENT_NAME", "manu7-mlops"),
    "PATH": "/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin",
}

default_args = {
    "owner": "manu7",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="model_retrain_dag",
    description="Weekly model retraining — diabetes risk prediction",
    schedule_interval="0 0 * * 1",    # every Monday 00:00 UTC
    start_date=datetime(2026, 5, 14),
    catchup=False,
    max_active_runs=1,                 # never run two retrains simultaneously
    default_args=default_args,
    tags=["mlops", "training", "diabetes"],
) as dag:

    # ── Task 1: pull latest data and cached model from S3 via DVC ─────────────
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"cd {PROJECT_DIR} && dvc pull --force",
        env=SHARED_ENV,
    )

    # ── Task 2: preprocess raw CSV → data/processed/data.csv ──────────────────
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && python src/preprocess.py",
        env=SHARED_ENV,
    )

    # ── Task 3: train RandomForest with GridSearchCV, log run to MLflow ────────
    train = BashOperator(
        task_id="train",
        bash_command=f"cd {PROJECT_DIR} && python src/train.py",
        env=SHARED_ENV,
    )

    # ── Task 4: evaluate model accuracy and log to MLflow ─────────────────────
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"cd {PROJECT_DIR} && python src/evaluate.py",
        env=SHARED_ENV,
    )

    # ── Task 5: run Evidently drift monitoring ─────────────────────────────────
    monitor = BashOperator(
        task_id="monitor",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "mkdir -p reports && "
            "python src/monitor.py"
        ),
        env=SHARED_ENV,
    )

    # ── Task 6: drift gate — block deployment if too many columns drifted ──────
    def check_drift(**context):
        report_path = os.path.join(PROJECT_DIR, "reports", "evidently_summary.json")

        if not os.path.exists(report_path):
            raise FileNotFoundError(
                f"[drift_gate] evidently_summary.json not found at {report_path}\n"
                "The monitor task must complete successfully before drift_gate."
            )

        with open(report_path) as f:
            summary = json.load(f)

        drift            = summary.get("drift_share", 0)
        glucose_drift    = summary.get("glucose_drift", 0)
        bmi_drift        = summary.get("bmi_drift", 0)
        prediction_drift = summary.get("prediction_drift", 0)

        print(f"[drift_gate] drift_share      = {drift:.2%}")
        print(f"[drift_gate] glucose_drift    = {glucose_drift}")
        print(f"[drift_gate] bmi_drift        = {bmi_drift}")
        print(f"[drift_gate] prediction_drift = {prediction_drift}")

        if drift > 0.5:
            raise ValueError(
                f"[drift_gate] BLOCKED: {drift:.2%} of columns drifted "
                f"(threshold = 50%). Fix data distribution before deploying."
            )

        print("[drift_gate] Drift within acceptable range — proceeding to deploy.")

    drift_gate = PythonOperator(
        task_id="drift_gate",
        python_callable=check_drift,
        provide_context=True,
    )

    # ── Task 7: push new model.pkl + dvc.lock back to S3 ──────────────────────
    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=f"cd {PROJECT_DIR} && dvc push",
        env=SHARED_ENV,
    )

    # ── Task 8: force new ECS deployment — same Docker image, fresh model ──────
    # ECS container restarts and reloads model.pkl from disk (mounted from S3/DVC)
    # No Docker build needed — only the model file changed, not the app code
    deploy_to_ecs = BashOperator(
        task_id="deploy_to_ecs",
        bash_command=(
            "aws ecs update-service "
            "--cluster ${ECS_CLUSTER} "
            "--service ${ECS_SERVICE} "
            "--force-new-deployment "
            "--region ${ECS_REGION}"
        ),
        env={
            **SHARED_ENV,
            "ECS_CLUSTER": os.environ.get("ECS_CLUSTER", "manu7-mlops-cluster"),
            "ECS_SERVICE": os.environ.get("ECS_SERVICE", "manu7-mlops-task-service2"),
            "ECS_REGION":  os.environ.get("ECS_REGION",  "ap-south-1"),
        },
    )

    # ── DAG dependency chain ───────────────────────────────────────────────────
    (
        dvc_pull
        >> preprocess
        >> train
        >> evaluate
        >> monitor
        >> drift_gate
        >> dvc_push
        >> deploy_to_ecs
    )
