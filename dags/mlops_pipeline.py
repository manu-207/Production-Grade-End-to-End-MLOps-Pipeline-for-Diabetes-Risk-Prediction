from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json, os

PROJECT_DIR = "/opt/airflow/project"   # where your repo is mounted in Airflow

default_args = {
    "owner": "manu7",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="mlops_diabetes_retraining",
    description="Weekly retraining pipeline for diabetes model",
    schedule_interval="@weekly",          # every Monday at midnight
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "diabetes"],
) as dag:

    # ── Task 1: pull latest data from S3 via DVC ──────────────────────────
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"cd {PROJECT_DIR} && dvc pull",
        env={
            "AWS_ACCESS_KEY_ID":     "{{ var.value.AWS_ACCESS_KEY_ID }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.AWS_SECRET_ACCESS_KEY }}",
            "AWS_DEFAULT_REGION":    "ap-south-1",
        },
    )

    # ── Task 2: preprocess ────────────────────────────────────────────────
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && python src/preprocess.py",
    )

    # ── Task 3: train + log to MLflow ────────────────────────────────────
    train = BashOperator(
        task_id="train",
        bash_command=f"cd {PROJECT_DIR} && python src/train.py",
        env={
            "MLFLOW_TRACKING_URI":     "{{ var.value.MLFLOW_TRACKING_URI }}",
            "MLFLOW_EXPERIMENT_NAME":  "manu7-mlops",
        },
    )

    # ── Task 4: evaluate ─────────────────────────────────────────────────
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"cd {PROJECT_DIR} && python src/evaluate.py",
        env={
            "MLFLOW_TRACKING_URI": "{{ var.value.MLFLOW_TRACKING_URI }}",
        },
    )

    # ── Task 5: drift monitoring via Evidently ────────────────────────────
    monitor = BashOperator(
        task_id="monitor",
        bash_command=f"cd {PROJECT_DIR} && python src/monitor.py",
        env={
            "MLFLOW_TRACKING_URI": "{{ var.value.MLFLOW_TRACKING_URI }}",
        },
    )

    # ── Task 6: check drift threshold — fail DAG if too high ─────────────
    def check_drift():
        report_path = f"{PROJECT_DIR}/reports/evidently_summary.json"
        with open(report_path) as f:
            summary = json.load(f)
        drift = summary.get("drift_share", 0)
        print(f"Drift share: {drift:.2%}")
        if drift > 0.5:
            raise ValueError(f"Drift too high ({drift:.2%}) — blocking deployment!")
        print("Drift acceptable — proceeding to deploy.")

    drift_gate = PythonOperator(
        task_id="drift_gate",
        python_callable=check_drift,
    )

    # ── Task 7: push updated model/dvc.lock back to S3 ───────────────────
    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=f"cd {PROJECT_DIR} && dvc push",
        env={
            "AWS_ACCESS_KEY_ID":     "{{ var.value.AWS_ACCESS_KEY_ID }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.AWS_SECRET_ACCESS_KEY }}",
            "AWS_DEFAULT_REGION":    "ap-south-1",
        },
    )

    # ── Task 8: redeploy ECS with the new model ───────────────────────────
    deploy = BashOperator(
        task_id="deploy_to_ecs",
        bash_command=(
            "aws ecs update-service "
            "--cluster manu7-mlops-cluster "
            "--service manu7-mlops-task-service2 "
            "--force-new-deployment "
            "--region ap-south-1"
        ),
        env={
            "AWS_ACCESS_KEY_ID":     "{{ var.value.AWS_ACCESS_KEY_ID }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.AWS_SECRET_ACCESS_KEY }}",
            "AWS_DEFAULT_REGION":    "ap-south-1",
        },
    )

    # ── DAG dependency chain ──────────────────────────────────────────────
    dvc_pull >> preprocess >> train >> evaluate >> monitor >> drift_gate >> dvc_push >> deploy
