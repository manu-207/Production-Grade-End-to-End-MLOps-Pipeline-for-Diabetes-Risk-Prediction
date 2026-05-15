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

# ── Shared env for tasks that need AWS + MLflow ───────────────────────────────
# append_env=True (set per-task below) MERGES this dict with the container's
# existing environment instead of replacing it, so PATH / HOME / etc. survive.
# AWS_* and MLFLOW_* are already injected by docker-compose from the .env file;
# we re-state them here only as an explicit, readable record of what each task
# actually needs.  Jinja var.value templates are intentionally NOT used because
# BashOperator does NOT render Jinja inside env= dicts — the literal template
# string would be passed to the shell, not the resolved secret value.
AWS_ENV = {
    "AWS_ACCESS_KEY_ID":     os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "AWS_DEFAULT_REGION":    os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
    # Ensure the airflow user's local bin (where dvc / aws CLI live) is on PATH
    "PATH": "/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin",
}

MLFLOW_ENV = {
    "MLFLOW_TRACKING_URI":    os.environ.get("MLFLOW_TRACKING_URI", ""),
    "MLFLOW_EXPERIMENT_NAME": os.environ.get("MLFLOW_EXPERIMENT_NAME", "manu7-mlops"),
    "PATH": "/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin",
}

with DAG(
    dag_id="mlops_diabetes_retraining",
    description="Weekly retraining pipeline for diabetes model",
    schedule_interval="*/10 * * * *",
    start_date=datetime(2026, 5, 13),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "diabetes"],
) as dag:

    # ── Task 1: pull latest data from S3 via DVC ──────────────────────────
    # FIX 1: append_env=True so the container PATH is preserved and dvc is found.
    # FIX 2: env= reads from os.environ (injected by docker-compose) instead of
    #         Jinja var.value templates, which are NOT rendered in env= dicts.
    # FIX 3: added --force so a dirty working directory never blocks the pull.
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"cd {PROJECT_DIR} && dvc pull --force",
        env=AWS_ENV,
        append_env=True,   # merge with container env, not replace
    )

    # ── Task 2: preprocess ────────────────────────────────────────────────
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && python src/preprocess.py",
        append_env=True,
    )

    # ── Task 3: train + log to MLflow ────────────────────────────────────
    train = BashOperator(
        task_id="train",
        bash_command=f"cd {PROJECT_DIR} && python src/train.py",
        env=MLFLOW_ENV,
        append_env=True,
    )

    # ── Task 4: evaluate ─────────────────────────────────────────────────
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"cd {PROJECT_DIR} && python src/evaluate.py",
        env=MLFLOW_ENV,
        append_env=True,
    )

    # ── Task 5: drift monitoring via Evidently ────────────────────────────
    monitor = BashOperator(
        task_id="monitor",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "mkdir -p reports && "
            "python src/monitor.py"
        ),
        env=MLFLOW_ENV,
        append_env=True,
    )

    # ── Task 6: check drift threshold — fail DAG if too high ─────────────
    def check_drift():
        report_path = f"{PROJECT_DIR}/reports/evidently_summary.json"
        if not os.path.exists(report_path):
            raise FileNotFoundError(
                f"[drift_gate] evidently_summary.json not found at {report_path}\n"
                "The monitor task must complete successfully before drift_gate runs."
            )
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
        env=AWS_ENV,
        append_env=True,
    )

    # ── Task 8: redeploy ECS with the new model ───────────────────────────
    deploy = BashOperator(
        task_id="deploy_to_ecs",
        bash_command=(
            "aws ecs update-service "
            "--cluster ${ECS_CLUSTER} "
            "--service ${ECS_SERVICE} "
            "--force-new-deployment "
            "--region ${ECS_REGION}"
        ),
        env={
            **AWS_ENV,
            "ECS_CLUSTER": os.environ.get("ECS_CLUSTER", "manu7-mlops-cluster"),
            "ECS_SERVICE": os.environ.get("ECS_SERVICE", "manu7-mlops-task-service2"),
            "ECS_REGION":  os.environ.get("ECS_REGION",  "ap-south-1"),
        },
        append_env=True,
    )

    # ── DAG dependency chain ──────────────────────────────────────────────
    dvc_pull >> preprocess >> train >> evaluate >> monitor >> drift_gate >> dvc_push >> deploy
