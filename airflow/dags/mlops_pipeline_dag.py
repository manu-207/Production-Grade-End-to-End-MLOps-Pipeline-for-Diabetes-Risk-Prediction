"""
mlops_pipeline_dag.py
======================
Master orchestration DAG — runs the complete end-to-end MLOps pipeline
by triggering data_quality_check → model_retrain_dag in sequence.

This is the DAG you should monitor in production. It gives you a single
view of the entire weekly pipeline run.

Tasks:
  1. trigger_data_quality  — trigger data_quality_check DAG and wait
  2. trigger_model_retrain — trigger model_retrain_dag and wait
  3. pipeline_success      — log final success summary

Schedule: every Monday 00:00 UTC (same as model_retrain_dag but acts
as the parent orchestrator)
"""

from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    "owner": "manu7",
    "depends_on_past": False,
    "retries": 0,                   # master DAG does not retry — child DAGs handle retries
    "email_on_failure": False,
}

with DAG(
    dag_id="mlops_pipeline_dag",
    description="Master orchestration: data quality → model retrain → deploy",
    schedule_interval="0 0 * * 1",  # every Monday 00:00 UTC
    start_date=datetime(2026, 5, 14),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["mlops", "orchestration", "diabetes"],
) as dag:

    # ── Task 1: Trigger data quality check and wait for completion ────────────
    trigger_data_quality = TriggerDagRunOperator(
        task_id="trigger_data_quality",
        trigger_dag_id="data_quality_check",
        wait_for_completion=True,       # block until data_quality_check finishes
        poke_interval=30,               # check status every 30 seconds
        failed_states=["failed"],
        reset_dag_run=True,
        conf={"triggered_by": "mlops_pipeline_dag"},
    )

    # ── Task 2: Trigger model retraining and wait for completion ──────────────
    trigger_model_retrain = TriggerDagRunOperator(
        task_id="trigger_model_retrain",
        trigger_dag_id="model_retrain_dag",
        wait_for_completion=True,       # block until full retrain + deploy finishes
        poke_interval=60,               # check every 60 seconds (training takes time)
        failed_states=["failed"],
        reset_dag_run=True,
        conf={"triggered_by": "mlops_pipeline_dag"},
    )

    # ── Task 3: Log pipeline completion summary ───────────────────────────────
    def pipeline_success_summary(**context):
        run_id = context["run_id"]
        print("=" * 60)
        print("MLOps Pipeline completed successfully!")
        print("=" * 60)
        print(f"  DAG run ID   : {run_id}")
        print(f"  Completed at : {datetime.utcnow().isoformat()} UTC")
        print()
        print("  Steps completed:")
        print("    ✓ data_quality_check  — raw data validated")
        print("    ✓ model_retrain_dag   — model retrained + deployed to ECS")
        print()
        print("  Next run: next Monday 00:00 UTC")
        print("=" * 60)

    pipeline_success = PythonOperator(
        task_id="pipeline_success",
        python_callable=pipeline_success_summary,
        provide_context=True,
    )

    # ── Chain ─────────────────────────────────────────────────────────────────
    trigger_data_quality >> trigger_model_retrain >> pipeline_success
