"""
data_quality_dag.py
====================
Runs BEFORE model retraining. Validates that the incoming data
meets minimum quality standards so we never train on bad data.

Schedule: runs 30 minutes before model_retrain_dag (every Monday 00:00 UTC)
so retraining always has fresh, validated data available.

Tasks:
  1. check_file_exists   — confirm data/raw/data.csv is present
  2. check_row_count     — fail if fewer than 100 rows
  3. check_columns       — fail if any required column is missing
  4. check_nulls         — fail if null % in critical columns > 20%
  5. check_class_balance — warn if class imbalance > 80/20
  6. generate_dq_report  — write reports/data_quality_report.json
"""

from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import json
import pandas as pd

PROJECT_DIR = "/opt/airflow/project"
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "raw", "data.csv")
REPORT_DIR  = os.path.join(PROJECT_DIR, "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "data_quality_report.json")

REQUIRED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
CRITICAL_COLUMNS = ["Glucose", "BMI", "BloodPressure", "Outcome"]
MIN_ROW_COUNT    = 100
MAX_NULL_PCT     = 0.20    # 20%
MAX_CLASS_RATIO  = 0.80    # warn if majority class > 80%


default_args = {
    "owner": "manu7",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}

with DAG(
    dag_id="data_quality_check",
    description="Validate raw data quality before model retraining",
    schedule_interval="30 23 * * 0",   # Sunday 23:30 UTC (30 min before retraining)
    start_date=datetime(2026, 5, 14),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["mlops", "data-quality", "diabetes"],
) as dag:

    # ── Task 1: confirm data file exists ──────────────────────────────────────
    def check_file_exists():
        print(f"[DQ] Checking file: {DATA_PATH}")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Data file not found: {DATA_PATH}\n"
                "Run dvc pull first or check your S3 remote."
            )
        size_kb = os.path.getsize(DATA_PATH) / 1024
        print(f"[DQ] File exists — size: {size_kb:.1f} KB")

    task_file_exists = PythonOperator(
        task_id="check_file_exists",
        python_callable=check_file_exists,
    )

    # ── Task 2: check minimum row count ──────────────────────────────────────
    def check_row_count():
        df = pd.read_csv(DATA_PATH)
        n  = len(df)
        print(f"[DQ] Row count: {n}")
        if n < MIN_ROW_COUNT:
            raise ValueError(
                f"Too few rows: {n} (minimum required: {MIN_ROW_COUNT}). "
                "Dataset may be corrupted or incomplete."
            )
        print(f"[DQ] Row count OK: {n} >= {MIN_ROW_COUNT}")

    task_row_count = PythonOperator(
        task_id="check_row_count",
        python_callable=check_row_count,
    )

    # ── Task 3: check all required columns are present ────────────────────────
    def check_columns():
        df      = pd.read_csv(DATA_PATH)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Found columns: {list(df.columns)}"
            )
        print(f"[DQ] All {len(REQUIRED_COLUMNS)} required columns present.")

    task_columns = PythonOperator(
        task_id="check_columns",
        python_callable=check_columns,
    )

    # ── Task 4: check null percentage in critical columns ─────────────────────
    def check_nulls():
        df     = pd.read_csv(DATA_PATH)
        errors = []
        for col in CRITICAL_COLUMNS:
            null_pct = df[col].isnull().mean()
            print(f"[DQ] {col} null%: {null_pct:.2%}")
            if null_pct > MAX_NULL_PCT:
                errors.append(f"{col}: {null_pct:.2%} nulls (max {MAX_NULL_PCT:.0%})")
        if errors:
            raise ValueError(f"Null threshold exceeded:\n" + "\n".join(errors))
        print("[DQ] Null checks passed for all critical columns.")

    task_nulls = PythonOperator(
        task_id="check_nulls",
        python_callable=check_nulls,
    )

    # ── Task 5: check class balance ───────────────────────────────────────────
    def check_class_balance():
        df         = pd.read_csv(DATA_PATH)
        counts     = df["Outcome"].value_counts(normalize=True)
        majority   = counts.max()
        print(f"[DQ] Class distribution: {counts.to_dict()}")
        if majority > MAX_CLASS_RATIO:
            # Warning only — do not fail the DAG, just log it
            print(
                f"[DQ] WARNING: Class imbalance detected. "
                f"Majority class = {majority:.1%} (threshold = {MAX_CLASS_RATIO:.0%}). "
                "Consider resampling before training."
            )
        else:
            print(f"[DQ] Class balance OK: majority class = {majority:.1%}")

    task_class_balance = PythonOperator(
        task_id="check_class_balance",
        python_callable=check_class_balance,
    )

    # ── Task 6: write data quality report JSON ────────────────────────────────
    def generate_dq_report():
        df = pd.read_csv(DATA_PATH)
        os.makedirs(REPORT_DIR, exist_ok=True)

        report = {
            "timestamp":      datetime.utcnow().isoformat(),
            "data_path":      DATA_PATH,
            "row_count":      len(df),
            "column_count":   len(df.columns),
            "columns":        list(df.columns),
            "null_counts":    df.isnull().sum().to_dict(),
            "null_pct":       (df.isnull().mean() * 100).round(2).to_dict(),
            "class_balance":  df["Outcome"].value_counts().to_dict(),
            "status":         "passed",
        }

        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[DQ] Report saved: {REPORT_PATH}")
        print(json.dumps(report, indent=2))

    task_dq_report = PythonOperator(
        task_id="generate_dq_report",
        python_callable=generate_dq_report,
    )

    # ── Chain ─────────────────────────────────────────────────────────────────
    (
        task_file_exists
        >> task_row_count
        >> task_columns
        >> task_nulls
        >> task_class_balance
        >> task_dq_report
    )
