# Airflow Setup — Diabetes MLOps Project

## Folder structure

```
airflow/
├── .env.example              ← copy to .env and fill your secrets
├── Dockerfile.airflow        ← custom image with all ML packages pre-installed
├── docker-compose.yml        ← runs webserver + scheduler + postgres
├── requirements-airflow.txt  ← ML packages baked into the image
├── README.md                 ← this file
└── dags/
    ├── __init__.py
    ├── data_quality_dag.py   ← validates raw data before training
    ├── model_retrain_dag.py  ← full retrain + deploy pipeline
    └── mlops_pipeline_dag.py ← master DAG that triggers both in sequence
```

## 3 DAGs explained

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `data_quality_check` | Sunday 23:30 UTC | Validates row count, columns, nulls, class balance |
| `model_retrain_dag` | Monday 00:00 UTC | dvc pull → preprocess → train → evaluate → monitor → drift gate → dvc push → ECS deploy |
| `mlops_pipeline_dag` | Monday 00:00 UTC | Master DAG — triggers both above in sequence |

## Quick start

```bash
# 1. Go to the airflow/ folder
cd airflow/

# 2. Copy .env and fill in your secrets
cp .env.example .env
nano .env

# 3. Build the custom Docker image (includes all ML packages)
docker-compose build

# 4. Initialize the database (first time only)
docker-compose run --rm airflow-init

# 5. Start Airflow
docker-compose up -d airflow-webserver airflow-scheduler

# 6. Open UI
# http://YOUR_EC2_IP:8080  (admin / admin)
```

## Set Airflow Variables (UI → Admin → Variables)

| Key | Value |
|-----|-------|
| `AWS_ACCESS_KEY_ID` | your AWS key |
| `AWS_SECRET_ACCESS_KEY` | your AWS secret |
| `MLFLOW_TRACKING_URI` | `http://YOUR_MLFLOW_IP:5000` |

## Useful commands

```bash
# View scheduler logs
docker logs $(docker ps -qf name=scheduler) -f

# Trigger pipeline manually
docker exec -it $(docker ps -qf name=scheduler) \
  airflow dags trigger mlops_pipeline_dag

# Check for DAG import errors
docker exec -it $(docker ps -qf name=scheduler) \
  airflow dags list-import-errors

# Stop everything
docker-compose down
```
