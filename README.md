# 🩺 MLOps End-to-End Project — Diabetes Risk Predictor

An end-to-end MLOps pipeline for predicting diabetes risk using the Pima Indians dataset. The project covers the full ML lifecycle: data versioning, model training with hyperparameter tuning, experiment tracking, containerised deployment to AWS ECS, and live model monitoring with drift detection.

---

## 📐 Architecture Overview

![MLOps Architecture](mlops_architecture.png)

The system is composed of three main zones:

- **Training** — DVC pipeline stages (preprocess → train → evaluate → monitor), data versioned in S3, experiments tracked in MLflow
- **Monitoring** — Evidently AI generates data-drift and model-drift reports on every pipeline run and in real time inside the Flask app
- **Deployment** — Docker image pushed to AWS ECR, deployed as an ECS Fargate service behind an ALB, with Prometheus + Grafana for observability

---

## 🚀 Tech Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC + AWS S3 |
| Experiment tracking | MLflow (hosted on EC2) |
| Model | scikit-learn RandomForest + GridSearchCV |
| Drift monitoring | Evidently AI |
| Metrics / alerting | Prometheus + Grafana |
| API server | Flask + Gunicorn |
| Containerisation | Docker |
| Registry | AWS ECR |
| Orchestration | AWS ECS Fargate |
| CI/CD | GitHub Actions |

---

## 📁 Project Structure

```
├── src/
│   ├── preprocess.py      # Stage 1 – data cleaning
│   ├── train.py           # Stage 2 – RF training + MLflow logging
│   ├── evaluate.py        # Stage 3 – model evaluation
│   └── monitor.py         # Stage 4 – Evidently drift report
├── tests/
│   └── test_basic.py      # pytest suite (preprocess, model, data, Flask)
├── app.py                 # Flask API with live drift + Prometheus metrics
├── dvc.yaml               # DVC pipeline definition
├── dvc.lock               # Locked pipeline state
├── params.yaml            # Hyperparameters and paths
├── Dockerfile             # Container image
├── requirements.txt       # Python dependencies
└── .github/
    └── workflows/
        └── main.yml       # CI/CD pipeline
```

---

## ⚙️ Setup & Local Run

### Prerequisites

- Python 3.10+
- Docker
- AWS CLI configured (`ap-south-1`)
- DVC with S3 remote (`pip install dvc dvc-s3`)

### 1 — Clone and install dependencies

```bash
git clone https://github.com/<your-username>/manu7-final-mlops-end-to-end-project.git
cd manu7-final-mlops-end-to-end-project
pip install -r requirements.txt
```

### 2 — Pull data from DVC remote (S3)

```bash
dvc pull
```

### 3 — Run the full pipeline

```bash
dvc repro
```

This executes all four stages in order:

```
preprocess → train → evaluate → monitor
```

### 4 — Start the Flask API locally

```bash
python app.py
# or with gunicorn:
gunicorn --bind 0.0.0.0:5001 --workers 2 app:app
```

Open `http://localhost:5001` in your browser to use the prediction UI.

---

## 🔄 DVC Pipeline & Data Versioning

![DVC Workflow](dvc_workflow.png)

DVC tracks data files and model artifacts using pointer files (`.dvc`) committed to Git, while the actual bytes live in S3. `dvc.yaml` defines the four pipeline stages and their dependencies, so `dvc repro` only reruns stages whose inputs have changed.

**Key DVC commands:**

| Command | Purpose |
|---|---|
| `dvc init` | Set up DVC in the repo |
| `dvc add data/raw/data.csv` | Start tracking a data file |
| `dvc push` | Upload artifacts to S3 |
| `dvc pull` | Download artifacts from S3 |
| `dvc repro` | Re-run changed pipeline stages |

---

## 🧪 Running Tests

```bash
# Set MLflow to local filesystem so tests don't need network access
export MLFLOW_TRACKING_URI=file:///tmp/mlruns

pytest tests/ -v
```

The test suite covers:
- **Preprocess** — output file creation, row count preservation, missing-input error
- **Model** — training, binary predictions, pickle round-trip
- **Data** — CSV schema, column presence, null checks
- **Flask** — home page, health endpoint, JSON predict, form predict, 400 on missing field

---

## 🤖 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Prediction UI (HTML form) |
| `POST` | `/predict` | Predict — accepts JSON or form data |
| `GET` | `/health` | Health check (`{"status": "ok"}`) |
| `GET` | `/drift-report` | Latest Evidently HTML drift report |
| `GET` | `/drift-summary` | Latest Evidently JSON summary |
| `GET` | `/metrics` | Prometheus scrape endpoint |

### Example JSON request

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35
  }'
```

```json
{"prediction": 0, "label": "Non-Diabetic"}
```

---

## 🔁 CI/CD Pipeline

![CI/CD Pipeline](cicd_pipeline.png)

The GitHub Actions workflow (`main.yml`) has two jobs:

**Job 1 — `test`** (runs on every push and pull request)
- Installs dependencies
- Runs `pytest tests/ -v` with `MLFLOW_TRACKING_URI=file:///tmp/mlruns` (no EC2 network call)

**Job 2 — `train-evaluate`** (runs on push to `main` only, after tests pass)
1. Pulls data from DVC/S3 (`dvc pull`)
2. Runs the full pipeline (`dvc repro`)
3. Uploads Evidently HTML + JSON reports as CI artifacts
4. **Drift gate** — fails the build if `drift_share > 0.5` (more than 50% of columns drifted)
5. Pushes updated `dvc.lock` back to the repo
6. Builds Docker image → pushes to AWS ECR (tagged with Git SHA + `latest`)
7. Forces a new ECS Fargate deployment
8. Verifies the Prometheus scrape target is healthy

### Required GitHub secrets

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM key with ECR + ECS permissions |
| `AWS_SECRET_ACCESS_KEY` | IAM secret |
| `MLFLOW_TRACKING_URI` | `http://<ec2-ip>:5000` |
| `EC2_PUBLIC_IP` | Public IP of EC2 Prometheus instance |

---

## 📊 Monitoring & Observability

![AWS Monitoring Architecture](aws_monitoring.png)

### Evidently AI — drift detection

- **Offline** (`monitor.py`): runs as a DVC stage on every pipeline execution, compares training reference split vs current split, saves `reports/data_drift_report.html` and `reports/evidently_summary.json`, logs metrics to MLflow.
- **Live** (`app.py`): accumulates incoming predictions in a rolling buffer; every 50 predictions a background thread runs a fresh Evidently report and pushes drift scores to Prometheus Gauges.

### Prometheus metrics (exposed at `/metrics`)

| Metric | Type | Description |
|---|---|---|
| `diabetes_predictions_total` | Counter | Total predictions, labelled by result |
| `diabetes_prediction_latency_seconds` | Histogram | End-to-end prediction latency |
| `diabetes_model_loaded` | Gauge | 1 if model is loaded |
| `diabetes_input_glucose` | Histogram | Distribution of Glucose input values |
| `evidently_drift_share` | Gauge | Share of columns with detected drift |
| `evidently_glucose_drift` | Gauge | 1 if Glucose drift detected |
| `evidently_bmi_drift` | Gauge | 1 if BMI drift detected |
| `evidently_prediction_drift` | Gauge | 1 if prediction drift detected |

### Infrastructure

- **Flask app** runs on ECS Fargate (port 5001), fronted by an ALB
- **Prometheus** (port 9090) runs on EC2 and scrapes `/metrics` from the ALB
- **Grafana** (port 3000) queries Prometheus and renders dashboards
- Security group rules: EC2 → ALB outbound 5001; ALB → EC2 security group inbound

---

## 🐳 Docker

```bash
# Build locally
docker build -t diabetes-predictor .

# Run locally
docker run -p 5001:5001 \
  -e MLFLOW_TRACKING_URI=http://<ec2-ip>:5000 \
  diabetes-predictor
```

The Dockerfile uses `python:3.10-slim`, installs `gunicorn`, and copies `src/`, `models/`, `params.yaml`, and `app.py`.

---

## 🧑‍💼 MLflow Experiment Tracking

All training runs are logged under the experiment `manu7-mlops`:

- **Params**: `best_n_estimators`, `best_max_depth`, `best_min_samples_split`, `best_min_samples_leaf`
- **Metrics**: `accuracy` (train), `accuracy` (evaluate), Evidently drift metrics
- **Artifacts**: `confusion_matrix.txt`, `classification_report.txt`, Evidently HTML + JSON reports, registered model `Best Model`

Access the MLflow UI at `http://<ec2-ip>:5000`.

---

## 📈 Model Details

- **Dataset**: Pima Indians Diabetes (768 rows, 8 features)
- **Algorithm**: Random Forest Classifier
- **Tuning**: GridSearchCV with 3-fold CV over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Target**: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://13.234.38.124:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `manu7-mlops` | MLflow experiment name |
| `REFERENCE_DATA_PATH` | `data/raw/data.csv` | Path to training CSV for Evidently reference |

---

## 📄 License

MIT License — feel free to use and extend this project.

---

*Built by Manu7 · MLOps end-to-end portfolio project*
