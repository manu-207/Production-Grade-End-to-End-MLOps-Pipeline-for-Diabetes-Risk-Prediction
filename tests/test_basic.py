import os
import pickle
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────
#  SAMPLE DATA  (same columns as Pima dataset)
# ─────────────────────────────────────────────

SAMPLE_DATA = {
    "Pregnancies": [6, 1, 8, 1, 0],
    "Glucose":     [148, 85, 183, 89, 137],
    "BloodPressure":[72, 66, 64, 66, 40],
    "SkinThickness":[35, 29, 0, 23, 35],
    "Insulin":     [0, 0, 0, 94, 168],
    "BMI":         [33.6, 26.6, 23.3, 28.1, 43.1],
    "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288],
    "Age":         [50, 31, 32, 21, 33],
    "Outcome":     [1, 0, 1, 0, 1],
}


# ─────────────────────────────────────────────
#  1. PREPROCESS TESTS
# ─────────────────────────────────────────────

def test_preprocess_creates_output_file(tmp_path):
    """Output CSV file must be created after preprocessing."""
    input_file  = tmp_path / "input.csv"
    output_file = tmp_path / "processed" / "output.csv"
    pd.DataFrame(SAMPLE_DATA).to_csv(input_file, index=False)

    os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"
    with patch("yaml.safe_load", return_value={
        "preprocess": {"input": str(input_file), "output": str(output_file)}
    }):
        from src.preprocess import preprocess
        preprocess(str(input_file), str(output_file))

    assert os.path.exists(output_file)


def test_preprocess_keeps_same_row_count(tmp_path):
    """Row count must be the same before and after preprocessing."""
    input_file  = tmp_path / "input.csv"
    output_file = tmp_path / "processed" / "output.csv"
    df = pd.DataFrame(SAMPLE_DATA)
    df.to_csv(input_file, index=False)

    os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"
    with patch("yaml.safe_load", return_value={
        "preprocess": {"input": str(input_file), "output": str(output_file)}
    }):
        from src.preprocess import preprocess
        preprocess(str(input_file), str(output_file))

    result = pd.read_csv(output_file, header=None)
    assert len(result) == len(df)


def test_preprocess_missing_input_raises_error(tmp_path):
    """Must raise FileNotFoundError if input CSV does not exist."""
    os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"
    with patch("yaml.safe_load", return_value={
        "preprocess": {"input": "x.csv", "output": "y.csv"}
    }):
        from src.preprocess import preprocess
        with pytest.raises(FileNotFoundError):
            preprocess(str(tmp_path / "no_file.csv"), str(tmp_path / "out.csv"))


# ─────────────────────────────────────────────
#  2. MODEL TESTS
# ─────────────────────────────────────────────

def test_model_trains_without_error():
    """RandomForestClassifier must fit without raising any exception."""
    df = pd.DataFrame(SAMPLE_DATA)
    X  = df.drop(columns=["Outcome"])
    y  = df["Outcome"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    assert model is not None


def test_model_predictions_are_0_or_1():
    """Model must only predict 0 (Non-Diabetic) or 1 (Diabetic)."""
    df = pd.DataFrame(SAMPLE_DATA)
    X  = df.drop(columns=["Outcome"])
    y  = df["Outcome"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)
    for pred in predictions:
        assert pred in [0, 1]


def test_model_saves_and_loads_correctly(tmp_path):
    """Model saved with pickle must load back as a RandomForestClassifier."""
    df = pd.DataFrame(SAMPLE_DATA)
    X  = df.drop(columns=["Outcome"])
    y  = df["Outcome"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    pickle.dump(model, open(model_path, "wb"))

    loaded = pickle.load(open(model_path, "rb"))
    assert isinstance(loaded, RandomForestClassifier)


def test_loaded_model_predicts_correctly(tmp_path):
    """Loaded model must produce same predictions as original model."""
    df = pd.DataFrame(SAMPLE_DATA)
    X  = df.drop(columns=["Outcome"])
    y  = df["Outcome"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    original_preds = model.predict(X)

    model_path = tmp_path / "model.pkl"
    pickle.dump(model, open(model_path, "wb"))
    loaded = pickle.load(open(model_path, "rb"))

    assert list(loaded.predict(X)) == list(original_preds)


# ─────────────────────────────────────────────
#  3. DATA TESTS
# ─────────────────────────────────────────────

def test_csv_loads_with_correct_shape(tmp_path):
    """CSV must load with correct number of rows and columns."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame(SAMPLE_DATA).to_csv(csv_file, index=False)

    df = pd.read_csv(csv_file)
    assert df.shape == (5, 9)   # 5 rows, 9 columns


def test_outcome_column_exists(tmp_path):
    """Outcome column must be present in the dataset."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame(SAMPLE_DATA).to_csv(csv_file, index=False)

    df = pd.read_csv(csv_file)
    assert "Outcome" in df.columns


def test_outcome_column_is_binary(tmp_path):
    """Outcome column must only contain 0 and 1."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame(SAMPLE_DATA).to_csv(csv_file, index=False)

    df = pd.read_csv(csv_file)
    assert set(df["Outcome"].unique()).issubset({0, 1})


def test_no_null_values_in_glucose_and_outcome(tmp_path):
    """Glucose and Outcome columns must have no missing values."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame(SAMPLE_DATA).to_csv(csv_file, index=False)

    df = pd.read_csv(csv_file)
    assert df["Glucose"].isnull().sum() == 0
    assert df["Outcome"].isnull().sum() == 0


def test_all_required_columns_present(tmp_path):
    """All 8 feature columns + Outcome must be present."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame(SAMPLE_DATA).to_csv(csv_file, index=False)

    df = pd.read_csv(csv_file)
    required = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


# ─────────────────────────────────────────────
#  4. FLASK APP TESTS
# ─────────────────────────────────────────────

@pytest.fixture
def client():
    """Flask test client with a tiny mock model injected."""
    # Build a real tiny model so predict() works
    df = pd.DataFrame(SAMPLE_DATA)
    X  = df.drop(columns=["Outcome"])
    y  = df["Outcome"]
    mock_model = RandomForestClassifier(n_estimators=3, random_state=42)
    mock_model.fit(X, y)

    import app as flask_app
    flask_app.model = mock_model          # inject model directly
    flask_app.app.config["TESTING"] = True

    with flask_app.app.test_client() as c:
        yield c


def test_home_page_returns_200(client):
    """Home page must return HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_home_page_has_form(client):
    """Home page must contain an HTML form."""
    response = client.get("/")
    assert b"<form" in response.data


def test_health_endpoint_returns_ok(client):
    """Health check endpoint must return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert b"ok" in response.data


def test_predict_json_returns_200(client):
    """JSON predict request with valid data must return 200."""
    payload = {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 80, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 35
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_json_has_prediction_and_label(client):
    """JSON response must contain prediction and label keys."""
    import json
    payload = {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 80, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 35
    }
    response = client.post("/predict", json=payload)
    data = json.loads(response.data)
    assert "prediction" in data
    assert "label" in data


def test_predict_label_is_diabetic_or_non_diabetic(client):
    """Label must be either Diabetic or Non-Diabetic."""
    import json
    payload = {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 80, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 35
    }
    response = client.post("/predict", json=payload)
    data = json.loads(response.data)
    assert data["label"] in ["Diabetic", "Non-Diabetic"]


def test_predict_missing_field_returns_400(client):
    """Missing a required field must return HTTP 400."""
    # Glucose is missing
    payload = {
        "Pregnancies": 2, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 80, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 35
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
