import pickle
import time
import json
import os
import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

# ── Evidently 0.7.x correct imports ───────────────────────────────────────────
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset

app = Flask(__name__)

# ── FIX 1: Guard all Prometheus metric registrations against duplicate
#           registration when pytest imports app.py multiple times.
# ─────────────────────────────────────────────────────────────────────────────
if not hasattr(app, '_metrics_registered'):
    metrics = PrometheusMetrics(app)
    metrics.info("app_info", "Diabetes Predictor", version="1.0.0")

    # Existing Prometheus metrics
    PREDICTION_COUNTER = Counter(
        "diabetes_predictions_total", "Total predictions", ["result"]
    )
    PREDICTION_LATENCY = Histogram(
        "diabetes_prediction_latency_seconds", "Prediction latency",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    MODEL_LOADED = Gauge("diabetes_model_loaded", "1 if model is loaded")
    GLUCOSE_HISTOGRAM = Histogram(
        "diabetes_input_glucose", "Glucose input distribution",
        buckets=[50, 80, 100, 120, 140, 160, 180, 200, 250]
    )

    # Prometheus metrics for Evidently drift scores
    DRIFT_SHARE       = Gauge("evidently_drift_share",       "Share of drifted columns")
    GLUCOSE_DRIFT     = Gauge("evidently_glucose_drift",     "1 if Glucose drift detected")
    BMI_DRIFT         = Gauge("evidently_bmi_drift",         "1 if BMI drift detected")
    PREDICTION_DRIFT  = Gauge("evidently_prediction_drift",  "1 if prediction drift detected")
    DRIFT_WINDOW_SIZE = Gauge("evidently_drift_window_size", "Predictions in current window")

    # ── FIX: Initialize all Evidently gauges to 0 so they appear in /metrics
    #         immediately. Without this, Grafana shows "no data" because
    #         Prometheus never sees these metrics until 50 predictions trigger
    #         the first drift check.
    DRIFT_SHARE.set(0)
    GLUCOSE_DRIFT.set(0)
    BMI_DRIFT.set(0)
    PREDICTION_DRIFT.set(0)
    DRIFT_WINDOW_SIZE.set(0)

    app._metrics_registered = True

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
DRIFT_CHECK_EVERY = 50   # run drift check every N predictions

FEATURE_ICONS = {
    "Pregnancies": "🤰", "Glucose": "🍬", "BloodPressure": "💓",
    "SkinThickness": "🩹", "Insulin": "💉", "BMI": "⚖️",
    "DiabetesPedigreeFunction": "🧬", "Age": "🎂"
}

HOME = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DiabetesAI Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --coral: #FF6B6B; --orange: #FF8E53; --yellow: #FFD93D;
            --teal: #4ECDC4; --purple: #A855F7; --navy: #0F172A;
            --card-bg: #1E293B; --border: rgba(255,255,255,0.08);
            --text: #F1F5F9; --muted: #94A3B8;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'DM Sans', sans-serif; background: var(--navy); color: var(--text); min-height: 100vh; overflow-x: hidden; }
        body::before, body::after { content: ''; position: fixed; border-radius: 50%; filter: blur(80px); opacity: 0.18; pointer-events: none; z-index: 0; animation: drift 12s ease-in-out infinite alternate; }
        body::before { width: 600px; height: 600px; background: radial-gradient(circle, var(--coral), var(--purple)); top: -150px; left: -150px; }
        body::after { width: 500px; height: 500px; background: radial-gradient(circle, var(--teal), var(--yellow)); bottom: -100px; right: -100px; animation-delay: -6s; }
        @keyframes drift { from { transform: translate(0,0) scale(1); } to { transform: translate(40px, 30px) scale(1.1); } }
        .page-wrapper { position: relative; z-index: 1; max-width: 780px; margin: 0 auto; padding: 48px 24px 80px; }
        header { text-align: center; margin-bottom: 48px; animation: fadeDown 0.6s ease both; }
        .badge { display: inline-block; background: linear-gradient(135deg, var(--coral), var(--purple)); color: #fff; font-size: 11px; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; padding: 6px 16px; border-radius: 999px; margin-bottom: 18px; }
        header h1 { font-family: 'Syne', sans-serif; font-size: clamp(2.4rem, 6vw, 3.8rem); font-weight: 800; line-height: 1.1; background: linear-gradient(135deg, var(--coral) 0%, var(--yellow) 50%, var(--teal) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 12px; }
        header p { color: var(--muted); font-size: 1rem; font-weight: 300; max-width: 420px; margin: 0 auto; line-height: 1.6; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 24px; padding: 40px; backdrop-filter: blur(12px); animation: fadeUp 0.7s ease 0.15s both; box-shadow: 0 32px 80px rgba(0,0,0,0.4); }
        .card-title { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 28px; display: flex; align-items: center; gap: 10px; }
        .card-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }
        .fields-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px; }
        @media (max-width: 520px) { .fields-grid { grid-template-columns: 1fr; } .card { padding: 24px; } }
        .field { position: relative; }
        .field label { display: flex; align-items: center; gap: 7px; font-size: 0.78rem; font-weight: 500; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .field input { width: 100%; background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 12px; padding: 13px 16px; color: var(--text); font-family: 'DM Sans', sans-serif; font-size: 0.95rem; transition: border-color 0.2s, box-shadow 0.2s, background 0.2s; outline: none; }
        .field input:focus { border-color: var(--teal); background: rgba(78,205,196,0.06); box-shadow: 0 0 0 3px rgba(78,205,196,0.12); }
        .field input::placeholder { color: #475569; }
        .btn-wrap { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
        button[type="submit"] { position: relative; overflow: hidden; background: linear-gradient(135deg, var(--coral), var(--purple)); color: #fff; border: none; border-radius: 14px; padding: 16px 56px; font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; letter-spacing: 0.5px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 8px 32px rgba(255,107,107,0.35); }
        button[type="submit"]:hover { transform: translateY(-2px); box-shadow: 0 16px 48px rgba(255,107,107,0.5); }
        .btn-report { background: linear-gradient(135deg, var(--teal), var(--purple)); color: #fff; border: none; border-radius: 14px; padding: 16px 32px; font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-block; box-shadow: 0 8px 32px rgba(78,205,196,0.3); transition: transform 0.2s; }
        .btn-report:hover { transform: translateY(-2px); }
        .result-banner { margin-top: 28px; border-radius: 16px; padding: 24px 28px; display: flex; align-items: center; gap: 18px; animation: popIn 0.4s cubic-bezier(0.34,1.56,0.64,1) both; }
        .result-banner.diabetic { background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(168,85,247,0.12)); border: 1px solid rgba(255,107,107,0.35); }
        .result-banner.non-diabetic { background: linear-gradient(135deg, rgba(78,205,196,0.15), rgba(255,217,61,0.10)); border: 1px solid rgba(78,205,196,0.35); }
        .result-icon { font-size: 2.4rem; line-height: 1; }
        .result-text h2 { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800; margin-bottom: 4px; }
        .result-banner.diabetic .result-text h2 { color: var(--coral); }
        .result-banner.non-diabetic .result-text h2 { color: var(--teal); }
        .result-text p { color: var(--muted); font-size: 0.88rem; }
        .dots { display: flex; justify-content: center; gap: 8px; margin-top: 48px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse 2s ease-in-out infinite; }
        .dot:nth-child(1) { background: var(--coral); animation-delay: 0s; }
        .dot:nth-child(2) { background: var(--yellow); animation-delay: 0.3s; }
        .dot:nth-child(3) { background: var(--teal); animation-delay: 0.6s; }
        .dot:nth-child(4) { background: var(--purple); animation-delay: 0.9s; }
        @keyframes pulse { 0%,100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(1.5); opacity: 1; } }
        @keyframes fadeDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes popIn { from { opacity: 0; transform: scale(0.92); } to { opacity: 1; transform: scale(1); } }
    </style>
</head>
<body>
<div class="page-wrapper">
    <header>
        <div class="badge">✦ AI Health Tool</div>
        <h1>Diabetes Risk<br>Predictor</h1>
        <p>Enter your health metrics below and let our AI model assess your diabetes risk instantly.</p>
    </header>
    <div class="card">
        <div class="card-title">📋 Health Parameters</div>
        <form action="/predict" method="POST">
            <div class="fields-grid">
                {% set icons = {"Pregnancies": "🤰", "Glucose": "🍬", "BloodPressure": "💓", "SkinThickness": "🩹", "Insulin": "💉", "BMI": "⚖️", "DiabetesPedigreeFunction": "🧬", "Age": "🎂"} %}
                {% for feature in features %}
                <div class="field">
                    <label><span class="icon">{{ icons[feature] }}</span>{{ feature }}</label>
                    <input type="text" name="{{ feature }}" placeholder="Enter value" required>
                </div>
                {% endfor %}
            </div>
            <div class="btn-wrap">
                <button type="submit">⚡ Analyse Now</button>
                <a href="/drift-report" class="btn-report">📊 Drift Report</a>
            </div>
        </form>
        {% if result %}
        {% if result == "Diabetic" %}
        <div class="result-banner diabetic">
            <div class="result-icon">⚠️</div>
            <div class="result-text"><h2>{{ result }}</h2><p>High risk detected. Please consult your healthcare provider.</p></div>
        </div>
        {% else %}
        <div class="result-banner non-diabetic">
            <div class="result-icon">✅</div>
            <div class="result-text"><h2>{{ result }}</h2><p>Low risk detected. Keep maintaining a healthy lifestyle!</p></div>
        </div>
        {% endif %}
        {% endif %}
    </div>
    <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
</div>
</body>
</html>
"""

# ── Global state ───────────────────────────────────────────────────────────────
model          = None
reference_data = None      # training CSV loaded once at startup
prediction_log = []        # rolling buffer of incoming feature dicts
drift_lock     = threading.Lock()


def load_model():
    global model
    if model is None:
        model = pickle.load(open("models/model.pkl", "rb"))
        MODEL_LOADED.set(1)
    return model


def load_reference_data():
    """Load training CSV once at startup — used as Evidently reference dataset."""
    global reference_data
    csv_path = os.getenv("REFERENCE_DATA_PATH", "data/raw/data.csv")
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        reference_data = data[FEATURES].copy()
        print(f"[Evidently] Reference data loaded: {len(reference_data)} rows")
    else:
        print(f"[Evidently] WARNING: reference data not found at {csv_path}")


def run_live_drift_check():
    """
    Background thread: runs every DRIFT_CHECK_EVERY predictions.
    Uses evidently 0.7.x API: report.run() returns Snapshot.
    Pushes drift scores to Prometheus Gauges.
    Saves live HTML report to reports/live_drift_report.html.
    """
    global prediction_log

    with drift_lock:
        if reference_data is None or len(prediction_log) < DRIFT_CHECK_EVERY:
            return
        current_df = pd.DataFrame(
            prediction_log[-DRIFT_CHECK_EVERY:], columns=FEATURES
        )

    try:
        report = Report(metrics=[
            DriftedColumnsCount(),
            ValueDrift(column="Glucose"),
            ValueDrift(column="BMI"),
            ValueDrift(column="prediction") if "prediction" in current_df.columns else ValueDrift(column="Age"),
            DataDriftPreset(),
        ])

        # Add predictions column to current data
        current_with_pred = current_df.copy()
        m = load_model()
        current_with_pred["prediction"] = m.predict(current_df)

        ref_with_pred = reference_data.copy()
        ref_with_pred["prediction"] = m.predict(reference_data)

        snapshot = report.run(reference_data=ref_with_pred, current_data=current_with_pred)

        # ── Push to Prometheus ────────────────────────────────────────────────
        for _, result in snapshot.metric_results.items():
            name = result.display_name

            if hasattr(result, "share") and "Count of Drifted" in name:
                share = float(result.share.value)
                DRIFT_SHARE.set(share)
                print(f"[Evidently] live drift_share={share:.3f}")

            elif hasattr(result, "value") and "Value drift for" in name:
                p_value  = float(result.value)
                detected = int(p_value < 0.05)
                if "Glucose"      in name: GLUCOSE_DRIFT.set(detected)
                elif "BMI"        in name: BMI_DRIFT.set(detected)
                elif "prediction" in name: PREDICTION_DRIFT.set(detected)

        DRIFT_WINDOW_SIZE.set(len(prediction_log))

        # ── Save live HTML report ─────────────────────────────────────────────
        os.makedirs("reports", exist_ok=True)
        snapshot.save_html("reports/live_drift_report.html")
        print("[Evidently] Live report updated -> reports/live_drift_report.html")

    except Exception as e:
        print(f"[Evidently] Drift check error: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME, features=FEATURES)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/drift-report", methods=["GET"])
def drift_report():
    """Serves the latest Evidently HTML report directly in the browser."""
    for path in ["reports/live_drift_report.html", "reports/data_drift_report.html"]:
        if os.path.exists(path):
            with open(path) as f:
                return f.read(), 200, {"Content-Type": "text/html"}
    return (
        f"<h2 style='font-family:sans-serif;padding:40px'>📊 No drift report yet."
        f"<br><small>Make at least {DRIFT_CHECK_EVERY} predictions to generate a live report.</small></h2>",
        404,
    )


@app.route("/drift-summary", methods=["GET"])
def drift_summary():
    """Returns latest Evidently JSON summary — useful for alerting."""
    json_path = "reports/evidently_summary.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            return jsonify(json.load(f)), 200
    return jsonify({"error": "No summary available yet"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    m     = load_model()
    start = time.time()

    if request.is_json:
        data = request.get_json()
        try:
            vals = np.array([float(data[f]) for f in FEATURES]).reshape(1, -1)
        except KeyError as e:
            return jsonify({"error": f"Missing field: {e}"}), 400

        prediction = int(m.predict(vals)[0])
        label      = "Diabetic" if prediction == 1 else "Non-Diabetic"

        PREDICTION_LATENCY.observe(time.time() - start)
        PREDICTION_COUNTER.labels(result="diabetic" if prediction == 1 else "non_diabetic").inc()
        GLUCOSE_HISTOGRAM.observe(float(data.get("Glucose", 0)))
        _log_prediction(data)

        return jsonify({"prediction": prediction, "label": label})

    else:
        try:
            vals = np.array([float(request.form[f]) for f in FEATURES]).reshape(1, -1)
        except (KeyError, ValueError) as e:
            return f"Error: {e}", 400

        prediction = int(m.predict(vals)[0])
        label      = "Diabetic" if prediction == 1 else "Non-Diabetic"

        PREDICTION_LATENCY.observe(time.time() - start)
        PREDICTION_COUNTER.labels(result="diabetic" if prediction == 1 else "non_diabetic").inc()
        GLUCOSE_HISTOGRAM.observe(float(request.form.get("Glucose", 0)))
        _log_prediction(request.form)

        return render_template_string(HOME, features=FEATURES, result=label)


def _log_prediction(source):
    """Appends incoming feature values to buffer; triggers drift check every N."""
    global prediction_log
    try:
        row = {f: float(source[f]) for f in FEATURES}
        with drift_lock:
            prediction_log.append(row)
            count = len(prediction_log)
        if count % DRIFT_CHECK_EVERY == 0:
            t = threading.Thread(target=run_live_drift_check, daemon=True)
            t.start()
    except Exception as e:
        print(f"[Evidently] Prediction log error: {e}")


# ── Startup ────────────────────────────────────────────────────────────────────
# FIX 2: Removed load_model() from here.
#         It was called at import time, crashing pytest because models/model.pkl
#         does not exist in CI. The test fixture injects the model directly via
#         flask_app.model = mock_model, so load_model() at startup is not needed.
#         load_reference_data() is safe — it only prints a warning if CSV missing.
with app.app_context():
    load_reference_data()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
