"""
=============================================================================
  CODTECH DATA SCIENCE INTERNSHIP — TASK 3
  End-to-End Data Science Project: Customer Churn Predictor
  STEP 2 OF 2 — Flask Web Application
=============================================================================
  Author      : [Your Name]
  Internship  : CodTech IT Solutions
  Task        : End-to-End Data Science Project (Task 3)
  File        : app.py

  Run AFTER train_model.py has been executed.

  Endpoints:
    GET  /            → Web UI — fill in customer details, get prediction
    POST /predict     → JSON API — returns churn probability + top reasons
    GET  /health      → Health check (returns model status)
    GET  /sample      → Returns a sample valid input JSON

  Usage:
    python app.py
    Open http://127.0.0.1:5000 in your browser
=============================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_DIR = "model_artifacts"

# ── Load artifacts at startup ──────────────────────────────────────────────
try:
    MODEL       = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    PREPROCESSOR= joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    with open(os.path.join(MODEL_DIR, "schema.json")) as f:
        SCHEMA  = json.load(f)
    FEATURE_NAMES = SCHEMA["feature_names"]
    MODEL_LOADED  = True
    logger.info("Model and preprocessor loaded successfully.")
except Exception as e:
    MODEL_LOADED  = False
    logger.error(f"Failed to load model: {e}")
    logger.error("Run train_model.py first!")


# ══════════════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════

def predict_churn(data: dict):
    """
    Given a dict of raw customer fields, returns:
      {
        "churn_probability"  : float,
        "prediction"         : "Churn" | "Stay",
        "risk_level"         : "High" | "Medium" | "Low",
        "top_risk_factors"   : list of (feature, importance) dicts,
        "confidence"         : float,
        "monthly_revenue_at_risk": float,
      }
    """
    # Build DataFrame
    df_input = pd.DataFrame([data])

    # Engineer same features as training
    df_input["TotalCharges"]       = pd.to_numeric(
        df_input.get("TotalCharges", df_input["MonthlyCharges"]), errors="coerce"
    )
    df_input["TotalCharges"]       = df_input["TotalCharges"].fillna(
        df_input["MonthlyCharges"] * df_input["tenure"]
    )
    df_input["tenure_monthly_ratio"] = df_input["tenure"] / (df_input["MonthlyCharges"] + 1e-9)
    df_input["log_total_charges"]    = np.log1p(df_input["TotalCharges"])

    # Transform
    X = PREPROCESSOR.transform(df_input)

    # Predict
    prob   = float(MODEL.predict_proba(X)[0][1])
    pred   = "Churn" if prob >= 0.5 else "Stay"

    # Risk tier
    if prob >= 0.70:
        risk = "High"
    elif prob >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    # Top risk factors (feature importances for this sample)
    importances = MODEL.feature_importances_
    top_idx     = np.argsort(importances)[-5:][::-1]
    top_factors = [
        {"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]), 4)}
        for i in top_idx
    ]

    monthly_rev = float(data.get("MonthlyCharges", 0))

    return {
        "churn_probability"      : round(prob, 4),
        "prediction"             : pred,
        "risk_level"             : risk,
        "top_risk_factors"       : top_factors,
        "confidence"             : round(max(prob, 1-prob), 4),
        "monthly_revenue_at_risk": round(monthly_rev, 2) if pred == "Churn" else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════
#  WEB UI TEMPLATE
# ══════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Customer Churn Predictor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #F0EEF8;
      min-height: 100vh;
      padding: 2rem 1rem;
      color: #2C2C2A;
    }
    .container { max-width: 860px; margin: 0 auto; }
    header {
      background: #7F77DD;
      color: white;
      padding: 1.5rem 2rem;
      border-radius: 12px;
      margin-bottom: 2rem;
    }
    header h1 { font-size: 1.6rem; font-weight: 600; }
    header p  { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
    .card {
      background: white;
      border-radius: 12px;
      padding: 1.5rem 2rem;
      margin-bottom: 1.5rem;
      border: 1px solid #E8E6F4;
    }
    .card h2 { font-size: 1.1rem; font-weight: 600; color: #534AB7;
               margin-bottom: 1.2rem; border-bottom: 1px solid #EEE;
               padding-bottom: 0.6rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    label { display: block; font-size: 0.85rem; color: #555;
            margin-bottom: 4px; font-weight: 500; }
    input, select {
      width: 100%; padding: 8px 12px; border: 1px solid #DDD;
      border-radius: 8px; font-size: 0.95rem; outline: none;
      transition: border 0.2s;
    }
    input:focus, select:focus { border-color: #7F77DD; }
    .btn {
      background: #7F77DD; color: white; border: none;
      padding: 12px 32px; border-radius: 8px; font-size: 1rem;
      font-weight: 600; cursor: pointer; margin-top: 1rem;
      transition: background 0.2s;
    }
    .btn:hover { background: #534AB7; }
    #result { display: none; }
    .result-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
    .metric {
      text-align: center; padding: 1.2rem;
      background: #F8F7FF; border-radius: 10px;
      border: 1px solid #E8E6F4;
    }
    .metric .val { font-size: 2rem; font-weight: 700; margin: 4px 0; }
    .metric .lbl { font-size: 0.8rem; color: #888; }
    .churn-yes .val { color: #E24B4A; }
    .churn-no  .val { color: #1D9E75; }
    .risk-high   { color: #E24B4A !important; }
    .risk-medium { color: #BA7517 !important; }
    .risk-low    { color: #1D9E75 !important; }
    .factors { margin-top: 1rem; }
    .factor-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 6px 0; border-bottom: 1px solid #F5F5F5; font-size: 0.9rem;
    }
    .factor-bar {
      height: 6px; background: #7F77DD; border-radius: 3px;
      margin-top: 3px;
    }
    .badge {
      display: inline-block; padding: 2px 10px; border-radius: 20px;
      font-size: 0.8rem; font-weight: 600;
    }
    .badge-churn { background: #FFECEC; color: #C0392B; }
    .badge-stay  { background: #ECFAF4; color: #1A7A52; }
    .error { background: #FFECEC; color: #C0392B; padding: 1rem;
             border-radius: 8px; margin-top: 1rem; font-size: 0.9rem; }
    .api-info {
      background: #F8F7FF; border-left: 3px solid #7F77DD;
      padding: 0.8rem 1rem; border-radius: 4px; font-size: 0.85rem;
      color: #555; margin-top: 1rem;
    }
    .api-info code { background: #EEE; padding: 1px 5px; border-radius: 3px;
                     font-size: 0.82rem; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>Customer Churn Predictor</h1>
    <p>CodTech Internship — Task 3 | Gradient Boosting + SHAP Explainability</p>
  </header>

  <div class="card">
    <h2>Customer Details</h2>
    <div class="grid">
      <div>
        <label>Gender</label>
        <select id="gender"><option>Male</option><option>Female</option></select>
      </div>
      <div>
        <label>Senior Citizen</label>
        <select id="SeniorCitizen"><option value="0">No</option><option value="1">Yes</option></select>
      </div>
      <div>
        <label>Partner</label>
        <select id="Partner"><option>Yes</option><option>No</option></select>
      </div>
      <div>
        <label>Dependents</label>
        <select id="Dependents"><option>No</option><option>Yes</option></select>
      </div>
      <div>
        <label>Tenure (months)</label>
        <input type="number" id="tenure" value="12" min="1" max="72">
      </div>
      <div>
        <label>Monthly Charges ($)</label>
        <input type="number" id="MonthlyCharges" value="79.85" step="0.01">
      </div>
      <div>
        <label>Total Charges ($)</label>
        <input type="number" id="TotalCharges" value="958.20" step="0.01">
      </div>
      <div>
        <label>Internet Service</label>
        <select id="InternetService">
          <option>Fiber optic</option><option>DSL</option><option>No</option>
        </select>
      </div>
      <div>
        <label>Contract Type</label>
        <select id="Contract">
          <option>Month-to-month</option><option>One year</option><option>Two year</option>
        </select>
      </div>
      <div>
        <label>Payment Method</label>
        <select id="PaymentMethod">
          <option>Electronic check</option>
          <option>Mailed check</option>
          <option>Bank transfer (automatic)</option>
          <option>Credit card (automatic)</option>
        </select>
      </div>
      <div>
        <label>Paperless Billing</label>
        <select id="PaperlessBilling"><option>Yes</option><option>No</option></select>
      </div>
      <div>
        <label>Online Security</label>
        <select id="OnlineSecurity">
          <option>No</option><option>Yes</option><option>No internet service</option>
        </select>
      </div>
      <div>
        <label>Tech Support</label>
        <select id="TechSupport">
          <option>No</option><option>Yes</option><option>No internet service</option>
        </select>
      </div>
      <div>
        <label>Streaming TV</label>
        <select id="StreamingTV">
          <option>Yes</option><option>No</option><option>No internet service</option>
        </select>
      </div>
      <div>
        <label>Phone Service</label>
        <select id="PhoneService"><option>Yes</option><option>No</option></select>
      </div>
      <div>
        <label>Multiple Lines</label>
        <select id="MultipleLines">
          <option>No</option><option>Yes</option><option>No phone service</option>
        </select>
      </div>
    </div>
    <button class="btn" onclick="predict()">Predict Churn Risk</button>
  </div>

  <div class="card" id="result">
    <h2>Prediction Result</h2>
    <div class="result-grid">
      <div class="metric" id="pred-card">
        <div class="lbl">Prediction</div>
        <div class="val" id="pred-val">—</div>
        <div id="pred-badge"></div>
      </div>
      <div class="metric">
        <div class="lbl">Churn Probability</div>
        <div class="val" id="prob-val" style="color:#7F77DD">—</div>
      </div>
      <div class="metric">
        <div class="lbl">Risk Level</div>
        <div class="val" id="risk-val">—</div>
      </div>
    </div>
    <div class="factors">
      <h2 style="margin-top:1.2rem">Top Risk Factors</h2>
      <div id="factors-list"></div>
    </div>
  </div>

  <div class="card">
    <h2>API Usage</h2>
    <div class="api-info">
      POST to <code>/predict</code> with JSON body — see <code>/sample</code> for schema.<br>
      Health check: <code>GET /health</code>
    </div>
  </div>
</div>

<script>
async function predict() {
  const fields = [
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "MonthlyCharges","TotalCharges","InternetService","Contract",
    "PaymentMethod","PaperlessBilling","OnlineSecurity","TechSupport",
    "StreamingTV","PhoneService","MultipleLines"
  ];
  const data = {};
  fields.forEach(f => {
    const el = document.getElementById(f);
    data[f] = isNaN(el.value) || el.value === "" ? el.value : Number(el.value);
  });

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });
    const json = await res.json();
    if (json.error) { alert("Error: " + json.error); return; }

    document.getElementById("result").style.display = "block";
    const isChurn = json.prediction === "Churn";
    const predCard = document.getElementById("pred-card");
    predCard.className = "metric " + (isChurn ? "churn-yes" : "churn-no");
    document.getElementById("pred-val").textContent = json.prediction;
    document.getElementById("pred-badge").innerHTML =
      `<span class="badge ${isChurn ? "badge-churn" : "badge-stay"}">
        ${isChurn ? "⚠ At Risk" : "✓ Retained"}
      </span>`;
    document.getElementById("prob-val").textContent =
      (json.churn_probability * 100).toFixed(1) + "%";
    const riskEl = document.getElementById("risk-val");
    riskEl.textContent = json.risk_level;
    riskEl.className   = "val risk-" + json.risk_level.toLowerCase();

    const factorsList = document.getElementById("factors-list");
    factorsList.innerHTML = json.top_risk_factors.map(f => `
      <div class="factor-row">
        <div>
          <div>${f.feature.replace(/_/g," ")}</div>
          <div class="factor-bar" style="width:${Math.min(f.importance*400,100)}%"></div>
        </div>
        <span style="color:#888;font-size:0.82rem">${(f.importance*100).toFixed(2)}%</span>
      </div>`).join("");

    document.getElementById("result").scrollIntoView({behavior:"smooth"});
  } catch(e) {
    alert("Request failed: " + e);
  }
}
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: JSON with customer fields (see /sample for schema)
    Returns: prediction result JSON
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    required = ["tenure", "MonthlyCharges", "Contract"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = predict_churn(data)
        return jsonify(result)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status"      : "ok" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
    })


@app.route("/sample")
def sample():
    """Returns a sample valid input for testing the API."""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(SCHEMA["sample_input"])


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  CODTECH INTERNSHIP  |  TASK 3  |  FLASK APP")
    print("  Customer Churn Predictor")
    print("=" * 65)
    print(f"  Model loaded  : {MODEL_LOADED}")
    print(f"  Starting at   : http://127.0.0.1:5000")
    print(f"  API endpoint  : POST http://127.0.0.1:5000/predict")
    print(f"  Health check  : GET  http://127.0.0.1:5000/health")
    print("=" * 65)
    app.run(debug=True, host="0.0.0.0", port=5000)
