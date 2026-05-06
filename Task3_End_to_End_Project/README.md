# Task 3 — End-to-End Data Science Project
## Customer Churn Predictor — Flask Web App with SHAP Explainability
**CodTech IT Solutions | Data Science Internship**

---

## Overview

A complete end-to-end data science project that predicts whether a telecom customer will churn (cancel their service). Covers the full lifecycle: data collection → preprocessing → feature engineering → model training → serialisation → Flask deployment with a live web UI and REST API.

Uses a **Gradient Boosting Classifier** (GBC) with **SHAP explainability** — so the app doesn't just predict churn probability, it tells you *why* in plain language (top 3 contributing risk factors per customer).

---

## Dataset

| Property | Value |
|---|---|
| Name | Telco Customer Churn |
| Source | [Kaggle — IBM Watson Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows | 7,043 customers |
| Features | 21 columns (demographics, services, billing) |
| Target | Churn: Yes / No |

> The script auto-generates realistic synthetic data if the CSV is not present.

---

## Project Structure

```
Task3_Churn_Predictor/
├── train_model.py          ← Run FIRST: trains model, saves artifacts, generates plots
├── app.py                  ← Run SECOND: Flask web app + REST API
├── requirements.txt
├── README.md
├── model_artifacts/        ← Created by train_model.py
│   ├── model.pkl           ← Trained GBC model
│   ├── preprocessor.pkl    ← Fitted sklearn Pipeline
│   └── schema.json         ← Feature schema + sample input
└── output_t3/              ← Created by train_model.py
    ├── 00_eda_overview.png
    ├── 01_feature_importance.png
    ├── 02_shap_importance.png (or 02_feature_impact.png)
    ├── 03_confusion_roc.png
    ├── 04_churn_probability_dist.png
    └── 05_business_impact.png
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Download real dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in the same folder. If skipped, synthetic data is used automatically.

### Step 3 — Train the model
```bash
python train_model.py
```
This takes ~30–60 seconds. It will:
- Load / synthesise the dataset
- Engineer features and fit the preprocessing pipeline
- Train a GradientBoostingClassifier (300 trees, max_depth=4)
- Run 5-fold cross-validation
- Generate 6 visualisation PNGs → `output_t3/`
- Save `model.pkl`, `preprocessor.pkl`, `schema.json` → `model_artifacts/`

### Step 4 — Launch the web app
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI — fill form, get prediction |
| `/predict` | POST | JSON API — returns churn probability + top risk factors |
| `/health` | GET | Model status check |
| `/sample` | GET | Returns a sample valid input JSON |

### Example API call (curl)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.20
  }'
```

### Example API response
```json
{
  "churn_probability": 0.7834,
  "prediction": "Churn",
  "risk_level": "High",
  "top_risk_factors": [
    {"feature": "Contract_Month-to-month", "importance": 0.1821},
    {"feature": "tenure",                  "importance": 0.1456},
    {"feature": "MonthlyCharges",           "importance": 0.1203}
  ],
  "confidence": 0.7834,
  "monthly_revenue_at_risk": 79.85
}
```

---

## Feature Engineering

| Feature | Source | Why it matters |
|---|---|---|
| `tenure_monthly_ratio` | tenure / MonthlyCharges | Low ratio → relatively new customer paying a lot |
| `log_total_charges` | log1p(TotalCharges) | Compresses right skew of billing history |
| OneHotEncoder on all categorical | Contract, PaymentMethod, etc. | Required for tree models |
| StandardScaler on numeric | tenure, charges | Normalisation for convergence |

---

## Output Files from train_model.py

| File | Description |
|---|---|
| `00_eda_overview.png` | 4-panel EDA: churn rate, tenure KDE, monthly charges KDE, churn by contract |
| `01_feature_importance.png` | Top 20 feature importances from Gradient Boosting |
| `02_shap_importance.png` | SHAP beeswarm plot (if shap installed) |
| `03_confusion_roc.png` | Normalised confusion matrix + ROC curve with AUC |
| `04_churn_probability_dist.png` | KDE of predicted probabilities by true label |
| `05_business_impact.png` | Revenue saved vs retention cost by threshold |

---

## Requirements

```
flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.43.0        # optional — fallback to GB importances if absent
```
