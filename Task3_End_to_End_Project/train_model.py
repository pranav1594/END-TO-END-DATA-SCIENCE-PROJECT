"""
=============================================================================
  CODTECH DATA SCIENCE INTERNSHIP — TASK 3
  End-to-End Data Science Project: Customer Churn Predictor
  STEP 1 OF 2 — Model Training Script
=============================================================================
  Author      : [Your Name]
  Internship  : CodTech IT Solutions
  Task        : End-to-End Data Science Project (Task 3)
  File        : train_model.py

  Run this file FIRST. It:
    1. Loads / synthesises the Telco Customer Churn dataset
    2. Performs full EDA and preprocessing
    3. Trains a Gradient Boosting classifier (GradientBoostingClassifier)
       with SHAP explainability
    4. Saves the trained model to model_artifacts/
    5. Generates 6 EDA + evaluation visualisations to output_t3/

  Then run app.py to launch the Flask prediction API.
=============================================================================
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, f1_score,
)
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import OneHotEncoder

# ── SHAP (optional) ───────────────────────────────────────────────────────
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SEED        = 42
OUTPUT_DIR  = "output_t3"
MODEL_DIR   = "model_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────
C = {
    "churn"   : "#E24B4A",
    "stay"    : "#1D9E75",
    "accent"  : "#7F77DD",
    "blue"    : "#378ADD",
    "warn"    : "#BA7517",
    "bg"      : "#F8F7F4",
    "grid"    : "#E0DED9",
    "text"    : "#2C2C2A",
}
plt.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["bg"],
    "axes.edgecolor": C["grid"], "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load Telco Customer Churn dataset.
    Real dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    (WA_Fn-UseC_-Telco-Customer-Churn.csv — 7043 rows, 21 columns)

    Falls back to a realistic synthetic version if CSV is not present.
    """
    candidates = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_churn.csv",
        "churn.csv",
    ]
    for f in candidates:
        if os.path.exists(f):
            print(f"[DATA] Loading '{f}' …")
            df = pd.read_csv(f)
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            return df

    print("[DATA] No CSV found — generating synthetic Telco churn data …")
    return _make_synthetic(n=7043)


def _make_synthetic(n=7043):
    rng = np.random.default_rng(SEED)

    tenure       = rng.integers(1, 73, n)
    monthly_chg  = rng.uniform(18, 118, n).round(2)
    total_chg    = (tenure * monthly_chg * rng.uniform(0.85, 1.05, n)).round(2)
    senior       = (rng.random(n) < 0.16).astype(int)
    contract     = rng.choice(["Month-to-month","One year","Two year"],
                               n, p=[0.55, 0.21, 0.24])
    internet_svc = rng.choice(["DSL","Fiber optic","No"], n, p=[0.34, 0.44, 0.22])
    payment      = rng.choice(
        ["Electronic check","Mailed check","Bank transfer (automatic)",
         "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21])
    paperless    = rng.choice(["Yes","No"], n, p=[0.59, 0.41])
    partner      = rng.choice(["Yes","No"], n, p=[0.48, 0.52])
    dependents   = rng.choice(["Yes","No"], n, p=[0.30, 0.70])
    phone_svc    = rng.choice(["Yes","No"], n, p=[0.90, 0.10])
    multiple_lines = rng.choice(["Yes","No","No phone service"], n, p=[0.42, 0.48, 0.10])
    online_sec   = rng.choice(["Yes","No","No internet service"], n, p=[0.29, 0.50, 0.21])
    tech_support = rng.choice(["Yes","No","No internet service"], n, p=[0.29, 0.50, 0.21])
    streaming_tv = rng.choice(["Yes","No","No internet service"], n, p=[0.38, 0.41, 0.21])

    # Churn probability: higher for month-to-month, fiber optic, electronic check
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.10 * (internet_svc == "Fiber optic")
        + 0.08 * (payment == "Electronic check")
        - 0.15 * np.clip((tenure - 1) / 72, 0, 1)
        + 0.05 * senior
        + rng.uniform(-0.05, 0.05, n)
    ).clip(0.02, 0.90)
    churn = (rng.random(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "customerID"    : [f"CUST-{i:05d}" for i in range(n)],
        "gender"        : rng.choice(["Male","Female"], n),
        "SeniorCitizen" : senior,
        "Partner"       : partner,
        "Dependents"    : dependents,
        "tenure"        : tenure,
        "PhoneService"  : phone_svc,
        "MultipleLines" : multiple_lines,
        "InternetService": internet_svc,
        "OnlineSecurity": online_sec,
        "TechSupport"   : tech_support,
        "StreamingTV"   : streaming_tv,
        "Contract"      : contract,
        "PaperlessBilling": paperless,
        "PaymentMethod" : payment,
        "MonthlyCharges": monthly_chg,
        "TotalCharges"  : total_chg,
        "Churn"         : pd.Series(churn).map({1:"Yes", 0:"No"}),
    })


# ══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

# Columns used for prediction
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges",
                "tenure_monthly_ratio", "log_total_charges"]
CATEG_COLS   = ["gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "TechSupport", "StreamingTV",
                "Contract", "PaperlessBilling", "PaymentMethod"]


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing:
      1. Drop customerID (identifier, not predictive)
      2. Fix TotalCharges dtype (sometimes loaded as str)
      3. Engineer features: tenure_monthly_ratio, log_total_charges, churn risk bins
      4. Encode target
      5. Build sklearn Pipeline (OneHotEncoder + StandardScaler)
      6. Train/test split

    Returns X_train, X_test, y_train, y_test, pipeline, feature_names, df_clean
    """
    print("[PREPROCESS] Starting …")
    df = df.copy()

    # Drop ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    null_total = df["TotalCharges"].isnull().sum()
    if null_total > 0:
        df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"], inplace=True)
        print(f"          Imputed {null_total} null TotalCharges")

    # Feature engineering
    df["tenure_monthly_ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1e-9)
    df["log_total_charges"]    = np.log1p(df["TotalCharges"])

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    # Align column lists to what's actually present
    num_cols  = [c for c in NUMERIC_COLS  if c in X.columns]
    cat_cols  = [c for c in CATEG_COLS    if c in X.columns]

    # sklearn Pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),       num_cols),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), cat_cols),
    ])

    pipeline = Pipeline([("prep", preprocessor)])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    pipeline.fit(X_train_raw)
    X_train = pipeline.transform(X_train_raw)
    X_test  = pipeline.transform(X_test_raw)

    # Feature names
    ohe_names = list(pipeline.named_steps["prep"]
                     .named_transformers_["cat"]
                     .get_feature_names_out(cat_cols))
    feature_names = num_cols + ohe_names

    churn_rate = y.mean() * 100
    print(f"          Dataset shape : {df.shape}")
    print(f"          Churn rate    : {churn_rate:.1f}%")
    print(f"          Train / Test  : {len(X_train)} / {len(X_test)}")
    print(f"          Features      : {len(feature_names)}")

    return X_train, X_test, y_train, y_test, pipeline, feature_names, df


# ══════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train):
    """
    Train Gradient Boosting Classifier.
    GBC is chosen over XGBoost for zero-dependency installation.
    Results are competitive with XGBoost on tabular churn data.
    """
    print("\n[TRAIN] Training Gradient Boosting Classifier …")
    model = GradientBoostingClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.08,
        subsample         = 0.8,
        min_samples_split = 20,
        random_state      = SEED,
        verbose           = 0,
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1", n_jobs=-1)
    print(f"          5-fold CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n[EVAL] Test Set Results")
    print(classification_report(y_test, y_pred,
                                 target_names=["Stay", "Churn"]))
    return y_pred, y_proba


# ══════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════

def _save(name):
    p = os.path.join(OUTPUT_DIR, name)
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"          Saved → {p}")


def plot_eda(df):
    """4-panel EDA overview."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Telco Customer Churn — EDA Overview", fontsize=14, fontweight="bold")

    # 1. Churn distribution
    ax = axes[0, 0]
    counts = df["Churn"].value_counts()
    bars = ax.bar(["Stay (0)", "Churn (1)"],
                  [counts.get(0, 0), counts.get(1, 0)],
                  color=[C["stay"], C["churn"]], width=0.45,
                  edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, [counts.get(0,0), counts.get(1,0)]):
        pct = cnt / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 40,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Churn Distribution"); ax.set_ylabel("Customers")
    ax.yaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)

    # 2. Tenure by churn
    ax = axes[0, 1]
    stay_tenure  = df.loc[df["Churn"]==0, "tenure"]
    churn_tenure = df.loc[df["Churn"]==1, "tenure"]
    sns.kdeplot(stay_tenure,  ax=ax, color=C["stay"],
                fill=True, alpha=0.35, label="Stay", linewidth=1.8)
    sns.kdeplot(churn_tenure, ax=ax, color=C["churn"],
                fill=True, alpha=0.45, label="Churn", linewidth=1.8)
    ax.set_title("Tenure Distribution by Churn")
    ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Density")
    ax.legend(frameon=False); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    # 3. Monthly charges
    ax = axes[1, 0]
    stay_mc  = df.loc[df["Churn"]==0, "MonthlyCharges"]
    churn_mc = df.loc[df["Churn"]==1, "MonthlyCharges"]
    sns.kdeplot(stay_mc,  ax=ax, color=C["stay"],
                fill=True, alpha=0.35, label="Stay", linewidth=1.8)
    sns.kdeplot(churn_mc, ax=ax, color=C["churn"],
                fill=True, alpha=0.45, label="Churn", linewidth=1.8)
    ax.set_title("Monthly Charges by Churn")
    ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Density")
    ax.legend(frameon=False); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    # 4. Churn rate by contract type
    ax = axes[1, 1]
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    bars = ax.bar(contract_churn.index, contract_churn.values * 100,
                  color=[C["churn"], C["warn"], C["stay"]],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("")
    ax.yaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)

    plt.tight_layout()
    _save("00_eda_overview.png")


def plot_feature_importance(model, feature_names):
    """Top 20 feature importances from Gradient Boosting."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Top 20 Feature Importances — Gradient Boosting",
                 fontsize=14, fontweight="bold", pad=12)
    bars = ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color=C["accent"] + "99", edgecolor=C["accent"], linewidth=1.2,
    )
    ax.set_xlabel("Feature Importance (Gini)")
    ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    _save("01_feature_importance.png")


def plot_shap_or_permutation(model, X_test, feature_names):
    """
    SHAP summary plot if shap is installed, otherwise
    permutation-importance-style bar chart.
    """
    if HAS_SHAP:
        print("          Generating SHAP summary plot …")
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test[:500])
        fig, ax    = plt.subplots(figsize=(10, 7))
        ax.set_title("SHAP Feature Impact on Churn Prediction",
                     fontsize=14, fontweight="bold")
        shap.summary_plot(shap_vals, X_test[:500],
                          feature_names=feature_names,
                          show=False, plot_size=(10, 7))
        _save("02_shap_importance.png")
    else:
        feat_imp = pd.Series(model.feature_importances_,
                             index=feature_names).sort_values(ascending=False)[:15]
        fig, ax  = plt.subplots(figsize=(10, 6))
        ax.set_title("Feature Impact on Churn Prediction\n(Gradient Boosting Importances)",
                     fontsize=13, fontweight="bold")
        ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                color=C["accent"] + "99", edgecolor=C["accent"], linewidth=1.2)
        ax.set_xlabel("Importance Score")
        ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        plt.tight_layout()
        _save("02_feature_impact.png")


def plot_confusion_and_roc(y_test, y_pred, y_proba):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Evaluation — Gradient Boosting Churn Predictor",
                 fontsize=14, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", ax=axes[0],
                cmap="Blues",
                xticklabels=["Stay", "Churn"],
                yticklabels=["Stay", "Churn"],
                linewidths=0.5, linecolor=C["grid"],
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Normalised Confusion Matrix", pad=10)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, lw=2.5, color=C["accent"],
                 label=f"GradientBoosting  (AUC = {roc_auc:.3f})")
    axes[1].plot([0,1],[0,1], "k--", lw=1.2, label="Random")
    axes[1].set_title("ROC Curve", pad=10)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1.02)
    axes[1].legend(frameon=False, fontsize=10)
    axes[1].yaxis.grid(True, alpha=0.4); axes[1].set_axisbelow(True)

    plt.tight_layout()
    _save("03_confusion_roc.png")


def plot_churn_probability_dist(y_proba, y_test):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Predicted Churn Probability Distribution",
                 fontsize=14, fontweight="bold")
    sns.kdeplot(y_proba[y_test==0], ax=ax, color=C["stay"],
                fill=True, alpha=0.35, label="Actual: Stay", linewidth=1.8)
    sns.kdeplot(y_proba[y_test==1], ax=ax, color=C["churn"],
                fill=True, alpha=0.45, label="Actual: Churn", linewidth=1.8)
    ax.axvline(0.5, color=C["warn"], linestyle="--", lw=1.5, label="Decision boundary (0.5)")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Density"); ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    _save("04_churn_probability_dist.png")


def plot_business_impact(y_test, y_pred, y_proba):
    """
    Business-focused plot: revenue at risk from churning customers.
    Assumes average monthly revenue per customer = $65.
    """
    AVG_MONTHLY_REV = 65
    RETENTION_COST  = 20   # cost of a retention offer

    thresholds = np.linspace(0.1, 0.9, 50)
    revenues_saved, costs = [], []

    for t in thresholds:
        flagged   = (y_proba >= t).sum()
        tp        = ((y_proba >= t) & (y_test == 1)).sum()
        rev_saved = tp * AVG_MONTHLY_REV * 12     # annual revenue saved
        cost      = flagged * RETENTION_COST
        revenues_saved.append(rev_saved)
        costs.append(cost)

    net = np.array(revenues_saved) - np.array(costs)
    best_t = thresholds[np.argmax(net)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Business Impact — Revenue Saved vs Retention Cost by Threshold",
                 fontsize=13, fontweight="bold")
    ax.plot(thresholds, np.array(revenues_saved)/1000, color=C["stay"],
            lw=2, label="Revenue saved ($k/yr)")
    ax.plot(thresholds, np.array(costs)/1000, color=C["churn"],
            lw=2, linestyle="--", label="Retention cost ($k)")
    ax.plot(thresholds, net/1000, color=C["accent"],
            lw=2.5, label="Net benefit ($k)")
    ax.axvline(best_t, color=C["warn"], linestyle=":", lw=1.8,
               label=f"Optimal threshold: {best_t:.2f}")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Value ($k)")
    ax.legend(frameon=False, fontsize=10)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    _save("05_business_impact.png")


# ══════════════════════════════════════════════════════════════════════════
#  SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════

def save_artifacts(model, pipeline, feature_names, df):
    """Save model, preprocessor pipeline, feature names, and schema."""
    joblib.dump(model,    os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    schema = {
        "feature_names"  : feature_names,
        "numeric_cols"   : NUMERIC_COLS,
        "categ_cols"     : CATEG_COLS,
        "sample_input"   : {
            "gender"           : "Male",
            "SeniorCitizen"    : 0,
            "Partner"          : "Yes",
            "Dependents"       : "No",
            "tenure"           : 12,
            "PhoneService"     : "Yes",
            "MultipleLines"    : "No",
            "InternetService"  : "Fiber optic",
            "OnlineSecurity"   : "No",
            "TechSupport"      : "No",
            "StreamingTV"      : "Yes",
            "Contract"         : "Month-to-month",
            "PaperlessBilling" : "Yes",
            "PaymentMethod"    : "Electronic check",
            "MonthlyCharges"   : 79.85,
            "TotalCharges"     : 958.20,
        },
    }
    with open(os.path.join(MODEL_DIR, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\n[SAVE] Artifacts written to ./{MODEL_DIR}/")
    print("         model.pkl | preprocessor.pkl | schema.json")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  CODTECH INTERNSHIP  |  TASK 3  |  END-TO-END PROJECT")
    print("  Customer Churn Predictor — Model Training")
    print("=" * 65)

    df = load_data()
    X_train, X_test, y_train, y_test, pipeline, feature_names, df_clean = preprocess(df)

    model = train_model(X_train, y_train)
    y_pred, y_proba = evaluate(model, X_test, y_test)

    print("\n[VIZ] Generating visualisations …")
    plot_eda(df_clean)
    plot_feature_importance(model, feature_names)
    plot_shap_or_permutation(model, X_test, feature_names)
    plot_confusion_and_roc(y_test, y_pred, y_proba)
    plot_churn_probability_dist(y_proba, y_test)
    plot_business_impact(y_test, y_pred, y_proba)

    save_artifacts(model, pipeline, feature_names, df_clean)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"  Test Accuracy : {acc:.4f}  |  F1 Score : {f1:.4f}")
    print(f"  Model saved   : ./{MODEL_DIR}/model.pkl")
    print(f"  Plots saved   : ./{OUTPUT_DIR}/")
    print(f"{'='*65}")
    print(f"\n  Next step → run:  python app.py")


if __name__ == "__main__":
    main()
