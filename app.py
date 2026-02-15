# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2 - Bank Marketing", layout="wide")

# ----------------------------
# Config
# ----------------------------
TARGET_COL = "deposit"          # your dataset target
MODEL_DIR = Path("model")       # folder where you saved *.pkl
FEATURE_COLS_FILE = MODEL_DIR / "feature_columns.pkl"  # recommended to save during training
SCALER_FILE = MODEL_DIR / "scaler.pkl"

# Models expected (files must exist in /model)
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic.pkl",
    "Decision Tree": MODEL_DIR / "dt.pkl",
    "KNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "nb.pkl",
    "Random Forest": MODEL_DIR / "rf.pkl",
    "XGBoost": MODEL_DIR / "xgb.pkl",
}

SCALED_MODELS = {"Logistic Regression", "KNN"}

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_artifacts():
    models = {}
    missing = []
    for name, fpath in MODEL_FILES.items():
        if fpath.exists():
            models[name] = joblib.load(fpath)
        else:
            missing.append(str(fpath))

    scaler = joblib.load(SCALER_FILE) if SCALER_FILE.exists() else None

    feature_cols = None
    if FEATURE_COLS_FILE.exists():
        feature_cols = joblib.load(FEATURE_COLS_FILE)

    return models, scaler, feature_cols, missing


def preprocess_features(df_raw: pd.DataFrame, feature_cols: list | None):
    """
    Preprocess to match training:
    1) Drop target if present
    2) One-hot encode categoricals
    3) Align columns to training feature columns (recommended)
    """
    df = df_raw.copy()

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # One-hot encode (must match training approach)
    df_enc = pd.get_dummies(df, drop_first=True)

    # Align columns to training columns if available (BEST PRACTICE)
    if feature_cols is not None:
        df_enc = df_enc.reindex(columns=feature_cols, fill_value=0)

    return df_enc


def evaluate(model, X, y_true):
    y_pred = model.predict(X)

    # AUC needs probabilities; handle models without predict_proba
    y_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            y_prob = proba[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["AUC"] = np.nan

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    return metrics, cm, report, y_pred, y_prob


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    st.pyplot(fig)


# ----------------------------
# UI
# ----------------------------
st.title("Machine Learning Assignment 2 — Bank Marketing Classifier")

models, scaler, feature_cols, missing_files = load_artifacts()

if missing_files:
    st.warning(
        "Some model files are missing. Please ensure you saved all models into the `model/` folder:\n\n"
        + "\n".join(f"- {m}" for m in missing_files)
    )

if scaler is None:
    st.warning("Scaler file not found at `model/scaler.pkl`. Logistic Regression / KNN scaling will not work correctly.")

if feature_cols is None:
    st.info(
        "Recommended: save training feature columns to `model/feature_columns.pkl` "
        "so uploaded test data can be aligned exactly (important for one-hot encoding)."
    )

st.markdown("### 1) Upload Test CSV")
uploaded = st.file_uploader("Upload CSV (test data)", type=["csv"])

st.markdown("### 2) Choose Model")
model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))

if uploaded is None:
    st.stop()

# Read CSV safely
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.write("Preview:")
st.dataframe(df.head(10), use_container_width=True)

# Extract y if present
y_true = None
if TARGET_COL in df.columns:
    # Convert target yes/no to 1/0 if needed
    y_series = df[TARGET_COL].copy()
    if y_series.dtype == "object":
        y_series = y_series.str.strip().str.lower().map({"yes": 1, "no": 0})
    y_true = pd.to_numeric(y_series, errors="coerce")
    # If any NaNs remain, warn
    if y_true.isna().any():
        st.warning("Some target values could not be parsed. Ensure deposit is 'yes'/'no' or 1/0.")
        y_true = None
else:
    st.info(f"Target column `{TARGET_COL}` not found. App will show predictions only (no metrics).")

# Preprocess X
X = preprocess_features(df, feature_cols)

# If we didn't save feature_cols, we still need consistent columns.
# We can try a fallback: align to model's expected number of features (best-effort),
# but the correct way is saving feature_columns.pkl from training.
selected_model = models.get(model_name)
if selected_model is None:
    st.error("Selected model could not be loaded. Check `model/` folder.")
    st.stop()

# Best-effort check if feature_cols not present
if feature_cols is None and hasattr(selected_model, "n_features_in_"):
    expected = int(selected_model.n_features_in_)
    if X.shape[1] != expected:
        st.warning(
            f"Feature count mismatch: model expects {expected} features, but uploaded data produced {X.shape[1]}. "
            "Please save `feature_columns.pkl` from training and redeploy."
        )

# Scale if needed
X_for_model = X
if model_name in SCALED_MODELS:
    if scaler is None:
        st.error("Scaler missing. Please save `model/scaler.pkl` during training.")
        st.stop()
    X_for_model = scaler.transform(X)

st.markdown("### 3) Results")

# If target exists, compute metrics + confusion matrix + report
if y_true is not None:
    metrics, cm, report, y_pred, y_prob = evaluate(selected_model, X_for_model, y_true)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Evaluation Metrics")
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ["Value"]
        st.dataframe(metrics_df, use_container_width=True)

    with col2:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(cm)

    st.subheader("Classification Report")
    st.code(report)

else:
    # Predictions only
    y_pred = selected_model.predict(X_for_model)
    st.subheader("Predictions (first 20)")
    st.write(pd.Series(y_pred).head(20))

    st.subheader("Prediction Distribution")
    st.write(pd.Series(y_pred).value_counts())

st.markdown("---")
st.caption(
    "Note: For one-hot encoding consistency, save training columns as `model/feature_columns.pkl`. "
    "This avoids feature mismatch during deployment."
)

