# src/evaluate.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import mlflow

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.utils import setup_mlflow

# ================= paths =================
FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/model.cbm")

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"
ROC_PATH = REPORTS_DIR / "roc_curve.png"
CM_PATH = REPORTS_DIR / "confusion_matrix.png"

# ================= config =================
TARGET_COL = "Churn"
ID_COL = "customerID"

MIN_ROC_AUC = 0.75
EXPERIMENT_NAME = "telco-churn-catboost"

# ========================================


def load_artifacts():
    df = pd.read_parquet(FEATURES_PATH)

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    return df, model


def validate(df: pd.DataFrame, model: CatBoostClassifier):
    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_proba)

    return y_val, y_pred, y_proba, roc_auc


def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(ROC_PATH, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CM_PATH, bbox_inches="tight")
    plt.close()


def save_metrics(roc_auc: float):
    metrics = {
        "roc_auc": float(roc_auc),
        "quality_gate": {
            "threshold": float(MIN_ROC_AUC),
            "passed": bool(roc_auc >= MIN_ROC_AUC),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    setup_mlflow(EXPERIMENT_NAME)

    df, model = load_artifacts()
    y_val, y_pred, y_proba, roc_auc = validate(df, model)

    plot_roc(y_val, y_proba)
    plot_confusion_matrix(y_val, y_pred)

    metrics = save_metrics(roc_auc)

    # ===== MLflow logging =====
    with mlflow.start_run(run_name="validation"):
        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_param("quality_gate_threshold", MIN_ROC_AUC)
        mlflow.log_param(
            "quality_gate_passed",
            bool(roc_auc >= MIN_ROC_AUC),
        )

        mlflow.log_artifact(ROC_PATH)
        mlflow.log_artifact(CM_PATH)
        mlflow.log_artifact(METRICS_PATH)

    print(metrics)

    # ===== quality gate =====
    if roc_auc < MIN_ROC_AUC:
        raise RuntimeError(
            f"Quality gate failed: ROC-AUC {roc_auc:.3f} < {MIN_ROC_AUC}"
        )


if __name__ == "__main__":
    main()
