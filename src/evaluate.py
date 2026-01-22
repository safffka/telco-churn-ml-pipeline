# src/evaluate.py
import json
from pathlib import Path

import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.utils import setup_mlflow
from src.config import load_config, cfg_get


def load_artifacts(cfg):
    features_path = Path(cfg_get(cfg, "data", "features_path"))
    model_path = Path(cfg_get(cfg, "paths", "model_path"))

    df = pd.read_parquet(features_path)

    model = CatBoostClassifier()
    model.load_model(model_path)

    return df, model


def validate(df: pd.DataFrame, model: CatBoostClassifier, cfg):
    target_col = cfg_get(cfg, "data", "target_col")
    id_col = cfg_get(cfg, "data", "id_col")

    test_size = cfg_get(cfg, "split", "test_size")
    seed = cfg_get(cfg, "project", "seed")

    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_proba)

    return y_val, y_pred, y_proba, roc_auc


def plot_roc(y_true, y_proba, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_metrics(roc_auc: float, passed: bool, cfg) -> dict:
    reports_dir = Path(cfg_get(cfg, "paths", "reports_dir"))
    metrics_path = reports_dir / "metrics.json"

    min_roc_auc = cfg_get(cfg, "quality_gate", "min_roc_auc")

    metrics = {
        "roc_auc": float(roc_auc),
        "quality_gate": {
            "threshold": float(min_roc_auc),
            "passed": bool(passed),
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    cfg = load_config()

    experiment_name = cfg_get(cfg, "project", "experiment_name")
    setup_mlflow(experiment_name)

    reports_dir = Path(cfg_get(cfg, "paths", "reports_dir"))
    roc_path = reports_dir / "roc_curve.png"
    cm_path = reports_dir / "confusion_matrix.png"

    min_roc_auc = cfg_get(cfg, "quality_gate", "min_roc_auc")

    df, model = load_artifacts(cfg)
    y_val, y_pred, y_proba, roc_auc = validate(df, model, cfg)

    plot_roc(y_val, y_proba, roc_path)
    plot_confusion_matrix(y_val, y_pred, cm_path)

    passed = roc_auc >= min_roc_auc
    metrics = save_metrics(roc_auc, passed, cfg)

    # ===== MLflow logging =====
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_param("quality_gate_threshold", float(min_roc_auc))
        mlflow.log_param("quality_gate_passed", bool(passed))

        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(reports_dir / "metrics.json")

    print(metrics)

    # ===== hard quality gate =====
    if not passed:
        raise RuntimeError(
            f"Quality gate failed: ROC-AUC {roc_auc:.3f} < {min_roc_auc}"
        )


if __name__ == "__main__":
    main()
