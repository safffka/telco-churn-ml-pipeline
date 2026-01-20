# src/train.py
import json
from pathlib import Path

import mlflow
import mlflow.catboost

from src.utils import setup_mlflow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# paths
FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/model.cbm")
METRICS_PATH = Path("reports/metrics.json")

TARGET_COL = "Churn"
ID_COL = "customerID"

# categorical features (должны совпадать с этапом 1)
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "is_long_contract",
    "has_fiber",
]


def load_features(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def train_model(df: pd.DataFrame) -> dict:
    # drop id
    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_COLS,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    # predictions
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_pred_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, val_pred_proba)),
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "features_count": int(X.shape[1]),
    }

    return model, metrics


def save_artifacts(model, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    setup_mlflow("telco-churn-catboost")

    df = load_features(FEATURES_PATH)

    with mlflow.start_run(run_name="catboost_baseline"):
        model, metrics = train_model(df)

        # log params
        mlflow.log_params({
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
        })

        # log metrics
        mlflow.log_metrics({
            "roc_auc": metrics["roc_auc"],
            "accuracy": metrics["accuracy"],
        })

        # log model
        mlflow.catboost.log_model(
            model,
            artifact_path="model"
        )

        save_artifacts(model, metrics)



if __name__ == "__main__":
    main()
