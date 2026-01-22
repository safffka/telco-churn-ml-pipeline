# src/train.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from src.utils import setup_mlflow
from src.config import load_config, cfg_get


def load_features(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def train_model(df: pd.DataFrame, cfg,use_stratify: bool = True):
    target_col = cfg_get(cfg, "data", "target_col")
    id_col = cfg_get(cfg, "data", "id_col")

    test_size = cfg_get(cfg, "split", "test_size")
    seed = cfg_get(cfg, "project", "seed")

    model_params = cfg_get(cfg, "model", "params").copy()
    model_params["random_seed"] = seed

    categorical_cols = cfg_get(cfg, "model", "categorical_cols", default=[])

    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    stratify_arg = y if use_stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_arg,
    )

    model = CatBoostClassifier(**model_params)

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_pred_proba >= 0.5).astype(int)

    if len(set(y_val)) < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(y_val, val_pred_proba))

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "features_count": int(X.shape[1]),
    }

    return model, metrics, model_params, test_size, seed



def save_artifacts(model, metrics: dict, cfg) -> None:
    model_path = Path(cfg_get(cfg, "paths", "model_path"))
    metrics_path = Path(cfg_get(cfg, "paths", "reports_dir")) / "metrics.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    cfg = load_config()

    experiment_name = cfg_get(cfg, "project", "experiment_name")
    setup_mlflow(experiment_name, cfg)
    features_path = Path(cfg_get(cfg, "data", "features_path"))
    df = load_features(features_path)
    with mlflow.start_run(run_name="train_catboost"):
        model, metrics, model_params, test_size, seed = train_model(df, cfg)

        # log params (РЕАЛЬНЫЕ)
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("seed", seed)

        # log metrics
        mlflow.log_metrics({
            "val_roc_auc": metrics["roc_auc"],
            "val_accuracy": metrics["accuracy"],
        })

        mlflow.catboost.log_model(model, artifact_path="model")

        save_artifacts(model, metrics, cfg)


if __name__ == "__main__":
    main()
