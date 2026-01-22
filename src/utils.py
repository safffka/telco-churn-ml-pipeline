
import mlflow
from src.config import cfg_get


def setup_mlflow(experiment_name: str, cfg=None):
    if cfg is not None:
        mlruns_dir = cfg_get(cfg, "paths", "mlruns_dir")
        if mlruns_dir:
            mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    mlflow.set_experiment(experiment_name)
