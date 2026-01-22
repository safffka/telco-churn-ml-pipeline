# tests/test_train.py
from src.train import train_model
from src.config import load_config

def test_train_model_runs(sample_df):
    cfg = load_config()

    model, metrics, model_params, test_size, seed = train_model(sample_df, cfg, use_stratify=False,)

    # model
    assert model is not None

    # metrics
    assert "roc_auc" in metrics
    assert "accuracy" in metrics

    # types (очень важно для json / mlflow)
    assert metrics["roc_auc"] is None or isinstance(metrics["roc_auc"], float)


    # config-derived values
    assert isinstance(test_size, float)
    assert isinstance(seed, int)
