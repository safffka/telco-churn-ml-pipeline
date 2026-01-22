# tests/test_config.py
from src.config import load_config, cfg_get

def test_config_loads():
    cfg = load_config()
    assert cfg is not None

def test_required_config_fields_exist():
    cfg = load_config()

    assert cfg_get(cfg, "data", "features_path") is not None
    assert cfg_get(cfg, "data", "target_col") is not None
    assert cfg_get(cfg, "split", "test_size") is not None
    assert cfg_get(cfg, "model", "params") is not None
    assert cfg_get(cfg, "quality_gate", "min_roc_auc") is not None

def test_test_size_valid():
    cfg = load_config()
    test_size = cfg_get(cfg, "split", "test_size")
    assert 0 < test_size < 1
