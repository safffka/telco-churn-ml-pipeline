from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw: dict

def load_config(path: str = "configs/config.yaml") -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)

def cfg_get(cfg: Config, *keys, default=None):
    cur = cfg.raw
    for k in keys:
        cur = cur.get(k, None) if isinstance(cur, dict) else None
        if cur is None:
            return default
    return cur
