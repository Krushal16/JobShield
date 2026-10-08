"""Project paths, configuration loading and random seeds."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=CONFIG_PATH):
    """Read configs/config.yaml into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative):
    """Turn a path from the config (relative to the project root) into an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed=None):
    """Fix random seeds so experiments can be repeated."""
    if seed is None:
        seed = load_config()["seed"]
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
