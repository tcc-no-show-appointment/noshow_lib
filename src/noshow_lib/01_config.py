# src/glpr/config.py

from pathlib import Path
import yaml


def load_config(path) -> dict:
    """
    Lê um arquivo YAML e devolve um dict Python.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
