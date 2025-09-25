from pathlib import Path

import yaml


def load_config(path: str = "../config/aqua_rag_config.yaml"):
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)["rag_config"]
