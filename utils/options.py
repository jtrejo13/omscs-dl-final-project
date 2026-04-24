import yaml
import os


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a dict"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)
    return opt
