import yaml
import os

def load_config(config_path='config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please create it.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
