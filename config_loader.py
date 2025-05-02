import yaml
import sys
import os

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller exe"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

def exe_dir_path(relative_path):
    """
    Returns a path relative to the directory of the executable (or script, if unfrozen).
    Use for user data/output folders.
    """
    if getattr(sys, 'frozen', False):  # Running as exe
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.normpath(os.path.join(base_path, relative_path))

def load_config():
    config_path = resource_path("config.yaml")
    print(f"DEBUG: Trying to open config at {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please create it.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
