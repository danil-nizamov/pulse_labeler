import os
import json
from config_loader import exe_dir_path

# Constants
STATE_FILE = "labeling_state.json"
if not os.path.isabs(STATE_FILE):
    STATE_FILE = exe_dir_path(STATE_FILE)

def save_state(file_path):
    with open(STATE_FILE, 'w') as f:
        json.dump({"last_file": file_path}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None
