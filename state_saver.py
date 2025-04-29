import os
import json

# Constants
STATE_FILE = "labeling_state.json"

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
