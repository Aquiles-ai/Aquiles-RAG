import os
from platformdirs import user_data_dir
import json

data_dir = user_data_dir("aquiles", "AquilesRAG")
os.makedirs(data_dir, exist_ok=True)

AQUILES_CONFIG = os.path.join(data_dir, "aquiles_cofig.json")

def load_aquiles_config():
    if os.path.exists(AQUILES_CONFIG):
        try:
            with open(AQUILES_CONFIG, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_aquiles_configs(configs):
    with open(AQUILES_CONFIG, "w") as f:
        json.dump(configs, f)