import os
import yaml

# src/common/common.py
#   → SRC_DIR  = src/common
#   → SRC_DIR  = src/
#   → ROOT_DIR = Projet/   ✅
COMMON_DIR = os.path.dirname(os.path.abspath(__file__))  # src/common/
SRC_DIR    = os.path.dirname(COMMON_DIR)                  # src/
ROOT_DIR   = os.path.dirname(SRC_DIR)                     # Projet/  ← racine

CONFIG_PATH = os.path.join(ROOT_DIR, "config.yml")

def get_full_path(rel_path):
    return os.path.normpath(os.path.join(ROOT_DIR, rel_path))

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)
    for key, value in CONFIG["paths"].items():
        CONFIG["paths"][key] = get_full_path(value)

# Création automatique des dossiers
for key, path in CONFIG["paths"].items():
    folder = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)