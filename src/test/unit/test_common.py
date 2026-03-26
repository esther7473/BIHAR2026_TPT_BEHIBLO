import pytest
import os
from src.common.common import CONFIG, ROOT_DIR, get_full_path  


def test_get_full_path():
    rel_path  = "./data/weather.db"
    full_path = get_full_path(rel_path)
    expected  = os.path.normpath(os.path.join(ROOT_DIR, rel_path))
    assert full_path == expected

def test_config_loaded():
    assert isinstance(CONFIG, dict)

def test_config_sections():
    assert "paths"  in CONFIG
    assert "model"  in CONFIG
    assert "data"   in CONFIG
    assert "mlflow" in CONFIG

def test_config_paths_are_absolute():
    for key, path in CONFIG["paths"].items():
        assert os.path.isabs(path), f"{key} n'est pas un chemin absolu : {path}"

def test_config_folders_created():
    for key, path in CONFIG["paths"].items():
        folder = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
        assert os.path.exists(folder), f"Dossier manquant pour {key} : {folder}"
        
