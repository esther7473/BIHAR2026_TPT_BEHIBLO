import os
import yaml
from src.common.common import CONFIG


def get_latest_run_metadata():
    mlruns_path = CONFIG["mlflow"]["tracking_uri"]
    models_path = os.path.join(mlruns_path, "models")
    
    if not os.path.exists(models_path):
        return None

    versions = []
    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        for version_dir in os.listdir(model_path):
            meta_path = os.path.join(model_path, version_dir, "meta.yaml")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = yaml.safe_load(f)
                versions.append(meta)

    versions.sort(key=lambda x: x.get("version", 0), reverse=True)
    return versions[0] if versions else None  