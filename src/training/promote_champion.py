import os
import pickle
import mlflow
from mlflow import MlflowClient
from src.common.common import CONFIG, ROOT_DIR


def promote_best_model():
    tracking_uri = CONFIG["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    client     = MlflowClient(tracking_uri=tracking_uri)
    model_name = CONFIG["model"]["name"]
    experiment = client.get_experiment_by_name(CONFIG["mlflow"]["experiment_name"])

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse_real ASC"]
    )

    if not runs:
        print(" Aucun run trouvé")
        return

    best_run = runs[0]
    print(f"\n Meilleur run : {best_run.info.run_name}")
    print(f"   RMSE : {best_run.data.metrics['rmse_real']}")

    mv = mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name=model_name
    )
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=mv.version
    )
    print(f"✅ Modèle v{mv.version} ({best_run.info.run_name}) promu en champion")

    model_path  = CONFIG["paths"]["model_path"]   
    scaler_path = CONFIG["paths"]["scaler_path"]  

    model = mlflow.keras.load_model(f"models:/{model_name}@champion")
    model.save(model_path)
    print(f"✅ Modèle sauvegardé : {model_path}")

    scaler_artifact_path = mlflow.artifacts.download_artifacts(
        run_id=best_run.info.run_id,
        artifact_path="scaler/scaler.pkl"
    )
    with open(scaler_artifact_path, "rb") as f:
        scaler = pickle.load(f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f" Scaler sauvegardé : {scaler_path}")


if __name__ == "__main__":
    promote_best_model()
