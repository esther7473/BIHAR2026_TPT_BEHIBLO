import os
import pickle
import mlflow
from mlflow import MlflowClient
from src.common.common import CONFIG, ROOT_DIR
from src.data.database import save_model_version


def promote_best_model():
    tracking_uri = CONFIG["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    client     = MlflowClient(tracking_uri=tracking_uri)
    model_name = CONFIG["model"]["name"]

    experiment = client.get_experiment_by_name(CONFIG["mlflow"]["experiment_name"])
    if experiment is None:
        print(" Expérience introuvable")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED' and metrics.rmse_real < 999",
        order_by=["metrics.rmse_real ASC"],
        max_results=1
    )
    if not runs:
        print(" Aucun run valide trouvé")
        return

    best_run  = runs[0]
    best_rmse = best_run.data.metrics["rmse_real"]
    print(f"\n Meilleur run : {best_run.info.run_name} | RMSE : {best_rmse:.4f}")

    try:
        current = client.get_model_version_by_alias(model_name, "champion")
        current_rmse = client.get_run(current.run_id).data.metrics.get("rmse_real", float("inf"))
        print(f" Champion actuel : v{current.version} | RMSE : {current_rmse:.4f}")
    except Exception:
        current_rmse = float("inf")
        print(" Pas de champion actuel")

    if best_rmse >= current_rmse:
        print(f" Pas de promotion — champion actuel déjà meilleur")
        return

    mv = mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name=model_name
    )
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=mv.version
    )
    print(f" Modèle v{mv.version} promu en champion")


    save_model_version(
        model_name=model_name,
        version=int(mv.version),   
        run_id=best_run.info.run_id,
        rmse=best_rmse
    )

    model_path  = CONFIG["paths"]["model_path"]
    scaler_path = CONFIG["paths"]["scaler_path"]

    model = mlflow.keras.load_model(f"models:/{model_name}@champion")
    model.save(model_path)
    print(f" Modèle sauvegardé : {model_path}")

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