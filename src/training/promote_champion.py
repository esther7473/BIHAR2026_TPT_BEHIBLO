from mlflow import MlflowClient
from src.common.common import CONFIG
import mlflow


def promote_best_model():
    tracking_uri = CONFIG["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri) 

    client     = MlflowClient(tracking_uri=CONFIG["mlflow"]["tracking_uri"])
    model_name = CONFIG["model"]["name"]
    experiment = client.get_experiment_by_name(CONFIG["mlflow"]["experiment_name"])

    # ── Récupère tous les runs lstm triés par rmse_real croissant ──
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        # filter_string="tags.mlflow.runName LIKE 'lstm%'",  
        order_by=["metrics.rmse_real ASC"]
    )

    if not runs:
        print(" Aucun run trouvé")
        return

    # ── Affiche le top 3 ──
    print("\n Top 3 runs :")
    for i, run in enumerate(runs[:3]):
        print(f"  {i+1}. {run.info.run_name}")
        print(f"     MAE  : {run.data.metrics.get('mae_real', 'N/A')}")
        print(f"     RMSE : {run.data.metrics.get('rmse_real', 'N/A')}")
        print(f"     Run ID : {run.info.run_id}")

    best_run = runs[0]
    print(f"\n Meilleur run : {best_run.info.run_name}")
    print(f"   RMSE : {best_run.data.metrics['rmse_real']}")

    # ── Enregistre et promeut en champion ──
    mv = mlflow.register_model(
            model_uri=f"runs:/{best_run.info.run_id}/model",
            name=model_name
    )
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=mv.version
    )

    print(f" Modèle v{mv.version} ({best_run.info.run_name}) promu en champion")


if __name__ == "__main__":
    promote_best_model()