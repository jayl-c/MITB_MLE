from mlflow.tracking import MlflowClient

def get_best_model_run(experiment_name: str = "loan-default-predictor"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.f1 DESC"],  # or "metrics.oot_auc DESC"
        max_results=1
    )

    if not runs:
        raise ValueError("No completed runs found.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/loan-default-predictor"

    print("Best model run ID:", run_id)
    print("F1 score:", best_run.data.metrics.get("f1"))
    print("Snapshot date:", best_run.data.tags.get("snapshot_date"))
    print("Model type:", best_run.data.tags.get("model_type"))

    return model_uri, best_run