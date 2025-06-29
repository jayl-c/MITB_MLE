from mlflow.tracking import MlflowClient
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://mlflow:5000/")

model_name = "Loan-default-prod"
client = MlflowClient(tracking_uri="http://mlflow:5000/")

# Create the registered model if it doesn't exist
# try:
#     client.get_registered_model(model_name)
# except mlflow.exceptions.RestException:
#     client.create_registered_model(model_name)
#     print(f"Registered model '{model_name}' created.")

def get_latest_model_by_type(experiment_name, model_type, metric):
    client = MlflowClient(tracking_uri="http://mlflow:5000/")
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_type = '{model_type}'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise Exception(f"No runs found for model_type = '{model_type}'")

    run = runs[0]
    return run, run.data.metrics.get(metric)


def compare_and_promote_best_model():
    experiment_name = "Loan-default-model"
    metric = "oot_auc"  # Use OOT AUC for comparison

    logreg_run, logreg_score = get_latest_model_by_type(experiment_name, "LogisticRegression", metric)
    xgb_run, xgb_score = get_latest_model_by_type(experiment_name, "XGBClassifier", metric)

    print(f"Logistic Regression OOT AUC: {logreg_score:.4f}")
    print(f"XGBoost OOT AUC: {xgb_score:.4f}")

    best_run = logreg_run if logreg_score > xgb_score else xgb_run
    best_model_type = best_run.data.params['model_type']
    best_model_name = "Loan-default-prod"  # Registered model name

    client = MlflowClient(tracking_uri="http://mlflow:5000/")

    artifact_uri = f"{best_run.info.artifact_uri}/model"
    print(artifact_uri)
    # Register new model version
    mv = client.create_model_version(
        name=best_model_name,
        source=f"{best_run.info.artifact_uri}/model",
        run_id=best_run.info.run_id
    )

    # Promote to production
    client.transition_model_version_stage(
        name=best_model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Promoted {best_model_type} (run_id={best_run.info.run_id}) to Production.")
    return best_run, mv.version


def get_production_model():
    client = MlflowClient(tracking_uri="http://mlflow:5000/")
    model_name = "Loan-default-prod"

    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        raise Exception("No production model found.")
    logger.info(f"Model '{model_name}' version {latest_versions[0].version} is now in Production.")

    mv = latest_versions[0]
    model_uri = f"models:/{model_name}/Production"

    print(f"Production Model: {model_name}")
    print(f"Version: {mv.version}")
    print(f"URI: {model_uri}")

    return model_uri, mv


if __name__ == "__main__":
    best_run, model_version = compare_and_promote_best_model()
