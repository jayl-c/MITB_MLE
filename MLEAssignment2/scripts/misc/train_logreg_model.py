from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import shutil
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from pyspark.ml.classification import GBTClassifier 
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import uuid

import mlflow 
from mlflow.models.signature import infer_signature
import optuna

from utils.helper import *
import config
# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

parser = argparse.ArgumentParser()
parser.add_argument('--snapshotdate', required=True)
args = parser.parse_args()

model_train_date_str = args.snapshotdate
model_train_date = pendulum.parse(model_train_date_str).replace(day=1).naive()
train_test_period_months = 2
oot_period_months = 1
train_test_ratio = 0.8

config = {}
config["model_train_date_str"] = model_train_date_str
config["train_test_period_months"] = train_test_period_months
config["oot_period_months"] =  oot_period_months
config["model_train_date"] =  model_train_date
config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
config["train_test_ratio"] = train_test_ratio 


# connect to label store
label_dir = "datamart/gold/label/"
parquet_files_label = []
for folder in os.listdir(label_dir):
    if folder.endswith(".parquet"):
        # Extract date part from filename
        date_str = folder[-18:-8]
        folder_date = datetime.strptime(date_str, "%Y_%m_%d")

        if config["train_test_start_date"] <= folder_date <= config["oot_end_date"]:
            parquet_files_label.append(os.path.join(label_dir, folder))


# extract label store
labels_sdf = spark.read.parquet(*parquet_files_label)
print("extracted labels_sdf", labels_sdf.count(), config["train_test_start_date"], config["oot_end_date"])


feature_dir = "datamart/gold/feature/"
parquet_files = []
for file in os.listdir(feature_dir):
    if file.endswith(".parquet") and "gold_feature_store_monthly" in file:
        parquet_files.append(os.path.join(feature_dir, file))

# Load only selected files
features_store_sdf = spark.read.parquet(*parquet_files)
features_store_sdf.show()
print(f"Number of rows in features df: {features_store_sdf.count()}")
labels_sdf.show()
print(f"Number of rows in labels df: {labels_sdf.count()}")

features_store_sdf = features_store_sdf.drop("snapshot_date", "Type_of_Loan")
data_pdf = labels_sdf.join(features_store_sdf, on="Customer_ID", how="left").toPandas()

data_pdf = data_pdf.dropna()

oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]


excluded_cols = {"loan_id", "Customer_ID", "label", "label_def", "snapshot_date"}

feature_cols = [
    fe_col for fe_col in data_pdf.columns
    if fe_col not in excluded_cols
]
X_oot = oot_pdf[feature_cols]
y_oot = oot_pdf["label"]
X_train, X_test, y_train, y_test = train_test_split(
    train_test_pdf[feature_cols], train_test_pdf["label"], 
    test_size= 1 - config["train_test_ratio"],
    random_state=88,     # Ensures reproducibility
    shuffle=True,        # Shuffle the data before splitting
    stratify=train_test_pdf["label"]           # Stratify based on the label column
)

mlflow.set_tracking_uri(uri="http://mlflow:5000")
mlflow.set_experiment("Loan Default Prediction")
with mlflow.start_run(run_name=f"lr_{model_train_date_str}") as run:

    #set up standard scalar preprocessing
    scaler = StandardScaler()

    transformer_stdscaler = scaler.fit(X_train)
    with open("standard_scaler.pkl", "wb") as f:
        pickle.dump(transformer_stdscaler, f)

    # Log it to MLflow as an artifact
    mlflow.log_artifact("standard_scaler.pkl")

    #save the features to json
    with open("selected_features.json", "w") as f:
        json.dump(feature_cols, f)

    # Log to MLflow
    mlflow.log_artifact("selected_features.json")

    # transform data
    X_train_processed = transformer_stdscaler.transform(X_train)
    X_test_processed = transformer_stdscaler.transform(X_test)
    X_oot_processed = transformer_stdscaler.transform(X_oot)

    print('X_train_processed', X_train_processed.shape[0])
    print('X_test_processed', X_test_processed.shape[0])
    print('X_oot_processed', X_oot_processed.shape[0])


    run_id = run.info.run_id

    # Log parameters for traceability
    mlflow.log_param("model_type", "LogRegClassifier")
    mlflow.log_param("snapshot_date", config["train_test_start_date"])

    # mlflow.log_param("feature_selection", "top_20_importance_cutoff_0.01")

    # Train the model
    logreg_model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    logreg_model.fit(X_train_processed, y_train)


   # Define the hyperparameter space to search
    param_dist = {
        'penalty': ['l1', 'l2'],  # L1 can induce sparsity
        'C': np.logspace(-4, 4, 20),  # Regularization strength
        'solver': ['liblinear', 'saga']  # solvers that support both L1 and L2
    }

    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)

    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=logreg_model,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=50,     # Fewer iterations since the space is smaller
        cv=3,          # 3-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Perform the random search
    random_search.fit(X_train_processed, y_train)
    
    # Log best hyperparameters
    mlflow.log_params(random_search.best_params_)
    best_model = random_search.best_estimator_

    metric_dict = {}
    for dataset_name, X_data, y_data in [
            ("train", X_train_processed, y_train),
            ("test", X_test_processed, y_test),
            ("oot", X_oot_processed, y_oot),
        ]:
            y_pred_proba = best_model.predict_proba(X_data)[:, 1]
            auc = roc_auc_score(y_data, y_pred_proba)
            mlflow.log_metric(f"{dataset_name}_auc", auc)
            metric_dict[dataset_name] = auc
            
    new_model_score = metric_dict["test"]  # or "oot" if preferred
    stability_score = metric_dict["oot"]  # or "oot" if preferred
    mlflow.log_metric("comparison_metric", new_model_score)
        
    mlflow.sklearn.log_model(best_model, artifact_path="model")
    model_uri = f"runs:/{run_id}/model"

    # ADD THESE TWO MISSING LINES:
    registered_model_name = "loan_default_logistic_regression"
    client = mlflow.tracking.MlflowClient()

    # --- Compare against latest registered model ---
    try:
        prod_versions = client.get_latest_versions(registered_model_name, stages=["Production"])
        if len(prod_versions) > 0:
            prod_version = prod_versions[0]
            prod_run_id = prod_version.run_id

            # Fetch comparison metric from production run
            prod_metric_history = client.get_metric_history(prod_run_id, "comparison_metric")
            if prod_metric_history:
                old_model_score = prod_metric_history[-1].value
                print(f"New: {new_model_score:.4f}, Production: {old_model_score:.4f}")

                if new_model_score > old_model_score and stability_score>0.60: #assuming threshold is better than a random guess
                    print(" New model is better. Registering...")
                    result = mlflow.register_model(
                        model_uri=model_uri,
                        name=registered_model_name
                    )
                    # Optionally auto-transition to "Production"
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=result.version,
                        stage="Staging",
                        archive_existing_versions=True
                    )
                else:
                    print(" New model is NOT better. Skipping registration.")
            else:
                print(" No metric history found in production. Registering new model...")
                result = mlflow.register_model(
                    model_uri=model_uri,
                    name=registered_model_name
                )
                client.transition_model_version_stage(
                        name=registered_model_name,
                        version=result.version,
                        stage="Staging",
                        archive_existing_versions=True
                    )
        else:
            print("No production model yet. Registering new model...")
            result = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )
            client.transition_model_version_stage(
                    name=registered_model_name,
                    version=result.version,
                    stage="Staging",
                    archive_existing_versions=True
                )
    except mlflow.exceptions.MlflowException as e:
        print(f"Registry error: {e}. Creating a new registered model.")
        result = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )
        client.transition_model_version_stage(
        name=registered_model_name,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True
    )