import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import roc_auc_score
import mlflow
from mlflow.tracking import MlflowClient


def detect_data_drift_without_labels(training_df, prod_df, snapshotdate, run_id, model_type, psi_threshold: float = 0.2) -> bool:
    """
    Detect data drift using Evidently PSI method
    """
    report = Report([DataDriftPreset(method="psi")])
    train_pd = training_df.toPandas()
    prod_pd = prod_df.toPandas()

    # Drop text columns
    text_cols = train_pd.select_dtypes(include=['object', 'string']).columns.tolist()
    train_pd = train_pd.drop(columns=text_cols)
    prod_pd = prod_pd.drop(columns=text_cols)

    # Run Evidently report
    report.run(reference_data=train_pd, current_data=prod_pd)
    snapshotdate_str = str(snapshotdate)

    # Save report
    report_path = f"opt/airflow/reports/{model_type}/{snapshotdate_str}_data_drift.html"
    report_dir = os.path.dirname(report_path)
    os.makedirs(report_dir, exist_ok=True)
    report.save_html(report_path)

    # Log report to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(report_path)

    # Extract drift metrics
    drift_json = report.as_dict()
    drifted_features = []

    for metric in drift_json["metrics"]:
        metric_val = metric["value"]
        feature = metric["metric_id"]
        if feature.startswith("ValueDrift(column="):
            start = feature.find("column=") + len("column=")
            end = feature.find(",", start) if "," in feature[start:] else feature.find(")", start)
            column_name = feature[start:end]
            if isinstance(metric_val, (float, int)) and metric_val > psi_threshold:
                drifted_features.append(column_name)

    if drifted_features:
        print(f"Data drift detected in: {drifted_features}")
        return True
    else:
        print("No significant data drift detected.")
        return False


def detect_model_performance(pred_df, label_df, run_id):
    """
    Detect model performance drift using AUC
    """
    label_pd = label_df.toPandas()
    pred_pd = pred_df.toPandas()

    # Align predictions and labels
    min_len = min(len(label_pd), len(pred_pd))
    label_pd_cut = label_pd.reset_index(drop=True).iloc[:min_len]
    pred_pd_cut = pred_pd.reset_index(drop=True).iloc[:min_len]

    # Calculate AUC
    auc = roc_auc_score(label_pd_cut['label'], pred_pd_cut['model_predictions'])

    # Log AUC to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("auc", auc)

    if auc < 0.6:
        print(f"Model performance degraded (AUC: {auc:.4f})")
        return True
    else:
        print(f"Model performance is acceptable (AUC: {auc:.4f})")
        return False


def get_model_train_date():
    """
    Retrieve model training date and metadata from MLflow
    """
    client = MlflowClient("http://mlflow:5000")
    model_name = "Loan-default-prod"

    latest = client.get_latest_versions(model_name, stages=["Production"])[0]
    run_id = latest.run_id
    run = client.get_run(run_id)
    run_data = run.data
    model_type = run_data.params["model_type"]
    snapshot_date = run_data.params.get("snapshot_date", "Unknown")[:10]

    return model_type, snapshot_date, run_id


def model_monitor_prod(snapshotdate):
    """
    Main function for model monitoring
    """
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Get model metadata
    model_type, train_date, run_id = get_model_train_date()

    # Define paths
    base_path = "../datamart/gold/"
    train_path = os.path.join(base_path, "feature", f"gold_feature_store_monthly_{train_date}.parquet")
    current_path = os.path.join(base_path, "feature", f"gold_feature_store_monthly_{snapshotdate.replace('-', '_')}.parquet")
    label_path = os.path.join(base_path, "label", f"gold_label_store_monthly_{snapshotdate.replace('-', '_')}.parquet")
    prediction_path = os.path.join(base_path, "model_predictions", model_type.lower(), f"{snapshotdate.replace('-', '_')}.parquet")

    # Load data
    train_df = spark.read.parquet(train_path)
    current_df = spark.read.parquet(current_path)
    label_df = spark.read.parquet(label_path)
    prediction_df = spark.read.parquet(prediction_path)

    # Detect data drift
    data_drift_detected = detect_data_drift_without_labels(train_df, current_df, snapshotdate, run_id, model_type)

    # Detect performance drift
    performance_drift_detected = detect_model_performance(prediction_df, label_df, run_id)

    # Output results
    if data_drift_detected or performance_drift_detected:
        sys.stdout.write("True")
    else:
        sys.stdout.write("False")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model monitoring job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    model_monitor_prod(args.snapshotdate)
