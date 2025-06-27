from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import traceback
import pandas as pd
import numpy as np
import os
import glob
import json
import logging

import config
from utils import data_processing_gold_table as dg

from pyspark.sql import SparkSession
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO)
logging.info("Data drift report saved at: %s", config.PATH_DIR_REPORT)

def read_pred(gold_db, spark):
    """
    Helper function to read all partitions of model predictions
    """
    # Updated path to match your inference output
    folder_path = os.path.join(gold_db, 'model_predictions')
    
    # Read recursively to handle model_type subfolders (xgb, logreg)
    files_list = glob.glob(os.path.join(folder_path, '**', '*.parquet'), recursive=True)
   
    if not files_list:
        print(f"Looking in: {folder_path}")
        print(f"Available paths: {os.listdir(gold_db) if os.path.exists(gold_db) else 'Directory not found'}")
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    
    print(f"Found prediction files: {len(files_list)}")
    df = spark.read.option("mergeSchema", "true").parquet(*files_list)
    return df

def performance_report(y_true, y_pred_binary, y_pred_proba):
    """
    Generate performance metrics for model evaluation
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred_binary), 4),
        "precision": round(precision_score(y_true, y_pred_binary, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_binary, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred_binary, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_pred_proba), 4)
    }

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) for a single numeric feature.
    """
    # Define bins based on expected (reference) distribution
    quantiles = np.percentile(expected.dropna(), np.linspace(0, 100, buckets + 1))
    quantiles = np.unique(quantiles)  # ensure no duplicate edges
    
    # Handle edge case where all values are the same
    if len(quantiles) < 2:
        return 0.0
    
    # Assign bins
    expected_bins = pd.cut(expected, bins=quantiles, include_lowest=True, duplicates='drop')
    actual_bins = pd.cut(actual, bins=quantiles, include_lowest=True, duplicates='drop')

    # Calculate proportions
    expected_dist = expected_bins.value_counts(normalize=True, sort=False)
    actual_dist = actual_bins.value_counts(normalize=True, sort=False)

    # Ensure both distributions have the same index
    all_bins = expected_dist.index.union(actual_dist.index)
    expected_dist = expected_dist.reindex(all_bins, fill_value=0)
    actual_dist = actual_dist.reindex(all_bins, fill_value=0)

    # Avoid divide-by-zero
    expected_dist = expected_dist.replace(0, 0.0001)
    actual_dist = actual_dist.replace(0, 0.0001)

    psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()
    return round(psi, 4)

def check_model_drift(spark, snapshot_date: str, beta: float = 1.5):
    """
    Check for model drift by comparing current batch to reference OOT period from config.
    Handles cases where labels are missing for the current batch.
    """
    # Convert snapshot_date
    snapshot_dt = datetime.strptime(snapshot_date, "%Y-%m-%d").date()

    # --- Define OOT reference window using config ---
    oot_end = snapshot_dt - timedelta(days=1)
    oot_start = snapshot_dt - relativedelta(months=config.oot_period_months)

    cur_start = snapshot_dt  # current batch is for this date

    # --- Read predictions and labels from store ---
    preds_df = read_pred("/opt/airflow/datamart/gold", spark).toPandas()  # Use local function
    
    try:
        labels_df = dg.read_gold_label("/opt/airflow/datamart/gold", snapshot_date, spark).toPandas()
    except:
        labels_df = pd.DataFrame()  # Handle case where no labels exist

    # Convert date columns to proper format
    preds_df['snapshot_date'] = pd.to_datetime(preds_df['snapshot_date']).dt.date
    if not labels_df.empty:
        labels_df['snapshot_date'] = pd.to_datetime(labels_df['snapshot_date']).dt.date

    # --- Reference (OOT) batch ---
    ref_preds = preds_df[(preds_df["snapshot_date"] >= oot_start) & (preds_df["snapshot_date"] <= oot_end)]
    ref_labels = labels_df[(labels_df["snapshot_date"] >= oot_start) & (labels_df["snapshot_date"] <= oot_end)] if not labels_df.empty else pd.DataFrame()

    # --- Current batch ---
    cur_preds = preds_df[preds_df["snapshot_date"] == cur_start]
    cur_labels = labels_df[labels_df["snapshot_date"] == cur_start] if not labels_df.empty else pd.DataFrame()

    print(f"Reference period: {len(ref_preds)} predictions, {len(ref_labels)} labels")
    print(f"Current batch: {len(cur_preds)} predictions, {len(cur_labels)} labels")

    if len(ref_preds) == 0:
        return {"error": "No reference predictions found"}
    
    if len(cur_preds) == 0:
        return {"error": "No current predictions found"}

    # --- Join predictions with labels for reference period ---
    if len(ref_labels) > 0:
        ref_df = pd.merge(ref_preds, ref_labels, on=["customer_id", "snapshot_date"], how="inner")
    else:
        ref_df = ref_preds.copy()
        ref_df["label"] = None

    # --- Join predictions with labels for current period ---
    if len(cur_labels) > 0:
        cur_df = pd.merge(cur_preds, cur_labels, on=["customer_id", "snapshot_date"], how="inner")
    else:
        cur_df = cur_preds.copy()
        cur_df["label"] = None

    # --- Extract arrays using correct column names from your inference script ---
    reference_probs = ref_df["model_predictions"].values          # Raw probabilities
    reference_binary = ref_df["model_predictions_binary"].values  # Binary predictions
    reference_labels = ref_df["label"].values if "label" in ref_df.columns and ref_df["label"].notna().any() else None
    reference_threshold = ref_df["model_threshold"].iloc[0] if len(ref_df) > 0 else 0.5

    current_probs = cur_df["model_predictions"].values
    current_binary = cur_df["model_predictions_binary"].values
    current_labels = cur_df["label"].values if "label" in cur_df.columns and cur_df["label"].notna().any() else None

    # --- 1. Prediction Drift (PSI) ---
    psi_score = calculate_psi(pd.Series(reference_probs), pd.Series(current_probs))
    
    # Also check binary prediction drift
    psi_binary = calculate_psi(pd.Series(reference_binary.astype(float)), 
                              pd.Series(current_binary.astype(float)), buckets=2)
    
    result = {
        "prediction_psi": round(psi_score, 4),
        "prediction_drift": psi_score > 0.1,
        "binary_psi": round(psi_binary, 4),
        "binary_drift": psi_binary > 0.1,
        "model_threshold": reference_threshold,
        "reference_samples": len(ref_df),
        "current_samples": len(cur_df)
    }

    # --- 2. Performance Drift (if labels are available) ---
    if reference_labels is not None and current_labels is not None:
        try:
            # Use both probability and binary predictions for comparison
            ref_perf_prob = performance_report(reference_labels, reference_binary, reference_probs)
            cur_perf_prob = performance_report(current_labels, current_binary, current_probs)

            result.update({
                "ref_perf": ref_perf_prob,
                "cur_perf": cur_perf_prob,
                "perf_drift_auc": round(ref_perf_prob["auc"] - cur_perf_prob["auc"], 4),
                f"perf_drift_f{beta}": round(
                    fbeta_score(reference_labels, reference_binary, beta=beta, zero_division=0) -
                    fbeta_score(current_labels, current_binary, beta=beta, zero_division=0), 4),
                "performance_drift": cur_perf_prob["auc"] < ref_perf_prob["auc"] - 0.05
            })
        except Exception as e:
            logging.warning(f"Error calculating performance metrics: {e}")
    else:
        logging.warning("Labels are not available for comparison. Skipping performance drift calculation.")

    return result

def check_model_consistency(spark, snapshot_date: str):
    """
    Check if model type and version are consistent across predictions
    """
    try:
        preds_df = read_pred("/opt/airflow/datamart/gold", spark).toPandas()
        preds_df['snapshot_date'] = pd.to_datetime(preds_df['snapshot_date']).dt.date
        snapshot_dt = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
        current_preds = preds_df[preds_df["snapshot_date"] == snapshot_dt]
        
        if len(current_preds) == 0:
            return {"error": "No predictions found for the given date"}
        
        # Check model consistency
        unique_models = current_preds["model_name"].unique()
        unique_versions = current_preds["model_version"].unique()
        unique_thresholds = current_preds["model_threshold"].unique()
        
        return {
            "model_types": unique_models.tolist(),
            "model_versions": unique_versions.tolist(),
            "model_thresholds": unique_thresholds.tolist(),
            "model_consistent": len(unique_models) == 1 and len(unique_versions) == 1,
            "total_predictions": len(current_preds),
            "avg_probability": round(current_preds["model_predictions"].mean(), 4),
            "default_rate": round(current_preds["model_predictions_binary"].mean(), 4)
        }
    except Exception as e:
        return {"error": f"Failed to check model consistency: {str(e)}"}

def run_monitoring(spark, snapshot_date: str):
    """
    Run complete monitoring suite
    """
    print(f"Running model monitoring for {snapshot_date}")
    
    try:
        # Check model consistency
        consistency_report = check_model_consistency(spark, snapshot_date)
        print("Model Consistency Report:")
        print(consistency_report)
        
        # Check model drift
        drift_report = check_model_drift(spark, snapshot_date)
        print("\nModel Drift Report:")
        print(drift_report)
        
        # Save reports
        report_path = f"/opt/airflow/reports/monitoring_{snapshot_date.replace('-', '_')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        full_report = {
            "snapshot_date": snapshot_date,
            "timestamp": datetime.now().isoformat(),
            "consistency": consistency_report,
            "drift": drift_report
        }
        
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
            
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Monitoring failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder.appName("ModelMonitoring").getOrCreate()
    run_monitoring(spark, args.snapshotdate)  # Fixed parameter order
    spark.stop()