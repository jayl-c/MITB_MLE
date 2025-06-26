from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import os
import datetime
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import config
from scripts.utils import data_processing_gold_table as dg

from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

import logging

logging.basicConfig(level=logging.INFO)
logging.info("Data drift report saved at: %s", config.PATH_DIR_REPORT)

def read_pred(gold_db, spark):
    """
    Helper function to read all partitions of model predictions
    """
    folder_path = os.path.join(gold_db, 'model_predictions')
    
    files_list = glob.glob(os.path.join(folder_path, '**', '*.parquet'), recursive=True)
   
    if not files_list:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) for a single numeric feature.
    """
    # Define bins based on expected (reference) distribution
    quantiles = np.percentile(expected.dropna(), np.linspace(0, 100, buckets + 1))
    quantiles = np.unique(quantiles)  # ensure no duplicate edges
    
    # Assign bins
    expected_bins = pd.cut(expected, bins=quantiles, include_lowest=True)
    actual_bins = pd.cut(actual, bins=quantiles, include_lowest=True)

    # Calculate proportions
    expected_dist = expected_bins.value_counts(normalize=True, sort=False)
    actual_dist = actual_bins.value_counts(normalize=True, sort=False)

    # Avoid divide-by-zero
    expected_dist = expected_dist.replace(0, 0.0001)
    actual_dist = actual_dist.replace(0, 0.0001)

    psi = ((expected_dist - actual_dist) * np.log(expected_dist / actual_dist)).sum()
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
    cur_end = snapshot_dt    # daily batch

    # --- Read predictions and labels from store ---
    preds_df = dg.read_pred("datamart/gold", spark).toPandas()
    labels_df = dg.read_gold_labels("datamart/gold", spark).toPandas()

    # --- Reference (OOT) batch ---
    ref_preds = preds_df[(preds_df["snapshot_date"] >= oot_start) & (preds_df["snapshot_date"] <= oot_end)]
    ref_labels = labels_df[(labels_df["snapshot_date"] >= oot_start) & (labels_df["snapshot_date"] <= oot_end)]

    # --- Current batch ---
    cur_preds = preds_df[preds_df["snapshot_date"] == cur_start]
    cur_labels = labels_df[labels_df["snapshot_date"] == cur_start] if not labels_df.empty else None

    # --- Join predictions with labels ---
    ref_df = pd.merge(ref_preds, ref_labels, on="customer_id")
    cur_df = pd.merge(cur_preds, cur_labels, on="customer_id") if cur_labels is not None else cur_preds

    # --- Extract arrays ---
    reference_probs = ref_df["model_predictions"].values
    reference_labels = ref_df["label"].values

    current_probs = cur_df["model_predictions"].values
    current_labels = cur_df["label"].values if "label" in cur_df.columns else None

    # --- 1. Prediction Drift (PSI) ---
    psi_score = calculate_psi(pd.Series(reference_probs), pd.Series(current_probs))
    result = {
        "prediction_psi": round(psi_score, 4),
        "prediction_drift": psi_score > 0.1
    }

    # --- 2. Performance Drift (if labels are available) ---
    if current_labels is not None:
        threshold = 0.5
        ref_pred = reference_probs > threshold
        cur_pred = current_probs > threshold

        ref_perf = performance_report(reference_labels, ref_pred, reference_probs)
        cur_perf = performance_report(current_labels, cur_pred, current_probs)

        result.update({
            "ref_perf": ref_perf,
            "cur_perf": cur_perf,
            "perf_drift_auc": round(ref_perf["auc"] - cur_perf["auc"], 4),
            f"perf_drift_f{beta}": round(
                fbeta_score(reference_labels, ref_pred, beta=beta) -
                fbeta_score(current_labels, cur_pred, beta=beta), 4),
            "performance_drift": cur_perf["auc"] < ref_perf["auc"] - 0.05  # adjustable
        })
    else:
        logging.warning("Labels are not available for the current batch. Skipping performance drift calculation.")

    return result

def performance_report(y_true: np.array, y_pred: np.array, y_prob: np.array) -> dict:
    """
    Generate performance report for a model.
    """

    report = dict()
    report["dataset size"] = y_true.shape[0]
    report["positive rate"] = y_true.sum() / y_true.shape[0]
    report["accuracy"] = accuracy_score(y_true, y_pred)
    report["f1"] = f1_score(y_true, y_pred)
    report["precision"] = precision_score(y_true, y_pred)
    report["recall"] = recall_score(y_true, y_pred)
    report["auc"] = roc_auc_score(y_true, y_prob)
    return report




