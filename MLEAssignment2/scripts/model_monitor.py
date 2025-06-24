from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import os
import datetime

import config

from sklearn.ensemble import BaseEnsemble
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

import logging
logging.basicConfig(level=logging.INFO)
logging.info("Data drift report saved at: %s", config.PATH_DIR_REPORT)

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



def check_data_drift(ref_df:pd.DataFrame, latest_df:pd.DataFrame):
    """
    Using Evidently to check for data drift 
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=latest_df)

    # Convert report to dict to extract drift information
    results = report.as_dict()
    
    # Check drift status
    drift_info = results["metrics"][0]["result"]
    drift_detected = drift_info.get("dataset_drift", False)
    n_drifted = drift_info.get("n_drifted_features", 0)
    retrain = drift_detected or (n_drifted > 0)

    # Get PSI for numeric features
    psi_report = {}
    for col in config.PREDICTORS:
        if pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(latest_df[col]):
            psi = calculate_psi(ref_df[col].dropna(), latest_df[col].dropna())
            psi_report[col] = round(psi, 4)

    # Save report as HTML
    os.makedirs(config.PATH_DIR_REPORT, exist_ok=True)
    report_path = os.path.join(config.PATH_DIR_REPORT, f"{job_id}_data_drift_report.html")
    
    try:
        report.save_html(report_path)
        print(f"[INFO] Data drift report saved at: {report_path}")
    except Exception as e:
        print(f"[WARNING] Could not save drift report: {e}")

    return {
        "report": report,
        "retrain": retrain,
        "drifted_features": drift_info.get("drift_by_columns", []),
        "n_drifted": n_drifted,
        "results": results
    }

def check_model_drift():
    pass

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




