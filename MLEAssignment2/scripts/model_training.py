import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyspark
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay
from utils import model_monitor
import numpy as np

import config as cg
import pickle
import json

def read_gold_table(table, gold_db, type, spark):
    """
    Helper function to read all partitions of a gold table
    """
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

def train_model(snapshot_date, spark: SparkSession):
    """
    Train model 
    """   
    X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
    y_spark = read_gold_table('label_store', 'datamart/gold', spark)

    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')

    model_train_date_str = snapshot_date # pretend we're training the model at this time

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = cg.train_test_period_months
    config["oot_period_months"] =  cg.oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = cg.oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = cg.train_test_period_months)
    config["train_test_ratio"] = cg.train_test_ratio 

    # Consider data from model training date
    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

    # Create OOT split
    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & (y_model_df['snapshot_date'] <= config['oot_end_date'])]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]

    # Everything else goes into train-test
    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

    X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest, 
                                                        test_size=config['train_test_ratio'], 
                                                        random_state=611, 
                                                        shuffle=True, 
                                                        stratify=y_traintest['label'])

    # Transform data into numpy arrays
    X_train_arr = X_train.drop(columns=['customer_id', 'snapshot_date']).values
    X_test_arr = X_test.drop(columns=['customer_id', 'snapshot_date']).values
    X_oot_arr = X_oot.drop(columns=['customer_id', 'snapshot_date']).values

    y_train_arr = y_train['label'].values
    y_test_arr = y_test['label'].values
    y_oot_arr = y_oot['label'].values

    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train_arr)
    X_test_arr = scaler.transform(X_test_arr)
    X_oot_arr = scaler.transform(X_oot_arr)

    # Train models
    clf = LogisticRegression()
    clf.fit(X_train_arr, y_train_arr)

    # Save model
    model_path = f"model_bank/model_{snapshot_date}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({'model': clf, 'scaler': scaler}, f)

    # Predict and evaluate
    y_pred_proba_train = clf.predict_proba(X_train_arr)[:, 1]
    train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)

    y_pred_proba_test = clf.predict_proba(X_test_arr)[:, 1]
    test_auc = roc_auc_score(y_test_arr, y_pred_proba_test)

    y_pred_proba_oot = clf.predict_proba(X_oot_arr)[:, 1]
    oot_auc = roc_auc_score(y_oot_arr, y_pred_proba_oot)

    # F2 score across thresholds
    thresholds = np.arange(0.0, 1.0, 0.01)
    beta = 1.5
    f1_scores_train = [fbeta_score(y_train_arr, y_pred_proba_train > t, beta=beta) for t in thresholds]
    f1_scores_test = [fbeta_score(y_test_arr, y_pred_proba_test > t, beta=beta) for t in thresholds]
    f1_scores_oot = [fbeta_score(y_oot_arr, y_pred_proba_oot > t, beta=beta) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores_train)]

    metrics = {
    "train_auc": train_auc,
    "test_auc": test_auc,
    "oot_auc": oot_auc,
    f"train_f{beta}": max(f1_scores_train),
    f"test_f{beta}": max(f1_scores_test),
    f"oot_f{beta}": max(f1_scores_oot),
    "best_threshold": float(best_threshold)
    }

    metrics_path = f"model_bank/model_{snapshot_date}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def pick_model_and_deploy(job_id):
    """
    Selects the newly trained model based on job_id and deploys it
    if it's better than the current production model.
    """
    new_model_path = f"model_bank/model_{job_id}.pkl"
    new_metrics_path = f"model_bank/model_{job_id}_metrics.json"
    
    prod_model_path = "model_bank/model_production.pkl"
    prod_metrics_path = "model_bank/model_production_metrics.json"

    # Check if the new model and metrics exist
    if not os.path.exists(new_model_path) or not os.path.exists(new_metrics_path):
        raise FileNotFoundError("Trained model or metrics not found.")

    # Load new model metrics
    with open(new_metrics_path, "r") as f:
        new_metrics = json.load(f)
    new_score = new_metrics.get("oot_auc", 0)  # or "oot_f1"

    # Load production model metrics
    if os.path.exists(prod_metrics_path):
        with open(prod_metrics_path, "r") as f:
            prod_metrics = json.load(f)
        prod_score = prod_metrics.get("oot_auc", 0)
    else:
        prod_score = 0  # No model deployed yet

    # Compare and update if better
    if new_score > prod_score:
        shutil.copyfile(new_model_path, prod_model_path)
        shutil.copyfile(new_metrics_path, prod_metrics_path)
        print(f"Deployed model_{job_id}.pkl as new production model (score improved from {prod_score:.4f} to {new_score:.4f})")
        return prod_model_path
    else:
        print(f"ℹModel_{job_id}.pkl not deployed. Score {new_score:.4f} did not exceed production score {prod_score:.4f}")
        return prod_model_path 
    
def load_production_model():
    """
    Helper function to load the production model
    """
    prod_model_path = "model_bank/model_production.pkl"
    prod_metrics_path = "model_bank/model_production_metrics.json"
    
    if not os.path.exists(prod_model_path):
        raise FileNotFoundError("No production model found.")
    
    with open(prod_model_path, "rb") as f:
        artifacts = pickle.load(f)
        model = artifacts['model']
        scaler = artifacts['scaler']
    with open(prod_metrics_path, "r") as f:
        metrics = json.load(f)
    
    return model, scaler, metrics