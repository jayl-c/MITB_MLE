import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import config as cg

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse
import pprint
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def load_best():
    """Load the deployed best model"""
    best_path = "/opt/airflow/model_bank/production/best_model.pkl"
    if os.path.exists(best_path):
        return joblib.load(best_path)
    else:
        print("No champion model found")
        return None

def model_pred(snapshotdate, spark: SparkSession):
    
    # Load the champion model
    champion_package = load_best()
    
    # Extract components from champion package
    model = champion_package['model']
    scaler = champion_package['scaler']
    feature_cols = champion_package['feature_columns']
    metrics = champion_package['performance']
    best_params = champion_package['best_params']
    
    # Determine model type
    model_type = type(model).__name__
    if 'XGBClassifier' in model_type:
        model_type_name = 'xgb'
    elif 'LogisticRegression' in model_type:
        model_type_name = 'logreg'
    else:
        model_type_name = model_type.lower()

    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = f"{model_type_name}" 
    config["model_version"] = f"{model_type_name}_{datetime.now().strftime('%Y%m%d')}"
    
    pprint.pprint(config)
    print("Production model loaded successfully!")
    print(f"Using {model_type_name} with OOT AUC: {metrics['oot_auc']:.4f}")
    print(f"OOT F1 Score: {metrics['max_fbeta_oot']:.4f}")
    print(f"Best Threshold: {metrics['best_threshold']:.4f}")

    # --- load feature store ---
    gold_db = "/opt/airflow/datamart/gold"
    partition_name = snapshotdate.replace('-','_') + '.parquet'
    feature_filepath = os.path.join(gold_db, 'feature_store', partition_name)
     
    # Load parquet file
    features_store_sdf = spark.read.parquet(feature_filepath)
        
    # extract feature store for specific date
    features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date_str"]))
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date_str"])
    
    features_pdf = features_sdf.toPandas()
    
    # Ensure we have all required features in correct order
    missing_features = set(feature_cols) - set(features_pdf.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
       
    X_inference = features_pdf[feature_cols]
    
    # apply transformer - use scaler from trained model
    X_inference_scaled = scaler.transform(X_inference)
    
    print('X_inference', X_inference_scaled.shape[0])

    # --- model prediction inference ---
    # Get probability predictions
    y_inference_proba = model.predict_proba(X_inference_scaled)[:, 1]
    
    # Get binary predictions using the best threshold from training
    y_inference_binary = (y_inference_proba > metrics['best_threshold']).astype(int)
    
    # prepare output with both probability and binary predictions
    y_inference_pdf = features_pdf[["customer_id","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_version"] = config["model_version"]
    y_inference_pdf["model_type"] = model_type_name
    y_inference_pdf["model_predictions"] = y_inference_proba  # Keep your original column name
    y_inference_pdf["model_predictions_binary"] = y_inference_binary
    y_inference_pdf["model_threshold"] = metrics['best_threshold']
    y_inference_pdf["model_oot_auc"] = metrics['oot_auc']
    y_inference_pdf["model_oot_f1"] = metrics['max_fbeta_oot']
    
    # --- save model inference to datamart gold table ---
    # create gold directory
    gold_directory = f"/opt/airflow/datamart/gold/model_predictions/{config['model_name']}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table with more detailed naming
    # partition_name = config["model_name"] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    partition_name = config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
      
    print('\n\n---completed job---\n\n')
    
    return y_inference_pdf

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder.appName("ModelInference").getOrCreate()
    
    # Call main with correct parameters
    model_pred(args.snapshotdate, spark)
    
    # Clean up
    spark.stop()