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

from model_training import load_production_model

# to call this script: python model_train.py --snapshotdate "2024-09-01"
def model_pred(snapshotdate, spark: SparkSession):
    print('\n\n---starting job---\n\n')
    
    # Load production model directly
    model, scaler, metrics = load_production_model()
    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = "production_model"  # Fixed: use string not model object
    
    pprint.pprint(config)
    print("Production model loaded successfully!")

    # --- load feature store ---
    gold_db = "datamart/gold"
    partition_name = snapshotdate.replace('-','_') + '.parquet'
    feature_filepath = os.path.join(gold_db, 'online_feature_store', partition_name)
     
    # Load parquet file
    features_store_sdf = spark.read.parquet(feature_filepath)
        
    # extract feature store for specific date
    features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date_str"]))
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date_str"])
    
    features_pdf = features_sdf.toPandas()
    
    # --- preprocess data for modeling ---
    # prepare X_inference
    feature_cols = cg.PREDICTORS
    X_inference = features_pdf[feature_cols]
    
    # apply transformer - use scaler from load_production_model()
    X_inference_scaled = scaler.transform(X_inference)
    
    print('X_inference', X_inference_scaled.shape[0])

    # --- model prediction inference ---
    # Use model from load_production_model()
    y_inference = model.predict_proba(X_inference_scaled)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["customer_id","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    
    # --- save model inference to datamart gold table ---
    # create gold directory
    gold_directory = f"datamart/gold/model_predictions/{config['model_name']}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table
    partition_name = config["model_name"] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    # Removed --modelname since we're using production model
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder.appName("ModelInference").getOrCreate()
    
    # Call main with correct parameters
    model_pred(args.snapshotdate, spark)
    
    # Clean up
    spark.stop()