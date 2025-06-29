import os
import argparse
import glob
import pyspark
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_date
from pyspark.sql.types import DateType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql.types import StructType, StructField, DateType, StringType, IntegerType
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, fbeta_score
import config as cg  
import matplotlib.pyplot as plt
import pickle
import xgboost as xgb
import pandas as pd
import json
import numpy as np
import logging



def read_gold_table(table, gold_db, spark):
    """
    Helper function to read all partitions of a gold table
    """
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

mlflow.set_tracking_uri(uri="http://mlflow:5000/")
mlflow.set_experiment("Loan-default-model")
client = MlflowClient("http://mlflow:5000/")
# registered_model_name = "xgb-loan-default"

def xgb_train(snapshot_date,spark:SparkSession):

    X_spark = read_gold_table('feature_store', '/opt/airflow/datamart/gold', spark)
    y_spark = read_gold_table('label_store', '/opt/airflow/datamart/gold', spark)
    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')

    model_train_date_str = snapshot_date
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
    y_traintest = y_model_df[y_model_df['snapshot_date'] < config['oot_start_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]


    X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest, 
                                                        test_size=config['train_test_ratio'], 
                                                        random_state=611, 
                                                        shuffle=True, 
                                                        stratify=y_traintest['label'])
    # Data filtering
    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & 
                      (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

    # Create OOT split
    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & 
                       (y_model_df['snapshot_date'] <= config['oot_end_date'])]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]

    # Train-test split
    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    y_traintest = y_model_df[y_model_df['snapshot_date'] < config['oot_start_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

    X_train, X_test, y_train, y_test = train_test_split(
        X_traintest, y_traintest, 
        test_size=config['train_test_ratio'], 
        random_state=42, 
        shuffle=True, 
        stratify=y_traintest['label']
    )
    feature_columns = X_train.drop(columns=['customer_id', 'snapshot_date']).columns.tolist()
    # Print dataset information
    print(f'X_train: {X_train.shape[0]} samples')
    print(f'X_test: {X_test.shape[0]} samples')
    print(f'X_oot: {X_oot.shape[0]} samples')
    print(f'y_train: {y_train.shape[0]} samples, bad rate: {y_train["label"].mean():.3f}')
    print(f'y_test: {y_test.shape[0]} samples, bad rate: {y_test["label"].mean():.3f}')
    print(f'y_oot: {y_oot.shape[0]} samples, bad rate: {y_oot["label"].mean():.3f}')

    # Validate bad rates are similar
    train_bad_rate = y_train["label"].mean()
    test_bad_rate = y_test["label"].mean()
    oot_bad_rate = y_oot["label"].mean()
    
    if abs(train_bad_rate - test_bad_rate) > 0.05 or abs(train_bad_rate - oot_bad_rate) > 0.05:
        return print("WARNING: Bad rates differ significantly between splits. Consider investigating data distribution.")
      
    # Prepare arrays
    X_train_arr = X_train.drop(columns=['customer_id', 'snapshot_date']).values
    X_test_arr = X_test.drop(columns=['customer_id', 'snapshot_date']).values
    X_oot_arr = X_oot.drop(columns=['customer_id', 'snapshot_date']).values

    y_train_arr = y_train['label'].values
    y_test_arr = y_test['label'].values
    y_oot_arr = y_oot['label'].values

    # Scale features
    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train_arr)
    X_test_processed = scaler.transform(X_test_arr)
    X_oot_processed = scaler.transform(X_oot_arr)

    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)
    
    # Define the hyperparameter space to search
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3, 6],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)
    
    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=20,  # Number of iterations for random search
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )
    
    # Perform the random search
    random_search.fit(X_train_processed, y_train_arr)
    
    # Output the best parameters and best score
    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC score: ", random_search.best_score_)

    best_model = random_search.best_estimator_
    
    # Evaluate the model on the train set
    best_model = random_search.best_estimator_
    y_pred_proba_train = best_model.predict_proba(X_train_processed)[:, 1]
    train_auc_score = roc_auc_score(y_train_arr, y_pred_proba_train)
    print("Train AUC score: ", train_auc_score)
    
    # Evaluate the model on the test set
    y_pred_proba_test = best_model.predict_proba(X_test_processed)[:, 1]
    test_auc_score = roc_auc_score(y_test_arr, y_pred_proba_test)
    print("Test AUC score: ", test_auc_score)
    
    # Evaluate the model on the oot set
    y_pred_proba_oot = best_model.predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot_arr, y_pred_proba_oot)
    print("OOT AUC score: ", oot_auc_score)
    
    print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
    print("Test GINI score: ", round(2*test_auc_score-1,3))
    print("OOT GINI score: ", round(2*oot_auc_score-1,3))

    thresholds = np.arange(0.01, 1.0, 0.01)
    beta = 1.5

    f1_scores_train = [fbeta_score(y_train_arr, (y_pred_proba_train > t).astype(int), beta=beta, zero_division=0) 
                    for t in thresholds]
    f1_scores_test = [fbeta_score(y_test_arr, (y_pred_proba_test > t).astype(int), beta=beta, zero_division=0) 
                    for t in thresholds]
    f1_scores_oot = [fbeta_score(y_oot_arr, (y_pred_proba_oot > t).astype(int), beta=beta, zero_division=0) 
                    for t in thresholds]

    best_threshold = thresholds[np.argmax(f1_scores_train)]

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, f1_scores_train, label='Train', linewidth=2, color='blue')
    plt.plot(thresholds, f1_scores_test, label='Test', linewidth=2, color='orange')
    plt.plot(thresholds, f1_scores_oot, label='OOT', linewidth=2, color='green')

    # Mark the best threshold
    plt.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2, 
            label=f'Best Threshold = {best_threshold:.3f}')
    plt.scatter([best_threshold], [max(f1_scores_train)], color='red', s=100, zorder=5)

    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel(f'F-beta Score (β={beta})', fontsize=12)
    plt.title(f'F-beta Score vs Probability Threshold\nXGBoost Model - {snapshot_date}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, max(max(f1_scores_train), max(f1_scores_test), max(f1_scores_oot)) * 1.1)

    # Add text annotations
    plt.text(0.7, 0.8, f'Max Train F-beta: {max(f1_scores_train):.3f}', 
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    plt.tight_layout()

    # Save plot
    plot_filename = f"fbeta_threshold_plot_{snapshot_date}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

    with mlflow.start_run(run_name=f"xgb_{snapshot_date}"):
            mlflow.log_params(random_search.best_params_)
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("snapshot_date", snapshot_date)
            mlflow.log_param("best_threshold", best_threshold)
            mlflow.log_param("beta_value", beta)

            # Log metrics for all datasets
            for name, X, y in [
                ("train", X_train_processed, y_train_arr),
                ("test", X_test_processed, y_test_arr),
                ("oot", X_oot_processed, y_oot_arr)
            ]:
                y_pred = best_model.predict(X)
                y_proba = best_model.predict_proba(X)[:, 1]
                y_pred_threshold = (y_proba > best_threshold).astype(int)
                auc = roc_auc_score(y, y_proba)
                f1 = f1_score(y, y_pred)
                fbeta_optimal = fbeta_score(y, y_pred_threshold, beta=beta, zero_division=0)
                
                mlflow.log_metric(f"{name}_auc", auc)
                mlflow.log_metric(f"{name}_f1", f1)
                mlflow.log_metric(f"{name}_gini", 2*auc-1)
                mlflow.log_metric(f"{name}_fbeta_optimal", fbeta_optimal)

            mlflow.log_artifact(plot_filename)

            fbeta_data = {
                "thresholds": thresholds.tolist(),
                "fbeta_train": f1_scores_train,
                "fbeta_test": f1_scores_test,
                "fbeta_oot": f1_scores_oot,
                "best_threshold": float(best_threshold),
                "beta_value": float(beta)
            }
            
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            fbeta_json_filename = os.path.join(reports_dir, f"fbeta_scores_{snapshot_date}.json")
            with open(fbeta_json_filename, 'w') as f:
                json.dump(fbeta_data, f, indent=2)
            mlflow.log_artifact(fbeta_json_filename, artifact_path="model")
            with open("best_threshold.txt", "w") as f:
                f.write(str(best_threshold))
            mlflow.log_artifact("best_threshold.txt", artifact_path="model")
            with open("feature_columns.pkl", "wb") as f:
                pickle.dump(feature_columns, f)
            mlflow.log_artifact("feature_columns.pkl", artifact_path="model")
            
            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact("scaler.pkl", artifact_path="model")

    signature = infer_signature(X_train_processed, best_model.predict(X_train_processed))
    mlflow.sklearn.log_model(
    sk_model=best_model,
    name="model",
    registered_model_name='Loan-default-prod',
    signature=signature
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True, help="Training date (YYYY-MM-DD)")
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("XGBoost_Training") \
        .master("local[2]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    result = xgb_train(args.snapshotdate, spark)
    spark.catalog.clearCache()
    spark.stop()

