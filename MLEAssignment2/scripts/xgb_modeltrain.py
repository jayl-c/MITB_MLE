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
    
            fbeta_json_filename = f"fbeta_scores_{snapshot_date}.json"
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


    
# def read_gold_table(table, gold_db, spark):
#     folder_path = os.path.join('/opt/airflow', gold_db, table)
#     files_list = glob.glob(os.path.join(folder_path, '**', '*.parquet'), recursive=True)
#     if not files_list:
#         raise FileNotFoundError(f"No Parquet files found in {folder_path}")
    
#     # ✅ Read files individually and filter empty ones
#     valid_dfs = []
    
#     for file_path in files_list:
#         try:
#             temp_df = spark.read.parquet(file_path)
            
#             # ✅ Check if DataFrame is empty
#             if temp_df.count() > 0:
#                 print(f"✅ Loaded: {os.path.basename(file_path)} ({temp_df.count()} rows)")
#                 valid_dfs.append(temp_df)
#             else:
#                 print(f"⚠️ Skipping empty file: {os.path.basename(file_path)}")
                
#         except Exception as e:
#             print(f"❌ Error reading {file_path}: {e}")
#             continue
    
#     if not valid_dfs:
#         raise ValueError(f"No valid data found in {table}")
    
#     # ✅ Union all valid DataFrames
#     df = valid_dfs[0]
#     for temp_df in valid_dfs[1:]:
#         df = df.union(temp_df)
    
#     # ✅ Standardize snapshot_date column
#     if 'snapshot_date' in df.columns:
#         df = df.withColumn("snapshot_date", to_date(col("snapshot_date").cast("string")))
    
#     print(f"📊 Final {table}: {df.count()} total rows")
#     return df

# def prepare_data_splits(X_spark, y_spark, config):
#     df = X_spark.join(y_spark, on=['customer_id', 'snapshot_date'], how='inner')
#     df = df.withColumn('snapshot_date', col('snapshot_date').cast(DateType()))

#     model_df = df.filter(
#         (col('snapshot_date') >= lit(config['train_test_start_date'])) &
#         (col('snapshot_date') <= lit(config['model_train_date']))
#     )

#     oot_df = model_df.filter(
#         (col('snapshot_date') >= lit(config['oot_start_date'])) &
#         (col('snapshot_date') <= lit(config['oot_end_date']))
#     )

#     traintest_df = model_df.filter(
#         col('snapshot_date') <= lit(config['train_test_end_date'])
#     )

#     return traintest_df, oot_df

# # def train_xgb_model(spark, snapshot_date: str, model_name: str):
# #     """
# #     Enhanced XGBoost training with best practices from both scripts
# #     """
    
# #     # MLflow setup
# #     mlflow.set_tracking_uri(uri="http://mlflow:5000")
# #     mlflow.set_experiment("Loan Default Prediction")
# #     client = MlflowClient("http://mlflow:5000")
# #     registered_model_name = "loan_default_prediction"
    
# #     with mlflow.start_run(run_name=f"{model_name}_{snapshot_date}") as run:
        
# #         print(f"🚀 Training Enhanced XGBoost model for {snapshot_date}")
# #         print("=" * 60)
        
# #         # ✅ Configuration setup
# #         config = {
# #             "model_name": model_name,
# #             "model_train_date_str": snapshot_date,
# #             "train_test_period_months": getattr(cg, 'train_test_period_months', 12),
# #             "oot_period_months": getattr(cg, 'oot_period_months', 3),
# #             "train_test_ratio": getattr(cg, 'train_test_ratio', 0.2)
# #         }
        
# #         config["model_train_date"] = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
# #         config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
# #         config["oot_start_date"] = config['model_train_date'] - relativedelta(months=config['oot_period_months'])
# #         config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
# #         config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=config['train_test_period_months'])
        
# #         # Log configuration
# #         mlflow.log_params({
# #             "model_type": "XGBoostClassifier",
# #             "snapshot_date": snapshot_date,
# #             "train_test_period_months": config["train_test_period_months"],
# #             "oot_period_months": config["oot_period_months"],
# #             "train_test_ratio": config["train_test_ratio"]
# #         })
        
# #         print(f"📅 Date ranges:")
# #         print(f"   Train-test: {config['train_test_start_date']} to {config['train_test_end_date']}")
# #         print(f"   OOT: {config['oot_start_date']} to {config['oot_end_date']}")
        
# #         # ✅ Data loading with robust error handling
# #         try:
# #             X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
# #             y_spark = read_gold_table('label_store', 'datamart/gold', spark)
            
# #             traintest_df, oot_df = prepare_data_splits(X_spark, y_spark, config)
            
# #             # Convert to Pandas for XGBoost (hybrid approach)
# #             traintest_pdf = traintest_df.toPandas().dropna()
# #             oot_pdf = oot_df.toPandas().dropna()
            
# #             print(f"📊 Final datasets - Train/Test: {len(traintest_pdf)}, OOT: {len(oot_pdf)}")
            
# #         except Exception as e:
# #             print(f"❌ Data loading failed: {e}")
# #             return None, None, None
        
# #         # ✅ Feature preparation
# #         excluded_cols = {"customer_id", "snapshot_date", "label", "label_def", "loan_id"}
# #         feature_cols = [col for col in traintest_pdf.columns if col not in excluded_cols]
        
# #         # Split train/test
# #         X, y = traintest_pdf[feature_cols], traintest_pdf["label"]
# #         X_oot, y_oot = oot_pdf[feature_cols], oot_pdf["label"]
        
# #         X_train, X_test, y_train, y_test = train_test_split(
# #             X, y, test_size=config["train_test_ratio"], 
# #             random_state=42, stratify=y
# #         )
        
# #         print(f"📊 Final splits - Train: {len(X_train)}, Test: {len(X_test)}, OOT: {len(X_oot)}")
        
# #         # ✅ Preprocessing with MLflow logging
# #         scaler = StandardScaler()
# #         X_train_scaled = scaler.fit_transform(X_train)
# #         X_test_scaled = scaler.transform(X_test)
# #         X_oot_scaled = scaler.transform(X_oot)
        
# #         # Save scaler
# #         with open("standard_scaler.pkl", "wb") as f:
# #             pickle.dump(scaler, f)
# #         mlflow.log_artifact("standard_scaler.pkl")
        
# #         # ✅ Initial model for feature importance (like reference script)
# #         print("🔍 Calculating feature importance...")
# #         initial_model = XGBClassifier(
# #             n_estimators=100, max_depth=6, learning_rate=0.1,
# #             random_state=42, use_label_encoder=False
# #         )
# #         initial_model.fit(X_train_scaled, y_train)
        
# #         # Feature importance analysis
# #         importance_df = pd.DataFrame({
# #             "feature": feature_cols,
# #             "importance": initial_model.feature_importances_
# #         }).sort_values(by="importance", ascending=False)
        
# #         # Save feature importance plot
# #         os.makedirs("output/plots", exist_ok=True)
# #         plt.figure(figsize=(10, 8))
# #         plt.barh(importance_df["feature"][:20], importance_df["importance"][:20])
# #         plt.xlabel("Importance")
# #         plt.ylabel("Feature")
# #         plt.title("Top 20 Feature Importances")
# #         plt.gca().invert_yaxis()
# #         plt.tight_layout()
# #         plt.savefig("output/plots/feature_importance.png", dpi=300)
# #         mlflow.log_artifact("output/plots/feature_importance.png")
        
# #         # ✅ Feature selection (from reference script)
# #         importance_cutoff = 0.01
# #         top_features_df = importance_df.head(20)
# #         selected_features = top_features_df[top_features_df["importance"] >= importance_cutoff]["feature"].tolist()
        
# #         print(f"🎯 Selected {len(selected_features)} features with importance >= {importance_cutoff}")
        
# #         # Save selected features
# #         with open("selected_features.json", "w") as f:
# #             json.dump(selected_features, f)
# #         mlflow.log_artifact("selected_features.json")
        
# #         # Apply feature selection
# #         selected_indices = [feature_cols.index(feat) for feat in selected_features]
# #         X_train_final = X_train_scaled[:, selected_indices]
# #         X_test_final = X_test_scaled[:, selected_indices]
# #         X_oot_final = X_oot_scaled[:, selected_indices]
        
# #         # ✅ Hyperparameter tuning (from reference script)
# #         print("🎯 Starting hyperparameter tuning...")
        
# #         param_dist = {
# #             'n_estimators': [50, 100, 200],
# #             'max_depth': [3, 6, 9],
# #             'learning_rate': [0.01, 0.1, 0.2],
# #             'subsample': [0.8, 0.9, 1.0],
# #             'colsample_bytree': [0.8, 0.9, 1.0],
# #             'gamma': [0, 0.1, 0.2],
# #             'min_child_weight': [1, 3, 5],
# #             'reg_alpha': [0, 0.1, 1],
# #             'reg_lambda': [1, 1.5, 2]
# #         }
        
# #         xgb_clf = XGBClassifier(objective="binary:logistic", random_state=42, use_label_encoder=False)
        
# #         random_search = RandomizedSearchCV(
# #             estimator=xgb_clf,
# #             param_distributions=param_dist,
# #             scoring=make_scorer(roc_auc_score),
# #             n_iter=50,  # Reduced for faster execution
# #             cv=3,
# #             verbose=1,
# #             random_state=42,
# #             n_jobs=-1
# #         )
        
# #         random_search.fit(X_train_final, y_train)
# #         best_model = random_search.best_estimator_
        
# #         # Log best parameters
# #         mlflow.log_params(random_search.best_params_)
        
# #         # ✅ Model evaluation
# #         print("📊 Evaluating model performance...")
        
# #         metrics = {}
# #         for dataset_name, X_data, y_data in [
# #             ("train", X_train_final, y_train),
# #             ("test", X_test_final, y_test),
# #             ("oot", X_oot_final, y_oot),
# #         ]:
# #             y_pred_proba = best_model.predict_proba(X_data)[:, 1]
# #             auc = roc_auc_score(y_data, y_pred_proba)
# #             metrics[f"{dataset_name}_auc"] = auc
# #             mlflow.log_metric(f"{dataset_name}_auc", auc)
# #             print(f"   {dataset_name.upper()} AUC: {auc:.4f}")
        
# #         # ✅ Model comparison and registration (from reference script)
# #         new_model_score = metrics["test_auc"]
# #         stability_score = metrics["oot_auc"]
# #         mlflow.log_metric("comparison_metric", new_model_score)
        
# #         # Log model
# #         mlflow.xgboost.log_model(best_model, artifact_path="model")
# #         model_uri = f"runs:/{run.info.run_id}/model"
        
# #         # ✅ Intelligent model registration
# #         try:
# #             prod_versions = client.get_latest_versions(registered_model_name, stages=["Production"])
            
# #             if len(prod_versions) > 0:
# #                 prod_version = prod_versions[0]
# #                 prod_run_id = prod_version.run_id
                
# #                 # Compare with production model
# #                 prod_metric_history = client.get_metric_history(prod_run_id, "comparison_metric")
# #                 if prod_metric_history:
# #                     old_model_score = prod_metric_history[-1].value
# #                     print(f"🔍 Model comparison - New: {new_model_score:.4f}, Production: {old_model_score:.4f}")
                    
# #                     if new_model_score > old_model_score and stability_score > 0.60:
# #                         print("✅ New model is better. Registering...")
# #                         result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
# #                         client.transition_model_version_stage(
# #                             name=registered_model_name,
# #                             version=result.version,
# #                             stage="Staging",
# #                             archive_existing_versions=True
# #                         )
# #                     else:
# #                         print("❌ New model is not better. Skipping registration.")
# #                 else:
# #                     print("🆕 No production metrics found. Registering new model...")
# #                     result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
# #             else:
# #                 print("🆕 No production model yet. Registering new model...")
# #                 result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
                
# #         except Exception as e:
# #             print(f"⚠️ Registration error: {e}")
# #             result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        
# #         print(f"\n🎉 Training completed successfully!")
# #         print(f"   Best parameters: {random_search.best_params_}")
# #         print(f"   Test AUC: {metrics['test_auc']:.4f}")
# #         print(f"   OOT AUC: {metrics['oot_auc']:.4f}")
# #         print(f"   MLflow Run ID: {run.info.run_id}")
        
# #         return best_model, config, metrics

# # if __name__ == "__main__":
# #     import argparse
    
# #     parser = argparse.ArgumentParser(description='Enhanced XGBoost training')
# #     parser.add_argument("--snapshotdate", type=str, required=True, help="Training date (YYYY-MM-DD)")
# #     parser.add_argument("--modelname", type=str, default="xgb_enhanced", help="Model name")
# #     args = parser.parse_args()

# #     # Initialize Spark
# #     spark = SparkSession.builder \
# #         .appName("Enhanced_XGBoost_Training") \
# #         .master("local[*]") \
# #         .config("spark.sql.adaptive.enabled", "true") \
# #         .getOrCreate()
    
# #     spark.sparkContext.setLogLevel("ERROR")
    
# #     try:
# #         model, config, metrics = train_xgb_model(
# #             spark=spark,
# #             snapshot_date=args.snapshotdate,
# #             model_name=args.modelname
# #         )
        
# #         if model is not None:
# #             print("✅ Enhanced training completed successfully!")
# #         else:
# #             print("❌ Training failed!")
            
# #     except Exception as e:
# #         print(f"❌ Script error: {e}")
# #         import traceback
# #         traceback.print_exc()
# #     finally:
# #         spark.stop()


# def train_xgb_model_pyspark(spark, snapshot_date: str, model_name: str):
#     config = {
#         "model_name": model_name,
#         "model_train_date_str": snapshot_date,
#         "train_test_period_months": cg.train_test_period_months,
#         "oot_period_months": cg.oot_period_months,
#         "model_train_date": datetime.strptime(snapshot_date, "%Y-%m-%d").date()
#     }
#     config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
#     config["oot_start_date"] = config['model_train_date'] - relativedelta(months=cg.oot_period_months)
#     config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
#     config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=cg.train_test_period_months)
#     config["train_test_ratio"] = cg.train_test_ratio

#     X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
#     y_spark = read_gold_table('label_store', 'datamart/gold', spark)
#     traintest_df, oot_df = prepare_data_splits(X_spark, y_spark, config)
#     train_df, test_df = traintest_df.randomSplit([0.8, 0.2], seed=611)

#     feature_cols = [c for c in train_df.columns if c not in ['customer_id', 'snapshot_date', 'label', 'label_def', 'loan_id']]
#     assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
#     scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

#     gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100, maxDepth=6, seed=42)
#     pipeline = Pipeline(stages=[assembler, scaler, gbt])

#     param_grid = (ParamGridBuilder()
#         .addGrid(gbt.maxIter, [50, 100])
#         .addGrid(gbt.maxDepth, [4, 6])
#         .addGrid(gbt.stepSize, [0.05, 0.1])
#         .build())

#     evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
#     f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

#     cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

#     mlflow.set_tracking_uri(uri="http://mlflow:5000")
#     mlflow.set_experiment("LoanEligibilityModel")

#     with mlflow.start_run(run_name=f"{model_name}_{snapshot_date}") as run:
#         cv_model = cv.fit(train_df)
#         best_model = cv_model.bestModel

#         train_predictions = best_model.transform(train_df)
#         test_predictions = best_model.transform(test_df)
#         oot_predictions = best_model.transform(oot_df)

#         train_auc = evaluator.evaluate(train_predictions)
#         test_auc = evaluator.evaluate(test_predictions)
#         oot_auc = evaluator.evaluate(oot_predictions)

#         f1_train = f1_evaluator.evaluate(train_predictions)
#         f1_test = f1_evaluator.evaluate(test_predictions)
#         f1_oot = f1_evaluator.evaluate(oot_predictions)

#         mlflow.log_param("model_name", model_name)
#         mlflow.log_param("snapshot_date", snapshot_date)
#         mlflow.log_param("maxIter", best_model.stages[-1]._java_obj.getMaxIter())
#         mlflow.log_param("maxDepth", best_model.stages[-1]._java_obj.getMaxDepth())
#         mlflow.log_param("stepSize", best_model.stages[-1]._java_obj.getStepSize())

#         mlflow.log_metric("train_auc", train_auc)
#         mlflow.log_metric("test_auc", test_auc)
#         mlflow.log_metric("oot_auc", oot_auc)
#         mlflow.log_metric("train_f1", f1_train)
#         mlflow.log_metric("test_f1", f1_test)
#         mlflow.log_metric("oot_f1", f1_oot)

#         mlflow.spark.log_model(best_model, artifact_path="spark-model")

#         print(f"Train AUC: {train_auc:.4f}")
#         print(f"Test AUC: {test_auc:.4f}")
#         print(f"OOT AUC: {oot_auc:.4f}")
#         print(f"Train F1: {f1_train:.4f}")
#         print(f"Test F1: {f1_test:.4f}")
#         print(f"OOT F1: {f1_oot:.4f}")
#         print(f"Model saved to MLflow under run ID: {run.info.run_id}")

#     return best_model, config


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--snapshotdate", type=str, required=True, help="Training date (YYYY-MM-DD)")
#     parser.add_argument("--modelname", type=str, default="xgb", help="Model name")
#     args = parser.parse_args()

#     spark = SparkSession.builder \
#         .appName("PySpark_GBT_Training") \
#         .master("local[*]") \
#         .config("spark.sql.adaptive.enabled", "true") \
#         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
#         .getOrCreate()

#     spark.sparkContext.setLogLevel("ERROR")
#     train_xgb_model_pyspark(spark, args.snapshotdate, args.modelname)




# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import pyspark
# from tqdm import tqdm

# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import make_scorer, roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

# # from utils import *
# # from config import *
# # from utils.helper import *
# # import numpy as np
# # import config as cg
# # import pickle
# # import json

# # import argparse
# # import os
# # import pickle
# # from datetime import datetime, timedelta
# # from dateutil.relativedelta import relativedelta
# # import optuna
# # import joblib
# # import argparse
# # from xgboost import XGBClassifier

# # def read_gold_table(table, gold_db, type, spark):
# #     """
# #     Helper function to read all partitions of a gold table
# #     """
# #     folder_path = os.path.join(gold_db, table)
# #     files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
# #     df = spark.read.option("header", "true").parquet(*files_list)
# #     return df

# # def read_gold_table(table, gold_db, spark):
# #     """
# #     Helper function to read all partitions of a gold table
# #     """
# #     folder_path = os.path.join('/opt/airflow', gold_db, table)
    
# #     # Read all .parquet files (recursive handles nested partitions)
# #     files_list = glob.glob(os.path.join(folder_path, '**', '*.parquet'), recursive=True)
    
# #     if not files_list:
# #         raise FileNotFoundError(f"No Parquet files found in {folder_path}")
    
# #     df = spark.read.option("header", "true").parquet(*files_list)
# #     return df


# # def train_xgb_model(spark, snapshot_date: str, model_name: str, n_trials: int = 100):
# #     """
# #     Train Logistic Regression model
# #     """
# #     snapshot_date = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
# #     print(f"Training XGBoost model for {model_name} on {snapshot_date}")
# #     print("=" * 60)
    
# #     # Your exact data loading logic
# #     X_spark = read_gold_table('feature_store', '/opt/airflow/datamart/gold', spark)
# #     y_spark = read_gold_table('label_store', '/opt/airflow/datamart/gold', spark)

# #     X_df = X_spark.toPandas().sort_values(by='customer_id')
# #     y_df = y_spark.toPandas().sort_values(by='customer_id')
# #     X_df['snapshot_date'] = pd.to_datetime(X_df['snapshot_date']).dt.date
# #     y_df['snapshot_date'] = pd.to_datetime(y_df['snapshot_date']).dt.date
# #     # snapshot_date = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
# #     # Configuration
# #     model_train_date_str = snapshot_date
# #     # snapshot_date = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
# #     config = {}
# #     config["model_name"] = model_name
# #     config["model_train_date_str"] = model_train_date_str
# #     config["train_test_period_months"] = cg.train_test_period_months
# #     config["oot_period_months"] = cg.oot_period_months
# #     config["model_train_date"] = model_train_date_str
# #     config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
# #     config["oot_start_date"] = config['model_train_date'] - relativedelta(months=cg.oot_period_months)
# #     config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
# #     config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=cg.train_test_period_months)
# #     config["train_test_ratio"] = cg.train_test_ratio
# #     # Data filtering
# #     y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & 
# #                       (y_df['snapshot_date'] <= config['model_train_date'])]
# #     X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

# #     # Create OOT split
# #     y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & 
# #                        (y_model_df['snapshot_date'] <= config['oot_end_date'])]
# #     X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]

# #     # Train-test split
# #     y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
# #     y_traintest = y_model_df[y_model_df['snapshot_date'] < config['oot_start_date']]
# #     X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X_traintest, y_traintest, 
# #         test_size=config['train_test_ratio'], 
# #         random_state=42, 
# #         shuffle=True, 
# #         stratify=y_traintest['label']
# #     )
    
# #     # Print dataset information
# #     print(f'X_train: {X_train.shape[0]} samples')
# #     print(f'X_test: {X_test.shape[0]} samples')
# #     print(f'X_oot: {X_oot.shape[0]} samples')
# #     print(f'y_train: {y_train.shape[0]} samples, bad rate: {y_train["label"].mean():.3f}')
# #     print(f'y_test: {y_test.shape[0]} samples, bad rate: {y_test["label"].mean():.3f}')
# #     print(f'y_oot: {y_oot.shape[0]} samples, bad rate: {y_oot["label"].mean():.3f}')

# #     # Validate bad rates are similar
# #     train_bad_rate = y_train["label"].mean()
# #     test_bad_rate = y_test["label"].mean()
# #     oot_bad_rate = y_oot["label"].mean()
    
# #     if abs(train_bad_rate - test_bad_rate) > 0.05 or abs(train_bad_rate - oot_bad_rate) > 0.05:
# #         return print("WARNING: Bad rates differ significantly between splits. Consider investigating data distribution.")
# #     else:     
# #         # Prepare arrays
# #         X_train_arr = X_train.drop(columns=['customer_id', 'snapshot_date']).values
# #         X_test_arr = X_test.drop(columns=['customer_id', 'snapshot_date']).values
# #         X_oot_arr = X_oot.drop(columns=['customer_id', 'snapshot_date']).values

# #         y_train_arr = y_train['label'].values
# #         y_test_arr = y_test['label'].values
# #         y_oot_arr = y_oot['label'].values

# #         # Scale features
# #         scaler = StandardScaler()
# #         X_train_arr = scaler.fit_transform(X_train_arr)
# #         X_test_arr = scaler.transform(X_test_arr)
# #         X_oot_arr = scaler.transform(X_oot_arr)

# #         print(f"Starting hyperparameter tuning for Logistic Regression...")
        
# #         # XGBoost hyperparameter tuning with Optuna - Minimal version
# #         def objective(trial):
# #             params = {
# #                 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
# #                 'max_depth': trial.suggest_int('max_depth', 3, 8),
# #                 'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
# #                 'random_state': 42,
# #                 'n_jobs': -1
# #             }
            
# #             model = XGBClassifier(**params)
# #             model.fit(X_train_arr, y_train_arr)
# #             y_pred = model.predict_proba(X_test_arr)[:, 1]
# #             return roc_auc_score(y_test_arr, y_pred)

# #         # Run the optimization
# #         study = optuna.create_study(direction='maximize')
# #         study.optimize(objective, n_trials=100)

# #         # Get best parameters
# #         best_params = study.best_params
# #         print(f"Best ROC AUC: {study.best_value:.4f}")
# #         print(f"Best parameters: {best_params}")

# #         # Train final model with best parameters
# #         best_model = XGBClassifier(**best_params)
# #         best_model.fit(X_train_arr, y_train_arr)
        
# #         print(f"Best parameters for XGBoost: {study.best_params}")
# #         print(f"Best cross-validation AUC: {study.best_value:.4f}")

# #         # Evaluate model on all splits
# #         y_pred_proba_train = best_model.predict_proba(X_train_arr)[:, 1]
# #         y_pred_proba_test = best_model.predict_proba(X_test_arr)[:, 1]
# #         y_pred_proba_oot = best_model.predict_proba(X_oot_arr)[:, 1]

# #         # Calculate AUC scores
# #         train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)
# #         test_auc = roc_auc_score(y_test_arr, y_pred_proba_test)
# #         oot_auc = roc_auc_score(y_oot_arr, y_pred_proba_oot)

# #         # Calculate F-beta scores across thresholds
# #         thresholds = np.arange(0.01, 1.0, 0.01)
# #         beta = 1.5
        
# #         f1_scores_train = [fbeta_score(y_train_arr, (y_pred_proba_train > t).astype(int), beta=beta, zero_division=0) 
# #                            for t in thresholds]
# #         f1_scores_test = [fbeta_score(y_test_arr, (y_pred_proba_test > t).astype(int), beta=beta, zero_division=0) 
# #                           for t in thresholds]
# #         f1_scores_oot = [fbeta_score(y_oot_arr, (y_pred_proba_oot > t).astype(int), beta=beta, zero_division=0) 
# #                          for t in thresholds]
        
# #         best_threshold = thresholds[np.argmax(f1_scores_train)]
        
# #         # Print results
# #         print(f"\n XGBoost Model Results:")
# #         print(f"Train AUC: {train_auc:.4f}")
# #         print(f"Test AUC: {test_auc:.4f}")
# #         print(f"OOT AUC: {oot_auc:.4f}")
# #         print(f"Best Threshold: {best_threshold:.3f}")
# #         print(f"Max F-beta (Train): {max(f1_scores_train):.4f}")
# #         print(f"Max F-beta (Test): {max(f1_scores_test):.4f}")
# #         print(f"Max F-beta (OOT): {max(f1_scores_oot):.4f}")
        
# #         # Save model
# #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# #         model_path = f"/opt/airflow/model_bank/{model_name}_{snapshot_date}_{timestamp}.pkl"
        
# #         model_package = {
# #             'model': best_model,
# #             'scaler': scaler,
# #             'best_params': study.best_params,
# #             'performance': {
# #                 'train_auc': train_auc,
# #                 'test_auc': test_auc,
# #                 'oot_auc': oot_auc,
# #                 'best_threshold': best_threshold,
# #                 'max_fbeta_train': max(f1_scores_train),
# #                 'max_fbeta_test': max(f1_scores_test),
# #                 'max_fbeta_oot': max(f1_scores_oot)
# #             },
# #             'config': config,
# #             'feature_columns': X_train.drop(columns=['customer_id', 'snapshot_date']).columns.tolist()
# #         }
        
# #         # Need to create directory first if not exists
# #         # Save model
# #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# #         model_dir = "/opt/airflow/model_bank"  # Directory path
# #         model_path = f"{model_dir}/xgb_{model_train_date_str}_{timestamp}.pkl"  # File path
# #         if not os.path.exists(model_dir):
# #             os.makedirs(model_dir)
# #         joblib.dump(model_package, model_path)
# #         print(f"\nnXGBoost model saved to: {model_path}")
        
# #         return best_model, model_path, model_package

# # # Example usage
# # if __name__ == "__main__":

# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")    
# #     args = parser.parse_args()

# #     # Initialize Spark
# #     # spark = set_spark()
# #     spark = pyspark.sql.SparkSession.builder \
# #     .appName("dev") \
# #     .master("local[*]") \
# #     .getOrCreate()
    
# #     spark.sparkContext.setLogLevel("ERROR")
    
# #     # Train Logistic Regression model
# #     model, model_path, package = train_xgb_model(
# #         spark=spark,
# #         snapshot_date=args.snapshotdate,
# #         model_name="xgb",
# #         n_trials=50
# #     )
    
# #     print(f"Model saved to: {model_path}")
# #     print(f"OOT AUC: {package['performance']['oot_auc']:.4f}")


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
import mlflow
import mlflow.spark
import config as cg


# def prepare_data_splits(X_spark, y_spark, config):
#     """
#     Prepare data splits for training
#     """
#     print("🔧 Preparing data splits...")
    
#     # Join features and labels
#     df = X_spark.join(y_spark, on=['customer_id', 'snapshot_date'], how='inner')
#     df = df.withColumn('snapshot_date', col('snapshot_date').cast(DateType()))
    
#     print(f"📊 Joined data: {df.count()} rows")

#     # Filter data within model date range
#     model_df = df.filter(
#         (col('snapshot_date') >= lit(config['train_test_start_date'])) &
#         (col('snapshot_date') <= lit(config['model_train_date']))
#     )
    
#     print(f"📊 Model data after date filtering: {model_df.count()} rows")

#     # Create OOT split
#     oot_df = model_df.filter(
#         (col('snapshot_date') >= lit(config['oot_start_date'])) &
#         (col('snapshot_date') <= lit(config['oot_end_date']))
#     )

#     # ✅ FIXED: Train-test split excludes OOT period
#     traintest_df = model_df.filter(
#         col('snapshot_date') < lit(config['oot_start_date'])
#     )
    
#     print(f"📊 Train/test data: {traintest_df.count()} rows")
#     print(f"📊 OOT data: {oot_df.count()} rows")

#     return traintest_df, oot_df

# def train_xgb_model_pyspark(spark, snapshot_date: str, model_name: str):
#     """
#     Train GBT model using PySpark MLlib (simplified without model comparison)
#     """
#     print(f"🚀 Training GBT model for {model_name} on {snapshot_date}")
#     print("=" * 60)
    
#     # Configuration setup
#     config = {
#         "model_name": model_name,
#         "model_train_date_str": snapshot_date,
#         "train_test_period_months": cg.train_test_period_months,
#         "oot_period_months": cg.oot_period_months,
#         "model_train_date": datetime.strptime(snapshot_date, "%Y-%m-%d").date()
#     }
    
#     config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
#     config["oot_start_date"] = config['model_train_date'] - relativedelta(months=cg.oot_period_months)
#     config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
#     config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=cg.train_test_period_months)
#     config["train_test_ratio"] = cg.train_test_ratio

#     print(f"📅 Date configuration:")
#     print(f"   Train/test period: {config['train_test_start_date']} to {config['train_test_end_date']}")
#     print(f"   OOT period: {config['oot_start_date']} to {config['oot_end_date']}")

#     # Data loading
#     print("📊 Loading data...")
#     X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
#     y_spark = read_gold_table('label_store', 'datamart/gold', spark)
    
#     # Prepare data splits
#     traintest_df, oot_df = prepare_data_splits(X_spark, y_spark, config)
    
#     # Split train and test
#     train_ratio = 1.0 - config["train_test_ratio"]
#     test_ratio = config["train_test_ratio"]
#     train_df, test_df = traintest_df.randomSplit([train_ratio, test_ratio], seed=611)

#     print(f"📊 Final splits:")
#     print(f"   Train: {train_df.count()} rows")
#     print(f"   Test: {test_df.count()} rows")
#     print(f"   OOT: {oot_df.count()} rows")

#     # Feature preparation
#     feature_cols = [c for c in train_df.columns if c not in ['customer_id', 'snapshot_date', 'label', 'label_def', 'loan_id']]
#     print(f"🎯 Using {len(feature_cols)} features")

#     # ML Pipeline setup
#     assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
#     scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)  # Changed withMean=False for stability
#     gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100, maxDepth=6, seed=42)
#     pipeline = Pipeline(stages=[assembler, scaler, gbt])

#     # Hyperparameter tuning
#     param_grid = (ParamGridBuilder()
#         .addGrid(gbt.maxIter, [50, 100])
#         .addGrid(gbt.maxDepth, [4, 6])
#         .addGrid(gbt.stepSize, [0.05, 0.1])
#         .build())

#     evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
#     f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

#     cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2)  # Reduced folds

#     # ✅ SIMPLIFIED: MLflow setup without model comparison
#     print("🎯 Setting up MLflow...")
#     try:
#         mlflow.set_tracking_uri(uri="http://mlflow:5000")
#         mlflow.set_experiment("LoanEligibilityModel")
        
#         with mlflow.start_run(run_name=f"{model_name}_{snapshot_date}") as run:
#             print("🎯 Training model with cross-validation...")
            
#             # Train model
#             cv_model = cv.fit(train_df)
#             best_model = cv_model.bestModel

#             # Make predictions
#             train_predictions = best_model.transform(train_df)
#             test_predictions = best_model.transform(test_df)
#             oot_predictions = best_model.transform(oot_df) if oot_df.count() > 0 else None

#             # Evaluate model
#             train_auc = evaluator.evaluate(train_predictions)
#             test_auc = evaluator.evaluate(test_predictions)
#             oot_auc = evaluator.evaluate(oot_predictions) if oot_predictions else 0.0

#             f1_train = f1_evaluator.evaluate(train_predictions)
#             f1_test = f1_evaluator.evaluate(test_predictions)
#             f1_oot = f1_evaluator.evaluate(oot_predictions) if oot_predictions else 0.0

#             # ✅ SIMPLIFIED: Log parameters and metrics only
#             mlflow.log_param("model_name", model_name)
#             mlflow.log_param("snapshot_date", snapshot_date)
#             mlflow.log_param("maxIter", best_model.stages[-1]._java_obj.getMaxIter())
#             mlflow.log_param("maxDepth", best_model.stages[-1]._java_obj.getMaxDepth())
#             mlflow.log_param("stepSize", best_model.stages[-1]._java_obj.getStepSize())

#             mlflow.log_metric("train_auc", train_auc)
#             mlflow.log_metric("test_auc", test_auc)
#             mlflow.log_metric("oot_auc", oot_auc)
#             mlflow.log_metric("train_f1", f1_train)
#             mlflow.log_metric("test_f1", f1_test)
#             mlflow.log_metric("oot_f1", f1_oot)

#             # ✅ SIMPLIFIED: Just log the model without comparison
#             mlflow.spark.log_model(best_model, artifact_path="spark-model")

#             # Print results
#             print(f"\n🎉 GBT Model Results:")
#             print(f"   Train AUC: {train_auc:.4f}")
#             print(f"   Test AUC: {test_auc:.4f}")
#             print(f"   OOT AUC: {oot_auc:.4f}")
#             print(f"   Train F1: {f1_train:.4f}")
#             print(f"   Test F1: {f1_test:.4f}")
#             print(f"   OOT F1: {f1_oot:.4f}")
#             print(f"   Model logged to MLflow under run ID: {run.info.run_id}")

#     except Exception as e:
#         print(f"⚠️ MLflow error: {e}. Training without MLflow...")
        
#         # Train model without MLflow
#         print("🎯 Training model with cross-validation (no MLflow)...")
#         cv_model = cv.fit(train_df)
#         best_model = cv_model.bestModel

#         # Make predictions and evaluate
#         train_predictions = best_model.transform(train_df)
#         test_predictions = best_model.transform(test_df)
#         oot_predictions = best_model.transform(oot_df) if oot_df.count() > 0 else None

#         train_auc = evaluator.evaluate(train_predictions)
#         test_auc = evaluator.evaluate(test_predictions)
#         oot_auc = evaluator.evaluate(oot_predictions) if oot_predictions else 0.0

#         f1_train = f1_evaluator.evaluate(train_predictions)
#         f1_test = f1_evaluator.evaluate(test_predictions)
#         f1_oot = f1_evaluator.evaluate(oot_predictions) if oot_predictions else 0.0

#         print(f"\n🎉 GBT Model Results (No MLflow):")
#         print(f"   Train AUC: {train_auc:.4f}")
#         print(f"   Test AUC: {test_auc:.4f}")
#         print(f"   OOT AUC: {oot_auc:.4f}")
#         print(f"   Train F1: {f1_train:.4f}")
#         print(f"   Test F1: {f1_test:.4f}")
#         print(f"   OOT F1: {f1_oot:.4f}")

#     return best_model, config

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--snapshotdate", type=str, required=True, help="Training date (YYYY-MM-DD)")
#     parser.add_argument("--modelname", type=str, default="gbt", help="Model name")
#     args = parser.parse_args()

#     # Initialize Spark with better configuration
#     spark = SparkSession.builder \
#         .appName("PySpark_GBT_Training") \
#         .master("local[2]") \
#         .config("spark.sql.adaptive.enabled", "true") \
#         .config("spark.driver.memory", "2g") \
#         .config("spark.driver.maxResultSize", "1g") \
#         .config("spark.sql.shuffle.partitions", "50") \
#         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
#         .getOrCreate()

#     spark.sparkContext.setLogLevel("ERROR")
    
#     print("✅ Starting GBT model training...")
#     model, config = train_xgb_model_pyspark(spark, args.snapshotdate, args.modelname)
#     print("✅ Training completed successfully!")

#     spark.catalog.clearCache()
#     spark.stop()
