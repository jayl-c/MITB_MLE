import os
import argparse
import pandas as pd
import joblib
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://mlflow:5000/")

def load_production_model():
    """
    Load the production model and associated artifacts from MLflow.
    """
    client = MlflowClient("http://mlflow:5000/")
    model_name = "Loan-default-prod"

    # Get the latest production model
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        raise Exception("No production model found.")

    mv = latest_versions[0]
    model_uri = f"models:/{model_name}/Production"
    print(f"Loading production model from URI: {model_uri}")

    # Load the model
    model = mlflow.sklearn.load_model(model_uri)

    # Load associated artifacts (e.g., scaler, feature columns)
    best_threshold_path = os.path.join(mv.source, "best_threshold.txt")
    scaler_path = os.path.join(mv.source, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise Exception(f"Scaler not found at {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)

    # Load feature columns
    feature_columns_path = os.path.join(mv.source, "feature_columns.pkl")
    if not os.path.exists(feature_columns_path):
        raise Exception(f"Feature columns not found at {feature_columns_path}")
    with open(feature_columns_path, "rb") as f:
        feature_columns = joblib.load(f)

    if not os.path.exists(best_threshold_path):
        raise Exception(f"Best threshold not found at {best_threshold_path}")
    with open(best_threshold_path, "r") as f:
        best_threshold = float(f.read().strip())

    return model, scaler, feature_columns, best_threshold


def model_pred(snapshotdate, spark: SparkSession):
    """
    Perform inference using the production model.
    """
    # Load the production model and artifacts
    model, scaler, feature_cols, best_threshold = load_production_model()

    # Define paths
    gold_db = "datamart/gold"
    partition_name = snapshotdate.replace("-", "_") + ".parquet"
    feature_filepath = os.path.join(gold_db, "feature_store", partition_name)

    # Load feature data
    features_store_sdf = spark.read.parquet(feature_filepath)
    features_sdf = features_store_sdf.filter((col("snapshot_date") == snapshotdate))
    print(f"Extracted features: {features_sdf.count()} rows for snapshot date {snapshotdate}")

    features_pdf = features_sdf.toPandas()

    # Ensure all required features are present
    missing_features = set(feature_cols) - set(features_pdf.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")

    X_inference = features_pdf[feature_cols]

    # Apply scaler
    X_inference_scaled = scaler.transform(X_inference)
    print(f"Scaled inference data: {X_inference_scaled.shape[0]} rows")

    # Perform inference
    y_inference_proba = model.predict_proba(X_inference_scaled)[:, 1]
    y_inference_binary = (y_inference_proba > best_threshold).astype(int) 



    # Prepare output
    y_inference_pdf = features_pdf[["customer_id", "snapshot_date"]].copy()
    y_inference_pdf["model_predictions"] = y_inference_proba
    y_inference_pdf["model_predictions_binary"] = y_inference_binary

    # Extract top 10 features based on importance
    if hasattr(model, "feature_importances_"):  # For tree-based models
        feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):  # For linear models
        feature_importances = model.coef_[0]
    else:
        raise ValueError("Model does not support feature importance extraction.")

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importances
    })

    # Sort by importance and get the top 10 features
    top_10_features = feature_importance_df.sort_values(by="importance", ascending=False).head(10)
    print("Top 10 features:")
    print(top_10_features)

        # Plot the top 10 features
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="importance", 
        y="feature", 
        data=top_10_features.sort_values(by="importance", ascending=True), 
        palette="viridis"
    )
    plt.title("Top 10 Features by Importance", fontsize=16)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    # Save the plot
    plot_output_path = f"/datamart/gold/model_predictions/top_10_features_{snapshotdate}.png"
    plt.savefig(plot_output_path)
    print(f"Top 10 features plot saved to: {plot_output_path}")

    # Save predictions
    output_dir = f"/datamart/gold/model_predictions/"
    output_path = os.path.join(output_dir, partition_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save predictions
    output_dir = f"/opt/airflow/datamart/gold/model_predictions/"
    output_path = os.path.join(output_dir, partition_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(output_path)
    print(f"Predictions saved to: {output_path}")

    return y_inference_pdf


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Run model inference job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Create Spark session
    spark = SparkSession.builder.appName("ModelInference").getOrCreate()

    # Call main with correct parameters
    model_pred(args.snapshotdate, spark)

    # Clean up
    spark.stop()
# def load_best():
#     """Load the deployed best model"""
#     best_path = "/opt/airflow/model_bank/production/best_model.pkl"
#     if os.path.exists(best_path):
#         return joblib.load(best_path)
#     else:
#         print("No champion model found")
#         return None

# def model_pred(snapshotdate, spark: SparkSession):
    
#     # Load the champion model
#     champion_package = load_best()
    
#     # Extract components from champion package
#     model = champion_package['model']
#     scaler = champion_package['scaler']
#     feature_cols = champion_package['feature_columns']
#     metrics = champion_package['performance']
#     best_params = champion_package['best_params']
    
#     # Determine model type
#     model_type = type(model).__name__
#     if 'XGBClassifier' in model_type:
#         model_type_name = 'xgb'
#     elif 'LogisticRegression' in model_type:
#         model_type_name = 'logreg'
#     else:
#         model_type_name = model_type.lower()

#     config = {}
#     config["snapshot_date_str"] = snapshotdate
#     config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
#     config["model_name"] = f"{model_type_name}" 
#     config["model_version"] = f"{model_type_name}_{datetime.now().strftime('%Y%m%d')}"
    
#     pprint.pprint(config)
#     print("Production model loaded successfully!")
#     print(f"Using {model_type_name} with OOT AUC: {metrics['oot_auc']:.4f}")
#     print(f"OOT F1 Score: {metrics['max_fbeta_oot']:.4f}")
#     print(f"Best Threshold: {metrics['best_threshold']:.4f}")

#     # --- load feature store ---
#     gold_db = "/opt/airflow/datamart/gold"
#     partition_name = snapshotdate.replace('-','_') + '.parquet'
#     feature_filepath = os.path.join(gold_db, 'feature_store', partition_name)
     
#     # Load parquet file
#     features_store_sdf = spark.read.parquet(feature_filepath)
        
#     # extract feature store for specific date
#     features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date_str"]))
#     print("extracted features_sdf", features_sdf.count(), config["snapshot_date_str"])
    
#     features_pdf = features_sdf.toPandas()
    
#     # Ensure we have all required features in correct order
#     missing_features = set(feature_cols) - set(features_pdf.columns)
#     if missing_features:
#         raise ValueError(f"Missing features in data: {missing_features}")
       
#     X_inference = features_pdf[feature_cols]
    
#     # apply transformer - use scaler from trained model
#     X_inference_scaled = scaler.transform(X_inference)
    
#     print('X_inference', X_inference_scaled.shape[0])

#     # --- model prediction inference ---
#     # Get probability predictions
#     y_inference_proba = model.predict_proba(X_inference_scaled)[:, 1]
    
#     # Get binary predictions using the best threshold from training
#     y_inference_binary = (y_inference_proba > metrics['best_threshold']).astype(int)
    
#     # prepare output with both probability and binary predictions
#     y_inference_pdf = features_pdf[["customer_id","snapshot_date"]].copy()
#     y_inference_pdf["model_name"] = config["model_name"]
#     y_inference_pdf["model_version"] = config["model_version"]
#     y_inference_pdf["model_type"] = model_type_name
#     y_inference_pdf["model_predictions"] = y_inference_proba  # Keep your original column name
#     y_inference_pdf["model_predictions_binary"] = y_inference_binary
#     y_inference_pdf["model_threshold"] = metrics['best_threshold']
#     y_inference_pdf["model_oot_auc"] = metrics['oot_auc']
#     y_inference_pdf["model_oot_f1"] = metrics['max_fbeta_oot']
    
#     # --- save model inference to datamart gold table ---
#     # create gold directory
#     gold_directory = f"/opt/airflow/datamart/gold/model_predictions/{config['model_name']}/"
#     print(gold_directory)
    
#     if not os.path.exists(gold_directory):
#         os.makedirs(gold_directory)
    
#     # save gold table with more detailed naming
#     # partition_name = config["model_name"] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
#     partition_name = config["snapshot_date_str"].replace('-','_') + '.parquet'
#     filepath = gold_directory + partition_name
#     spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
#     print('saved to:', filepath)
      
#     print('\n\n---completed job---\n\n')
    
#     return y_inference_pdf

# if __name__ == "__main__":
#     # Setup argparse to parse command-line arguments
#     parser = argparse.ArgumentParser(description="run job")
#     parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
#     args = parser.parse_args()
    
#     # Create Spark session
#     spark = SparkSession.builder.appName("ModelInference").getOrCreate()
    
#     # Call main with correct parameters
#     model_pred(args.snapshotdate, spark)
    
#     # Clean up
#     spark.stop()