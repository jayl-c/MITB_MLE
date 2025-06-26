import optuna
import mlflow
import shutil
import uuid
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from utils.helper import *
import config
import argparse

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from mlflow.models.signature import infer_signature

def run_optuna_gbt(train_df, test_df, feature_col, snapshot_date):
    """Only tune on train/test data"""
    def objective(trial):
        params = {
            "stepSize": trial.suggest_float("stepSize", 0.01, 0.3),
            "maxDepth": trial.suggest_int("maxDepth", 3, 10),
            "maxBins": trial.suggest_int("maxBins", 32, 500),
            "lossType": 'logistic'
        }
        
        classifier = GBTClassifier(**params, labelCol="label", featuresCol=feature_col)
        trained_model = classifier.fit(train_df)
        test_predictions = trained_model.transform(test_df)
        
        eval_f1 = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="f1"
        )
        f1_score = eval_f1.evaluate(test_predictions)
        return f1_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study

def register_model_mlflow(run_name, params, model_class, train_df, test_df, oot_df, model_name, feature_col):
    """
    Register and log a Spark ML binary classifier with MLflow including train/test/OOT metrics.
    """
    with mlflow.start_run(run_name=run_name):

        classifier = model_class(**params, labelCol="label", featuresCol=feature_col)
        trained_model = classifier.fit(train_df)

        test_predictions = trained_model.transform(test_df).persist()
        oot_predictions = trained_model.transform(oot_df).persist()
        _ = test_predictions.count()
        _ = oot_predictions.count()

        # Binary evaluators - fixed label column name
        eval_auc = BinaryClassificationEvaluator(
            labelCol="label", 
            rawPredictionCol="rawPrediction", 
            metricName="areaUnderROC"
        )
        eval_f1 = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="f1"
        )

        test_metrics = {
            "test_auc": eval_auc.evaluate(test_predictions),
            "test_f1": eval_f1.evaluate(test_predictions)
        }

        oot_metrics = {
            "oot_auc": eval_auc.evaluate(oot_predictions),
            "oot_f1": eval_f1.evaluate(oot_predictions)
        }

        mlflow.log_params(params)
        mlflow.log_metrics({**test_metrics, **oot_metrics})

        # Save test predictions as CSV - fixed column names
        log_dir = f"/tmp/spark_predictions_log_{uuid.uuid4()}"
        shutil.rmtree(log_dir, ignore_errors=True)

        test_predictions.select("label", "prediction") \
            .withColumnRenamed("label", "ground_truth") \
            .withColumnRenamed("prediction", "predictions") \
            .coalesce(1) \
            .write.option("header", True).mode("overwrite").csv(log_dir)

        csv_file = next((f for f in os.listdir(log_dir) if f.endswith(".csv")), None)
        if csv_file:
            mlflow.log_artifact(os.path.join(log_dir, csv_file), artifact_path="predictions")

        # Log datasets
        for name, df in [("train", train_df), ("test", test_df), ("oot", oot_df)]:
            path = f"/tmp/{name}_spark_{uuid.uuid4()}.parquet"
            df.select(feature_col, "label").write.mode("overwrite").parquet(path)
            mlflow.log_artifact(path, artifact_path=f"{name}_data")

        # Log model
        signature = infer_signature(
            test_df.select(feature_col), 
            test_predictions.select("prediction")
        )
        mlflow.spark.log_model(
            spark_model=trained_model,
            artifact_path="loan-default-predictor",
            signature=signature
        )

        mlflow.set_tags({
            "Training Info": f"GBT model for loan-default-predictor",
            "model_type": "GBTClassifier",
            "model_name": model_name
        })

        return trained_model, test_metrics["test_f1"]

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train GBT for credit risk prediction")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, default="loan-default-model")
    parser.add_argument("--ntrials", type=int, default=50)
    args = parser.parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "loan-default-predictor-gbt"
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()

    # Create Spark session
    spark = set_spark() 

    X_spark = read_gold_table('feature_store', 'gold/', spark)
    y_spark = read_gold_table('label_store', 'gold/', spark)

    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')

    # Configuration setup
    model_dict = {}
    model_dict["model_train_date_str"] = args.snapshotdate
    model_dict["train_test_period_months"] = config.train_test_period_months
    model_dict["oot_period_months"] = config.oot_period_months
    model_dict["model_train_date"] = datetime.strptime(args.snapshotdate, "%Y-%m-%d")
    model_dict["oot_end_date"] = model_dict['model_train_date'] - timedelta(days=1)
    model_dict["oot_start_date"] = model_dict['model_train_date'] - relativedelta(months=config.oot_period_months)
    model_dict["train_test_end_date"] = model_dict["oot_start_date"] - timedelta(days=1)
    model_dict["train_test_start_date"] = model_dict["oot_start_date"] - relativedelta(months=config.train_test_period_months)
    model_dict["train_test_ratio"] = config.train_test_ratio

    # Create temporal splits
    y_model_df = y_df[
        (y_df['snapshot_date'] >= model_dict['train_test_start_date']) & 
        (y_df['snapshot_date'] <= model_dict['model_train_date'])
    ]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

    # Create OOT split
    y_oot = y_model_df[
        (y_model_df['snapshot_date'] >= model_dict['oot_start_date']) & 
        (y_model_df['snapshot_date'] <= model_dict['oot_end_date'])
    ]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]

    # Everything else goes into train-test
    y_traintest = y_model_df[y_model_df['snapshot_date'] <= model_dict['train_test_end_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

    # Create combined DataFrames
    traintest_df = X_traintest.merge(y_traintest, on='customer_id', how='inner')
    oot_df = X_oot.merge(y_oot, on='customer_id', how='inner')

    # Split train/test
    train_df, test_df = train_test_split(
        traintest_df, 
        test_size=model_dict['train_test_ratio'], 
        random_state=611, 
        shuffle=True, 
        stratify=traintest_df['label']
    )

    # Identify feature columns
    feature_cols = [col for col in train_df.columns if col not in ['customer_id', 'snapshot_date', 'label']]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}, OOT size: {len(oot_df)}")

    # Convert back to Spark DataFrames and prepare features
    train_spark = spark.createDataFrame(train_df)
    test_spark = spark.createDataFrame(test_df)
    oot_spark = spark.createDataFrame(oot_df)

    # Prepare feature vectors
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_spark = assembler.transform(train_spark)
    test_spark = assembler.transform(test_spark)
    oot_spark = assembler.transform(oot_spark)

    # Run hyperparameter optimization (only using train/test)
    print("Starting hyperparameter optimization...")
    study = run_optuna_gbt(train_spark, test_spark, "features", args.snapshotdate)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Train final model and evaluate on all sets including OOT
    print("\nTraining final model with best parameters...")
    final_model, final_f1 = register_model_mlflow(
        f"gbt_{args.snapshotdate}_final",
        study.best_params,
        GBTClassifier,
        train_spark,
        test_spark,
        oot_spark,
        "GBTClassifier_Final",
        "features"
    )
    
    spark.stop()