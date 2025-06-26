import argparse
import mlflow
import mlflow.spark
import optuna
import uuid
import shutil
import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from mlflow.models.signature import infer_signature
from utils.helper import *
import numpy as np
import config


def run_optuna_logreg(train_df, test_df, feature_col, snapshot_date):
    """
    Run Optuna hyperparameter optimization for Logistic Regression
    Only uses train/test data - OOT is reserved for final evaluation
    """
    def objective(trial):
        params = {
            "regParam": trial.suggest_float("regParam", 0.001, 1.0, log=True),
            "elasticNetParam": trial.suggest_float("elasticNetParam", 0.0, 1.0),
            "maxIter": trial.suggest_int("maxIter", 50, 500),
            "threshold": trial.suggest_float("threshold", 0.3, 0.7),
            "standardization": trial.suggest_categorical("standardization", [True, False])
        }
        
        # Only train and evaluate on train/test - NO OOT data used in tuning
        classifier = LogisticRegression(
            labelCol="label", 
            featuresCol=feature_col,
            **params
        )
        
        trained_model = classifier.fit(train_df)
        test_predictions = trained_model.transform(test_df)
        
        # Evaluate only on test set for hyperparameter selection
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


def register_model_mlflow(run_name, params, model_class, train_df, test_df, oot_df, model_name, feature_col, feature_names=None):
    """
    Register and log a Spark ML Logistic Regression classifier with MLflow including train/test/OOT metrics.
    """
    with mlflow.start_run(run_name=run_name):
        
        # Initialize Logistic Regression classifier
        classifier = model_class(
            labelCol="label", 
            featuresCol=feature_col,
            **params
        )
        
        # Train the model
        trained_model = classifier.fit(train_df)

        # Make predictions on test and OOT sets
        test_predictions = trained_model.transform(test_df).persist()
        oot_predictions = trained_model.transform(oot_df).persist()
        
        # Force evaluation to persist data
        _ = test_predictions.count()
        _ = oot_predictions.count()

        # Binary evaluators
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

        # Calculate metrics
        test_metrics = {
            "test_auc": eval_auc.evaluate(test_predictions),
            "test_f1": eval_f1.evaluate(test_predictions)
        }

        oot_metrics = {
            "oot_auc": eval_auc.evaluate(oot_predictions),
            "oot_f1": eval_f1.evaluate(oot_predictions)
        }

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics({**test_metrics, **oot_metrics})

        # Log model coefficients and intercept
        coefficients = trained_model.coefficients.toArray()
        intercept = trained_model.intercept
        
        mlflow.log_param("intercept", intercept)
        mlflow.log_param("num_features", len(coefficients))
        
        # Get feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        # Create feature importance dictionary with names and coefficients
        feature_importance = {}
        feature_abs_importance = {}
        
        for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
            feature_importance[f"coef_{name}"] = float(coef)
            feature_abs_importance[name] = abs(float(coef))
        
        # Log all coefficients
        mlflow.log_params(feature_importance)
        
        # Log top 10 most important features by absolute coefficient value
        top_features = sorted(feature_abs_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_features_dict = {f"top_{i+1}_feature": f"{name} ({abs_coef:.4f})" 
                           for i, (name, abs_coef) in enumerate(top_features)}
        mlflow.log_params(top_features_dict)
        
        # Create and log feature importance plot data
        import pandas as pd
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': [abs(c) for c in coefficients]
        }).sort_values('abs_coefficient', ascending=False)
        
        # Save feature importance as CSV
        importance_path = f"/tmp/feature_importance_{uuid.uuid4()}.csv"
        feature_importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")

        # Save test predictions as CSV
        log_dir = f"/tmp/spark_predictions_log_{uuid.uuid4()}"
        shutil.rmtree(log_dir, ignore_errors=True)

        test_predictions.select("label", "prediction", "probability") \
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
            test_predictions.select("prediction", "probability")
        )
        mlflow.spark.log_model(
            spark_model=trained_model,
            artifact_path="loan-default-predictor-logreg",
            signature=signature
        )

        # Set tags
        mlflow.set_tags({
            "Training Info": f"Logistic Regression model for loan-default-predictor",
            "model_type": "LogisticRegression",
            "model_name": model_name,
            "regularization": f"ElasticNet(alpha={params.get('regParam', 0)}, l1_ratio={params.get('elasticNetParam', 0)})"
        })

        return trained_model, test_metrics["test_f1"]


def prepare_spark_features(df, feature_cols, feature_col_name="features"):
    """
    Convert feature columns to a single vector column for Spark ML
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=feature_col_name)
    return assembler.transform(df)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Logistic Regression for credit risk prediction")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, default="loan-default-model")
    parser.add_argument("--ntrials", type=int, default=50)
    args = parser.parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "loan-default-predictor-logreg"
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()

    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Create Spark session
    spark = set_spark()  # assume this exists

    # Load data
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
    
    print(f"Training Logistic Regression with {len(feature_cols)} features")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}, OOT size: {len(oot_df)}")

    # Convert back to Spark DataFrames and prepare features
    train_spark = spark.createDataFrame(train_df)
    test_spark = spark.createDataFrame(test_df)
    oot_spark = spark.createDataFrame(oot_df)

    # Prepare feature vectors
    train_spark = prepare_spark_features(train_spark, feature_cols, "features")
    test_spark = prepare_spark_features(test_spark, feature_cols, "features")
    oot_spark = prepare_spark_features(oot_spark, feature_cols, "features")

    # Run hyperparameter optimization (only using train/test)
    print("Starting hyperparameter optimization...")
    study = run_optuna_logreg(train_spark, test_spark, "features", args.snapshotdate)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Train final model with best parameters (only on train data) and evaluate on test + OOT
    print("\nTraining final model with best parameters...")
    
    # Train only on training data
    final_classifier = LogisticRegression(
        labelCol="label", 
        featuresCol="features",
        **study.best_params
    )
    final_model = final_classifier.fit(train_spark)
    
    # Evaluate on test and OOT (no retraining)
    test_predictions = final_model.transform(test_spark).persist()
    oot_predictions = final_model.transform(oot_spark).persist()
    
    # Calculate final metrics
    eval_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    
    final_metrics = {
        "final_test_auc": eval_auc.evaluate(test_predictions),
        "final_test_f1": eval_f1.evaluate(test_predictions),
        "final_oot_auc": eval_auc.evaluate(oot_predictions),
        "final_oot_f1": eval_f1.evaluate(oot_predictions)
    }
    
    # Log final model and metrics
    with mlflow.start_run(run_name=f"logreg_{args.snapshotdate}_final"):
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(final_metrics)
        
        # Log feature importance
        coefficients = final_model.coefficients.toArray()
        intercept = final_model.intercept
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': coefficients,
            'abs_coefficient': [abs(c) for c in coefficients]
        }).sort_values('abs_coefficient', ascending=False)
        
        # Log top 10 features
        top_features = feature_importance_df.head(10)
        for i, row in top_features.iterrows():
            mlflow.log_param(f"top_{len(top_features)-i}_feature", f"{row['feature']} ({row['abs_coefficient']:.4f})")
        
        # Save feature importance
        importance_path = f"/tmp/feature_importance_{uuid.uuid4()}.csv"
        feature_importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")
        
        # Log model
        signature = infer_signature(test_spark.select("features"), test_predictions.select("prediction", "probability"))
        mlflow.spark.log_model(
            spark_model=final_model,
            artifact_path="loan-default-predictor-logreg",
            signature=signature
        )
    
    print(f"Final Test AUC: {final_metrics['final_test_auc']:.4f} | Final OOT AUC: {final_metrics['final_oot_auc']:.4f}")
    print(f"Final Test F1: {final_metrics['final_test_f1']:.4f} | Final OOT F1: {final_metrics['final_oot_f1']:.4f}")
