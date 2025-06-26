import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyspark
from tqdm import tqdm


from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

from utils import *
from config import *
from utils.helper import *
import numpy as np
import config as cg
import pickle
import json

import argparse
import os
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import optuna
import joblib
import argparse

def read_gold_table(table, gold_db, type, spark):
    """
    Helper function to read all partitions of a gold table
    """
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df


def train_logistic_regression_model(spark, snapshot_date: str, model_name: str, n_trials: int = 100):
    """
    Train Logistic Regression model using your exact data splitting methodology.
    """
    
    print(f"Training Logistic Regression model for {model_name} on {snapshot_date}")
    print("=" * 60)
    
    # Your exact data loading logic
    X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
    y_spark = read_gold_table('label_store', 'datamart/gold', spark)

    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')

    # Configuration
    model_train_date_str = snapshot_date
    config = {}
    config["model_name"] = model_name
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = cg.train_test_period_months
    config["oot_period_months"] = cg.oot_period_months
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
    config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
    config["oot_start_date"] = config['model_train_date'] - relativedelta(months=cg.oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=cg.train_test_period_months)
    config["train_test_ratio"] = cg.train_test_ratio

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
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

    X_train, X_test, y_train, y_test = train_test_split(
        X_traintest, y_traintest, 
        test_size=config['train_test_ratio'], 
        random_state=611, 
        shuffle=True, 
        stratify=y_traintest['label']
    )
    
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
    else:     
        # Prepare arrays
        X_train_arr = X_train.drop(columns=['customer_id', 'snapshot_date']).values
        X_test_arr = X_test.drop(columns=['customer_id', 'snapshot_date']).values
        X_oot_arr = X_oot.drop(columns=['customer_id', 'snapshot_date']).values

        y_train_arr = y_train['label'].values
        y_test_arr = y_test['label'].values
        y_oot_arr = y_oot['label'].values

        # Scale features
        scaler = StandardScaler()
        X_train_arr = scaler.fit_transform(X_train_arr)
        X_test_arr = scaler.transform(X_test_arr)
        X_oot_arr = scaler.transform(X_oot_arr)

        print(f"Starting hyperparameter tuning for Logistic Regression...")
        
        # Logistic Regression hyperparameter tuning with Optuna
        def objective(trial):
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
            
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': penalty,
                'max_iter': trial.suggest_int('max_iter', 100, 2000),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Set solver based on penalty
            if penalty == 'l1':
                params['solver'] = 'liblinear'
            elif penalty == 'l2':
                params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'sag', 'saga'])
            else:  # elasticnet
                params['solver'] = 'saga'
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            
            model = LogisticRegression(**params)
            model.fit(X_train_arr, y_train_arr)
            y_pred = model.predict_proba(X_test_arr)[:, 1]
            return roc_auc_score(y_test_arr, y_pred)
        
        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model
        best_model = LogisticRegression(**study.best_params)
        best_model.fit(X_train_arr, y_train_arr)
        
        print(f"Best parameters for Logistic Regression: {study.best_params}")
        print(f"Best cross-validation AUC: {study.best_score:.4f}")

        # Evaluate model on all splits
        y_pred_proba_train = best_model.predict_proba(X_train_arr)[:, 1]
        y_pred_proba_test = best_model.predict_proba(X_test_arr)[:, 1]
        y_pred_proba_oot = best_model.predict_proba(X_oot_arr)[:, 1]

        # Calculate AUC scores
        train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)
        test_auc = roc_auc_score(y_test_arr, y_pred_proba_test)
        oot_auc = roc_auc_score(y_oot_arr, y_pred_proba_oot)

        # Calculate F-beta scores across thresholds
        thresholds = np.arange(0.01, 1.0, 0.01)
        beta = 1.5
        
        f1_scores_train = [fbeta_score(y_train_arr, (y_pred_proba_train > t).astype(int), beta=beta, zero_division=0) 
                           for t in thresholds]
        f1_scores_test = [fbeta_score(y_test_arr, (y_pred_proba_test > t).astype(int), beta=beta, zero_division=0) 
                          for t in thresholds]
        f1_scores_oot = [fbeta_score(y_oot_arr, (y_pred_proba_oot > t).astype(int), beta=beta, zero_division=0) 
                         for t in thresholds]
        
        best_threshold = thresholds[np.argmax(f1_scores_train)]
        
        # Print results
        print(f"\nLogistic Regression Model Results:")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"OOT AUC: {oot_auc:.4f}")
        print(f"Best Threshold: {best_threshold:.3f}")
        print(f"Max F-beta (Train): {max(f1_scores_train):.4f}")
        print(f"Max F-beta (Test): {max(f1_scores_test):.4f}")
        print(f"Max F-beta (OOT): {max(f1_scores_oot):.4f}")
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"model_bank/{model_name}_{snapshot_date}_{timestamp}.pkl"
        
        model_package = {
            'model': best_model,
            'scaler': scaler,
            'best_params': study.best_params,
            'performance': {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'oot_auc': oot_auc,
                'best_threshold': best_threshold,
                'max_fbeta_train': max(f1_scores_train),
                'max_fbeta_test': max(f1_scores_test),
                'max_fbeta_oot': max(f1_scores_oot)
            },
            'config': config,
            'feature_columns': X_train.drop(columns=['customer_id', 'snapshot_date']).columns.tolist()
        }
        
        joblib.dump(model_package, model_path)
        print(f"\nLogistic Regression model saved to: {model_path}")
        
        return best_model, model_path, model_package

# Example usage
if __name__ == "__main__":

    # Initialize Spark
    spark = set_spark()
    
    # Train Logistic Regression model
    model, model_path, package = train_logistic_regression_model(
        spark=spark,
        snapshot_date=args.snapshotdate,
        model_name="logreg_train",
        n_trials=100
    )
    
    print(f"\nLogistic Regression training completed!")
    print(f"Model saved to: {model_path}")
    print(f"OOT AUC: {package['performance']['oot_auc']:.4f}")