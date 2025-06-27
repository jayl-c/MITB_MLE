from pyspark.sql import SparkSession
import logging

import joblib
import os
import glob
import shutil
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_latest_model(model_type, model_bank_path="./model_bank/"):
    """Get the latest model file for a given model type"""
    pattern = os.path.join(model_bank_path, f"*{model_type}*_*.pkl")
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"No {model_type} models found in {model_bank_path}")
        return None, None
    
    # Sort by modification time to get the latest
    latest_file = max(model_files, key=os.path.getmtime)
    model_package = joblib.load(latest_file)
    
    return model_package, latest_file

def deploy_best_model(primary_metric='oot_f1', deployment_path="./model_bank/production/"):
    """
    Compare XGBoost vs Logistic Regression and deploy the better one
    
    Parameters:
    primary_metric: str - 'oot_f1', 'oot_auc', 'test_f1', etc.
    deployment_path: str - where to deploy the winning model
    """
    
    # Load latest models
    xgb_package, xgb_path = get_latest_model("xgb")
    logreg_package, logreg_path = get_latest_model("logreg")
    
    if xgb_package is None or logreg_package is None:
        print("Could not find both model types")
        return None
    
    # Get performance scores
    xgb_perf = xgb_package['performance']
    logreg_perf = logreg_package['performance']
    
    # Map metric names
    metric_map = {
        'oot_f1': 'max_fbeta_oot',
        'oot_auc': 'oot_auc',
        'test_f1': 'max_fbeta_test',
        'test_auc': 'test_auc'
    }
    
    metric_key = metric_map.get(primary_metric, 'max_fbeta_oot')
    
    xgb_score = xgb_perf[metric_key]
    logreg_score = logreg_perf[metric_key]
    
    # Determine winner
    if xgb_score > logreg_score:
        winner_package, winner_path, winner_name = xgb_package, xgb_path, "XGBoost"
        winner_score, loser_score = xgb_score, logreg_score
    else:
        winner_package, winner_path, winner_name = logreg_package, logreg_path, "Logistic Regression"
        winner_score, loser_score = logreg_score, xgb_score
    
    # Print comparison
    print(f"Model Comparison ({primary_metric.upper()}):")
    print(f"XGBoost: {xgb_score:.4f}")
    print(f"Logistic Regression: {logreg_score:.4f}")
    print(f"Winner: {winner_name} ({winner_score:.4f})")
    print(f"Margin: {abs(winner_score - loser_score):.4f}")
    
    # Deploy winner
    os.makedirs(deployment_path, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    production_path = os.path.join(deployment_path, f"best_model_{timestamp}.pkl")
    latest_path = os.path.join(deployment_path, "best_model.pkl")
    
    # Copy model files
    shutil.copy2(winner_path, production_path)
    if os.path.exists(latest_path):
        os.remove(latest_path)
    shutil.copy2(winner_path, latest_path)
    
    # Save deployment log
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'winner': winner_name,
        'metric': primary_metric,
        'winner_score': winner_score,
        'xgb_score': xgb_score,
        'logreg_score': logreg_score,
        'production_path': production_path
    }
    
    log_path = os.path.join(deployment_path, f"deployment_log_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nDeployed: {production_path}")
    print(f"Log saved: {log_path}")
    
    return winner_package, production_path



# Usage
if __name__ == "__main__":
    # Deploy best model based on OOT F1 (better for risk prediction)
    champion, path = deploy_best_model(primary_metric='oot_f1')
    
    # # Load for inference
    # model = load_best()
    # if model:
    #     print(f"Production Model ready: {type(model['model']).__name__}")
    #     print(f"Threshold: {model['performance']['best_threshold']:.3f}")