import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import joblib
import numpy as np


def set_spark():
    spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    return spark

def prepare_spark_features(df, feature_cols, feature_col_name="features"):
    """
    Convert feature columns to a single vector column for Spark ML
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=feature_col_name)
    return assembler.transform(df)

def generate_first_of_month_dates(start_date_str, end_date_str):
    """
    Generate list of dates to process
    """
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

def generate_first_of_month_dates_past_6_months(snapshot_date_str):
    """
    Generate a list of the first day of the month for the past 6 months.

    Parameters:
    -----------
    snapshot_date_str : str
        The snapshot date in the format 'YYYY-MM-DD'.

    Returns:
    --------
    list : A list of the first day of the month for the past 12 months.
    """
    # Convert the snapshot date to a datetime object
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Generate the first day of the month for the past 12 months
    for i in range(6):
        first_of_month = (snapshot_date - relativedelta(months=i)).replace(day=1)
        first_of_month_dates.append(first_of_month.strftime("%Y-%m-%d"))

    return first_of_month_dates

import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession

def get_gold_features_and_labels(spark, start_date, end_date, data_root_path="/opt/airflow/data"):
    """
    Load gold features and labels from local file system in Airflow.
    
    Args:
        spark: SparkSession object
        start_date: Start date (datetime object)
        end_date: End date (datetime object)
        data_root_path: Root path to data directory
        
    Returns:
        tuple: (feature_df, label_df) - Spark DataFrames or (None, None) if no files found
    """
    
    print(f"Loading gold data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Define paths to feature and label stores
    feature_store_path = os.path.join(data_root_path, "datamart", "gold", "feature_store")
    label_store_path = os.path.join(data_root_path, "datamart", "gold", "label_store")
    
    stores = {
        "features": feature_store_path,
        "labels": label_store_path
    }
    
    # Initialize file lists
    found_files = {"features": [], "labels": []}
    
    try:
        # Generate month list (first day of each month)
        month_list = get_month_list(start_date, end_date)
        print(f"Looking for files in months: {month_list}")
        
        # Find files for each store
        for store_name, store_path in stores.items():
            print(f"\nProcessing {store_name} from: {store_path}")
            
            if not os.path.exists(store_path):
                print(f"Warning: Directory does not exist: {store_path}")
                continue
            
            # Look for files for each month
            for month_date in month_list:
                file_pattern = f"{month_date}.parquet"
                file_path = os.path.join(store_path, file_pattern)
                
                # Check for exact file match
                if os.path.exists(file_path):
                    print(f"  Found: {file_path}")
                    found_files[store_name].append(file_path)
                else:
                    # Try pattern matching in case of nested structure
                    pattern_path = os.path.join(store_path, f"{month_date}*", "*.parquet")
                    matching_files = glob.glob(pattern_path)
                    
                    if matching_files:
                        print(f"  Found nested files: {len(matching_files)} files")
                        found_files[store_name].extend(matching_files)
                    else:
                        print(f"  Not found: {file_pattern}")
        
        # Load data into Spark DataFrames
        feature_df = None
        label_df = None
        
        if found_files["features"]:
            print(f"\nLoading {len(found_files['features'])} feature files...")
            try:
                feature_df = spark.read.parquet(*found_files["features"])
                print(f"Feature DataFrame: {feature_df.count()} rows, {len(feature_df.columns)} columns")
            except Exception as e:
                print(f"Error loading feature files: {str(e)}")
                feature_df = None
        
        if found_files["labels"]:
            print(f"Loading {len(found_files['labels'])} label files...")
            try:
                label_df = spark.read.parquet(*found_files["labels"])
                print(f"Label DataFrame: {label_df.count()} rows, {len(label_df.columns)} columns")
            except Exception as e:
                print(f"Error loading label files: {str(e)}")
                label_df = None
        
        # Return results
        if feature_df is not None and label_df is not None:
            print("Successfully loaded both features and labels")
            return feature_df, label_df
        elif feature_df is not None:
            print("Warning: Only features loaded, no labels found")
            return feature_df, None
        elif label_df is not None:
            print("Warning: Only labels loaded, no features found")
            return None, label_df
        else:
            print("No data files found for the specified date range")
            return None, None
            
    except Exception as e:
        print(f"Error loading gold data: {str(e)}")
        raise e

def get_month_list(start_date, end_date):
    """
    Generate list of month strings (YYYY-MM-DD format) between start and end dates.
    Returns first day of each month.
    
    Args:
        start_date: Start date (datetime object)
        end_date: End date (datetime object)
        
    Returns:
        list: List of date strings in YYYY-MM-DD format
    """
    month_list = []
    
    # Start from first day of start_date month
    current_date = datetime(start_date.year, start_date.month, 1)
    end_month = datetime(end_date.year, end_date.month, 1)
    
    while current_date <= end_month:
        month_list.append(current_date.strftime('%Y-%m-%d'))
        # Move to first day of next month
        current_date = current_date + relativedelta(months=1)
    
    return month_list

def read_gold_table(table_name, path_prefix, spark, data_root_path="/opt/airflow/datamart"):
    """
    Read a single gold table from local file system.
    
    Args:
        table_name: Name of the table (e.g., 'feature_store', 'label_store')
        path_prefix: Path prefix (e.g., 'datamart/gold')
        spark: SparkSession object
        data_root_path: Root path to data directory
        
    Returns:
        DataFrame: Spark DataFrame or None if not found
    """
    
    # Build full path
    table_path = os.path.join(data_root_path, path_prefix, table_name)
    
    print(f"Reading table: {table_name} from {table_path}")
    
    try:
        if os.path.exists(table_path):
            # Check if it's a single parquet file or directory with parquet files
            if os.path.isfile(table_path) and table_path.endswith('.parquet'):
                df = spark.read.parquet(table_path)
            elif os.path.isdir(table_path):
                # Look for parquet files in directory
                parquet_files = glob.glob(os.path.join(table_path, "*.parquet"))
                if parquet_files:
                    df = spark.read.parquet(*parquet_files)
                else:
                    # Try nested directories
                    nested_parquet = glob.glob(os.path.join(table_path, "**", "*.parquet"), recursive=True)
                    if nested_parquet:
                        df = spark.read.parquet(*nested_parquet)
                    else:
                        print(f"No parquet files found in {table_path}")
                        return None
            else:
                print(f"Path exists but is not a valid parquet source: {table_path}")
                return None
            
            print(f"Successfully loaded {table_name}: {df.count()} rows, {len(df.columns)} columns")
            return df
            
        else:
            print(f"Table path does not exist: {table_path}")
            return None
            
    except Exception as e:
        print(f"Error reading table {table_name}: {str(e)}")
        raise e

def get_gold_file_if_exist(start_date, end_date, spark, data_root_path="/opt/airflow/data"):
    """
    Wrapper function that matches your original function signature.
    Airflow-compatible version without Google Drive dependency.
    
    Args:
        start_date: Start date (datetime object)
        end_date: End date (datetime object)
        spark: SparkSession object
        data_root_path: Root path to data directory
        
    Returns:
        tuple: (feature_df, label_df) - Spark DataFrames or (None, None) if no files found
    """
    
    return get_gold_features_and_labels(
        spark=spark,
        start_date=start_date,
        end_date=end_date,
        data_root_path=data_root_path
    )


def generate_config(snapshot_date: str, model_name: str, cg: object) -> dict:
    config = {}
    config["model_name"] = model_name
    config["model_train_date_str"] = snapshot_date
    config["train_test_period_months"] = cg.train_test_period_months
    config["oot_period_months"] = cg.oot_period_months
    config["train_test_ratio"] = cg.train_test_ratio
    config["model_train_date"] = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
    config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
    config["oot_start_date"] = config['model_train_date'] - relativedelta(months=cg.oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=cg.train_test_period_months)
    return config

def get_data_splits(X_df, y_df, config):
    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & 
                      (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & 
                       (y_model_df['snapshot_date'] <= config['oot_end_date'])]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]

    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

    return X_traintest, y_traintest, X_oot, y_oot

def validate_bad_rate(y_train, y_test, y_oot) -> bool:
    rates = [y["label"].mean() for y in [y_train, y_test, y_oot]]
    if any(abs(rates[i] - rates[j]) > 0.05 for i in range(3) for j in range(i+1, 3)):
        print("WARNING: Bad rates differ significantly between splits.")
        return False
    return True
