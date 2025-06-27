import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import tqdm
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import read_silver_table, build_label_store, process_gold_label

from utils.helper import *

# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def bronze_label(snapshotdate, spark:SparkSession):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = set_spark()

    # load arguments
    date_str = snapshotdate
    
    # create bronze datalake
    bronze_lms_directory = "/opt/airflow/datamart/bronze"
    bronze_lms_path = "/opt/airflow/data/lms_loan_daily.csv"
    if not os.path.exists(bronze_lms_directory):
        os.makedirs(bronze_lms_directory)

    # run data processing
    process_bronze_table('lms', bronze_lms_path, bronze_lms_directory, date_str, spark)
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

def silver_label(snapshotdate, spark:SparkSession):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = set_spark()

    # Use consistent pathing
    date_str = snapshotdate
    bronze_directory = "/opt/airflow/datamart/bronze"
    silver_directory = "/opt/airflow/datamart/silver"

    # Ensure directory exists (safe to do regardless of mount)
    os.makedirs(silver_directory, exist_ok=True)

    # Run transformation
    process_silver_table('lms', bronze_directory, silver_directory, date_str, spark)
    
    spark.stop()
    print('\n\n---completed job---\n\n')


def gold_label(snapshotdate, spark:SparkSession):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = set_spark()

    date_str = snapshotdate
    silver_directory = "/opt/airflow/datamart/silver"
    gold_directory = "/opt/airflow/datamart/gold"

    os.makedirs(gold_directory, exist_ok=True)

    process_gold_label(silver_directory, gold_directory, date_str, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":

    spark = set_spark()

    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, required=True, help="Which task to run")
    
    args = parser.parse_args()

    TASK_REGISTRY = {
        "bronze_label": bronze_label,
        "silver_label": silver_label,
        "gold_label": gold_label 
    }

    task_func = TASK_REGISTRY[args.task]

    task_func(args.snapshotdate, spark)
    

