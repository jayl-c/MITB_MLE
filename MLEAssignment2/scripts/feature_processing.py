import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import *
from utils.helper import *

# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def bronze_clickstream(snapshotdate, spark:SparkSession):
    print('\n\n---starting job---\n\n')

    bronze_directory = "/opt/airflow/datamart/bronze"
    os.makedirs(bronze_directory, exist_ok=True)

    process_bronze_table(
        'clickstream',
        '/opt/airflow/data/feature_clickstream.csv',
        bronze_directory,
        snapshotdate,
        spark
    )
    spark.stop()
    print('\n\n---completed job---\n\n')


def bronze_attributes(snapshotdate, spark: SparkSession):
    print('\n\n---starting job---\n\n')

    bronze_directory = "/opt/airflow/datamart/bronze"
    os.makedirs(bronze_directory, exist_ok=True)

    process_bronze_table(
        'attributes',
        '/opt/airflow/data/features_attributes.csv',
        bronze_directory,
        snapshotdate,
        spark
    )
    spark.stop()
    print('\n\n---completed job---\n\n')


def bronze_financials(snapshotdate, type, spark: SparkSession):
    print('\n\n---starting job---\n\n')

    bronze_directory = "/opt/airflow/datamart/bronze"
    os.makedirs(bronze_directory, exist_ok=True)

    process_bronze_table(
        'financials',
        '/opt/airflow/data/features_financials.csv',
        bronze_directory,
        snapshotdate,
        spark
    )
    spark.stop()
    print('\n\n---completed job---\n\n')


def silver_clickstream(snapshotdate, spark: SparkSession):
    print('\n\n---starting job---\n\n')

    bronze_directory = "/opt/airflow/datamart/bronze"
    silver_directory = "/opt/airflow/datamart/silver"
    os.makedirs(silver_directory, exist_ok=True)

    process_silver_table('clickstream', bronze_directory, silver_directory, snapshotdate, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


def silver_attributes(snapshotdate, spark: SparkSession):
    print('\n\n---starting job---\n\n')

    bronze_directory = "/opt/airflow/datamart/bronze"
    silver_directory = "/opt/airflow/datamart/silver"
    os.makedirs(silver_directory, exist_ok=True)

    process_silver_table('attributes', bronze_directory, silver_directory, snapshotdate, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


def silver_financials(snapshotdate, spark: SparkSession):
    print('\n\n---starting job---\n\n')

    bronze_directory = "/opt/airflow/datamart/bronze"
    silver_directory = "/opt/airflow/datamart/silver"
    os.makedirs(silver_directory, exist_ok=True)

    process_silver_table('financials', bronze_directory, silver_directory, snapshotdate, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


def gold_features(snapshotdate, type, spark: SparkSession):
    print('\n\n---starting job---\n\n')

    silver_db = "/opt/airflow/datamart/silver"
    gold_db = "/opt/airflow/datamart/gold"
    os.makedirs(gold_db, exist_ok=True)

    process_gold_features(silver_db, gold_db, snapshotdate, spark)

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":

    spark = set_spark()

    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    # parser.add_argument("--task", type=str, required=True, help="Which task to run")
    parser.add_argument("--type", type=str, required=True, help="training or inference")
    args = parser.parse_args()

    TASK_REGISTRY = {
        "bronze_clickstream": bronze_clickstream,
        "bronze_attributes": bronze_attributes,
        "bronze_financials": bronze_financials,
        "silver_clickstream": silver_clickstream,
        "silver_attributes": silver_attributes,
        "silver_financials": silver_financials,
        "gold_features": gold_features        
        }
    
    task_func = TASK_REGISTRY[args.task]

    # dates_str_lst = generate_first_of_month_dates(args.snapshotdate)
    # for date_str in tqdm.tqdm(dates_str_lst, total=len(dates_str_lst), desc=f"Processing {args.task}"):
    task_func(args.snapshotdate, spark)  




    
        