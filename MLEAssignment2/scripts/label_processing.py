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
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import read_silver_table, build_label_store, process_gold_label

# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"
def set_spark():
    spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    return spark

def bronze_label(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = set_spark()

    # load arguments
    date_str = snapshotdate
    
    # create bronze datalake
    bronze_lms_directory = "datamart/bronze"
    
    if not os.path.exists(bronze_lms_directory):
        os.makedirs(bronze_lms_directory)

    # run data processing
    process_bronze_table('lms', 'data/lms_loan_daily.csv', bronze_lms_directory, date_str, spark)
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

def silver_label(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = set_spark()

    # load arguments
    date_str = snapshotdate

    # bronze directory
    bronze_directory = "datamart/bronze"
    
    # create silver datalake
    silver_directory = "datamart/silver"

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    # run data processing
    process_silver_table('lms', bronze_directory, silver_directory, date_str, spark)
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

def gold_label(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = set_spark()

    # load arguments
    date_str = snapshotdate
 
    # silver datalake
    silver_directory = "datamart/silver"

    # gold directory
    gold_directory = "datamart/gold"

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    # run data processing
    process_gold_label(silver_directory, gold_directory, date_str, spark)

    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, required=True, help="Which task to run")
    
    args = parser.parse_args()

    # Route to different functions based on task argument
    if args.task == "bronze_label":
        bronze_label(args.snapshotdate)
    elif args.task == "silver_label":
        silver_label(args.snapshotdate)
    elif args.task == "gold_label":
        gold_label(args.snapshotdate)
    else:
        raise ValueError(f"Unknown task: {args.task}")   
    
    # # Call main with arguments explicitly passed
    # main(args.snapshotdate)
