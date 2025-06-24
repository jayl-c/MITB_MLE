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

def silver_clickstream(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    spark = set_spark()

    date_str = snapshotdate

    bronze_directory = "datamart/bronze"
    silver_directory = "datamart/silver"

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
    
    process_silver_table('clickstream', bronze_directory, silver_directory, date_str, spark)
    
    spark.stop()
    print('\n\n---completed job---\n\n')

def silver_attributes(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    spark = set_spark()

    date_str = snapshotdate

    bronze_directory = "datamart/bronze"
    silver_directory = "datamart/silver"

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
    
    process_silver_table('attributes', bronze_directory, silver_directory, date_str, spark)
    
    spark.stop()
    print('\n\n---completed job---\n\n')

def silver_financials(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    spark = set_spark()

    date_str = snapshotdate

    bronze_directory = "datamart/bronze"
    silver_directory = "datamart/silver"

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
    
    process_silver_table('financials', bronze_directory, silver_directory, date_str, spark)
    
    spark.stop()
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, required=True, help="Which task to run")
    
    args = parser.parse_args()
    print("Building silver tables...")
    # Route to different functions based on task argument
    if args.task == "silver_clickstream":
        silver_clickstream(args.snapshotdate)
    elif args.task == "silver_attributes":
        silver_attributes(args.snapshotdate)
    elif args.task == "silver_financials":
        silver_financials(args.snapshotdate)
    else:
        raise ValueError(f"Unknown task: {args.task}")   
    
    # # Call main with arguments explicitly passed
    # main(args.snapshotdate)
