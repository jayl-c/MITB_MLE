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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write, partition by date
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_bronze_feature_tables(date_str, bronze_directory, spark):
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    # Dictionary to store the loaded DataFrames
    dfs = {}
    
    # Process non-partitioned data (financials and attributes) - only once
    non_partitioned_data = {
        'financials': "data/features_financials.csv",
        'attributes': "data/features_attributes.csv"
    }
    
    for data_type, file_path in non_partitioned_data.items():
        # print(f"Processing {data_type} data...")
        
        # Create directory if it doesn't exist
        type_directory = f"{bronze_directory}"
        os.makedirs(type_directory, exist_ok=True)
        
        # Load data - IRL ingest from back end source system
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        # Store the DataFrame
        dfs[data_type] = df
        
        # print(f"{data_type} data row count: {df.count()}")
        
        # Save as single file (not partitioned by date)
        partition_name = f"bronze_{data_type}.csv"
        filepath = type_directory + partition_name
        
        # Save as CSV
        df.toPandas().to_csv(filepath, index=False)
        # print(f'Saved {data_type} data to: {filepath}')
    
    # Process clickstream data (partitioned by date) - saved in one folder with date-partitioned filenames
    print(f"Processing clickstream data for {date_str}...")
    
    clickstream_directory = f"{bronze_directory}/clickstream/"
    # Load clickstream data
    df = spark.read.csv("data/feature_clickstream.csv", header=True, inferSchema=True)
    
    if 'snapshot_date' in df.columns:
        df = df.filter(col('snapshot_date') == snapshot_date)
    # Store the DataFrame
    dfs['clickstream'] = df
    
    print(f"clickstream data for {date_str} row count: {df.count()}")
    
    # Save with date-partitioned filename in the clickstream folder
    partition_name = f"bronze_clickstream_{date_str.replace('-','_')}.csv"
    filepath = clickstream_directory + partition_name
    
    # Save as CSV
    df.toPandas().to_csv(filepath, index=False)
    print(f'Saved clickstream data to: {filepath}')
    
    print(f"All bronze feature tables for {date_str} processed successfully.")
    return dfs