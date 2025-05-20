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

def process_bronze_feature_tables(date_str, bronze_base_directory, spark):
    # Prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Source data paths - IRL connect to back end source system
    data_paths = {
        'financials': "data/features_financials.csv",
        'attributes': "data/features_attributes.csv",
        'clickstream': "data/feature_clickstream.csv"
    }
    
    # Dictionary to store the loaded DataFrames
    dfs = {}
    
    # Process each data type separately
    for data_type, file_path in data_paths.items():
        print(f"Processing {data_type} data for {date_str}...")
        
        # Create directory if it doesn't exist
        type_directory = f"{bronze_base_directory}/{data_type}/"
        os.makedirs(type_directory, exist_ok=True)
        
        # Load data - IRL ingest from back end source system
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        
        # Filter by snapshot date if the column exists
        if 'snapshot_date' in df.columns:
            df = df.filter(col('snapshot_date') == snapshot_date)
        
        # Store the DataFrame
        dfs[data_type] = df
        
        print(f"{data_type} data for {date_str} row count: {df.count()}")
        
        # Save bronze table to datamart
        partition_name = f"bronze_{data_type}_{date_str.replace('-','_')}.csv"
        filepath = os.path.join(type_directory, partition_name)
        
        # Save as CSV
        df.toPandas().to_csv(filepath, index=False)
        print(f'Saved {data_type} data to: {filepath}')
    
    print(f"All bronze feature tables for {date_str} processed successfully.")
    return dfs