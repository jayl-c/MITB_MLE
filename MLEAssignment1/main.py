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

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
import utils.data_processing_features
import utils.generator_fn as fn

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2023-02-01"

dates_str_lst = fn.generate_first_of_month_dates(start_date_str, end_date_str)
print('generating data from', start_date_str, 'to', end_date_str)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)

bronze_feature_directory = "datamart/bronze/features/"
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_feature_tables(date_str, bronze_feature_directory, spark)

# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# create datalake for features
silver_feature_directory = "datamart/silver/features/"

if not os.path.exists(silver_feature_directory):
    os.makedirs(silver_feature_directory)

for date_str in dates_str_lst:
    utils.data_processing_features.process_customer_features(date_str, bronze_feature_directory, silver_feature_directory, spark)
    utils.data_processing_features.process_silver_cs_features(date_str, bronze_feature_directory, silver_feature_directory, spark)


# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 60, mob = 6)

# create gold tables for features
gold_feature_directory = "datamart/gold/feature_store/"

if not os.path.exists(silver_feature_directory):
    os.makedirs(silver_feature_directory)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_features_gold_table(date_str, end_date_str, silver_feature_directory, gold_feature_directory, spark)


folder_path = gold_label_store_directory
files_list = glob.glob(os.path.join(folder_path, '*.parquet'))
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()



    