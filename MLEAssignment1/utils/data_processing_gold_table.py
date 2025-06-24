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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(date_str, snapshot_date, silver_feature_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    enddate_str = datetime.strptime(snapshot_date, "%Y-%m-%d")
    
    # connect to customer features silver table
    customer_features_partition_name = "silver_financial_attributes.parquet"
    custf_filepath = silver_feature_directory + customer_features_partition_name
    custf_df = spark.read.parquet(custf_filepath)
    print('loaded from:', custf_filepath, 'row count:', custf_df.count())
    # custf_df = custf_df.filter(col("snapshot_date") <= enddate_str)
    custf_df = custf_df.drop("snapshot_date")

    # connect to clickstream features silver table
    clickstream_features_partition_name = "clickstream/silver_clickstream_daily_" + date_str.replace('-','_') + '.parquet'
    cs_filepath = silver_feature_directory + clickstream_features_partition_name
    cs_df = spark.read.parquet(cs_filepath)

    cs_df = cs_df.filter(col("snapshot_date") <= enddate_str)
    fstore = cs_df.join(custf_df, on=["Customer_ID"], how="inner")

    gold_partition_name = "gold_feature_store_" + date_str.replace('-','_') + '.parquet'
    gold_filepath = gold_feature_store_directory + gold_partition_name
    fstore.write.mode("overwrite").parquet(gold_filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', gold_filepath)
    
    return fstore