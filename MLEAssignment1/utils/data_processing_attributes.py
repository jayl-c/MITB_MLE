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
import seaborn as sns
import math

from pyspark.sql.functions import col, round, expr, count, mean, min, max, stddev, skewness, kurtosis, percentile_approx, lit, split, explode, trim, regexp_replace, transform, array_contains, when, regexp_extract, coalesce, isnull
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import col, regexp_replace
import utils.generator_fn as fn

spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

def process_customer_attributes(attributes_filepath, spark):
    
    # snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # # connect to financials table
    # attributes_partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    # financials_filepath = bronze_feature_directory + "attributes/" + attributes_partition_name

    # financials_df = spark.read.csv(financials_filepath, header=True, inferSchema=True)
    # connect to financials table
    attributes_df = spark.read.csv(attributes_filepath, header=True, inferSchema=True)
    print('loaded from:', attributes_df, 'row count:', attributes_df.count())

    attributes_df = attributes_df.withColumn("Occupation", F.when(~F.col("Occupation").rlike('[a-zA-Z]'), "Others")
                             .otherwise(F.col("Occupation"))) \
                             .withColumn("has_SSN", F.when(~F.col("SSN").rlike("[0-9]{3}-[0-9]{2}-[0-9]{4}"), F.lit(0))
                             .otherwise(1)) 
    
    attributes_df = fn.one_hot_encode_categorical(attributes_df, "Occupation")

    attributes_df = attributes_df.withColumn("Age", regexp_replace(F.col("Age"), "[^0-9]", ""))
    attributes_df = attributes_df.withColumn("Age", F.col("Age").cast("int"))
    median_age = attributes_df.approxQuantile("Age", [0.5], 0.01)[0]
    attributes_df = attributes_df.withColumn("Age", F.when((F.col("Age") > 100) | F.col("Age").isNull(), median_age)  
                                                     .otherwise(F.col("Age")))
    new_attributes_df = attributes_df.drop("Occupation", "SSN", "Name")

    return new_attributes_df

