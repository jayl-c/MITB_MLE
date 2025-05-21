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
from utils.data_processing_attributes import process_customer_attributes
import utils.generator_fn as fn

spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

def process_customer_features(date_str, bronze_feature_directory, silver_feature_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    print(f'------------Processing for', date_str, '------------')
    # connect to financials table
    financials_partition_name = "bronze_financials_" + date_str.replace('-','_') + '.csv'
    financials_filepath = bronze_feature_directory + "financials/" + financials_partition_name

    financials_df = spark.read.csv(financials_filepath, header=True, inferSchema=True)
    print('loaded from:', financials_filepath, 'row count:', financials_df.count())

    # Format numerical columns to remove non-numeric characters
    df1 = financials_df.withColumn("Annual_Income", regexp_replace(F.col("Annual_Income"), "[^0-9\\.]", "")) \
        .withColumn("Num_of_Loan", regexp_replace(F.col("Num_of_Loan"), "[^0-9s\\.]", "")) \
        .withColumn("Num_of_Delayed_Payment", regexp_replace(F.col("Num_of_Delayed_Payment"), "[^0-9\\.]", "")) \
        .withColumn("Changed_Credit_Limit", regexp_replace(F.col("Changed_Credit_Limit"), "[^0-9\\.]", "")) \
        .withColumn("Outstanding_Debt", regexp_replace(F.col("Outstanding_Debt"), "[^0-9\\.]", "")) \
        .withColumn("Amount_invested_monthly", regexp_replace(F.col("Amount_invested_monthly"), "[^0-9\\.]", "")) \
        .withColumn("Monthly_Balance", regexp_replace(F.col("Monthly_Balance"), "[^0-9\\.]", ""))

    # Format data types
    df2 = df1.withColumn("Annual_Income", F.col("Annual_Income").cast("double")) \
        .withColumn("Num_of_Loan", F.col("Num_of_Loan").cast("int")) \
        .withColumn("Num_of_Delayed_Payment", F.col("Num_of_Delayed_Payment").cast("int")) \
        .withColumn("Num_Credit_Inquiries", F.col("Num_Credit_Inquiries").cast("int")) \
        .withColumn("Changed_Credit_Limit", F.col("Changed_Credit_Limit").cast("double")) \
        .withColumn("Amount_invested_monthly", F.col("Amount_invested_monthly").cast("double"))

    df3 = df2.withColumn("Annual_Income", F.round(F.col("Annual_Income"), 2)) \
        .withColumn("Monthly_Inhand_Salary", F.round(F.col("Monthly_Inhand_Salary"), 2)) \
        .withColumn("Num_of_Loan", F.round(F.col("Num_of_Loan"), 2)) \
        .withColumn("Num_of_Delayed_Payment", F.round(F.col("Num_of_Delayed_Payment"), 2)) \
        .withColumn("Changed_Credit_Limit", F.round(F.col("Changed_Credit_Limit"), 2)) \
        .withColumn("Outstanding_Debt", F.round(F.col("Outstanding_Debt"), 2)) \
        .withColumn("Credit_Utilization_Ratio", F.round(F.col("Credit_Utilization_Ratio"), 2)) \
        .withColumn("Total_EMI_per_month", F.round(F.col("Total_EMI_per_month"), 2)) \
        .withColumn("Amount_invested_monthly", F.round(F.col("Amount_invested_monthly"), 2)) \
        .withColumn("Monthly_Balance", F.round(F.col("Monthly_Balance"), 2))

    df4 = df3.withColumn("Credit_History_Age_Months", 
                        (coalesce(regexp_extract(F.col("Credit_History_Age"), "([0-9]+) Years?", 1).cast("float"), F.lit(0)) * 12) + 
                        coalesce(regexp_extract(F.col("Credit_History_Age"), "([0-9]+) Months?", 1).cast("float"), F.lit(0))
                        )

    df5 = df4.drop("Credit_History_Age")

    # Remove extreme outlier (3 Stdev) and recheck stats
    percentile_997_mb = df5.approxQuantile("Monthly_Balance", [0.997], 0.01)[0]

    df6 = df5.filter(F.col("Monthly_Balance") < percentile_997_mb)

    # Remove number of bank accounts < 0 as it should not be negative
    df7 = df6.filter(F.col("Num_Bank_Accounts") >= 0)
    df7 = df7.withColumn("Changed_Credit_Limit", F.when(F.col("Changed_Credit_Limit").isNull(), 0).otherwise(F.col("Changed_Credit_Limit")))

    # Stats for item counts looks odd. Therefore, we will conduct winsorisation to cap the number of accounts value to those in the 95th percentile. 
    percentile_95_acc = df7.approxQuantile("Num_Bank_Accounts", [0.95], 0.01)[0]
    percentile_95_cc = df7.approxQuantile("Num_Credit_Card", [0.95], 0.01)[0]  
    percentile_95_loan = df7.approxQuantile("Num_of_Loan", [0.95], 0.01)[0]  
    percentile_95_dp = df7.approxQuantile("Num_of_Delayed_Payment", [0.95], 0.01)[0]  
    percentile_95_ci = df7.approxQuantile("Num_Credit_Inquiries", [0.95], 0.01)[0]  

    df8 = df7.withColumn("Num_Bank_Accounts", F.when(F.col("Num_Bank_Accounts") > percentile_95_acc, percentile_95_acc).otherwise(F.col("Num_Bank_Accounts"))) \
        .withColumn("Num_Credit_Card", F.when(F.col("Num_Credit_Card") > percentile_95_cc, percentile_95_cc).otherwise(F.col("Num_Credit_Card"))) \
        .withColumn("Num_of_Loan", F.when(F.col("Num_of_Loan") > percentile_95_loan, percentile_95_loan).otherwise(F.col("Num_of_Loan"))) \
        .withColumn("Num_of_Delayed_Payment", F.when(F.col("Num_of_Delayed_Payment") > percentile_95_dp, percentile_95_dp).otherwise(F.col("Num_of_Delayed_Payment"))) \
        .withColumn("Num_Credit_Inquiries", F.when(F.col("Num_Credit_Inquiries") > percentile_95_ci, percentile_95_ci).otherwise(F.col("Num_Credit_Inquiries")))

    df8 = df8.withColumn("Num_Bank_Accounts", F.col("Num_Bank_Accounts").cast("int")) \
        .withColumn("Num_Credit_Card", F.col("Num_Credit_Card").cast("int")) \
        .withColumn("Num_of_Loan", F.col("Num_of_Loan").cast("int")) \
        .withColumn("Num_of_Delayed_Payment", F.col("Num_of_Delayed_Payment").cast("int")) \
        .withColumn("Num_Credit_Inquiries", F.col("Num_Credit_Inquiries").cast("int"))

    percentile_95_int = df8.approxQuantile("Interest_Rate", [0.95], 0.01)[0]  
    df8 = df8.withColumn("Interest_Rate", F.when(F.col("Interest_Rate") > percentile_95_int, percentile_95_int).otherwise(F.col("Interest_Rate")))
    df8 = df8.withColumn("Interest_Rate", F.round(F.col("Interest_Rate"), 2))

    # Clean and split the text (handle nulls first)
    df9 = df8.withColumn("Type_of_Loan", when(col("Type_of_Loan").isNull(), "No loan").otherwise(col("Type_of_Loan")))

    df9 = df9.withColumn("loan_types_array", 
                    F.split(regexp_replace(F.col("Type_of_Loan"), " and ", ", "), ","))

    # Use built-in trim function
    df9 = df9.withColumn("loan_types_array", F.transform(F.col("loan_types_array"), F.trim))

    # Remove empty strings from the arrays themselves
    df9 = df9.withColumn("loan_types_array", 
                        F.filter(F.col("loan_types_array"), lambda x: (x != "") & (x.isNotNull())))

    # Handle edge case: arrays that become empty after filtering
    df9 = df9.withColumn("loan_types_array",
                        when(F.size(col("loan_types_array")) == 0, F.array(F.lit("No loan")))
                        .otherwise(col("loan_types_array")))

    # Get all unique loan types
    loan_types = df9.select(explode("loan_types_array").alias("loan_type")).distinct().collect()
    unique_loan_types = [row['loan_type'] for row in loan_types]

    # One-hot encode loan types
    for loan_type in unique_loan_types:
        safe_column_name = loan_type.replace(" ", "_").replace("-", "_").lower()
        df9 = df9.withColumn(f"loan_type_{safe_column_name}", 
                            F.array_contains("loan_types_array", loan_type).cast("int"))
    
    df9 = df9.drop("loan_type_no_loan")

    # One hot encode categorical variables
    df10 = df9.withColumn("Credit_Mix", F.when(F.col("Credit_Mix") == "_", "Invalid").otherwise(F.col("Credit_Mix"))) 
    df10 = fn.one_hot_encode_categorical(df10, "Credit_Mix")
    df10 = fn.one_hot_encode_categorical(df10, "Payment_of_Min_Amount")

    # Extract spending level
    df11 = df10.withColumn("Spending_Level", 
                        F.when(F.col("Payment_Behaviour").contains("High"), 1)
                        .when(F.col("Payment_Behaviour").contains("Low"), 0)
                        .otherwise(-1)) \
                .withColumn("Payment_Size",
                        F.when(F.col("Payment_Behaviour").contains("Small"), 1)
                        .when(F.col("Payment_Behaviour").contains("Medium"), 2)
                        .when(F.col("Payment_Behaviour").contains("Large"), 3)
                        .otherwise(0))
   
    new_financials_df = df11.drop("Monthly_Inhand_Salary", "Type_of_Loan", "Payment_Behaviour", "loan_types_array", "payment_of_min_amount", "payment_of_min_amount_no", "credit_mix")
    # new_financials_df = new_financials_df.filter(F.col("snapshot_date") <= snapshot_date)

    print('Processing financials data complete')
    print('Financials data row count:', new_financials_df.count())

    print('Begin to process attributes data')
    # connect to attributes table
    attributes_partition_name = "bronze_attributes_" + date_str.replace('-','_') + '.csv'
    attributes_filepath = bronze_feature_directory + "attributes/" + attributes_partition_name

    processed_attributes = process_customer_attributes(attributes_filepath, spark)

    print('Processing attributes data complete')
    print('Attributes data row count:', processed_attributes.count())

    # Join financials and attributes data
    print('Merging financials and attributes data complete')
    
    features_df = new_financials_df.join(processed_attributes, on=["Customer_ID","snapshot_date"], how="inner")
    print('Merge complete. features_df row count:', features_df.count())

    print('Final financials features data row count:', new_financials_df.count())
    partition_name = "/attributes/silver_customer_features_daily_" + date_str.replace('-','_') + '.parquet'
    filepath = silver_feature_directory + partition_name
    new_financials_df.write.mode("overwrite").parquet(filepath)
    print('\n \n Data processing has completed')
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    return new_financials_df

def process_silver_cs_features(snapshot_date_str, bronze_feature_directory, silver_feature_directory, spark):
    
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    cs_partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    cs_filepath = bronze_feature_directory + "clickstream/" + cs_partition_name
    
    cs_df = spark.read.csv(cs_filepath, header=True, inferSchema=True)
    print('processing clickstream data starting....')
    
    fe_count = [num for num in range(1,21)]
    column_type_map = {
        f'fe_{num}': FloatType() for num in fe_count
    }
    for column, new_type in column_type_map.items():
        cs_df = cs_df.withColumn(column, col(column).cast(new_type))
    column_type_map['CustomerID'] = StringType()

    print("clickstream processing complete")
    print("clickstream row_count:",cs_df.count())

    partition_name = "clickstream/silver_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_feature_directory + partition_name
    cs_df.write.mode("overwrite").parquet(filepath)
    print('\n \n Data processing has completed')
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    return cs_df 