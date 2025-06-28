
import os
import glob
import pyspark
import pyspark.sql.functions as F

from tqdm import tqdm

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType, NumericType, ArrayType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler

def read_silver_table(table, silver_db, spark):
    """
    Helper function to read a silver table and ensure consistent schema
    """
    folder_path = os.path.join(silver_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    print(folder_path)
    # Read the table
    df = spark.read.option("header", "true").parquet(*files_list)
    
    # # Ensure snapshot_date is consistently string
    # if 'snapshot_date' in df.columns:
    #     df = df.withColumn("snapshot_date", col("snapshot_date").cast("string"))
    
    return df

def read_gold_label(gold_db, snapshot_date, spark):
    """
    Helper to read gold label data filtered by snapshot_date from a partitioned folder.
    """
    df = spark.read.option("header", "true").parquet(os.path.join(gold_db, 'label_store'))
    return df.filter(col('snapshot_date') == snapshot_date)

############################
# Label Store
############################
def build_label_store(mob, dpd, df):
    """
    Function to build label store
    """
    ####################
    # Create labels
    ####################

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    return df

############################
# Feature Store
############################
def one_hot_encoder(df, category_col):
    """
    Utility function for one hot encoding
    """
    # Get label encoding
    indexer = StringIndexer(inputCol=category_col, outputCol=f"{category_col}_index", handleInvalid="keep")
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)

    # Transform into one hot encoding
    encoder = OneHotEncoder(inputCol=f"{category_col}_index", outputCol=f"{category_col}_ohe", dropLast=False)
    df = encoder.fit(df).transform(df)
    vector_to_array_udf = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df = df.withColumn(f"{category_col}_array", vector_to_array_udf(f"{category_col}_ohe"))

    # Split into columns
    categories = [cat.lower() for cat in indexer_model.labels]

    for i, cat in enumerate(categories):
        df = df.withColumn(f"{category_col}_{cat}", df[f"{category_col}_array"][i])
        df = df.withColumn(f"{category_col}_{cat}", col(f"{category_col}_{cat}").cast(IntegerType()))

    # Optional: drop intermediate columns
    df = df.drop(category_col, f"{category_col}_index", f"{category_col}_ohe", f"{category_col}_array")
    return df

def build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_lms, df_label):
  
    #############
    # Join attributes and financials into a single matrix
    #############
    df_joined = df_attributes.join(df_financials, on=["customer_id", "snapshot_date"], how="inner")       
    df_joined = df_joined.join(df_loan_type, on=["customer_id", "snapshot_date"], how="inner")
    df_joined = df_joined.drop("name", "ssn", "type_of_loan", "credit_history_age", "type_of_loan") # drop identifiers and duplicated columns

    df_joined = df_joined.join(df_label.select("customer_id"), on="customer_id", how="left_semi") # filter by user IDs that have labels
 
    # Merge credit history age into one column
    df_joined = df_joined.withColumn("credit_history_age_month", F.col("credit_history_age_year") * 12 + F.col("credit_history_age_month"))
    df_joined = df_joined.drop("credit_history_age_year")

    print("1. Joined dataframes")

    #############
    # Impute mean into null numeric variables
    #############
    df_joined = df_joined.withColumn("snapshot_date", F.col("snapshot_date").cast("date"))
    df_joined = df_joined.withColumn("num_bank_accounts", F.col("num_bank_accounts").cast(IntegerType()))
    df_joined = df_joined.withColumn("num_credit_card", F.col("num_credit_card").cast(IntegerType()))
    df_joined = df_joined.withColumn("interest_rate", F.col("interest_rate").cast(IntegerType()))
    df_joined = df_joined.withColumn("num_of_loan", F.col("num_of_loan").cast(IntegerType()))
    df_joined = df_joined.withColumn("num_of_delayed_payment", F.col("num_of_delayed_payment").cast(IntegerType()))

    numeric_columns = [col.name for col in df_joined.schema.fields 
                   if isinstance(col.dataType, NumericType) 
                   and col.name not in ["snapshot_date", "customer_id"]]    
    
    imputer = Imputer(inputCols=numeric_columns, outputCols=numeric_columns)
    df_joined = imputer.fit(df_joined).transform(df_joined)

    print("2. Imputed mean into numeric variables")

    #############
    # Turn categorical variables into one hot encoded columns
    #############
    df_joined = one_hot_encoder(df_joined, "occupation")
    df_joined = one_hot_encoder(df_joined, "payment_of_min_amount")
    df_joined = one_hot_encoder(df_joined, "credit_mix")
    df_joined = one_hot_encoder(df_joined, "payment_behaviour_spent")
    df_joined = one_hot_encoder(df_joined, "payment_behaviour_value")

    print("3. Performed one-hot encoding")

    #############
    # Aggregate mean clickstream data for each user
    #############

    # Filter clickstream data
    df_lms_mob0 = df_lms.filter(col("mob") == 0)
    df_lms_mob0 = df_lms_mob0.withColumnRenamed("snapshot_date", "mob_date")
    df_lms_mob0 = df_lms_mob0.select("customer_id", "mob_date") 
    df_clickstream_filtered = df_clickstream.join(df_lms_mob0, on="customer_id", how="inner") # filter by user IDs that have labels
    df_clickstream_filtered = df_clickstream_filtered.filter(col("snapshot_date") <= col("mob_date"))

    # Do mean aggregation
    agg_exprs = [F.avg(f'fe_{i}').alias(f"avg_fe_{i}") for i in range(1, 21)]
    df_clickstream_filtered = df_clickstream_filtered.groupBy("customer_id").agg(*agg_exprs)

    print("4. Processed clickstream data")

    #############
    # Join clickstream data with attributes and financials
    #############
    df_joined = df_joined.join(df_clickstream_filtered, on=["customer_id"], how="left")

    print("5. Joined clickstream data with the rest of the features")

    return df_joined

############################
# Pipeline
############################
def process_gold_table(silver_db, gold_db, snapshot_date, spark):
    """
    Wrapper function to build all gold tables
    """
    # Read silver tables
    df_attributes = read_silver_table('attributes', silver_db, spark)
    df_clickstream = read_silver_table('clickstream', silver_db, spark)
    df_financials = read_silver_table('financials', silver_db, spark)
    df_loan_type = read_silver_table('loan_type', silver_db, spark)
    df_lms = read_silver_table('lms', silver_db, spark)

    # Build label store
    print("Building label store...")
    df_label = build_label_store(6, 30, df_lms)
 
    # Build features
    print("Building features...")
    df_features = build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_lms, df_label)

    partition_name = snapshot_date

    # Partition and save features
    # for date_str in tqdm.tqdm(partitions_list, total=len(dates_str_lst), desc="Saving features"):
    partition_name = snapshot_date.replace('-','_') + '.parquet'
    feature_filepath = os.path.join(gold_db, 'feature_store', partition_name)
    df_features.filter(col('snapshot_date')==snapshot_date).write.mode('overwrite').parquet(feature_filepath)

    # Partition and save labels
    # for date_str in tqdm.tqdm(partitions_list, total=len(dates_str_lst), desc="Saving labels"):
    partition_name = snapshot_date.replace('-','_') + '.parquet'
    label_filepath = os.path.join(gold_db, 'label_store', partition_name)
    df_label.filter(col('snapshot_date')==snapshot_date).write.mode('overwrite').parquet(label_filepath)

    return df_features, df_label

def process_gold_label(silver_db, gold_db, date_str, spark):
    """
    Wrapper function to build gold label table
    """
    df_lms = read_silver_table('lms', silver_db, spark)

    # Build label store
    print("Building label store...")
    df_label = build_label_store(6, 30, df_lms)

    partition_name = date_str.replace('-','_') + '.parquet'
    label_filepath = os.path.join(gold_db, 'label_store', partition_name)
    # df_label = spark.read.option("header", "true").parquet(label_filepath)
    df_label.filter(col('snapshot_date')==date_str).write.mode('overwrite').parquet(label_filepath)

    return df_label

def process_gold_features(silver_db, gold_db, date_str, spark):
    """
    Wrapper function to build gold features
    """
    # Read silver tables
    df_attributes = read_silver_table('attributes', silver_db, spark)
    df_clickstream = read_silver_table('clickstream', silver_db, spark)
    df_financials = read_silver_table('financials', silver_db, spark)
    df_loan_type = read_silver_table('loan_type', silver_db, spark)
    df_lms = read_silver_table('lms', silver_db, spark)
    
    df_label = build_label_store(6, 30, df_lms)

    df_label = df_label.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

    partition_name = date_str.replace('-', '_') + '.parquet'
    label_filepath = os.path.join(gold_db, 'label_store', partition_name)

    df_label.filter(F.col('snapshot_date') == date_str) \
            .write.mode('overwrite').parquet(label_filepath)

    ##############################
    # 3. Build features
    ##############################
    print("Building features...")
    df_features = build_feature_store(
        df_attributes, df_financials, df_loan_type,
        df_clickstream, df_lms, df_label
    )

    df_features = df_features.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

    feature_filepath = os.path.join(gold_db, "feature_store", partition_name)

    # Filter for the correct date
    df_features.filter(F.col('snapshot_date') == date_str) \
               .write.mode("overwrite") \
               .parquet(feature_filepath)

    print(f"Feature store saved at: {feature_filepath}")

    return df_features