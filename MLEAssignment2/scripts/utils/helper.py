import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def set_spark():
    spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    return spark

def generate_first_of_month_dates_past_6_months(snapshot_date_str):
    """
    Generate a list of the first day of the month for the past 6 months.

    Parameters:
    -----------
    snapshot_date_str : str
        The snapshot date in the format 'YYYY-MM-DD'.

    Returns:
    --------
    list : A list of the first day of the month for the past 6 months.
    """
    # Convert the snapshot date to a datetime object
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Generate the first day of the month for the past 6 months
    for i in range(6):
        first_of_month = (snapshot_date - relativedelta(months=i)).replace(day=1)
        first_of_month_dates.append(first_of_month.strftime("%Y-%m-%d"))

    return first_of_month_dates