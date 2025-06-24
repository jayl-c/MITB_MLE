from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import col, when

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def one_hot_encode_categorical(df, categorical_column):

    # Get unique values
    unique_values = df.select(categorical_column).distinct().collect()
    unique_values = [row[categorical_column] for row in unique_values]
    
    # Create column prefix based on the original column name
    column_prefix = categorical_column.lower()
    
    # One-hot encode each value
    for value in unique_values:
        # Clean column name to ensure it's valid
        if value is not None:  # Handle None values
            safe_value = str(value).replace(" ", "_").replace("-", "_").replace(".", "_").lower()
            
            # Create the one-hot encoded column
            df = df.withColumn(f"{column_prefix}_{safe_value}",
                             when(col(categorical_column) == value, 1).otherwise(0))
    
    return df

