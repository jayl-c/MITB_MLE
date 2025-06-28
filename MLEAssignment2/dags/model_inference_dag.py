from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'inference_pipeline',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2024, 11, 1), # Eg. we make inference on this day
    end_date=datetime(2024, 12, 1),
    catchup=True,

) as dag:

    inference_task = BashOperator(
        task_id='make_inference',
        bash_command =(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py '
            '--snapshotdate "{{ ds }}"'
        ) 
    )

    inference_report = BashOperator(
        task_id='generate_report',
        bash_command =(
            'cd /opt/airflow/scripts && '
            'python3 model_monitor.py '
            '--snapshotdate "{{ ds }}"'
        ) 
    )

inference_task >> inference_report