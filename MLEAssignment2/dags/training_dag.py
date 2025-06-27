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
    'training_pipeline',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2024, 11, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,

) as dag:

    gold_feature_store = BashOperator(
        task_id='gold_features',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 feature_processing.py '
            '--snapshotdate "{{ ds }}" '
            '--task gold_features '
        )
    )

    gold_label_store = BashOperator(
    task_id='run_gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 label_processing.py '
            '--snapshotdate "{{ ds }}" '
            '--task gold_label '
        ),
    )
    
    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    gold_feature_store_completed = DummyOperator(task_id="gold_feature_store_completed")
    gold_label_store_completed = DummyOperator(task_id="gold_label_store_completed")
    gold_stores_completed = DummyOperator(task_id="gold_store_completed")

    train_xgb_model = BashOperator(
        task_id='training_model_xgb',
        bash_command =(
            'cd /opt/airflow/scripts && '
            'python3 xgb_modeltrain.py '
            '--snapshotdate "{{ ds }}"'
        ) 
    )

    train_lg_model = BashOperator(
        task_id='training_model_lg',
        bash_command =(
            'cd /opt/airflow/scripts && '
            'python3 logreg_modeltrain.py '
            '--snapshotdate "{{ ds }}"'
        ) 
    )


    gold_feature_store >> gold_feature_store_completed
    gold_label_store >> gold_label_store_completed

    [gold_feature_store_completed,gold_label_store_completed] >> train_xgb_model
    [gold_feature_store_completed,gold_label_store_completed] >> train_lg_model

