from airflow import DAG
# from airflow.providers.standard.operators.bash import BashOperator
# from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
# from scripts.model_monitor import check_model_drift, calculate_psi, performance_report
# from MLEAssignment2.scripts.logreg_modeltrain import train_model, pick_model_and_deploy
import uuid

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'data_pipeline',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,

) as dag:

    #################
    # data pipeline #
    #################    

    # --- LABEL STORE --- #
    # Parallel processing of financial, clickstream, attribute and lms data.
    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 label_processing.py '
            '--snapshotdate "{{ ds }}" '
            '--task bronze_label '
        ),
    )

    # silver_label_store = DummyOperator(task_id="silver_label_store")
    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 label_processing.py '
            '--snapshotdate "{{ ds }}" '
            '--task silver_label '
        ),
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
    label_store_completed = DummyOperator(task_id="label_store_completed")

    # --- feature store --- #
    # chaining multiple bronze table to silver table
    bronze_clickstream_task = BashOperator(
    task_id='bronze_clickstream',
    bash_command=(
        'cd /opt/airflow/scripts && '
        'python3 feature_processing.py '
        '--snapshotdate "{{ ds }}" '
        '--task bronze_clickstream '
        ),
    )

    bronze_attributes_task = BashOperator(
        task_id='bronze_attributes', 
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 feature_processing.py'
            ' --snapshotdate "{{ ds }}" '
            '--task bronze_attributes '
        )
    )

    bronze_financials_task = BashOperator(
        task_id='bronze_financials',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 feature_processing.py '
            '--snapshotdate "{{ ds }}" '
            '--task bronze_financials '
        )
    )

    silver_clickstream_task = BashOperator(
    task_id='silver_clickstream',
    bash_command=(
        'cd /opt/airflow/scripts && '
        'python3 feature_processing.py '
        '--snapshotdate "{{ ds }}" '
        '--task silver_clickstream ' \
        )
    )

    silver_attributes_task = BashOperator(
        task_id='silver_attributes', 
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 feature_processing.py '
            '--snapshotdate "{{ ds }}" '
            '--task silver_attributes '
        )
    )

    silver_financials_task = BashOperator(
        task_id='silver_financials',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 feature_processing.py '
            '--snapshotdate "{{ ds }}" '
            '--task silver_financials'
        )
    )

    # gold_feature_store = BashOperator(
    #     task_id='gold_features',
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 feature_processing.py '
    #         '--snapshotdate "{{ ds }}" '
    #         '--task gold_features '
    #     )
    # )
    # Dummy operators
    bronze_feature_processing_completed = DummyOperator(task_id="bronze_feature_processing_completed")
    bronze_label_processing_completed = DummyOperator(task_id="bronze_label_processing_completed")
    silver_processing_completed = DummyOperator(task_id="silver_processing_completed")
    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    gold_stores_completed = DummyOperator(task_id="gold_store_completed")
    
    # Define task dependencies to run scripts sequentially
    # Bronze layer processing
    [bronze_clickstream_task, bronze_attributes_task, bronze_financials_task] >> bronze_feature_processing_completed
    bronze_label_store >> bronze_label_processing_completed    

    # Silver layer processing
    bronze_feature_processing_completed >> [silver_clickstream_task, silver_attributes_task, silver_financials_task] >> silver_processing_completed
    bronze_label_processing_completed >> silver_label_store >> silver_processing_completed

    # Silver layer processing
    bronze_feature_processing_completed >> [silver_clickstream_task, silver_attributes_task, silver_financials_task] >> silver_processing_completed
    bronze_label_processing_completed >> silver_label_store >> silver_processing_completed   

    # Gold layer processing
    # silver_processing_completed >> [gold_feature_store, gold_label_store] >> gold_stores_completed
    
    # gold_feature_store >> feature_store_completed
    # gold_label_store >> label_store_completed

