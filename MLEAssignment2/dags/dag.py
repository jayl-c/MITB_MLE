from airflow import DAG
# from airflow.providers.standard.operators.bash import BashOperator
# from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import uuid

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'credifraud_ml_pipeline',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2023, 1, 1),
    # end_date=datetime(2024, 12, 1),
    catchup=True,

) as dag:

    job_id = str(uuid.uuid4()).replace("-", "")
    #################
    # data pipeline #
    #################
    # --- label store ---
    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data") # fake task 

    # Parallel processing of financial, clickstream, attribute and lms data.
    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
            '--task bronze_label'
        ),
    )
    # ds stands for date. this is where to input the date based on the schedule provided above. 
    # silver_label_store = DummyOperator(task_id="silver_label_store")
    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 label_store.py '
            '--snapshotdate "{{ ds }}"'
            '--task silver_label'
        ),
    )

    # gold_label_store = DummyOperator(task_id="gold_label_store")
    gold_label_store = BashOperator(
        task_id='run_gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 label_store.py '
            '--snapshotdate "{{ ds }}"'
            '--task gold_label'
        ),
    )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    # --- feature store --- chaining multiple bronze table to silver table
    dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_bronze_1")

    dep_check_source_data_bronze_2 = DummyOperator(task_id="dep_check_source_data_bronze_2")

    dep_check_source_data_bronze_3 = DummyOperator(task_id="dep_check_source_data_bronze_3")

    bronze_clickstream_task = BashOperator(
    task_id='bronze_clickstream',
    bash_command='cd /opt/airflow/scripts && python3 feature_store.py --snapshotdate "{{ ds }}" --task bronze_clickstream'
    )

    bronze_attributes_task = BashOperator(
        task_id='bronze_attributes', 
        bash_command='cd /opt/airflow/scripts && python3 feature_store.py --snapshotdate "{{ ds }}" --task bronze_attributes'
    )

    bronze_financials_task = BashOperator(
        task_id='bronze_financials',
        bash_command='cd /opt/airflow/scripts && python3 feature_store.py --snapshotdate "{{ ds }}" --task bronze_financials'
    )

    silver_clickstream_task = BashOperator(
    task_id='silver_clickstream',
    bash_command='cd /opt/airflow/scripts && python3 feature_store.py --snapshotdate "{{ ds }}" --task silver_clickstream'
    )

    silver_attributes_task = BashOperator(
        task_id='silver_attributes', 
        bash_command='cd /opt/airflow/scripts && python3 feature_store.py --snapshotdate "{{ ds }}" --task silver_attributes'
    )

    silver_financials_task = BashOperator(
        task_id='silver_financials',
        bash_command='cd /opt/airflow/scripts && python3 feature_store.py --snapshotdate "{{ ds }}" --task silver_financials'
    )

    # gold_feature_store = DummyOperator(task_id="gold_feature_store")
    gold_feature_store = BashOperator(
        task_id='gold_features',
        bash_command='cd /opt/airflow/scripts && python3 feature_store.py --snapshotdate "{{ ds }}" --task gold_features', 
        dag=dag
    )
    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    # Bronze layer parallel processing
    dep_check_source_data_bronze_1 >> [bronze_clickstream_task, bronze_attributes_task, bronze_financials_task]
    [bronze_clickstream_task, bronze_attributes_task, bronze_financials_task] >> dep_check_source_data_bronze_2

    # Silver layer processing
    dep_check_source_data_bronze_2 >> [silver_clickstream_task, silver_attributes_task, silver_financials_task]
    [silver_clickstream_task, silver_attributes_task, silver_financials_task] >> dep_check_source_data_bronze_3

    dep_check_source_label_data >> bronze_label_store >> silver_label_store

    # Gold layer processing
    dep_check_source_data_bronze_3 >> gold_feature_store
    gold_feature_store >> feature_store_completed
    gold_label_store >> label_store_completed

    # dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    # dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store
    # dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2 >> gold_feature_store
    model_train = PythonBranchaOperator(
        task_id='training_model',
        bash_command = 'cd /opt/airflow/scripts && python3 model_training.py --job train--snapshotdate "{{ ds }}"'
    )

    deploy_model = BashOperator(
        task_id='deploy_best',
        bash_command = 'cd /opt/airflow/scripts && python3 model_training.py --job deploy --snapshotdate "{{ ds }}"'
    )

    # --- model inference ---
    model_inference_start = DummyOperator(task_id="batch_inference_start")
    
    model_1_inference = DummyOperator(task_id="model_1_inference")

    model_2_inference = DummyOperator(task_id="model_2_inference")

    model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed
    model_inference_start >> model_2_inference >> model_inference_completed

    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_1_monitor = DummyOperator(task_id="model_1_monitor")

    model_2_monitor = DummyOperator(task_id="model_2_monitor")

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed
    model_monitor_start >> model_2_monitor >> model_monitor_completed


    # --- model auto training ---
    # model_automl_start = DummyOperator(task_id="model_automl_start")
    # model_1_automl = DummyOperator(task_id="model_1_automl")
    # model_2_automl = DummyOperator(task_id="model_2_automl")
    # model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    # feature_store_completed >> model_automl_start
    # label_store_completed >> model_automl_start
    # model_automl_start >> model_1_automl >> model_automl_completed
    # model_automl_start >> model_2_automl >> model_automl_completed