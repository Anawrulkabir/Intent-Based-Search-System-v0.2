from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import boto3

def ingest_data():
    data = pd.DataFrame({"product": ["red shoes"], "category": ["footwear"]})
    s3 = boto3.client("s3")
    data.to_csv("/tmp/data.csv")
    s3.upload_file("/tmp/data.csv", "your-bucket-name", "data.csv")
    print("Data ingested")

with DAG("ecommerce_pipeline", start_date=datetime(2025, 4, 1), schedule_interval="@daily", catchup=False) as dag:
    ingest_task = PythonOperator(task_id="ingest_data", python_callable=ingest_data)