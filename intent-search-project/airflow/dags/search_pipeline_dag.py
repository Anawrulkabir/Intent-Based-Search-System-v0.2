from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Add scripts directory to path
sys.path.append('/app/scripts')

from classification import classify_intent
from labelling import train_ner_model, clean_entities
from pinecone_indexing import index_data
from pinecone_query import query_pinecone

# Define tasks
def run_classification():
    text = "gaming pc under 50000"
    result = classify_intent(text)
    print(f"Classification result: {result}")

def run_ner_training():
    train_ner_model()

def run_pinecone_indexing():
    index_data()

def run_pinecone_query():
    query = "Samsung smartwatch space gray color"
    query_pinecone(query)

# Define DAG
with DAG(
    'search_pipeline',
    start_date=datetime(2025, 4, 24),
    schedule_interval='@daily',
    catchup=False
) as dag:
    t1 = PythonOperator(
        task_id='classify_intent',
        python_callable=run_classification
    )
    t2 = PythonOperator(
        task_id='train_ner_model',
        python_callable=run_ner_training
    )
    t3 = PythonOperator(
        task_id='index_pinecone',
        python_callable=run_pinecone_indexing
    )
    t4 = PythonOperator(
        task_id='query_pinecone',
        python_callable=run_pinecone_query
    )

    t1 >> t2 >> t3 >> t4