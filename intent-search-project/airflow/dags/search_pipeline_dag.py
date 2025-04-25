# from datetime import datetime
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# import sys
# import os

# # Add scripts directory to path
# sys.path.append('/opt/airflow/scripts')

# from classification import classify_intent
# from labelling import train_ner_model, clean_entities
# from pinecone_indexing import index_data
# from pinecone_query import query_pinecone

# # Define tasks
# def run_classification():
#     text = "gaming pc under 50000"
#     result = classify_intent(text)

#     print(f"Classification result: {result}")

# def run_ner_training():
#     train_ner_model()

# def run_pinecone_indexing():
#     index_data()

# def run_pinecone_query():
#     query = "Samsung smartwatch space gray color"
#     query_pinecone(query)

# # Define DAG
# with DAG(
#     'search_pipeline',
#     start_date=datetime(2025, 4, 24),
#     schedule_interval='@daily',
#     catchup=False
# ) as dag:
#     t1 = PythonOperator(
#         task_id='classify_intent',
#         python_callable=run_classification
#     )
#     t2 = PythonOperator(
#         task_id='train_ner_model',
#         python_callable=run_ner_training
#     )
#     t3 = PythonOperator(
#         task_id='index_pinecone',
#         python_callable=run_pinecone_indexing
#     )
#     t4 = PythonOperator(
#         task_id='query_pinecone',
#         python_callable=run_pinecone_query
#     )

#     t1 >> t2 >> t3 >> t4


from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import sys
import mlflow
import mlflow.sklearn
import mlflow.transformers

# Add the project directory to the path
sys.path.append('/opt/airflow/scripts')

# Import the scripts
from scripts.text_classifier import classify_intent
from scripts.text_labeler import train_ner_model, extract_entities
from scripts.pinecone_indexer import index_data
from scripts.pinecone_querier import query_index

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'intent_search_pipeline',
    default_args=default_args,
    description='Intent-based search pipeline with NER and vector search',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['intent-search', 'nlp', 'vector-db'],
)

# Task 1: Intent Classification
def run_intent_classification(**kwargs):
    with mlflow.start_run(run_name="intent_classification") as run:
        query = kwargs.get('query', "gaming pc under 50000")
        result = classify_intent(query)
        
        # Log parameters and results to MLflow
        mlflow.log_param("query", query)
        mlflow.log_param("model", "roundspecs/minilm-finetuned-intent-classification")
        mlflow.log_metric("confidence", result[0]['score'])
        
        # Return the classification result
        return {
            'intent_type': result[0]['label'],
            'confidence': result[0]['score'],
            'query': query
        }

# Task 2: Train NER Model (if needed)
def run_ner_training(**kwargs):
    with mlflow.start_run(run_name="ner_training") as run:
        model_path, metrics = train_ner_model()
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "f1_score": metrics.get("f1", 0),
        })
        
        # Log the model to MLflow
        mlflow.transformers.log_model(
            transformers_model={"model": model_path},
            artifact_path="ner_model"
        )
        
        return {"model_path": model_path}

# Task 3: Entity Extraction
def run_entity_extraction(**kwargs):
    ti = kwargs['ti']
    intent_result = ti.xcom_pull(task_ids='intent_classification')
    
    with mlflow.start_run(run_name="entity_extraction") as run:
        query = intent_result['query']
        entities = extract_entities(query)
        
        # Log parameters and results
        mlflow.log_param("query", query)
        mlflow.log_dict(entities, "extracted_entities.json")
        
        return {
            "query": query,
            "entities": entities,
            "intent_type": intent_result['intent_type']
        }

# Task 4: Index Data to Pinecone
def run_indexing(**kwargs):
    with mlflow.start_run(run_name="pinecone_indexing") as run:
        index_name = "tech-products-v2"
        data_path = "data/startech_products_small.csv"
        
        # Index the data
        num_records = index_data(data_path, index_name)
        
        # Log metrics
        mlflow.log_param("index_name", index_name)
        mlflow.log_param("data_path", data_path)
        mlflow.log_metric("indexed_records", num_records)
        
        return {"index_name": index_name, "indexed_records": num_records}

# Task 5: Query Pinecone
def run_query(**kwargs):
    ti = kwargs['ti']
    entity_result = ti.xcom_pull(task_ids='entity_extraction')
    index_result = ti.xcom_pull(task_ids='indexing')
    
    with mlflow.start_run(run_name="pinecone_query") as run:
        query = entity_result['query']
        index_name = index_result['index_name']
        
        # Run the query
        results = query_index(query, index_name)
        
        # Log parameters and results
        mlflow.log_param("query", query)
        mlflow.log_param("index_name", index_name)
        mlflow.log_dict(results, "search_results.json")
        
        return {"query": query, "results": results}

# Define the tasks
intent_classification_task = PythonOperator(
    task_id='intent_classification',
    python_callable=run_intent_classification,
    op_kwargs={'query': "{{ dag_run.conf.get('query', 'gaming pc under 50000') }}"},
    dag=dag,
)

ner_training_task = PythonOperator(
    task_id='ner_training',
    python_callable=run_ner_training,
    dag=dag,
)

entity_extraction_task = PythonOperator(
    task_id='entity_extraction',
    python_callable=run_entity_extraction,
    provide_context=True,
    dag=dag,
)

indexing_task = PythonOperator(
    task_id='indexing',
    python_callable=run_indexing,
    dag=dag,
)

query_task = PythonOperator(
    task_id='query',
    python_callable=run_query,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
intent_classification_task >> entity_extraction_task
ner_training_task >> entity_extraction_task
entity_extraction_task >> query_task
indexing_task >> query_task
