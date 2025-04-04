FROM apache/airflow:2.10.5-python3.11

WORKDIR /opt/airflow
COPY app/requirements.txt .
RUN pip install -r requirements.txt
COPY app/dags /opt/airflow/dags