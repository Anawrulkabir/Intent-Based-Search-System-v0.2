FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install mlflow
   

COPY . .
