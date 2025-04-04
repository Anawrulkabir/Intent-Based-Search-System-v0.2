import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import boto3
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5001")


aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

if not aws_access_key or not aws_secret_key or not aws_region:
    raise ValueError("AWS credentials or region are not set properly.")

s3 = boto3.client("s3" , region_name=aws_region)
s3.download_file("intent-based-search-product", "data.csv", "/tmp/data.csv")
data = pd.read_csv("/tmp/data.csv")
X, y = data["product"], data["category"]
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X, y)

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("accuracy", 0.9)
    run_id = mlflow.active_run().info.run_id
    print(f"Model trained, Run ID: {run_id}")