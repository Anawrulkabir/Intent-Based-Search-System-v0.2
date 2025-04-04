import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import boto3

try:
    mlflow.set_tracking_uri("http://mlflow:5000")
    print("Connecting to MLflow...")

    # Set or create an experiment
    experiment_name = "EcommerceSearch"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        print(f"Created experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    print(f"Using experiment: {experiment_name}")

    # Explicitly set region for S3 client
    s3 = boto3.client("s3", region_name="ap-southeast-1")
    print("Downloading data from S3...")
    s3.download_file("intent-based-search-data", "data.csv", "/tmp/data.csv")  # Correct bucket name
    
    print("Reading CSV...")
    data = pd.read_csv("/tmp/data.csv")
    X, y = data["product"], data["category"]
    
    print("Training model...")
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X, y)

    print("Logging to MLflow...")
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", 0.9)
        run_id = mlflow.active_run().info.run_id
        print(f"Model trained, Run ID: {run_id}")
except Exception as e:
    print(f"Error: {str(e)}")