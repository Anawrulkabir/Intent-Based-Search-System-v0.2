from fastapi import FastAPI
import mlflow.sklearn
from prometheus_client import Counter, start_http_server
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

app = FastAPI()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5001")

# Replace with actual run_id after training
run_id = "<run_id>"  # Update this after running train.py

try:
    model = mlflow.sklearn.load_model("runs:/{run_id}/model")  # Replace after training
except:
    print("Training dummy model...")
    X = ["red shoes", "blue shirt", "green hat"]
    y = ["footwear", "clothing", "accessories"]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X, y)

requests_counter = Counter("api_requests_total", "Total API requests")

@app.get("/predict")
def predict(query: str):
    requests_counter.inc()
    prediction = model.predict([query])[0]
    return {"query": query, "category": prediction}

if __name__ == "__main__":
    start_http_server(8001)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)