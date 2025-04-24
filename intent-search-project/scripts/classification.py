import mlflow
import mlflow.pytorch
from transformers import pipeline

mlflow.set_experiment("intent_classification")

with mlflow.start_run():
    # Load the pre-trained model
    pipe = pipeline("text-classification", model="roundspecs/minilm-finetuned-intent-classification")
    
    # Log model parameters
    mlflow.log_param("model_name", "roundspecs/minilm-finetuned-intent-classification")
    
    # Example input
    text = "gaming pc under 50000"
    result = pipe(text)
    
    # Log result as a metric or artifact
    mlflow.log_metric("classification_score", result[0]["score"])
    mlflow.log_artifact("classification_result.txt")
    
    # Save model
    mlflow.pytorch.log_model(pipe, "intent_classifier")
    
    print(result)


def classify_intent(text):
    mlflow.set_experiment("intent_classification")
    with mlflow.start_run():
        pipe = pipeline("text-classification", model="roundspecs/minilm-finetuned-intent-classification")
        mlflow.log_param("model_name", "roundspecs/minilm-finetuned-intent-classification")
        result = pipe(text)
        mlflow.log_metric("classification_score", result[0]["score"])
        mlflow.pytorch.log_model(pipe, "intent_classifier")
        return result