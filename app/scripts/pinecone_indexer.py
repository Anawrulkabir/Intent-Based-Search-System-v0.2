import os
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import mlflow

def index_data(data_path="data/startech_products_small.csv", index_name="tech-products-v2"):
    """
    Index product data into Pinecone.
    
    Args:
        data_path (str): Path to the CSV file with product data
        index_name (str): Name of the Pinecone index
        
    Returns:
        int: Number of records indexed
    """
    # Initialize Pinecone client
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Load CSV
    df = pd.read_csv(data_path)

    # Log data stats to MLflow
    mlflow.log_metric("num_products", len(df))
    
    # Use 'title' column for text input
    # Define a function to generate text based on the row data
    def get_text(x):
        if pd.isna(x.Color):
            return f"{x.title} {x.category} {x.brand}. {x.short_description}"
        return f"{x.title} {x.category} {x.brand} {x.Color}. {x.short_description}"

    # Apply the function to create a new 'text' column
    df["text"] = df.apply(get_text, axis=1)

    # Generate embeddings using the 'text' column
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Save the model locally for MLflow tracking
    model_path = "./models/sentence-transformer"
    model.save(model_path)
    mlflow.log_artifact(model_path, "sentence_transformer_model")
    
    df["embedding"] = df["text"].apply(lambda x: model.encode(x, convert_to_tensor=False).tolist())

    # Create index if not exists
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Wait until index is ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    # Upsert to Pinecone
    index = pc.Index(index_name)
    to_upsert = [
        (str(i), emb, {
            "title": df.iloc[i]["title"],
            "price": str(df.iloc[i]["price"])
        })
        for i, emb in enumerate(df["embedding"])
    ]

    batch_size = 100
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i:i+batch_size]
        index.upsert(batch)

    print(f"✅ {len(df)} records indexed into Pinecone successfully.")
    return len(df)

if __name__ == "__main__":
    index_data()
