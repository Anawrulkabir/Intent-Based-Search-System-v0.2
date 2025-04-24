import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import mlflow

def query_index(query, index_name="tech-products-v2", top_k=5):
    """
    Query the Pinecone index with the given query.
    
    Args:
        query (str): The search query
        index_name (str): Name of the Pinecone index
        top_k (int): Number of results to return
        
    Returns:
        list: Search results with scores and metadata
    """
    # Initialize Pinecone client
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    # Load the model
    model_path = "./models/sentence-transformer"
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Save the model locally
        model.save(model_path)
    
    # Log the query to MLflow
    mlflow.log_param("query", query)
    mlflow.log_param("top_k", top_k)
    
    # Generate query vector
    query_vec = model.encode(query).tolist()

    # Vector search
    result = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )

    # Format results
    formatted_results = []
    for match in result["matches"]:
        formatted_results.append({
            "score": match["score"],
            "title": match["metadata"].get("title"),
            "price": match["metadata"].get("price")
        })
    
    # Log the number of results
    mlflow.log_metric("num_results", len(formatted_results))
    
    return formatted_results

if __name__ == "__main__":
    query = "Samsung smartwatch space gray color"
    results = query_index(query)
    
    # Print results
    for result in results:
        print(f"Score: {result['score']}")
        print(f"Title: {result['title']}")
        print(f"Price: {result['price']}")
        print("-" * 30)
