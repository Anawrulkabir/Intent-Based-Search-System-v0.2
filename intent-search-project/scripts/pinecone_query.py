import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

def query_pinecone(query):
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("tech-products")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode(query).tolist()

    result = index.query(
        vector=query_vec,
        top_k=5,
        include_metadata=True
    )

    for match in result["matches"]:
        print(f"Score: {match['score']}")
        print(f"Title: {match['metadata'].get('title')}")
        print(f"Price: {match['metadata'].get('price')}")
        print("-" * 30)
    return result