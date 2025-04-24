import os
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

def index_data():
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    df = pd.read_csv("/app/data/startech_products_small.csv")

    def get_text(x):
        if pd.isna(x.Color):
            return f"{x.title} {x.category} {x.brand}. {x.short_description}"
        return f"{x.title} {x.category} {x.brand} {x.Color}. {x.short_description}"

    df["text"] = df.apply(get_text, axis=1)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df["embedding"] = df["text"].apply(lambda x: model.encode(x, convert_to_tensor=False).tolist())

    index_name = "tech-products-v2"
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    index = pc.Index(index_name)
    to_upsert = [
        (str(i), emb, {"title": df.iloc[i]["title"], "price": str(df.iloc[i]["price"])})
        for i, emb in enumerate(df["embedding"])
    ]

    batch_size = 100
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i:i+batch_size]
        index.upsert(batch)

    print("✅ Data indexed into Pinecone successfully.")