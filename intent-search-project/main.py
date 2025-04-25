from fastapi import FastAPI
from scripts.classification import classify_intent
from scripts.labelling import clean_entities
from scripts.pinecone_query import query_pinecone
from transformers import pipeline

app = FastAPI()

print('hjelll')

@app.get("/search")
async def search(query: str):
    # Classify intent
    classification = classify_intent(query)
    if classification[0]["label"] == "keyword":
        # Implement fuzzy search (not provided in your code)
        return {"message": "Fuzzy search not implemented"}
    else:
        # NER
        ner_pipe = pipeline("ner", model="/app/models/ner_model", tokenizer="/app/models/ner_model")
        entities = ner_pipe(query)
        cleaned_entities = clean_entities(entities)
        # Pinecone query
        result = query_pinecone(query)
        return {"entities": cleaned_entities, "search_results": result}