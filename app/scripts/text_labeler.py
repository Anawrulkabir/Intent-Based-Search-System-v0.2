import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
import torch
from huggingface_hub import login
from seqeval.metrics import classification_report, f1_score
import mlflow
import mlflow.transformers

def train_ner_model():
    """
    Train a Named Entity Recognition model for tech product entities.
    
    Returns:
        tuple: (model_path, metrics) - Path to the saved model and training metrics
    """
    # Load dataset
    dataset = load_dataset("roundspecs/tech-product-ner")

    # Prepare label mappings
    label_set = set(tag for tags in dataset["train"]["ner_tags"] for tag in tags)
    label_list = sorted(list(label_set))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)

    # Encode labels
    def encode_tags(example):
        example["labels"] = [label2id[tag] for tag in example["ner_tags"]]
        return example

    encoded_dataset = dataset.map(encode_tags)

    # Tokenize and align labels
    model_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            max_length=128
        )

        all_labels = []
        for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples["tokens"]))):
            labels = []
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(-100)
                else:
                    labels.append(label2id[examples["ner_tags"][i][word_idx]])
            all_labels.append(labels)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    tokenized_dataset = encoded_dataset.map(tokenize_and_align_labels, batched=True)

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/bert-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        push_to_hub=False,
        report_to="mlflow"
    )

    # Compute metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(-1)
        true_labels = [
            [id2label[label] for label in label_row if label != -100]
            for label_row in labels
        ]
        true_preds = [
            [id2label[pred] for pred, label in zip(pred_row, label_row) if label != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]
        
        f1 = f1_score(true_labels, true_preds)
        report = classification_report(true_labels, true_preds)
        
        return {
            "f1": f1,
            "report": report
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()
    
    # Save the model
    model_save_path = "./models/bert-ner-final"
    trainer.save_model(model_save_path)
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    
    return model_save_path, eval_results

def extract_entities(query):
    """
    Extract entities from a query using the NER model.
    
    Args:
        query (str): The input query
        
    Returns:
        dict: Extracted entities categorized by type
    """
    # Load the model
    model_path = "./models/bert-ner-final"
    
    # If model doesn't exist, use a pre-trained model
    if not os.path.exists(model_path):
        model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create NER pipeline
    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    # Get entities
    entities = pipe(query)
    
    # Clean and organize entities
    return clean_entities(entities)

def clean_entities(entities):
    """
    Clean and organize extracted entities.
    
    Args:
        entities (list): List of extracted entities
        
    Returns:
        dict: Organized entities by category
    """
    result = {
        "CATEGORY": [],
        "PRICE_MAX": "",
        "PRICE_MIN": "",
        "BRAND": [],
        "COLOR": [],
        "REMAINDER": ""
    }

    colors = ["red", "green", "blue", "yellow", "black", "white", "pink", "gray", "brown"]

    for entity in entities:
        label = entity["entity_group"]
        word = entity["word"]
        if label == "CATEGORY":
            result["CATEGORY"].append(word)
        elif label == "PRICE_MAX":
            result["PRICE_MAX"] += "".join(c for c in word if c.isdigit())
        elif label == "PRICE_MIN":
            result["PRICE_MIN"] += "".join(c for c in word if c.isdigit())
        elif label == "BRAND":
            result["BRAND"].append(word)
        elif label == "COLOR" and word.lower() in colors:
            result["COLOR"].append(word)
        else:
            result["REMAINDER"] += word + " "

    return result

if __name__ == "__main__":
    query = "Samsung smartphones between 10000 and 20000 BDT in red color"
    entities = extract_entities(query)
    print(entities)
