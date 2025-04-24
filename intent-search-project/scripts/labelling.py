import os
import mlflow
import mlflow.pytorch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
from seqeval.metrics import classification_report, f1_score
from dotenv import load_dotenv
from huggingface_hub import login

# Set up MLflow
mlflow.set_experiment("ner_model_training")

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)

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
    output_dir="./bert-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    push_to_hub=False,
    report_to="none"
)

# Compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)
    true_labels = [[id2label[label] for label in label_row if label != -100] for label_row in labels]
    true_preds = [[id2label[pred] for pred, label in zip(pred_row, label_row) if label != -100]
                  for pred_row, label_row in zip(predictions, labels)]
    f1 = f1_score(true_labels, true_preds)
    report = classification_report(true_labels, true_preds)
    return {"f1": f1, "report": report}

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_path", model_path)
    mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
    mlflow.log_param("learning_rate", training_args.learning_rate)

    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Log metrics
    eval_results = trainer.evaluate()
    mlflow.log_metric("f1_score", eval_results["eval_f1"])
    
    # Log model
    mlflow.pytorch.log_model(model, "ner_model")
    
    # Save classification report
    with open("classification_report.txt", "w") as f:
        f.write(eval_results["eval_report"])
    mlflow.log_artifact("classification_report.txt")

    # Inference
    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    query = "Samsung smartphones between 10000 and 20000 BDT in red color"
    entities = pipe(query)

    # Post-process
    def clean_entities(entities):
        result = {
            "CATEGORY": [], "PRICE_MAX": "", "PRICE_MIN": "", "BRAND": [], "COLOR": [], "REMAINDER": ""
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

    print(clean_entities(entities))


# (Already modified above, ensure train_ner_model and clean_entities are exposed as functions)
def train_ner_model():
    # Existing training code from labelling.py
    # Ensure MLflow logging is included
    pass

def clean_entities(entities):
    # Existing post-processing code
    pass