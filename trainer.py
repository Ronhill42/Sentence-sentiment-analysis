import pickle

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, list_metrics, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# import evaluate

# print(list_metrics())
# metric = load_metric("precision")


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

def load_data_corpus():
    with open("./training_data_no_neutral.csv", "rb") as file:
        data = pd.read_csv(file,encoding = 'unicode_escape')
    
    data.reset_index(inplace=True)
    data = data.dropna()
    data["label"] = data["latest_review"]
    return data[["text", "label"]]


corpus_ds = Dataset.from_pandas(load_data_corpus())
corpus_ds = corpus_ds.cast_column("label", ClassLabel(names=["negative", "positive", "neutral"]))
#corpus_ds = corpus_ds.train_test_split(0.8, stratify_by_column="label")


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")


def tokenize_function(text):
    return tokenizer(text["text"], padding="max_length", truncation=True)


tokenized_dataset = corpus_ds.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert",
    num_labels=3,
    ignore_mismatched_sizes=True,
)
training_args = TrainingArguments(
    output_dir="./sentence_sentiment_analysis/test_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    # compute_metrics=compute_metrics,
    args=training_args,
    train_dataset=tokenized_dataset,#["train"],
    eval_dataset=tokenized_dataset,#["test"],
)
train_result = trainer.train()


metrics = train_result.metrics
trainer.log_metrics(metrics)
trainer.save_metrics(metrics)
trainer.save_model("./finbert_trained")
print("stop for interactive predictions")
