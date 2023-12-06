import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

"""
bert-base-cased
bert-large-cased
"""

MODEL_NAME = "bert-large-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)


training_args = TrainingArguments(output_dir="test_trainer")


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()