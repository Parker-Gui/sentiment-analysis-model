import os

from transformers.trainer_utils import EvaluationStrategy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import evaluate
import inspect
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

#parameters
TRAIN_N=2000
TEST_N=1000
MAX_LEN=256
SEED=42
BATCH_TRAIN = 8
BATCH_EVAL = 16
GRAD_ACCUM = 2
LR=2e-5

#load dataset
ds=load_dataset("stanfordnlp/imdb")

train_ds=ds["train"].shuffle(seed=SEED).select(range(min(TRAIN_N,len(ds["train"]))))
test_ds=ds["test"].shuffle(seed=SEED).select(range(min(TEST_N,len(ds["test"]))))

#for example
sample=train_ds[0]
print("\n--- one sample ---")
print("label=", sample["label"])
print("text preview", sample["text"][:200].replace("\n",""))

#text segmentation
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

tokenized_train=train_ds.map(tokenize, batched=True, remove_columns=["text"])
tokenized_test=test_ds.map(tokenize, batched=True, remove_columns=["text"])

#show results
print("\n--- tokenized dataset ---")
print("before keys:", train_ds[0].keys()) #text, label
print("after keys:", tokenized_train[0].keys())

#model training
collator=DataCollatorWithPadding(tokenizer=tokenizer)
model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

acc=evaluate.load("accuracy")
f1=evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels=eval_pred
    preds=np.argmax(logits, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"],
    }

def make_training_args(**kwargs)->TrainingArguments:
    #old version: evaluation_strategy | new version: eval_strategy
    sig=inspect.signature(TrainingArguments.__init__)
    params=set(sig.parameters.keys())
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"]=kwargs.pop("evaluation_strategy")
    if "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"]=kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)

args=make_training_args(
    output_dir="./output",
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=50,

    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer=Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

#training & evaluation
print("\n--- Training ---")
trainer.train()
print("\n--- Evaluation ---")
metrics = trainer.evaluate()
print("\nFinal metrics:", metrics)


