import inspect
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer, ConfigUnion, LoRAConfig

#parameters
MODEL_NAME="distilbert-base-uncased"
TRAIN_N=2000
TEST_N=1000
MAX_LEN=256
SEED=42

BATCH = 8
EPOCHS=1

LR_NO_FINETUNE=5e-4
LR_LORA=1e-4


#load dataset
set_seed(SEED)
ds=load_dataset("stanfordnlp/imdb")

train_ds=ds["train"].shuffle(seed=SEED).select(range(min(TRAIN_N,len(ds["train"]))))
test_ds=ds["test"].shuffle(seed=SEED).select(range(min(TEST_N,len(ds["test"]))))


#text segmentation
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

tokenized_train=train_ds.map(tokenize, batched=True, remove_columns=["text"])
tokenized_test=test_ds.map(tokenize, batched=True, remove_columns=["text"])


#DataCollator
collator=DataCollatorWithPadding(tokenizer=tokenizer)


#evaluate
acc=evaluate.load("accuracy")
f1=evaluate.load("f1")
prec=evaluate.load("precision")
rec=evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels=eval_pred
    preds=np.argmax(logits, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"],
        "recall": rec.compute(predictions=preds, references=labels, average="binary")["recall"],
        "precision": prec.compute(predictions=preds, references=labels, average="binary")["precision"],
    }


#step1: training loop
#model
model_no_fine_tune=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

for name, p in model_no_fine_tune.named_parameters():
    if ("pre_classifier" in name) or ("classifier" in name):
        p.requires_grad = True
    else:
        p.requires_grad = False

training_args_no_fine_tune=TrainingArguments(
    output_dir="./out_no_finetune",
    learning_rate=LR_NO_FINETUNE,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=50,

)

trainer=Trainer(
    model=model_no_fine_tune,
    args=training_args_no_fine_tune,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

#training & evaluation
print("\n=== Step1: No-Finetune ===")
trainer.train()
metrics=trainer.evaluate()
print("Final metrics:", metrics)


#step2: add LoRA into pre_training model
model_lora=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

config={
    "rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_lin", "k_lin", "v_lin", "out_lin"],
}

lora_config=LoraConfig(
    r=config["rank"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    target_modules=config["target_modules"],
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["classifier", "pre_classifier"],
)

model_LORA=get_peft_model(model_lora, lora_config)

training_args_lora= TrainingArguments(
    output_dir="./out_lora",
    learning_rate=LR_LORA,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=50,

)

trainer=Trainer(
    model=model_LORA,
    args=training_args_lora,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

print("\n=== Step2: LORA ===")
trainer.train()
metrics=trainer.evaluate()
print("Final metrics:", metrics)

#step3: add Adapter
model_adapter=AutoAdapterModel.from_pretrained(MODEL_NAME)
model_adapter.add_classification_head("imdb", num_labels=2)
model_adapter.active_head = "imdb"

adapter_config=AdapterConfig.load(
    "pfeiffer",
    reduction_factor=16,
    non_linearity="relu",
)

lora_cfg=LoRAConfig(
    r=8,
    alpha=16,
    dropout=0.05,
)

mixed_cfg=ConfigUnion(adapter_config, lora_cfg)

adapter_name="lora_adapter"
model_adapter.add_adapter(adapter_name, config=mixed_cfg)
model_adapter.train_adapter(adapter_name)
model_adapter.set_active_adapters(adapter_name)

training_args_adapt=TrainingArguments(
    output_dir="./out_adapt",
    learning_rate=1e-4,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=50,

)

trainer=AdapterTrainer(
    model=model_adapter,
    args=training_args_adapt,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

print("\n=== Step3: LoRA+Adapter ===")
trainer.train()
metrics=trainer.evaluate()
print("Final metrics:", metrics)