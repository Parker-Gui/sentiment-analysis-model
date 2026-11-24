import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer, ConfigUnion, LoRAConfig
import math

#parameters
MODEL_NAME="distilbert-base-uncased"
TRAIN_N=2000
TEST_N=1000
MAX_LEN=256
SEED=42

BATCH = 8
EPOCHS=3

BASE_LR=1e-4
BASE_BATCH=8



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

#lora+adapter model->baseline
def build_lora_adapter_model(lora_r=8, lora_alpha=16, lora_dropout=0.05,
                             reduction_factor=16, non_linearity="relu"):
    model_adapter = AutoAdapterModel.from_pretrained(MODEL_NAME)
    model_adapter.add_classification_head("imdb", num_labels=2)
    model_adapter.active_head = "imdb"

    adapter_config = AdapterConfig.load(
        "pfeiffer",
        reduction_factor=reduction_factor,
        non_linearity=non_linearity,
    )

    lora_cfg = LoRAConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    mixed_cfg = ConfigUnion(adapter_config, lora_cfg)

    adapter_name = "lora_adapter"
    model_adapter.add_adapter(adapter_name, config=mixed_cfg)
    model_adapter.train_adapter(adapter_name)
    model_adapter.set_active_adapters(adapter_name)

    return model_adapter


def scale_lr(base_lr: float, base_batch: int, batch_size: int, grad_accum: int, rule: str = "none") -> float:
    eff_batch = batch_size * grad_accum
    ratio = eff_batch / base_batch

    if rule == "none":
        return base_lr
    elif rule == "linear":
        return base_lr * ratio
    elif rule == "sqrt":
        return base_lr * math.sqrt(ratio)
    else:
        raise ValueError(f"Unknown lr scaling rule: {rule}")

def run_experiment(exp_cfg:dict):
    set_seed(SEED)
    model=build_lora_adapter_model(
        lora_r=exp_cfg["lora_r"],
        lora_alpha=exp_cfg["lora_alpha"],
        lora_dropout=exp_cfg["lora_dropout"],
        reduction_factor=exp_cfg["reduction_factor"],
        non_linearity=exp_cfg["non_linearity"],
    )

    args=TrainingArguments(
        output_dir=exp_cfg["output_dir"],
        learning_rate=exp_cfg["learning_rate"],
        per_device_train_batch_size=exp_cfg["batch_size"],
        per_device_eval_batch_size=exp_cfg["batch_size"],
        num_train_epochs=exp_cfg["epochs"],

        #learing rate
        lr_scheduler_type=exp_cfg["lr_scheduler_type"],
        warmup_ratio=exp_cfg["warmup_ratio"],

        #gradient accumulation
        gradient_accumulation_steps=exp_cfg["gradient_accumulation_steps"],
        max_grad_norm=exp_cfg.get("max_grad_norm", 1.0),

        #regulation
        weight_decay=exp_cfg["weight_decay"],

        #混合精度
        fp16=exp_cfg["fp16"],

        #optimizer
        optim=exp_cfg["optim"],

        #eval strategy
        eval_strategy="epoch",
        report_to="none",
        logging_steps=50,

        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )

    trainer = AdapterTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics

BEST_WD = 0.05
BEST_LORA_DROPOUT = 0.05


BEST_SCHEDULER = "cosine"
BEST_WARMUP_RATIO = 0.06
BEST_GRAD_ACCUM = 8


BATCH_LIST = [8, 16, 32, 64]


LR_RULES = ["linear", "sqrt"]


BASE_EXP_COMMON = {
    "epochs": EPOCHS,

    "lr_scheduler_type": BEST_SCHEDULER,
    "warmup_ratio": BEST_WARMUP_RATIO,

    "gradient_accumulation_steps": BEST_GRAD_ACCUM,
    "max_grad_norm": 1.0,

    "weight_decay": BEST_WD,
    "fp16": torch.cuda.is_available(),
    "optim": "adamw_torch",


    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": BEST_LORA_DROPOUT,
    "reduction_factor": 16,
    "non_linearity": "relu",
}


batch_lr_experiments = []
for rule in LR_RULES:
    for bs in BATCH_LIST:
        lr_scaled = scale_lr(
            base_lr=BASE_LR,
            base_batch=BASE_BATCH,
            batch_size=bs,
            grad_accum=BEST_GRAD_ACCUM,
            rule=rule,
        )
        tag = f"bs{bs}_{rule}"
        batch_lr_experiments.append({
            **BASE_EXP_COMMON,
            "tag": tag,
            "output_dir": f"./runs/batch_lr/{tag}",
            "batch_size": bs,
            "learning_rate": lr_scaled,
            "lr_scale_rule": rule,
        })


batch_lr_results = []
for exp in batch_lr_experiments:
    metrics = run_experiment(exp)
    print("Final metrics:", metrics)

    eff_batch = exp["batch_size"] * exp["gradient_accumulation_steps"]
    batch_lr_results.append({
        "tag": exp["tag"],
        "rule": exp["lr_scale_rule"],
        "batch_size": exp["batch_size"],
        "grad_accum": exp["gradient_accumulation_steps"],
        "effective_batch": eff_batch,
        "learning_rate": exp["learning_rate"],
        **metrics
    })


print("\n===== Batch Size & LR Scaling Results =====")
print(f"Fixed: scheduler={BEST_SCHEDULER} | warmup_ratio={BEST_WARMUP_RATIO} | grad_accum={BEST_GRAD_ACCUM}")
print(f"Fixed: weight_decay={BEST_WD} | lora_dropout={BEST_LORA_DROPOUT} | base_lr={BASE_LR} | base_batch={BASE_BATCH}")
print(f"{'Config':14s} | {'Rule':6s} | {'BS':4s} | {'EffBS':6s} | {'LR':10s} | {'Acc':6s} | {'F1':6s} | {'Loss':8s}")
print("-" * 110)

for r in batch_lr_results:
    print(
        f"{r['tag']:14s} | "
        f"{r['rule']:6s} | "
        f"{r['batch_size']:<4d} | "
        f"{r['effective_batch']:<6d} | "
        f"{r['learning_rate']:<10.6g} | "
        f"{r.get('eval_accuracy', 0):.4f} | "
        f"{r.get('eval_f1', 0):.4f} | "
        f"{r.get('eval_loss', 0):.4f}"
    )

print("-" * 110)