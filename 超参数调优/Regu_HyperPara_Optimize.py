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

    print("\n==============================")
    print("Running experiment:", exp_cfg["tag"])
    print(f"Scheduler: {exp_cfg['lr_scheduler_type']} | warmup_ratio={exp_cfg['warmup_ratio']}")
    eff_batch = exp_cfg["batch_size"] * exp_cfg["gradient_accumulation_steps"]
    print(f"Batch: {exp_cfg['batch_size']} | accum: {exp_cfg['gradient_accumulation_steps']} | effective_batch: {eff_batch}")
    print(f"LR scale rule: {exp_cfg.get('lr_scale_rule', 'none')} | learning_rate: {exp_cfg['learning_rate']}")
    print("==============================")

    trainer.train()
    metrics = trainer.evaluate()
    return metrics

BEST_SCHEDULER = "cosine"
BEST_WARMUP_RATIO = 0.06

BEST_BATCH_SIZE = BASE_BATCH
BEST_GRAD_ACCUM = 8
LR_SCALE_RULE = "linear"

BASE_EXP_COMMON = {
    "epochs": EPOCHS,

    "lr_scheduler_type": BEST_SCHEDULER,
    "warmup_ratio": BEST_WARMUP_RATIO,

    "batch_size": BEST_BATCH_SIZE,
    "gradient_accumulation_steps": BEST_GRAD_ACCUM,
    "max_grad_norm": 1.0,

    "fp16": torch.cuda.is_available(),
    "optim": "adamw_torch",

    # LoRA+Adapter config (keep same)
    "lora_r": 8,
    "lora_alpha": 16,
    "reduction_factor": 16,
    "non_linearity": "relu",

    "lr_scale_rule": LR_SCALE_RULE,
}

BASE_LR_SCALED = scale_lr(BASE_LR, BASE_BATCH, BEST_BATCH_SIZE, BEST_GRAD_ACCUM, LR_SCALE_RULE)

#weight decay
WEIGHT_DECAYS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
FIXED_LORA_DROPOUT = 0.05



wd_experiments = []
for wd in WEIGHT_DECAYS:
    tag = f"wd_{wd:g}".replace(".", "p")
    wd_experiments.append(
        {
            **BASE_EXP_COMMON,
            "tag": tag,
            "output_dir": f"./runs/regularization/{tag}",
            "learning_rate": BASE_LR_SCALED,
            "weight_decay": wd,
            "lora_dropout": FIXED_LORA_DROPOUT,
            **BASE_EXP_COMMON,
        }
    )

wd_results = []
for exp in wd_experiments:
    metrics = run_experiment(exp)
    print("Final metrics:", metrics)
    wd_results.append({"tag": exp["tag"], "weight_decay": exp["weight_decay"], "lora_dropout": exp["lora_dropout"], **metrics})

#best_wd=0
#dropout
best_wd = 0.05  # TODO: 跑完 reg_wd_* 后，把这里替换成你最优的 weight_decay

drop_list = [0.0, 0.05, 0.1, 0.2]
drop_experiments = []
for dp in drop_list:
    tag = f"loradrop_{dp:g}".replace(".", "p")
    drop_experiments.append(
        {
            "tag": tag,
            "output_dir": f"./runs/regularization/{tag}_wd0",
            "learning_rate": BASE_LR_SCALED,
            "weight_decay": best_wd,
            "lora_dropout": dp,
            **BASE_EXP_COMMON,
        }
    )

drop_results = []
for exp in drop_experiments:
    metrics = run_experiment(exp)
    print("Final metrics:", metrics)
    drop_results.append({
        "tag": exp["tag"],
        "weight_decay": exp["weight_decay"],
        "lora_dropout": exp["lora_dropout"],
        **metrics
    })


print("\n===== Regularization (Weight Decay) Results =====")
print(f"Fixed lora_dropout = {FIXED_LORA_DROPOUT} | base_lr_scaled = {BASE_LR_SCALED}")
print(f"{'Config':12s} | {'WD':7s} | {'Acc':6s} | {'F1':6s} | {'Loss':8s}")
print("-" * 70)
for r in wd_results:
    print(
        f"{r['tag']:12s} | "
        f"{r['weight_decay']:<7g} | "
        f"{r.get('eval_accuracy', 0):.4f} | "
        f"{r.get('eval_f1', 0):.4f} | "
        f"{r.get('eval_loss', 0):.4f}"
    )
print("-" * 70)


print("\n===== Regularization (LoRA Dropout) Results =====")
print(f"Fixed weight_decay = {best_wd} | base_lr_scaled = {BASE_LR_SCALED}")
print(f"{'Config':14s} | {'Dropout':8s} | {'Acc':6s} | {'F1':6s} | {'Loss':8s}")
print("-" * 80)

for r in drop_results:
    print(
        f"{r['tag']:14s} | "
        f"{r['lora_dropout']:<8g} | "
        f"{r.get('eval_accuracy', 0):.4f} | "
        f"{r.get('eval_f1', 0):.4f} | "
        f"{r.get('eval_loss', 0):.4f}"
    )

print("-" * 80)
