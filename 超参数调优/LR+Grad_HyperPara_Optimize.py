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

#gradient optimzation
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

#experiment
experiments=[
    #constant learning rate
    {
        "tag": "sched_constant",
        "output_dir": "./runs/sched_constant",
        "learning_rate": BASE_LR,
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,

        "gradient_accumulation_steps": 1,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",
    },

    #linear
    {
        "tag": "sched_linear",
        "output_dir": "./runs/sched_linear",
        "learning_rate": BASE_LR,
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 1,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",
    },

    #cosine
    {
        "tag": "sched_cosine",
        "output_dir": "./runs/sched_cosine",
        "learning_rate": BASE_LR,
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 1,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",
    },


]

#gradient experiments
experiments+=[
    # accum=4, 不缩放学习率（eff batch = 8*4=32）
    {
        "tag": "gradacc_4_none",
        "output_dir": "./runs/gradacc_4_none",
        "learning_rate": scale_lr(BASE_LR, BASE_BATCH, BASE_BATCH, 4, "none"),
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",

        "lr_scale_rule": "none",
    },

    # accum=4, 线性缩放学习率（lr *= 32/8 = 4）
    {
        "tag": "gradacc_4_linear",
        "output_dir": "./runs/gradacc_4_linear",
        "learning_rate": scale_lr(BASE_LR, BASE_BATCH, BASE_BATCH, 4, "linear"),
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",

        "lr_scale_rule": "linear",
    },

    # accum=4, 平方根缩放学习率（lr *= sqrt(32/8)=2）
    {
        "tag": "gradacc_4_sqrt",
        "output_dir": "./runs/gradacc_4_sqrt",
        "learning_rate": scale_lr(BASE_LR, BASE_BATCH, BASE_BATCH, 4, "sqrt"),
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",

        "lr_scale_rule": "sqrt",
    },

    # accum=8, 不缩放学习率（eff batch = 8*8=64）
    {
        "tag": "gradacc_8_none",
        "output_dir": "./runs/gradacc_8_none",
        "learning_rate": scale_lr(BASE_LR, BASE_BATCH, BASE_BATCH, 8, "none"),
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",

        "lr_scale_rule": "none",
    },

    # accum=8, 线性缩放学习率（lr *= 64/8 = 8）
    {
        "tag": "gradacc_8_linear",
        "output_dir": "./runs/gradacc_8_linear",
        "learning_rate": scale_lr(BASE_LR, BASE_BATCH, BASE_BATCH, 8, "linear"),
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",

        "lr_scale_rule": "linear",
    },

    # accum=8, 平方根缩放学习率（lr *= sqrt(64/8)=sqrt(8)=2.828）
    {
        "tag": "gradacc_8_sqrt",
        "output_dir": "./runs/gradacc_8_sqrt",
        "learning_rate": scale_lr(BASE_LR, BASE_BATCH, BASE_BATCH, 8, "sqrt"),
        "batch_size": BASE_BATCH,
        "epochs": EPOCHS,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "optim": "adamw_torch",

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",

        "lr_scale_rule": "sqrt",
    },
]

#regularization experiments
REG_BASE={
    "output_dir": "./runs/reg_base",
    "learning_rate": scale_lr(BASE_LR, BASE_BATCH, BASE_BATCH, 8, "linear"),
    "batch_size": BASE_BATCH,
    "epochs": EPOCHS,

    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.06,

    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,

    "weight_decay": 0.01,
    "fp16": torch.cuda.is_available(),
    "optim": "adamw_torch",

    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "reduction_factor": 16,
    "non_linearity": "relu",

    "lr_scale_rule": "linear",
}



all_results=[]
for exp in experiments:
    metrics = run_experiment(exp)
    print("Final metrics:", metrics)
    all_results.append({
        "tag": exp["tag"],
        "weight_decay": exp.get("weight_decay"),
        "lora_dropout": exp.get("lora_dropout"),
        "grad_accum": exp.get("gradient_accumulation_steps"),
        "learning_rate": exp.get("learning_rate"),
        **metrics
    })

print("\n===== Learning Rate Scheduler Optimization Results =====")
print(f"{'Scheduler':18s} | {'Acc':6s} | {'F1':6s} | {'Loss':8s}")
print("-" * 60)

for r in all_results:
    if r["tag"].startswith("sched_"):
        print(
            f"{r['tag']:18s} | "
            f"{r.get('eval_accuracy', 0):.4f} | "
            f"{r.get('eval_f1', 0):.4f} | "
            f"{r.get('eval_loss', 0):.4f}"
        )


print("\n===== Gradient Accumulation Optimization Results =====")
print(f"{'Config':20s} | {'Acc':6s} | {'F1':6s} | {'Loss':8s}")
print("-" * 65)

for r in all_results:
    if r["tag"].startswith("gradacc_"):
        print(
            f"{r['tag']:20s} | "
            f"{r.get('eval_accuracy', 0):.4f} | "
            f"{r.get('eval_f1', 0):.4f} | "
            f"{r.get('eval_loss', 0):.4f}"
        )









