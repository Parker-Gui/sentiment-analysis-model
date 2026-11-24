import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time
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


MODEL_NAME = "distilbert-base-uncased"
SEED = 42


TRAIN_N = 2000
TEST_N = 1000
MAX_LEN = 256
EPOCHS = 3


BASE_LR = 1e-4
BASE_BATCH = 8


MAX_STEPS = None


set_seed(SEED)
ds = load_dataset("stanfordnlp/imdb")

train_ds = ds["train"].shuffle(seed=SEED).select(range(min(TRAIN_N, len(ds["train"]))))
test_ds = ds["test"].shuffle(seed=SEED).select(range(min(TEST_N, len(ds["test"]))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)


acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")
prec = evaluate.load("precision")
rec = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"],
        "recall": rec.compute(predictions=preds, references=labels, average="binary")["recall"],
        "precision": prec.compute(predictions=preds, references=labels, average="binary")["precision"],
    }


def build_lora_adapter_model(
    lora_r=8, lora_alpha=16, lora_dropout=0.05,
    reduction_factor=16, non_linearity="relu"
):
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


def final_run():
    set_seed(SEED)


    final_cfg = {
        "tag": "FINAL_best_combo",

        # training
        "epochs": EPOCHS,
        "batch_size": 8,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,

        # scheduler & warmup
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,

        # optimizer
        "optim": "adamw_torch",

        # regularization
        "weight_decay": 0.05,

        # mixed precision
        "fp16": bool(torch.cuda.is_available()),

        # LoRA+Adapter
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "reduction_factor": 16,
        "non_linearity": "relu",

        # lr scaling rule
        "lr_scale_rule": "linear",
    }

    final_cfg["learning_rate"] = scale_lr(
        BASE_LR, BASE_BATCH,
        final_cfg["batch_size"],
        final_cfg["gradient_accumulation_steps"],
        final_cfg["lr_scale_rule"],
    )

    final_cfg["output_dir"] = "./runs/final_run_best"

    model = build_lora_adapter_model(
        lora_r=final_cfg["lora_r"],
        lora_alpha=final_cfg["lora_alpha"],
        lora_dropout=final_cfg["lora_dropout"],
        reduction_factor=final_cfg["reduction_factor"],
        non_linearity=final_cfg["non_linearity"],
    )


    args = TrainingArguments(
        output_dir=final_cfg["output_dir"],
        learning_rate=final_cfg["learning_rate"],
        per_device_train_batch_size=final_cfg["batch_size"],
        per_device_eval_batch_size=final_cfg["batch_size"],
        num_train_epochs=final_cfg["epochs"],

        lr_scheduler_type=final_cfg["lr_scheduler_type"],
        warmup_ratio=final_cfg["warmup_ratio"],

        gradient_accumulation_steps=final_cfg["gradient_accumulation_steps"],
        max_grad_norm=final_cfg["max_grad_norm"],

        weight_decay=final_cfg["weight_decay"],
        fp16=final_cfg["fp16"],
        optim=final_cfg["optim"],


        eval_strategy="epoch",


        max_steps=MAX_STEPS if MAX_STEPS is not None else -1,

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


    eff_batch = final_cfg["batch_size"] * final_cfg["gradient_accumulation_steps"]
    print("\n==============================")
    print("Running FINAL experiment:", final_cfg["tag"])
    print(f"Scheduler: {final_cfg['lr_scheduler_type']} | warmup_ratio={final_cfg['warmup_ratio']}")
    print(f"Batch: {final_cfg['batch_size']} | accum: {final_cfg['gradient_accumulation_steps']} | effective_batch: {eff_batch}")
    print(f"LR scale rule: {final_cfg['lr_scale_rule']} | base_lr={BASE_LR} -> lr={final_cfg['learning_rate']}")
    print(f"Weight decay: {final_cfg['weight_decay']} | LoRA dropout: {final_cfg['lora_dropout']}")
    print(f"Optim: {final_cfg['optim']} | fp16: {final_cfg['fp16']} | MAX_STEPS: {MAX_STEPS}")
    print("==============================\n")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    metrics = trainer.evaluate()

    print("\n===== FINAL RUN RESULT =====")
    print("TrainTime(s):", round(train_time, 2))
    print("Final metrics:", metrics)

    return metrics

if __name__ == "__main__":
    final_run()
