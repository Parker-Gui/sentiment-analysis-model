import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.environ["HF_HUB_ETAG_TIMEOUT"] = "200"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "200"
os.environ["HF_HUB_HTTP_TIMEOUT"] = "200"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import time
import math
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
CACHE_DIR = "./hf_cache"

SEED = 42
MAX_LEN = 256

TRAIN_N = 800
TEST_N = 400

BATCH = 4
EPOCHS = 1
MAX_STEPS = 80
LOGGING_STEPS = 20

BASE_LR = 8e-4
WARMUP_RATIO = 0.06
SCHEDULER = "cosine"

WEIGHT_DECAY = 0.05
LORA_DROPOUT = 0.05

GRAD_ACCUM = 4
MAX_GRAD_NORM = 1.0


set_seed(SEED)
ds = load_dataset("stanfordnlp/imdb", cache_dir=CACHE_DIR)

train_ds = ds["train"].shuffle(seed=SEED).select(range(min(TRAIN_N, len(ds["train"]))))
test_ds  = ds["test"].shuffle(seed=SEED).select(range(min(TEST_N, len(ds["test"]))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
tokenized_test  = test_ds.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)


acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"],
    }


def build_lora_adapter_model(lora_r=8, lora_alpha=16, lora_dropout=0.05,
                             reduction_factor=16, non_linearity="relu"):
    model_adapter = AutoAdapterModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
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


    model_adapter.gradient_checkpointing_enable()

    return model_adapter


def run_one(tag: str, fp16_flag: bool):
    set_seed(SEED)

    use_cuda = torch.cuda.is_available()
    if fp16_flag and not use_cuda:
        print(f"[SKIP] {tag}: fp16=True 但未检测到CUDA，跳过（CPU上fp16没意义/可能报错）。")
        return None

    model = build_lora_adapter_model(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=LORA_DROPOUT,
        reduction_factor=16,
        non_linearity="relu",
    )

    args = TrainingArguments(
        output_dir=f"./runs/mp_light/{tag}",
        learning_rate=BASE_LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,


        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,

        lr_scheduler_type=SCHEDULER,
        warmup_ratio=WARMUP_RATIO,

        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=MAX_GRAD_NORM,

        weight_decay=WEIGHT_DECAY,
        fp16=fp16_flag,
        optim="adamw_torch",

        eval_strategy="no",
        save_strategy="no",
        logging_steps=LOGGING_STEPS,
        report_to="none",

        dataloader_num_workers=0,
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
    print(f"Running: {tag}")
    print(f"fp16={fp16_flag} | cuda={torch.cuda.is_available()}")
    eff_bs = BATCH * GRAD_ACCUM
    print(f"batch={BATCH} | grad_accum={GRAD_ACCUM} | effective_batch={eff_bs}")
    print(f"max_steps={MAX_STEPS} | lr={BASE_LR} | wd={WEIGHT_DECAY} | lora_dropout={LORA_DROPOUT}")
    print("==============================")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    metrics = trainer.evaluate()
    metrics["train_time_sec"] = train_time
    return metrics


if __name__ == "__main__":
    results = []

    m0 = run_one("fp32", fp16_flag=False)
    if m0 is not None:
        results.append(("fp32", m0))

    m1 = run_one("fp16", fp16_flag=True)
    if m1 is not None:
        results.append(("fp16", m1))

    print("\n===== Mixed Precision Light Comparison =====")
    print(f"TRAIN_N={TRAIN_N} | TEST_N={TEST_N} | MAX_STEPS={MAX_STEPS} | BATCH={BATCH} | GRAD_ACCUM={GRAD_ACCUM}")
    print(f"{'Mode':6s} | {'Acc':6s} | {'F1':6s} | {'Loss':8s} | {'TrainTime(s)':12s}")
    print("-" * 60)
    for tag, met in results:
        print(
            f"{tag:6s} | "
            f"{met.get('eval_accuracy', 0):.4f} | "
            f"{met.get('eval_f1', 0):.4f} | "
            f"{met.get('eval_loss', 0):.4f} | "
            f"{met.get('train_time_sec', 0):.1f}"
        )
    print("-" * 60)
