import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    get_scheduler,
)

from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer, ConfigUnion, LoRAConfig


MODEL_NAME = "distilbert-base-uncased"
TRAIN_N = 800
TEST_N  = 400
MAX_LEN = 256
SEED = 42

EPOCHS = 3


MAX_STEPS = 80

BATCH = 4
GRAD_ACCUM = 4


WEIGHT_DECAY = 0.05
LORA_DROPOUT = 0.05


SCHEDULER = "cosine"
WARMUP_RATIO = 0.06


FP16_FLAG = torch.cuda.is_available()


set_seed(SEED)
ds = load_dataset("stanfordnlp/imdb")

train_ds = ds["train"].shuffle(seed=SEED).select(range(min(TRAIN_N, len(ds["train"]))))
test_ds  = ds["test"].shuffle(seed=SEED).select(range(min(TEST_N,  len(ds["test"]))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
tokenized_test  = test_ds.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)


acc  = evaluate.load("accuracy")
f1   = evaluate.load("f1")
prec = evaluate.load("precision")
rec  = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1":        f1.compute(predictions=preds, references=labels)["f1"],
        "recall":    rec.compute(predictions=preds, references=labels, average="binary")["recall"],
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



def build_param_groups(model, weight_decay: float):

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_params = []
    nodecay_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

def try_build_optimizer(model, opt_name: str, lr: float, weight_decay: float):

    param_groups = build_param_groups(model, weight_decay)

    opt_name = opt_name.lower()

    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)

    if opt_name == "adam":

        return torch.optim.Adam(param_groups, lr=lr)

    if opt_name == "sgd":

        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)

    if opt_name == "adafactor":

        try:
            from transformers import Adafactor
        except Exception as e:
            raise RuntimeError(f"Adafactor import failed: {e}")

        return Adafactor(
            param_groups,
            lr=lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    if opt_name == "lamb":

        try:
            from pytorch_optimizer import Lamb
            return Lamb(param_groups, lr=lr, weight_decay=weight_decay)
        except Exception as e:
            raise RuntimeError(f"LAMB not available (need extra lib). {e}")

    if opt_name in ["adam8bit", "adamw8bit", "8bitadam"]:

        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(param_groups, lr=lr)
        except Exception as e:
            raise RuntimeError(f"8-bit Adam not available (need bitsandbytes). {e}")

    raise ValueError(f"Unknown optimizer: {opt_name}")


def run_experiment(exp_cfg: dict):
    set_seed(SEED)
    model = build_lora_adapter_model(
        lora_r=exp_cfg["lora_r"],
        lora_alpha=exp_cfg["lora_alpha"],
        lora_dropout=exp_cfg["lora_dropout"],
        reduction_factor=exp_cfg["reduction_factor"],
        non_linearity=exp_cfg["non_linearity"],
    )

    # TrainingArguments
    args = TrainingArguments(
        output_dir=exp_cfg["output_dir"],
        learning_rate=exp_cfg["learning_rate"],  # 只是记录用，真正 lr 由 optimizer 控制
        per_device_train_batch_size=exp_cfg["batch_size"],
        per_device_eval_batch_size=exp_cfg["batch_size"],
        num_train_epochs=exp_cfg["epochs"],

        lr_scheduler_type=exp_cfg["lr_scheduler_type"],
        warmup_ratio=exp_cfg["warmup_ratio"],

        gradient_accumulation_steps=exp_cfg["gradient_accumulation_steps"],
        max_grad_norm=exp_cfg.get("max_grad_norm", 1.0),

        weight_decay=exp_cfg["weight_decay"],
        fp16=exp_cfg["fp16"],

        # 轻量化：限制步数（注意：max_steps > 0 会覆盖 epochs）
        max_steps=exp_cfg.get("max_steps", -1),

        # 评估/日志
        eval_strategy="steps",
        eval_steps=exp_cfg.get("eval_steps", 40),
        logging_steps=exp_cfg.get("logging_steps", 20),
        report_to="none",

        save_strategy="no",
        load_best_model_at_end=False,
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


    if args.max_steps and args.max_steps > 0:
        num_training_steps = args.max_steps
    else:
        steps_per_epoch = math.ceil(len(tokenized_train) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
        num_training_steps = steps_per_epoch * int(args.num_train_epochs)

    warmup_steps = int(num_training_steps * exp_cfg["warmup_ratio"])


    optimizer = try_build_optimizer(
        model,
        opt_name=exp_cfg["optimizer_name"],
        lr=exp_cfg["learning_rate"],
        weight_decay=exp_cfg["weight_decay"],
    )


    lr_scheduler = get_scheduler(
        name=exp_cfg["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler

    print("\n==============================")
    print("Running experiment:", exp_cfg["tag"])
    eff_batch = exp_cfg["batch_size"] * exp_cfg["gradient_accumulation_steps"]
    print(f"Optimizer: {exp_cfg['optimizer_name']} | LR={exp_cfg['learning_rate']} | WD={exp_cfg['weight_decay']}")
    print(f"fp16={exp_cfg['fp16']} | batch={exp_cfg['batch_size']} | accum={exp_cfg['gradient_accumulation_steps']} | eff_batch={eff_batch}")
    print(f"scheduler={exp_cfg['lr_scheduler_type']} | warmup_ratio={exp_cfg['warmup_ratio']} | max_steps={exp_cfg.get('max_steps', -1)}")
    print("==============================")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    metrics = trainer.evaluate()
    metrics["train_time_sec"] = train_time
    return metrics

BASE_EXP = {
    "epochs": EPOCHS,
    "max_steps": MAX_STEPS,
    "eval_steps": 40,
    "logging_steps": 20,

    "lr_scheduler_type": SCHEDULER,
    "warmup_ratio": WARMUP_RATIO,

    "batch_size": BATCH,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "max_grad_norm": 1.0,

    "weight_decay": WEIGHT_DECAY,
    "fp16": FP16_FLAG,

    # LoRA+Adapter
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": LORA_DROPOUT,
    "reduction_factor": 16,
    "non_linearity": "relu",
}


OPTIM_LR_CANDIDATES = {
    "adamw":     [8e-4],
    "adam":      [4e-4],
    "adafactor": [8e-4],
    "sgd":       [1e-2],
    "lamb":      [8e-4],
    "adam8bit":  [8e-4],
}

experiments = []
for opt_name, lr_list in OPTIM_LR_CANDIDATES.items():
    for lr in lr_list:
        tag = f"{opt_name}_lr{lr:g}".replace(".", "p")
        experiments.append({
            **BASE_EXP,
            "tag": tag,
            "output_dir": f"./runs/optim/{tag}",
            "optimizer_name": opt_name,
            "learning_rate": lr,
        })

results = []
print("\n===== Optimizer Comparison (Light) =====")
print(f"TRAIN_N={TRAIN_N} | TEST_N={TEST_N} | MAX_STEPS={MAX_STEPS} | BATCH={BATCH} | GRAD_ACCUM={GRAD_ACCUM}")
print(f"Fixed: wd={WEIGHT_DECAY} | lora_dropout={LORA_DROPOUT} | scheduler={SCHEDULER} | warmup_ratio={WARMUP_RATIO} | fp16={FP16_FLAG}")
print("------------------------------------------------------------")

for exp in experiments:
    try:
        m = run_experiment(exp)
        results.append({
            "tag": exp["tag"],
            "optimizer": exp["optimizer_name"],
            "lr": exp["learning_rate"],
            "acc": m.get("eval_accuracy", 0.0),
            "f1": m.get("eval_f1", 0.0),
            "loss": m.get("eval_loss", 0.0),
            "time": m.get("train_time_sec", 0.0),
        })
        print("Final metrics:", m)
    except Exception as e:

        print(f"[SKIP] {exp['tag']} failed: {repr(e)}")


print("\n===== Optimizer Compare Results =====")
print(f"{'Config':22s} | {'Opt':9s} | {'LR':10s} | {'Acc':6s} | {'F1':6s} | {'Loss':8s} | {'Time(s)':7s}")
print("-" * 90)
for r in results:
    print(
        f"{r['tag'][:22]:22s} | "
        f"{r['optimizer']:<9s} | "
        f"{r['lr']:<10g} | "
        f"{r['acc']:.4f} | "
        f"{r['f1']:.4f} | "
        f"{r['loss']:.4f} | "
        f"{r['time']:.1f}"
    )
print("-" * 90)
