import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import time
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import textattack
from textattack.attack_recipes import (
    HotFlipEbrahimi2017,
    TextFoolerJin2019,
    BERTAttackLi2020,
    BAEGarg2019,
    DeepWordBugGao2018
)
from textattack.models.wrappers import HuggingFaceModelWrapper, PyTorchModelWrapper
from textattack.models.wrappers import ModelWrapper

# Note: Using AdapterModelWrapper instead of PyTorchModelWrapper
# because we use AutoAdapterModel (adapter-based model) which requires
# special handling for adapter activation. The functionality is equivalent.

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256

def clip_text_by_tokens(text: str, tokenizer, max_len: int):
    ids = tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_model_from_checkpoint(ckpt_path: str, device: str,
                               adapter_name="lora_adapter", head_name="imdb"):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    adapter_dir = os.path.join(ckpt_path, adapter_name)
    head_dir = os.path.join(ckpt_path, head_name)

    # load adapter + head
    model.load_adapter(adapter_dir, load_as=adapter_name, set_active=True)
    model.set_active_adapters(adapter_name)
    model.load_head(head_dir)
    model.active_head = head_name

    model.to(device)
    model.eval()
    return model, tokenizer

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def predict_conf(model, tokenizer, texts, device):
    enc = tokenizer(
        texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    conf = torch.max(probs, dim=-1).values
    return pred.detach().cpu(), conf.detach().cpu(), probs.detach().cpu()

def eval_metrics(model, tokenizer, dataset, device, batch_size=16):
    n=len(dataset)
    correct = 0
    confs=[]
    for i in range(0, n, batch_size):
        batch=dataset[i: i+batch_size]
        texts=batch["text"]
        labels=torch.tensor(batch["label"], dtype=torch.long)

        pred, conf, _= predict_conf(model, tokenizer, texts, device)
        correct+=(pred==labels).sum().item()
        confs.append(conf.numpy())

    acc=correct/n
    mean_conf=float(np.concatenate(confs).mean())
    return acc, mean_conf

def eval_attack_success_rate(model, tokenizer, attack, dataset, device, n_samples=100, batch_size=16):
    """
    Evaluate attack success rate on a subset of the dataset
    
    Returns:
        attack_success_rate: Percentage of successful attacks
        avg_conf_drop: Average confidence drop (Conf(original) - Conf(adv))
        successful_attacks: Number of successful attacks
        total_attempts: Total number of attack attempts
    """
    model.eval()
    if hasattr(model, 'set_active_adapters'):
        model.set_active_adapters("lora_adapter")
    
    # Sample subset for evaluation
    n = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), size=n, replace=False)
    
    successful_attacks = 0
    total_attempts = 0
    conf_drops = []
    
    for idx in indices:
        idx = int(idx)  # Convert numpy.int64 to Python int
        text = dataset[idx]["text"]
        label = dataset[idx]["label"]
        
        # Get original prediction and confidence
        pred_orig, conf_orig, _ = predict_conf(model, tokenizer, [text], device)
        pred_orig = pred_orig[0].item()
        conf_orig = conf_orig[0].item()
        
        # Only attack if model predicts correctly
        if pred_orig == label:
            total_attempts += 1
            
            try:
                # Generate adversarial example
                res = attack.attack(text, label)
                
                if res.perturbed_result is not None:
                    adv_text = res.perturbed_result.attacked_text.text
                    
                    # Get adversarial prediction and confidence
                    pred_adv, conf_adv, _ = predict_conf(model, tokenizer, [adv_text], device)
                    pred_adv = pred_adv[0].item()
                    conf_adv = conf_adv[0].item()
                    
                    # Check if attack succeeded (prediction changed)
                    if pred_adv != pred_orig:
                        successful_attacks += 1
                        conf_drops.append(conf_orig - conf_adv)
                    
            except Exception as e:
                # Attack failed, skip
                pass
    
    attack_success_rate = (successful_attacks / total_attempts * 100) if total_attempts > 0 else 0.0
    avg_conf_drop = float(np.mean(conf_drops)) if conf_drops else 0.0
    
    return attack_success_rate, avg_conf_drop, successful_attacks, total_attempts

class AdapterModelWrapper(ModelWrapper):
    """Custom wrapper for adapter models to work with textattack"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_name = "lora_adapter"
        
    def __call__(self, text_input_list):
        """Process text inputs and return model outputs as list of lists"""
        # For PGD, we may need gradients, so check if model is in train mode
        # But for inference, use eval mode
        was_training = self.model.training
        if not was_training:
            self.model.eval()
        
        if hasattr(self.model, 'set_active_adapters'):
            self.model.set_active_adapters(self.adapter_name)
        
        encodings = self.tokenizer(
            text_input_list,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        encodings = {k: v.to(device) for k, v in encodings.items()}

        # For PGD attacks, we need gradients, so don't use no_grad()
        # But for other attacks, we can use no_grad for efficiency
        if was_training:
            outputs = self.model(**encodings)
            logits = outputs.logits
        else:
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits

        logits_np = logits.detach().cpu().numpy()
        return logits_np.tolist()
    
    def get_grad(self, text_input, loss_fn):
        """Get gradients for gradient-based attacks like PGD"""
        self.model.train()
        if hasattr(self.model, 'set_active_adapters'):
            self.model.set_active_adapters(self.adapter_name)

        encodings = self.tokenizer(
            text_input,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Forward pass with gradients
        outputs = self.model(**encodings)
        loss = loss_fn(outputs.logits)
        loss.backward()
        
        return loss.item()

def make_attack(recipe_name:str, model, tokenizer, pgd_eps=0.03, pgd_alpha=0.001, pgd_steps=5):
    """
    Create attack recipe
    
    Args:
        recipe_name: Name of the attack recipe
        model: The model to attack
        tokenizer: The tokenizer
        pgd_eps: Not used (kept for compatibility)
        pgd_alpha: Not used (kept for compatibility)
        pgd_steps: Not used (kept for compatibility)
    
    Available recipes:
        - hotflip: Gradient-based attack (fast, effective for adversarial training)
        - textfooler: Word-level substitution attack (popular, effective)
        - bertattack: BERT-based attack (slower but powerful)
        - bae: BERT-based Adversarial Examples
        - deepwordbug: Character-level attack (fast)
    """
    # Use custom wrapper for adapter models
    model_wrapper = AdapterModelWrapper(model, tokenizer)
    
    recipe_name = recipe_name.lower()
    
    if recipe_name == "hotflip":
        # HotFlip: Gradient-based attack, good for adversarial training
        return HotFlipEbrahimi2017.build(model_wrapper)
    elif recipe_name == "textfooler":
        # TextFooler: Popular word substitution attack
        return TextFoolerJin2019.build(model_wrapper)
    elif recipe_name == "bertattack":
        # BERTAttack: BERT-based attack
        return BERTAttackLi2020.build(model_wrapper)
    elif recipe_name == "bae":
        # BAE: BERT-based Adversarial Examples
        return BAEGarg2019.build(model_wrapper)
    elif recipe_name == "deepwordbug":
        # DeepWordBug: Character-level attack
        return DeepWordBugGao2018.build(model_wrapper)
    else:
        raise ValueError(
            f"Unknown recipe: {recipe_name}. "
            f"Available options: 'hotflip', 'textfooler', 'bertattack', 'bae', 'deepwordbug'"
        )

def adv_train(
    ckpt_path: str,
    out_dir: str,
    recipe: str,
    train_n: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    fp16: bool,
    adv_ratio: float,
    mix_textfooler_ratio: float,
    mix_deepwordbug_ratio: float,
    seed: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA 不可用，请检查 GPU/驱动/torch 版本"
    print(f"[INFO] Using device: {device}")
    set_all_seeds(seed)

    model, tokenizer = load_model_from_checkpoint(ckpt_path, device=device)
    model.train()

    model.train_adapter("lora_adapter")

    ds=load_dataset("stanfordnlp/imdb")
    train_ds = ds["train"].shuffle(seed=seed).select(range(train_n))
    train_texts = train_ds["text"]
    train_labels = train_ds["label"]

    test_ds = ds["test"].shuffle(seed=seed).select(range(min(400, len(ds["test"]))))

    # Skip evaluation before training to speed up runtime
    model.train()

    # attack(s)
    recipe_lower = recipe.lower()
    if recipe_lower == "mix":
        attack_textfooler = make_attack("textfooler", model, tokenizer)
        attack_deepwordbug = make_attack("deepwordbug", model, tokenizer)
        mix_total = mix_textfooler_ratio + mix_deepwordbug_ratio
        if mix_total <= 0:
            raise ValueError("mix_textfooler_ratio + mix_deepwordbug_ratio must be > 0")
        textfooler_prob = mix_textfooler_ratio / mix_total
    else:
        attack = make_attack(recipe, model, tokenizer)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device == "cuda")) if device == "cuda" else None

    os.makedirs(out_dir, exist_ok=True)

    step=0
    t0=time.time()

    while step < max_steps:
        idxs = np.random.choice(len(train_texts), size=batch_size, replace=False)
        # Convert numpy indices to Python int to avoid type issues
        idxs = [int(i) for i in idxs]
        texts = [train_texts[i] for i in idxs]
        ys = [int(train_labels[i]) for i in idxs]
        labels = torch.tensor(ys, dtype=torch.long, device=device)

        # clip clean
        texts = [clip_text_by_tokens(t, tokenizer, MAX_LEN) for t in texts]

        adv_texts = []
        adv_ok = 0
        adv_try = 0

        # Generate adversarial examples
        # For gradient-based attacks, model needs to be in train mode to compute gradients
        # But we'll switch back to train mode after generating adv examples
        use_gradients = (recipe_lower == "hotflip")
        
        if use_gradients:
            # For gradient-based attacks, model needs to allow gradients
            # But textattack will handle this internally
            model.eval()  # Start with eval mode, attack will switch if needed
        else:
            model.eval()
        
        # Ensure adapters are active before attack
        if hasattr(model, 'set_active_adapters'):
            model.set_active_adapters("lora_adapter")
        
        for t, y in zip(texts, ys):
            if random.random() < adv_ratio:
                adv_try += 1
                adv = t

                try:
                    # Ensure adapters are active for each attack
                    if hasattr(model, 'set_active_adapters'):
                        model.set_active_adapters("lora_adapter")
                    
                    # Attack the sample
                    if recipe_lower == "mix":
                        if random.random() < textfooler_prob:
                            res = attack_textfooler.attack(t, y)
                        else:
                            res = attack_deepwordbug.attack(t, y)
                    else:
                        res = attack.attack(t, y)
                    
                    # Check if attack succeeded
                    if res.perturbed_result is not None:
                        adv = res.perturbed_result.attacked_text.text
                        adv = clip_text_by_tokens(adv, tokenizer, MAX_LEN)
                        # Check if text actually changed
                        if adv != t:
                            adv_ok += 1
                        else:
                            adv = t  # If no change, use original
                    else:
                        adv = t  # Attack failed, use original

                except Exception as e:
                    if step < 3:
                        print(f"[debug] attack failed for sample: {repr(e)}")
                    adv = t  # On error, use original text
            else:
                adv = t

            adv_texts.append(adv)

        model.train()

        if (step + 1) % 10 == 0 and adv_try > 0:
            print(f"[debug] adv_gen_success_rate={adv_ok / adv_try:.2f} ({adv_ok}/{adv_try})")
            print("[debug] example clean:", texts[0][:120].replace("\n", " "))
            print("[debug] example adv  :", adv_texts[0][:120].replace("\n", " "))

        def forward_loss(text_list, labels_):
            enc = tokenizer(text_list, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            return torch.nn.functional.cross_entropy(out.logits, labels_)

        if fp16 and device == "cuda":
            with torch.amp.autocast('cuda'):
                loss_clean = forward_loss(texts, labels)
                loss_adv = forward_loss(adv_texts, labels)
                loss = (1 - adv_ratio) * loss_clean + adv_ratio * loss_adv
        else:
            loss_clean = forward_loss(texts, labels)
            loss_adv = forward_loss(adv_texts, labels)
            loss = (1 - adv_ratio) * loss_clean + adv_ratio * loss_adv

        loss = loss / grad_accum
        
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 10 == 0:
            print(f"[step {step + 1}/{max_steps}] loss={loss.item() * grad_accum:.4f}")

        step += 1

    train_time=time.time()-t0

    # Skip evaluation after training to speed up runtime

    # Build run tag
    if recipe_lower == "mix":
        run_tag = (
            f"{recipe}_N{train_n}_S{max_steps}_bs{batch_size}_ga{grad_accum}"
            f"_lr{lr}_ar{adv_ratio}_tf{mix_textfooler_ratio}_dw{mix_deepwordbug_ratio}_seed{seed}_{int(time.time())}"
        )
    else:
        run_tag = (
            f"{recipe}_N{train_n}_S{max_steps}_bs{batch_size}_ga{grad_accum}"
            f"_lr{lr}_ar{adv_ratio}_seed{seed}_{int(time.time())}"
        )
    save_dir = os.path.join(out_dir, f"adv_trained_{run_tag}")
    os.makedirs(save_dir, exist_ok=True)

    model.save_adapter(os.path.join(save_dir, "lora_adapter"), "lora_adapter")
    model.save_head(os.path.join(save_dir, "imdb"), "imdb")
    tokenizer.save_pretrained(save_dir)

    print("\n" + "="*60)
    print("===== ADVERSARIAL TRAINING RESULTS =====")
    print("="*60)
    print(f"Recipe: {recipe}")
    if recipe_lower == "mix":
        print(f"Mix Ratios: textfooler={mix_textfooler_ratio:.2f} | deepwordbug={mix_deepwordbug_ratio:.2f}")
    print(f"Training Config: N={train_n} | Steps={max_steps} | BS={batch_size} | GA={grad_accum} | LR={lr} | FP16={fp16} | AdvRatio={adv_ratio}")
    print(f"\nTraining Time: {train_time:.2f}s")
    print(f"\n--- Model Performance Comparison ---")
    print("Evaluation skipped to speed up runtime.")
    print("\n--- Summary ---")
    print("✓ Evaluation skipped to speed up runtime.")
    print(f"\n[SAVED] Model saved to: {save_dir}")
    print("="*60)

    return save_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to a specific checkpoint folder (e.g., /path/to/checkpoint-93)")
    parser.add_argument("--out_dir", type=str, default="./runs/adv_train_textattack",
                        help="Output directory for saving adversarial trained models")
    parser.add_argument("--recipe", type=str, default="textfooler", 
                        choices=["hotflip", "textfooler", "bertattack", "bae", "deepwordbug", "mix"], 
                        help="Attack recipe: 'hotflip' (gradient-based, fast), 'textfooler' (word substitution), "
                             "'bertattack' (BERT-based), 'bae' (BERT adversarial), 'deepwordbug' (char-level), "
                             "'mix' (textfooler + deepwordbug)")
    parser.add_argument("--train_n", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--adv_ratio", type=float, default=0.5)
    parser.add_argument("--mix_textfooler_ratio", type=float, default=0.6,
                        help="When recipe=mix, probability weight for TextFooler")
    parser.add_argument("--mix_deepwordbug_ratio", type=float, default=0.4,
                        help="When recipe=mix, probability weight for DeepWordBug")
    parser.add_argument("--seed", type=int, default=42)

    args=parser.parse_args()

    adv_ckpt_dir=adv_train(
        ckpt_path=args.ckpt_path,
        out_dir=args.out_dir,
        recipe=args.recipe,
        train_n=args.train_n,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        fp16=args.fp16,
        adv_ratio=args.adv_ratio,
        mix_textfooler_ratio=args.mix_textfooler_ratio,
        mix_deepwordbug_ratio=args.mix_deepwordbug_ratio,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()