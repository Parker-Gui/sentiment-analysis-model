import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import random
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256

SENTIMENT_SYNONYMS = {
    # positive
    "good": ["great", "nice", "decent", "fine", "solid"],
    "great": ["excellent", "amazing", "fantastic", "wonderful"],
    "excellent": ["outstanding", "superb", "brilliant"],
    "amazing": ["incredible", "awesome", "remarkable"],
    "love": ["like", "enjoy", "adore"],
    "awesome": ["fantastic", "great", "amazing"],
    "wonderful": ["great", "lovely", "excellent"],

    # negative
    "bad": ["awful", "poor", "terrible", "horrible"],
    "terrible": ["awful", "horrible", "dreadful"],
    "awful": ["terrible", "horrible", "bad"],
    "boring": ["dull", "tedious", "uninteresting"],
    "hate": ["dislike", "detest"],
    "worst": ["awful", "terrible", "horrible"],

    # common movie-review adjectives
    "funny": ["humorous", "amusing", "comical"],
    "sad": ["depressing", "tragic", "miserable"],
    "slow": ["sluggish", "dragging", "lethargic"],
}

NEGATION_WORDS = ["not", "never", "no", "hardly", "scarcely"]

DOUBLE_NEGATION = {
    "good": "not bad",
    "great": "not terrible",
    "excellent": "not unimpressive",
    "amazing": "not awful",
    "awesome": "not bad",
    "bad": "not good",
    "terrible": "not great",
    "awful": "not good",
    "boring": "not interesting",
}


CONDITIONAL_PREFIX = {
    "positive": ["If you ignore the flaws, ", "Despite some issues, "],
    "negative": ["Although it has some merits, ", "Even with good points, "],
}


_word_re = re.compile(r"\b\w+\b", re.UNICODE)

def synonym_substitution(text: str, max_replacements: int=6, p: float=0.8, seed: int=42) -> str:
    rng = random.Random(seed)
    words = text.split()
    idxs = list(range(len(words)))
    rng.shuffle(idxs)

    replaced = 0
    for i in idxs:
        if replaced >= max_replacements:
            break
        w = re.sub(r"[^A-Za-z]", "", words[i]).lower()
        if not w or w not in SENTIMENT_SYNONYMS:
            continue
        if rng.random() > p:
            continue

        cand = rng.choice(SENTIMENT_SYNONYMS[w])
        # preserve simple capitalization
        if words[i][:1].isupper():
            cand = cand.capitalize()

        suffix = ""
        m = re.search(r"([^\w]+)$", words[i])
        if m:
            suffix = m.group(1)
        words[i] = cand + suffix
        replaced += 1

    return " ".join(words)


POS_WORDS = {"good","great","excellent","amazing","love","awesome","wonderful","funny"}
NEG_WORDS = {"bad","terrible","awful","boring","hate","worst","sad","slow"}

def insert_negation(text: str, p: float = 0.8, seed: int = 42) -> str:
    rng = random.Random(seed)
    words = text.split()
    if len(words) < 2:
        return text
    if rng.random() > p:
        return text

    candidates = []
    for i, w in enumerate(words):
        key = re.sub(r"[^A-Za-z]", "", w).lower()
        if key in POS_WORDS or key in NEG_WORDS:
            candidates.append(i)

    if candidates:
        i = rng.choice(candidates)
        # 在情感词前插入，效果通常最强
        neg = rng.choice(["not", "never", "no", "hardly", "scarcely"])
        words.insert(max(0, i), neg)
        return " ".join(words)

    # 找不到情感词，就退化为原来的随机插入
    pos = rng.randint(1, len(words) - 1)
    neg = rng.choice(NEGATION_WORDS)
    words.insert(pos, neg)
    return " ".join(words)


def double_negation(text: str, max_replacements: int = 2, seed: int = 42) -> str:
    rng = random.Random(seed)
    out = text
    for _ in range(max_replacements):
        tokens = _word_re.findall(out.lower())
        candidates = [t for t in tokens if t in DOUBLE_NEGATION]
        if not candidates:
            break
        target = rng.choice(candidates)
        repl = DOUBLE_NEGATION[target]
        out = re.sub(rf"\b{re.escape(target)}\b", repl, out, count=1, flags=re.IGNORECASE)
    return out


def conditional_statement(text: str, true_label: int, seed: int=42)-> str:
    rng=random.Random(seed)
    if true_label == 1:
        prefix = rng.choice(CONDITIONAL_PREFIX["positive"])
    else:
        prefix = rng.choice(CONDITIONAL_PREFIX["negative"])
    return prefix + text

ATTACK_FUNCS= {
    "synonym": lambda t, y, s: synonym_substitution(t, seed=s),
    "negation": lambda t, y, s: insert_negation(t, seed=s),
    "double": lambda t, y, s: double_negation(t, seed=s),
    "conditional": lambda t, y, s: conditional_statement(t, y, seed=s),
}

def apply_attack(text: str, label: int, attack: str, seed: int)-> str:
    return ATTACK_FUNCS[attack](text, label, seed)

"""
model loading + adapter
"""


@torch.no_grad()
def predict_batch(model, tokenizer, texts, device, batch_size: int = 16):
    all_logits = []
    all_probs = []
    all_preds = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        logits = out.logits

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        confs = torch.max(probs, dim=-1).values

        all_logits.append(logits.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_confs.append(confs.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.cat(all_probs, dim=0)
    preds = torch.cat(all_preds, dim=0)
    confs = torch.cat(all_confs, dim=0)
    return logits, probs, preds, confs


def evaluate_attack(model, tokenizer, samples, attack: str, seed: int, device):
    texts=[t for t, y in samples]
    labels=torch.tensor([y for t, y in samples], dtype=torch.long)

    #original
    _, probs_o, preds_o, conf_o=predict_batch(model, tokenizer, texts, device, batch_size=16)

    #adversarial texts
    adv_texts=[apply_attack(t,y,attack, seed+i) for i, (t, y) in enumerate(samples)]
    _, probs_a, preds_a, conf_a=predict_batch(model, tokenizer, adv_texts, device, batch_size=16)

    #metrics
    correct_o=(preds_o==labels)
    correct_a=(preds_a==labels)
    success=(correct_o & (~correct_a)).sum().item()
    asr_total=success / len(samples)

    conf_drop = (conf_o - conf_a).mean().item()

    # adv accuracy
    acc_adv = correct_a.float().mean().item()
    acc_orig = correct_o.float().mean().item()

    return {
        "asr_total": asr_total,
        "acc_orig": acc_orig,
        "acc_adv": acc_adv,
        "conf_drop": conf_drop,
    }


#Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to a specific checkpoint folder")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attacks", type=str, default="synonym,negation,double,conditional",
                        help="Comma-separated: synonym,negation,double,conditional")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = args.ckpt_path.strip()

    # model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    adapter_dir = os.path.join(ckpt, "lora_adapter")
    head_dir = os.path.join(ckpt, "imdb")

    adapter_name = "lora_adapter"
    model.load_adapter(adapter_dir, load_as=adapter_name, set_active=False)
    model.load_head(head_dir)

    model.active_head = "imdb"
    model.set_active_adapters(adapter_name)

    model.to(device)
    model.eval()


    # Load dataset + sample
    ds = load_dataset("stanfordnlp/imdb")
    test_ds = ds["test"].shuffle(seed=args.seed).select(range(args.n_samples))
    samples = [(x["text"], int(x["label"])) for x in test_ds]

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    print("\n===== Manual Rule-based Adversarial Evaluation =====")
    print(f"Samples: {args.n_samples} | Seed: {args.seed} | Attacks: {attacks}")
    print("-" * 100)
    print(
        f"{'Attack':12s} | {'OrigAcc':7s} | {'AdvAcc':7s}  | {'ASR_all':7s} | {'ΔConf(mean)':11s}")
    print("-" * 100)

    results = []
    for atk in attacks:
        r = evaluate_attack(model, tokenizer, samples, atk, seed=args.seed, device=device)
        r["attack"] = atk
        results.append(r)
        print(
            f"{atk:12s} | "
            f"{r['acc_orig']:.4f}  | "
            f"{r['acc_adv']:.4f}  | "
            f"{r['asr_total']:.4f} | "
            f"{r['conf_drop']:.4f}"
        )

    print("-" * 100)


if __name__ == "__main__":
    main()

