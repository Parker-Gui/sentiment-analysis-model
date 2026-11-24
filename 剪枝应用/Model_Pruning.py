import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from datasets import load_dataset


NUM_LAYERS = 6
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN_SIZE = 768
FFN_INTERMEDIATE_SIZE = 3072

def load_model(ckpt_path, device="cuda"):
    model = AutoAdapterModel.from_pretrained("distilbert-base-uncased")

    adapter_path = os.path.join(ckpt_path, "lora_adapter")
    model.load_adapter(adapter_path, load_as="lora_adapter", set_active=True)
    model.set_active_adapters("lora_adapter")

    head_path = os.path.join(ckpt_path, "imdb")
    model.load_head(head_path)
    model.active_head = "imdb"

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    return model, tokenizer

def load_test_data(n_samples=500):
    dataset = load_dataset("imdb", split="test")
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    return dataset['text'], dataset['label']


def evaluate_model(model, tokenizer, texts, labels, device, batch_size=16):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="评估中"):
            batch_texts = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            correct += sum(p == l for p, l in zip(preds, batch_labels))
            total += len(batch_labels)

    accuracy = correct / total * 100
    return accuracy

def prune_attention_heads(model, heads_to_prune, device="cuda"):
    heads_by_layer = {}
    for layer, head in heads_to_prune:
        heads_by_layer.setdefault(layer, []).append(head)

    pruning_stats = {
        'total_heads_before': NUM_LAYERS * NUM_HEADS,
        'total_heads_pruned': len(heads_to_prune),
        'layer_stats': []
    }

    for layer_idx in range(NUM_LAYERS):
        heads_to_prune_in_layer = heads_by_layer.get(layer_idx, [])
        if not heads_to_prune_in_layer:
            pruning_stats['layer_stats'].append({
                'layer': layer_idx,
                'pruned': 0,
                'kept': NUM_HEADS
            })
            continue

        num_heads_to_prune = len(heads_to_prune_in_layer)
        print(f"  剪枝 {num_heads_to_prune}/{NUM_HEADS} 个头: {heads_to_prune_in_layer}")
        heads_to_keep = [h for h in range(NUM_HEADS) if h not in heads_to_prune_in_layer]
        print(f"  保留头: {heads_to_keep}")

        attention = model.distilbert.transformer.layer[layer_idx].attention

        with torch.no_grad():
            for proj_name, proj_layer in [('q_lin', attention.q_lin), ('k_lin', attention.k_lin), ('v_lin', attention.v_lin)]:
                weight = proj_layer.weight.data.view(NUM_HEADS, HEAD_DIM, HIDDEN_SIZE)
                bias = proj_layer.bias.data.view(NUM_HEADS, HEAD_DIM)

                weight[heads_to_prune_in_layer, :, :] = 0.0
                bias[heads_to_prune_in_layer, :] = 0.0

            out_weight = attention.out_lin.weight.data.view(HIDDEN_SIZE, NUM_HEADS, HEAD_DIM)
            out_weight[:, heads_to_prune_in_layer, :] = 0.0

        pruning_stats['layer_stats'].append({
            'layer': layer_idx,
            'pruned': num_heads_to_prune,
            'kept': NUM_HEADS - num_heads_to_prune
        })

    pruning_stats['total_heads_after'] = pruning_stats['total_heads_before'] - pruning_stats['total_heads_pruned']

    return model, pruning_stats

def load_heads_to_prune(csv_path, prune_ratio=0.2):
    df=pd.read_csv(csv_path)
    df_sorted=df.sort_values('acc_drop_pct', ascending=True)

    n_prune=int(len(df_sorted) * prune_ratio)
    heads_to_prune=[]

    for i in range(n_prune):
        row=df_sorted.iloc[i]
        heads_to_prune.append((int(row['layer']), int(row['head'])))

    return heads_to_prune

def prune_ffn_weights(model, sparsity_ratio=0.2, method='global'):
    ffn_weights={}

    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn

        ffn_weights[(layer_idx, 'lin1')]=ffn.lin1.weight
        ffn_weights[(layer_idx, 'lin2')]=ffn.lin2.weight

    if method == 'global':
        all_weights = []
        for weight in ffn_weights.values():
            all_weights.append(weight.abs().flatten())
        all_weights = torch.cat(all_weights)

        k = int(len(all_weights) * sparsity_ratio)
        if k == 0:
            threshold = 0.0
        else:
            threshold = torch.kthvalue(all_weights, k).values.item()

        thresholds = {key: threshold for key in ffn_weights.keys()}
    else:
        thresholds = {}
        for key, weight in ffn_weights.items():
            abs_weight = weight.abs().flatten()
            k = int(len(abs_weight) * sparsity_ratio)
            if k == 0:
                threshold = 0.0
            else:
                threshold = torch.kthvalue(abs_weight, k).values.item()
            thresholds[key] = threshold

    total_pruned = 0
    total_weights = 0
    layer_stats = []
    pruning_masks = {}

    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn
        layer_pruned=0
        layer_total=0

        for lin_name, lin_layer in [('lin1', ffn.lin1), ('lin2', ffn.lin2)]:
            key=(layer_idx, lin_name)
            threshold=thresholds[key]

            mask=lin_layer.weight.abs() <threshold

            n_pruned=mask.sum().item()
            n_total=mask.numel()

            with torch.no_grad():
                lin_layer.weight.data[mask] = 0.0

            keep_mask = ~mask
            pruning_masks[key] = keep_mask.detach().cpu()

            layer_pruned +=n_pruned
            layer_total +=n_total

        total_pruned += layer_pruned
        total_weights += layer_total

        layer_stats.append({
            'layer': layer_idx,
            'pruned': layer_pruned,
            'total': layer_total,
            'sparsity_pct': layer_pruned / layer_total * 100
        })

    pruning_stats = {
        'total_pruned': total_pruned,
        'total_weights': total_weights,
        'actual_sparsity': total_pruned / total_weights,
        'layer_stats': layer_stats
    }

    return pruning_stats, pruning_masks

def pruning(ckpt_path, importance_csv, attention_prune_ratio=0.2, ffn_sparsity=0.2, output_dir="./pruned_model", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model(ckpt_path, device)
    texts, labels = load_test_data(n_samples=500)

    baseline_acc = evaluate_model(model, tokenizer, texts, labels, device)

    heads_to_prune = load_heads_to_prune(importance_csv, attention_prune_ratio)
    model, attention_stats = prune_attention_heads(model, heads_to_prune, device)

    acc_after_attention = evaluate_model(model, tokenizer, texts, labels, device)

    ffn_stats, pruning_masks = prune_ffn_weights(model, ffn_sparsity, method='global')
    acc_after_ffn = evaluate_model(model, tokenizer, texts, labels, device)


    model_save_path = os.path.join(output_dir, "pruned_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'attention_stats': attention_stats,
        'ffn_stats': ffn_stats,
        'pruning_masks': pruning_masks,
        'baseline_acc': baseline_acc,
        'final_acc': acc_after_ffn,
    }, model_save_path)

    mask_save_path = os.path.join(output_dir, "pruning_masks.pt")
    torch.save(pruning_masks, mask_save_path)
    print(f"  掩码已保存: {mask_save_path}")

    generate_pruning_report(baseline_acc, acc_after_attention, acc_after_ffn,
                            attention_stats, ffn_stats, output_dir)

    return model, {
        'baseline_acc': baseline_acc,
        'acc_after_attention': acc_after_attention,
        'acc_after_ffn': acc_after_ffn,
        'attention_stats': attention_stats,
        'ffn_stats': ffn_stats
    }
def generate_pruning_report(baseline_acc, acc_after_attention, acc_after_ffn,attention_stats, ffn_stats, output_dir):
    report = [
        "=" * 60,
        "模型剪枝报告",
        "=" * 60,
        "",
        "1. 准确率变化",
        "-" * 40,
        f"  原始准确率: {baseline_acc:.2f}%",
        f"  结构化剪枝后: {acc_after_attention:.2f}% (下降 {baseline_acc - acc_after_attention:.2f}%)",
        f"  非结构化剪枝后: {acc_after_ffn:.2f}% (总下降 {baseline_acc - acc_after_ffn:.2f}%)",
        "",
        "2. 结构化剪枝（注意力头）",
        "-" * 40,
        f"  剪枝前: {attention_stats['total_heads_before']} 个头",
        f"  剪枝后: {attention_stats['total_heads_after']} 个头",
        f"  剪枝比例: {attention_stats['total_heads_pruned'] / attention_stats['total_heads_before'] * 100:.1f}%",
        "",
        "  各层剪枝情况:",
    ]

    for layer_stat in attention_stats['layer_stats']:
        report.append(
            f"    Layer {layer_stat['layer']}: 保留 {layer_stat['kept']}/{layer_stat['kept'] + layer_stat['pruned']} 个头")

    report.extend([
        "",
        "3. 非结构化剪枝（FFN）",
        "-" * 40,
        f"  总权重数: {ffn_stats['total_weights']:,}",
        f"  剪枝权重: {ffn_stats['total_pruned']:,}",
        f"  实际稀疏率: {ffn_stats['actual_sparsity'] * 100:.2f}%",
        "",
        "  各层稀疏率:",
    ])

    for layer_stat in ffn_stats['layer_stats']:
        report.append(f"    Layer {layer_stat['layer']}: {layer_stat['sparsity_pct']:.2f}%")


    attention_params_removed = attention_stats['total_heads_pruned'] * (HEAD_DIM * HIDDEN_SIZE * 4)  # Q, K, V, O
    ffn_params_removed = ffn_stats['total_pruned']
    total_params_removed = attention_params_removed + ffn_params_removed
    original_params = 68_170_844 

    report.extend([
        "",
        "4. 压缩效果",
        "-" * 40,
        f"  移除参数（注意力）: {attention_params_removed:,}",
        f"  移除参数（FFN）: {ffn_params_removed:,}",
        f"  总移除参数: {total_params_removed:,}",
        f"  原始参数: {original_params:,}",
        f"  压缩率: {total_params_removed / original_params * 100:.2f}%",
        "",
        "=" * 60,
    ])

    report_text = "\n".join(report)

    report_path = os.path.join(output_dir, "pruning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[已保存] {report_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="综合模型剪枝（结构化 + 非结构化）")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--importance_csv", type=str, required=True,
                        help="注意力头重要性分析结果")
    parser.add_argument("--attention_prune_ratio", type=float, default=0.2,
                        help="注意力头剪枝比例")
    parser.add_argument("--ffn_sparsity", type=float, default=0.2,
                        help="FFN 权重稀疏率")
    parser.add_argument("--output_dir", type=str, default="./pruned_model",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")

    args = parser.parse_args()

    model, results = pruning(
        ckpt_path=args.ckpt_path,
        importance_csv=args.importance_csv,
        attention_prune_ratio=args.attention_prune_ratio,
        ffn_sparsity=args.ffn_sparsity,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()