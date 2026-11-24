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

# DistilBERT 常量
NUM_LAYERS = 6
FFN_INTERMEDIATE_SIZE = 3072
FFN_HIDDEN_SIZE = 768

def load_model(ckpt_path, device="cuda"):
    model=AutoAdapterModel.from_pretrained("distilbert-base-uncased")
    adapter_path=os.path.join(ckpt_path, "lora_adapter")
    model.load_adapter(adapter_path, load_as="lora_adapter", set_active=True)
    model.set_active_adapters("lora_adapter")

    head_path=os.path.join(ckpt_path, "imdb")
    model.load_head(head_path)
    model.active_head="imdb"

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    return model, tokenizer

def get_ffn_weight(model):
    ffn_weights={}  # 字典，不是列表

    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn
        ffn_weights[(layer_idx, 'lin1')]=ffn.lin1.weight
        ffn_weights[(layer_idx, 'lin2')] = ffn.lin2.weight

    return ffn_weights

def compute_global_threshold(ffn_weights, sparsity_ratio):
    all_weights=[]
    for weight in ffn_weights.values():
        all_weights.append(weight.abs().flatten())

    all_weights=torch.cat(all_weights)

    k=int(len(all_weights) *sparsity_ratio)
    if k==0:
        threshold=0.0
    else:
        threshold=torch.kthvalue(all_weights, k).values.item()
    return threshold

def compute_layerwise_threshold(ffn_weights, sparsity_ratio):
    thresholds={}
    for key, weight in ffn_weights.items():
        layer_idx, lin_name=key
        abs_weight=weight.abs().flatten()

        k=int(len(abs_weight)*sparsity_ratio)
        if k==0:
            threshold=0.0
        else:
            threshold=torch.kthvalue(abs_weight, k).values.item()

        thresholds[key]=threshold

    return thresholds

def apply_pruning(model, sparsity_ratio, method='global'):
    ffn_weights=get_ffn_weight(model)

    if method=='global':
        threshold=compute_global_threshold(ffn_weights, sparsity_ratio)
        thresholds={key: threshold for key in ffn_weights.keys()}
    else:
        thresholds=compute_layerwise_threshold(ffn_weights, sparsity_ratio)

    total_pruned=0
    total_weights=0
    layer_stats=[]

    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn

        layer_pruned=0
        layer_total=0

        for lin_name, lin_layer in [('lin1', ffn.lin1), ('lin2', ffn.lin2)]:
            key=(layer_idx, lin_name)
            threshold=thresholds[key]

            mask=lin_layer.weight.abs()<threshold

            n_pruned=mask.sum().item()
            n_total=mask.numel()

            with torch.no_grad():
                lin_layer.weight.data[mask]=0.0

            layer_pruned+=n_pruned
            layer_total+=n_total

        total_pruned+=layer_pruned
        total_weights+=layer_total

        layer_stats.append({
            'layer': layer_idx,
            'pruned': layer_pruned,
            'total': layer_total,
            'sparsity': layer_pruned/layer_total *100
        })

    # return 在循环外部
    return {
        'total_pruned': total_pruned,
        'total_weights': total_weights,
        'actual_sparsity': total_pruned / total_weights,
        'layer_stats': layer_stats
    }

def load_test_data(n_samples=500):
    dataset=load_dataset("imdb", split="test")
    dataset=dataset.shuffle(seed=42).select(range(n_samples))
    return dataset['text'], dataset['label']

def evaluate_model(model, tokenizer, texts, labels, device, batch_size=16):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts=texts[i:i+batch_size]
            batch_labels=labels[i:i+batch_size]

            inputs=tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
            outputs=model(**inputs)
            logits=outputs.logits
            preds=torch.argmax(logits, dim=1).cpu().numpy()
            correct+=sum(p==l for p, l in zip(preds, batch_labels))
            total+=len(batch_labels)

    return correct/total * 100


def test_sparsity_ratios(ckpt_path, tokenizer, texts, labels, device,
                         ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    results = []

    model, _ = load_model(ckpt_path, device)
    baseline_acc = evaluate_model(model, tokenizer, texts, labels, device)
    results.append((0.0, baseline_acc, 0))
    del model
    torch.cuda.empty_cache()

    for ratio in ratios:
        print(f"\n[测试] 稀疏率 {ratio * 100:.0f}%:")

        model, _ = load_model(ckpt_path, device)

        stats = apply_pruning(model, ratio, method='global')
        acc = evaluate_model(model, tokenizer, texts, labels, device)
        print(f"  准确率: {acc:.2f}% (下降 {baseline_acc - acc:.2f}%)")

        results.append((ratio, acc, stats['total_pruned']))

        del model
        torch.cuda.empty_cache()

    return results, baseline_acc


def plot_sparsity_results(results, baseline_acc, output_dir):
    ratios = [r[0] * 100 for r in results]
    accuracies = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ratios, accuracies, 'go-', linewidth=2, markersize=8)

    ax.axhline(y=baseline_acc, color='r', linestyle='--',
               label=f'Baseline: {baseline_acc:.2f}%')

    ax.axhline(y=baseline_acc - 1, color='orange', linestyle='--',
               label=f'1% drop: {baseline_acc - 1:.2f}%')

    ax.set_xlabel('Sparsity Ratio (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('FFN Pruning: Accuracy vs Sparsity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ratio, acc, _ in results:
        ax.annotate(f'{acc:.1f}%', (ratio * 100, acc),
                    textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ffn_sparsity_accuracy.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_pruned_model(model, output_dir, sparsity_ratio):
    save_path = os.path.join(output_dir, f"pruned_model_sparsity{int(sparsity_ratio * 100)}.pt")

    ffn_state = {}
    for layer_idx in range(NUM_LAYERS):
        ffn = model.distilbert.transformer.layer[layer_idx].ffn
        ffn_state[f'layer{layer_idx}.lin1.weight'] = ffn.lin1.weight.data.cpu()
        ffn_state[f'layer{layer_idx}.lin1.bias'] = ffn.lin1.bias.data.cpu()
        ffn_state[f'layer{layer_idx}.lin2.weight'] = ffn.lin2.weight.data.cpu()
        ffn_state[f'layer{layer_idx}.lin2.bias'] = ffn.lin2.bias.data.cpu()

    torch.save(ffn_state, save_path)
    print(f"[已保存] {save_path}")

    return save_path


def generate_report(results, baseline_acc, recommended_sparsity, output_dir):
    report = [
        "=" * 60,
        "FFN 非结构化剪枝报告",
        "=" * 60,
        "",
        "1. 剪枝策略",
        "-" * 40,
        "   方法: 基于权重绝对值的全局剪枝 (Magnitude Pruning)",
        "   粒度: 权重级 (Weight-level)",
        "   范围: 全局阈值",
        "",
        "2. 测试结果",
        "-" * 40,
        f"   基准准确率: {baseline_acc:.2f}%",
        "",
    ]

    for ratio, acc, n_pruned in results:
        if ratio == 0:
            continue
        drop = baseline_acc - acc
        report.append(f"   稀疏率 {ratio * 100:>3.0f}%: 准确率 {acc:.2f}% (下降 {drop:.2f}%)")

    best_ratio = 0
    for ratio, acc, _ in results:
        if baseline_acc - acc <= 1.0:
            best_ratio = ratio

    report.extend([
        "",
        "3. 推荐配置",
        "-" * 40,
        f"   推荐稀疏率: {recommended_sparsity * 100:.0f}%",
        f"   预期准确率下降: <= 1%",
        "",
        "4. 结论",
        "-" * 40,
        f"   - FFN 权重可以安全地剪枝约 {best_ratio * 100:.0f}% 而性能下降 < 1%",
        f"   - 更高的稀疏率会导致更明显的性能下降",
        "",
        "=" * 60,
    ])

    report_text = "\n".join(report)

    report_path = os.path.join(output_dir, "ffn_pruning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[已保存] {report_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="FFN 非结构化剪枝")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./ffn_pruning_output",
                        help="输出目录")
    parser.add_argument("--n_samples", type=int, default=500, help="测试样本数")
    parser.add_argument("--recommended_sparsity", type=float, default=0.3,
                        help="推荐稀疏率")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("FFN 非结构化剪枝 (Magnitude Pruning)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    texts, labels = load_test_data(args.n_samples)

    results, baseline_acc = test_sparsity_ratios(
        args.ckpt_path, tokenizer, texts, labels, device,
        ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )

    # 3. 可视化
    plot_sparsity_results(results, baseline_acc, args.output_dir)

    # 4. 保存推荐稀疏率的模型
    print(f"\n保存推荐稀疏率 ({args.recommended_sparsity * 100:.0f}%) 的剪枝模型...")
    model, _ = load_model(args.ckpt_path, device)
    apply_pruning(model, args.recommended_sparsity, method='global')
    save_pruned_model(model, args.output_dir, args.recommended_sparsity)

    # 5. 生成报告
    report = generate_report(results, baseline_acc, args.recommended_sparsity, args.output_dir)
    print("\n" + report)

    # 6. 保存结果
    results_df = pd.DataFrame(results, columns=['sparsity', 'accuracy', 'n_pruned'])
    results_df.to_csv(os.path.join(args.output_dir, "ffn_pruning_results.csv"), index=False)

    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()