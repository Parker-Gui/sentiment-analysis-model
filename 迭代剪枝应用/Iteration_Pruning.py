import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from datasets import load_dataset


NUM_LAYERS = 6
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN_SIZE = 768
FFN_INTERMEDIATE_SIZE = 3072

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_model(ckpt_path, n_test_samples=500, device="cuda"):
    print(f"[加载模型] 路径: {ckpt_path}")
    model=AutoAdapterModel.from_pretrained("distilbert-base-uncased")
    adapter_path=os.path.join(ckpt_path, "lora_adapter")
    model.load_adapter(adapter_path, load_as="lora_adapter", set_active=True)
    model.set_active_adapters("lora_adapter")

    head_path=os.path.join(ckpt_path, "imdb")
    model.load_head(head_path)
    model.active_head="imdb"

    model=model.to(device)
    model.eval()

    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print(f"[加载数据] 测试样本数: {n_test_samples}")
    dataset=load_dataset("imdb", split="test")
    dataset=dataset.shuffle(seed=42).select(range(n_test_samples))

    test_texts=dataset['text']
    test_labels=dataset['label']

    return model, tokenizer, test_texts, test_labels

def evaluate_model(model, tokenizer, texts, labels, device, batch_size=16):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts=texts[i:i+batch_size]
            batch_labels=labels[i:i+batch_size]

            inputs=tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(device)

            outputs=model(**inputs)
            logits=outputs.logits
            preds=torch.argmax(logits, dim=1).cpu().numpy()

            correct+=sum(p==l for p, l in zip(preds, batch_labels))
            total+=len(batch_labels)
    accuracy=correct/total *100
    return accuracy

def calculate_sparsity(model):
    total_zeros=0
    total_elements=0

    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn

        for lin_layer in [ffn.lin1, ffn.lin2]:
            weight=lin_layer.weight.data
            total_zeros+=(weight==0).sum().item()
            total_elements+=weight.numel()

    sparsity=total_zeros / total_elements if total_elements>0 else 0.0
    return sparsity

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def finetune_data(n_samples=1000, seed=42):
    print(f"[准备微调数据] 样本数: {n_samples}")
    ds=load_dataset("imdb")
    train_subset=ds["train"].shuffle(seed=seed).select(range(n_samples))

    split_idx=int(n_samples *0.8)

    train_texts=train_subset['text'][:split_idx]
    train_labels=train_subset['label'][:split_idx]

    val_texts=train_subset['text'][split_idx:]
    val_labels=train_subset['label'][split_idx:]

    return train_texts, train_labels, val_texts, val_labels

def apply_pruning_mask_gradients(model, pruning_masks):
    if not pruning_masks:
        return

    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn

        key=(layer_idx, 'lin1')
        if key in pruning_masks:
            mask=pruning_masks[key].to(ffn.lin1.weight.device)
            if ffn.lin1.weight.grad is not None:
                ffn.lin1.weight.grad.data*=mask

        key=(layer_idx, 'lin2')
        if key in pruning_masks:
            mask=pruning_masks[key].to(ffn.lin2.weight.device)
            if ffn.lin2.weight.grad is not None:
                ffn.lin2.weight.grad.data*=mask

def apply_pruning_mask_weights(model, pruning_masks):
    if not pruning_masks:
        return

    with torch.no_grad():
        for layer_idx in range(NUM_LAYERS):
            ffn=model.distilbert.transformer.layer[layer_idx].ffn

            key=(layer_idx, 'lin1')
            if key in pruning_masks:
                mask=pruning_masks[key].to(ffn.lin1.weight.device)
                ffn.lin1.weight.data*=mask

            key=(layer_idx, 'lin2')
            if key in pruning_masks:
                mask=pruning_masks[key].to(ffn.lin2.weight.device)
                ffn.lin2.weight.data*=mask

def finetune_recovery(model, tokenizer, pruning_masks, device, epochs=2, batch_size=16, lr=1e-5,n_samples=1000):
    print(f"\n{'=' * 50}")
    print("[微调] 开始性能恢复")
    print(f"{'=' * 50}")
    print(f"  Epochs: {epochs}")
    print(f"  学习率: {lr}")
    print(f"  批大小: {batch_size}")
    print(f"  数据量: {n_samples}")

    train_texts, train_labels, val_texts, val_labels=finetune_data(n_samples=n_samples, seed=42)
    model.train()
    model.train_adapter("lora_adapter")

    trainable_params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=0.01
    )

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        total_loss=0
        n_batches=0

        indices=list(range(len(train_texts)))
        random.shuffle(indices) #打乱数据

        for i in range(0, len(train_texts), batch_size):
            batch_indices=indices[i:i+batch_size]
            batch_texts=[train_texts[j] for j in batch_indices]
            batch_labels=torch.tensor(
                [train_labels[j] for j in batch_indices],
                dtype=torch.long,
                device=device
            )

            inputs=tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(device)

            #Forward pass
            outputs=model(**inputs)
            loss=torch.nn.functional.cross_entropy(outputs.logits, batch_labels)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()

            apply_pruning_mask_gradients(model, pruning_masks)
            optimizer.step() #Update Parameters
            apply_pruning_mask_weights(model, pruning_masks)

            total_loss+=loss.item()
            n_batches+=1

        avg_loss=total_loss / n_batches

        model.eval()
        val_acc=evaluate_model(model, tokenizer, val_texts, val_labels, device=device, batch_size=16)
        model.train()
        
        print(f"  Loss: {avg_loss:.4f} | 验证准确率: {val_acc:.2f}%")

    model.eval()
    final_acc=evaluate_model(model, tokenizer, val_texts, val_labels, device=device)
    print(f"\n[微调完成] 最终准确率: {final_acc:.2f}%")
    print(f"{'=' * 50}\n")

    return final_acc


def prune_attention_heads(model, heads_to_prune, device="cuda"):
    heads_by_layer={}
    for layer, head in heads_to_prune:
        heads_by_layer.setdefault(layer, []).append(head)

    pruning_stats={
        'total_heads_before': NUM_LAYERS * NUM_HEADS,
        'total_heads_pruned': len(heads_to_prune),
        'layer_stats':[]
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

        attention=model.distilbert.transformer.layer[layer_idx].attention

        with torch.no_grad():
            for proj_name, proj_layer in [('q_lin', attention.q_lin), ('k_lin', attention.k_lin), ('v_lin', attention.v_lin)]:
                weight=proj_layer.weight.data.view(NUM_HEADS, HEAD_DIM, HIDDEN_SIZE)
                bias=proj_layer.bias.data.view(NUM_HEADS, HEAD_DIM)

                weight[heads_to_prune_in_layer, :, :]=0.0
                bias[heads_to_prune_in_layer, :]=0.0

            out_weight=attention.out_lin.weight.data.view(HIDDEN_SIZE, NUM_HEADS, HEAD_DIM)
            out_weight[:, heads_to_prune_in_layer, :]=0.0

        pruning_stats['layer_stats'].append({
            'layer': layer_idx,
            'pruned': len(heads_to_prune_in_layer),
            'kept': NUM_HEADS-len(heads_to_prune_in_layer)
        })

    pruning_stats['total_heads_after']=pruning_stats['total_heads_before'] - pruning_stats['total_heads_pruned']
    return model, pruning_stats

def prune_ffn_weights(model, sparsity_ratio=0.2):
    ffn_weights={}
    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn
        ffn_weights[(layer_idx, 'lin1')]=ffn.lin1.weight
        ffn_weights[(layer_idx, 'lin2')]=ffn.lin2.weight

    all_weights=[]
    for weight in ffn_weights.values():
        all_weights.append(weight.abs().flatten())
    all_weights=torch.cat(all_weights)

    k=int(len(all_weights) * sparsity_ratio)
    if k==0:
        threshold=0.0
    else:
        threshold=torch.kthvalue(all_weights,k).values.item()

    thresholds={key: threshold for key in ffn_weights.keys()}

    total_pruned=0
    total_weights=0
    pruning_masks={}

    for layer_idx in range(NUM_LAYERS):
        ffn=model.distilbert.transformer.layer[layer_idx].ffn

        for lin_name, lin_layer in [('lin1', ffn.lin1), ('lin2', ffn.lin2)]:
            key = (layer_idx, lin_name)
            threshold = thresholds[key]

            # 生成mask：绝对值小于阈值的位置设为True
            mask = lin_layer.weight.abs() < threshold

            n_pruned = mask.sum().item()
            n_total = mask.numel()

            with torch.no_grad():
                lin_layer.weight.data[mask] = 0.0

            # 保存keep_mask（1=保留，0=剪枝）
            keep_mask = ~mask
            pruning_masks[key] = keep_mask.detach().cpu()

            total_pruned += n_pruned
            total_weights += n_total

    pruning_stats = {
            'total_pruned': total_pruned,
            'total_weights': total_weights,
            'actual_sparsity': total_pruned / total_weights
        }

    return pruning_stats, pruning_masks


def load_heads_importance(csv_path):
    print(f"[加载重要性] 文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # 按acc_drop_pct升序排序
    df_sorted = df.sort_values('acc_drop_pct', ascending=True)

    all_heads = []
    for i in range(len(df_sorted)):
        row = df_sorted.iloc[i]
        all_heads.append((int(row['layer']), int(row['head'])))

    print(f"  共{len(all_heads)}个注意力头")
    print(f"  最不重要的5个: {all_heads[:5]}")

    return all_heads

def iterative_prune_finetune(
        model,
        tokenizer,
        test_texts,
        test_labels,
        importance_csv,
        device,
        total_iterations=6,
        initial_accuracy=None,
        finetune_threshold=0.02,
        stop_threshold=0.1,
):
    print("\n" + "=" * 60)
    print("开始迭代剪枝")
    print("=" * 60)
    print(f"  总迭代次数: {total_iterations}")
    print(f"  微调阈值: {finetune_threshold:.0%}")
    print(f"  早停阈值: {stop_threshold:.0%}")

    all_heads_sorted=load_heads_importance(importance_csv)

    if initial_accuracy is None:
        initial_accuracy = evaluate_model(model, tokenizer, test_texts, test_labels, device)

    initial_params=count_parameters(model)
    initial_sparsity=calculate_sparsity(model)
    
    print(f"\n{'='*60}")
    print("[初始状态]")
    print(f"{'='*60}")
    print(f"  准确率: {initial_accuracy:.2f}%")
    print(f"  参数量: {initial_params:,}")
    print(f"  稀疏度: {initial_sparsity:.4f}")
    
    history=[{
        "iteration": 0,
        "accuracy": initial_accuracy,
        "params": initial_params,
        "sparsity": initial_sparsity,
        "pruning_type": "none",  # ← 改成字符串
        "pruning_ratio": 0.0,
        "finetuned": False
    }]

    current_pruning_masks={}
    heads_already_pruned=0

    for iteration in range(1, total_iterations+1):
        print(f"\n{'=' * 60}")
        print(f"迭代 {iteration}/{total_iterations}")
        print(f"{'=' * 60}")

        if iteration <= total_iterations //2 :
            pruning_type="attention"
            pruning_ratio=0.15+0.05* iteration

            n_heads_to_prune=int(NUM_LAYERS * NUM_HEADS * pruning_ratio) -heads_already_pruned
            heads_to_prune=all_heads_sorted[heads_already_pruned: heads_already_pruned + n_heads_to_prune]
            heads_already_pruned+=n_heads_to_prune

            print(f"[剪枝类型] 结构化剪枝（注意力头）")
            print(f"  目标比例: {pruning_ratio:.0%}")
            print(f"  本次剪枝: {len(heads_to_prune)} 个头")
            print(f"  累计剪枝: {heads_already_pruned}/{NUM_LAYERS * NUM_HEADS} 个头")

            model, attention_stats= prune_attention_heads(model, heads_to_prune, device)

        else:
            pruning_type="ffn"

            sparsity=0.15 + 0.05 * (iteration - total_iterations //2)
            print(f"[剪枝类型] 非结构化剪枝（FFN）")
            print(f"  目标稀疏度: {sparsity:.0%}")

            ffn_stats, new_masks = prune_ffn_weights(model, sparsity)
            current_pruning_masks = new_masks

            print(f"  实际稀疏度: {ffn_stats['actual_sparsity']:.2%}")

        current_accuracy=evaluate_model(model, tokenizer, test_texts, test_labels, device)
        current_params=count_parameters(model)
        current_sparsity=calculate_sparsity(model)

        accuracy_drop=history[-1]["accuracy"]- current_accuracy
        print(f"\n[剪枝后状态]")
        print(f"  准确率: {current_accuracy:.2f}% (下降 {accuracy_drop:.2f}%)")
        print(f"  参数量: {current_params:,}")
        print(f"  稀疏度: {current_sparsity:.4f}")

        if current_accuracy < initial_accuracy *(1- stop_threshold):
            print(f"\n[早停] 准确率下降超过{stop_threshold:.0%}，停止剪枝")
            break

        need_finetune=accuracy_drop > finetune_threshold

        if need_finetune:
            print(f"\n[触发微调] 性能下降{accuracy_drop:.2%} > {finetune_threshold:.0%}")
            if accuracy_drop > 0.05:
                epochs=3
                lr=1e-4
            else:
                epochs=2
                lr=1e-5

            finetuned_accuracy=finetune_recovery(
                model=model,
                tokenizer=tokenizer,
                pruning_masks=current_pruning_masks if pruning_type == "ffn" else {},
                device=device,
                epochs=epochs,
                lr=lr,
                n_samples=1000,
            )

            recovery=finetuned_accuracy - current_accuracy
            print(f"[微调效果] 准确率: {finetuned_accuracy:.2f}% (恢复 {recovery:.2f}%)")

            current_accuracy=finetuned_accuracy

        else:
            print(f"\n[跳过微调] 性能下降{accuracy_drop:.2%} <= {finetune_threshold:.0%}")

        history.append({
            "iteration": iteration,
            "accuracy": current_accuracy,
            "params": current_params,
            "sparsity": current_sparsity,
            "pruning_type": pruning_type,
            "pruning_ratio": pruning_ratio if pruning_type == "attention" else sparsity,
            "finetuned": need_finetune
        })

    print(f"\n{'=' * 60}")
    print("迭代剪枝完成")
    print(f"{'=' * 60}")
    print(f"  最终准确率: {history[-1]['accuracy']:.2f}%")
    print(f"  准确率损失: {initial_accuracy - history[-1]['accuracy']:.2f}%")
    print(f"  最终稀疏度: {history[-1]['sparsity']:.4f}")

    return history, model

def generate_analysis_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    iterations=[h["iteration"] for h in history]
    accuracies=[h["accuracy"] for h in history]
    sparsities=[h["sparsity"] for h in history]
    finetuned=[h["finetuned"] for h in history]

    # Plot 1: Accuracy vs Iteration
    plt.figure(figsize=(10,6))
    plt.plot(iterations, accuracies, marker='o', linewidth=2, markersize=8, color='blue', label="Accuracy")
    for i, ft in enumerate(finetuned):
        if ft:
            plt.scatter(iterations[i], accuracies[i], s=200, c='red', marker='*', edgecolors='black', linewidths=1.5, zorder=5, label='Finetuned' if i == next((j for j, x in enumerate(finetuned) if x), None) else '')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Iteration', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig1_path=os.path.join(output_dir, '1_accuracy_vs_iteration.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot 1 saved: {fig1_path}")

    # Plot 2: Accuracy & Sparsity vs Iteration (Dual Y-axis)
    fig, ax1=plt.subplots(figsize=(10,6))
    color='tab:blue'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12)
    ax1.plot(iterations, accuracies, marker='o', color=color, linewidth=2, markersize=8, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2=ax1.twinx()
    color='tab:red'
    ax2.set_ylabel('Sparsity', color=color, fontsize=12)
    ax2.plot(iterations, sparsities, marker='s', color=color, linewidth=2, label='Sparsity')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Accuracy & Sparsity vs Iteration', fontsize=14, fontweight='bold')
    fig.tight_layout()

    fig2_path = os.path.join(output_dir, '2_accuracy_sparsity_vs_iteration.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot 2 saved: {fig2_path}")

    # Plot 3: Accuracy vs Sparsity (Pareto Frontier)
    plt.figure(figsize=(10,6))
    scatter=plt.scatter(sparsities, accuracies, s=100, c=iterations, cmap='viridis', edgecolors='black', linewidths=1)
    plt.colorbar(scatter, label='Iteration')

    plt.plot(sparsities, accuracies, 'k--', alpha=0.3, linewidth=1)

    initial_acc=history[0]["accuracy"]
    best_idx=0
    best_score= -float('inf')

    for i, h in enumerate(history[1:], 1):
        acc_loss=initial_acc - h["accuracy"]
        if acc_loss <= 0.05 * initial_acc:
            score=h["sparsity"]
            if score > best_score:
                best_score=score
                best_idx=i

    if best_idx >0:
        best_point=history[best_idx]
        plt.scatter([best_point["sparsity"]], [best_point["accuracy"]], s=300, c='red', marker='*', edgecolors='black', linewidths=2, label=f'Best Point (Iter {best_idx})', zorder=5)
        plt.annotate(
            f'Best\nAcc: {best_point["accuracy"]:.2f}%\nSparsity: {best_point["sparsity"]:.4f}',
            xy=(best_point["sparsity"], best_point["accuracy"]),
            xytext=(20, 20),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red')
        )

        plt.legend(fontsize=10)

    plt.xlabel('Sparsity', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Sparsity (Pareto Frontier)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig3_path = os.path.join(output_dir, '3_accuracy_vs_sparsity.png')
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot 3 saved: {fig3_path}")

    return best_idx


def generate_text_report(history, best_idx, output_dir):
    initial = history[0]
    final = history[-1]

    # 构建报告内容
    report = [
        "=" * 70,
        "迭代剪枝分析报告",
        "=" * 70,
        "",
        "【1】总体效果",
        "-" * 70,
        f"  初始准确率: {initial['accuracy']:.2f}%",
        f"  最终准确率: {final['accuracy']:.2f}%",
        f"  准确率损失: {initial['accuracy'] - final['accuracy']:.2f}%",
        f"  损失比例: {(initial['accuracy'] - final['accuracy']) / initial['accuracy'] * 100:.2f}%",
        "",
        f"  初始稀疏度: {initial['sparsity']:.4f}",
        f"  最终稀疏度: {final['sparsity']:.4f}",
        f"  稀疏度提升: {(final['sparsity'] - initial['sparsity']):.4f}",
        "",
        f"  初始参数量: {initial['params']:,}",
        f"  最终参数量: {final['params']:,}",
        "",
        "【2】迭代详情",
        "-" * 70,
    ]

    for h in history:
        if h["iteration"] == 0:
            report.append(f"  初始状态: 准确率={h['accuracy']:.2f}%, 稀疏度={h['sparsity']:.4f}")
        else:
            ft_mark = " [✓微调]" if h["finetuned"] else ""
            pruning_info = f"剪枝{h['pruning_ratio']:.0%}" if h[
                                                                  "pruning_type"] == "attention" else f"稀疏度{h['pruning_ratio']:.0%}"
            report.append(
                f"  迭代{h['iteration']}: {h['pruning_type']:10s} | {pruning_info:10s} | "
                f"准确率={h['accuracy']:.2f}% | 稀疏度={h['sparsity']:.4f}{ft_mark}"
            )

    report.extend([
        "",
        "【3】最优权衡点",
        "-" * 70,
    ])

    if best_idx > 0:
        best = history[best_idx]
        report.extend([
            f"  最优迭代: 第{best_idx}次",
            f"  剪枝类型: {best['pruning_type']}",
            f"  准确率: {best['accuracy']:.2f}% (损失 {initial['accuracy'] - best['accuracy']:.2f}%)",
            f"  稀疏度: {best['sparsity']:.4f}",
            f"  说明: 在准确率下降不超过5%的前提下，达到最大稀疏度",
        ])
    else:
        report.append("  未找到最优点（所有迭代准确率下降超过5%）")

    report.extend([
        "",
        "【4】剪枝策略总结",
        "-" * 70,
        f"  前半段: 结构化剪枝（注意力头）",
        f"  后半段: 非结构化剪枝（FFN权重）",
        f"  微调次数: {sum(1 for h in history if h['finetuned'])} 次",
        "",
        "=" * 70,
    ])

    report_text = "\n".join(report)

    report_path = os.path.join(output_dir, "iterative_pruning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[Report Generation]")
    print(f"  ✓ Text report saved: {report_path}")

    print("\n" + report_text)

    return report_path

def main():
    parser=argparse.ArgumentParser(description="综合迭代剪枝")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--importance_csv", type=str, required=True, help="注意力头重要性CSV文件")
    parser.add_argument("--output_dir", type=str, default="./iterative_pruning_output", help="输出目录")
    parser.add_argument("--total_iterations", type=int, default=6, help="总迭代次数")
    parser.add_argument("--n_test_samples", type=int, default=500, help="测试样本数")
    parser.add_argument("--finetune_threshold", type=float, default=0.02, help="触发微调的准确率下降阈值")
    parser.add_argument("--stop_threshold", type=float, default=0.10, help="早停阈值（准确率下降）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args=parser.parse_args()

    print("\n" + "=" * 70)
    print("Iterative Pruning - Start")
    print("=" * 70)
    print(f"  Model Path: {args.ckpt_path}")
    print(f"  Importance CSV: {args.importance_csv}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Total Iterations: {args.total_iterations}")
    print(f"  Test Samples: {args.n_test_samples}")
    print("=" * 70)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, test_texts, test_labels=load_model(args.ckpt_path, n_test_samples=args.n_test_samples, device=args.device)

    history, final_model=iterative_prune_finetune(
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        test_labels=test_labels,
        importance_csv=args.importance_csv,
        device=args.device,
        total_iterations=args.total_iterations,
        finetune_threshold=args.finetune_threshold,
        stop_threshold=args.stop_threshold,
    )

    best_idx=generate_analysis_plots(history, args.output_dir)
    generate_text_report(history, best_idx, args.output_dir)

    model_save_path = os.path.join(args.output_dir, "final_pruned_model.pt")
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'history': history,
        'best_iteration': best_idx,
    }, model_save_path)
    print(f"  ✓ Model saved: {model_save_path}")

    history_path = os.path.join(args.output_dir, "history.json")
    # Convert numpy types to Python native types for JSON serialization
    history_serializable = []
    for h in history:
        h_copy = {}
        for key, value in h.items():
            # Convert numpy bool to Python bool
            if hasattr(value, 'item'):
                h_copy[key] = value.item()
            else:
                h_copy[key] = value
        history_serializable.append(h_copy)
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=2, ensure_ascii=False)
    print(f"  ✓ History saved: {history_path}")

    config_path = os.path.join(args.output_dir, "config.json")
    config = {
        "ckpt_path": args.ckpt_path,
        "importance_csv": args.importance_csv,
        "total_iterations": args.total_iterations,
        "n_test_samples": args.n_test_samples,
        "finetune_threshold": args.finetune_threshold,
        "stop_threshold": args.stop_threshold,
        "seed": args.seed,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Config saved: {config_path}")

    print("\n" + "=" * 70)
    print("✅ All Done!")
    print("=" * 70)
    print(f"\nOutput Files:")
    print(f"  - Plot: {args.output_dir}/1_accuracy_vs_iteration.png")
    print(f"  - Plot: {args.output_dir}/2_accuracy_sparsity_vs_iteration.png")
    print(f"  - Plot: {args.output_dir}/3_accuracy_vs_sparsity.png")
    print(f"  - Report: {args.output_dir}/iterative_pruning_report.txt")
    print(f"  - Model: {args.output_dir}/final_pruned_model.pt")
    print(f"  - History: {args.output_dir}/history.json")
    print(f"  - Config: {args.output_dir}/config.json")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()