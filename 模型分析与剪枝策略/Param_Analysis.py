import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import re

#plot setting
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#model name
MODEL_NAME="distilbert-base-uncased"

def load_model(ckpt_path):
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model=AutoAdapterModel.from_pretrained(MODEL_NAME)

    #load LoRA adapter
    adapter_path=os.path.join(ckpt_path, "lora_adapter")
    model.load_adapter(adapter_path, load_as="lora_adapter", set_active=True)
    model.set_active_adapters("lora_adapter")

    #load classification head
    head_path=os.path.join(ckpt_path, "imdb")
    model.load_head(head_path)
    model.active_head="imdb"

    model.eval()

    return model, tokenizer

def classify_param(name):
    name_lower = name.lower()
    if "embedding" in name_lower:
        return "Embedding"
    elif "lora" in name_lower:
        return "LoRA"
    elif "q_lin" in name_lower or "k_lin" in name_lower or "v_lin" in name_lower or "out_lin" in name_lower:
        return "Attention"
    elif "ffn" in name_lower or "lin1" in name_lower or "lin2" in name_lower:
        return "FFN"
    elif "layernorm" in name_lower or "layer_norm" in name_lower:
        return "LayerNorm"
    elif "heads" in name_lower:
        return "Classification Head"
    else:
        return "Other"

def get_layer_index(name):
    pattern=r'layer\.(\d+)'
    match=re.search(pattern,name)
    if match:
        return int(match.group(1))
    else:
        return -1

def analyze_parameters(model, sparsity_threshold=1e-3):
    stats_list=[]
    all_params=[]

    for name, param in model.named_parameters():
        data=param.detach().cpu().numpy()
        data_flat=data.flatten()

        all_params.extend(data_flat.tolist())

        count=param.numel()
        mean_val=float(data.mean())
        std_val=float(data.std())
        min_val=float(data.min())
        max_val=float(data.max())
        abs_mean=float(np.abs(data).mean())
        abs_max=float(np.abs(data).max())

        near_zero_count=np.sum(np.abs(data)<sparsity_threshold)
        sparsity=near_zero_count/count

        category=classify_param(name)
        layer_index=get_layer_index(name)

        stats={
            "name": name,
            "shape": str(list(param.shape)),
            "count": count,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "abs_mean": abs_mean,
            "abs_max": abs_max,
            "sparsity": sparsity,
            "category": category,
            "layer_index": layer_index,
        }
        stats_list.append(stats)

    return stats_list, np.array(all_params)

def plot_histogram(all_params, save_path):
    fig, axes=plt.subplots(1,2,figsize=(14,5))

    #ax1: parameter value distribution
    ax1=axes[0]
    ax1.hist(all_params, bins=100, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax1.set_xlabel("Parameters Value")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Global Parameter Distribution")
    ax1.legend()

    #ax2: abs distribution
    ax2=axes[1]
    abs_params=np.abs(all_params)
    ax2.hist(abs_params, bins=100, color="coral", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("|Parameter Value|")
    ax2.set_ylabel("Frequency(log scale)")
    ax2.set_title("Absolute Parameter Distribution")
    ax2.set_yscale("log")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(stats_list, save_path):
    plot_data = []

    for stats in stats_list:
        plot_data.append({
            "Category": stats["category"],
            "Abs Mean": stats["abs_mean"]
        })
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(x="Category", y="Abs Mean", data=df, ax=ax, palette="Set2")

    ax.set_xlabel("Parameter Category")
    ax.set_ylabel("Absolute Mean Value")
    ax.set_title("Parameter Distribution by Category")

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sparsity_bar(stats_list, save_path):
    category_sparsity = defaultdict(list)

    for stats in stats_list:
        category = stats["category"]
        sparsity = stats["sparsity"]
        category_sparsity[category].append(sparsity)

    categories = []
    avg_sparsities = []

    for category, sparsity_list in sorted(category_sparsity.items()):
        categories.append(category)
        avg_sparsity = np.mean(sparsity_list) * 100
        avg_sparsities.append(avg_sparsity)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(categories, avg_sparsities, color='teal', alpha=0.8, edgecolor='black')

    for bar, val in zip(bars, avg_sparsities):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height() + 0.3
        ax.text(x, y, f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel("Parameter Category")
    ax.set_ylabel("Average Sparsity (%)")
    ax.set_title("Sparsity by Category (|w| < 0.001)")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pie(stats_list, save_path):
    category_count = defaultdict(int)

    for stats in stats_list:
        category = stats["category"]
        count = stats["count"]
        category_count[category] += count

    labels = []
    sizes = []
    total = sum(category_count.values())  # 总参数量

    for category, count in sorted(category_count.items(), key=lambda x: -x[1]):
        if count / total > 0.01:
            label = f"{category}\n({count / 1e6:.2f}M)"
            labels.append(label)
            sizes.append(count)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75
    )

    ax.set_title(f"Parameter Distribution by Category\n(Total: {total / 1e6:.2f}M parameters)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_heatmap(stats_list, save_path):
    num_layers=6
    categories=["Attention", "FFN", "LayerNorm", "LoRA"]

    matrix=np.zeros((num_layers, len(categories)))
    count_matrix=np.zeros((num_layers, len(categories)))

    for stats in stats_list:
        layer_idx=stats["layer_index"]
        category=stats["category"]
        abs_mean=stats["abs_mean"]

        if layer_idx >=0 and category in categories:
            row=layer_idx
            col=categories.index(category)
            matrix[row, col]+=abs_mean
            count_matrix[row, col]+=1
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix=np.divide(matrix, count_matrix)
        matrix=np.nan_to_num(matrix)

    if matrix.max()>0:
        matrix=matrix/matrix.max()

    fig, ax=plt.subplots(figsize=(10, 6))

    sns.heatmap(
        matrix,
        xticklabels=categories,
        yticklabels=[f"Layer {i} "for i in range(num_layers)],
        annot = True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax
    )

    ax.set_title("Normalized Abs Mean by Layer and Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Layer")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_variance_analysis(stats_list, save_path):
    """
    绘制各类别的方差分析图
    分析参数方差分布，识别稳定性和异常性
    """
    category_stats = defaultdict(lambda: {"mean_list": [], "std_list": []})
    
    for stats in stats_list:
        cat = stats["category"]
        category_stats[cat]["mean_list"].append(stats["mean"])
        category_stats[cat]["std_list"].append(stats["std"])
    
    categories = []
    avg_std = []
    max_std = []
    min_std = []
    
    for cat, data in sorted(category_stats.items()):
        categories.append(cat)
        avg_std.append(np.mean(data["std_list"]))
        max_std.append(np.max(data["std_list"]))
        min_std.append(np.min(data["std_list"]))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.6
    
    # 绘制柱状图（平均标准差）
    bars = ax.bar(x, avg_std, width, color='purple', alpha=0.7, 
                   edgecolor='black', label='Average Std')
    
    # 添加误差线（显示范围）
    errors = [(avg - min_v, max_v - avg) for avg, min_v, max_v in 
              zip(avg_std, min_std, max_std)]
    yerr_lower = [e[0] for e in errors]
    yerr_upper = [e[1] for e in errors]
    ax.errorbar(x, avg_std, yerr=[yerr_lower, yerr_upper], 
                fmt='none', ecolor='black', capsize=5, alpha=0.5)
    
    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, avg_std)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Parameter Category", fontsize=11)
    ax.set_ylabel("Standard Deviation", fontsize=11)
    ax.set_title("Parameter Variance Analysis by Category\n(Higher variance indicates more diverse parameter values)", 
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(stats_list, all_params, output_dir):
    report_path = os.path.join(output_dir, "param_analysis_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("=" * 60 + "\n")
        f.write("模型参数分布分析报告\n")
        f.write("MODEL PARAMETER DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # ===== 1. 全局统计 =====
        f.write("1. 全局统计 (Global Statistics)\n")
        f.write("-" * 40 + "\n")

        total_params = len(all_params)
        f.write(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)\n")
        f.write(f"全局均值: {np.mean(all_params):.6f}\n")
        f.write(f"全局标准差: {np.std(all_params):.6f}\n")
        f.write(f"最小值: {np.min(all_params):.6f}\n")
        f.write(f"最大值: {np.max(all_params):.6f}\n")
        f.write(f"绝对值均值: {np.mean(np.abs(all_params)):.6f}\n")

        # 全局稀疏性
        sparsity_global = np.sum(np.abs(all_params) < 1e-3) / total_params
        f.write(f"全局稀疏性 (|w|<0.001): {sparsity_global * 100:.2f}%\n\n")

        # ===== 2. 按类别统计 =====
        f.write("2. 按类别统计 (Statistics by Category)\n")
        f.write("-" * 40 + "\n")

        # 汇总每个类别
        category_stats = defaultdict(lambda: {"count": 0, "sparsity_list": []})

        for stats in stats_list:
            cat = stats["category"]
            category_stats[cat]["count"] += stats["count"]
            category_stats[cat]["sparsity_list"].append(stats["sparsity"])

        # 按参数量排序输出
        for cat, info in sorted(category_stats.items(), key=lambda x: -x[1]["count"]):
            pct = info["count"] / total_params * 100
            avg_sparsity = np.mean(info["sparsity_list"]) * 100
            f.write(f"\n{cat}:\n")
            f.write(f"  参数量: {info['count']:,} ({pct:.2f}%)\n")
            f.write(f"  平均稀疏性: {avg_sparsity:.2f}%\n")

        # ===== 3. 剪枝建议 =====
        f.write("\n\n3. 剪枝建议 (Pruning Recommendations)\n")
        f.write("-" * 40 + "\n")

        # 找出稀疏性最高的类别
        sparsity_ranking = []
        for cat, info in category_stats.items():
            avg_sp = np.mean(info["sparsity_list"])
            sparsity_ranking.append((cat, avg_sp, info["count"]))

        sparsity_ranking.sort(key=lambda x: -x[1])  # 按稀疏性降序

        f.write("\n稀疏性排名（从高到低）:\n")
        for rank, (cat, sp, count) in enumerate(sparsity_ranking, 1):
            f.write(f"  {rank}. {cat}: {sp * 100:.2f}% (参数量: {count:,})\n")

        f.write("\n建议:\n")
        f.write("  - FFN 层通常稀疏性较高，适合做非结构化剪枝\n")
        f.write("  - Attention 层可考虑结构化剪枝（移除整个 head）\n")
        f.write("  - Embedding 层参数量大但稀疏性低，不建议剪枝\n")
        f.write("  - LoRA/LayerNorm 参数量小，保持不变\n")

        f.write("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="模型参数分布分析 (Model Parameter Distribution Analysis)"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_output",
        help="输出目录（默认: ./analysis_output）"
    )
    parser.add_argument(
        "--sparsity_threshold",
        type=float,
        default=1e-3,
        help="稀疏性阈值（默认: 0.001）"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("模型参数分布分析 (Model Parameter Distribution Analysis)")
    print("=" * 60)
    print(f"模型路径: {args.ckpt_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"稀疏性阈值: {args.sparsity_threshold}")
    print()

    model, tokenizer = load_model(args.ckpt_path)

    stats_list, all_params = analyze_parameters(model, args.sparsity_threshold)

    df = pd.DataFrame(stats_list)
    csv_path = os.path.join(args.output_dir, "param_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"[已保存] {csv_path}")

    plot_histogram(
        all_params,
        os.path.join(args.output_dir, "param_histogram.png")
    )

    plot_boxplot(
        stats_list,
        os.path.join(args.output_dir, "param_boxplot.png")
    )

    plot_sparsity_bar(
        stats_list,
        os.path.join(args.output_dir, "sparsity_bar.png")
    )

    plot_pie(
        stats_list,
        os.path.join(args.output_dir, "param_pie.png")
    )

    plot_heatmap(
        stats_list,
        os.path.join(args.output_dir, "layer_heatmap.png")
    )

    plot_variance_analysis(
        stats_list,
        os.path.join(args.output_dir, "variance_analysis.png")
    )

    generate_report(stats_list, all_params, args.output_dir)

    print("=" * 60)
    print("分析结果总结:")
    print("-" * 40)

    total = len(all_params)
    print(f"总参数量: {total:,} ({total / 1e6:.2f}M)")

    sparsity = np.sum(np.abs(all_params) < args.sparsity_threshold) / total
    print(f"全局稀疏性: {sparsity * 100:.2f}%")

    print("\n各类别参数量:")
    category_count = defaultdict(int)
    for s in stats_list:
        category_count[s["category"]] += s["count"]

    for cat, count in sorted(category_count.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")

    print("=" * 60)
    print(f"\n所有结果已保存到: {args.output_dir}")
    print("  - param_stats.csv (统计数据)")
    print("  - param_histogram.png (直方图)")
    print("  - param_boxplot.png (箱线图)")
    print("  - sparsity_bar.png (稀疏性柱状图)")
    print("  - param_pie.png (饼图)")
    print("  - layer_heatmap.png (热力图)")
    print("  - variance_analysis.png (方差分析图)")
    print("  - param_analysis_report.txt (分析报告)")

if __name__ == "__main__":
    main()