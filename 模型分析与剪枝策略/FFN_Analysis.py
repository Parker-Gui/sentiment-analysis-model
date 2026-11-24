# -*- coding: utf-8 -*-


import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

# ========== 设置环境 ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from adapters import AutoAdapterModel



def load_model(ckpt_path: str, device: str = "cuda"):

    print(f"\n加载模型: {ckpt_path}")
    
    # 加载基础模型
    model = AutoAdapterModel.from_pretrained("distilbert-base-uncased")
    
    # 加载 LoRA adapter
    adapter_path = os.path.join(ckpt_path, "lora_adapter")
    model.load_adapter(adapter_path, load_as="lora_adapter", set_active=True)
    model.set_active_adapters("lora_adapter")
    
    # 加载分类头
    head_path = os.path.join(ckpt_path, "imdb")
    model.load_head(head_path)
    model.active_head = "imdb"
    
    # 移动到设备
    model = model.to(device)
    model.eval()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    print(f"  模型加载完成！设备: {device}")
    return model, tokenizer


# ============================================================
# 第二部分：加载测试数据
# ============================================================

def load_test_data(n_samples=500):
    """
    加载 IMDB 测试数据
    """
    print("  加载 IMDB 数据集...")
    
    # 方法1：尝试使用 ModelScope
    try:
        print("  尝试从 ModelScope 加载...")
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load('imdb', split='test')
        texts = [item['text'] for item in ds]
        labels = [item['label'] for item in ds]
        
        import random
        random.seed(42)
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        combined = combined[:n_samples]
        texts, labels = zip(*combined)
        
        class SimpleDataset:
            def __init__(self, texts, labels):
                self.data = {"text": list(texts), "label": list(labels)}
            def __len__(self):
                return len(self.data["text"])
            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return {"text": self.data["text"][idx], "label": self.data["label"][idx]}
                return {"text": self.data["text"][idx], "label": self.data["label"][idx]}
        
        dataset = SimpleDataset(texts, labels)
        print(f"  成功加载 {len(dataset)} 条数据")
        return dataset
    except Exception as e:
        print(f"  ModelScope 加载失败: {e}")
    
    # 方法2：尝试 HuggingFace
    try:
        print("  尝试从 HuggingFace 加载...")
        from datasets import load_dataset
        dataset = load_dataset("imdb", split="test", trust_remote_code=True)
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        print(f"  成功加载 {len(dataset)} 条数据")
        return dataset
    except Exception as e:
        print(f"  HuggingFace 加载失败: {e}")
    
    # 方法3：从 Stanford 下载
    try:
        print("  尝试从 Stanford 下载...")
        import urllib.request
        import tarfile
        import tempfile
        
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "imdb.tar.gz")
            urllib.request.urlretrieve(url, tar_path)
            
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)
            
            texts, labels = [], []
            test_dir = os.path.join(tmpdir, "aclImdb", "test")
            
            for label, folder in [(1, "pos"), (0, "neg")]:
                folder_path = os.path.join(test_dir, folder)
                for fname in os.listdir(folder_path)[:n_samples//2]:
                    with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(label)
            
            import random
            random.seed(42)
            combined = list(zip(texts, labels))
            random.shuffle(combined)
            texts, labels = zip(*combined)
            
            class SimpleDataset:
                def __init__(self, texts, labels):
                    self.data = {"text": list(texts), "label": list(labels)}
                def __len__(self):
                    return len(self.data["text"])
                def __getitem__(self, idx):
                    if isinstance(idx, slice):
                        return {"text": self.data["text"][idx], "label": self.data["label"][idx]}
                    return {"text": self.data["text"][idx], "label": self.data["label"][idx]}
            
            dataset = SimpleDataset(texts, labels)
            print(f"  成功加载 {len(dataset)} 条数据")
            return dataset
    except Exception as e:
        print(f"  Stanford 下载失败: {e}")
    
    raise RuntimeError("无法加载 IMDB 数据集")


# ============================================================
# 第三部分：FFN 权重范数分析
# ============================================================

def analyze_ffn_weight_norm(model):
    """
    分析 FFN 层每个神经元的权重范数
    
    原理:
        FFN 结构: x -> Linear1 -> GELU -> Linear2 -> out
        Linear1: (768, 3072) - 每列对应一个中间神经元的输入权重
        Linear2: (3072, 768) - 每行对应一个中间神经元的输出权重
        
        神经元重要性 = ||W1的列|| * ||W2的行||
        
    返回:
        importance_dict: {layer_idx: [3072个神经元的重要性分数]}
    """
    print("\n" + "="*60)
    print("FFN 权重范数分析")
    print("="*60)
    
    importance_dict = {}
    
    # DistilBERT 有 6 层 Transformer
    n_layers = 6
    
    for layer_idx in range(n_layers):
        # 获取 FFN 的两个线性层权重
        # DistilBERT 中 FFN 的命名: distilbert.transformer.layer.{i}.ffn.lin1/lin2
        try:
            lin1_weight = model.distilbert.transformer.layer[layer_idx].ffn.lin1.weight.data  # (3072, 768)
            lin2_weight = model.distilbert.transformer.layer[layer_idx].ffn.lin2.weight.data  # (768, 3072)
        except AttributeError:
            print(f"  Layer {layer_idx}: 无法获取 FFN 权重，跳过")
            continue
        
        # 计算每个神经元的重要性
        # lin1_weight: (3072, 768) - 第 i 行是神经元 i 的输入权重
        # lin2_weight: (768, 3072) - 第 i 列是神经元 i 的输出权重
        
        n_neurons = lin1_weight.shape[0]  # 3072
        neuron_importance = []
        
        for neuron_idx in range(n_neurons):
            # 输入权重的 L2 范数
            input_norm = torch.norm(lin1_weight[neuron_idx, :], p=2).item()
            # 输出权重的 L2 范数
            output_norm = torch.norm(lin2_weight[:, neuron_idx], p=2).item()
            # 综合重要性 = 输入范数 × 输出范数
            importance = input_norm * output_norm
            neuron_importance.append(importance)
        
        importance_dict[layer_idx] = neuron_importance
        
        # 统计信息
        imp_array = np.array(neuron_importance)
        print(f"  Layer {layer_idx}: mean={imp_array.mean():.4f}, "
              f"std={imp_array.std():.4f}, min={imp_array.min():.4f}, max={imp_array.max():.4f}")
    
    return importance_dict


# ============================================================
# 第四部分：FFN 激活值分析
# ============================================================

def analyze_ffn_activation(model, tokenizer, dataset, device, n_samples=200):
    """
    分析 FFN 中间层神经元的平均激活值
    
    原理:
        通过 hook 捕获 FFN 中间层（GELU 之后）的激活值
        激活值越大，说明神经元在处理数据时越活跃
        
    参数:
        model: 模型
        tokenizer: 分词器
        dataset: 测试数据集
        device: 设备
        n_samples: 用于统计的样本数
        
    返回:
        activation_dict: {layer_idx: [3072个神经元的平均激活值]}
    """
    print("\n" + "="*60)
    print("FFN 激活值分析")
    print("="*60)
    
    n_layers = 6
    n_neurons = 3072
    
    # 存储每个神经元的激活值累加
    activation_sum = {i: np.zeros(n_neurons) for i in range(n_layers)}
    activation_count = 0
    
    # 存储 hook 捕获的激活值
    activations = {}
    
    def make_hook(layer_idx):
        """创建捕获激活值的 hook"""
        def hook(module, input, output):
            # output 形状: (batch_size, seq_len, 3072)
            # 我们取所有位置的平均激活值
            act = output.detach().cpu().numpy()  # (batch, seq, 3072)
            act = np.abs(act).mean(axis=(0, 1))  # (3072,) 取绝对值的平均
            activations[layer_idx] = act
        return hook
    
    # 注册 hooks
    hooks = []
    for layer_idx in range(n_layers):
        # Hook 放在 lin1 之后（即 GELU 激活之前的线性变换输出）
        # 但我们更关心 GELU 之后的激活，所以 hook 放在整个 ffn 模块
        # DistilBERT 的 FFN 输出是经过 lin2 之后的，我们需要 hook lin1 的输出
        layer = model.distilbert.transformer.layer[layer_idx]
        hook = layer.ffn.lin1.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    # 前向传播收集激活值
    model.eval()
    n_samples = min(n_samples, len(dataset))
    
    print(f"  收集 {n_samples} 个样本的激活值...")
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="  分析激活值"):
            text = dataset[i]["text"]
            
            # 编码
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(device)
            
            # 前向传播（会触发 hooks）
            _ = model(**inputs)
            
            # 累加激活值
            for layer_idx in range(n_layers):
                if layer_idx in activations:
                    activation_sum[layer_idx] += activations[layer_idx]
            
            activation_count += 1
            activations.clear()
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    # 计算平均激活值
    activation_dict = {}
    for layer_idx in range(n_layers):
        avg_activation = activation_sum[layer_idx] / activation_count
        activation_dict[layer_idx] = avg_activation.tolist()
        
        print(f"  Layer {layer_idx}: mean={avg_activation.mean():.4f}, "
              f"std={avg_activation.std():.4f}, min={avg_activation.min():.4f}, max={avg_activation.max():.4f}")
    
    return activation_dict


# ============================================================
# 第五部分：综合重要性评分
# ============================================================

def compute_combined_importance(weight_importance, activation_importance, alpha=0.5):
    """
    综合权重范数和激活值，计算最终重要性评分
    
    参数:
        weight_importance: 权重范数重要性 {layer: [scores]}
        activation_importance: 激活值重要性 {layer: [scores]}
        alpha: 权重范数的权重（0-1之间），激活值权重为 1-alpha
        
    返回:
        combined: {layer: [综合分数]}
    """
    print("\n" + "="*60)
    print(f"综合重要性评分 (权重范数:{alpha}, 激活值:{1-alpha})")
    print("="*60)
    
    combined = {}
    
    for layer_idx in weight_importance:
        # 归一化到 [0, 1]
        w_scores = np.array(weight_importance[layer_idx])
        a_scores = np.array(activation_importance[layer_idx])
        
        # Min-Max 归一化
        w_norm = (w_scores - w_scores.min()) / (w_scores.max() - w_scores.min() + 1e-8)
        a_norm = (a_scores - a_scores.min()) / (a_scores.max() - a_scores.min() + 1e-8)
        
        # 加权组合
        combined_scores = alpha * w_norm + (1 - alpha) * a_norm
        combined[layer_idx] = combined_scores.tolist()
        
        print(f"  Layer {layer_idx}: mean={combined_scores.mean():.4f}, "
              f"std={combined_scores.std():.4f}")
    
    return combined


# ============================================================
# 第六部分：识别可剪枝神经元
# ============================================================

def identify_prunable_neurons(combined_importance, threshold_percentile=20):
    """
    识别可剪枝的神经元（重要性最低的一部分）
    
    参数:
        combined_importance: 综合重要性分数
        threshold_percentile: 阈值百分位（低于此百分位的视为可剪枝）
        
    返回:
        prunable_dict: {layer: [可剪枝的神经元索引]}
        stats: 统计信息
    """
    print("\n" + "="*60)
    print(f"识别可剪枝神经元 (阈值: 最低 {threshold_percentile}%)")
    print("="*60)
    
    prunable_dict = {}
    all_scores = []
    
    # 收集所有分数
    for layer_idx, scores in combined_importance.items():
        all_scores.extend(scores)
    
    # 计算全局阈值
    global_threshold = np.percentile(all_scores, threshold_percentile)
    print(f"  全局阈值: {global_threshold:.4f}")
    
    total_prunable = 0
    total_neurons = 0
    
    for layer_idx, scores in combined_importance.items():
        scores = np.array(scores)
        prunable_indices = np.where(scores < global_threshold)[0].tolist()
        prunable_dict[layer_idx] = prunable_indices
        
        total_prunable += len(prunable_indices)
        total_neurons += len(scores)
        
        print(f"  Layer {layer_idx}: {len(prunable_indices)}/{len(scores)} "
              f"神经元可剪枝 ({100*len(prunable_indices)/len(scores):.1f}%)")
    
    stats = {
        "threshold": global_threshold,
        "total_prunable": total_prunable,
        "total_neurons": total_neurons,
        "prunable_ratio": total_prunable / total_neurons
    }
    
    print(f"\n  总计: {total_prunable}/{total_neurons} 神经元可剪枝 "
          f"({100*stats['prunable_ratio']:.1f}%)")
    
    return prunable_dict, stats


# ============================================================
# 第七部分：可视化
# ============================================================

def plot_importance_heatmap(importance_dict, output_dir, filename, title):
    """
    绘制重要性热力图
    
    每行是一层，每列是神经元（由于 3072 太多，我们采样显示）
    """
    print(f"\n绘制热力图: {title}")
    
    n_layers = len(importance_dict)
    n_neurons = len(importance_dict[0])
    
    # 采样显示（3072 太多，采样 100 个代表性神经元）
    sample_indices = np.linspace(0, n_neurons-1, 100, dtype=int)
    
    # 构建热力图数据
    heatmap_data = np.zeros((n_layers, len(sample_indices)))
    for layer_idx in range(n_layers):
        scores = np.array(importance_dict[layer_idx])
        heatmap_data[layer_idx, :] = scores[sample_indices]
    
    # 绘图
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        xticklabels=50,  # 每 50 个显示一个标签
        yticklabels=[f"Layer {i}" for i in range(n_layers)],
        cbar_kws={"label": "Importance Score"}
    )
    plt.xlabel("Neuron Index (sampled)")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [已保存] {save_path}")


def plot_layer_distribution(importance_dict, output_dir, filename, title):
    """
    绘制每层神经元重要性的分布（箱线图）
    """
    print(f"\n绘制分布图: {title}")
    
    # 准备数据
    data = []
    for layer_idx, scores in importance_dict.items():
        for score in scores:
            data.append({"Layer": f"Layer {layer_idx}", "Importance": score})
    
    df = pd.DataFrame(data)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Layer", y="Importance", data=df, palette="Set2")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [已保存] {save_path}")


def plot_importance_histogram(importance_dict, output_dir, filename):
    """
    绘制全局重要性分布直方图
    """
    print(f"\n绘制直方图...")
    
    # 收集所有分数
    all_scores = []
    for scores in importance_dict.values():
        all_scores.extend(scores)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Importance Score")
    plt.ylabel("Count")
    plt.title("FFN Neuron Importance Distribution")
    
    # 添加统计信息
    mean_val = np.mean(all_scores)
    median_val = np.median(all_scores)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.4f}')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [已保存] {save_path}")


def plot_prunable_summary(prunable_dict, output_dir):
    """
    绘制可剪枝神经元数量的柱状图
    """
    print(f"\n绘制可剪枝神经元统计图...")
    
    layers = [f"Layer {i}" for i in prunable_dict.keys()]
    prunable_counts = [len(indices) for indices in prunable_dict.values()]
    total_counts = [3072] * len(layers)  # 每层 3072 个神经元
    
    x = np.arange(len(layers))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, total_counts, width, label='Total Neurons', color='lightgray')
    bars2 = ax.bar(x + width/2, prunable_counts, width, label='Prunable', color='coral')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Prunable FFN Neurons per Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    
    # 添加数值标签
    for bar, count in zip(bars2, prunable_counts):
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "prunable_neurons.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [已保存] {save_path}")


# ============================================================
# 第八部分：生成报告
# ============================================================

def generate_report(weight_importance, activation_importance, combined_importance,
                    prunable_dict, stats, output_dir):
    """
    生成详细的分析报告
    """
    print("\n生成分析报告...")
    
    report_lines = [
        "=" * 60,
        "FFN 层重要性分析报告",
        "=" * 60,
        "",
        "1. 分析概述",
        "-" * 40,
        f"   - 分析层数: 6 (DistilBERT)",
        f"   - 每层神经元数: 3072",
        f"   - 总神经元数: 18432",
        f"   - 分析方法: 权重范数 + 激活值统计",
        "",
        "2. 权重范数分析结果",
        "-" * 40,
    ]
    
    for layer_idx, scores in weight_importance.items():
        scores = np.array(scores)
        report_lines.append(
            f"   Layer {layer_idx}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"min={scores.min():.4f}, max={scores.max():.4f}"
        )
    
    report_lines.extend([
        "",
        "3. 激活值分析结果",
        "-" * 40,
    ])
    
    for layer_idx, scores in activation_importance.items():
        scores = np.array(scores)
        report_lines.append(
            f"   Layer {layer_idx}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"min={scores.min():.4f}, max={scores.max():.4f}"
        )
    
    report_lines.extend([
        "",
        "4. 综合重要性分析",
        "-" * 40,
    ])
    
    for layer_idx, scores in combined_importance.items():
        scores = np.array(scores)
        report_lines.append(
            f"   Layer {layer_idx}: mean={scores.mean():.4f}, std={scores.std():.4f}"
        )
    
    report_lines.extend([
        "",
        "5. 可剪枝神经元统计",
        "-" * 40,
        f"   - 阈值: {stats['threshold']:.4f} (最低 20% 分位)",
        f"   - 总可剪枝: {stats['total_prunable']} / {stats['total_neurons']} "
        f"({100*stats['prunable_ratio']:.1f}%)",
        "",
    ])
    
    for layer_idx, indices in prunable_dict.items():
        report_lines.append(
            f"   Layer {layer_idx}: {len(indices)} 个神经元可剪枝 "
            f"({100*len(indices)/3072:.1f}%)"
        )
    
    report_lines.extend([
        "",
        "6. 结论与建议",
        "-" * 40,
        "   - 权重范数小且激活值低的神经元贡献度较低",
        "   - 建议优先剪枝综合得分最低的 20% 神经元",
        "   - 剪枝后需要微调以恢复性能",
        "",
        "=" * 60,
    ])
    
    # 保存报告
    report_path = os.path.join(output_dir, "ffn_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  [已保存] {report_path}")
    
    return "\n".join(report_lines)


def save_importance_data(weight_importance, activation_importance, combined_importance,
                         prunable_dict, output_dir):
    """
    保存重要性数据到 CSV
    """
    print("\n保存数据到 CSV...")
    
    rows = []
    for layer_idx in weight_importance.keys():
        for neuron_idx in range(len(weight_importance[layer_idx])):
            rows.append({
                "layer": layer_idx,
                "neuron": neuron_idx,
                "weight_importance": weight_importance[layer_idx][neuron_idx],
                "activation_importance": activation_importance[layer_idx][neuron_idx],
                "combined_importance": combined_importance[layer_idx][neuron_idx],
                "prunable": neuron_idx in prunable_dict[layer_idx]
            })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "ffn_importance.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [已保存] {csv_path}")
    
    return df



def main():
    parser = argparse.ArgumentParser(description="FFN层重要性分析")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="./ffn_analysis_output", help="输出目录")
    parser.add_argument("--n_samples", type=int, default=200, help="用于激活值分析的样本数")
    parser.add_argument("--alpha", type=float, default=0.5, help="权重范数在综合评分中的权重")
    parser.add_argument("--prune_percentile", type=int, default=20, help="可剪枝阈值百分位")
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印配置
    print("\n" + "="*60)
    print("FFN 层重要性分析 (FFN Layer Importance Analysis)")
    print("="*60)
    print(f"模型路径: {args.ckpt_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"激活值分析样本数: {args.n_samples}")
    print(f"综合评分权重: 权重范数={args.alpha}, 激活值={1-args.alpha}")
    print(f"剪枝阈值: 最低 {args.prune_percentile}%")
    print(f"设备: {device}")
    
    # 1. 加载模型
    model, tokenizer = load_model(args.ckpt_path, device)
    
    # 2. 加载数据
    print("\n加载测试数据...")
    dataset = load_test_data(args.n_samples)
    
    # 3. 权重范数分析
    weight_importance = analyze_ffn_weight_norm(model)
    
    # 4. 激活值分析
    activation_importance = analyze_ffn_activation(
        model, tokenizer, dataset, device, args.n_samples
    )
    
    # 5. 综合重要性评分
    combined_importance = compute_combined_importance(
        weight_importance, activation_importance, args.alpha
    )
    
    # 6. 识别可剪枝神经元
    prunable_dict, stats = identify_prunable_neurons(
        combined_importance, args.prune_percentile
    )
    
    # 7. 可视化
    print("\n" + "="*60)
    print("生成可视化图表")
    print("="*60)
    
    plot_importance_heatmap(
        weight_importance, args.output_dir,
        "weight_importance_heatmap.png", "FFN Weight Norm Importance"
    )
    
    plot_importance_heatmap(
        activation_importance, args.output_dir,
        "activation_importance_heatmap.png", "FFN Activation Importance"
    )
    
    plot_importance_heatmap(
        combined_importance, args.output_dir,
        "combined_importance_heatmap.png", "FFN Combined Importance"
    )
    
    plot_layer_distribution(
        combined_importance, args.output_dir,
        "importance_distribution.png", "FFN Neuron Importance Distribution by Layer"
    )
    
    plot_importance_histogram(combined_importance, args.output_dir, "importance_histogram.png")
    
    plot_prunable_summary(prunable_dict, args.output_dir)
    
    # 8. 保存数据和报告
    save_importance_data(
        weight_importance, activation_importance, combined_importance,
        prunable_dict, args.output_dir
    )
    
    report = generate_report(
        weight_importance, activation_importance, combined_importance,
        prunable_dict, stats, args.output_dir
    )
    
    # 9. 打印总结
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"总神经元数: 18432 (6 层 × 3072 神经元)")
    print(f"可剪枝神经元: {stats['total_prunable']} ({100*stats['prunable_ratio']:.1f}%)")
    print(f"\n所有结果已保存到: {args.output_dir}")
    print(f"  - ffn_importance.csv (数据)")
    print(f"  - weight_importance_heatmap.png (权重范数热力图)")
    print(f"  - activation_importance_heatmap.png (激活值热力图)")
    print(f"  - combined_importance_heatmap.png (综合重要性热力图)")
    print(f"  - importance_distribution.png (分布箱线图)")
    print(f"  - importance_histogram.png (直方图)")
    print(f"  - prunable_neurons.png (可剪枝统计)")
    print(f"  - ffn_analysis_report.txt (分析报告)")


if __name__ == "__main__":
    main()
