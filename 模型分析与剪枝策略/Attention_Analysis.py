import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from datasets import load_dataset


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL_NAME="distilbert-base-uncased"
NUM_LAYERS=6
NUM_HEADS=12
HEAD_DIM=64

#load model
def load_model(ckpt_path, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    # load LoRA adapter
    adapter_path = os.path.join(ckpt_path, "lora_adapter")
    model.load_adapter(adapter_path, load_as="lora_adapter", set_active=True)
    model.set_active_adapters("lora_adapter")

    head_path = os.path.join(ckpt_path, "imdb")
    model.load_head(head_path)
    model.active_head = "imdb"

    model.to(device)
    model.eval()

    print(f"  模型加载完成！设备: {device}")
    return model, tokenizer

def load_test_data(n_samples=500):
    """
    加载 IMDB 测试数据
    尝试多种方式加载真实数据集
    """
    print("  加载 IMDB 数据集...")
    
    # 方法1：尝试使用 ModelScope（阿里云镜像）
    try:
        print("  尝试从 ModelScope 加载...")
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load('imdb', split='test')
        # 转换格式
        texts = [item['text'] for item in ds]
        labels = [item['label'] for item in ds]
        
        # 打乱并选取
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
            def select(self, indices):
                return SimpleDataset([self.data["text"][i] for i in indices], 
                                    [self.data["label"][i] for i in indices])
        
        dataset = SimpleDataset(texts, labels)
        print(f"  成功从 ModelScope 加载 {len(dataset)} 条数据")
        return dataset
    except Exception as e:
        print(f"  ModelScope 加载失败: {e}")
    
    # 方法2：尝试从 HuggingFace 加载
    try:
        print("  尝试从 HuggingFace 加载...")
        dataset = load_dataset("imdb", split="test", trust_remote_code=True)
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        print(f"  成功从 HuggingFace 加载 {len(dataset)} 条数据")
        return dataset
    except Exception as e:
        print(f"  HuggingFace 加载失败: {e}")
    
    # 方法3：从 stanfordnlp 直接下载
    try:
        print("  尝试从 Stanford 直接下载...")
        import urllib.request
        import tarfile
        import tempfile
        
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "imdb.tar.gz")
            print("    下载中...")
            urllib.request.urlretrieve(url, tar_path)
            
            print("    解压中...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)
            
            # 读取测试数据
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
                def select(self, indices):
                    return SimpleDataset([self.data["text"][i] for i in indices],
                                        [self.data["label"][i] for i in indices])
            
            dataset = SimpleDataset(texts, labels)
            print(f"  成功从 Stanford 下载 {len(dataset)} 条数据")
            return dataset
    except Exception as e:
        print(f"  Stanford 下载失败: {e}")
    
    raise RuntimeError("无法加载 IMDB 数据集，请检查网络连接")

def evaluate_accuracy(model, tokenizer, dataset, device="cpu", batch_size=16):
    model.eval()
    correct=0
    total=0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch=dataset[i:i+batch_size]
            texts=batch["text"]
            labels=batch["label"]

            inputs=tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,\
                return_tensors="pt",
            ).to(device)

            outputs=model(**inputs)
            logits=outputs.logits

            predictions=logits.argmax(dim=-1).cpu().numpy()

            correct+=(predictions==np.array(labels)).sum()
            total+=len(labels)

    accuracy=correct/total
    return accuracy

def analyze_head_importance(model, tokenizer, dataset, device="cpu"):
    baseline_acc=evaluate_accuracy(model, tokenizer, dataset, device)
    importance_scores={}

    total_heads=NUM_LAYERS*NUM_HEADS

    with tqdm(total=total_heads, desc="分析进度") as pbar:
        for layer_idx in range(NUM_LAYERS):
            for head_idx in range(NUM_HEADS):
                head_mask=torch.ones(NUM_LAYERS, NUM_HEADS).to(device)
                head_mask[layer_idx, head_idx]=0

                masked_acc=evaluate_with_head_mask(model, tokenizer, dataset, head_mask, device)

                importance=baseline_acc-masked_acc
                importance_scores[(layer_idx, head_idx)]={
                    "importance": importance,
                    "masked_acc": masked_acc,
                    "acc_drop": importance*100
                }

                pbar.update(1)
                pbar.set_postfix({
                    "Layer": layer_idx,
                    "Head": head_idx,
                    "Drop": f"{importance*100:.2f}%"
                })
    return importance_scores, baseline_acc


def evaluate_with_head_mask(model, tokenizer, dataset, head_mask, device="cpu", batch_size=16):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            texts = batch["text"]
            labels = batch["label"]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs, head_mask=head_mask)
            logits = outputs.logits

            predictions = logits.argmax(dim=-1).cpu().numpy()
            correct += (predictions == np.array(labels)).sum()
            total += len(labels)

    return correct / total

def analyze_attention_entropy(model, tokenizer, dataset, device="cpu", n_samples=100):
    entropy_accumulator=defaultdict(list)
    model.eval()
    samples=dataset.select(range(min(n_samples, len(dataset))))

    with torch.no_grad():
        for sample in tqdm(samples, desc="计算注意力熵"):
            text=sample["text"]
            inputs=tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            outputs=model(**inputs, output_attentions=True)
            attentions=outputs.attentions

            for layer_idx, attn in enumerate(attentions):
                attn=attn.squeeze(0)

                for head_idx in range(NUM_HEADS):
                    head_attn=attn[head_idx]
                    entropy=compute_entropy(head_attn)
                    entropy_accumulator[(layer_idx, head_idx)].append(entropy)

    entropy_scores={}
    for key, values in entropy_accumulator.items():
        entropy_scores[key]=np.mean(values)

    return entropy_scores

def compute_entropy(attention_matrix):
    attn=attention_matrix.float()
    attn=attn+1e-10
    entropy_per_row=-torch.sum(attn*torch.log(attn), dim=-1)
    avg_entropy=entropy_per_row.mean().item()
    return avg_entropy


def plot_importance_heatmap(importance_scores, save_path, title="Head Importance"):
    """
    画 Attention Head 重要性热力图

    横轴：Head 索引 (0-11)
    纵轴：Layer 索引 (0-5)
    颜色：重要性分数（越红越重要）
    """
    # 创建矩阵
    matrix = np.zeros((NUM_LAYERS, NUM_HEADS))

    for (layer_idx, head_idx), info in importance_scores.items():
        # 如果 info 是字典，取 importance 值
        if isinstance(info, dict):
            matrix[layer_idx, head_idx] = info["importance"]
        else:
            matrix[layer_idx, head_idx] = info

    # 画热力图
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        matrix * 100,  # 转为百分比
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",  # 黄-橙-红
        xticklabels=[f"Head {i}" for i in range(NUM_HEADS)],
        yticklabels=[f"Layer {i}" for i in range(NUM_LAYERS)],
        ax=ax,
        cbar_kws={"label": "Accuracy Drop (%)"}
    )

    ax.set_title(title)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[已保存] {save_path}")


def plot_entropy_heatmap(entropy_scores, save_path):
    """
    画注意力熵热力图

    颜色：熵值（越蓝熵越低=越集中=可能越重要）
    """
    matrix = np.zeros((NUM_LAYERS, NUM_HEADS))

    for (layer_idx, head_idx), entropy in entropy_scores.items():
        matrix[layer_idx, head_idx] = entropy

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu_r",  # 反向：蓝色=低熵=集中
        xticklabels=[f"Head {i}" for i in range(NUM_HEADS)],
        yticklabels=[f"Layer {i}" for i in range(NUM_LAYERS)],
        ax=ax,
        cbar_kws={"label": "Entropy"}
    )

    ax.set_title("Attention Entropy by Head (Lower = More Focused)")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[已保存] {save_path}")


def plot_importance_ranking(importance_scores, save_path, top_n=20):
    """
    画 Head 重要性排名柱状图
    显示最重要和最不重要的 N 个 Head
    """
    # 转换为列表并排序
    items = []
    for (layer_idx, head_idx), info in importance_scores.items():
        if isinstance(info, dict):
            importance = info["importance"]
        else:
            importance = info
        items.append({
            "head": f"L{layer_idx}H{head_idx}",
            "layer": layer_idx,
            "head_idx": head_idx,
            "importance": importance * 100  # 转为百分比
        })

    # 按重要性排序
    items.sort(key=lambda x: -x["importance"])

    # 画图：最重要的 top_n 个
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：最重要的 Heads
    top_items = items[:top_n]
    ax1 = axes[0]
    bars1 = ax1.barh(
        [item["head"] for item in top_items][::-1],
        [item["importance"] for item in top_items][::-1],
        color='coral'
    )
    ax1.set_xlabel("Accuracy Drop (%)")
    ax1.set_title(f"Top {top_n} Most Important Heads")
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    # 右图：最不重要的 Heads
    bottom_items = items[-top_n:]
    ax2 = axes[1]
    bars2 = ax2.barh(
        [item["head"] for item in bottom_items][::-1],
        [item["importance"] for item in bottom_items][::-1],
        color='steelblue'
    )
    ax2.set_xlabel("Accuracy Drop (%)")
    ax2.set_title(f"Top {top_n} Least Important Heads (Candidates for Pruning)")
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(importance_scores, entropy_scores, baseline_acc, output_dir):
    """
    生成注意力头重要性分析报告
    """
    report_path = os.path.join(output_dir, "head_importance_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("注意力头重要性分析报告\n")
        f.write("ATTENTION HEAD IMPORTANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # 1. 基本信息
        f.write("1. 基本信息\n")
        f.write("-" * 40 + "\n")
        f.write(f"模型: {MODEL_NAME}\n")
        f.write(f"总层数: {NUM_LAYERS}\n")
        f.write(f"每层 Head 数: {NUM_HEADS}\n")
        f.write(f"总 Head 数: {NUM_LAYERS * NUM_HEADS}\n")
        f.write(f"原始准确率 (Baseline): {baseline_acc * 100:.2f}%\n\n")

        # 2. 重要性排名
        f.write("2. Head 重要性排名 (按准确率下降排序)\n")
        f.write("-" * 40 + "\n")

        # 排序
        items = []
        for (layer_idx, head_idx), info in importance_scores.items():
            if isinstance(info, dict):
                importance = info["importance"]
                masked_acc = info.get("masked_acc", 0)
            else:
                importance = info
                masked_acc = baseline_acc - importance

            entropy = entropy_scores.get((layer_idx, head_idx), 0)

            items.append({
                "layer": layer_idx,
                "head": head_idx,
                "importance": importance,
                "masked_acc": masked_acc,
                "entropy": entropy
            })

        items.sort(key=lambda x: -x["importance"])

        f.write("\n最重要的 10 个 Head:\n")
        f.write(f"{'排名':<4} {'Layer':<6} {'Head':<6} {'准确率下降':<12} {'熵值':<8}\n")
        for i, item in enumerate(items[:10], 1):
            f.write(f"{i:<4} {item['layer']:<6} {item['head']:<6} "
                    f"{item['importance'] * 100:>8.2f}%    {item['entropy']:.3f}\n")

        f.write("\n最不重要的 10 个 Head (剪枝候选):\n")
        f.write(f"{'排名':<4} {'Layer':<6} {'Head':<6} {'准确率下降':<12} {'熵值':<8}\n")
        for i, item in enumerate(items[-10:][::-1], 1):
            f.write(f"{i:<4} {item['layer']:<6} {item['head']:<6} "
                    f"{item['importance'] * 100:>8.2f}%    {item['entropy']:.3f}\n")

        # 3. 按层汇总
        f.write("\n\n3. 按层汇总\n")
        f.write("-" * 40 + "\n")

        for layer_idx in range(NUM_LAYERS):
            layer_items = [item for item in items if item["layer"] == layer_idx]
            avg_importance = np.mean([item["importance"] for item in layer_items])
            avg_entropy = np.mean([item["entropy"] for item in layer_items])

            f.write(f"\nLayer {layer_idx}:\n")
            f.write(f"  平均准确率下降: {avg_importance * 100:.2f}%\n")
            f.write(f"  平均熵值: {avg_entropy:.3f}\n")

            # 找出该层最重要的 head
            layer_items.sort(key=lambda x: -x["importance"])
            f.write(f"  最重要 Head: Head {layer_items[0]['head']} "
                    f"(下降 {layer_items[0]['importance'] * 100:.2f}%)\n")
            f.write(f"  最不重要 Head: Head {layer_items[-1]['head']} "
                    f"(下降 {layer_items[-1]['importance'] * 100:.2f}%)\n")

        # 4. 剪枝建议
        f.write("\n\n4. 剪枝建议\n")
        f.write("-" * 40 + "\n")

        # 找出准确率下降 < 0.5% 的 head
        prunable = [item for item in items if item["importance"] < 0.005]
        f.write(f"\n可安全剪枝的 Head (准确率下降 < 0.5%): {len(prunable)} 个\n")

        if prunable:
            for item in prunable[:10]:
                f.write(f"  - Layer {item['layer']}, Head {item['head']}: "
                        f"下降 {item['importance'] * 100:.2f}%\n")
            if len(prunable) > 10:
                f.write(f"  ... 还有 {len(prunable) - 10} 个\n")

        f.write("\n" + "=" * 60 + "\n")


def main():
    """
    主函数：程序入口
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="注意力头重要性分析 (Attention Head Importance Analysis)"
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
        help="输出目录"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="用于评估的样本数（默认500，越多越准但越慢）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备: 'cuda', 'cpu', 或 'auto'"
    )

    args = parser.parse_args()

    # 自动选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印配置
    print("\n" + "=" * 60)
    print("注意力头重要性分析 (Attention Head Importance Analysis)")
    print("=" * 60)
    print(f"模型路径: {args.ckpt_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"评估样本数: {args.n_samples}")
    print(f"设备: {device}")

    # 加载模型
    model, tokenizer = load_model(args.ckpt_path, device)

    # 加载测试数据
    dataset = load_test_data(args.n_samples)

    # 分析 1: Head Masking
    importance_scores, baseline_acc = analyze_head_importance(
        model, tokenizer, dataset, device
    )

    # 分析 2: 注意力熵
    entropy_scores = analyze_attention_entropy(
        model, tokenizer, dataset, device, n_samples=min(100, args.n_samples)
    )

    # 保存数据到 CSV
    print("\n保存分析数据...")

    data_rows = []
    for (layer_idx, head_idx), info in importance_scores.items():
        row = {
            "layer": layer_idx,
            "head": head_idx,
            "importance": info["importance"] if isinstance(info, dict) else info,
            "acc_drop_pct": (info["importance"] if isinstance(info, dict) else info) * 100,
            "entropy": entropy_scores.get((layer_idx, head_idx), 0)
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(args.output_dir, "head_importance.csv")
    df.to_csv(csv_path, index=False)
    print(f"[已保存] {csv_path}")

    # 生成可视化
    print("\n生成可视化图表...")

    plot_importance_heatmap(
        importance_scores,
        os.path.join(args.output_dir, "head_importance_heatmap.png"),
        title="Attention Head Importance (Accuracy Drop %)"
    )

    plot_entropy_heatmap(
        entropy_scores,
        os.path.join(args.output_dir, "head_entropy_heatmap.png")
    )

    plot_importance_ranking(
        importance_scores,
        os.path.join(args.output_dir, "head_importance_ranking.png")
    )

    # 生成报告
    print("\n生成分析报告...")
    generate_report(importance_scores, entropy_scores, baseline_acc, args.output_dir)

    # 打印总结
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"原始准确率: {baseline_acc * 100:.2f}%")

    # 找出最重要和最不重要的 head
    items = [(k, v["importance"] if isinstance(v, dict) else v)
             for k, v in importance_scores.items()]
    items.sort(key=lambda x: -x[1])

    print(f"\n最重要的 Head: Layer {items[0][0][0]}, Head {items[0][0][1]} "
          f"(遮掉后准确率下降 {items[0][1] * 100:.2f}%)")
    print(f"最不重要的 Head: Layer {items[-1][0][0]}, Head {items[-1][0][1]} "
          f"(遮掉后准确率下降 {items[-1][1] * 100:.2f}%)")

    # 统计可剪枝的 head
    prunable = [item for item in items if item[1] < 0.005]
    print(f"\n可安全剪枝的 Head (准确率下降 < 0.5%): {len(prunable)} 个")

    print(f"\n所有结果已保存到: {args.output_dir}")
    print("  - head_importance.csv (数据)")
    print("  - head_importance_heatmap.png (重要性热力图)")
    print("  - head_entropy_heatmap.png (熵热力图)")
    print("  - head_importance_ranking.png (排名图)")
    print("  - head_importance_report.txt (分析报告)")


if __name__ == "__main__":
    main()