import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from Attention_Analysis import NUM_LAYERS, NUM_HEADS

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from datasets import load_dataset

NUM_LAYERS=6
NUM_HEADS=12
TOTAL_HEADS=NUM_LAYERS*NUM_HEADS

def load_importance_scores(csv_path):
    df=pd.read_csv(csv_path)
    return df

def select_heads_by_ratio(df, prune_ratio=0.3):
    df_sorted=df.sort_values('acc_drop_pct', ascending=True)
    n_prune=int(len(df_sorted)*prune_ratio)

    heads_to_prune=[]
    for i in range(n_prune):
        row=df_sorted.iloc[i]
        heads_to_prune.append((int(row['layer']), int(row['head'])))

    heads_to_keep=[]
    for i in range(n_prune, len(df_sorted)):
        row=df_sorted.iloc[i]
        heads_to_keep.append((int(row['layer']), int(row['head'])))

    return heads_to_prune, heads_to_keep

def select_heads_by_threshold(df, threshold=0.5):
    heads_to_prune=[]
    heads_to_keep=[]

    for _, row in df.iterrows():
        layer=int(row['layer'])
        head=int(row['head'])
        importance=row['acc_drop_pct']

        if importance < threshold:
            heads_to_prune.append((layer, head))
        else:
            heads_to_keep.append((layer, head))

    return heads_to_prune, heads_to_keep

def create_head_mask(heads_to_prune, num_layers=NUM_LAYERS, num_heads=NUM_HEADS):
    head_mask=torch.ones(num_layers, num_heads)

    for layer, head in heads_to_prune:
        head_mask[layer, head]=0

    return head_mask

def load_model(ckpt_path, device="cuda"):
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
    return model, tokenizer

def load_test_data(n_samples=500):
    dataset=load_dataset("imdb", split="test")
    dataset=dataset.shuffle(seed=42).select(range(n_samples))
    return dataset['text'], dataset['label']

def evaluate_with_mask(model, tokenizer, texts, labels, head_mask, device, batch_size=16):
    model.eval()
    correct = 0
    total = 0

    if head_mask is not None:
        head_mask = head_mask.to(device)  # 保持 2D (num_layers, num_heads)，transformers 会自动处理

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="评估中"):
            batch_texts=texts[i:i+batch_size]
            batch_labels=labels[i:i+batch_size]

            inputs=tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(device)

            outputs=model(**inputs, head_mask=head_mask)
            logits=outputs.logits

            preds=torch.argmax(logits, dim=1).cpu().numpy()
            correct+=sum(p==l for p, l in zip(preds, batch_labels))
            total+=len(batch_labels)

    accuracy=correct/total * 100
    return accuracy

def test_pruning_ratios(model, tokenizer, texts, labels, df, device, ratios=[0.1, 0.2, 0.3, 0.4, 0.5]):
    results=[]

    base_line_acc=evaluate_with_mask(model, tokenizer, texts, labels, None, device)
    results.append((0.0, base_line_acc, 0))

    for ratio in ratios:
        heads_to_prune, _=select_heads_by_ratio(df, ratio)
        head_mask=create_head_mask(heads_to_prune)
        acc=evaluate_with_mask(model, tokenizer, texts, labels, head_mask, device)
        results.append((ratio, acc, len(heads_to_prune)))

    return results, base_line_acc


def plot_pruning_results(results, baseline_acc, output_dir):
    ratios = [r[0] * 100 for r in results]
    accuracies = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ratios, accuracies, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline: {baseline_acc:.2f}%')
    ax.axhline(y=baseline_acc - 1, color='orange', linestyle='--',
               label=f'1% drop: {baseline_acc - 1:.2f}%')

    ax.set_xlabel('Pruning Ratio (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Attention Head Pruning: Accuracy vs Pruning Ratio', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ratio, acc, _ in results:
        ax.annotate(f'{acc:.1f}%', (ratio * 100, acc),
                    textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "pruning_accuracy_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_pruning_config(df, recommended_ratio, output_dir):
    heads_to_prune, heads_to_keep = select_heads_by_ratio(df, recommended_ratio)
    head_mask = create_head_mask(heads_to_prune)

    mask_path = os.path.join(output_dir, "head_mask.pt")
    torch.save(head_mask, mask_path)
    print(f"\n[已保存] {mask_path}")
    prune_df = pd.DataFrame(heads_to_prune, columns=['layer', 'head'])
    prune_df.to_csv(os.path.join(output_dir, "heads_to_prune.csv"), index=False)

    report = [
        "=" * 60,
        "注意力头剪枝配置报告",
        "=" * 60,
        "",
        f"推荐剪枝比例: {recommended_ratio * 100:.0f}%",
        f"剪掉头数: {len(heads_to_prune)} / {TOTAL_HEADS}",
        "",
        "剪掉的头列表:",
        "-" * 40,
    ]

    for layer, head in sorted(heads_to_prune):
        importance = df[(df['layer'] == layer) & (df['head'] == head)]['acc_drop_pct'].values[0]
        report.append(f"  Layer {layer}, Head {head}: 重要性 {importance:.4f}%")

    report.extend([
        "",
        "各层剪枝统计:",
        "-" * 40,
    ])

    for layer in range(NUM_LAYERS):
        pruned_in_layer = sum(1 for l, h in heads_to_prune if l == layer)
        report.append(f"  Layer {layer}: 剪掉 {pruned_in_layer}/{NUM_HEADS} 个头")

    report.append("=" * 60)

    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "pruning_config_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[已保存] {report_path}")

    return head_mask, report_text


def main():
    parser = argparse.ArgumentParser(description="注意力头剪枝策略定制")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--importance_csv", type=str, required=True,
                        help="Step 2 的 head_importance.csv 路径")
    parser.add_argument("--output_dir", type=str, default="./pruning_output",
                        help="输出目录")
    parser.add_argument("--n_samples", type=int, default=500, help="测试样本数")
    parser.add_argument("--recommended_ratio", type=float, default=0.3,
                        help="推荐的剪枝比例")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("注意力头结构化剪枝策略定制")
    print("=" * 60)

    df = load_importance_scores(args.importance_csv)
    model, tokenizer = load_model(args.ckpt_path, device)
    texts, labels = load_test_data(args.n_samples)

    results, baseline_acc = test_pruning_ratios(
        model, tokenizer, texts, labels, df, device,
        ratios=[0.1, 0.2, 0.3, 0.4, 0.5]
    )

    plot_pruning_results(results, baseline_acc, args.output_dir)

    print("\n" + "=" * 60)
    print(f"生成推荐剪枝配置 (比例: {args.recommended_ratio * 100:.0f}%)")
    print("=" * 60)
    head_mask, report = generate_pruning_config(df, args.recommended_ratio, args.output_dir)

    print("\n" + report)

    results_df = pd.DataFrame(results, columns=['ratio', 'accuracy', 'n_pruned'])
    results_df.to_csv(os.path.join(args.output_dir, "pruning_results.csv"), index=False)

    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()