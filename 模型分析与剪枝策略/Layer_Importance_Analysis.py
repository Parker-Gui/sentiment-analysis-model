import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_importance_data(attention_csv, ffn_csv):
    attention_df=pd.read_csv(attention_csv)
    ffn_df=pd.read_csv(ffn_csv)
    return attention_df, ffn_df

def aggregate_attention_by_layer(attention_df):
    layer_stats=attention_df.groupby('layer')['acc_drop_pct'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()

    return layer_stats

def aggregate_ffn_by_layer(ffn_df):
    layer_stats=ffn_df.groupby('layer')['combined_importance'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()

    prunable_ratio=ffn_df.groupby('layer')['prunable'].mean()
    layer_stats['prunable_ratio']=prunable_ratio.values
    return layer_stats

def compute_layer_importance(attention_stats, ffn_stats, attention_weight=0.5):
    att_mean=attention_stats['mean'].values
    ffn_mean=ffn_stats['mean'].values

    att_norm=(att_mean-att_mean.min())/(att_mean.max()-att_mean.min()+1e-8)
    ffn_norm=(ffn_mean-ffn_mean.min())/(ffn_mean.max()-ffn_mean.min()+1e-8)

    combined=attention_weight*att_norm+(1-attention_weight)*ffn_norm

    result=pd.DataFrame({
        'layer': range(len(combined)),
        'attention_importance': att_mean,
        'attention_normalized': att_norm,
        'ffn_importance': ffn_mean,
        'ffn_normalized': ffn_norm,
        'combined_importance': combined
    })

    result['rank']=result['combined_importance'].rank(ascending=False).astype(int)
    
    return result


def plot_layer_comparison(layer_importance, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    layers = layer_importance['layer'].values

    # 图1：Attention vs FFN 对比
    ax1 = axes[0]
    x = np.arange(len(layers))
    width = 0.35
    ax1.bar(x - width / 2, layer_importance['attention_normalized'], width,
            label='Attention', color='steelblue')
    ax1.bar(x + width / 2, layer_importance['ffn_normalized'], width,
            label='FFN', color='coral')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Normalized Importance')
    ax1.set_title('Attention vs FFN Importance by Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{i}' for i in layers])
    ax1.legend()

    ax2 = axes[1]
    colors = plt.cm.RdYlGn(layer_importance['combined_importance'])
    ax2.bar(layers, layer_importance['combined_importance'], color=colors)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Combined Importance')
    ax2.set_title('Combined Layer Importance')
    ax2.set_xticks(layers)

    ax3 = axes[2]
    sorted_df = layer_importance.sort_values('combined_importance', ascending=True)
    ax3.barh(sorted_df['layer'].astype(str), sorted_df['combined_importance'],
             color='teal')
    ax3.set_xlabel('Combined Importance')
    ax3.set_ylabel('Layer')
    ax3.set_title('Layer Importance Ranking')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "layer_importance_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_summary_report(attention_stats, ffn_stats, layer_importance, output_dir):
    most_important = layer_importance.loc[layer_importance['rank'] == 1, 'layer'].values[0]
    least_important = layer_importance.loc[layer_importance['rank'] == layer_importance['rank'].max(), 'layer'].values[
        0]

    report = [
        "=" * 60,
        "层级平均重要性分析报告",
        "=" * 60,
        "",
        "1. 注意力层统计",
        "-" * 40,
    ]

    for _, row in attention_stats.iterrows():
        report.append(f"   Layer {int(row['layer'])}: "
                      f"mean={row['mean']:.4f}, std={row['std']:.4f}")

    report.extend([
        "",
        "2. FFN层统计",
        "-" * 40,
    ])

    for _, row in ffn_stats.iterrows():
        report.append(f"   Layer {int(row['layer'])}: "
                      f"mean={row['mean']:.4f}, prunable={row['prunable_ratio'] * 100:.1f}%")

    report.extend([
        "",
        "3. 层级综合重要性排名",
        "-" * 40,
    ])

    sorted_layers = layer_importance.sort_values('rank')
    for _, row in sorted_layers.iterrows():
        report.append(f"   第{int(row['rank'])}名: Layer {int(row['layer'])} "
                      f"(综合分数: {row['combined_importance']:.4f})")

    report.extend([
        "",
        "4. 结论",
        "-" * 40,
        f"   - 最重要的层: Layer {most_important}",
        f"   - 最不重要的层: Layer {least_important} (优先剪枝候选)",
        "",
        "=" * 60,
    ])

    report_text = "\n".join(report)

    # 保存
    report_path = os.path.join(output_dir, "layer_importance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  [已保存] {report_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="计算层级平均重要性")
    parser.add_argument("--attention_csv", type=str, required=True,
                        help="Step 2 输出的 head_importance.csv 路径")
    parser.add_argument("--ffn_csv", type=str, required=True,
                        help="Step 3 输出的 ffn_importance.csv 路径")
    parser.add_argument("--output_dir", type=str, default="./layer_analysis_output",
                        help="输出目录")
    parser.add_argument("--attention_weight", type=float, default=0.5,
                        help="Attention 在综合评分中的权重")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("层级平均重要性分析")
    print("=" * 60)

    attention_df, ffn_df = load_importance_data(args.attention_csv, args.ffn_csv)

    # 2. 按层聚合
    attention_stats = aggregate_attention_by_layer(attention_df)
    ffn_stats = aggregate_ffn_by_layer(ffn_df)

    layer_importance = compute_layer_importance(
        attention_stats, ffn_stats, args.attention_weight
    )

    layer_importance.to_csv(
        os.path.join(args.output_dir, "layer_importance.csv"),
        index=False
    )

    plot_layer_comparison(layer_importance, args.output_dir)

    report = generate_summary_report(
        attention_stats, ffn_stats, layer_importance, args.output_dir
    )

    print("\n" + report)
    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()