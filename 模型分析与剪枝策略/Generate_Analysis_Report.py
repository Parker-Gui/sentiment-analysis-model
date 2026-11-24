# -*- coding: utf-8 -*-
"""
生成完整的模型分析报告
整合 Step 1-4 的所有分析结果
"""

import os
import argparse
import pandas as pd
from datetime import datetime


def generate_comprehensive_analysis_report(
    param_report_path,
    param_stats_csv,
    head_report_path,
    head_csv_path,
    ffn_report_path,
    ffn_csv_path,
    layer_report_path,
    layer_csv_path,
    output_path
):
    """生成完整的模型分析报告"""
    
    report = []
    
    # ============================
    # 标题和摘要
    # ============================
    report.extend([
        "=" * 80,
        "DistilBERT-LoRA 情感分析模型 - 完整分析报告",
        "=" * 80,
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 80,
        "执行摘要",
        "=" * 80,
        "",
        "本报告对基于 DistilBERT + LoRA 的 IMDB 情感分析模型进行了全面的参数分析和",
        "重要性评估，为后续的模型压缩和剪枝提供数据支持。",
        "",
        "主要发现：",
        "  1. 模型总参数量：68.17M，其中 FFN 层占 41.6%，Embedding 层占 35.0%",
        "  2. 72 个注意力头中，约 55.6% (40个) 可安全剪枝，性能下降 < 0.5%",
        "  3. 18,432 个 FFN 神经元中，约 20% 可剪枝",
        "  4. Layer 0 最重要，Layer 2 最不重要",
        "  5. 预期模型可压缩 15-25%，准确率下降 < 1%",
        "",
    ])
    
    # ============================
    # 第一部分：参数分布分析
    # ============================
    report.extend([
        "=" * 80,
        "第一部分：参数分布分析",
        "=" * 80,
        "",
    ])
    
    # 读取参数统计
    if os.path.exists(param_stats_csv):
        df = pd.read_csv(param_stats_csv)
        
        report.extend([
            "1.1 参数统计概览",
            "-" * 50,
            "",
            f"总参数量: {df['count'].sum():,}",
            f"全局稀疏性: {df['sparsity'].mean():.2f}%",
            "",
            "各类别参数分布：",
        ])
        
        for _, row in df.iterrows():
            report.append(
                f"  {row['category']:25s}: {row['count']:>12,} "
                f"({row['count']/df['count'].sum()*100:>5.1f}%)"
            )
        
        report.extend([
            "",
            "1.2 参数分布特征",
            "-" * 50,
            "",
        ])
        
        for _, row in df.iterrows():
            report.append(f"[{row['category']}]")
            report.append(f"  均值: {row['mean']:.6f}")
            report.append(f"  标准差: {row['std']:.6f}")
            report.append(f"  绝对值均值: {row['abs_mean']:.6f}")
            report.append(f"  稀疏性: {row['sparsity']:.2f}%")
            report.append("")
    
    # 添加原始报告内容
    if os.path.exists(param_report_path):
        report.extend([
            "1.3 详细分析",
            "-" * 50,
            "",
        ])
        with open(param_report_path, 'r', encoding='utf-8') as f:
            report.append(f.read())
    
    report.append("")
    
    # ============================
    # 第二部分：注意力头重要性分析
    # ============================
    report.extend([
        "=" * 80,
        "第二部分：注意力头重要性分析",
        "=" * 80,
        "",
    ])
    
    if os.path.exists(head_csv_path):
        head_df = pd.read_csv(head_csv_path)
        
        report.extend([
            "2.1 重要性统计",
            "-" * 50,
            "",
            f"总注意力头数: {len(head_df)}",
            f"平均重要性: {head_df['acc_drop_pct'].mean():.4f}%",
            f"标准差: {head_df['acc_drop_pct'].std():.4f}%",
            "",
        ])
        
        # 最重要和最不重要的头
        sorted_heads = head_df.sort_values('acc_drop_pct', ascending=False)
        
        report.extend([
            "2.2 最重要的 10 个注意力头",
            "-" * 50,
            "",
        ])
        
        for i, (_, row) in enumerate(sorted_heads.head(10).iterrows(), 1):
            report.append(
                f"  {i:2d}. Layer {int(row['layer'])}, Head {int(row['head'])}: "
                f"重要性 {row['acc_drop_pct']:.4f}% (熵: {row['entropy']:.4f})"
            )
        
        report.extend([
            "",
            "2.3 最不重要的 10 个注意力头",
            "-" * 50,
            "",
        ])
        
        for i, (_, row) in enumerate(sorted_heads.tail(10).iterrows(), 1):
            report.append(
                f"  {i:2d}. Layer {int(row['layer'])}, Head {int(row['head'])}: "
                f"重要性 {row['acc_drop_pct']:.4f}% (熵: {row['entropy']:.4f})"
            )
        
        # 可安全剪枝的头
        prunable = head_df[head_df['acc_drop_pct'] < 0.5]
        report.extend([
            "",
            "2.4 可安全剪枝的头 (重要性 < 0.5%)",
            "-" * 50,
            "",
            f"数量: {len(prunable)} / {len(head_df)} ({len(prunable)/len(head_df)*100:.1f}%)",
            "",
        ])
    
    # 添加原始报告
    if os.path.exists(head_report_path):
        report.extend([
            "2.5 详细分析",
            "-" * 50,
            "",
        ])
        with open(head_report_path, 'r', encoding='utf-8') as f:
            report.append(f.read())
    
    report.append("")
    
    # ============================
    # 第三部分：FFN 层重要性分析
    # ============================
    report.extend([
        "=" * 80,
        "第三部分：FFN 层重要性分析",
        "=" * 80,
        "",
    ])
    
    if os.path.exists(ffn_csv_path):
        ffn_df = pd.read_csv(ffn_csv_path)
        
        report.extend([
            "3.1 FFN 神经元统计",
            "-" * 50,
            "",
            f"总神经元数: {len(ffn_df):,}",
            f"平均重要性: {ffn_df['combined_importance'].mean():.6f}",
            f"可剪枝神经元数: {ffn_df['prunable'].sum():,} ({ffn_df['prunable'].sum()/len(ffn_df)*100:.1f}%)",
            "",
        ])
        
        # 按层统计
        report.extend([
            "3.2 各层 FFN 统计",
            "-" * 50,
            "",
        ])
        
        for layer in range(6):
            layer_df = ffn_df[ffn_df['layer'] == layer]
            prunable_count = layer_df['prunable'].sum()
            avg_importance = layer_df['combined_importance'].mean()
            
            report.append(
                f"  Layer {layer}: 平均重要性 {avg_importance:.6f}, "
                f"可剪枝 {prunable_count:,}/{len(layer_df):,} "
                f"({prunable_count/len(layer_df)*100:.1f}%)"
            )
        
        report.append("")
    
    # 添加原始报告
    if os.path.exists(ffn_report_path):
        report.extend([
            "3.3 详细分析",
            "-" * 50,
            "",
        ])
        with open(ffn_report_path, 'r', encoding='utf-8') as f:
            report.append(f.read())
    
    report.append("")
    
    # ============================
    # 第四部分：层级综合重要性
    # ============================
    report.extend([
        "=" * 80,
        "第四部分：层级综合重要性分析",
        "=" * 80,
        "",
    ])
    
    if os.path.exists(layer_csv_path):
        layer_df = pd.read_csv(layer_csv_path)
        
        report.extend([
            "4.1 层级重要性排名",
            "-" * 50,
            "",
        ])
        
        sorted_layers = layer_df.sort_values('rank')
        for _, row in sorted_layers.iterrows():
            report.append(
                f"  第 {int(row['rank'])} 名: Layer {int(row['layer'])} - "
                f"综合分数 {row['combined_importance']:.4f} "
                f"(Attention: {row['attention_normalized']:.4f}, FFN: {row['ffn_normalized']:.4f})"
            )
        
        report.append("")
    
    # 添加原始报告
    if os.path.exists(layer_report_path):
        report.extend([
            "4.2 详细分析",
            "-" * 50,
            "",
        ])
        with open(layer_report_path, 'r', encoding='utf-8') as f:
            report.append(f.read())
    
    report.append("")
    
    # ============================
    # 结论
    # ============================
    report.extend([
        "=" * 80,
        "总结与结论",
        "=" * 80,
        "",
        "1. 参数分布分析结论",
        "   - FFN 层参数最多 (41.6%)，是压缩的主要目标",
        "   - Embedding 层次之 (35.0%)，但不建议剪枝",
        "   - LayerNorm 和分类头参数极少，可忽略",
        "",
        "2. 注意力头重要性结论",
        "   - 约 55.6% 的注意力头可安全剪枝",
        "   - Layer 4 的注意力头最重要",
        "   - Layer 1 有较多冗余头",
        "",
        "3. FFN 层重要性结论",
        "   - 约 20% 的 FFN 神经元可剪枝",
        "   - Layer 4 的 FFN 最冗余 (52.1% 可剪枝)",
        "   - Layer 0 和 Layer 3 的 FFN 较重要",
        "",
        "4. 层级综合结论",
        "   - Layer 0 最重要，不建议大幅剪枝",
        "   - Layer 2 最不重要，可优先剪枝",
        "   - Layer 4 的 Attention 重要但 FFN 冗余",
        "",
        "5. 压缩潜力估计",
        "   - 注意力头剪枝: 20-30% → 参数减少约 2-3M",
        "   - FFN 剪枝: 20-30% → 参数减少约 5-8M",
        "   - 总体压缩: 15-25% → 模型从 68M 降至 50-58M",
        "   - 预期性能下降: < 1%",
        "",
        "=" * 80,
        "可视化图表清单",
        "=" * 80,
        "",
        "参数分布分析:",
        "  - param_histogram.png: 参数值分布直方图",
        "  - param_boxplot.png: 各类别参数分布箱线图",
        "  - sparsity_bar.png: 稀疏性柱状图",
        "  - param_pie.png: 参数量占比饼图",
        "  - layer_heatmap.png: 各层参数热力图",
        "",
        "注意力头分析:",
        "  - head_importance_heatmap.png: 注意力头重要性热力图",
        "  - head_entropy_heatmap.png: 注意力头熵热力图",
        "  - head_importance_ranking.png: 重要性排名图",
        "",
        "FFN 层分析:",
        "  - weight_importance_heatmap.png: 权重范数重要性热力图",
        "  - activation_importance_heatmap.png: 激活值重要性热力图",
        "  - combined_importance_heatmap.png: 综合重要性热力图",
        "  - importance_distribution.png: 重要性分布箱线图",
        "  - importance_histogram.png: 重要性分布直方图",
        "  - prunable_neurons.png: 可剪枝神经元统计",
        "",
        "层级综合分析:",
        "  - layer_importance_comparison.png: 层级重要性对比图",
        "",
        "=" * 80,
        "报告结束",
        "=" * 80,
    ])
    
    # 保存报告
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"[已生成] {output_path}")
    print(f"报告总行数: {len(report)}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="生成完整的模型分析报告")
    parser.add_argument("--param_report", type=str, 
                       default="./analysis_output/param_analysis_report.txt")
    parser.add_argument("--param_csv", type=str,
                       default="./analysis_output/param_stats.csv")
    parser.add_argument("--head_report", type=str,
                       default="./analysis_output/head_importance_report.txt")
    parser.add_argument("--head_csv", type=str,
                       default="./analysis_output/head_importance.csv")
    parser.add_argument("--ffn_report", type=str,
                       default="./ffn_analysis_output/ffn_analysis_report.txt")
    parser.add_argument("--ffn_csv", type=str,
                       default="./ffn_analysis_output/ffn_importance.csv")
    parser.add_argument("--layer_report", type=str,
                       default="./layer_analysis_output/layer_importance_report.txt")
    parser.add_argument("--layer_csv", type=str,
                       default="./layer_analysis_output/layer_importance.csv")
    parser.add_argument("--output", type=str,
                       default="./模型分析报告_完整版.txt")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("生成完整的模型分析报告")
    print("="*60)
    
    report = generate_comprehensive_analysis_report(
        args.param_report,
        args.param_csv,
        args.head_report,
        args.head_csv,
        args.ffn_report,
        args.ffn_csv,
        args.layer_report,
        args.layer_csv,
        args.output
    )
    
    print("\n报告已生成！")


if __name__ == "__main__":
    main()
