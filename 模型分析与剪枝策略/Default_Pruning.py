import os
import argparse
import json
import torch
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from adapters import AutoAdapterModel

def classify_parameter(name):
    name_lower=name.lower()
    if 'lora' in name_lower:
        return 'LoRA'
    if 'heads' in name_lower or 'classifier' in name_lower:
        return 'Classification Head'
    if 'embedding' in name_lower:
        if 'word' in name_lower:
            return 'Word Embedding'
        elif 'position' in name_lower:
            return 'Position Embedding'
        return 'Embedding'
    if 'layernorm' in name_lower or 'layer_norm' in name_lower:
        return 'LayerNorm'
    if any(x in name_lower for x in ['q_lin', 'k_lin', 'v_lin', 'out_lin']):
        return 'Attention'
    if 'ffn' in name_lower or 'lin1' in name_lower or 'lin2' in name_lower:
        return 'FFN'
    return 'Other'

def analyze_other_layers(model):
    stats=defaultdict(lambda: {'params': 0, 'count': 0, 'names': []})
    for name, param in model.named_parameters():
        category=classify_parameter(name)
        stats[category]['params']+=param.numel()
        stats[category]['count']+=1
        stats[category]['names'].append(name)

    return dict(stats)

def get_default_strategy():
    strategies={
        'Word Embedding': {
            'action': '不剪枝',
            'sparsity': 0.0,
            'reason': '词嵌入是每个词的语义表示基础，剪枝会直接破坏词的含义理解'
        },
        'Position Embedding': {
            'action': '不剪枝',
            'sparsity': 0.0,
            'reason': '位置嵌入编码序列位置信息，对理解词序至关重要'
        },
        'LayerNorm': {
            'action': '不剪枝',
            'sparsity': 0.0,
            'reason': '参数量极少(~18K)，且对训练稳定性关键，剪枝收益几乎为零'
        },
        'LoRA': {
            'action': '保留',
            'sparsity': 0.0,
            'reason': 'LoRA是微调的核心，包含任务特定知识，参数已很少(~0.6M)'
        },
        'Classification Head': {
            'action': '不剪枝',
            'sparsity': 0.0,
            'reason': '分类头直接决定输出结果，参数极少(~1.5K)，剪枝无意义'
        },
        'Other': {
            'action': '保守处理',
            'sparsity': 0.0,
            'reason': '未知层类型默认保留，避免意外损失'
        }
    }
    return strategies

def generate_default_strategy_report(stats, strategies, output_dir):
    report=[
        "=" *60,
        "默认剪枝策略配置报告",
        "=" * 60,
        "",
        "说明：此报告定义了除 Attention 和 FFN 之外的其他层类型的剪枝策略",
        "",
        "1. 参数统计（非 Attention/FFN 层）",
        "-" * 50,
    ]

    other_categories = ['Word Embedding', 'Position Embedding', 'LayerNorm',
                        'LoRA', 'Classification Head', 'Other']

    total_other = 0

    for category in other_categories:
        if category in stats:
            params = stats[category]['params']
            total_other += params
            report.append(f"   {category:25s}: {params:>12,} 参数")
    report.append(f"   {'小计':25s}: {total_other:>12,} 参数")

    report.extend([
        "",
        "2. 默认策略定义",
        "-" * 50,
    ])

    for category in other_categories:
        if category in strategies:
            s = strategies[category]
            report.append(f"\n   【{category}】")
            report.append(f"      策略: {s['action']}")
            report.append(f"      稀疏率: {s['sparsity']*100:.0f}%")
            report.append(f"      理由: {s['reason']}")

    report.extend([
        "",
        "3. 总结",
        "-" * 50,
        "   - 所有其他层类型均采用「不剪枝」或「保留」策略",
        "   - 原因：这些层参数量小、作用关键、剪枝收益极低",
        "   - 模型压缩主要通过 Attention 结构化剪枝和 FFN 非结构化剪枝实现",
        "",
        "=" * 60,
    ])

    report_text="\n".join(report)

    report_path=os.path.join(output_dir, "default_strategy_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text

def save_strategy_config(stats, strategies, output_dir):
    config={}
    for category, strategy in strategies.items():
        param_count=stats.get(category, {}).get('params', 0)
        config[category]={
            'action': strategy['action'],
            'sparsity': strategy['sparsity'],
            'reason': strategy['reason'],
            'params_count': param_count,
        }

    config_path=os.path.join(output_dir, "strategy_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def main():
    parser=argparse.ArgumentParser(description="默认剪枝策略")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./default_strategy_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("默认剪枝策略配置")
    print("=" * 60)

    model = AutoAdapterModel.from_pretrained("distilbert-base-uncased")
    adapter_path = os.path.join(args.ckpt_path, "lora_adapter")
    model.load_adapter(adapter_path, load_as="lora_adapter", set_active=True)
    head_path = os.path.join(args.ckpt_path, "imdb")
    model.load_head(head_path)

    stats = analyze_other_layers(model)

    strategies = get_default_strategy()

    report = generate_default_strategy_report(stats, strategies, args.output_dir)
    print("\n" + report)

    # 5. 保存配置文件
    save_strategy_config(stats, strategies, args.output_dir)

    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
