"""
模型结构可视化工具

用于输出自适应调整后的AutoEncoder网络结构，提供清晰易读的格式。

作者: Claude Code
日期: 2025-01-26
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict


def format_number(num: int) -> str:
    """
    格式化数字，添加千位分隔符

    Args:
        num: 输入数字

    Returns:
        格式化后的字符串 (e.g., "12,544")
    """
    return f"{num:,}"


def get_parameter_count(model: nn.Module) -> Tuple[int, int]:
    """
    计算模型的参数量

    Args:
        model: PyTorch模型

    Returns:
        (total_params, trainable_params): 总参数量和可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model: nn.Module, model_name: str = "AutoEncoder", mode: str = "summary"):
    """
    打印模型结构摘要

    Args:
        model: AutoEncoder模型实例
        model_name: 模型名称
        mode: 输出模式
            - 'summary': 简洁摘要（推荐）
            - 'detailed': 详细结构
            - 'pytorch': PyTorch原生输出
    """
    print(f"\n{'='*80}")
    print(f"{'模型结构摘要':^80}")
    print(f"{'='*80}\n")

    print(f"模型名称: {model_name}")
    print(f"模型类: {type(model).__name__}")

    # 获取模型信息
    model_info = None
    if hasattr(model, 'get_model_info'):
        try:
            model_info = model.get_model_info()
        except:
            pass

    # 基本信息
    print(f"\n{'-'*80}")
    print("基本信息")
    print(f"{'-'*80}")

    if model_info:
        # 如果有get_model_info，使用其信息
        print(f"模式: {model_info.get('mode', 'Unknown')}")
        print(f"架构: {model_info.get('architecture', 'Unknown')}")
        print(f"隐空间维度: {model_info.get('latent_dim', 'Unknown')}")
        print(f"输入通道数: {model_info.get('input_channels', 'Unknown')}")
        print(f"Dropout率: {model_info.get('dropout_rate', 'Unknown')}")

        # 输入/输出形状
        if 'input_shape' in model_info:
            print(f"输入形状: {model_info['input_shape']}")
        if 'output_shape' in model_info:
            print(f"输出形状: {model_info['output_shape']}")

    # 参数统计
    total_params, trainable_params = get_parameter_count(model)
    print(f"\n总参数量: {format_number(total_params)}")
    print(f"可训练参数: {format_number(trainable_params)}")

    # 网络结构详情
    if model_info and 'fc_structure' in model_info:
        print(f"\n{'-'*80}")
        print("全连接层结构（自适应调整）")
        print(f"{'-'*80}")

        fc_structure = model_info['fc_structure']
        print(f"\n{fc_structure}")

        # 中间维度
        if 'intermediate_dims' in model_info:
            intermediate_dims = model_info['intermediate_dims']
            print(f"\n中间层维度: {intermediate_dims}")

        # 层数
        if 'num_fc_layers' in model_info:
            num_fc_layers = model_info['num_fc_layers']
            print(f"全连接层数: {num_fc_layers}")

        # 压缩比
        if 'compression_ratios' in model_info:
            compression_ratios = model_info['compression_ratios']
            print(f"\n各阶段压缩比:")
            for i, ratio in enumerate(compression_ratios, 1):
                print(f"  阶段 {i}: {ratio}")

    # 详细模式：显示所有层
    if mode == 'detailed':
        print(f"\n{'-'*80}")
        print("详细网络结构")
        print(f"{'-'*80}\n")

        # 打印encoder
        if hasattr(model, 'encoder'):
            print("【Encoder】")
            _print_module_details(model.encoder, indent=2)

        # 打印decoder
        if hasattr(model, 'decoder'):
            print("\n【Decoder】")
            _print_module_details(model.decoder, indent=2)

    # PyTorch原生模式
    elif mode == 'pytorch':
        print(f"\n{'-'*80}")
        print("PyTorch原生结构")
        print(f"{'-'*80}\n")
        print(model)

    print(f"\n{'='*80}\n")


def _print_module_details(module: nn.Module, indent: int = 0):
    """
    递归打印模块详情

    Args:
        module: PyTorch模块
        indent: 缩进级别
    """
    prefix = " " * indent

    if isinstance(module, nn.Sequential):
        for i, layer in enumerate(module):
            print(f"{prefix}({i}): {_layer_repr(layer)}")
    elif isinstance(module, nn.ModuleList):
        for i, layer in enumerate(module):
            print(f"{prefix}({i}): {_layer_repr(layer)}")
    else:
        print(f"{prefix}{_layer_repr(module)}")


def _layer_repr(layer: nn.Module) -> str:
    """
    获取层的简洁表示

    Args:
        layer: PyTorch层

    Returns:
        层的字符串表示
    """
    layer_type = type(layer).__name__

    if isinstance(layer, nn.Linear):
        return f"Linear(in={layer.in_features}, out={layer.out_features})"
    elif isinstance(layer, nn.Conv2d):
        return f"Conv2d(in={layer.in_channels}, out={layer.out_channels}, kernel={layer.kernel_size})"
    elif isinstance(layer, nn.ConvTranspose2d):
        return f"ConvTranspose2d(in={layer.in_channels}, out={layer.out_channels}, kernel={layer.kernel_size})"
    elif isinstance(layer, nn.BatchNorm2d):
        return f"BatchNorm2d({layer.num_features})"
    elif isinstance(layer, nn.BatchNorm1d):
        return f"BatchNorm1d({layer.num_features})"
    elif isinstance(layer, nn.Dropout):
        return f"Dropout(p={layer.p})"
    elif isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
        return f"{layer_type}()"
    elif isinstance(layer, nn.Flatten):
        return "Flatten()"
    else:
        return f"{layer_type}"


def compare_models(models: List[Tuple[nn.Module, str]], show_diff: bool = True):
    """
    对比多个模型的结构

    Args:
        models: 模型列表，每个元素是 (model, name) 元组
        show_diff: 是否突出显示差异
    """
    print(f"\n{'='*80}")
    print(f"{'模型结构对比':^80}")
    print(f"{'='*80}\n")

    # 收集所有模型信息
    all_info = []
    for model, name in models:
        info = {
            'name': name,
            'class': type(model).__name__,
            'total_params': get_parameter_count(model)[0],
            'trainable_params': get_parameter_count(model)[1],
        }

        if hasattr(model, 'get_model_info'):
            try:
                model_info = model.get_model_info()
                info.update({
                    'mode': model_info.get('mode', 'N/A'),
                    'architecture': model_info.get('architecture', 'N/A'),
                    'latent_dim': model_info.get('latent_dim', 'N/A'),
                    'fc_structure': model_info.get('fc_structure', 'N/A'),
                    'num_fc_layers': model_info.get('num_fc_layers', 'N/A'),
                })
            except:
                pass

        all_info.append(info)

    # 打印对比表格
    print(f"{'模型名称':<30} {'类名':<30} {'参数量':<15} {'隐空间维度':<12}")
    print(f"{'-'*80}")

    for info in all_info:
        name = info['name'][:28]
        class_name = info['class'][:28]
        params = format_number(info['total_params'])
        latent = str(info.get('latent_dim', 'N/A'))
        print(f"{name:<30} {class_name:<30} {params:<15} {latent:<12}")

    # 详细对比
    print(f"\n{'-'*80}")
    print("全连接层结构对比")
    print(f"{'-'*80}\n")

    for info in all_info:
        print(f"{info['name']}:")
        if 'fc_structure' in info:
            print(f"  结构: {info['fc_structure']}")
            print(f"  层数: {info.get('num_fc_layers', 'N/A')}")
        else:
            print(f"  （无结构信息）")
        print()

    print(f"{'='*80}\n")


def export_structure_to_dict(model: nn.Module) -> Dict[str, Any]:
    """
    导出模型结构为字典（便于保存/序列化）

    Args:
        model: AutoEncoder模型

    Returns:
        包含结构信息的字典
    """
    structure = {
        'class_name': type(model).__name__,
        'total_params': get_parameter_count(model)[0],
        'trainable_params': get_parameter_count(model)[1],
    }

    if hasattr(model, 'get_model_info'):
        try:
            model_info = model.get_model_info()
            structure.update(model_info)
        except:
            pass

    return structure


# ============================================================================
# 便捷函数
# ============================================================================

def quick_summary(model: nn.Module, name: str = "AutoEncoder"):
    """
    快速打印模型摘要（推荐使用）

    Args:
        model: AutoEncoder模型
        name: 模型名称
    """
    print_model_summary(model, name, mode='summary')


def detailed_summary(model: nn.Module, name: str = "AutoEncoder"):
    """
    打印详细模型结构

    Args:
        model: AutoEncoder模型
        name: 模型名称
    """
    print_model_summary(model, name, mode='detailed')


def pytorch_summary(model: nn.Module, name: str = "AutoEncoder"):
    """
    打印PyTorch原生格式结构

    Args:
        model: AutoEncoder模型
        name: 模型名称
    """
    print_model_summary(model, name, mode='pytorch')


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 示例：测试不同latent_dim的WaveletAutoEncoder
    from autoencoder.models import WaveletAutoEncoder, DirectAutoEncoder

    print("\n测试自适应结构输出功能\n")

    # 测试1: WaveletAutoEncoder with latent_dim=32
    model_w32 = WaveletAutoEncoder(latent_dim=32, num_frequencies=2)
    quick_summary(model_w32, "WaveletAutoEncoder (latent_dim=32)")

    # 测试2: WaveletAutoEncoder with latent_dim=16
    model_w16 = WaveletAutoEncoder(latent_dim=16, num_frequencies=2)
    quick_summary(model_w16, "WaveletAutoEncoder (latent_dim=16)")

    # 测试3: DirectAutoEncoder with latent_dim=32
    model_d32 = DirectAutoEncoder(latent_dim=32, num_frequencies=2)
    quick_summary(model_d32, "DirectAutoEncoder (latent_dim=32)")

    # 对比
    compare_models([
        (model_w32, "Wavelet-32"),
        (model_w16, "Wavelet-16"),
        (model_d32, "Direct-32"),
    ])
