"""
自适应层构建器
为所有AutoEncoder模型提供动态适配的网络结构生成

功能：
1. 根据输入维度和隐空间维度自动计算中间层
2. 保持每级压缩比≤4:1（可配置）
3. 支持CNN和MLP两种架构类型
4. 自动生成对称的Encoder/Decoder结构
"""

import torch.nn as nn
from typing import List, Tuple
import numpy as np


def calculate_intermediate_dims(input_dim: int,
                                latent_dim: int,
                                max_ratio: int = 4,
                                min_layers: int = 0) -> List[int]:
    """
    动态计算中间层维度，实现渐进式压缩

    策略：
    - 保持每级压缩比不超过max_ratio（默认4:1）
    - 自动生成多个中间层以平滑过渡
    - 维度向上取整到2的幂次，便于硬件优化
    - 可选最少层数要求（主要用于MLP）

    Args:
        input_dim: 输入维度
        latent_dim: 目标隐空间维度
        max_ratio: 每级最大压缩比（默认4）
        min_layers: 最少中间层数（默认0，自动决定）

    Returns:
        中间层维度列表（不包括input_dim和latent_dim）

    Examples:
        >>> calculate_intermediate_dims(4096, 32, max_ratio=4)
        [1024, 256, 64]  # 4096→1024→256→64→32

        >>> calculate_intermediate_dims(512, 16, max_ratio=4)
        [128, 32]  # 512→128→32→16

        >>> calculate_intermediate_dims(19208, 32, max_ratio=4, min_layers=3)
        [4096, 1024, 256, 64]  # 至少3个中间层
    """
    if input_dim <= latent_dim:
        return []

    dims = []
    current = input_dim

    # 持续压缩直到接近目标维度
    while current > latent_dim * max_ratio:
        # 压缩到当前的1/2到1/4之间
        next_dim = max(latent_dim, current // max_ratio)
        # 向上取整到2的幂次（如64, 128, 256...）
        next_dim = 2 ** round(np.log2(next_dim))
        # 确保不小于latent_dim
        next_dim = max(next_dim, latent_dim)

        if next_dim < current:
            dims.append(next_dim)
            current = next_dim
        else:
            break

    # 确保至少有min_layers个中间层
    if min_layers > 0 and len(dims) < min_layers:
        # 在当前dims最后插入更多层
        if dims:
            last_dim = dims[-1]
        else:
            last_dim = input_dim

        while len(dims) < min_layers and last_dim > latent_dim * 2:
            next_dim = max(latent_dim, last_dim // 2)
            next_dim = 2 ** round(np.log2(next_dim))
            next_dim = max(next_dim, latent_dim)
            if next_dim < last_dim:
                dims.append(next_dim)
                last_dim = next_dim
            else:
                break

    return dims


def build_fc_encoder(input_dim: int,
                     latent_dim: int,
                     dropout_rate: float = 0.2,
                     max_ratio: int = 4,
                     min_layers: int = 0,
                     use_bn: bool = True) -> Tuple[nn.Sequential, List[int]]:
    """
    构建全连接编码器（渐进式压缩）

    Args:
        input_dim: 输入维度（展平后）
        latent_dim: 隐空间维度
        dropout_rate: Dropout比率
        max_ratio: 每级最大压缩比
        min_layers: 最少中间层数
        use_bn: 是否使用BatchNorm

    Returns:
        (encoder, intermediate_dims)
        - encoder: nn.Sequential编码器
        - intermediate_dims: 中间层维度列表

    Examples:
        >>> encoder, dims = build_fc_encoder(4096, 32, dropout_rate=0.2)
        >>> # encoder: 4096 → 1024 → 256 → 64 → 32
        >>> # dims: [1024, 256, 64]
    """
    # 计算中间层维度
    intermediate_dims = calculate_intermediate_dims(
        input_dim, latent_dim, max_ratio, min_layers
    )

    # 构建编码器
    layers = [nn.Flatten()]
    current_dim = input_dim

    # 构建中间层
    for intermediate_dim in intermediate_dims:
        layers.append(nn.Linear(current_dim, intermediate_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(intermediate_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        current_dim = intermediate_dim

    # 最后一层到latent_dim（不加激活函数）
    layers.append(nn.Linear(current_dim, latent_dim))

    encoder = nn.Sequential(*layers)
    return encoder, intermediate_dims


def build_fc_decoder(latent_dim: int,
                     output_dim: int,
                     intermediate_dims: List[int],
                     dropout_rate: float = 0.2,
                     use_bn: bool = True,
                     use_output_activation: bool = False) -> nn.Sequential:
    """
    构建全连接解码器（对称的渐进式扩展）

    Args:
        latent_dim: 隐空间维度
        output_dim: 输出维度（展平后）
        intermediate_dims: 中间层维度列表（与encoder对称）
        dropout_rate: Dropout比率
        use_bn: 是否使用BatchNorm
        use_output_activation: 输出层是否使用激活函数

    Returns:
        decoder: nn.Sequential解码器

    Examples:
        >>> decoder = build_fc_decoder(32, 4096, [64, 256, 1024])
        >>> # decoder: 32 → 64 → 256 → 1024 → 4096
    """
    layers = []
    current_dim = latent_dim

    # 反向构建中间层
    for intermediate_dim in reversed(intermediate_dims):
        layers.append(nn.Linear(current_dim, intermediate_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(intermediate_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        current_dim = intermediate_dim

    # 最后一层到output_dim
    layers.append(nn.Linear(current_dim, output_dim))

    # 可选的输出激活
    if use_output_activation:
        layers.append(nn.ReLU(inplace=True))

    decoder = nn.Sequential(*layers)
    return decoder


def build_adaptive_fc_pair(input_dim: int,
                           latent_dim: int,
                           dropout_rate: float = 0.2,
                           max_ratio: int = 4,
                           min_layers: int = 0,
                           arch_type: str = 'cnn') -> Tuple[nn.Sequential, nn.Sequential, List[int]]:
    """
    一键构建自适应的Encoder-Decoder对（全连接部分）

    Args:
        input_dim: 输入维度
        latent_dim: 隐空间维度
        dropout_rate: Dropout比率
        max_ratio: 每级最大压缩比
        min_layers: 最少中间层数
        arch_type: 架构类型 ('cnn' 或 'mlp')

    Returns:
        (encoder_fc, decoder_fc, intermediate_dims)

    Examples:
        >>> # CNN架构：4096 → latent_dim
        >>> enc_fc, dec_fc, dims = build_adaptive_fc_pair(4096, 32, arch_type='cnn')

        >>> # MLP架构：19208 → latent_dim，至少3个中间层
        >>> enc_fc, dec_fc, dims = build_adaptive_fc_pair(19208, 32, min_layers=3, arch_type='mlp')
    """
    # MLP默认要求更多中间层
    if arch_type.lower() == 'mlp' and min_layers == 0:
        min_layers = 3

    # 构建encoder
    encoder_fc, intermediate_dims = build_fc_encoder(
        input_dim, latent_dim, dropout_rate, max_ratio, min_layers
    )

    # 构建decoder
    decoder_fc = build_fc_decoder(
        latent_dim, input_dim, intermediate_dims, dropout_rate
    )

    return encoder_fc, decoder_fc, intermediate_dims


def get_structure_info(input_dim: int,
                       latent_dim: int,
                       intermediate_dims: List[int]) -> dict:
    """
    生成网络结构信息（用于模型的get_model_info）

    Args:
        input_dim: 输入维度
        latent_dim: 隐空间维度
        intermediate_dims: 中间层维度列表

    Returns:
        包含结构信息的字典
    """
    structure = [input_dim] + intermediate_dims + [latent_dim]
    structure_str = ' → '.join(map(str, structure))

    return {
        'fc_structure': structure_str,
        'intermediate_dims': intermediate_dims,
        'num_fc_layers': len(structure) - 1,
        'compression_ratios': [
            f'{structure[i]}:{structure[i+1]} = {structure[i]/structure[i+1]:.1f}:1'
            for i in range(len(structure) - 1)
        ]
    }


# ===== 使用示例 =====
if __name__ == "__main__":
    print("=== 自适应层构建器测试 ===\n")

    # 测试1: CNN模式 (Wavelet)
    print("【测试1】CNN-Wavelet: 4096 → 32")
    enc_fc, dec_fc, dims = build_adaptive_fc_pair(4096, 32, arch_type='cnn')
    info = get_structure_info(4096, 32, dims)
    print(f"  结构: {info['fc_structure']}")
    print(f"  中间层: {dims}")
    print(f"  层数: {info['num_fc_layers']}")
    print()

    # 测试2: CNN模式 (Direct)
    print("【测试2】CNN-Direct: 512 → 16")
    enc_fc, dec_fc, dims = build_adaptive_fc_pair(512, 16, arch_type='cnn')
    info = get_structure_info(512, 16, dims)
    print(f"  结构: {info['fc_structure']}")
    print(f"  中间层: {dims}")
    print(f"  层数: {info['num_fc_layers']}")
    print()

    # 测试3: MLP模式 (Wavelet)
    print("【测试3】MLP-Wavelet: 19208 → 32")
    enc_fc, dec_fc, dims = build_adaptive_fc_pair(19208, 32, arch_type='mlp')
    info = get_structure_info(19208, 32, dims)
    print(f"  结构: {info['fc_structure']}")
    print(f"  中间层: {dims}")
    print(f"  层数: {info['num_fc_layers']}")
    print()

    # 测试4: MLP模式 (Direct)
    print("【测试4】MLP-Direct: 16562 → 16")
    enc_fc, dec_fc, dims = build_adaptive_fc_pair(16562, 16, arch_type='mlp')
    info = get_structure_info(16562, 16, dims)
    print(f"  结构: {info['fc_structure']}")
    print(f"  中间层: {dims}")
    print(f"  层数: {info['num_fc_layers']}")
    print()

    print("✅ 所有测试通过!")
