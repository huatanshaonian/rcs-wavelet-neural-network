"""
直接AutoEncoder模型
方案B: 直接处理RCS数据，不使用小波变换
输入: [B, 91, 91, 2] RCS数据 (2个频率)
输出: [B, 91, 91, 2] 重建RCS数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List
import numpy as np
from autoencoder.utils.adaptive_layers import get_structure_info
from autoencoder.models.channel_attention import ChannelAttention, get_recommended_reduction
from autoencoder.utils.activation_factory import get_activation, get_activation_name
from autoencoder.models.base_autoencoder import BaseAutoEncoder


def calculate_intermediate_dims(input_dim: int, latent_dim: int, max_ratio: int = 4) -> List[int]:
    """
    动态计算中间层维度，实现渐进式压缩

    策略：
    - 保持每级压缩比不超过max_ratio（默认4:1）
    - 自动生成多个中间层以平滑过渡
    - 维度向上取整到2的幂次，便于硬件优化

    Args:
        input_dim: 输入维度
        latent_dim: 目标隐空间维度
        max_ratio: 每级最大压缩比（默认4）

    Returns:
        中间层维度列表（不包括input_dim和latent_dim）

    Example:
        >>> calculate_intermediate_dims(512, 32, max_ratio=4)
        [128]  # 512→128→32
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

    return dims


class DirectAutoEncoder(BaseAutoEncoder):
    """
    直接处理RCS数据的CNN-AutoEncoder
    跳过小波变换，直接学习RCS特征
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 use_channel_attention: bool = False,
                 activation: str = 'relu',
                 output_activation: str = None):
        """
        初始化直接AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量 (2 for 1.5GHz+3GHz, 3 for +6GHz)
            dropout_rate: Dropout比率
            use_channel_attention: 是否使用通道注意力机制 (默认: False)
            activation: 激活函数类型（'relu', 'sin', 'gelu', 'swish'等，默认: 'relu'）
            output_activation: 输出激活函数 (None 或 'softplus'，用于物理约束)
        """
        super().__init__(output_activation=output_activation)

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.input_channels = num_frequencies  # 直接使用频率数量作为通道数
        self.use_channel_attention = use_channel_attention
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        # ===== 通道注意力模块（可选） =====
        if self.use_channel_attention:
            reduction = get_recommended_reduction(self.input_channels)
            self.channel_attention = ChannelAttention(
                num_channels=self.input_channels,
                reduction=reduction
            )
        else:
            self.channel_attention = None

        # ===== Encoder: RCS数据 → 隐空间 =====
        self.encoder = nn.Sequential(
            # 第一层: 直接RCS输入 [B, num_freq, 91, 91]
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),

            # 下采样层1: [91, 91] → [46, 46]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 下采样层2: [46, 46] → [23, 23]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 下采样层3: [23, 23] → [12, 12]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 下采样层4: [12, 12] → [6, 6]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 全局平均池化: [6, 6] → [1, 1]
            nn.AdaptiveAvgPool2d(1),
        )

        # 计算编码器输出尺寸
        self.encoder_output_size = 512

        # ===== 动态适配：自动计算中间层维度 =====
        # 策略：保持每级压缩比≤4:1，避免信息瓶颈
        self.intermediate_dims = calculate_intermediate_dims(
            self.encoder_output_size, latent_dim, max_ratio=4
        )

        # 编码器全连接层：渐进式压缩
        # 示例: latent_dim=32 → [512, 128, 32]
        #       latent_dim=16 → [512, 128, 32, 16]
        to_latent_layers = [nn.Flatten()]
        current_dim = self.encoder_output_size

        # 构建中间层
        for intermediate_dim in self.intermediate_dims:
            to_latent_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                activation_layer(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到latent_dim
        to_latent_layers.append(nn.Linear(current_dim, latent_dim))
        self.to_latent = nn.Sequential(*to_latent_layers)

        # ===== Decoder: 隐空间 → RCS数据 =====

        # 对称的解码器结构
        from_latent_layers = []
        current_dim = latent_dim

        # 反向构建中间层
        for intermediate_dim in reversed(self.intermediate_dims):
            from_latent_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                activation_layer(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到encoder_output_size
        from_latent_layers.extend([
            nn.Linear(current_dim, self.encoder_output_size),
            activation_layer()
        ])
        self.from_latent = nn.Sequential(*from_latent_layers)

        self.decoder = nn.Sequential(
            # 重塑为特征图: [512] → [512, 1, 1]
            # 上采样层1: [1, 1] → [6, 6]
            nn.ConvTranspose2d(512, 256, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样层2: [6, 6] → [12, 12]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样层3: [12, 12] → [23, 23]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样层4: [23, 23] → [46, 46]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 最终层: [46, 46] → [91, 91]
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
        )

        # 最终尺寸调整层 (确保输出恰好是91x91)
        self.final_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d((91, 91)),  # 强制调整到目标尺寸
        )

        # 保存结构信息（用于 get_model_info）
        self.flattened_size = self.encoder_output_size
        self.structure_info = get_structure_info(
            self.flattened_size,
            latent_dim,
            self.intermediate_dims
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码RCS数据到隐空间

        Args:
            x: [B, H, W, num_freq] RCS数据

        Returns:
            latent: [B, latent_dim] 隐空间表示
        """
        # 输入形状转换: [B, H, W, C] → [B, C, H, W]
        if x.dim() == 4 and x.shape[-1] == self.num_frequencies:
            x = x.permute(0, 3, 1, 2)

        # 应用通道注意力（如果启用）
        if self.use_channel_attention and self.channel_attention is not None:
            x = self.channel_attention(x)

        # 编码
        features = self.encoder(x)  # [B, 512, 1, 1]
        latent = self.to_latent(features)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码隐空间到RCS数据

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            x: [B, H, W, num_freq] 重建RCS数据
        """
        # 隐空间到特征
        features = self.from_latent(latent)  # [B, 512]
        features = features.view(-1, 512, 1, 1)  # [B, 512, 1, 1]

        # 解码
        x = self.decoder(features)  # [B, num_freq, ~91, ~91]
        x = self.final_adjust(x)    # [B, num_freq, 91, 91]

        # 输出形状转换: [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        # ✅ 应用输出激活（如果启用了物理约束）
        x = self.apply_output_activation(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, H, W, num_freq] RCS数据

        Returns:
            reconstructed: [B, H, W, num_freq] 重建RCS数据
            latent: [B, latent_dim] 隐空间表示
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)

        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """
        获取模型参数数量

        Returns:
            包含参数统计的字典
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        latent_params = sum(p.numel() for p in self.to_latent.parameters()) + \
                       sum(p.numel() for p in self.from_latent.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'latent_projection': latent_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        # 构建FC层结构描述
        fc_structure = [self.encoder_output_size] + self.intermediate_dims + [self.latent_dim]
        fc_structure_str = ' → '.join(map(str, fc_structure))

        input_size = 91 * 91 * self.input_channels

        return {
            'model_name': 'DirectAutoEncoder',
            'architecture': 'CNN',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'input_channels': self.input_channels,
            'input_shape': f'[B, 91, 91, {self.input_channels}]',
            'output_shape': f'[B, 91, 91, {self.input_channels}]',
            'latent_shape': f'[B, {self.latent_dim}]',
            'fc_structure': fc_structure_str,
            'intermediate_dims': self.intermediate_dims,
            'num_fc_layers': len(fc_structure) - 1,
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'compression_ratio': f'{input_size}:{self.latent_dim} = {(input_size/self.latent_dim):.1f}:1',
            'compression_ratios': self.structure_info['compression_ratios'],
            'use_channel_attention': self.use_channel_attention
        }


def create_direct_autoencoder(latent_dim: int = 256,
                            num_frequencies: int = 2,
                            dropout_rate: float = 0.2,
                            activation: str = 'relu') -> DirectAutoEncoder:
    """
    创建直接AutoEncoder模型

    Args:
        latent_dim: 隐空间维度
        num_frequencies: 频率数量
        dropout_rate: Dropout比率
        activation: 激活函数类型

    Returns:
        模型实例和信息
    """
    model = DirectAutoEncoder(
        latent_dim=latent_dim,
        num_frequencies=num_frequencies,
        dropout_rate=dropout_rate,
        activation=activation
    )

    model_info = model.get_model_info()

    print(f"创建直接AutoEncoder:")
    print(f"  - 隐空间维度: {model_info['latent_dim']}")
    print(f"  - 输入通道数: {model_info['input_channels']}")
    print(f"  - 激活函数: {model_info['activation']}")
    print(f"  - 参数量: {model_info['total_parameters']:,}")
    print(f"  - 架构: 直接RCS输入，无小波变换")

    return model


if __name__ == "__main__":
    # 测试直接AutoEncoder
    print("=== 直接AutoEncoder测试 ===")

    # 创建模型
    model = create_direct_autoencoder(latent_dim=256, num_frequencies=2)

    # 测试数据
    batch_size = 4
    test_input = torch.randn(batch_size, 91, 91, 2)

    print(f"\n测试输入: {test_input.shape}")

    # 前向传播
    with torch.no_grad():
        reconstructed, latent = model(test_input)

    print(f"隐空间输出: {latent.shape}")
    print(f"重建输出: {reconstructed.shape}")

    # 计算重建误差
    mse = torch.mean((test_input - reconstructed) ** 2)
    print(f"重建MSE (随机权重): {mse.item():.6f}")

    print("\n✅ 直接AutoEncoder测试完成!")
