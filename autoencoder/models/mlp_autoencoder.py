"""
MLP-AutoEncoder模型
提供全连接网络架构的AutoEncoder实现
支持小波模式和直接模式
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List
import numpy as np
from autoencoder.utils.adaptive_layers import get_structure_info
from autoencoder.utils.activation_factory import get_activation, get_activation_name


def calculate_mlp_dims(input_dim: int, latent_dim: int, max_ratio: int = 4, min_layers: int = 3) -> List[int]:
    """
    为MLP动态计算中间层维度

    策略：
    - MLP适合更深的层级结构
    - 保持每级压缩比≤max_ratio
    - 至少min_layers个中间层
    - 维度向上取整到2的幂次

    Args:
        input_dim: 输入维度（展平后）
        latent_dim: 目标隐空间维度
        max_ratio: 每级最大压缩比
        min_layers: 最少中间层数

    Returns:
        中间层维度列表

    Example:
        >>> calculate_mlp_dims(19208, 32, max_ratio=4, min_layers=3)
        [8192, 2048, 512, 128]  # 19208→8192→2048→512→128→32
    """
    if input_dim <= latent_dim:
        return []

    dims = []
    current = input_dim

    # 持续压缩直到接近目标
    while current > latent_dim * max_ratio:
        next_dim = max(latent_dim, current // max_ratio)
        # 向上取整到2的幂次
        next_dim = 2 ** round(np.log2(next_dim))
        next_dim = max(next_dim, latent_dim)

        if next_dim < current:
            dims.append(next_dim)
            current = next_dim
        else:
            break

    # 确保至少有min_layers个中间层
    if len(dims) < min_layers:
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


class WaveletMLPAutoEncoder(nn.Module):
    """
    小波增强的MLP-AutoEncoder
    使用全连接网络处理小波系数

    输入: [B, 49, 49, 8] 小波系数 (2频率 × 4频带)
    输出: [B, 49, 49, 8] 重建小波系数
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 input_size: int = 49,
                 activation: str = 'relu'):
        """
        初始化小波MLP-AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量 (2 for 1.5GHz+3GHz, 3 for +6GHz)
            wavelet_bands: 小波频带数 (通常为4: LL, LH, HL, HH)
            dropout_rate: Dropout比例
            input_size: 输入小波系数尺寸 (49 for db4小波变换后的尺寸)
            activation: 激活函数类型 (例如 'relu', 'sin', 'gelu', 'swish')
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.activation_type = get_activation_name(activation)

        # 计算展平后的输入维度
        self.input_dim = input_size * input_size * self.input_channels  # 49×49×8 = 19,208

        # ===== 动态适配：自动计算MLP中间层维度 =====
        # 策略：MLP适合深度网络，至少3个中间层，每级压缩比≤4:1
        self.intermediate_dims = calculate_mlp_dims(
            self.input_dim, latent_dim, max_ratio=4, min_layers=3
        )

        # ===== Encoder: 小波系数 → 隐空间 =====
        # 动态构建encoder
        # 示例: latent_dim=32 → [19208, 8192, 2048, 512, 128, 32]
        encoder_layers = [nn.Flatten()]
        current_dim = self.input_dim

        # 构建中间层
        for intermediate_dim in self.intermediate_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到latent_dim（不加激活函数）
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ===== Decoder: 隐空间 → 小波系数 =====
        # 对称的decoder结构
        decoder_layers = []
        current_dim = latent_dim

        # 反向构建中间层
        for intermediate_dim in reversed(self.intermediate_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到input_dim（不加激活函数，允许小波系数任意范围）
        decoder_layers.append(nn.Linear(current_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # 保存结构信息（用于 get_model_info）
        self.structure_info = get_structure_info(
            self.input_dim,
            latent_dim,
            self.intermediate_dims
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码器: 小波系数 → 隐空间

        Args:
            x: [B, input_size, input_size, num_freq*4] 小波系数

        Returns:
            latent: [B, latent_dim] 隐空间表示
        """
        # 调整维度: [B, input_size, input_size, num_freq*4] → [B, num_freq*4, input_size, input_size]
        x = x.permute(0, 3, 1, 2)

        # MLP会在内部自动Flatten
        latent = self.encoder(x)

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码器: 隐空间 → 小波系数

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            x_recon: [B, input_size, input_size, num_freq*4] 重建小波系数
        """
        # 解码
        x_flat = self.decoder(latent)

        # 重塑: [B, input_dim] → [B, input_channels, input_size, input_size]
        x_recon = x_flat.view(-1, self.input_channels, self.input_size, self.input_size)

        # 调整维度: [B, num_freq*4, input_size, input_size] → [B, input_size, input_size, num_freq*4]
        x_recon = x_recon.permute(0, 2, 3, 1)

        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, input_size, input_size, num_freq*4] 小波系数

        Returns:
            x_recon: [B, input_size, input_size, num_freq*4] 重建小波系数
            latent: [B, latent_dim] 隐空间表示
        """
        latent = self.encode(x)
        x_recon = self.decode(latent)

        return x_recon, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
            'trainable': trainable_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        param_count = self.get_parameter_count()

        # 构建MLP层结构描述
        mlp_structure = [self.input_dim] + self.intermediate_dims + [self.latent_dim]
        mlp_structure_str = ' → '.join(map(str, mlp_structure))

        input_size = self.input_size * self.input_size * self.input_channels
        return {
            'model_name': 'WaveletMLPAutoEncoder',
            'architecture': 'MLP',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'wavelet_bands': self.wavelet_bands,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'input_dim': self.input_dim,
            'input_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'output_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'latent_shape': f'[B, {self.latent_dim}]',
            'mlp_structure': mlp_structure_str,
            'intermediate_dims': self.intermediate_dims,
            'num_mlp_layers': len(mlp_structure) - 1,
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'compression_ratio': f'{input_size}:{self.latent_dim} = {(input_size/self.latent_dim):.1f}:1',
            'compression_ratios': self.structure_info['compression_ratios']
        }


class DirectMLPAutoEncoder(nn.Module):
    """
    直接处理RCS数据的MLP-AutoEncoder
    使用全连接网络处理原始RCS数据

    输入: [B, 91, 91, 2] RCS数据 (2个频率)
    输出: [B, 91, 91, 2] 重建RCS数据
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu'):
        """
        初始化直接MLP-AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量 (2 for 1.5GHz+3GHz, 3 for +6GHz)
            dropout_rate: Dropout比例
            activation: 激活函数类型 (例如 'relu', 'sin', 'gelu', 'swish')
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.input_channels = num_frequencies
        self.activation_type = get_activation_name(activation)

        # 计算展平后的输入维度
        self.input_dim = 91 * 91 * num_frequencies  # 91×91×2 = 16,562 或 91×91×3 = 24,843

        # ===== 动态适配：自动计算MLP中间层维度 =====
        # 策略：MLP适合深度网络，至少3个中间层，每级压缩比≤4:1
        self.intermediate_dims = calculate_mlp_dims(
            self.input_dim, latent_dim, max_ratio=4, min_layers=3
        )

        # ===== Encoder: RCS数据 → 隐空间 =====
        # 动态构建encoder
        # 示例: latent_dim=32 → [16562, 4096, 1024, 256, 64, 32]
        encoder_layers = [nn.Flatten()]
        current_dim = self.input_dim

        # 构建中间层
        for intermediate_dim in self.intermediate_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到latent_dim（不加激活函数）
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ===== Decoder: 隐空间 → RCS数据 =====
        # 对称的decoder结构
        decoder_layers = []
        current_dim = latent_dim

        # 反向构建中间层
        for intermediate_dim in reversed(self.intermediate_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到input_dim（不加激活函数，允许RCS任意范围）
        decoder_layers.append(nn.Linear(current_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # 保存结构信息（用于 get_model_info）
        self.structure_info = get_structure_info(
            self.input_dim,
            latent_dim,
            self.intermediate_dims
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码器: RCS数据 → 隐空间

        Args:
            x: [B, 91, 91, num_freq] RCS数据

        Returns:
            latent: [B, latent_dim] 隐空间表示
        """
        # 调整维度: [B, 91, 91, num_freq] → [B, num_freq, 91, 91]
        x = x.permute(0, 3, 1, 2)

        # MLP会在内部自动Flatten
        latent = self.encoder(x)

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码器: 隐空间 → RCS数据

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            x_recon: [B, 91, 91, num_freq] 重建RCS数据
        """
        # 解码
        x_flat = self.decoder(latent)

        # 重塑: [B, input_dim] → [B, num_freq, 91, 91]
        x_recon = x_flat.view(-1, self.input_channels, 91, 91)

        # 调整维度: [B, num_freq, 91, 91] → [B, 91, 91, num_freq]
        x_recon = x_recon.permute(0, 2, 3, 1)

        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, 91, 91, num_freq] RCS数据

        Returns:
            x_recon: [B, 91, 91, num_freq] 重建RCS数据
            latent: [B, latent_dim] 隐空间表示
        """
        latent = self.encode(x)
        x_recon = self.decode(latent)

        return x_recon, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
            'trainable': trainable_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        param_count = self.get_parameter_count()

        # 构建MLP层结构描述
        mlp_structure = [self.input_dim] + self.intermediate_dims + [self.latent_dim]
        mlp_structure_str = ' → '.join(map(str, mlp_structure))

        return {
            'model_name': 'DirectMLPAutoEncoder',
            'architecture': 'MLP',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'input_channels': self.input_channels,
            'input_dim': self.input_dim,
            'input_shape': f'[B, 91, 91, {self.input_channels}]',
            'output_shape': f'[B, 91, 91, {self.input_channels}]',
            'latent_shape': f'[B, {self.latent_dim}]',
            'mlp_structure': mlp_structure_str,
            'intermediate_dims': self.intermediate_dims,
            'num_mlp_layers': len(mlp_structure) - 1,
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'compression_ratio': f'{self.input_dim}:{self.latent_dim} = {(self.input_dim/self.latent_dim):.1f}:1',
            'compression_ratios': self.structure_info['compression_ratios']
        }


def test_wavelet_mlp():
    """测试小波MLP AutoEncoder"""
    print("=== 测试 WaveletMLPAutoEncoder ===")

    model = WaveletMLPAutoEncoder(latent_dim=256, num_frequencies=2)
    print(f"模型信息:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, 49, 49, 8)
    print(f"\n输入形状: {test_input.shape}")

    recon, latent = model(test_input)
    print(f"重建形状: {recon.shape}")
    print(f"隐空间形状: {latent.shape}")

    # 计算重建误差
    mse = torch.mean((test_input - recon)**2).item()
    print(f"初始MSE (未训练): {mse:.6f}")

    print("✓ WaveletMLPAutoEncoder 测试通过\n")


def test_direct_mlp():
    """测试直接MLP AutoEncoder"""
    print("=== 测试 DirectMLPAutoEncoder ===")

    model = DirectMLPAutoEncoder(latent_dim=256, num_frequencies=2)
    print(f"模型信息:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, 91, 91, 2)
    print(f"\n输入形状: {test_input.shape}")

    recon, latent = model(test_input)
    print(f"重建形状: {recon.shape}")
    print(f"隐空间形状: {latent.shape}")

    # 计算重建误差
    mse = torch.mean((test_input - recon)**2).item()
    print(f"初始MSE (未训练): {mse:.6f}")

    print("✓ DirectMLPAutoEncoder 测试通过\n")


if __name__ == "__main__":
    test_wavelet_mlp()
    test_direct_mlp()
