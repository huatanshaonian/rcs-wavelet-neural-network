"""
微隐空间AutoEncoder设计
专为极小隐空间维度(如10维)优化的架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class MicroLatentAutoEncoder(nn.Module):
    """
    微隐空间AutoEncoder
    支持极小隐空间维度(10-64维)的优化设计
    """

    def __init__(self,
                 latent_dim: int = 10,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.1):
        """
        初始化微隐空间AutoEncoder

        Args:
            latent_dim: 隐空间维度 (推荐10-64)
            num_frequencies: 频率数量
            wavelet_bands: 小波频带数
            dropout_rate: Dropout比率
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands  # 8
        self.dropout_rate = dropout_rate

        print(f"初始化微隐空间CNN-AE: 输入[{self.input_channels}, 49, 49] → {latent_dim}维隐空间")

        # ===== CNN编码器 (复用高效3层架构) =====

        # Stage 1: [8, 49, 49] → [64, 49, 49]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 2: [64, 49, 49] → [128, 25, 25]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 3: [128, 25, 25] → [256, 13, 13]
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # 全局平均池化 + 渐进式压缩FC层
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_features = 256 * 4 * 4  # 4096

        # ===== 关键：渐进式压缩到微隐空间 =====
        self.encoder_fc = self._create_progressive_encoder()

        # ===== 渐进式扩张解码器 =====
        self.decoder_fc = self._create_progressive_decoder()

        # ===== CNN解码器 (复用架构) =====

        # Stage 3 逆向: [256, 4, 4] → [128, 25, 25]
        self.deconv3 = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=3, padding=0, output_padding=0),  # 4→13
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # 13→25
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Stage 2 逆向: [128, 25, 25] → [64, 49, 49]
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 25→49
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1 逆向: [64, 49, 49] → [8, 49, 49]
        self.deconv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 小波系数可以有负值
        )

        self._initialize_weights()

    def _create_progressive_encoder(self) -> nn.Module:
        """
        创建渐进式编码器，逐步压缩到微隐空间
        """
        layers = [nn.Flatten()]

        # 根据隐空间维度动态调整架构
        current_dim = self.fc_features  # 4096

        if self.latent_dim <= 16:
            # 极小隐空间：需要更多层来平滑压缩
            intermediate_dims = [1024, 256, 64]
            print(f"  极小隐空间设计: {current_dim} → {' → '.join(map(str, intermediate_dims))} → {self.latent_dim}")
        elif self.latent_dim <= 64:
            # 小隐空间：中等压缩
            intermediate_dims = [1024, 256]
            print(f"  小隐空间设计: {current_dim} → {' → '.join(map(str, intermediate_dims))} → {self.latent_dim}")
        else:
            # 正常隐空间：标准压缩
            intermediate_dims = [512]
            print(f"  标准隐空间设计: {current_dim} → {' → '.join(map(str, intermediate_dims))} → {self.latent_dim}")

        # 添加中间层
        for dim in intermediate_dims:
            layers.extend([
                nn.Linear(current_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = dim

        # 最终压缩到隐空间
        layers.append(nn.Linear(current_dim, self.latent_dim))

        return nn.Sequential(*layers)

    def _create_progressive_decoder(self) -> nn.Module:
        """
        创建渐进式解码器，从微隐空间逐步扩张
        """
        layers = []

        # 根据隐空间维度动态调整架构（编码器的逆向）
        current_dim = self.latent_dim

        if self.latent_dim <= 16:
            # 极小隐空间：需要更多层来平滑扩张
            intermediate_dims = [64, 256, 1024]
        elif self.latent_dim <= 64:
            # 小隐空间：中等扩张
            intermediate_dims = [256, 1024]
        else:
            # 正常隐空间：标准扩张
            intermediate_dims = [512]

        # 添加中间层
        for dim in intermediate_dims:
            layers.extend([
                nn.Linear(current_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = dim

        # 最终扩张到特征空间
        layers.extend([
            nn.Linear(current_dim, self.fc_features),
            nn.ReLU(inplace=True)
        ])

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对于极小隐空间，使用更小的初始化方差
                if self.latent_dim <= 16:
                    nn.init.normal_(m.weight, 0, 0.005)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码器前向传播"""
        # 输入验证
        if x.shape[1:] != (8, 49, 49):
            raise ValueError(f"期望输入[B, 8, 49, 49], 得到{x.shape}")

        # CNN编码
        x1 = self.conv1(x)      # [B, 64, 49, 49]
        x2 = self.conv2(x1)     # [B, 128, 25, 25]
        x3 = self.conv3(x2)     # [B, 256, 13, 13]

        # 全局池化并渐进式压缩
        x_pool = self.global_pool(x3)  # [B, 256, 4, 4]
        latent = self.encoder_fc(x_pool)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码器前向传播"""
        # 渐进式扩张
        x = self.decoder_fc(latent)     # [B, 4096]

        # CNN解码
        x = self.deconv3(x)             # [B, 128, 25, 25]
        x = self.deconv2(x)             # [B, 64, 49, 49]
        x = self.deconv1(x)             # [B, 8, 49, 49]

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 计算压缩比
        input_size = 8 * 49 * 49  # 19208
        compression_ratio = input_size / self.latent_dim

        return {
            'model_name': f'Micro-Latent AutoEncoder ({self.latent_dim}D)',
            'architecture': 'Progressive compression for micro latent space',
            'input_shape': f'[batch, {self.input_channels}, 49, 49]',
            'latent_dim': self.latent_dim,
            'compression_ratio': f'{compression_ratio:.1f}x',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'encoder_path': self._get_encoder_path(),
            'decoder_path': self._get_decoder_path(),
            'advantages': [
                '渐进式压缩，避免信息骤降',
                '支持极小隐空间(10维)',
                '动态架构适应不同压缩比',
                '优化的权重初始化',
                '防止梯度消失'
            ]
        }

    def _get_encoder_path(self) -> str:
        """获取编码器路径描述"""
        if self.latent_dim <= 16:
            return "4096→1024→256→64→latent"
        elif self.latent_dim <= 64:
            return "4096→1024→256→latent"
        else:
            return "4096→512→latent"

    def _get_decoder_path(self) -> str:
        """获取解码器路径描述"""
        if self.latent_dim <= 16:
            return "latent→64→256→1024→4096"
        elif self.latent_dim <= 64:
            return "latent→256→1024→4096"
        else:
            return "latent→512→4096"


def test_micro_latent_dimensions():
    """测试不同微隐空间维度"""
    print("=== 测试微隐空间AutoEncoder ===")
    print()

    test_dims = [10, 16, 32, 64, 128, 256]
    test_input = torch.randn(2, 8, 49, 49)

    results = []

    for dim in test_dims:
        print(f"测试 {dim}维隐空间:")

        model = MicroLatentAutoEncoder(latent_dim=dim)

        with torch.no_grad():
            reconstructed, latent = model(test_input)
            mse = torch.mean((test_input - reconstructed)**2).item()

        info = model.get_model_info()

        print(f"  压缩比: {info['compression_ratio']}")
        print(f"  编码路径: {info['encoder_path']}")
        print(f"  参数量: {info['total_params']:,}")
        print(f"  重建MSE: {mse:.6f}")
        print()

        results.append({
            'dim': dim,
            'compression': info['compression_ratio'],
            'params': info['total_params'],
            'mse': mse
        })

    print("📊 维度对比总结:")
    print("维度  压缩比    参数量      重建MSE")
    print("-" * 40)
    for r in results:
        print(f"{r['dim']:3d}   {r['compression']:8s} {r['params']:8,}  {r['mse']:.6f}")

    return results


def compare_with_standard_ae():
    """与标准AutoEncoder对比"""
    print("\n=== 微隐空间 vs 标准架构对比 ===")

    from autoencoder.models.efficient_cnn_autoencoder import EfficientCNNAutoEncoder

    # 创建模型
    micro_ae = MicroLatentAutoEncoder(latent_dim=10)
    standard_ae = EfficientCNNAutoEncoder(latent_dim=10)

    # 参数统计
    micro_params = sum(p.numel() for p in micro_ae.parameters())
    standard_params = sum(p.numel() for p in standard_ae.parameters())

    print(f"微隐空间AE参数: {micro_params:,}")
    print(f"标准AE参数:     {standard_params:,}")
    print(f"参数减少:       {(standard_params-micro_params)/standard_params*100:.1f}%")

    # 测试重建质量
    test_input = torch.randn(4, 8, 49, 49)

    with torch.no_grad():
        micro_recon, micro_latent = micro_ae(test_input)
        standard_recon, standard_latent = standard_ae(test_input)

        micro_mse = torch.mean((test_input - micro_recon)**2).item()
        standard_mse = torch.mean((test_input - standard_recon)**2).item()

    print(f"微隐空间重建MSE: {micro_mse:.6f}")
    print(f"标准架构重建MSE: {standard_mse:.6f}")

    return micro_ae, standard_ae


if __name__ == "__main__":
    # 测试不同隐空间维度
    test_micro_latent_dimensions()

    # 对比标准架构
    compare_with_standard_ae()