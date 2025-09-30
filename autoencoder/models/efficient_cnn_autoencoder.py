"""
高效的3层CNN-AutoEncoder
针对49x49小波系数优化设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class EfficientCNNAutoEncoder(nn.Module):
    """
    高效的3层CNN-AutoEncoder
    输入: [B, 8, 49, 49] (正确的小波系数尺寸 - 91x91经小波变换后得到)
    设计原则: 相比原91x91输入，数据量减少3.4倍，层数也相应减少
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.1):
        """
        初始化高效CNN-AutoEncoder

        Args:
            latent_dim: 隐空间维度
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

        print(f"初始化高效CNN-AE: 输入[{self.input_channels}, 49, 49] → 3层架构")

        # ===== 高效Encoder (仅3层) =====

        # Layer 1: [8, 49, 49] → [64, 49, 49]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Layer 2: [64, 49, 49] → [128, 25, 25] (49//2 = 24, 但padding调整为25)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Layer 3: [128, 25, 25] → [256, 13, 13] (25//2 = 12, 但padding调整为13)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # 全局平均池化 + FC层
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # 固定到4x4
        self.fc_features = 256 * 4 * 4  # 4096

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, latent_dim)
        )

        # ===== 高效Decoder (3层逆向) =====

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.fc_features),
            nn.ReLU(inplace=True)
        )

        # Layer 3 逆向: [256, 4, 4] → [128, 25, 25] (精确路径)
        self.deconv3 = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=3, padding=0, output_padding=0),  # 4→13
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # 13→25
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Layer 2 逆向: [128, 25, 25] → [64, 49, 49] (精确路径)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 25→49
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Layer 1 逆向: [64, 49, 49] → [8, 49, 49]
        self.deconv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 小波系数可以有负值
        )

        self._initialize_weights()

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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码器 - 仅3层卷积"""
        # 输入验证 - 修正为正确的小波系数尺寸
        if x.shape[1:] != (8, 49, 49):
            raise ValueError(f"期望输入[B, 8, 49, 49], 得到{x.shape}")

        x1 = self.conv1(x)      # [B, 64, 49, 49]
        x2 = self.conv2(x1)     # [B, 128, 25, 25]
        x3 = self.conv3(x2)     # [B, 256, 13, 13]

        # 全局池化并编码
        x_pool = self.global_pool(x3)  # [B, 256, 4, 4]
        latent = self.encoder_fc(x_pool)

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码器 - 3层逆向卷积"""
        x = self.decoder_fc(latent)     # [B, 4096]
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

        return {
            'model_name': 'Efficient 3-Layer CNN-AutoEncoder',
            'architecture': '3-layer CNN optimized for 49x49 input',
            'input_shape': '[batch, 8, 49, 49]',
            'latent_dim': self.latent_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layers': [
                'Conv1: 8→64 [49×49]',
                'Conv2: 64→128 [25×25]',
                'Conv3: 128→256 [13×13]',
                f'FC: {self.fc_features}→{self.latent_dim}'
            ],
            'advantages': [
                '数据量减少3.4倍',
                '层数从5层减少到3层',
                '参数量大幅减少',
                '训练速度更快',
                '过拟合风险降低'
            ]
        }


class CompactCNNAutoEncoder(nn.Module):
    """
    更紧凑的2层CNN-AutoEncoder版本
    极简设计，适合快速原型
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.latent_dim = latent_dim

        # 编码器：仅2层
        self.encoder = nn.Sequential(
            # Layer 1: [8, 49, 49] → [32, 25, 25]
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: [32, 25, 25] → [64, 13, 13]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 全局池化 + FC
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)
        )

        # 解码器：2层逆向
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 4, 4)),

            # Layer 2 逆向: [64, 4, 4] → [32, 25, 25]
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 1 逆向: [32, 25, 25] → [8, 49, 49]
            nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def compare_architectures():
    """对比不同CNN架构的效率"""
    print("=== CNN架构效率对比 ===")
    print()

    # 创建不同架构
    efficient_3layer = EfficientCNNAutoEncoder(latent_dim=256)
    compact_2layer = CompactCNNAutoEncoder(latent_dim=128)

    # 参数统计
    params_3layer = sum(p.numel() for p in efficient_3layer.parameters())
    params_2layer = sum(p.numel() for p in compact_2layer.parameters())

    print(f"参数量对比:")
    print(f"  3层高效版: {params_3layer:,} 参数 ({params_3layer/1e6:.1f}M)")
    print(f"  2层紧凑版: {params_2layer:,} 参数 ({params_2layer/1e6:.1f}M)")
    print()

    # 测试输入
    test_input = torch.randn(4, 8, 49, 49)
    print(f"测试输入: {test_input.shape}")

    # 测试前向传播
    with torch.no_grad():
        out_3layer, latent_3layer = efficient_3layer(test_input)
        out_2layer, latent_2layer = compact_2layer(test_input)

    print(f"3层输出: {out_3layer.shape}, 隐空间: {latent_3layer.shape}")
    print(f"2层输出: {out_2layer.shape}, 隐空间: {latent_2layer.shape}")
    print()

    print("推荐选择:")
    print("  • 3层版本: 平衡性能与效率，推荐用于生产")
    print("  • 2层版本: 极简设计，适合快速实验")
    print("  • 相比原来5-6层: 大幅提升训练速度")

    return efficient_3layer, compact_2layer


def test_efficient_architecture():
    """测试高效架构"""
    print("=== 测试高效CNN-AutoEncoder ===")

    model = EfficientCNNAutoEncoder(latent_dim=256)

    # 模拟正确的小波输入
    test_input = torch.randn(2, 8, 49, 49)

    with torch.no_grad():
        reconstructed, latent = model(test_input)

    print(f"输入: {test_input.shape}")
    print(f"重建: {reconstructed.shape}")
    print(f"隐空间: {latent.shape}")

    # 重建误差
    mse = torch.mean((test_input - reconstructed)**2).item()
    print(f"重建MSE: {mse:.6f}")

    # 模型信息
    info = model.get_model_info()
    print(f"\\n模型详情:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    return model


if __name__ == "__main__":
    # 测试高效架构
    test_efficient_architecture()
    print()

    # 对比分析
    compare_architectures()