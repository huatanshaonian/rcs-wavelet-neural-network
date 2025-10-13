"""
正确的CNN-AutoEncoder架构
输入: [B, 8, 46, 46] - 正确的小波系数尺寸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class CorrectCNNAutoEncoder(nn.Module):
    """
    正确的CNN-AutoEncoder
    输入: [B, 8, 46, 46] (2频率 × 4小波频带 × 46×46尺寸)
    输出: [B, 8, 46, 46] (重建的小波系数)
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 wavelet_size: Tuple[int, int] = (46, 46),
                 dropout_rate: float = 0.1,
                 use_attention: bool = True):
        """
        初始化正确的CNN-AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量
            wavelet_bands: 小波频带数
            wavelet_size: 小波系数尺寸 (46, 46)
            dropout_rate: Dropout比率
            use_attention: 是否使用注意力机制
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands  # 8
        self.wavelet_h, self.wavelet_w = wavelet_size  # 46, 46
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        print(f"初始化正确CNN-AE: 输入[{self.input_channels}, {self.wavelet_h}, {self.wavelet_w}]")

        # ===== 正确的Encoder架构 =====
        # 输入: [B, 8, 46, 46]

        # Stage 1: [8, 46, 46] → [32, 46, 46]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 2: [32, 46, 46] → [64, 23, 23]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 3: [64, 23, 23] → [128, 12, 12]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 4: [128, 12, 12] → [256, 6, 6]
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 5: [256, 6, 6] → [512, 3, 3]
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # 注意力机制 (可选)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1),
                nn.Sigmoid()
            )

        # 编码器FC层: [512, 3, 3] → latent_dim
        self.final_conv_size = 512 * 3 * 3  # 4608
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_conv_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, latent_dim)
        )

        # ===== 正确的Decoder架构 =====

        # 解码器FC层
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, self.final_conv_size),
            nn.ReLU(inplace=True)
        )

        # Stage 5 逆向: [512, 3, 3] → [256, 6, 6]
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Stage 4 逆向: [256, 6, 6] → [128, 12, 12]
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Stage 3 逆向: [128, 12, 12] → [64, 23, 23]
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 2 逆向: [64, 23, 23] → [32, 46, 46]
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1 逆向: [32, 46, 46] → [8, 46, 46]
        self.deconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 小波系数可以有负值
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
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
        """编码器前向传播"""
        # 输入验证
        expected_shape = (self.input_channels, self.wavelet_h, self.wavelet_w)
        if x.shape[1:] != expected_shape:
            raise ValueError(f"输入形状错误: 期望 [B, {expected_shape}], 得到 {x.shape}")

        # 逐层编码
        x1 = self.conv1(x)      # [B, 32, 46, 46]
        x2 = self.conv2(x1)     # [B, 64, 23, 23]
        x3 = self.conv3(x2)     # [B, 128, 12, 12]
        x4 = self.conv4(x3)     # [B, 256, 6, 6]
        x5 = self.conv5(x4)     # [B, 512, 3, 3]

        # 注意力机制 (可选)
        if self.use_attention:
            attention_weights = self.attention(x5)
            x5 = x5 * attention_weights

        # 最终编码
        latent = self.encoder_fc(x5)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码器前向传播"""
        # 重建特征图
        x = self.decoder_fc(latent)
        x = x.view(-1, 512, 3, 3)  # [B, 512, 3, 3]

        # 逐层解码
        x = self.deconv5(x)        # [B, 256, 6, 6]
        x = self.deconv4(x)        # [B, 128, 12, 12]
        x = self.deconv3(x)        # [B, 64, 23, 23]
        x = self.deconv2(x)        # [B, 32, 46, 46]
        x = self.deconv1(x)        # [B, 8, 46, 46]

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
            'model_name': 'Correct CNN-AutoEncoder',
            'architecture': '5-stage CNN for 46x46 wavelet coefficients',
            'input_shape': f'[batch, {self.input_channels}, {self.wavelet_h}, {self.wavelet_w}]',
            'latent_dim': self.latent_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'stages': [
                f'Conv1: {self.input_channels}→32 [46×46]',
                'Conv2: 32→64 [23×23]',
                'Conv3: 64→128 [12×12]',
                'Conv4: 128→256 [6×6]',
                'Conv5: 256→512 [3×3]',
                f'FC: {self.final_conv_size}→{self.latent_dim}'
            ],
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout_rate,
            'wavelet_size': (self.wavelet_h, self.wavelet_w)
        }


def test_correct_cnn_architecture():
    """测试正确的CNN架构"""
    print("=== 测试正确的CNN-AutoEncoder架构 ===")

    # 创建模型
    model = CorrectCNNAutoEncoder(latent_dim=256)

    # 创建正确的输入 [B, 8, 46, 46]
    batch_size = 4
    test_input = torch.randn(batch_size, 8, 46, 46)
    print(f"输入形状: {test_input.shape}")

    # 前向传播
    with torch.no_grad():
        reconstructed, latent = model(test_input)

    print(f"重建输出形状: {reconstructed.shape}")
    print(f"隐空间形状: {latent.shape}")

    # 计算重建误差
    mse = torch.mean((test_input - reconstructed)**2).item()
    print(f"重建MSE (随机初始化): {mse:.6f}")

    # 模型信息
    info = model.get_model_info()
    print(f"\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    return model


def compare_correct_vs_wrong_architecture():
    """对比正确架构与错误架构的参数数量"""
    print("\n=== 正确架构 vs 错误架构对比 ===")

    # 正确架构 (46x46输入)
    correct_model = CorrectCNNAutoEncoder(latent_dim=256)
    correct_params = sum(p.numel() for p in correct_model.parameters())

    # 错误架构模拟 (91x91输入) - 参数会多很多
    wrong_input_size = 91 * 91 * 8  # 66248
    correct_input_size = 46 * 46 * 8  # 16928

    print(f"输入数据量对比:")
    print(f"  错误方法 (插值到91x91): {wrong_input_size:,} 像素")
    print(f"  正确方法 (保持46x46): {correct_input_size:,} 像素")
    print(f"  数据量减少: {wrong_input_size/correct_input_size:.1f}x")

    print(f"\n正确架构参数数量: {correct_params:,} ({correct_params/1e6:.1f}M)")

    print(f"\n正确方法的优势:")
    print(f"  1. 保持小波变换的数学正确性")
    print(f"  2. 数据量减少{wrong_input_size/correct_input_size:.1f}倍，训练更快")
    print(f"  3. 避免插值引入的噪声和失真")
    print(f"  4. 更好地利用小波的多分辨率特性")
    print(f"  5. 符合小波变换的理论基础")

    return correct_model


if __name__ == "__main__":
    # 测试正确架构
    model = test_correct_cnn_architecture()

    # 对比分析
    compare_correct_vs_wrong_architecture()