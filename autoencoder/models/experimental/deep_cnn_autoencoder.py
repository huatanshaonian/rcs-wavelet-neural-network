"""
深层CNN-AutoEncoder模型
基于用户反馈的深层架构改进
对比传统MLP-AE与CNN-AE的性能差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class DeepCNNAutoEncoder(nn.Module):
    """
    深层CNN-AutoEncoder
    架构参考传统MLP-AE: 8281→4096→2048→1024→512→256→latent_dim

    CNN vs MLP-AE 对比:
    1. 空间局部性: CNN保持空间结构，MLP完全打散
    2. 参数效率: CNN共享权重，参数数量更少
    3. 平移不变性: CNN对位置变化更鲁棒
    4. 特征提取: CNN自动学习空间模式，MLP需要更多数据
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True):
        """
        初始化深层CNN-AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量
            wavelet_bands: 小波频带数
            dropout_rate: Dropout比率
            use_attention: 是否使用注意力机制
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        # ===== 深层Encoder: 模拟8281→4096→2048→1024→512→256 =====

        # Stage 1: 输入处理 [8, 91, 91] → [32, 91, 91]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 2: [32, 91, 91] → [64, 46, 46] (相当于4096维)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 3: [64, 46, 46] → [128, 23, 23] (相当于2048维)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 4: [128, 23, 23] → [256, 12, 12] (相当于1024维)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 5: [256, 12, 12] → [512, 6, 6] (相当于512维)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 6: [512, 6, 6] → [512, 3, 3] (相当于256维)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
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

        # 最终编码层: [512, 3, 3] → latent_dim
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

        # ===== 深层Decoder: 逆向重建 =====

        # 解码全连接层
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

        # Stage 6 逆向: [512, 3, 3] → [512, 6, 6]
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Stage 5 逆向: [512, 6, 6] → [256, 12, 12]
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Stage 4 逆向: [256, 12, 12] → [128, 23, 23]
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Stage 3 逆向: [128, 23, 23] → [64, 46, 46]
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 2 逆向: [64, 46, 46] → [32, 91, 91]
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1 逆向: [32, 91, 91] → [8, 91, 91]
        self.deconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出激活，保持小波系数范围
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
        # 深层卷积编码
        x1 = self.conv1(x)      # [B, 32, 91, 91]
        x2 = self.conv2(x1)     # [B, 64, 46, 46]
        x3 = self.conv3(x2)     # [B, 128, 23, 23]
        x4 = self.conv4(x3)     # [B, 256, 12, 12]
        x5 = self.conv5(x4)     # [B, 512, 6, 6]
        x6 = self.conv6(x5)     # [B, 512, 3, 3]

        # 注意力机制 (可选)
        if self.use_attention:
            attention_weights = self.attention(x6)
            x6 = x6 * attention_weights

        # 最终编码
        latent = self.encoder_fc(x6)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码器前向传播"""
        # 重建特征图
        x = self.decoder_fc(latent)
        x = x.view(-1, 512, 3, 3)  # [B, 512, 3, 3]

        # 深层卷积解码
        x = self.deconv6(x)        # [B, 512, 6, 6]
        x = self.deconv5(x)        # [B, 256, 12, 12]
        x = self.deconv4(x)        # [B, 128, 23, 23]
        x = self.deconv3(x)        # [B, 64, 46, 46]
        x = self.deconv2(x)        # [B, 32, 91, 91]
        x = self.deconv1(x)        # [B, 8, 91, 91]

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
            'model_name': 'Deep CNN-AutoEncoder',
            'architecture': '6-stage deep CNN with attention',
            'input_shape': f'[batch, {self.input_channels}, 91, 91]',
            'latent_dim': self.latent_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'stages': [
                'Conv1: 8→32 [91×91]',
                'Conv2: 32→64 [46×46]',
                'Conv3: 64→128 [23×23]',
                'Conv4: 128→256 [12×12]',
                'Conv5: 256→512 [6×6]',
                'Conv6: 512→512 [3×3]',
                f'FC: {self.final_conv_size}→{self.latent_dim}'
            ],
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout_rate
        }


class MLPAutoEncoder(nn.Module):
    """
    传统MLP-AutoEncoder (用于对比)
    严格按照 8281→4096→2048→1024→512→256→latent_dim 架构
    """

    def __init__(self, latent_dim: int = 256, dropout_rate: float = 0.2):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = 91 * 91 * 8  # 8281 * 8 = 66328

        # 编码器
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, latent_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(4096, self.input_dim),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latent)
        return decoded.view(-1, 8, 91, 91)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def compare_architectures():
    """
    CNN-AE vs MLP-AE 架构对比分析
    """
    print("🔍 CNN-AutoEncoder vs MLP-AutoEncoder 对比分析")
    print("=" * 60)

    # 创建模型
    cnn_ae = DeepCNNAutoEncoder(latent_dim=256)
    mlp_ae = MLPAutoEncoder(latent_dim=256)

    # 参数统计
    cnn_params = sum(p.numel() for p in cnn_ae.parameters())
    mlp_params = sum(p.numel() for p in mlp_ae.parameters())

    print(f"📊 参数数量对比:")
    print(f"   CNN-AE: {cnn_params:,} 参数")
    print(f"   MLP-AE: {mlp_params:,} 参数")
    print(f"   参数比值: {mlp_params/cnn_params:.1f}x (MLP更大)")

    print(f"\n🏗️ 架构特点对比:")
    print(f"   CNN-AE优势:")
    print(f"   - 保持空间结构信息")
    print(f"   - 平移不变性，对角度变化鲁棒")
    print(f"   - 权重共享，参数效率高")
    print(f"   - 自动学习空间模式")
    print(f"   - 避免过拟合风险低")

    print(f"\n   MLP-AE特点:")
    print(f"   - 完全连接，表达能力强")
    print(f"   - 参数量大，需要更多数据")
    print(f"   - 容易过拟合")
    print(f"   - 丢失空间局部性")

    print(f"\n🔬 理论分析:")
    print(f"   对于RCS数据，CNN-AE更适合因为:")
    print(f"   1. RCS图像具有强空间相关性")
    print(f"   2. 角度变化需要平移不变性")
    print(f"   3. 小波频带间存在空间模式")
    print(f"   4. CNN的归纳偏置适合图像类数据")

    return cnn_ae, mlp_ae


if __name__ == "__main__":
    # 对比分析
    cnn_ae, mlp_ae = compare_architectures()

    # 模型信息
    print(f"\n📋 CNN-AE详细信息:")
    info = cnn_ae.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")