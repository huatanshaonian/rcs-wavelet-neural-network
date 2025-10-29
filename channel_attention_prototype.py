"""
通道注意力原型 - 最简单的LL/高频分离方案
可以快速集成到现有WaveletAutoEncoder中验证效果
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块

    自动学习每个通道（LL, LH, HL, HH）的重要性权重
    可以让网络自适应地调整LL和高频通道的贡献
    """

    def __init__(self, num_channels, reduction=4):
        """
        Args:
            num_channels: 输入通道数（如8 = 2freq×4bands）
            reduction: 压缩比（控制中间层大小）
        """
        super().__init__()

        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP：channels → channels//reduction → channels
        hidden_dim = max(num_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 小波系数

        Returns:
            weighted_x: [B, C, H, W] 加权后的小波系数
        """
        batch, channels, _, _ = x.size()

        # 全局池化 → [B, C, 1, 1] → [B, C]
        avg_out = self.fc(self.avg_pool(x).view(batch, channels))
        max_out = self.fc(self.max_pool(x).view(batch, channels))

        # 生成通道权重 [B, C] → [B, C, 1, 1]
        channel_weights = self.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)

        # 预期效果：LL通道权重接近0.7-0.8，高频通道0.2-0.3
        # （网络自动学习）

        return x * channel_weights  # 逐通道加权


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    原始论文: "Squeeze-and-Excitation Networks" (CVPR 2018)
    这是通道注意力的经典实现
    """

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # Squeeze: 全局平均池化
        squeezed = self.squeeze(x).view(batch, channels)

        # Excitation: 生成通道权重
        channel_weights = self.excitation(squeezed).view(batch, channels, 1, 1)

        return x * channel_weights


# ============= 集成示例 =============

class WaveletAutoEncoderWithAttention(nn.Module):
    """
    带通道注意力的小波AutoEncoder

    只需在第一层卷积前添加通道注意力模块
    """

    def __init__(self, latent_dim=256, num_frequencies=2, dropout_rate=0.2, input_size=49):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.input_channels = num_frequencies * 4  # 2freq × 4bands = 8

        # ===== 关键：添加通道注意力 =====
        self.channel_attention = ChannelAttention(
            num_channels=self.input_channels,
            reduction=2  # 8 → 4 → 8（reduction不宜太大）
        )

        # ===== 原有Encoder结构 =====
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.flattened_size = 256 * 4 * 4

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, latent_dim)
        )

        # Decoder省略...

    def encode(self, wavelet_coeffs):
        """
        Args:
            wavelet_coeffs: [B, H, W, C] 小波系数

        Returns:
            latent: [B, latent_dim]
        """
        # 转换为 [B, C, H, W]
        x = wavelet_coeffs.permute(0, 3, 1, 2)

        # ===== 关键：应用通道注意力 =====
        # 在第一层卷积前重新加权通道
        x = self.channel_attention(x)

        # 原有编码流程
        features = self.encoder(x)
        latent = self.encoder_fc(features)

        return latent


# ============= 使用示例 =============

if __name__ == "__main__":
    # 创建模型
    model = WaveletAutoEncoderWithAttention(
        latent_dim=256,
        num_frequencies=2,
        dropout_rate=0.2,
        input_size=49
    )

    # 测试输入
    batch_size = 4
    wavelet_coeffs = torch.randn(batch_size, 49, 49, 8)  # [B, H, W, 2freq×4bands]

    # 前向传播
    latent = model.encode(wavelet_coeffs)

    print(f"输入形状: {wavelet_coeffs.shape}")
    print(f"隐空间形状: {latent.shape}")
    print(f"\nPASS 通道注意力原型测试通过!")

    # 查看通道注意力权重（训练后）
    # 预期：LL通道权重 > 高频通道权重
    print(f"\n提示：训练后可以通过以下方式查看学到的通道权重：")
    print(f"  with torch.no_grad():")
    print(f"      x = wavelet_coeffs.permute(0, 3, 1, 2)")
    print(f"      weights = model.channel_attention(x)")
    print(f"      # 观察LL通道 vs 高频通道的权重差异")
