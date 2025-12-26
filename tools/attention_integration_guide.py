"""
通道注意力集成指南 - 适用于所有AutoEncoder

核心要点：
1. ChannelAttention类完全通用，不需要针对不同网络修改
2. 唯一参数：num_channels（输入通道数）
3. 集成只需3行代码

本文件展示如何集成到不同网络中
"""

import torch
import torch.nn as nn


# ============================================================
# Part 1: 通用的ChannelAttention类（所有网络共用）
# ============================================================

class ChannelAttention(nn.Module):
    """
    通道注意力模块 - 完全通用

    可用于：
    - WaveletAutoEncoder
    - DifferentiableWaveletAutoEncoder
    - EnhancedWaveletAutoEncoder
    - DeepWaveletAutoEncoder
    - DirectAutoEncoder（某些层）
    - 任何有通道维度的网络

    唯一需要知道的：有多少个通道（num_channels）
    """

    def __init__(self, num_channels, reduction=4):
        """
        Args:
            num_channels: 输入通道数
                - Wavelet模式(2freq): 8 (2×4)
                - Wavelet模式(3freq): 12 (3×4)
                - Direct模式: 2 (2个频率)
                - 中间层: 任意（如32, 64, 128）
            reduction: 压缩比，控制中间层大小
                - 建议：通道数<16时用2，≥16时用4
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 瓶颈层
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
            x: [B, C, H, W] 任意形状的特征图

        Returns:
            weighted_x: [B, C, H, W] 加权后的特征图
        """
        batch, channels, _, _ = x.size()

        # 全局池化
        avg_out = self.fc(self.avg_pool(x).view(batch, channels))
        max_out = self.fc(self.max_pool(x).view(batch, channels))

        # 生成权重
        channel_weights = self.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)

        # 加权
        return x * channel_weights


# ============================================================
# Part 2: 集成示例1 - WaveletAutoEncoder（Wavelet模式）
# ============================================================

class WaveletAutoEncoderWithAttention(nn.Module):
    """示例：在WaveletAutoEncoder中集成通道注意力"""

    def __init__(self, latent_dim=256, num_frequencies=2, dropout_rate=0.2, input_size=49):
        super().__init__()

        self.num_frequencies = num_frequencies
        self.input_channels = num_frequencies * 4  # 2freq×4bands = 8

        # ===== 添加通道注意力（1行）=====
        self.channel_attention = ChannelAttention(
            num_channels=self.input_channels,  # 8
            reduction=2  # 8→4→8
        )

        # ===== 原有网络结构（不需要修改）=====
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # ... 其他层
        )
        # ... decoder等

    def encode(self, wavelet_coeffs):
        # [B, H, W, C] → [B, C, H, W]
        x = wavelet_coeffs.permute(0, 3, 1, 2)

        # ===== 应用通道注意力（1行）=====
        x = self.channel_attention(x)

        # ===== 原有编码流程（不需要修改）=====
        features = self.encoder(x)
        # ...
        return features


# ============================================================
# Part 3: 集成示例2 - DifferentiableWaveletAutoEncoder
# ============================================================

class DifferentiableWaveletAutoEncoderWithAttention(nn.Module):
    """示例：在DifferentiableWaveletAutoEncoder中集成"""

    def __init__(self, latent_dim=256, num_frequencies=2, wavelet_type='db4', dropout_rate=0.2):
        super().__init__()

        self.num_frequencies = num_frequencies
        self.input_channels = num_frequencies * 4

        # ===== 完全相同的注意力模块（不需要修改）=====
        self.channel_attention = ChannelAttention(
            num_channels=self.input_channels,  # 8（与Wavelet相同）
            reduction=2
        )

        # ===== 集成可微分小波变换 =====
        from differentiable_wavelet_transform import DifferentiableWaveletTransform
        self.wavelet_transform = DifferentiableWaveletTransform(wavelet=wavelet_type)

        # ===== 原有CNN网络 =====
        self.encoder = nn.Sequential(...)
        # ...

    def encode(self, rcs_data):
        # RCS → 小波系数（可微分）
        wavelet_coeffs = self.wavelet_transform.forward_transform(rcs_data)  # [B, H, W, 8]

        # 转换格式
        x = wavelet_coeffs.permute(0, 3, 1, 2)  # [B, 8, H, W]

        # ===== 应用通道注意力（完全相同的调用）=====
        x = self.channel_attention(x)

        # 原有编码
        features = self.encoder(x)
        # ...
        return features


# ============================================================
# Part 4: 集成示例3 - DirectAutoEncoder（Direct模式）
# ============================================================

class DirectAutoEncoderWithAttention(nn.Module):
    """
    示例：在DirectAutoEncoder中集成

    Direct模式的通道数少（只有频率数），但同样可用
    """

    def __init__(self, latent_dim=256, num_frequencies=2, dropout_rate=0.2):
        super().__init__()

        self.num_frequencies = num_frequencies  # Direct模式: 2

        # ===== 注意力模块（参数不同，但类相同）=====
        self.channel_attention = ChannelAttention(
            num_channels=self.num_frequencies,  # 2（不是8！）
            reduction=1  # 通道数太少，不压缩
        )

        # ===== 原有网络 =====
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_frequencies, 32, kernel_size=3, padding=1),
            # ...
        )

    def encode(self, rcs_data):
        # [B, H, W, C] → [B, C, H, W]
        x = rcs_data.permute(0, 3, 1, 2)  # [B, 2, 91, 91]

        # ===== 应用通道注意力（调用方式完全相同）=====
        x = self.channel_attention(x)

        # 原有编码
        features = self.encoder(x)
        # ...
        return features


# ============================================================
# Part 5: 高级用法 - 多处添加注意力
# ============================================================

class MultiAttentionAutoEncoder(nn.Module):
    """示例：在多个位置添加注意力"""

    def __init__(self, latent_dim=256, num_frequencies=2, dropout_rate=0.2):
        super().__init__()

        input_channels = num_frequencies * 4  # 8

        # ===== 注意力1: 输入层 - 小波通道注意力 =====
        self.input_attention = ChannelAttention(
            num_channels=input_channels,  # 8
            reduction=2
        )

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)

        # ===== 注意力2: 中间层 - 特征通道注意力 =====
        self.mid_attention = ChannelAttention(
            num_channels=32,  # 第一层卷积输出的通道数
            reduction=4
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # ===== 注意力3: 深层 - 高级特征注意力 =====
        self.deep_attention = ChannelAttention(
            num_channels=64,
            reduction=8
        )

        # ...后续层

    def encode(self, wavelet_coeffs):
        x = wavelet_coeffs.permute(0, 3, 1, 2)

        # ===== 多级注意力 =====
        x = self.input_attention(x)   # 第1级：小波通道选择
        x = self.conv1(x)

        x = self.mid_attention(x)     # 第2级：浅层特征选择
        x = self.conv2(x)

        x = self.deep_attention(x)    # 第3级：深层特征选择
        # ...

        return x


# ============================================================
# Part 6: 参数指南
# ============================================================

def get_attention_config(mode, num_frequencies):
    """
    根据模式和频率数，返回注意力配置

    Args:
        mode: 'wavelet', 'direct', 'differentiable_wavelet'
        num_frequencies: 2 or 3

    Returns:
        dict: {
            'num_channels': 通道数,
            'reduction': 推荐的压缩比
        }
    """
    if mode in ('wavelet', 'differentiable_wavelet'):
        num_channels = num_frequencies * 4
    elif mode == 'direct':
        num_channels = num_frequencies
    else:
        raise ValueError(f"未知模式: {mode}")

    # 根据通道数选择合适的reduction
    if num_channels <= 4:
        reduction = 1  # 不压缩
    elif num_channels <= 8:
        reduction = 2  # 8→4→8
    else:
        reduction = 4  # 12→3→12

    return {
        'num_channels': num_channels,
        'reduction': reduction
    }


# ============================================================
# Part 7: 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("通道注意力集成示例")
    print("=" * 60)

    # ===== 示例1: Wavelet模式（2频率）=====
    print("\n【示例1】Wavelet模式（2频率）")
    config1 = get_attention_config('wavelet', num_frequencies=2)
    print(f"  配置: {config1}")

    attention1 = ChannelAttention(**config1)
    test_input1 = torch.randn(4, 8, 49, 49)  # [B, 2freq×4bands, H, W]
    output1 = attention1(test_input1)
    print(f"  输入形状: {test_input1.shape}")
    print(f"  输出形状: {output1.shape}")
    print(f"  ✓ 成功！")

    # ===== 示例2: Wavelet模式（3频率）=====
    print("\n【示例2】Wavelet模式（3频率）")
    config2 = get_attention_config('wavelet', num_frequencies=3)
    print(f"  配置: {config2}")

    attention2 = ChannelAttention(**config2)
    test_input2 = torch.randn(4, 12, 49, 49)  # [B, 3freq×4bands, H, W]
    output2 = attention2(test_input2)
    print(f"  输入形状: {test_input2.shape}")
    print(f"  输出形状: {output2.shape}")
    print(f"  ✓ 成功！")

    # ===== 示例3: Direct模式（2频率）=====
    print("\n【示例3】Direct模式（2频率）")
    config3 = get_attention_config('direct', num_frequencies=2)
    print(f"  配置: {config3}")

    attention3 = ChannelAttention(**config3)
    test_input3 = torch.randn(4, 2, 91, 91)  # [B, 2freq, H, W]
    output3 = attention3(test_input3)
    print(f"  输入形状: {test_input3.shape}")
    print(f"  输出形状: {output3.shape}")
    print(f"  ✓ 成功！")

    # ===== 示例4: 中间层特征（任意通道数）=====
    print("\n【示例4】中间层特征（任意通道数）")
    attention4 = ChannelAttention(num_channels=64, reduction=8)
    test_input4 = torch.randn(4, 64, 25, 25)  # [B, 64, H, W]
    output4 = attention4(test_input4)
    print(f"  输入形状: {test_input4.shape}")
    print(f"  输出形状: {output4.shape}")
    print(f"  ✓ 成功！")

    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print("✓ 同一个ChannelAttention类适用于所有网络")
    print("✓ 只需修改num_channels参数")
    print("✓ 集成只需3行代码:")
    print("    1. 创建: self.attention = ChannelAttention(...)")
    print("    2. 调用: x = self.attention(x)")
    print("    3. 完成！")
    print("=" * 60)
