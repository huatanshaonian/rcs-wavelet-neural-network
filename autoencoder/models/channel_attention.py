"""
通道注意力模块 (Channel Attention Module)

实现了Squeeze-and-Excitation风格的通道注意力机制，可用于所有AutoEncoder模型。
通过学习通道间的依赖关系，自适应地重新加权不同通道的贡献。

核心特性：
1. 完全通用 - 只需指定通道数即可
2. 即插即用 - 无需修改现有网络架构
3. 开销小 - 参数量和计算量都很小
4. 有效性 - 对小波系数特别有效（自动学习LL vs 高频权重）

使用方法：
    from autoencoder.models.channel_attention import ChannelAttention

    # 在AutoEncoder中集成
    self.channel_attention = ChannelAttention(num_channels=8, reduction=2)

    # 在forward中应用
    x = self.channel_attention(x)  # [B, C, H, W] -> [B, C, H, W]

参考文献：
    Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块

    通过全局池化 + 瓶颈MLP的方式学习通道间的依赖关系，
    生成每个通道的权重，实现自适应的通道重要性调整。

    工作流程：
    1. Squeeze: 全局平均池化 + 全局最大池化
    2. Excitation: 通过MLP学习通道权重 (C -> C/r -> C)
    3. Scale: 用学到的权重对输入进行加权

    Args:
        num_channels (int): 输入通道数
            - Wavelet模式(2freq): 8 (2×4bands)
            - Wavelet模式(3freq): 12 (3×4bands)
            - Direct模式: 2 (2个频率)
            - 中间层: 任意（如32, 64, 128）
        reduction (int): 压缩比，控制瓶颈层大小
            - 建议：通道数<16时用2，≥16时用4
            - 默认: 4

    Input:
        x: [B, C, H, W] 任意形状的特征图

    Output:
        weighted_x: [B, C, H, W] 加权后的特征图（形状不变）

    Example:
        >>> attention = ChannelAttention(num_channels=8, reduction=2)
        >>> x = torch.randn(4, 8, 49, 49)  # Wavelet coefficients
        >>> out = attention(x)
        >>> print(out.shape)  # torch.Size([4, 8, 49, 49])
    """

    def __init__(self, num_channels, reduction=4):
        super().__init__()

        self.num_channels = num_channels
        self.reduction = reduction

        # 全局池化层（Squeeze操作）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]

        # 瓶颈MLP（Excitation操作）
        # C -> C/r -> C
        hidden_dim = max(num_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels, bias=False)
        )

        # 归一化到[0, 1]
        self.sigmoid = nn.Sigmoid()

        # 保存最近一次的注意力权重（用于可视化和分析）
        self.last_attention_weights = None

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, C, H, W] 输入特征图

        Returns:
            weighted_x: [B, C, H, W] 加权后的特征图
        """
        batch, channels, _, _ = x.size()

        # Squeeze: 全局池化 -> [B, C, 1, 1] -> [B, C]
        avg_out = self.fc(self.avg_pool(x).view(batch, channels))  # [B, C]
        max_out = self.fc(self.max_pool(x).view(batch, channels))  # [B, C]

        # Excitation: 生成通道权重 [B, C] -> [B, C, 1, 1]
        channel_weights = self.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)

        # 保存权重用于后续分析（取batch平均值）
        with torch.no_grad():
            self.last_attention_weights = channel_weights.squeeze().mean(dim=0).cpu().numpy()

        # Scale: 逐通道加权
        # 预期效果（对小波系数）：
        #   - LL通道权重 ≈ 0.7-0.8（高权重）
        #   - LH/HL/HH通道权重 ≈ 0.2-0.3（低权重）
        # 网络会自动学习这种权重分配

        return x * channel_weights

    def get_attention_weights(self):
        """
        获取最近一次前向传播的注意力权重

        Returns:
            numpy.ndarray: [C] 每个通道的注意力权重，范围[0, 1]
                           如果还没有运行过forward，返回None

        Example:
            >>> attention = ChannelAttention(num_channels=8)
            >>> x = torch.randn(4, 8, 49, 49)
            >>> _ = attention(x)
            >>> weights = attention.get_attention_weights()
            >>> print(f"通道权重: {weights}")
        """
        return self.last_attention_weights

    def __repr__(self):
        return f"ChannelAttention(num_channels={self.num_channels}, reduction={self.reduction})"


def get_recommended_reduction(num_channels):
    """
    根据通道数推荐合适的压缩比

    Args:
        num_channels (int): 输入通道数

    Returns:
        int: 推荐的压缩比

    规则：
        - ≤4通道: reduction=1 (不压缩)
        - 5-8通道: reduction=2
        - 9-16通道: reduction=4
        - >16通道: reduction=8

    Example:
        >>> get_recommended_reduction(2)   # Direct mode
        1
        >>> get_recommended_reduction(8)   # Wavelet 2freq
        2
        >>> get_recommended_reduction(12)  # Wavelet 3freq
        4
        >>> get_recommended_reduction(64)  # Intermediate layer
        8
    """
    if num_channels <= 4:
        return 1
    elif num_channels <= 8:
        return 2
    elif num_channels <= 16:
        return 4
    else:
        return 8


if __name__ == "__main__":
    print("=" * 60)
    print("通道注意力模块测试")
    print("=" * 60)

    # 测试1: Wavelet模式 (2freq)
    print("\n[Test 1] Wavelet模式 (2freq): 8通道")
    attention1 = ChannelAttention(num_channels=8, reduction=2)
    x1 = torch.randn(4, 8, 49, 49)
    out1 = attention1(x1)
    print(f"  输入形状: {x1.shape}")
    print(f"  输出形状: {out1.shape}")
    print(f"  参数量: {sum(p.numel() for p in attention1.parameters()):,}")
    print("  PASS")

    # 测试2: Wavelet模式 (3freq)
    print("\n[Test 2] Wavelet模式 (3freq): 12通道")
    attention2 = ChannelAttention(num_channels=12, reduction=4)
    x2 = torch.randn(4, 12, 49, 49)
    out2 = attention2(x2)
    print(f"  输入形状: {x2.shape}")
    print(f"  输出形状: {out2.shape}")
    print(f"  参数量: {sum(p.numel() for p in attention2.parameters()):,}")
    print("  PASS")

    # 测试3: Direct模式
    print("\n[Test 3] Direct模式: 2通道")
    attention3 = ChannelAttention(num_channels=2, reduction=1)
    x3 = torch.randn(4, 2, 91, 91)
    out3 = attention3(x3)
    print(f"  输入形状: {x3.shape}")
    print(f"  输出形状: {out3.shape}")
    print(f"  参数量: {sum(p.numel() for p in attention3.parameters()):,}")
    print("  PASS")

    # 测试4: 中间层特征
    print("\n[Test 4] 中间层特征: 64通道")
    attention4 = ChannelAttention(num_channels=64, reduction=8)
    x4 = torch.randn(4, 64, 25, 25)
    out4 = attention4(x4)
    print(f"  输入形状: {x4.shape}")
    print(f"  输出形状: {out4.shape}")
    print(f"  参数量: {sum(p.numel() for p in attention4.parameters()):,}")
    print("  PASS")

    # 测试5: 推荐压缩比
    print("\n[Test 5] 推荐压缩比测试")
    test_channels = [2, 8, 12, 32, 64, 128]
    for nc in test_channels:
        rec = get_recommended_reduction(nc)
        print(f"  {nc}通道 -> reduction={rec}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
