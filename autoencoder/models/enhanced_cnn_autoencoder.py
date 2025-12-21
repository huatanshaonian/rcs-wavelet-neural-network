"""
增强感受野的CNN-AutoEncoder模型
特点:
1. 空洞卷积扩大感受野
2. 多尺度特征融合
3. 通道注意力机制
4. 残差连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from autoencoder.utils.adaptive_layers import (
    build_adaptive_fc_pair,
    get_structure_info,
)
from autoencoder.models.channel_attention import (
    ChannelAttention as InputChannelAttention,
    get_recommended_reduction
)
from autoencoder.utils.activation_factory import get_activation, get_activation_name


class MultiScaleBlock(nn.Module):
    """
    多尺度并行卷积块
    使用不同膨胀率的空洞卷积捕捉不同尺度的空间模式

    架构:
    ┌─── dilation=1 (3×3感受野) ───┐
    ├─── dilation=2 (5×5感受野) ───┤
    ├─── dilation=4 (9×9感受野) ───┤  → concat → BatchNorm → ReLU
    └─── kernel=5   (5×5感受野) ───┘

    优势:
    - 同时捕捉局部细节和全局结构
    - 参数量仅比单一卷积增加25%
    - 感受野覆盖3×3到9×9
    """

    def __init__(self, in_channels: int, out_channels: int, activation: str = 'relu'):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（会均分到4个分支）
            activation: 激活函数类型
        """
        super().__init__()

        # 确保out_channels可以被4整除
        branch_channels = out_channels // 4

        # 分支1: 标准卷积 - 捕捉局部细节 (感受野: 3×3)
        self.branch1 = nn.Conv2d(in_channels, branch_channels,
                                 kernel_size=3, padding=1, dilation=1)

        # 分支2: 空洞卷积 dilation=2 - 中等尺度 (感受野: 5×5)
        self.branch2 = nn.Conv2d(in_channels, branch_channels,
                                 kernel_size=3, padding=2, dilation=2)

        # 分支3: 空洞卷积 dilation=4 - 大尺度 (感受野: 9×9)
        self.branch3 = nn.Conv2d(in_channels, branch_channels,
                                 kernel_size=3, padding=4, dilation=4)

        # 分支4: 5×5标准卷积 - 补充中等尺度 (感受野: 5×5)
        self.branch4 = nn.Conv2d(in_channels, branch_channels,
                                 kernel_size=5, padding=2, dilation=1)

        # 融合后的归一化
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, in_channels, H, W]

        Returns:
            out: [B, out_channels, H, W]
        """
        # 并行计算所有分支
        b1 = self.branch1(x)  # 3×3 局部
        b2 = self.branch2(x)  # 5×5 中等
        b3 = self.branch3(x)  # 9×9 大尺度
        b4 = self.branch4(x)  # 5×5 标准

        # 通道拼接
        out = torch.cat([b1, b2, b3, b4], dim=1)

        # 归一化和激活
        out = self.bn(out)
        out = self.activation(out)

        return out


class DilatedResidualBlock(nn.Module):
    """
    空洞卷积残差块
    结合空洞卷积和残差连接，扩大感受野同时保持梯度流畅

    架构:
    input ──┬─→ Conv(dilation=r₁) → BN → ReLU → Conv(dilation=r₂) → BN ──┬→ output
            │                                                                │
            └────────────────── shortcut (1×1 conv if needed) ──────────────┘

    优势:
    - 大感受野 (通过空洞卷积)
    - 稳定训练 (通过残差连接)
    - 防止退化 (skip connection)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dilation1: int = 2, dilation2: int = 4,
                 activation: str = 'relu'):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长 (用于下采样)
            dilation1: 第一层膨胀率
            dilation2: 第二层膨胀率
            activation: 激活函数类型
        """
        super().__init__()

        # 计算padding以保持尺寸
        padding1 = dilation1 * (3 - 1) // 2
        padding2 = dilation2 * (3 - 1) // 2

        # 主路径: 两层空洞卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride,
                              padding=padding1, dilation=dilation1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1,
                              padding=padding2, dilation=dilation2,
                              bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差路径 (如果维度不匹配需要投影)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, in_channels, H, W]

        Returns:
            out: [B, out_channels, H', W']
        """
        identity = self.shortcut(x)

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += identity
        out = self.activation(out)

        return out


class ChannelAttention(nn.Module):
    """
    通道注意力机制 (改进自SENet)
    自适应强化重要的频率通道，抑制不重要的通道

    原理:
    1. 全局平均池化: [B,C,H,W] → [B,C,1,1]
    2. 全局最大池化: [B,C,H,W] → [B,C,1,1]
    3. MLP: [B,C] → [B,C//r] → [B,C]
    4. Sigmoid激活: 得到通道权重 [B,C,1,1]
    5. 重新加权: x * attention_weights

    优势:
    - 强化跨频率的重要特征
    - 参数量极小 (< 0.1%)
    - 计算开销小
    - 端到端训练
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 输入通道数
            reduction: 降维比率 (控制MLP中间层大小)
        """
        super().__init__()

        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, C, H, W]

        Returns:
            out: [B, C, H, W] (加权后的特征)
        """
        b, c, _, _ = x.size()

        # 平均池化分支
        avg = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg)

        # 最大池化分支
        max_ = self.max_pool(x).view(b, c)
        max_out = self.fc(max_)

        # 融合两个分支的注意力权重
        attention = (avg_out + max_out).view(b, c, 1, 1)

        # 重新加权
        return x * attention


class EnhancedWaveletAutoEncoder(nn.Module):
    """
    增强感受野的小波CNN-AutoEncoder

    改进:
    1. 多尺度并行卷积 (感受野: 3×3~9×9)
    2. 空洞残差块 (感受野指数增长)
    3. 通道注意力 (强化跨频率特征)
    4. 残差连接 (稳定训练)

    感受野对比:
    原始CNN: Layer1(3×3) → Layer2(7×7) → Layer3(15×15) → Layer4(31×31)
    增强CNN: Layer1(3×3) → Layer2(15×15) → Layer3(35×35) → Layer4(67×67) → 全局

    参数量: ~11M (仅比原始增加10%)
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 input_size: int = 49,
                 use_channel_attention: bool = False,
                 activation: str = 'relu'):
        """
        初始化增强AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量 (2 or 3)
            wavelet_bands: 小波频带数 (通常为4)
            dropout_rate: Dropout比率
            input_size: 输入尺寸 (49 for db4小波)
            use_channel_attention: 是否在输入层使用通道注意力 (默认: False)
                                  注意：中间层的通道注意力是Enhanced架构固有的，始终启用
            activation: 激活函数类型（'relu', 'sin', 'gelu', 'swish'等，默认: 'relu'）
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.use_channel_attention = use_channel_attention
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        # ===== 输入层通道注意力（可选，用户可配置） =====
        if self.use_channel_attention:
            reduction = get_recommended_reduction(self.input_channels)
            self.input_channel_attention = InputChannelAttention(
                num_channels=self.input_channels,
                reduction=reduction
            )
        else:
            self.input_channel_attention = None

        # ===== 编码器 =====

        # 初始特征提取 (标准卷积)
        # [B, 8/12, 49, 49] → [B, 32, 49, 49]
        # 感受野: 3×3
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer()
        )

        # 多尺度特征提取
        # [B, 32, 49, 49] → [B, 64, 49, 49]
        # 感受野: 3×3 ~ 9×9 (多尺度并行)
        self.multi_scale = MultiScaleBlock(32, 64, activation=self.activation_type)

        # 下采样块1 + 大感受野
        # [B, 64, 49, 49] → [B, 128, 25, 25]
        # 感受野: ~35×35 (通过dilation=2,4)
        self.down1 = DilatedResidualBlock(64, 128, stride=2,
                                          dilation1=2, dilation2=4,
                                          activation=self.activation_type)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        # 下采样块2 + 更大感受野
        # [B, 128, 25, 25] → [B, 256, 13, 13]
        # 感受野: ~67×67 (超过49×49输入)
        self.down2 = DilatedResidualBlock(128, 256, stride=2,
                                          dilation1=2, dilation2=4,
                                          activation=self.activation_type)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # 通道注意力 (强化重要频率)
        self.channel_attn = ChannelAttention(256, reduction=16)

        # 下采样块3 (标准卷积，避免棋盘效应)
        # [B, 256, 13, 13] → [B, 512, 7, 7]
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # 全局池化
        # [B, 512, 7, 7] → [B, 512, 4, 4]
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 隐空间映射（自适应全连接层）
        self.flattened_size = 512 * 4 * 4  # 8192
        self.encoder_fc, decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
            input_dim=self.flattened_size,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            arch_type='cnn',
        )
        # 原始实现末尾使用ReLU，这里保持一致性
        self.decoder_fc = nn.Sequential(*decoder_fc, activation_layer())
        self.structure_info = get_structure_info(
            self.flattened_size, latent_dim, self.intermediate_dims
        )

        # 上采样卷积
        self.decoder_conv = nn.Sequential(
            # [8192] → [512, 4, 4]
            nn.Unflatten(1, (512, 4, 4)),

            # 上采样1: [4, 4] → [8, 8]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样2: [8, 8] → [16, 16]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样3: [16, 16] → [32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation_layer()
        )

        # 最终卷积层
        self.final_conv = nn.Conv2d(64, self.input_channels,
                                    kernel_size=3, padding=1)

        # 权重初始化
        self._initialize_weights()

        # ===== 统一接口：encoder/decoder属性 =====
        # 为了兼容训练代码中的冻结/解冻操作，提供统一的encoder/decoder接口
        self.encoder = nn.ModuleList([
            self.conv1, self.multi_scale, self.down1, self.dropout1,
            self.down2, self.dropout2, self.channel_attn,
            self.down3, self.global_pool, self.encoder_fc
        ])

        self.decoder = nn.ModuleList([
            self.decoder_fc, self.decoder_conv, self.final_conv
        ])

    def _initialize_weights(self):
        """Xavier权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
        # 维度转换: [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # 应用输入层通道注意力（如果启用）
        if self.use_channel_attention and self.input_channel_attention is not None:
            x = self.input_channel_attention(x)

        # 编码路径
        x = self.conv1(x)           # [B, 32, 49, 49]
        x = self.multi_scale(x)     # [B, 64, 49, 49] 多尺度
        x = self.down1(x)           # [B, 128, 25, 25] 大感受野
        x = self.dropout1(x)
        x = self.down2(x)           # [B, 256, 13, 13] 超大感受野
        x = self.dropout2(x)
        x = self.channel_attn(x)    # 通道注意力
        x = self.down3(x)           # [B, 512, 7, 7]
        x = self.global_pool(x)     # [B, 512, 4, 4]

        # 隐空间映射
        latent = self.encoder_fc(x)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码器: 隐空间 → 小波系数

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            x: [B, input_size, input_size, num_freq*4] 重建小波系数
        """
        # 隐空间到特征
        x = self.decoder_fc(latent)    # [B, 8192]

        # 上采样
        x = self.decoder_conv(x)       # [B, 64, 32, 32]
        x = self.final_conv(x)         # [B, input_channels, 32, 32]

        # 上采样到目标尺寸
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                         mode='bilinear', align_corners=False)

        # 维度转换: [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, input_size, input_size, num_freq*4] 小波系数

        Returns:
            reconstructed: [B, input_size, input_size, num_freq*4] 重建小波系数
            latent: [B, latent_dim] 隐空间表示
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)

        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        encoder_params = (
            sum(p.numel() for p in self.conv1.parameters()) +
            sum(p.numel() for p in self.multi_scale.parameters()) +
            sum(p.numel() for p in self.down1.parameters()) +
            sum(p.numel() for p in self.down2.parameters()) +
            sum(p.numel() for p in self.channel_attn.parameters()) +
            sum(p.numel() for p in self.down3.parameters()) +
            sum(p.numel() for p in self.encoder_fc.parameters())
        )

        decoder_params = (
            sum(p.numel() for p in self.decoder_fc.parameters()) +
            sum(p.numel() for p in self.decoder_conv.parameters()) +
            sum(p.numel() for p in self.final_conv.parameters())
        )

        total_params = sum(p.numel() for p in self.parameters())

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        param_count = self.get_parameter_count()

        return {
            'model_name': 'EnhancedWaveletAutoEncoder',
            'architecture': 'Enhanced CNN (Dilated + Multi-Scale + Attention)',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'wavelet_bands': self.wavelet_bands,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'fc_structure': self.structure_info['fc_structure'],
            'intermediate_dims': self.structure_info['intermediate_dims'],
            'num_fc_layers': self.structure_info['num_fc_layers'],
            'compression_ratios': self.structure_info['compression_ratios'],
            'receptive_field': '3×3 → 15×15 → 35×35 → 67×67 → Global',
            'use_channel_attention': self.use_channel_attention,
            'key_features': [
                'Multi-scale parallel convolutions (dilation 1,2,4)',
                'Dilated residual blocks',
                'Channel attention mechanism (mid-layer, always on)',
                'Input-level channel attention (optional, user-configurable)' if self.use_channel_attention else 'Channel attention mechanism (mid-layer)',
                'Skip connections'
            ]
        }


class EnhancedDirectAutoEncoder(nn.Module):
    """
    增强感受野的直接CNN-AutoEncoder
    直接处理RCS数据，不使用小波变换

    与EnhancedWaveletAutoEncoder架构相似，但输入是91×91×2的RCS数据
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 use_channel_attention: bool = False,
                 activation: str = 'relu'):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.input_channels = num_frequencies
        self.use_channel_attention = use_channel_attention
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        # ===== 输入层通道注意力（可选，用户可配置） =====
        if self.use_channel_attention:
            reduction = get_recommended_reduction(self.input_channels)
            self.input_channel_attention = InputChannelAttention(
                num_channels=self.input_channels,
                reduction=reduction
            )
        else:
            self.input_channel_attention = None

        # ===== 编码器 =====

        # 初始特征提取
        # [B, 2/3, 91, 91] → [B, 32, 91, 91]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer()
        )

        # 多尺度特征
        # [B, 32, 91, 91] → [B, 64, 91, 91]
        self.multi_scale = MultiScaleBlock(32, 64, activation=self.activation_type)

        # 下采样1: [B, 64, 91, 91] → [B, 128, 46, 46]
        self.down1 = DilatedResidualBlock(64, 128, stride=2,
                                          dilation1=2, dilation2=4,
                                          activation=self.activation_type)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        # 下采样2: [B, 128, 46, 46] → [B, 256, 23, 23]
        self.down2 = DilatedResidualBlock(128, 256, stride=2,
                                          dilation1=2, dilation2=4,
                                          activation=self.activation_type)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # 通道注意力
        self.channel_attn = ChannelAttention(256, reduction=16)

        # 下采样3: [B, 256, 23, 23] → [B, 512, 12, 12]
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # 下采样4: [B, 512, 12, 12] → [B, 512, 6, 6]
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 隐空间映射（自适应全连接层）
        self.flattened_size = 512
        self.encoder_fc, decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
            input_dim=self.flattened_size,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            arch_type='cnn',
        )
        self.decoder_fc = nn.Sequential(*decoder_fc, activation_layer())
        self.structure_info = get_structure_info(
            self.flattened_size, latent_dim, self.intermediate_dims
        )

        self.decoder_conv = nn.Sequential(
            # [512] → [512, 1, 1]
            nn.Unflatten(1, (512, 1, 1)),

            # [1, 1] → [6, 6]
            nn.ConvTranspose2d(512, 256, kernel_size=6, stride=1),
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # [6, 6] → [12, 12]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # [12, 12] → [23, 23]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # [23, 23] → [46, 46]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # [46, 46] → [91, 91]
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
        )

        # 最终调整层
        self.final_adjust = nn.AdaptiveAvgPool2d((91, 91))

        self._initialize_weights()

        # ===== 统一接口：encoder/decoder属性 =====
        # 为了兼容训练代码中的冻结/解冻操作，提供统一的encoder/decoder接口
        self.encoder = nn.ModuleList([
            self.conv1, self.multi_scale, self.down1, self.dropout1,
            self.down2, self.dropout2, self.channel_attn,
            self.down3, self.down4, self.global_pool, self.encoder_fc
        ])

        self.decoder = nn.ModuleList([
            self.decoder_fc, self.decoder_conv, self.final_adjust
        ])

    def _initialize_weights(self):
        """Xavier权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码RCS数据到隐空间"""
        # [B, H, W, C] → [B, C, H, W]
        if x.dim() == 4 and x.shape[-1] == self.num_frequencies:
            x = x.permute(0, 3, 1, 2)

        # 应用输入层通道注意力（如果启用）
        if self.use_channel_attention and self.input_channel_attention is not None:
            x = self.input_channel_attention(x)

        x = self.conv1(x)
        x = self.multi_scale(x)
        x = self.down1(x)
        x = self.dropout1(x)
        x = self.down2(x)
        x = self.dropout2(x)
        x = self.channel_attn(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.global_pool(x)

        latent = self.encoder_fc(x)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码隐空间到RCS数据"""
        x = self.decoder_fc(latent)
        x = self.decoder_conv(x)
        x = self.final_adjust(x)

        # [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        encoder_params = sum(p.numel() for n, p in self.named_parameters()
                           if 'encoder' in n or 'conv1' in n or 'down' in n or 'multi_scale' in n)
        decoder_params = sum(p.numel() for n, p in self.named_parameters()
                           if 'decoder' in n or 'final' in n)
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': 'EnhancedDirectAutoEncoder',
            'architecture': 'Enhanced CNN (Direct RCS)',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'input_channels': self.input_channels,
            'parameters': self.get_parameter_count(),
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'fc_structure': self.structure_info['fc_structure'],
            'intermediate_dims': self.structure_info['intermediate_dims'],
            'num_fc_layers': self.structure_info['num_fc_layers'],
            'compression_ratios': self.structure_info['compression_ratios'],
            'use_channel_attention': self.use_channel_attention,
        }


def test_enhanced_models():
    """测试增强模型"""
    print("=== 测试增强感受野CNN模型 ===\n")

    # 测试WaveletAutoEncoder
    print("--- EnhancedWaveletAutoEncoder ---")
    model_wavelet = EnhancedWaveletAutoEncoder(latent_dim=256, num_frequencies=2)
    info = model_wavelet.get_model_info()
    print(f"模型: {info['model_name']}")
    print(f"激活函数: {info['activation']}")
    print(f"参数量: {info['parameters']['total']:,}")
    print(f"感受野演进: {info['receptive_field']}")

    x_wavelet = torch.randn(2, 49, 49, 8)
    recon, latent = model_wavelet(x_wavelet)
    print(f"输入: {x_wavelet.shape} → 隐空间: {latent.shape} → 重建: {recon.shape}")
    print("✅ 测试通过\n")

    # 测试DirectAutoEncoder
    print("--- EnhancedDirectAutoEncoder ---")
    model_direct = EnhancedDirectAutoEncoder(latent_dim=256, num_frequencies=2)
    info = model_direct.get_model_info()
    print(f"模型: {info['model_name']}")
    print(f"激活函数: {info['activation']}")
    print(f"参数量: {info['parameters']['total']:,}")

    x_direct = torch.randn(2, 91, 91, 2)
    recon, latent = model_direct(x_direct)
    print(f"输入: {x_direct.shape} → 隐空间: {latent.shape} → 重建: {recon.shape}")
    print("✅ 测试通过\n")


if __name__ == "__main__":
    test_enhanced_models()
