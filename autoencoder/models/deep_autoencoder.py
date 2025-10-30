"""
Deep CNN-AutoEncoder架构（4层深度，双卷积）
相比标准CNN更深，采用双卷积设计和通道注意力机制

Deep系列包括：
- DeepWaveletAutoEncoder: 小波增强模式（49×49输入）
- DeepDirectAutoEncoder: 直接处理模式（91×91输入）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from autoencoder.utils.adaptive_layers import (
    build_adaptive_fc_pair,
    get_structure_info,
)
from autoencoder.models.channel_attention import ChannelAttention, get_recommended_reduction
from autoencoder.utils.activation_factory import get_activation, get_activation_name


class DeepWaveletAutoEncoder(nn.Module):
    """
    深度CNN-AutoEncoder（小波模式）
    输入: [B, 49, 49, 8] 小波系数 (2频率 × 4频带)
    输出: [B, 49, 49, 8] 重建小波系数

    特点：
    - 4层深度CNN（针对49×49优化）
    - 双卷积设计（每stage两个3×3卷积）
    - 通道注意力机制
    - 更大的感受野
    - 参数量：~12M（介于标准CNN和原5层之间）
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 use_attention: bool = True,
                 input_size: int = 49,
                 use_channel_attention: bool = False,
                 activation: str = 'relu'):
        """
        初始化Deep Wavelet AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量
            wavelet_bands: 小波频带数
            dropout_rate: Dropout比率
            use_attention: 是否使用中间层通道注意力 (Deep架构固有特性)
            input_size: 小波系数尺寸 (49)
            use_channel_attention: 是否在输入层使用通道注意力 (用户可配置)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.input_size = input_size
        self.use_channel_attention = use_channel_attention
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        def replace_activation_layers(seq: nn.Sequential) -> nn.Sequential:
            layers = []
            for module in seq:
                if isinstance(module, nn.ReLU):
                    layers.append(activation_layer())
                else:
                    layers.append(module)
            return nn.Sequential(*layers)
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        def replace_activation_layers(seq: nn.Sequential) -> nn.Sequential:
            layers = []
            for module in seq:
                if isinstance(module, nn.ReLU):
                    layers.append(activation_layer())
                else:
                    layers.append(module)
            return nn.Sequential(*layers)

        # ===== 输入层通道注意力（可选，用户可配置） =====
        if self.use_channel_attention:
            reduction = get_recommended_reduction(self.input_channels)
            self.input_channel_attention = ChannelAttention(
                num_channels=self.input_channels,
                reduction=reduction
            )
        else:
            self.input_channel_attention = None

        # ===== 深度Encoder架构（4层，双卷积） =====
        # 输入: [B, 49, 49, 8] → permute → [B, 8, 49, 49]

        # Stage 1: [8, 49, 49] → [32, 49, 49] (双卷积，不下采样)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 双卷积
            nn.BatchNorm2d(32),
            activation_layer()
        )

        # Stage 2: [32, 49, 49] → [64, 25, 25]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 双卷积
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 3: [64, 25, 25] → [128, 13, 13]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 双卷积
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 4: [128, 13, 13] → [256, 7, 7]
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 双卷积
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # 通道注意力机制（可选）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),  # 1×1卷积学习通道权重
                nn.Sigmoid()  # 输出0-1的权重
            )

        # 编码器FC层: [256, 7, 7] → latent_dim（自适应构建）
        self.final_conv_size = 256 * 7 * 7  # 12544
        self.encoder_fc, decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
            input_dim=self.final_conv_size,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            arch_type='cnn',
        )
        self.encoder_fc = replace_activation_layers(self.encoder_fc)
        decoder_fc = replace_activation_layers(decoder_fc)
        self.decoder_fc = nn.Sequential(*list(decoder_fc.children()), activation_layer())
        self.structure_info = get_structure_info(
            self.final_conv_size, latent_dim, self.intermediate_dims
        )

        # ===== 深度Decoder架构 =====

        # Stage 4 逆向: [256, 7, 7] → [128, 13, 13]
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 双卷积
            nn.BatchNorm2d(128),
            activation_layer()
        )

        # Stage 3 逆向: [128, 13, 13] → [64, 25, 25]
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 双卷积
            nn.BatchNorm2d(64),
            activation_layer()
        )

        # Stage 2 逆向: [64, 25, 25] → [32, 49, 49]
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 双卷积
            nn.BatchNorm2d(32),
            activation_layer()
        )

        # Stage 1 逆向: [32, 49, 49] → [8, 49, 49]
        self.deconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1)
            # 不使用Tanh，允许小波系数任意值
        )

        self._initialize_weights()

        # ===== 统一接口：encoder/decoder属性 =====
        # 为了兼容训练代码中的冻结/解冻操作，提供统一的encoder/decoder接口
        encoder_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.encoder_fc]
        if self.use_attention:
            encoder_modules.insert(-1, self.attention)  # 在encoder_fc之前插入attention
        self.encoder = nn.ModuleList(encoder_modules)

        self.decoder = nn.ModuleList([
            self.decoder_fc, self.deconv4, self.deconv3, self.deconv2, self.deconv1
        ])

    def _initialize_weights(self):
        """Kaiming权重初始化（针对ReLU优化）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码器: 小波系数 → 隐空间

        Args:
            x: [B, 49, 49, 8] 小波系数

        Returns:
            latent: [B, latent_dim] 隐空间表示
        """
        # 调整维度: [B, 49, 49, 8] → [B, 8, 49, 49]
        x = x.permute(0, 3, 1, 2)

        # 应用输入层通道注意力（如果启用）
        if self.use_channel_attention and self.input_channel_attention is not None:
            x = self.input_channel_attention(x)

        # 逐层编码（双卷积设计）
        x1 = self.conv1(x)      # [B, 32, 49, 49]
        x2 = self.conv2(x1)     # [B, 64, 25, 25]
        x3 = self.conv3(x2)     # [B, 128, 13, 13]
        x4 = self.conv4(x3)     # [B, 256, 7, 7]

        # 通道注意力机制（可选）
        if self.use_attention:
            attention_weights = self.attention(x4)  # [B, 256, 7, 7]
            x4 = x4 * attention_weights  # 加权

        # 最终编码
        latent = self.encoder_fc(x4)  # [B, latent_dim]
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码器: 隐空间 → 小波系数

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            x_recon: [B, 49, 49, 8] 重建小波系数
        """
        # 重建特征图
        x = self.decoder_fc(latent)
        x = x.view(-1, 256, 7, 7)  # [B, 256, 7, 7]

        # 逐层解码（双卷积设计）
        x = self.deconv4(x)        # [B, 128, 13, 13]
        x = self.deconv3(x)        # [B, 64, 25, 25]
        x = self.deconv2(x)        # [B, 32, 49, 49]
        x = self.deconv1(x)        # [B, 8, 49, 49]

        # 调整维度: [B, 8, 49, 49] → [B, 49, 49, 8]
        x_recon = x.permute(0, 2, 3, 1)

        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        encoder_params = (
            sum(p.numel() for n, p in self.named_parameters() if 'conv' in n and 'deconv' not in n) +
            sum(p.numel() for p in self.encoder_fc.parameters())
        )
        if self.use_attention:
            encoder_params += sum(p.numel() for p in self.attention.parameters())

        decoder_params = (
            sum(p.numel() for n, p in self.named_parameters() if 'deconv' in n) +
            sum(p.numel() for p in self.decoder_fc.parameters())
        )

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

        return {
            'model_name': 'DeepWaveletAutoEncoder',
            'architecture': '4-stage Deep CNN with double convolutions',
            'input_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'output_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'latent_dim': self.latent_dim,
            'parameters': param_count,
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'fc_structure': self.structure_info['fc_structure'],
            'intermediate_dims': self.structure_info['intermediate_dims'],
            'num_fc_layers': self.structure_info['num_fc_layers'],
            'compression_ratios': self.structure_info['compression_ratios'],
            'stages': [
                f'Conv1: {self.input_channels}→32 [{self.input_size}×{self.input_size}] (双卷积)',
                'Conv2: 32→64 [25×25] (双卷积)',
                'Conv3: 64→128 [13×13] (双卷积)',
                'Conv4: 128→256 [7×7] (双卷积)',
                f'FC: {self.structure_info["fc_structure"]}'
            ]
        }


class DeepDirectAutoEncoder(nn.Module):
    """
    深度CNN-AutoEncoder（Direct模式）
    输入: [B, 91, 91, 2] RCS数据（直接处理，不经小波变换）
    输出: [B, 91, 91, 2] 重建RCS数据

    特点：
    - 4层深度CNN（针对91×91优化）
    - 双卷积设计
    - 通道注意力机制
    - 参数量：~15M
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 use_attention: bool = True,
                 input_size: int = 91,
                 use_channel_attention: bool = False,
                 activation: str = 'relu'):
        """
        初始化Deep Direct AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量
            dropout_rate: Dropout比率
            use_attention: 是否使用中间层通道注意力 (Deep架构固有特性)
            input_size: RCS数据尺寸 (91)
            use_channel_attention: 是否在输入层使用通道注意力 (用户可配置)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.input_channels = num_frequencies  # Direct模式：2个频率
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.input_size = input_size
        self.use_channel_attention = use_channel_attention
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        # ===== 输入层通道注意力（可选，用户可配置） =====
        if self.use_channel_attention:
            reduction = get_recommended_reduction(self.input_channels)
            self.input_channel_attention = ChannelAttention(
                num_channels=self.input_channels,
                reduction=reduction
            )
        else:
            self.input_channel_attention = None

        # ===== 深度Encoder架构（4层） =====
        # 91 → 91 → 46 → 23 → 12

        # Stage 1: [2, 91, 91] → [32, 91, 91]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer()
        )

        # Stage 2: [32, 91, 91] → [64, 46, 46]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 3: [64, 46, 46] → [128, 23, 23]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # Stage 4: [128, 23, 23] → [256, 12, 12]
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            activation_layer(),
            nn.Dropout2d(dropout_rate)
        )

        # 通道注意力
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.Sigmoid()
            )

        # 编码器FC层（自适应构建）
        self.final_conv_size = 256 * 12 * 12  # 36864
        self.encoder_fc, decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
            input_dim=self.final_conv_size,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            arch_type='cnn',
        )
        self.encoder_fc = replace_activation_layers(self.encoder_fc)
        decoder_fc = replace_activation_layers(decoder_fc)
        self.decoder_fc = nn.Sequential(*list(decoder_fc.children()), activation_layer())
        self.structure_info = get_structure_info(
            self.final_conv_size, latent_dim, self.intermediate_dims
        )

        # ===== 深度Decoder架构 =====

        # 解码器卷积层
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_layer()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_layer()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer()
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1)
        )

        self._initialize_weights()

        # ===== 统一接口：encoder/decoder属性 =====
        # 为了兼容训练代码中的冻结/解冻操作，提供统一的encoder/decoder接口
        encoder_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.encoder_fc]
        if self.use_attention:
            encoder_modules.insert(-1, self.attention)  # 在encoder_fc之前插入attention
        self.encoder = nn.ModuleList(encoder_modules)

        self.decoder = nn.ModuleList([
            self.decoder_fc, self.deconv4, self.deconv3, self.deconv2, self.deconv1
        ])

    def _initialize_weights(self):
        """Kaiming权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码: RCS → 隐空间"""
        # [B, 91, 91, 2] → [B, 2, 91, 91]
        x = x.permute(0, 3, 1, 2)

        # 应用输入层通道注意力（如果启用）
        if self.use_channel_attention and self.input_channel_attention is not None:
            x = self.input_channel_attention(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        if self.use_attention:
            attention_weights = self.attention(x4)
            x4 = x4 * attention_weights

        latent = self.encoder_fc(x4)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码: 隐空间 → RCS"""
        x = self.decoder_fc(latent)
        x = x.view(-1, 256, 12, 12)

        x = self.deconv4(x)   # [B, 128, 23, 23]
        x = self.deconv3(x)   # [B, 64, 46, 46]
        x = self.deconv2(x)   # [B, 32, 91, 91]
        x = self.deconv1(x)   # [B, 2, 91, 91]

        # [B, 2, 91, 91] → [B, 91, 91, 2]
        x_recon = x.permute(0, 2, 3, 1)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        param_count = self.get_parameter_count()

        return {
            'model_name': 'DeepDirectAutoEncoder',
            'architecture': '4-stage Deep CNN for direct RCS processing',
            'input_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'output_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'latent_dim': self.latent_dim,
            'parameters': param_count,
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'fc_structure': self.structure_info['fc_structure'],
            'intermediate_dims': self.structure_info['intermediate_dims'],
            'num_fc_layers': self.structure_info['num_fc_layers'],
            'compression_ratios': self.structure_info['compression_ratios'],
            'stages': [
                f'Conv1: {self.input_channels}→32 [{self.input_size}×{self.input_size}] (双卷积)',
                'Conv2: 32→64 [46×46] (双卷积)',
                'Conv3: 64→128 [23×23] (双卷积)',
                'Conv4: 128→256 [12×12] (双卷积)',
                f'FC: {self.structure_info["fc_structure"]}'
            ]
        }


if __name__ == "__main__":
    print("=== 测试DeepWaveletAutoEncoder ===")
    model_wavelet = DeepWaveletAutoEncoder(latent_dim=256, num_frequencies=2)
    test_input_wavelet = torch.randn(4, 49, 49, 8)
    recon_wavelet, latent_wavelet = model_wavelet(test_input_wavelet)
    print(f"输入: {test_input_wavelet.shape}")
    print(f"重建: {recon_wavelet.shape}")
    print(f"隐空间: {latent_wavelet.shape}")
    info_wavelet = model_wavelet.get_model_info()
    print(f"参数量: {info_wavelet['parameters']['total']:,}")
    print(f"激活函数: {info_wavelet['activation']}")

    print("\n=== 测试DeepDirectAutoEncoder ===")
    model_direct = DeepDirectAutoEncoder(latent_dim=256, num_frequencies=2)
    test_input_direct = torch.randn(4, 91, 91, 2)
    recon_direct, latent_direct = model_direct(test_input_direct)
    print(f"输入: {test_input_direct.shape}")
    print(f"重建: {recon_direct.shape}")
    print(f"隐空间: {latent_direct.shape}")
    info_direct = model_direct.get_model_info()
    print(f"参数量: {info_direct['parameters']['total']:,}")
    print(f"激活函数: {info_direct['activation']}")
