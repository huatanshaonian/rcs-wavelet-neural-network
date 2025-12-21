"""
叠加型双分支AutoEncoder
核心思想：双Decoder分别学习高频和低频特征，输出叠加

架构：
    输入 → Encoder → Latent
                        ↓
            ┌───────────┴───────────┐
            ↓                       ↓
    Decoder_HighFreq          Decoder_Smooth
    (Sin激活)                  (Tanh/Swish激活)
            ↓                       ↓
      高频特征重建              低频趋势重建
            └───────────┬───────────┘
                        ↓
              输出 = α·高频 + β·低频

优势：
- Sin激活：学习高频振荡、细节特征
- Smooth激活：学习低频趋势、整体统计特性
- 输出叠加：兼顾高频和低频，提升重建质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

from autoencoder.utils.activation_factory import get_activation, get_activation_name
from autoencoder.models.cnn_autoencoder import calculate_intermediate_dims


class AdditiveDualBranchWaveletAutoEncoder(nn.Module):
    """
    叠加型双分支小波AutoEncoder (Wavelet模式)
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 input_size: int = 49,
                 activation_encoder: str = 'relu',
                 activation_high: str = 'sin',
                 activation_smooth: str = 'tanh',
                 learnable_weights: bool = False,
                 alpha_high: float = 0.5,
                 alpha_smooth: float = 0.5):
        """
        初始化叠加型双分支AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量 (2 for 1.5GHz+3GHz)
            wavelet_bands: 小波频带数 (4: LL, LH, HL, HH)
            dropout_rate: Dropout比例
            input_size: 输入小波系数尺寸 (49)
            activation_encoder: Encoder激活函数 (默认'relu', 可选: 'gelu', 'swish', 'sin'等)
            activation_high: 高频分支Decoder激活函数 (默认'sin', 推荐: 'sin', 'gelu', 'swish')
            activation_smooth: 低频分支Decoder激活函数 (默认'tanh', 推荐: 'tanh', 'sigmoid', 'softplus')
            learnable_weights: 是否学习叠加权重 (默认False - 固定权重)
            alpha_high: 高频分支权重 (learnable_weights=False时使用)
            alpha_smooth: 低频分支权重 (learnable_weights=False时使用)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.activation_encoder_type = get_activation_name(activation_encoder)
        self.activation_high_type = get_activation_name(activation_high)
        self.activation_smooth_type = get_activation_name(activation_smooth)
        self.learnable_weights = learnable_weights

        # ===== 共享Encoder: 小波系数 → 隐空间 =====
        self.encoder = nn.Sequential(
            # 第一层: [B, num_freq*4, 49, 49]
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation_encoder),

            # 下采样层1: [49, 49] → [25, 25]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation_encoder),
            nn.Dropout2d(dropout_rate),

            # 下采样层2: [25, 25] → [13, 13]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation_encoder),
            nn.Dropout2d(dropout_rate),

            # 下采样层3: [13, 13] → [7, 7]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation_encoder),
            nn.Dropout2d(dropout_rate),

            # 全局平均池化: [7, 7] → [4, 4]
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 计算展平后的特征维度
        self.flattened_size = 256 * 4 * 4  # 4096

        # 动态计算中间层维度
        self.intermediate_dims = calculate_intermediate_dims(
            self.flattened_size, latent_dim, max_ratio=4
        )

        # Encoder全连接层
        encoder_fc_layers = [nn.Flatten()]
        current_dim = self.flattened_size

        for intermediate_dim in self.intermediate_dims:
            encoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation_encoder),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        encoder_fc_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder_fc = nn.Sequential(*encoder_fc_layers)

        # ===== 双分支Decoder =====
        # 分支1: 高频Decoder (可配置激活)
        self.decoder_high_fc = self._build_decoder_fc(activation_high)
        self.decoder_high_conv = self._build_decoder_conv(activation_high)
        self.final_conv_high = nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1)

        # 分支2: 低频Decoder (Smooth激活)
        self.decoder_smooth_fc = self._build_decoder_fc(activation_smooth)
        self.decoder_smooth_conv = self._build_decoder_conv(activation_smooth)
        self.final_conv_smooth = nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1)

        # ===== 叠加权重 =====
        if learnable_weights:
            # 可学习的权重（初始化为均等）
            self.alpha_high = nn.Parameter(torch.tensor(0.5))
            self.alpha_smooth = nn.Parameter(torch.tensor(0.5))
        else:
            # 固定权重
            self.register_buffer('alpha_high', torch.tensor(alpha_high))
            self.register_buffer('alpha_smooth', torch.tensor(alpha_smooth))

        # 权重初始化
        self._initialize_weights()

        # ===== 统一接口：encoder/decoder属性 =====
        self.decoder = nn.ModuleList([
            self.decoder_high_fc, self.decoder_high_conv, self.final_conv_high,
            self.decoder_smooth_fc, self.decoder_smooth_conv, self.final_conv_smooth
        ])

    def _build_decoder_fc(self, activation: str) -> nn.Sequential:
        """构建Decoder全连接层"""
        decoder_fc_layers = []
        current_dim = self.latent_dim

        # 反向构建中间层
        for intermediate_dim in reversed(self.intermediate_dims):
            decoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到flattened_size
        decoder_fc_layers.extend([
            nn.Linear(current_dim, self.flattened_size),
            get_activation(activation)
        ])

        return nn.Sequential(*decoder_fc_layers)

    def _build_decoder_conv(self, activation: str) -> nn.Sequential:
        """构建Decoder卷积层"""
        return nn.Sequential(
            # 重塑为特征图: [4096] → [256, 4, 4]
            nn.Unflatten(1, (256, 4, 4)),

            # 上采样层1: [4, 4] → [8, 8]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(self.dropout_rate),

            # 上采样层2: [8, 8] → [16, 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(self.dropout_rate),

            # 上采样层3: [16, 16] → [32, 32]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation)
        )

    def _initialize_weights(self):
        """Xavier权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
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
        # 转换为通道优先: [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # CNN编码
        features = self.encoder(x)  # [B, 256, 4, 4]

        # 全连接编码到隐空间
        latent = self.encoder_fc(features)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        双分支解码器: 隐空间 → 小波系数

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            recon: [B, input_size, input_size, num_freq*4] 重建小波系数
        """
        # ===== 分支1: 高频Decoder =====
        feat_high = self.decoder_high_fc(latent)  # [B, 4096]
        feat_high = self.decoder_high_conv(feat_high)  # [B, 32, 32, 32]
        recon_high = self.final_conv_high(feat_high)  # [B, num_freq*4, 32, 32]

        # ===== 分支2: 低频Decoder =====
        feat_smooth = self.decoder_smooth_fc(latent)  # [B, 4096]
        feat_smooth = self.decoder_smooth_conv(feat_smooth)  # [B, 32, 32, 32]
        recon_smooth = self.final_conv_smooth(feat_smooth)  # [B, num_freq*4, 32, 32]

        # ===== 输出叠加 =====
        # 确保权重归一化（可选，防止输出爆炸）
        if self.learnable_weights:
            # 可学习权重：直接使用，不强制归一化，允许模型自适应学习幅度
            alpha_high_norm = self.alpha_high
            alpha_smooth_norm = self.alpha_smooth
        else:
            # 固定权重：直接使用
            alpha_high_norm = self.alpha_high
            alpha_smooth_norm = self.alpha_smooth

        recon = alpha_high_norm * recon_high + alpha_smooth_norm * recon_smooth

        # 上采样到目标尺寸: [32, 32] → [49, 49]
        recon = F.interpolate(recon, size=(self.input_size, self.input_size),
                             mode='bilinear', align_corners=False)

        # 转换回空间优先: [B, C, H, W] → [B, H, W, C]
        recon = recon.permute(0, 2, 3, 1)

        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, input_size, input_size, num_freq*4] 小波系数

        Returns:
            recon: [B, input_size, input_size, num_freq*4] 重建小波系数
            latent: [B, latent_dim] 隐空间表示
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        # Encoder参数
        encoder_params = sum(p.numel() for p in self.encoder.parameters()) + \
                        sum(p.numel() for p in self.encoder_fc.parameters())

        # High-frequency Decoder参数
        decoder_high_params = sum(p.numel() for p in self.decoder_high_fc.parameters()) + \
                             sum(p.numel() for p in self.decoder_high_conv.parameters()) + \
                             sum(p.numel() for p in self.final_conv_high.parameters())

        # Low-frequency Decoder参数
        decoder_smooth_params = sum(p.numel() for p in self.decoder_smooth_fc.parameters()) + \
                               sum(p.numel() for p in self.decoder_smooth_conv.parameters()) + \
                               sum(p.numel() for p in self.final_conv_smooth.parameters())

        # 总decoder参数（两个分支之和）
        decoder_params = decoder_high_params + decoder_smooth_params

        # 总参数和可训练参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'decoder_high': decoder_high_params,
            'decoder_smooth': decoder_smooth_params,
            'total': total_params,
            'trainable': trainable_params
        }

    def get_model_info(self) -> dict:
        """返回模型信息"""
        return {
            'type': 'AdditiveDualBranchWaveletAutoEncoder',
            'latent_dim': self.latent_dim,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'activation_encoder': self.activation_encoder_type,
            'activation_high': self.activation_high_type,
            'activation_smooth': self.activation_smooth_type,
            'learnable_weights': self.learnable_weights,
            'alpha_high': self.alpha_high.item(),
            'alpha_smooth': self.alpha_smooth.item(),
            'intermediate_dims': self.intermediate_dims,
            'parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class AdditiveDualBranchDirectAutoEncoder(nn.Module):
    """
    叠加型双分支Direct AutoEncoder (Direct模式 - 无小波变换)
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 input_size: int = 91,
                 activation_encoder: str = 'relu',
                 activation_high: str = 'sin',
                 activation_smooth: str = 'tanh',
                 learnable_weights: bool = False,
                 alpha_high: float = 0.5,
                 alpha_smooth: float = 0.5):
        """
        初始化Direct模式叠加型双分支AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量 (2 for 1.5GHz+3GHz)
            dropout_rate: Dropout比例
            input_size: 输入RCS尺寸 (91)
            activation_encoder: Encoder激活函数 (默认'relu')
            activation_high: 高频分支Decoder激活函数 (默认'sin')
            activation_smooth: 低频分支Decoder激活函数 (默认'tanh')
            learnable_weights: 是否学习叠加权重
            alpha_high: 高频分支权重
            alpha_smooth: 低频分支权重
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.input_channels = num_frequencies
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.activation_encoder_type = get_activation_name(activation_encoder)
        self.activation_high_type = get_activation_name(activation_high)
        self.activation_smooth_type = get_activation_name(activation_smooth)
        self.learnable_weights = learnable_weights

        # ===== 共享Encoder: RCS → 隐空间 =====
        self.encoder = nn.Sequential(
            # 第一层: [B, num_freq, 91, 91]
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation_encoder),

            # 下采样层1: [91, 91] → [46, 46]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation_encoder),
            nn.Dropout2d(dropout_rate),

            # 下采样层2: [46, 46] → [23, 23]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation_encoder),
            nn.Dropout2d(dropout_rate),

            # 下采样层3: [23, 23] → [12, 12]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation_encoder),
            nn.Dropout2d(dropout_rate),

            # 下采样层4: [12, 12] → [6, 6]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            get_activation(activation_encoder),
            nn.Dropout2d(dropout_rate),

            # 全局平均池化: [6, 6] → [4, 4]
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 计算展平后的特征维度
        self.flattened_size = 512 * 4 * 4  # 8192

        # 动态计算中间层维度
        self.intermediate_dims = calculate_intermediate_dims(
            self.flattened_size, latent_dim, max_ratio=4
        )

        # Encoder全连接层
        encoder_fc_layers = [nn.Flatten()]
        current_dim = self.flattened_size

        for intermediate_dim in self.intermediate_dims:
            encoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation_encoder),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        encoder_fc_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder_fc = nn.Sequential(*encoder_fc_layers)

        # ===== 双分支Decoder =====
        # 分支1: 高频Decoder (可配置激活)
        self.decoder_high_fc = self._build_decoder_fc(activation_high)
        self.decoder_high_conv = self._build_decoder_conv(activation_high)
        self.final_conv_high = nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1)

        # 分支2: 低频Decoder
        self.decoder_smooth_fc = self._build_decoder_fc(activation_smooth)
        self.decoder_smooth_conv = self._build_decoder_conv(activation_smooth)
        self.final_conv_smooth = nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1)

        # ===== 叠加权重 =====
        if learnable_weights:
            self.alpha_high = nn.Parameter(torch.tensor(0.5))
            self.alpha_smooth = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('alpha_high', torch.tensor(alpha_high))
            self.register_buffer('alpha_smooth', torch.tensor(alpha_smooth))

        # 权重初始化
        self._initialize_weights()

        # 统一接口
        self.decoder = nn.ModuleList([
            self.decoder_high_fc, self.decoder_high_conv, self.final_conv_high,
            self.decoder_smooth_fc, self.decoder_smooth_conv, self.final_conv_smooth
        ])

    def _build_decoder_fc(self, activation: str) -> nn.Sequential:
        """构建Decoder全连接层"""
        decoder_fc_layers = []
        current_dim = self.latent_dim

        for intermediate_dim in reversed(self.intermediate_dims):
            decoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = intermediate_dim

        decoder_fc_layers.extend([
            nn.Linear(current_dim, self.flattened_size),
            get_activation(activation)
        ])

        return nn.Sequential(*decoder_fc_layers)

    def _build_decoder_conv(self, activation: str) -> nn.Sequential:
        """构建Decoder卷积层"""
        return nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation),
            nn.Dropout2d(self.dropout_rate),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(self.dropout_rate),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(self.dropout_rate),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation)
        )

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码器"""
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        features = self.encoder(x)
        latent = self.encoder_fc(features)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """双分支解码器"""
        # 高频分支
        feat_high = self.decoder_high_fc(latent)
        feat_high = self.decoder_high_conv(feat_high)
        recon_high = self.final_conv_high(feat_high)

        # 低频分支
        feat_smooth = self.decoder_smooth_fc(latent)
        feat_smooth = self.decoder_smooth_conv(feat_smooth)
        recon_smooth = self.final_conv_smooth(feat_smooth)

        # 叠加
        if self.learnable_weights:
            # 可学习权重：直接使用，不强制归一化
            alpha_high_norm = self.alpha_high
            alpha_smooth_norm = self.alpha_smooth
        else:
            alpha_high_norm = self.alpha_high
            alpha_smooth_norm = self.alpha_smooth

        recon = alpha_high_norm * recon_high + alpha_smooth_norm * recon_smooth

        # 上采样: [64, 64] → [91, 91]
        recon = F.interpolate(recon, size=(self.input_size, self.input_size),
                             mode='bilinear', align_corners=False)

        recon = recon.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        # Encoder参数
        encoder_params = sum(p.numel() for p in self.encoder.parameters()) + \
                        sum(p.numel() for p in self.encoder_fc.parameters())

        # High-frequency Decoder参数
        decoder_high_params = sum(p.numel() for p in self.decoder_high_fc.parameters()) + \
                             sum(p.numel() for p in self.decoder_high_conv.parameters()) + \
                             sum(p.numel() for p in self.final_conv_high.parameters())

        # Low-frequency Decoder参数
        decoder_smooth_params = sum(p.numel() for p in self.decoder_smooth_fc.parameters()) + \
                               sum(p.numel() for p in self.decoder_smooth_conv.parameters()) + \
                               sum(p.numel() for p in self.final_conv_smooth.parameters())

        # 总decoder参数（两个分支之和）
        decoder_params = decoder_high_params + decoder_smooth_params

        # 总参数和可训练参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'decoder_high': decoder_high_params,
            'decoder_smooth': decoder_smooth_params,
            'total': total_params,
            'trainable': trainable_params
        }

    def get_model_info(self) -> dict:
        """返回模型信息"""
        return {
            'type': 'AdditiveDualBranchDirectAutoEncoder',
            'latent_dim': self.latent_dim,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'activation_encoder': self.activation_encoder_type,
            'activation_high': self.activation_high_type,
            'activation_smooth': self.activation_smooth_type,
            'learnable_weights': self.learnable_weights,
            'alpha_high': self.alpha_high.item(),
            'alpha_smooth': self.alpha_smooth.item(),
            'intermediate_dims': self.intermediate_dims,
            'parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
