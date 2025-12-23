"""
叠加型双分支MLP AutoEncoder
核心思想：双Decoder分别学习高频和低频特征，输出叠加，使用MLP架构

架构：
    输入 → Encoder(MLP) → Latent
                           ↓
            ┌──────────────┴──────────────┐
            ↓                             ↓
    Decoder_HighFreq(MLP)         Decoder_Smooth(MLP)
    (Sin激活)                      (ReLU/Tanh激活)
            ↓                             ↓
      高频特征重建                   低频趋势重建
            └──────────────┬──────────────┘
                           ↓
                 输出 = α·高频 + β·低频

优势：
- 全连接结构更适合学习全局频率特征
- Sin激活在MLP中通常比在CNN中更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np

from autoencoder.utils.activation_factory import get_activation, get_activation_name
from autoencoder.models.cnn_autoencoder import calculate_intermediate_dims
from autoencoder.models.base_autoencoder import BaseAutoEncoder


class AdditiveDualBranchWaveletMLPAutoEncoder(BaseAutoEncoder):
    """
    叠加型双分支小波MLP AutoEncoder
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 input_size: int = 49,
                 activation_encoder: str = 'relu',
                 activation_high: str = 'sin',
                 activation_smooth: str = 'relu',
                 learnable_weights: bool = False,
                 alpha_high: float = 1.0,
                 alpha_smooth: float = 1.0,
                 output_activation: str = None):
        """
        初始化叠加型双分支MLP AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量
            wavelet_bands: 小波频带数
            dropout_rate: Dropout比例
            input_size: 输入尺寸
            activation_encoder: Encoder激活函数
            activation_high: 高频分支Decoder激活函数
            activation_smooth: 低频分支Decoder激活函数
            learnable_weights: 是否学习叠加权重
            alpha_high: 高频分支初始权重
            alpha_smooth: 低频分支初始权重
            output_activation: 输出激活函数 (None 或 'softplus'，用于物理约束)
        """
        super().__init__(output_activation=output_activation)

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.activation_encoder_type = get_activation_name(activation_encoder)
        self.activation_high_type = get_activation_name(activation_high)
        self.activation_smooth_type = get_activation_name(activation_smooth)
        self.learnable_weights = learnable_weights

        # 计算输入总维度
        self.input_dim = input_size * input_size * num_frequencies * wavelet_bands
        
        # 动态计算中间层维度 (3层逐步压缩)
        self.intermediate_dims = calculate_intermediate_dims(
            self.input_dim, latent_dim, max_ratio=4
        )

        # ===== Encoder: Flatten Input → Latent =====
        encoder_layers = []
        current_dim = self.input_dim

        for intermediate_dim in self.intermediate_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation_encoder),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ===== Decoder Branch 1: High Frequency (Detail) =====
        self.decoder_high = self._build_decoder_mlp(activation_high)

        # ===== Decoder Branch 2: Smooth (Mean/Trend) =====
        self.decoder_smooth = self._build_decoder_mlp(activation_smooth)

        # ===== 叠加权重 =====
        if learnable_weights:
            self.alpha_high = nn.Parameter(torch.tensor(float(alpha_high)))
            self.alpha_smooth = nn.Parameter(torch.tensor(float(alpha_smooth)))
        else:
            self.register_buffer('alpha_high', torch.tensor(float(alpha_high)))
            self.register_buffer('alpha_smooth', torch.tensor(float(alpha_smooth)))

        # 权重初始化
        self._initialize_weights()

        # 统一接口
        self.decoder = nn.ModuleList([self.decoder_high, self.decoder_smooth])

    def _build_decoder_mlp(self, activation: str) -> nn.Sequential:
        """构建Decoder MLP"""
        layers = []
        current_dim = self.latent_dim

        # 反向构建中间层
        for intermediate_dim in reversed(self.intermediate_dims):
            layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = intermediate_dim

        # 输出层
        layers.append(nn.Linear(current_dim, self.input_dim))
        
        # 注意：最后一层通常不加激活函数，或者根据数据范围加 Tanh/Sigmoid
        # 这里为了保持通用性不加，让激活函数在Loss计算前处理或由数据预处理决定
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对Sin激活函数使用特殊的初始化
                if hasattr(m, 'activation_type') and m.activation_type == 'sin':
                    # SIREN initialization (如果实现了的话)
                    nn.init.uniform_(m.weight, -np.sqrt(6/m.in_features) / 30, np.sqrt(6/m.in_features) / 30)
                else:
                    nn.init.xavier_uniform_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码
        x: [B, H, W, C]
        """
        batch_size = x.shape[0]
        # Flatten: [B, H, W, C] -> [B, H*W*C]
        x_flat = x.reshape(batch_size, -1)
        
        latent = self.encoder(x_flat)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码
        latent: [B, latent_dim]
        """
        # 高频分支
        out_high = self.decoder_high(latent)
        
        # 低频分支
        out_smooth = self.decoder_smooth(latent)
        
        # 叠加 (独立权重)
        if self.learnable_weights:
            # 可学习模式下，通常不限制权重范围，允许模型自适应
            # 但为了防止爆炸，可以加一个软约束或不加
            recon_flat = self.alpha_high * out_high + self.alpha_smooth * out_smooth
        else:
            recon_flat = self.alpha_high * out_high + self.alpha_smooth * out_smooth
            
        # Reshape back: [B, H*W*C] -> [B, H, W, C]
        recon = recon_flat.reshape(latent.shape[0], self.input_size, self.input_size, -1)

        # ✅ 应用输出激活（如果启用了物理约束）
        recon = self.apply_output_activation(recon)

        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def forward_with_branches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（返回分支输出，用于可视化和频域分析）

        Args:
            x: [B, H, W, C] 输入数据

        Returns:
            recon: [B, H, W, C] 叠加后的重建
            latent: [B, latent_dim] 隐空间表示
            recon_high: [B, H, W, C] 高频分支重建
            recon_smooth: [B, H, W, C] 低频分支重建
        """
        batch_size = x.shape[0]

        # 编码
        latent = self.encode(x)

        # 分别计算两个分支（MLP输出为1D向量）
        out_high_flat = self.decoder_high(latent)      # [B, input_dim]
        out_smooth_flat = self.decoder_smooth(latent)  # [B, input_dim]

        # Reshape到2D用于可视化和FFT分析
        recon_high = out_high_flat.reshape(batch_size, self.input_size, self.input_size, -1)
        recon_smooth = out_smooth_flat.reshape(batch_size, self.input_size, self.input_size, -1)

        # 叠加
        if self.learnable_weights:
            recon = self.alpha_high * recon_high + self.alpha_smooth * recon_smooth
        else:
            recon = self.alpha_high * recon_high + self.alpha_smooth * recon_smooth

        # 应用输出激活
        recon = self.apply_output_activation(recon)
        recon_high = self.apply_output_activation(recon_high)
        recon_smooth = self.apply_output_activation(recon_smooth)

        return recon, latent, recon_high, recon_smooth

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_high_params = sum(p.numel() for p in self.decoder_high.parameters())
        decoder_smooth_params = sum(p.numel() for p in self.decoder_smooth.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'encoder': encoder_params,
            'decoder_high': decoder_high_params,
            'decoder_smooth': decoder_smooth_params,
            'total': total_params
        }

    def get_model_info(self) -> dict:
        return {
            'type': 'AdditiveDualBranchWaveletMLPAutoEncoder',
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'activation_encoder': self.activation_encoder_type,
            'activation_high': self.activation_high_type,
            'activation_smooth': self.activation_smooth_type,
            'alpha_high': self.alpha_high.item(),
            'alpha_smooth': self.alpha_smooth.item(),
            'parameters': self.get_parameter_count()['total']
        }


class AdditiveDualBranchDirectMLPAutoEncoder(BaseAutoEncoder):
    """
    叠加型双分支Direct MLP AutoEncoder (无小波变换)
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 input_size: int = 91,
                 activation_encoder: str = 'relu',
                 activation_high: str = 'sin',
                 activation_smooth: str = 'relu',
                 learnable_weights: bool = False,
                 alpha_high: float = 1.0,
                 alpha_smooth: float = 1.0,
                 output_activation: str = None):
        super().__init__(output_activation=output_activation)

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.activation_encoder_type = get_activation_name(activation_encoder)
        self.activation_high_type = get_activation_name(activation_high)
        self.activation_smooth_type = get_activation_name(activation_smooth)
        self.learnable_weights = learnable_weights

        # 计算输入总维度
        self.input_dim = input_size * input_size * num_frequencies
        
        # 动态计算中间层维度
        self.intermediate_dims = calculate_intermediate_dims(
            self.input_dim, latent_dim, max_ratio=4
        )

        # ===== Encoder =====
        encoder_layers = []
        current_dim = self.input_dim
        for intermediate_dim in self.intermediate_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation_encoder),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ===== Decoders =====
        self.decoder_high = self._build_decoder_mlp(activation_high)
        self.decoder_smooth = self._build_decoder_mlp(activation_smooth)

        # ===== Weights =====
        if learnable_weights:
            self.alpha_high = nn.Parameter(torch.tensor(float(alpha_high)))
            self.alpha_smooth = nn.Parameter(torch.tensor(float(alpha_smooth)))
        else:
            self.register_buffer('alpha_high', torch.tensor(float(alpha_high)))
            self.register_buffer('alpha_smooth', torch.tensor(float(alpha_smooth)))

        self._initialize_weights()
        self.decoder = nn.ModuleList([self.decoder_high, self.decoder_smooth])

    def _build_decoder_mlp(self, activation: str) -> nn.Sequential:
        layers = []
        current_dim = self.latent_dim
        for intermediate_dim in reversed(self.intermediate_dims):
            layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = intermediate_dim
        layers.append(nn.Linear(current_dim, self.input_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.encoder(x_flat)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        out_high = self.decoder_high(latent)
        out_smooth = self.decoder_smooth(latent)
        
        recon_flat = self.alpha_high * out_high + self.alpha_smooth * out_smooth
        
        recon = recon_flat.reshape(latent.shape[0], self.input_size, self.input_size, -1)

        # ✅ 应用输出激活（如果启用了物理约束）
        recon = self.apply_output_activation(recon)

        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def forward_with_branches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（返回分支输出，用于可视化和频域分析）

        Args:
            x: [B, H, W, C] 输入数据

        Returns:
            recon: [B, H, W, C] 叠加后的重建
            latent: [B, latent_dim] 隐空间表示
            recon_high: [B, H, W, C] 高频分支重建
            recon_smooth: [B, H, W, C] 低频分支重建
        """
        batch_size = x.shape[0]

        # 编码
        latent = self.encode(x)

        # 分别计算两个分支（MLP输出为1D向量）
        out_high_flat = self.decoder_high(latent)      # [B, input_dim]
        out_smooth_flat = self.decoder_smooth(latent)  # [B, input_dim]

        # Reshape到2D用于可视化和FFT分析
        recon_high = out_high_flat.reshape(batch_size, self.input_size, self.input_size, -1)
        recon_smooth = out_smooth_flat.reshape(batch_size, self.input_size, self.input_size, -1)

        # 叠加
        recon = self.alpha_high * recon_high + self.alpha_smooth * recon_smooth

        # 应用输出激活
        recon = self.apply_output_activation(recon)
        recon_high = self.apply_output_activation(recon_high)
        recon_smooth = self.apply_output_activation(recon_smooth)

        return recon, latent, recon_high, recon_smooth

    def get_parameter_count(self) -> Dict[str, int]:
        return {
            'total': sum(p.numel() for p in self.parameters())
        }

    def get_model_info(self) -> dict:
        return {
            'type': 'AdditiveDualBranchDirectMLPAutoEncoder',
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'parameters': self.get_parameter_count()['total']
        }
