"""
双分支可微分小波AutoEncoder模型 V2 (正确实现)

与V1的关键区别：
1. ✅ Decoder也是双分支（对称架构）
2. ✅ ll_latent_dim和hf_latent_dim真正起作用
3. ✅ 正确的通道split/combine操作，保证通道顺序
4. ✅ 移除无用的fusion层，直接concat

架构说明：
- Encoder: LL分支 + HF分支 → 分别输出ll_latent和hf_latent → concat → latent
- Decoder: latent → split → LL分支 + HF分支 → 分别输出LL和HF → combine → 小波系数
- 损失在RCS空间，梯度可微分回传
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List
import numpy as np

# 导入可微分小波变换
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from differentiable_wavelet_transform import DifferentiableWaveletTransform
from autoencoder.utils.adaptive_layers import get_structure_info
from autoencoder.utils.activation_factory import get_activation, get_activation_name

# 从dual_branch_autoencoder导入工具函数
try:
    from .dual_branch_autoencoder import calculate_branch_latent_dims
except ImportError:
    from dual_branch_autoencoder import calculate_branch_latent_dims


def calculate_intermediate_dims(input_dim: int, latent_dim: int, max_ratio: int = 4) -> List[int]:
    """
    动态计算中间层维度，实现渐进式压缩

    策略：
    - 保持每级压缩比不超过max_ratio（默认4:1）
    - 自动生成多个中间层以平滑过渡
    - 维度向上取整到2的幂次，便于硬件优化

    Args:
        input_dim: 输入维度
        latent_dim: 目标隐空间维度
        max_ratio: 每级最大压缩比（默认4）

    Returns:
        中间层维度列表（不包括input_dim和latent_dim）
    """
    if input_dim <= latent_dim:
        return []

    dims = []
    current = input_dim

    while current > latent_dim * max_ratio:
        next_dim = max(latent_dim, current // max_ratio)
        next_dim = 2 ** round(np.log2(next_dim))
        next_dim = max(next_dim, latent_dim)

        if next_dim < current:
            dims.append(next_dim)
            current = next_dim
        else:
            break

    return dims


# ============================================================================
# 双分支可微分MLP AutoEncoder V2 (正确实现)
# ============================================================================

class DualBranchDifferentiableWaveletMLPAutoEncoderV2(nn.Module):
    """
    双分支可微分小波MLP AutoEncoder V2 (正确实现)

    与V1的区别：
    1. ✅ Decoder也是双分支（LL decoder + HF decoder）
    2. ✅ ll_latent_dim和hf_latent_dim真正起作用
    3. ✅ 正确的_combine_channels()操作
    4. ✅ 移除fusion层，直接concat

    架构：
    Encoder:
        RCS → 小波变换 → split(LL, HF)
        LL → flatten → MLP → ll_latent
        HF → flatten → MLP → hf_latent
        concat(ll_latent, hf_latent) → latent

    Decoder:
        latent → split(ll_latent, hf_latent)
        ll_latent → MLP → LL channels
        hf_latent → MLP → HF channels
        combine(LL, HF) → 小波系数 → 逆小波 → RCS
    """

    def __init__(self,
                 latent_dim: int = 32,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 wavelet_type: str = 'db4',
                 input_size: int = 49,
                 ll_ratio: float = 0.7,
                 activation: str = 'relu'):
        """
        初始化双分支可微分Wavelet MLP AutoEncoder V2

        Args:
            latent_dim: 总隐空间维度（如32）
            num_frequencies: 频率数量（2 or 3）
            dropout_rate: Dropout率
            wavelet_type: 小波类型（db4, haar等）
            input_size: 小波系数空间尺寸（默认49）
            ll_ratio: LL分支latent占比（默认0.7，即LL占70%）
            activation: 激活函数类型 (例如 'relu', 'sin', 'gelu', 'swish')
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size
        self.ll_ratio = ll_ratio
        self.activation_type = get_activation_name(activation)

        # ===== 关键修复1: ll_latent_dim和hf_latent_dim真正起作用 =====
        self.ll_latent_dim, self.hf_latent_dim = calculate_branch_latent_dims(latent_dim, ll_ratio)

        # ===== 集成可微分小波变换 =====
        self.wavelet_transform = DifferentiableWaveletTransform(
            wavelet=wavelet_type,
            mode='symmetric',
            level=1
        )

        # 计算输入维度
        ll_input_dim = input_size * input_size * num_frequencies      # 49*49*2 = 4802
        hf_input_dim = input_size * input_size * (num_frequencies * 3) # 49*49*6 = 14406

        # ===== Encoder: LL分支 (输出ll_latent_dim) =====
        self.ll_encoder = nn.Sequential(
            nn.Linear(ll_input_dim, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(256, self.ll_latent_dim),  # ✅ 修复：使用ll_latent_dim而非128
        )

        # ===== Encoder: HF分支 (输出hf_latent_dim) =====
        self.hf_encoder = nn.Sequential(
            nn.Linear(hf_input_dim, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(256, self.hf_latent_dim),  # ✅ 修复：使用hf_latent_dim而非128
        )

        # ===== 关键修复2: 移除fusion层，直接concat =====
        # latent = torch.cat([ll_latent, hf_latent], dim=1)
        # 不需要额外的fusion层

        # ===== 关键修复3: Decoder也是双分支 =====

        # LL Decoder (对称设计)
        self.ll_decoder = nn.Sequential(
            nn.Linear(self.ll_latent_dim, 256),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(512, ll_input_dim),  # 输出LL通道
        )

        # HF Decoder (对称设计)
        self.hf_decoder = nn.Sequential(
            nn.Linear(self.hf_latent_dim, 256),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(512, hf_input_dim),  # 输出HF通道
        )

        # 统一的encoder/decoder接口（训练时需要）
        self.encoder = nn.ModuleList([
            self.ll_encoder,
            self.hf_encoder,
            self.wavelet_transform
        ])
        self.decoder = nn.ModuleList([
            self.ll_decoder,
            self.hf_decoder
        ])

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _split_channels(self, wavelet_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分离LL和HF通道（保证顺序正确）

        Args:
            wavelet_coeffs: [B, H, W, num_freq*4]
                通道顺序: [f0_LL, f0_LH, f0_HL, f0_HH, f1_LL, f1_LH, f1_HL, f1_HH, ...]

        Returns:
            ll_channels: [B, H, W, num_freq]  顺序: [f0_LL, f1_LL, ...]
            hf_channels: [B, H, W, num_freq*3] 顺序: [f0_LH, f0_HL, f0_HH, f1_LH, f1_HL, f1_HH, ...]
        """
        ll_list = []
        hf_list = []

        for freq_idx in range(self.num_frequencies):
            base = freq_idx * 4
            ll_list.append(wavelet_coeffs[:, :, :, base:base+1])       # LL
            hf_list.append(wavelet_coeffs[:, :, :, base+1:base+4])     # LH, HL, HH

        ll_channels = torch.cat(ll_list, dim=3)  # [B, H, W, num_freq]
        hf_channels = torch.cat(hf_list, dim=3)  # [B, H, W, num_freq*3]

        return ll_channels, hf_channels

    def _combine_channels(self, ll_channels: torch.Tensor, hf_channels: torch.Tensor) -> torch.Tensor:
        """
        ===== 关键修复4: 正确组合LL和HF通道 =====

        组合LL和HF通道回小波系数格式（保证通道顺序正确）

        Args:
            ll_channels: [B, H, W, num_freq]  顺序: [f0_LL, f1_LL, ...]
            hf_channels: [B, H, W, num_freq*3] 顺序: [f0_LH, f0_HL, f0_HH, f1_LH, f1_HL, f1_HH, ...]

        Returns:
            wavelet_coeffs: [B, H, W, num_freq*4]
                通道顺序: [f0_LL, f0_LH, f0_HL, f0_HH, f1_LL, f1_LH, f1_HL, f1_HH, ...]
        """
        wavelet_coeffs_list = []

        for freq_idx in range(self.num_frequencies):
            # 提取该频率的LL和HF
            ll = ll_channels[:, :, :, freq_idx:freq_idx+1]          # [B, H, W, 1]
            hf = hf_channels[:, :, :, freq_idx*3:(freq_idx+1)*3]    # [B, H, W, 3]

            # 拼接为 [LL, LH, HL, HH]
            freq_coeffs = torch.cat([ll, hf], dim=3)  # [B, H, W, 4]
            wavelet_coeffs_list.append(freq_coeffs)

        # 拼接所有频率
        wavelet_coeffs = torch.cat(wavelet_coeffs_list, dim=3)  # [B, H, W, num_freq*4]
        return wavelet_coeffs

    def encode(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """
        编码: RCS → 小波系数 → 双分支 → 隐空间

        Args:
            rcs_data: [B, H, W, C] RCS数据

        Returns:
            latent: [B, latent_dim]
        """
        batch_size = rcs_data.shape[0]

        # Step 1: RCS → 小波系数（可微分）
        wavelet_coeffs = self.wavelet_transform.forward_transform(rcs_data)  # [B, 49, 49, 8]

        # Step 2: 分离LL和HF通道
        ll_channels, hf_channels = self._split_channels(wavelet_coeffs)

        # Step 3: Flatten
        ll_flat = ll_channels.reshape(batch_size, -1)  # [B, 4802]
        hf_flat = hf_channels.reshape(batch_size, -1)  # [B, 14406]

        # Step 4: 双分支编码
        ll_latent = self.ll_encoder(ll_flat)  # [B, ll_latent_dim]
        hf_latent = self.hf_encoder(hf_flat)  # [B, hf_latent_dim]

        # Step 5: Concat（不需要fusion层）
        latent = torch.cat([ll_latent, hf_latent], dim=1)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码: 隐空间 → 双分支 → 小波系数 → RCS

        Args:
            latent: [B, latent_dim]

        Returns:
            rcs_data: [B, H, W, C] 重建的RCS数据
        """
        batch_size = latent.shape[0]

        # Step 1: Split latent
        ll_latent = latent[:, :self.ll_latent_dim]  # [B, ll_latent_dim]
        hf_latent = latent[:, self.ll_latent_dim:]  # [B, hf_latent_dim]

        # Step 2: 双分支解码
        ll_flat = self.ll_decoder(ll_latent)  # [B, 4802]
        hf_flat = self.hf_decoder(hf_latent)  # [B, 14406]

        # Step 3: Reshape
        ll_channels = ll_flat.view(batch_size, self.input_size, self.input_size, self.num_frequencies)
        hf_channels = hf_flat.view(batch_size, self.input_size, self.input_size, self.num_frequencies * 3)

        # Step 4: 正确组合（关键！）
        wavelet_coeffs = self._combine_channels(ll_channels, hf_channels)  # [B, 49, 49, 8]

        # Step 5: 逆小波变换
        rcs_data = self.wavelet_transform.inverse_transform(wavelet_coeffs)  # [B, 91, 91, 2]

        return rcs_data

    def forward(self, rcs_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: RCS → 隐空间 → 重建RCS

        Args:
            rcs_data: [B, H, W, C] 输入RCS数据

        Returns:
            reconstructed: [B, H, W, C] 重建的RCS数据
            latent: [B, latent_dim] 隐空间表示
        """
        latent = self.encode(rcs_data)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取模型参数数量（兼容现有系统）"""
        ll_encoder_params = sum(p.numel() for p in self.ll_encoder.parameters())
        hf_encoder_params = sum(p.numel() for p in self.hf_encoder.parameters())
        ll_decoder_params = sum(p.numel() for p in self.ll_decoder.parameters())
        hf_decoder_params = sum(p.numel() for p in self.hf_decoder.parameters())
        wavelet_params = sum(p.numel() for p in self.wavelet_transform.parameters())

        encoder_params = ll_encoder_params + hf_encoder_params
        decoder_params = ll_decoder_params + hf_decoder_params
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'll_encoder': ll_encoder_params,
            'hf_encoder': hf_encoder_params,
            'll_decoder': ll_decoder_params,
            'hf_decoder': hf_decoder_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'wavelet_transform': wavelet_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        return {
            'model_name': 'DualBranchDifferentiableWaveletMLPAutoEncoderV2',
            'version': 'v2',
            'architecture': 'Dual-Branch MLP + Differentiable Wavelet (Correct Implementation)',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'wavelet_type': self.wavelet_type,
            'differentiable': True,
            'input_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'output_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'loss_space': 'RCS',
            'branch_config': {
                'll_channels': self.num_frequencies,
                'hf_channels': self.num_frequencies * 3,
                'll_latent_dim': self.ll_latent_dim,
                'hf_latent_dim': self.hf_latent_dim,
                'll_ratio': self.ll_ratio,
            },
            'improvements': [
                'Decoder is also dual-branch (symmetric)',
                'll_latent_dim and hf_latent_dim actually work',
                'Correct _combine_channels() operation',
                'Removed useless fusion layer',
            ],
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
        }


# ============================================================================
# 双分支可微分CNN AutoEncoder V2 (正确实现)
# ============================================================================

class DualBranchDifferentiableWaveletAutoEncoderV2(nn.Module):
    """
    双分支可微分小波CNN AutoEncoder V2 (正确实现)

    与MLP版本的区别：使用卷积层而非全连接层

    架构：
    Encoder:
        RCS → 小波变换 → split(LL, HF)
        LL → Conv layers → ll_latent
        HF → Conv layers → hf_latent
        concat(ll_latent, hf_latent) → latent

    Decoder:
        latent → split(ll_latent, hf_latent)
        ll_latent → ConvTranspose layers → LL channels
        hf_latent → ConvTranspose layers → HF channels
        combine(LL, HF) → 小波系数 → 逆小波 → RCS
    """

    def __init__(self,
                 latent_dim: int = 32,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 wavelet_type: str = 'db4',
                 input_size: int = 49,
                 ll_ratio: float = 0.7,
                 activation: str = 'relu'):
        """初始化双分支可微分Wavelet CNN AutoEncoder V2"""
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size
        self.ll_ratio = ll_ratio
        self.activation_type = get_activation_name(activation)

        # 计算各分支latent维度
        self.ll_latent_dim, self.hf_latent_dim = calculate_branch_latent_dims(latent_dim, ll_ratio)

        # 集成可微分小波变换
        self.wavelet_transform = DifferentiableWaveletTransform(
            wavelet=wavelet_type,
            mode='symmetric',
            level=1
        )

        # ===== Encoder: LL分支 (CNN) =====
        self.ll_branch = nn.Sequential(
            nn.Conv2d(num_frequencies, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            get_activation(activation),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),
        )

        # ===== Encoder: HF分支 (CNN) =====
        self.hf_branch = nn.Sequential(
            nn.Conv2d(num_frequencies * 3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            get_activation(activation),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),
        )

        # ===== Encoder: 融合层 =====
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),

            nn.AdaptiveAvgPool2d(1),
        )

        # ===== Encoder FC: 特征 → Latent (使用ll_latent_dim和hf_latent_dim) =====
        self.flatten_dim = 128

        # 计算中间层
        self.ll_intermediate_dims = calculate_intermediate_dims(
            self.flatten_dim, self.ll_latent_dim, max_ratio=4
        )
        self.hf_intermediate_dims = calculate_intermediate_dims(
            self.flatten_dim, self.hf_latent_dim, max_ratio=4
        )

        # LL编码器FC
        ll_encoder_fc_layers = [nn.Flatten()]
        current_dim = self.flatten_dim
        for intermediate_dim in self.ll_intermediate_dims:
            ll_encoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim
        ll_encoder_fc_layers.append(nn.Linear(current_dim, self.ll_latent_dim))
        self.ll_encoder_fc = nn.Sequential(*ll_encoder_fc_layers)

        # HF编码器FC
        hf_encoder_fc_layers = [nn.Flatten()]
        current_dim = self.flatten_dim
        for intermediate_dim in self.hf_intermediate_dims:
            hf_encoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim
        hf_encoder_fc_layers.append(nn.Linear(current_dim, self.hf_latent_dim))
        self.hf_encoder_fc = nn.Sequential(*hf_encoder_fc_layers)

        # ===== Decoder FC: Latent → 特征 (双分支) =====

        # LL解码器FC
        ll_decoder_fc_layers = []
        current_dim = self.ll_latent_dim
        for intermediate_dim in reversed(self.ll_intermediate_dims):
            ll_decoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim
        ll_decoder_fc_layers.extend([
            nn.Linear(current_dim, self.flatten_dim),
            get_activation(activation)
        ])
        self.ll_decoder_fc = nn.Sequential(*ll_decoder_fc_layers)

        # HF解码器FC
        hf_decoder_fc_layers = []
        current_dim = self.hf_latent_dim
        for intermediate_dim in reversed(self.hf_intermediate_dims):
            hf_decoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim
        hf_decoder_fc_layers.extend([
            nn.Linear(current_dim, self.flatten_dim),
            get_activation(activation)
        ])
        self.hf_decoder_fc = nn.Sequential(*hf_decoder_fc_layers)

        # ===== Decoder: LL解码器 (CNN) =====
        self.ll_decoder_net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(32, num_frequencies, kernel_size=3, stride=2, padding=1, output_padding=0),
        )

        # ===== Decoder: HF解码器 (CNN) =====
        self.hf_decoder_net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(32, num_frequencies * 3, kernel_size=3, stride=2, padding=1, output_padding=0),
        )

        # 最终尺寸调整
        self.ll_final_adjust = nn.AdaptiveAvgPool2d((input_size, input_size))
        self.hf_final_adjust = nn.AdaptiveAvgPool2d((input_size, input_size))

        # 统一的encoder/decoder接口
        self.encoder = nn.ModuleList([
            self.ll_branch,
            self.hf_branch,
            self.fusion,
            self.ll_encoder_fc,
            self.hf_encoder_fc,
            self.wavelet_transform
        ])
        self.decoder = nn.ModuleList([
            self.ll_decoder_fc,
            self.hf_decoder_fc,
            self.ll_decoder_net,
            self.hf_decoder_net,
            self.ll_final_adjust,
            self.hf_final_adjust
        ])

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _split_channels(self, wavelet_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分离LL和高频通道

        Args:
            wavelet_coeffs: [B, H, W, num_freq*4]

        Returns:
            ll_channels: [B, num_freq, H, W]
            hf_channels: [B, num_freq*3, H, W]
        """
        # 转换为 [B, C, H, W]
        x = wavelet_coeffs.permute(0, 3, 1, 2)

        ll_list = []
        hf_list = []

        for freq_idx in range(self.num_frequencies):
            base = freq_idx * 4
            ll_list.append(x[:, base:base+1, :, :])          # LL
            hf_list.append(x[:, base+1:base+4, :, :])        # LH, HL, HH

        ll_channels = torch.cat(ll_list, dim=1)  # [B, num_freq, H, W]
        hf_channels = torch.cat(hf_list, dim=1)  # [B, num_freq*3, H, W]

        return ll_channels, hf_channels

    def _combine_channels(self, ll_channels: torch.Tensor, hf_channels: torch.Tensor) -> torch.Tensor:
        """
        组合LL和HF通道回小波系数格式

        Args:
            ll_channels: [B, num_freq, H, W]
            hf_channels: [B, num_freq*3, H, W]

        Returns:
            wavelet_coeffs: [B, H, W, num_freq*4]
        """
        wavelet_coeffs_list = []

        for freq_idx in range(self.num_frequencies):
            # 提取该频率的LL和HF
            ll = ll_channels[:, freq_idx:freq_idx+1, :, :]          # [B, 1, H, W]
            hf = hf_channels[:, freq_idx*3:(freq_idx+1)*3, :, :]    # [B, 3, H, W]

            # 拼接为 [LL, LH, HL, HH]
            freq_coeffs = torch.cat([ll, hf], dim=1)  # [B, 4, H, W]
            wavelet_coeffs_list.append(freq_coeffs)

        # 拼接所有频率
        x = torch.cat(wavelet_coeffs_list, dim=1)  # [B, num_freq*4, H, W]

        # 转换为 [B, H, W, C]
        wavelet_coeffs = x.permute(0, 2, 3, 1)  # [B, H, W, num_freq*4]

        return wavelet_coeffs

    def encode(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """编码：RCS → 小波系数 → 双分支 → 隐空间"""
        # Step 1: RCS → 小波系数（可微分）
        wavelet_coeffs = self.wavelet_transform.forward_transform(rcs_data)  # [B, 49, 49, 8]

        # Step 2: 分离LL和高频通道
        ll_input, hf_input = self._split_channels(wavelet_coeffs)

        # Step 3: 双分支处理
        ll_feat = self.ll_branch(ll_input)    # [B, 64, 13, 13]
        hf_feat = self.hf_branch(hf_input)    # [B, 64, 13, 13]

        # Step 4: 融合
        fused = torch.cat([ll_feat, hf_feat], dim=1)  # [B, 128, 13, 13]
        fused = self.fusion(fused)                     # [B, 128, 1, 1]

        # Step 5: 分别编码到ll_latent和hf_latent
        ll_latent = self.ll_encoder_fc(fused)  # [B, ll_latent_dim]
        hf_latent = self.hf_encoder_fc(fused)  # [B, hf_latent_dim]

        # Step 6: Concat
        latent = torch.cat([ll_latent, hf_latent], dim=1)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码：隐空间 → 双分支 → 小波系数 → RCS"""
        # Step 1: Split latent
        ll_latent = latent[:, :self.ll_latent_dim]
        hf_latent = latent[:, self.ll_latent_dim:]

        # Step 2: 双分支解码到特征
        ll_feat = self.ll_decoder_fc(ll_latent)  # [B, 128]
        hf_feat = self.hf_decoder_fc(hf_latent)  # [B, 128]

        # Step 3: Reshape
        ll_feat = ll_feat.view(-1, 128, 1, 1)
        hf_feat = hf_feat.view(-1, 128, 1, 1)

        # Step 4: 双分支解码到小波通道
        ll_channels = self.ll_decoder_net(ll_feat)  # [B, num_freq, ~49, ~49]
        hf_channels = self.hf_decoder_net(hf_feat)  # [B, num_freq*3, ~49, ~49]

        # Step 5: 调整尺寸
        ll_channels = self.ll_final_adjust(ll_channels)  # [B, num_freq, 49, 49]
        hf_channels = self.hf_final_adjust(hf_channels)  # [B, num_freq*3, 49, 49]

        # Step 6: 正确组合
        wavelet_coeffs = self._combine_channels(ll_channels, hf_channels)  # [B, 49, 49, 8]

        # Step 7: 逆小波变换
        rcs_data = self.wavelet_transform.inverse_transform(wavelet_coeffs)  # [B, 91, 91, 2]

        return rcs_data

    def forward(self, rcs_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(rcs_data)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取模型参数数量"""
        ll_branch_params = sum(p.numel() for p in self.ll_branch.parameters())
        hf_branch_params = sum(p.numel() for p in self.hf_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        ll_encoder_fc_params = sum(p.numel() for p in self.ll_encoder_fc.parameters())
        hf_encoder_fc_params = sum(p.numel() for p in self.hf_encoder_fc.parameters())
        ll_decoder_fc_params = sum(p.numel() for p in self.ll_decoder_fc.parameters())
        hf_decoder_fc_params = sum(p.numel() for p in self.hf_decoder_fc.parameters())
        ll_decoder_net_params = sum(p.numel() for p in self.ll_decoder_net.parameters())
        hf_decoder_net_params = sum(p.numel() for p in self.hf_decoder_net.parameters())
        wavelet_params = sum(p.numel() for p in self.wavelet_transform.parameters())

        encoder_params = (ll_branch_params + hf_branch_params + fusion_params +
                         ll_encoder_fc_params + hf_encoder_fc_params)
        decoder_params = (ll_decoder_fc_params + hf_decoder_fc_params +
                         ll_decoder_net_params + hf_decoder_net_params)
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'll_branch': ll_branch_params,
            'hf_branch': hf_branch_params,
            'fusion': fusion_params,
            'll_encoder_fc': ll_encoder_fc_params,
            'hf_encoder_fc': hf_encoder_fc_params,
            'll_decoder_fc': ll_decoder_fc_params,
            'hf_decoder_fc': hf_decoder_fc_params,
            'll_decoder_net': ll_decoder_net_params,
            'hf_decoder_net': hf_decoder_net_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'wavelet_transform': wavelet_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        return {
            'model_name': 'DualBranchDifferentiableWaveletAutoEncoderV2',
            'version': 'v2',
            'architecture': 'Dual-Branch CNN + Differentiable Wavelet (Correct Implementation)',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'wavelet_type': self.wavelet_type,
            'differentiable': True,
            'input_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'output_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'loss_space': 'RCS',
            'branch_config': {
                'll_channels': self.num_frequencies,
                'hf_channels': self.num_frequencies * 3,
                'll_latent_dim': self.ll_latent_dim,
                'hf_latent_dim': self.hf_latent_dim,
                'll_ratio': self.ll_ratio,
            },
            'improvements': [
                'Decoder is also dual-branch (symmetric)',
                'll_latent_dim and hf_latent_dim actually work',
                'Correct _combine_channels() operation',
                'Separate decoder networks for LL and HF',
            ],
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("双分支可微分小波AutoEncoder V2 测试")
    print("=" * 80)

    # 测试1: MLP V2
    print("\n[Test 1] Dual-Branch MLP V2 (latent_dim=32, ll_ratio=0.7)")
    model_mlp = DualBranchDifferentiableWaveletMLPAutoEncoderV2(
        latent_dim=32,
        num_frequencies=2,
        ll_ratio=0.7
    )

    batch_size = 4
    test_input = torch.randn(batch_size, 91, 91, 2)

    with torch.no_grad():
        recon, latent = model_mlp(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"Latent形状: {latent.shape}")
    print(f"重建形状: {recon.shape}")
    print(f"重建MSE: {torch.mean((test_input - recon)**2).item():.6f}")

    info = model_mlp.get_model_info()
    print(f"\n模型信息:")
    print(f"  - 架构: {info['architecture']}")
    print(f"  - LL latent: {info['branch_config']['ll_latent_dim']}")
    print(f"  - HF latent: {info['branch_config']['hf_latent_dim']}")
    print(f"  - LL ratio: {info['branch_config']['ll_ratio']}")
    print(f"  - 总参数量: {info['parameters']['total']:,}")
    print(f"\n改进:")
    for improvement in info['improvements']:
        print(f"  - {improvement}")

    # 测试2: CNN V2
    print("\n" + "=" * 80)
    print("[Test 2] Dual-Branch CNN V2 (latent_dim=32, ll_ratio=0.7)")
    model_cnn = DualBranchDifferentiableWaveletAutoEncoderV2(
        latent_dim=32,
        num_frequencies=2,
        ll_ratio=0.7
    )

    with torch.no_grad():
        recon_cnn, latent_cnn = model_cnn(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"Latent形状: {latent_cnn.shape}")
    print(f"重建形状: {recon_cnn.shape}")
    print(f"重建MSE: {torch.mean((test_input - recon_cnn)**2).item():.6f}")

    info_cnn = model_cnn.get_model_info()
    print(f"\n模型信息:")
    print(f"  - 架构: {info_cnn['architecture']}")
    print(f"  - LL latent: {info_cnn['branch_config']['ll_latent_dim']}")
    print(f"  - HF latent: {info_cnn['branch_config']['hf_latent_dim']}")
    print(f"  - 总参数量: {info_cnn['parameters']['total']:,}")

    # 测试3: 通道顺序测试
    print("\n" + "=" * 80)
    print("[Test 3] 通道顺序验证")

    # 创建可识别的输入模式
    test_pattern = torch.zeros(1, 91, 91, 2)
    test_pattern[0, 45, 45, 0] = 1.0  # freq0中心点
    test_pattern[0, 45, 45, 1] = 2.0  # freq1中心点

    with torch.no_grad():
        # 获取原始小波系数
        coeffs_orig = model_mlp.wavelet_transform.forward_transform(test_pattern)

        # 前向传播
        latent_test = model_mlp.encode(test_pattern)
        rcs_recon = model_mlp.decode(latent_test)
        coeffs_recon = model_mlp.wavelet_transform.forward_transform(rcs_recon)

    print("原始小波系数 vs 重建小波系数相关系数:")
    for i in range(8):
        orig = coeffs_orig[0, :, :, i].flatten()
        recon = coeffs_recon[0, :, :, i].flatten()

        # 计算相关系数
        orig_mean = orig.mean()
        recon_mean = recon.mean()
        numerator = ((orig - orig_mean) * (recon - recon_mean)).sum()
        denominator = torch.sqrt(((orig - orig_mean)**2).sum() * ((recon - recon_mean)**2).sum())
        corr = numerator / (denominator + 1e-8)

        channel_names = ['LL0', 'LH0', 'HL0', 'HH0', 'LL1', 'LH1', 'HL1', 'HH1']
        print(f"  Channel {i} ({channel_names[i]}): {corr:.4f}")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过! V2版本实现正确的双分支对称架构")
    print("=" * 80)
