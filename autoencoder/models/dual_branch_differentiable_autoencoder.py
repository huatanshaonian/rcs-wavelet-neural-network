"""
双分支可微分小波AutoEncoder模型

结合双分支架构和可微分小波变换：
- 双分支：分离处理LL通道（>90%能量）和高频通道（<10%能量）
- 可微分：小波变换集成为nn.Module，损失在RCS空间，梯度可回传

设计思想：
- LL分支：大卷积核捕捉全局低频特征
- HF分支：小卷积核捕捉高频细节
- 端到端训练：RCS → 小波(torch) → 双分支AE → 逆小波(torch) → RCS

支持架构：
1. DualBranchDifferentiableWaveletAutoEncoder (CNN)
2. DualBranchDifferentiableWaveletMLPAutoEncoder (MLP)
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
from .dual_branch_autoencoder import calculate_branch_latent_dims


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

    Example:
        >>> calculate_intermediate_dims(4096, 32, max_ratio=4)
        [1024, 256, 64]  # 4096→1024→256→64→32
    """
    if input_dim <= latent_dim:
        return []

    dims = []
    current = input_dim

    # 持续压缩直到接近目标维度
    while current > latent_dim * max_ratio:
        # 压缩到当前的1/2到1/4之间
        next_dim = max(latent_dim, current // max_ratio)
        # 向上取整到2的幂次（如64, 128, 256...）
        next_dim = 2 ** round(np.log2(next_dim))
        # 确保不小于latent_dim
        next_dim = max(next_dim, latent_dim)

        if next_dim < current:
            dims.append(next_dim)
            current = next_dim
        else:
            break

    return dims


# ============================================================================
# 双分支可微分CNN AutoEncoder
# ============================================================================

class DualBranchDifferentiableWaveletAutoEncoder(nn.Module):
    """
    双分支可微分小波CNN AutoEncoder

    架构：
    - 集成可微分小波变换（nn.Module）
    - LL分支：处理LL通道（大卷积核7×7，全局特征）
    - HF分支：处理LH/HL/HH通道（小卷积核3×3，细节特征）
    - 融合层：特征图拼接 → 继续卷积
    - 统一latent：支持小隐空间（如16-32维）

    输入：RCS data [B, 91, 91, 2]
    输出：RCS data [B, 91, 91, 2]
    损失：RCS空间（梯度可回传通过小波变换）
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
        初始化双分支可微分Wavelet CNN AutoEncoder

        Args:
            latent_dim: 总隐空间维度（如32）
            num_frequencies: 频率数量（2 or 3）
            dropout_rate: Dropout率
            wavelet_type: 小波类型（db4, haar等）
            input_size: 小波系数空间尺寸（默认49）
            ll_ratio: LL分支latent占比（默认0.7）
            activation: 激活函数类型 (例如 'relu', 'sin', 'gelu', 'swish')
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size
        self.activation_type = get_activation_name(activation)

        # 计算各分支latent维度
        self.ll_latent_dim, self.hf_latent_dim = calculate_branch_latent_dims(latent_dim, ll_ratio)

        # ===== 关键：集成可微分小波变换 =====
        self.wavelet_transform = DifferentiableWaveletTransform(
            wavelet=wavelet_type,
            mode='symmetric',
            level=1
        )

        # ===== LL分支：处理低频通道 =====
        # 输入: [B, num_freq, 49, 49] (每个频率1个LL通道)
        self.ll_branch = nn.Sequential(
            # 第一层：大卷积核捕捉全局特征
            nn.Conv2d(num_frequencies, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            get_activation(activation),

            # 下采样1: [49, 49] → [25, 25]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            # 下采样2: [25, 25] → [13, 13]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),
        )

        # ===== HF分支：处理高频通道 =====
        # 输入: [B, num_freq*3, 49, 49] (每个频率3个高频通道: LH, HL, HH)
        self.hf_branch = nn.Sequential(
            # 第一层：小卷积核捕捉细节
            nn.Conv2d(num_frequencies * 3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            get_activation(activation),

            # 下采样1: [49, 49] → [25, 25]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            # 下采样2: [25, 25] → [13, 13]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),
        )

        # ===== 融合层 =====
        # 输入: [B, 128, 13, 13] (64 from LL + 64 from HF)
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            # 下采样3: [13, 13] → [7, 7]
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),

            # 全局池化: [7, 7] → [1, 1]
            nn.AdaptiveAvgPool2d(1),
        )

        # ===== Encoder FC: 特征 → Latent (支持小隐空间) =====
        self.flatten_dim = 128
        self.intermediate_dims = calculate_intermediate_dims(
            self.flatten_dim, latent_dim, max_ratio=4
        )

        # 构建渐进式压缩层
        encoder_fc_layers = [nn.Flatten()]
        current_dim = self.flatten_dim

        for intermediate_dim in self.intermediate_dims:
            encoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        encoder_fc_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder_fc = nn.Sequential(*encoder_fc_layers)

        # ===== Decoder FC: Latent → 特征 =====
        decoder_fc_layers = []
        current_dim = latent_dim

        for intermediate_dim in reversed(self.intermediate_dims):
            decoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        decoder_fc_layers.extend([
            nn.Linear(current_dim, self.flatten_dim),
            get_activation(activation)
        ])
        self.decoder_fc = nn.Sequential(*decoder_fc_layers)

        # ===== 解码器：重建小波系数 =====
        self.decoder_net = nn.Sequential(
            # 重塑: [128] → [128, 1, 1]
            # 上采样1: [1, 1] → [7, 7]
            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            # 上采样2: [7, 7] → [13, 13]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            # 上采样3: [13, 13] → [25, 25]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout_rate),

            # 上采样4: [25, 25] → [49, 49]
            nn.ConvTranspose2d(32, num_frequencies * 4, kernel_size=3, stride=2, padding=1, output_padding=0),
            # 最后一层不加激活，允许小波系数为负值
        )

        # 最终尺寸调整
        self.final_adjust = nn.AdaptiveAvgPool2d((input_size, input_size))

        # 统一的encoder/decoder接口（训练时需要）
        self.encoder = nn.ModuleList([
            self.ll_branch,
            self.hf_branch,
            self.fusion,
            self.encoder_fc,
            self.wavelet_transform  # 包含小波变换
        ])
        self.decoder = nn.ModuleList([
            self.decoder_fc,
            self.decoder_net,
            self.final_adjust,
            # 逆小波变换在decode()中调用self.wavelet_transform.inverse_transform
        ])

        # 保存结构信息
        self.structure_info = get_structure_info(
            self.flatten_dim, latent_dim, self.intermediate_dims
        )

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

    def encode(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """
        编码：RCS → 小波系数 → 双分支 → 隐空间

        Args:
            rcs_data: [B, H, W, C] RCS数据

        Returns:
            latent: [B, latent_dim]
        """
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

        # Step 5: 编码到latent
        latent = self.encoder_fc(fused)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码：隐空间 → 小波系数 → RCS

        Args:
            latent: [B, latent_dim]

        Returns:
            rcs_data: [B, H, W, C] 重建的RCS数据
        """
        # Step 1: Latent → 特征
        x = self.decoder_fc(latent)           # [B, 128]
        x = x.view(-1, 128, 1, 1)             # [B, 128, 1, 1]

        # Step 2: 解码到小波系数
        x = self.decoder_net(x)               # [B, num_freq*4, ~49, ~49]
        x = self.final_adjust(x)              # [B, num_freq*4, 49, 49]

        # Step 3: 转换为 [B, H, W, C]
        wavelet_coeffs = x.permute(0, 2, 3, 1)  # [B, 49, 49, 8]

        # Step 4: 小波系数 → RCS（可微分）
        rcs_data = self.wavelet_transform.inverse_transform(wavelet_coeffs)  # [B, 91, 91, 2]

        return rcs_data

    def forward(self, rcs_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：RCS → 隐空间 → 重建RCS

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
        ll_params = sum(p.numel() for p in self.ll_branch.parameters())
        hf_params = sum(p.numel() for p in self.hf_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        encoder_fc_params = sum(p.numel() for p in self.encoder_fc.parameters())
        decoder_fc_params = sum(p.numel() for p in self.decoder_fc.parameters())
        decoder_net_params = sum(p.numel() for p in self.decoder_net.parameters())
        wavelet_params = sum(p.numel() for p in self.wavelet_transform.parameters())

        encoder_params = ll_params + hf_params + fusion_params + encoder_fc_params
        decoder_params = decoder_fc_params + decoder_net_params
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'll_branch': ll_params,
            'hf_branch': hf_params,
            'fusion': fusion_params,
            'encoder_fc': encoder_fc_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'wavelet_transform': wavelet_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        # 构建FC层结构描述
        fc_structure = [self.flatten_dim] + self.intermediate_dims + [self.latent_dim]
        fc_structure_str = ' → '.join(map(str, fc_structure))

        return {
            'model_name': 'DualBranchDifferentiableWaveletAutoEncoder',
            'architecture': 'Dual-Branch CNN + Differentiable Wavelet',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'wavelet_type': self.wavelet_type,
            'differentiable': True,
            'input_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'output_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'loss_space': 'RCS',  # 关键：损失在RCS空间
            'branch_config': {
                'll_channels': self.num_frequencies,
                'hf_channels': self.num_frequencies * 3,
                'll_latent_dim': self.ll_latent_dim,
                'hf_latent_dim': self.hf_latent_dim,
            },
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'fc_structure': fc_structure_str,
            'intermediate_dims': self.intermediate_dims,
            **self.structure_info
        }


# ============================================================================
# 双分支可微分MLP AutoEncoder
# ============================================================================

class DualBranchDifferentiableWaveletMLPAutoEncoder(nn.Module):
    """
    双分支可微分小波MLP AutoEncoder

    架构：
    - 集成可微分小波变换（nn.Module）
    - LL分支：MLP处理LL通道
    - HF分支：MLP处理高频通道
    - 融合层：向量拼接 → 继续MLP
    - 统一latent：支持小隐空间

    与CNN版本的区别：
    - 使用全连接层而非卷积层
    - flatten输入后直接处理

    输入：RCS data [B, 91, 91, 2]
    输出：RCS data [B, 91, 91, 2]
    """

    def __init__(self,
                 latent_dim: int = 32,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 wavelet_type: str = 'db4',
                 input_size: int = 49,
                 ll_ratio: float = 0.7,
                 activation: str = 'relu'):
        """初始化双分支可微分Wavelet MLP AutoEncoder"""
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size
        self.activation_type = get_activation_name(activation)

        # 计算各分支latent维度
        self.ll_latent_dim, self.hf_latent_dim = calculate_branch_latent_dims(latent_dim, ll_ratio)

        # ===== 关键：集成可微分小波变换 =====
        self.wavelet_transform = DifferentiableWaveletTransform(
            wavelet=wavelet_type,
            mode='symmetric',
            level=1
        )

        # 计算输入维度
        ll_input_dim = input_size * input_size * num_frequencies      # 49*49*2 = 4802
        hf_input_dim = input_size * input_size * (num_frequencies * 3) # 49*49*6 = 14406

        # ===== LL分支 =====
        self.ll_branch = nn.Sequential(
            nn.Linear(ll_input_dim, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            get_activation(activation),
            nn.Dropout(dropout_rate),
        )

        # ===== HF分支 =====
        self.hf_branch = nn.Sequential(
            nn.Linear(hf_input_dim, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            get_activation(activation),
            nn.Dropout(dropout_rate),
        )

        # ===== 融合层 =====
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),  # 128(LL) + 128(HF) = 256
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(128, latent_dim),
        )

        # ===== 解码器 =====
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(512, ll_input_dim + hf_input_dim),  # 完整小波系数
        )

        # 统一的encoder/decoder接口
        self.encoder = nn.ModuleList([
            self.ll_branch,
            self.hf_branch,
            self.fusion,
            self.wavelet_transform
        ])
        self.decoder = nn.ModuleList([self.decoder_fc])

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _split_and_flatten(self, wavelet_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分离并展平LL和高频通道

        Args:
            wavelet_coeffs: [B, H, W, num_freq*4]

        Returns:
            ll_flat: [B, H*W*num_freq]
            hf_flat: [B, H*W*num_freq*3]
        """
        batch_size = wavelet_coeffs.shape[0]

        ll_list = []
        hf_list = []

        for freq_idx in range(self.num_frequencies):
            base = freq_idx * 4
            ll_list.append(wavelet_coeffs[:, :, :, base:base+1])       # LL
            hf_list.append(wavelet_coeffs[:, :, :, base+1:base+4])     # LH, HL, HH

        ll_channels = torch.cat(ll_list, dim=3)  # [B, H, W, num_freq]
        hf_channels = torch.cat(hf_list, dim=3)  # [B, H, W, num_freq*3]

        # Flatten
        ll_flat = ll_channels.reshape(batch_size, -1)
        hf_flat = hf_channels.reshape(batch_size, -1)

        return ll_flat, hf_flat

    def encode(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """
        编码：RCS → 小波系数 → 双分支 → 隐空间

        Args:
            rcs_data: [B, H, W, C] RCS数据

        Returns:
            latent: [B, latent_dim]
        """
        # Step 1: RCS → 小波系数（可微分）
        wavelet_coeffs = self.wavelet_transform.forward_transform(rcs_data)  # [B, 49, 49, 8]

        # Step 2: 分离并展平
        ll_input, hf_input = self._split_and_flatten(wavelet_coeffs)

        # Step 3: 双分支处理
        ll_feat = self.ll_branch(ll_input)  # [B, 128]
        hf_feat = self.hf_branch(hf_input)  # [B, 128]

        # Step 4: 融合
        fused = torch.cat([ll_feat, hf_feat], dim=1)  # [B, 256]
        latent = self.fusion(fused)                    # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码：隐空间 → 小波系数 → RCS

        Args:
            latent: [B, latent_dim]

        Returns:
            rcs_data: [B, H, W, C] 重建的RCS数据
        """
        # Step 1: 解码
        x = self.decoder_fc(latent)  # [B, H*W*num_freq*4]

        # Step 2: 重塑为小波系数
        batch_size = latent.shape[0]
        wavelet_coeffs = x.view(batch_size, self.input_size, self.input_size, self.num_frequencies * 4)

        # Step 3: 小波系数 → RCS（可微分）
        rcs_data = self.wavelet_transform.inverse_transform(wavelet_coeffs)  # [B, 91, 91, 2]

        return rcs_data

    def forward(self, rcs_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(rcs_data)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取模型参数数量（兼容现有系统）"""
        ll_params = sum(p.numel() for p in self.ll_branch.parameters())
        hf_params = sum(p.numel() for p in self.hf_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        decoder_params = sum(p.numel() for p in self.decoder_fc.parameters())
        wavelet_params = sum(p.numel() for p in self.wavelet_transform.parameters())

        encoder_params = ll_params + hf_params + fusion_params
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'll_branch': ll_params,
            'hf_branch': hf_params,
            'fusion': fusion_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'wavelet_transform': wavelet_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        return {
            'model_name': 'DualBranchDifferentiableWaveletMLPAutoEncoder',
            'architecture': 'Dual-Branch MLP + Differentiable Wavelet',
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
            },
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("双分支可微分小波AutoEncoder测试")
    print("=" * 80)

    # 测试1: 双分支可微分CNN (32维小隐空间)
    print("\n[Test 1] 双分支可微分Wavelet CNN (latent_dim=32)")
    model_cnn = DualBranchDifferentiableWaveletAutoEncoder(
        latent_dim=32,
        num_frequencies=2
    )

    batch_size = 4
    # 输入是RCS数据，不是小波系数
    test_input = torch.randn(batch_size, 91, 91, 2)  # RCS数据

    print(f"输入形状: {test_input.shape} (RCS数据)")

    with torch.no_grad():
        recon, latent = model_cnn(test_input)

    print(f"Latent形状: {latent.shape}")
    print(f"重建形状: {recon.shape} (RCS数据)")
    print(f"重建MSE: {torch.mean((test_input - recon)**2).item():.6f}")

    info = model_cnn.get_model_info()
    print(f"\n模型信息:")
    print(f"  - 架构: {info['architecture']}")
    print(f"  - 损失空间: {info['loss_space']}")
    print(f"  - LL分支通道: {info['branch_config']['ll_channels']}")
    print(f"  - HF分支通道: {info['branch_config']['hf_channels']}")
    print(f"  - LL latent维度: {info['branch_config']['ll_latent_dim']}")
    print(f"  - HF latent维度: {info['branch_config']['hf_latent_dim']}")
    print(f"  - 总参数量: {info['parameters']['total']:,}")
    print(f"  - FC结构: {info['fc_structure']}")

    # 测试2: 双分支可微分MLP (32维小隐空间)
    print("\n" + "=" * 80)
    print("[Test 2] 双分支可微分Wavelet MLP (latent_dim=32)")
    model_mlp = DualBranchDifferentiableWaveletMLPAutoEncoder(
        latent_dim=32,
        num_frequencies=2
    )

    with torch.no_grad():
        recon_mlp, latent_mlp = model_mlp(test_input)

    print(f"输入形状: {test_input.shape} (RCS数据)")
    print(f"Latent形状: {latent_mlp.shape}")
    print(f"重建形状: {recon_mlp.shape} (RCS数据)")
    print(f"重建MSE: {torch.mean((test_input - recon_mlp)**2).item():.6f}")

    info_mlp = model_mlp.get_model_info()
    print(f"\n模型信息:")
    print(f"  - 架构: {info_mlp['architecture']}")
    print(f"  - 损失空间: {info_mlp['loss_space']}")
    print(f"  - LL分支参数: {info_mlp['parameters']['ll_branch']:,}")
    print(f"  - HF分支参数: {info_mlp['parameters']['hf_branch']:,}")
    print(f"  - 小波变换参数: {info_mlp['parameters']['wavelet_transform']:,}")
    print(f"  - 总参数量: {info_mlp['parameters']['total']:,}")

    # 测试3: 不同隐空间维度
    print("\n" + "=" * 80)
    print("[Test 3] 测试不同隐空间维度")

    for latent_dim in [16, 32, 64, 128, 256]:
        model = DualBranchDifferentiableWaveletAutoEncoder(
            latent_dim=latent_dim,
            num_frequencies=2
        )
        ll_dim, hf_dim = calculate_branch_latent_dims(latent_dim, 0.7)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"  latent_dim={latent_dim:3d}: LL={ll_dim:3d}, HF={hf_dim:3d}, 参数量={total_params:,}")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过!")
    print("=" * 80)
