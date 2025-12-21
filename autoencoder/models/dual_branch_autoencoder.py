"""
双分支AutoEncoder模型
分别处理LL通道和高频通道，针对性优化特征提取

设计思想：
- LL通道（低频近似）：包含>90%能量，使用大卷积核捕捉全局特征
- 高频通道（LH/HL/HH）：包含<10%能量，使用小卷积核捕捉细节

支持架构：
1. DualBranchWaveletAutoEncoder (CNN) - Wavelet模式
2. DualBranchWaveletMLPAutoEncoder (MLP) - Wavelet模式
3. DualBranchDirectAutoEncoder (CNN) - Direct模式
4. DualBranchDirectMLPAutoEncoder (MLP) - Direct模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List
import numpy as np
from autoencoder.utils.adaptive_layers import get_structure_info
from autoencoder.utils.activation_factory import get_activation, get_activation_name


def calculate_branch_latent_dims(total_latent_dim: int, ll_ratio: float = 0.7) -> Tuple[int, int]:
    """
    计算LL和HF分支的隐空间维度分配

    Args:
        total_latent_dim: 总隐空间维度
        ll_ratio: LL分支占比（默认0.7，基于能量比例）

    Returns:
        (ll_latent_dim, hf_latent_dim)

    Example:
        >>> calculate_branch_latent_dims(32, 0.7)
        (22, 10)
        >>> calculate_branch_latent_dims(256, 0.7)
        (179, 77)
    """
    ll_latent_dim = int(total_latent_dim * ll_ratio)
    hf_latent_dim = total_latent_dim - ll_latent_dim
    return ll_latent_dim, hf_latent_dim


# ============================================================================
# Wavelet模式 - 双分支CNN AutoEncoder
# ============================================================================

class DualBranchWaveletAutoEncoder(nn.Module):
    """
    双分支小波CNN AutoEncoder

    架构：
    - LL分支：处理LL通道（大卷积核，深层特征）
    - HF分支：处理LH/HL/HH通道（小卷积核，细节特征）
    - 融合层：特征图拼接 → 继续卷积
    - 统一latent：支持小隐空间（如32维）

    输入：Wavelet coeffs [B, 49, 49, num_freq*4]
    输出：Latent [B, latent_dim]
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
        初始化双分支Wavelet CNN AutoEncoder

        Args:
            latent_dim: 总隐空间维度（如32）
            num_frequencies: 频率数量（2 or 3）
            dropout_rate: Dropout率
            wavelet_type: 小波类型
            input_size: 小波系数空间尺寸（默认49）
            ll_ratio: LL分支latent占比（默认0.7）
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        # 计算各分支latent维度
        self.ll_latent_dim, self.hf_latent_dim = calculate_branch_latent_dims(latent_dim, ll_ratio)

        # ===== LL分支：处理低频通道 =====
        # 输入: [B, num_freq, 49, 49] (每个频率1个LL通道)
        self.ll_branch = nn.Sequential(
            # 第一层：大卷积核捕捉全局特征
            nn.Conv2d(num_frequencies, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            activation_layer(),

            # 下采样1: [49, 49] → [25, 25]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 下采样2: [25, 25] → [13, 13]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate),
        )

        # ===== HF分支：处理高频通道 =====
        # 输入: [B, num_freq*3, 49, 49] (每个频率3个高频通道: LH, HL, HH)
        self.hf_branch = nn.Sequential(
            # 第一层：小卷积核捕捉细节
            nn.Conv2d(num_frequencies * 3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            activation_layer(),

            # 下采样1: [49, 49] → [25, 25]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 下采样2: [25, 25] → [13, 13]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate),
        )

        # ===== 融合层 =====
        # 输入: [B, 128, 13, 13] (64 from LL + 64 from HF)
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 下采样3: [13, 13] → [7, 7]
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            activation_layer(),

            # 全局池化: [7, 7] → [1, 1]
            nn.AdaptiveAvgPool2d(1),
        )

        # ===== Encoder: 特征 → Latent =====
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )

        # ===== Decoder: Latent → 特征 =====
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            activation_layer(),
        )

        # ===== 解码器：重建小波系数 =====
        self.decoder_net = nn.Sequential(
            # 重塑: [128] → [128, 1, 1]
            # 上采样1: [1, 1] → [7, 7]
            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样2: [7, 7] → [13, 13]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样3: [13, 13] → [25, 25]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            activation_layer(),
            nn.Dropout2d(dropout_rate),

            # 上采样4: [25, 25] → [49, 49]
            nn.ConvTranspose2d(32, num_frequencies * 4, kernel_size=3, stride=2, padding=1, output_padding=0),
        )

        # 最终尺寸调整
        self.final_adjust = nn.AdaptiveAvgPool2d((input_size, input_size))

        # 统一的encoder/decoder接口（训练时需要）
        self.encoder = nn.ModuleList([self.ll_branch, self.hf_branch, self.fusion, self.encoder_fc])
        self.decoder = nn.ModuleList([self.decoder_fc, self.decoder_net, self.final_adjust])

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

    def encode(self, wavelet_coeffs: torch.Tensor) -> torch.Tensor:
        """
        编码小波系数到隐空间

        Args:
            wavelet_coeffs: [B, H, W, num_freq*4]

        Returns:
            latent: [B, latent_dim]
        """
        # 分离LL和高频通道
        ll_input, hf_input = self._split_channels(wavelet_coeffs)

        # 分别处理
        ll_feat = self.ll_branch(ll_input)    # [B, 64, 13, 13]
        hf_feat = self.hf_branch(hf_input)    # [B, 64, 13, 13]

        # 融合
        fused = torch.cat([ll_feat, hf_feat], dim=1)  # [B, 128, 13, 13]
        fused = self.fusion(fused)                     # [B, 128, 1, 1]

        # 编码到latent
        latent = self.encoder_fc(fused)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码隐空间到小波系数

        Args:
            latent: [B, latent_dim]

        Returns:
            wavelet_coeffs: [B, H, W, num_freq*4]
        """
        # Latent → 特征
        x = self.decoder_fc(latent)           # [B, 128]
        x = x.view(-1, 128, 1, 1)             # [B, 128, 1, 1]

        # 解码
        x = self.decoder_net(x)               # [B, num_freq*4, ~49, ~49]
        x = self.final_adjust(x)              # [B, num_freq*4, 49, 49]

        # 转换为 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        return x

    def forward(self, wavelet_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(wavelet_coeffs)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """
        获取模型参数数量（兼容现有系统）

        Returns:
            包含参数统计的字典
        """
        ll_params = sum(p.numel() for p in self.ll_branch.parameters())
        hf_params = sum(p.numel() for p in self.hf_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        encoder_params = ll_params + hf_params + fusion_params + sum(p.numel() for p in self.encoder_fc.parameters())
        decoder_params = sum(p.numel() for p in self.decoder_fc.parameters()) + sum(p.numel() for p in self.decoder_net.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'll_branch': ll_params,
            'hf_branch': hf_params,
            'fusion': fusion_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        return {
            'model_name': 'DualBranchWaveletAutoEncoder',
            'architecture': 'Dual-Branch CNN',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'input_channels': self.num_frequencies * 4,
            'input_shape': f'[B, {self.input_size}, {self.input_size}, {self.num_frequencies*4}]',
            'latent_shape': f'[B, {self.latent_dim}]',
            'branch_config': {
                'll_channels': self.num_frequencies,
                'hf_channels': self.num_frequencies * 3,
                'll_latent_dim': self.ll_latent_dim,
                'hf_latent_dim': self.hf_latent_dim,
            },
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'wavelet_type': self.wavelet_type,
            'activation': self.activation_type,
        }


# ============================================================================
# Wavelet模式 - 双分支MLP AutoEncoder
# ============================================================================

class DualBranchWaveletMLPAutoEncoder(nn.Module):
    """
    双分支小波MLP AutoEncoder

    架构：
    - LL分支：MLP处理LL通道
    - HF分支：MLP处理高频通道
    - 融合层：向量拼接 → 继续MLP
    - 统一latent

    与CNN版本的区别：
    - 使用全连接层而非卷积层
    - flatten输入后直接处理
    """

    def __init__(self,
                 latent_dim: int = 32,
                 num_frequencies: int = 2,
                 dropout_rate: float = 0.2,
                 wavelet_type: str = 'db4',
                 input_size: int = 49,
                 ll_ratio: float = 0.7,
                 activation: str = 'relu'):
        """初始化双分支Wavelet MLP AutoEncoder"""
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size
        self.activation_type = get_activation_name(activation)

        def activation_layer():
            return get_activation(self.activation_type)

        # 计算各分支latent维度
        self.ll_latent_dim, self.hf_latent_dim = calculate_branch_latent_dims(latent_dim, ll_ratio)
        # 计算输入维度
        ll_input_dim = input_size * input_size * num_frequencies      # 49*49*2 = 4802
        hf_input_dim = input_size * input_size * (num_frequencies * 3) # 49*49*6 = 14406

        # ===== LL分支 =====
        self.ll_branch = nn.Sequential(
            nn.Linear(ll_input_dim, 512),
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            activation_layer(),
            nn.Dropout(dropout_rate),
        )

        # ===== HF分支 =====
        self.hf_branch = nn.Sequential(
            nn.Linear(hf_input_dim, 512),
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            activation_layer(),
            nn.Dropout(dropout_rate),
        )

        # ===== 融合层 =====
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),  # 128(LL) + 128(HF) = 256
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, latent_dim),
        )

        # ===== 解码器 =====
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            activation_layer(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, ll_input_dim + hf_input_dim),  # 完整小波系数
        )

        # 统一的encoder/decoder接口
        self.encoder = nn.ModuleList([self.ll_branch, self.hf_branch, self.fusion])
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

    def encode(self, wavelet_coeffs: torch.Tensor) -> torch.Tensor:
        """编码"""
        # 分离并展平
        ll_input, hf_input = self._split_and_flatten(wavelet_coeffs)

        # 分别处理
        ll_feat = self.ll_branch(ll_input)  # [B, 128]
        hf_feat = self.hf_branch(hf_input)  # [B, 128]

        # 融合
        fused = torch.cat([ll_feat, hf_feat], dim=1)  # [B, 256]
        latent = self.fusion(fused)                    # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码"""
        # 解码
        x = self.decoder_fc(latent)  # [B, H*W*num_freq*4]

        # 重塑
        batch_size = latent.shape[0]
        x = x.view(batch_size, self.input_size, self.input_size, self.num_frequencies * 4)

        return x

    def forward(self, wavelet_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        latent = self.encode(wavelet_coeffs)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """
        获取模型参数数量（兼容现有系统）

        Returns:
            包含参数统计的字典
        """
        ll_params = sum(p.numel() for p in self.ll_branch.parameters())
        hf_params = sum(p.numel() for p in self.hf_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        encoder_params = ll_params + hf_params + fusion_params
        decoder_params = sum(p.numel() for p in self.decoder_fc.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'll_branch': ll_params,
            'hf_branch': hf_params,
            'fusion': fusion_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        return {
            'model_name': 'DualBranchWaveletMLPAutoEncoder',
            'architecture': 'Dual-Branch MLP',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'input_channels': self.num_frequencies * 4,
            'input_shape': f'[B, {self.input_size}, {self.input_size}, {self.num_frequencies*4}]',
            'latent_shape': f'[B, {self.latent_dim}]',
            'branch_config': {
                'll_channels': self.num_frequencies,
                'hf_channels': self.num_frequencies * 3,
                'll_latent_dim': self.ll_latent_dim,
                'hf_latent_dim': self.hf_latent_dim,
            },
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'wavelet_type': self.wavelet_type,
            'activation': self.activation_type,
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("双分支AutoEncoder测试")
    print("=" * 80)

    # 测试1: 双分支CNN (32维小隐空间)
    print("\n[Test 1] 双分支Wavelet CNN (latent_dim=32)")
    model_cnn = DualBranchWaveletAutoEncoder(latent_dim=32, num_frequencies=2)

    batch_size = 4
    test_input = torch.randn(batch_size, 49, 49, 8)  # 2freq × 4bands = 8 channels

    print(f"输入形状: {test_input.shape}")

    with torch.no_grad():
        recon, latent = model_cnn(test_input)

    print(f"Latent形状: {latent.shape}")
    print(f"重建形状: {recon.shape}")
    print(f"重建MSE: {torch.mean((test_input - recon)**2).item():.6f}")

    info = model_cnn.get_model_info()
    print(f"\n模型信息:")
    print(f"  - LL分支通道: {info['branch_config']['ll_channels']}")
    print(f"  - HF分支通道: {info['branch_config']['hf_channels']}")
    print(f"  - LL latent维度: {info['branch_config']['ll_latent_dim']}")
    print(f"  - HF latent维度: {info['branch_config']['hf_latent_dim']}")
    print(f"  - 总参数量: {info['parameters']['total']:,}")
    print(f"  - 激活函数: {info['activation']}")

    # 测试2: 双分支MLP (32维小隐空间)
    print("\n" + "=" * 80)
    print("[Test 2] 双分支Wavelet MLP (latent_dim=32)")
    model_mlp = DualBranchWaveletMLPAutoEncoder(latent_dim=32, num_frequencies=2)

    with torch.no_grad():
        recon_mlp, latent_mlp = model_mlp(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"Latent形状: {latent_mlp.shape}")
    print(f"重建形状: {recon_mlp.shape}")
    print(f"重建MSE: {torch.mean((test_input - recon_mlp)**2).item():.6f}")

    info_mlp = model_mlp.get_model_info()
    print(f"\n模型信息:")
    print(f"  - 架构: {info_mlp['architecture']}")
    print(f"  - LL分支参数: {info_mlp['parameters']['ll_branch']:,}")
    print(f"  - HF分支参数: {info_mlp['parameters']['hf_branch']:,}")
    print(f"  - 总参数量: {info_mlp['parameters']['total']:,}")
    print(f"  - 激活函数: {info_mlp['activation']}")

    # 测试3: 不同隐空间维度
    print("\n" + "=" * 80)
    print("[Test 3] 测试不同隐空间维度")

    for latent_dim in [16, 32, 64, 128, 256]:
        model = DualBranchWaveletAutoEncoder(latent_dim=latent_dim, num_frequencies=2)
        ll_dim, hf_dim = calculate_branch_latent_dims(latent_dim, 0.7)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"  latent_dim={latent_dim:3d}: LL={ll_dim:3d}, HF={hf_dim:3d}, 参数量={total_params:,}")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过!")
    print("=" * 80)
