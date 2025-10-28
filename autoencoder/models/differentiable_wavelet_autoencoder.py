"""
可微分小波AutoEncoder模型

第三种模式：differentiable_wavelet
- 与传统wavelet模式不同，损失直接在RCS空间计算
- 小波变换集成到网络中，可以端到端训练
- 更容易添加物理约束（如ReLU保证RCS≥0）

架构对比：
- wavelet模式：  RCS → 小波(numpy) → AE → 逆小波(numpy) → RCS（损失在系数空间）
- differentiable模式：RCS → 小波(torch) → AE → 逆小波(torch) → RCS（损失在RCS空间）✓
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


class DifferentiableWaveletAutoEncoder(nn.Module):
    """
    可微分小波CNN AutoEncoder

    与WaveletAutoEncoder的区别：
    1. 集成了可微分小波变换（nn.Module）
    2. 输入/输出都是RCS数据（不是小波系数）
    3. 损失在RCS空间计算
    4. 可以端到端训练，梯度回传通过整个网络

    输入: [B, 91, 91, 2] RCS数据
    输出: [B, 91, 91, 2] 重建RCS数据
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 wavelet_type: str = 'db4',
                 input_size: int = 49):
        """
        初始化可微分小波CNN AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量
            wavelet_bands: 小波频带数（固定为4）
            dropout_rate: Dropout比率
            wavelet_type: 小波类型（db4, haar等）
            input_size: 小波系数尺寸（自动从wavelet_type计算）
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size

        # ===== 关键：集成可微分小波变换 =====
        self.wavelet_transform = DifferentiableWaveletTransform(
            wavelet=wavelet_type,
            mode='symmetric',
            level=1
        )

        # ===== CNN Encoder: 小波系数 → 隐空间 =====
        # [B, 49, 49, 8] → [B, 256]
        self.encoder = nn.Sequential(
            # 输入: [B, 8, input_size, input_size]
            nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=2, padding=1),
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

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ⚠️ 自动计算flatten维度（根据input_size动态调整）
        def calc_conv_output_size(input_size, num_layers=4):
            """计算经过num_layers层stride=2卷积后的输出尺寸"""
            h = input_size
            for _ in range(num_layers):
                h = (h + 2*1 - 3) // 2 + 1
            return h

        encoder_output_size = calc_conv_output_size(input_size, num_layers=4)
        self.flatten_dim = 256 * encoder_output_size * encoder_output_size
        self.encoder_output_size = encoder_output_size  # 保存用于decoder

        print(f"  CNN自适应: 小波{input_size}×{input_size} → Encoder输出{encoder_output_size}×{encoder_output_size}×256 = {self.flatten_dim}")

        # ===== 动态适配：自动计算中间层维度（支持小隐空间）=====
        # 策略：保持每级压缩比≤4:1，避免信息瓶颈
        self.intermediate_dims = calculate_intermediate_dims(
            self.flatten_dim, latent_dim, max_ratio=4
        )

        # 编码器全连接层：渐进式压缩
        # 示例: latent_dim=32 → [flatten_dim, 1024, 256, 64, 32]
        #       latent_dim=16 → [flatten_dim, 1024, 256, 64, 16]
        encoder_fc_layers = [nn.Flatten()]
        current_dim = self.flatten_dim

        # 构建中间层
        for intermediate_dim in self.intermediate_dims:
            encoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到latent_dim
        encoder_fc_layers.append(nn.Linear(current_dim, latent_dim))
        self.fc_encoder = nn.Sequential(*encoder_fc_layers)

        # ===== Decoder: 隐空间 → 小波系数 =====
        # 对称的解码器结构
        decoder_fc_layers = []
        current_dim = latent_dim

        # 反向构建中间层
        for intermediate_dim in reversed(self.intermediate_dims):
            decoder_fc_layers.extend([
                nn.Linear(current_dim, intermediate_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            current_dim = intermediate_dim

        # 最后一层到flatten_dim
        decoder_fc_layers.extend([
            nn.Linear(current_dim, self.flatten_dim),
            nn.ReLU(inplace=True)
        ])
        self.fc_decoder = nn.Sequential(*decoder_fc_layers)

        # 保存结构信息（用于 get_model_info）
        self.structure_info = get_structure_info(
            self.flatten_dim, latent_dim, self.intermediate_dims
        )

        self.decoder = nn.Sequential(
            # [B, 256, 4, 4]
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # → [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # → [B, 64, 31, 31]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.ConvTranspose2d(64, self.input_channels, kernel_size=3, stride=2, padding=1),  # → [B, 8, 61, 61]
            # 最后一层不加激活函数，允许小波系数为负值
        )

    def encode(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """
        编码：RCS → 小波系数 → 隐空间

        Args:
            rcs_data: [B, H, W, C] RCS数据

        Returns:
            latent: [B, latent_dim] 隐空间表示
        """
        # Step 1: RCS → 小波系数（可微分）
        wavelet_coeffs = self.wavelet_transform.forward_transform(rcs_data)  # [B, 49, 49, 8]

        # Step 2: 转换为[B, C, H, W]格式
        wavelet_coeffs = wavelet_coeffs.permute(0, 3, 1, 2)  # [B, 8, 49, 49]

        # Step 3: CNN编码
        features = self.encoder(wavelet_coeffs)  # [B, 256, 4, 4]

        # Step 4: FC到隐空间
        latent = self.fc_encoder(features)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码：隐空间 → 小波系数 → RCS

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            rcs_data: [B, H, W, C] 重建的RCS数据
        """
        # Step 1: FC展开
        features = self.fc_decoder(latent)  # [B, flatten_dim]
        features = features.view(-1, 256, self.encoder_output_size, self.encoder_output_size)  # [B, 256, H, H]

        # Step 2: CNN解码
        wavelet_coeffs = self.decoder(features)  # [B, 8, H', W']

        # Step 3: 裁剪到正确的小波尺寸
        target_size = self.wavelet_transform.wavelet_size
        if target_size is not None:
            h, w = target_size
            wavelet_coeffs = wavelet_coeffs[:, :, :h, :w]

        # Step 4: 转换为[B, H, W, C]格式
        wavelet_coeffs = wavelet_coeffs.permute(0, 2, 3, 1)  # [B, 49, 49, 8]

        # Step 5: 小波系数 → RCS（可微分）
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
        """获取参数数量"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        fc_enc_params = sum(p.numel() for p in self.fc_encoder.parameters() if p.requires_grad)
        fc_dec_params = sum(p.numel() for p in self.fc_decoder.parameters() if p.requires_grad)
        wavelet_params = sum(p.numel() for p in self.wavelet_transform.parameters() if p.requires_grad)

        total = encoder_params + decoder_params + fc_enc_params + fc_dec_params + wavelet_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params + fc_enc_params,
            'decoder': decoder_params + fc_dec_params,
            'wavelet_transform': wavelet_params,
            'total': total,
            'trainable': trainable
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_count = self.get_parameter_count()

        # 构建FC层结构描述
        fc_structure = [self.flatten_dim] + self.intermediate_dims + [self.latent_dim]
        fc_structure_str = ' → '.join(map(str, fc_structure))

        return {
            'type': 'DifferentiableWaveletAutoEncoder',
            'architecture': 'CNN',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'wavelet_type': self.wavelet_type,
            'differentiable': True,
            'input_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'output_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'loss_space': 'RCS',  # 关键：损失在RCS空间
            'total_params': param_count['total'],
            'trainable_params': param_count['trainable'],
            'fc_structure': fc_structure_str,
            'intermediate_dims': self.intermediate_dims,
            **self.structure_info  # 包含compression_ratios等详细信息
        }


class DifferentiableWaveletMLPAutoEncoder(nn.Module):
    """
    可微分小波MLP AutoEncoder

    使用全连接网络处理小波系数
    适合参数敏感性分析
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 wavelet_type: str = 'db4',
                 input_size: int = 49):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size

        # 可微分小波变换
        self.wavelet_transform = DifferentiableWaveletTransform(
            wavelet=wavelet_type,
            mode='symmetric',
            level=1
        )

        # 计算输入维度
        self.input_dim = input_size * input_size * num_frequencies * wavelet_bands

        # MLP Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, latent_dim)
        )

        # MLP Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(4096, self.input_dim)
        )

    def encode(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """RCS → 小波系数 → 隐空间"""
        wavelet_coeffs = self.wavelet_transform.forward_transform(rcs_data)
        latent = self.encoder(wavelet_coeffs)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """隐空间 → 小波系数 → RCS"""
        wavelet_coeffs = self.decoder(latent)
        # Reshape
        batch_size = latent.size(0)
        wavelet_coeffs = wavelet_coeffs.view(batch_size, self.input_size, self.input_size, -1)
        rcs_data = self.wavelet_transform.inverse_transform(wavelet_coeffs)
        return rcs_data

    def forward(self, rcs_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(rcs_data)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        wavelet_params = sum(p.numel() for p in self.wavelet_transform.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'wavelet_transform': wavelet_params,
            'total': encoder_params + decoder_params + wavelet_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'type': 'DifferentiableWaveletMLPAutoEncoder',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'total_params': total_params,
            'wavelet_type': self.wavelet_type,
            'differentiable': True,
            'input_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'output_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'loss_space': 'RCS'
        }


class DifferentiableSineWaveletMLPAutoEncoder(DifferentiableWaveletMLPAutoEncoder):
    """
    可微分Sine激活MLP AutoEncoder

    使用sin激活函数代替ReLU
    可能更适合周期性信号
    """

    def __init__(self, *args, **kwargs):
        # 先不调用父类初始化，我们要自定义
        nn.Module.__init__(self)

        latent_dim = kwargs.get('latent_dim', 256)
        num_frequencies = kwargs.get('num_frequencies', 2)
        wavelet_bands = kwargs.get('wavelet_bands', 4)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        wavelet_type = kwargs.get('wavelet_type', 'db4')
        input_size = kwargs.get('input_size', 49)

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.input_size = input_size

        # 可微分小波变换
        self.wavelet_transform = DifferentiableWaveletTransform(
            wavelet=wavelet_type,
            mode='symmetric',
            level=1
        )

        # 计算输入维度
        self.input_dim = input_size * input_size * num_frequencies * wavelet_bands

        # MLP Encoder (使用Sin激活)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 4096),
            nn.BatchNorm1d(4096),
            SinActivation(),
            nn.Dropout(dropout_rate),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            SinActivation(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            SinActivation(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, latent_dim)
        )

        # MLP Decoder (使用Sin激活)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            SinActivation(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            SinActivation(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            SinActivation(),
            nn.Dropout(dropout_rate),

            nn.Linear(4096, self.input_dim)
        )

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息（重写以返回正确类型名）"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'type': 'DifferentiableSineWaveletMLPAutoEncoder',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'total_params': total_params,
            'wavelet_type': self.wavelet_type,
            'differentiable': True,
            'activation': 'sin',
            'input_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'output_shape': f'[B, 91, 91, {self.num_frequencies}]',
            'loss_space': 'RCS'
        }


class SinActivation(nn.Module):
    """Sin激活函数"""
    def forward(self, x):
        return torch.sin(x)


# 导出
__all__ = [
    'DifferentiableWaveletAutoEncoder',
    'DifferentiableWaveletMLPAutoEncoder',
    'DifferentiableSineWaveletMLPAutoEncoder'
]
