"""
CNN-AutoEncoder核心模型
方案A: 小波预处理 + 单AutoEncoder
输入: [B, 49, 49, 8] 小波系数 (2频率 × 4频带，db4小波变换后尺寸)
输出: [B, 49, 49, 8] 重建小波系数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class WaveletAutoEncoder(nn.Module):
    """
    小波增强的CNN-AutoEncoder
    使用张量输入，保持空间结构
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,
                 wavelet_bands: int = 4,
                 dropout_rate: float = 0.2,
                 input_size: int = 49):
        """
        初始化AutoEncoder

        Args:
            latent_dim: 隐空间维度
            num_frequencies: 频率数量 (2 for 1.5GHz+3GHz, 3 for +6GHz)
            wavelet_bands: 小波频带数 (通常为4: LL,LH,HL,HH)
            dropout_rate: Dropout比率
            input_size: 输入小波系数尺寸 (49 for db4小波变换后的尺寸)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.wavelet_bands = wavelet_bands
        self.input_channels = num_frequencies * wavelet_bands
        self.dropout_rate = dropout_rate
        self.input_size = input_size

        # ===== Encoder: 小波系数 → 隐空间 =====
        self.encoder = nn.Sequential(
            # 第一层: 动态小波通道输入 [B, num_freq*4, input_size, input_size]
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 下采样层1: [49, 49] → [25, 25]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 下采样层2: [25, 25] → [13, 13]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 下采样层3: [13, 13] → [7, 7]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 全局平均池化: [7, 7] → [4, 4]
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 计算展平后的特征维度
        self.flattened_size = 256 * 4 * 4  # 4096

        # 编码器的最后几层
        # TODO: latent_dim优化点 - 待实验验证
        # 当前架构: 4096 → 1024 → latent_dim (单步压缩)
        # - latent_dim=256: 1024→256 (4:1) ✅ 温和，无问题
        # - latent_dim=128: 1024→128 (8:1) ⚠️ 中等，需验证Stage 1重建误差
        # - latent_dim=64:  1024→64 (16:1) ❌ 激进，建议增加512过渡层
        # 如果Stage 1 val_loss > 0.05，考虑多级压缩: 4096→1024→512→256→latent_dim
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, latent_dim)  # 压缩瓶颈
        )

        # ===== Decoder: 隐空间 → 小波系数 =====
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, self.flattened_size),
            nn.ReLU(inplace=True)
        )

        # 解码器 - 不包含最终的Upsample，在decode()中动态处理
        self.decoder_conv = nn.Sequential(
            # 重塑为特征图: [4096] → [256, 4, 4]
            nn.Unflatten(1, (256, 4, 4)),

            # 上采样层1: [4, 4] → [8, 8]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 上采样层2: [8, 8] → [16, 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 上采样层3: [16, 16] → [32, 32]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 最终卷积层
        self.final_conv = nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1)

        # 权重初始化
        self._initialize_weights()

        # ===== 统一接口：encoder/decoder属性 =====
        # 注意：encoder已经是nn.Sequential，这里只需要添加decoder
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
        # 调整维度: [B, input_size, input_size, num_freq*4] → [B, num_freq*4, input_size, input_size]
        x = x.permute(0, 3, 1, 2)

        # 卷积特征提取
        features = self.encoder(x)  # [B, 256, 4, 4]

        # 全连接编码
        latent = self.encoder_fc(features)  # [B, latent_dim]

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码器: 隐空间 → 小波系数

        Args:
            latent: [B, latent_dim] 隐空间表示

        Returns:
            x_recon: [B, input_size, input_size, num_freq*4] 重建小波系数
        """
        # 全连接解码
        features = self.decoder_fc(latent)  # [B, 4096]

        # 卷积重建
        x_recon = self.decoder_conv(features)  # [B, 32, 32, 32]

        # 上采样到目标尺寸
        x_recon = F.interpolate(x_recon, size=(self.input_size, self.input_size),
                               mode='bilinear', align_corners=False)

        # 最终卷积
        x_recon = self.final_conv(x_recon)  # [B, num_freq*4, input_size, input_size]

        # 调整维度: [B, num_freq*4, input_size, input_size] → [B, input_size, input_size, num_freq*4]
        x_recon = x_recon.permute(0, 2, 3, 1)

        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, input_size, input_size, num_freq*4] 小波系数

        Returns:
            x_recon: [B, input_size, input_size, num_freq*4] 重建小波系数
            latent: [B, latent_dim] 隐空间表示
        """
        latent = self.encode(x)
        x_recon = self.decode(latent)

        return x_recon, latent

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters()) + \
                        sum(p.numel() for p in self.encoder_fc.parameters())

        decoder_params = sum(p.numel() for p in self.decoder_fc.parameters()) + \
                        sum(p.numel() for p in self.decoder_conv.parameters()) + \
                        sum(p.numel() for p in self.final_conv.parameters())

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

        input_size = self.input_size * self.input_size * self.input_channels
        return {
            'model_name': 'WaveletAutoEncoder',
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'wavelet_bands': self.wavelet_bands,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'input_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'output_shape': f'[B, {self.input_size}, {self.input_size}, {self.input_channels}]',
            'latent_shape': f'[B, {self.latent_dim}]',
            'parameters': param_count,
            'dropout_rate': self.dropout_rate,
            'compression_ratio': f'{input_size}:{self.latent_dim} = {(input_size/self.latent_dim):.1f}:1'
        }


class ParameterMapper(nn.Module):
    """
    参数映射网络: 设计参数 → 隐空间
    """

    def __init__(self,
                 param_dim: int = 9,
                 latent_dim: int = 256,
                 hidden_dims: list = [64, 128],
                 dropout_rate: float = 0.3):
        """
        初始化参数映射器

        Args:
            param_dim: 设计参数维度
            latent_dim: 隐空间维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
        """
        super().__init__()

        self.param_dim = param_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # 构建网络层
        layers = []
        input_dim = param_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, latent_dim))

        self.network = nn.Sequential(*layers)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            params: [B, param_dim] 设计参数

        Returns:
            latent: [B, latent_dim] 隐空间表示
        """
        return self.network(params)

    def get_parameter_count(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())


def test_models():
    """测试模型功能"""
    print("=== 模型测试 ===")

    # 测试2频率配置 (当前默认)
    print("=== 测试2频率配置 (1.5GHz + 3GHz) ===")
    batch_size = 4
    wavelet_data_2freq = torch.randn(batch_size, 91, 91, 8)  # 2频率 × 4频带
    params_data = torch.randn(batch_size, 9)

    print(f"2频率小波数据形状: {wavelet_data_2freq.shape}")
    print(f"参数数据形状: {params_data.shape}")

    ae_2freq = WaveletAutoEncoder(latent_dim=256, num_frequencies=2)

    # 前向传播
    recon_2freq, latent_2freq = ae_2freq(wavelet_data_2freq)

    print(f"2频率重建形状: {recon_2freq.shape}")
    print(f"2频率隐空间形状: {latent_2freq.shape}")

    # 计算重建误差
    recon_error_2freq = F.mse_loss(recon_2freq, wavelet_data_2freq)
    print(f"2频率重建MSE误差: {recon_error_2freq:.6f}")

    # 模型信息
    model_info_2freq = ae_2freq.get_model_info()
    print(f"2频率模型参数量: {model_info_2freq['parameters']['total']:,}")
    print(f"2频率压缩比: {model_info_2freq['compression_ratio']}")

    # 测试3频率配置 (6GHz扩展)
    print("\n=== 测试3频率配置 (1.5GHz + 3GHz + 6GHz) ===")
    wavelet_data_3freq = torch.randn(batch_size, 91, 91, 12)  # 3频率 × 4频带
    print(f"3频率小波数据形状: {wavelet_data_3freq.shape}")

    ae_3freq = WaveletAutoEncoder(latent_dim=256, num_frequencies=3)
    recon_3freq, latent_3freq = ae_3freq(wavelet_data_3freq)

    print(f"3频率重建形状: {recon_3freq.shape}")
    print(f"3频率隐空间形状: {latent_3freq.shape}")

    recon_error_3freq = F.mse_loss(recon_3freq, wavelet_data_3freq)
    print(f"3频率重建MSE误差: {recon_error_3freq:.6f}")

    model_info_3freq = ae_3freq.get_model_info()
    print(f"3频率模型参数量: {model_info_3freq['parameters']['total']:,}")
    print(f"3频率压缩比: {model_info_3freq['compression_ratio']}")

    # 测试参数映射器
    print("\n--- 测试ParameterMapper ---")
    mapper = ParameterMapper(param_dim=9, latent_dim=256)
    mapped_latent = mapper(params_data)

    print(f"映射隐空间形状: {mapped_latent.shape}")
    print(f"映射器参数量: {mapper.get_parameter_count():,}")

    # 测试端到端流程 (2频率)
    print("\n--- 测试2频率端到端流程 ---")
    pred_latent_2freq = mapper(params_data)
    pred_wavelet_2freq = ae_2freq.decode(pred_latent_2freq)
    print(f"2频率端到端重建形状: {pred_wavelet_2freq.shape}")

    # 测试端到端流程 (3频率)
    print("\n--- 测试3频率端到端流程 ---")
    pred_latent_3freq = mapper(params_data)
    pred_wavelet_3freq = ae_3freq.decode(pred_latent_3freq)
    print(f"3频率端到端重建形状: {pred_wavelet_3freq.shape}")

    # 模型兼容性验证
    print("\n--- 模型兼容性验证 ---")
    print(f"隐空间维度一致性: {latent_2freq.shape[1] == latent_3freq.shape[1]}")
    print("✅ 不同频率配置使用相同的隐空间维度，参数映射器可以复用")

    return True


if __name__ == "__main__":
    test_models()