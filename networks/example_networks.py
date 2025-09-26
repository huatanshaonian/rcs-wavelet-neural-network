"""
网络架构示例
展示如何适配现有网络到插件化框架，以及如何创建新网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from network_registry import BaseNetwork, BaseLoss, NetworkConfig, LossConfig, register_network, register_loss

# ============================================================================
# 1. 适配现有的小波网络到新框架
# ============================================================================

@register_network
class WaveletRCSNetwork(BaseNetwork):
    """小波RCS网络 - 适配现有架构"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        # 参数编码器
        self.param_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 多尺度特征提取 (简化版)
        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 23*23*64),
            nn.Unflatten(1, (64, 23, 23)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 双频解码器
        self.decoder_1_5g = self._build_decoder()
        self.decoder_3g = self._build_decoder()

        # 输出激活
        if self.use_log_output:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Softplus()

    def _build_decoder(self):
        """构建解码器"""
        return nn.Sequential(
            # 23x23 -> 46x46
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 46x46 -> 91x91
            nn.Upsample((91, 91), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 参数编码
        encoded = self.param_encoder(x)  # [B, 256]

        # 特征提取
        features = self.feature_extractor(encoded)  # [B, 64, 23, 23]

        # 双频解码
        rcs_1_5g = self.decoder_1_5g(features)  # [B, 1, 91, 91]
        rcs_3g = self.decoder_3g(features)      # [B, 1, 91, 91]

        # 拼接频率维度
        output = torch.cat([rcs_1_5g, rcs_3g], dim=1)  # [B, 2, 91, 91]
        output = output.permute(0, 2, 3, 1)  # [B, 91, 91, 2]

        # 输出激活
        output = self.output_activation(output)

        return output

    @classmethod
    def get_name(cls) -> str:
        return "wavelet_rcs"

    @classmethod
    def get_description(cls) -> str:
        return "Multi-scale wavelet network for RCS prediction"

# ============================================================================
# 2. 创建全新的网络架构示例
# ============================================================================

@register_network
class SimpleFCNetwork(BaseNetwork):
    """简单全连接网络 - 作为基线对比"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        output_size = 1
        for dim in self.output_shape:
            output_size *= dim

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_size)
        )

        if self.use_log_output:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        output = self.network(x)
        output = output.view(batch_size, *self.output_shape)
        return self.output_activation(output)

    @classmethod
    def get_name(cls) -> str:
        return "simple_fc"

    @classmethod
    def get_description(cls) -> str:
        return "Simple fully connected baseline network"

@register_network
class ResNetRCS(BaseNetwork):
    """ResNet风格的RCS预测网络"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        # 参数编码器
        self.param_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        # ResNet块
        self.resnet_blocks = nn.ModuleList([
            self._make_resnet_block(512, 512) for _ in range(4)
        ])

        # 空间解码器
        self.spatial_decoder = nn.Sequential(
            nn.Linear(512, 32*32*64),
            nn.Unflatten(1, (64, 32, 32)),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample((91, 91), mode='bilinear'),
            nn.Conv2d(32, self.output_shape[-1], 1)
        )

        if self.use_log_output:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Softplus()

    def _make_resnet_block(self, in_features, out_features):
        """创建ResNet块"""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 参数编码
        features = self.param_encoder(x)

        # ResNet块 (带残差连接)
        for block in self.resnet_blocks:
            residual = features
            features = block(features) + residual

        # 空间解码
        output = self.spatial_decoder(features)  # [B, 2, 91, 91]
        output = output.permute(0, 2, 3, 1)      # [B, 91, 91, 2]

        return self.output_activation(output)

    @classmethod
    def get_name(cls) -> str:
        return "resnet_rcs"

    @classmethod
    def get_description(cls) -> str:
        return "ResNet-style network with residual connections for RCS prediction"

    @classmethod
    def validate_config(cls, config: NetworkConfig) -> bool:
        # ResNet需要更多的参数，所以需要更大的input_dim
        return config.input_dim >= 3

# ============================================================================
# 3. 新的损失函数示例
# ============================================================================

@register_loss
class SimpleMSELoss(BaseLoss):
    """简单MSE损失"""

    def __init__(self, config: LossConfig):
        super().__init__(config)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse_loss = self.mse(pred, target)
        return {
            'total': mse_loss,
            'mse': mse_loss
        }

    @classmethod
    def get_name(cls) -> str:
        return "simple_mse"

@register_loss
class RobustLoss(BaseLoss):
    """鲁棒损失函数组合"""

    def __init__(self, config: LossConfig):
        super().__init__(config)
        self.huber = nn.HuberLoss(delta=0.1)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        huber_loss = self.huber(pred, target)
        mse_loss = self.mse(pred, target)

        # 加权组合
        total_loss = (
            self.loss_weights.get('huber', 0.7) * huber_loss +
            self.loss_weights.get('mse', 0.3) * mse_loss
        )

        return {
            'total': total_loss,
            'huber': huber_loss,
            'mse': mse_loss
        }

    @classmethod
    def get_name(cls) -> str:
        return "robust_loss"

    @classmethod
    def get_default_config(cls) -> LossConfig:
        return LossConfig(
            loss_type='robust_loss',
            loss_weights={
                'huber': 0.7,
                'mse': 0.3
            }
        )

# ============================================================================
# 4. 自定义输出形状的网络示例
# ============================================================================

@register_network
class FlexibleOutputNetwork(BaseNetwork):
    """支持任意输出形状的灵活网络"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        # 计算输出总大小
        self.output_size = 1
        for dim in self.output_shape:
            self.output_size *= dim

        # 动态构建网络
        hidden_sizes = [256, 512, 1024]
        layers = []

        input_size = self.input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size

        # 输出层
        layers.append(nn.Linear(input_size, self.output_size))

        self.network = nn.Sequential(*layers)

        if self.use_log_output:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        output = self.network(x)
        output = output.view(batch_size, *self.output_shape)
        return self.output_activation(output)

    @classmethod
    def get_name(cls) -> str:
        return "flexible_output"

    @classmethod
    def get_description(cls) -> str:
        return "Flexible network supporting arbitrary output shapes"

    @classmethod
    def validate_config(cls, config: NetworkConfig) -> bool:
        # 检查输出形状是否合理
        output_size = 1
        for dim in config.output_shape:
            output_size *= dim
        return output_size <= 1000000  # 避免过大的输出

if __name__ == "__main__":
    # 演示如何使用
    from network_registry import create_network, create_loss, list_available_networks, list_available_losses

    print("=== 演示插件化网络架构 ===")

    # 列出所有可用网络
    list_available_networks()
    print()

    # 列出所有可用损失函数
    list_available_losses()
    print()

    # 创建不同的网络
    networks_to_test = ['wavelet_rcs', 'simple_fc', 'resnet_rcs', 'flexible_output']

    for net_name in networks_to_test:
        try:
            print(f"测试网络: {net_name}")
            network = create_network(net_name, input_dim=9, output_shape=(91, 91, 2))
            print(f"  参数统计: {network.get_parameter_count()}")

            # 测试前向传播
            x = torch.randn(2, 9)
            with torch.no_grad():
                output = network(x)
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            print()

        except Exception as e:
            print(f"  错误: {e}")
            print()

    # 测试损失函数
    print("=== 测试损失函数 ===")
    loss_functions = ['simple_mse', 'robust_loss']

    for loss_name in loss_functions:
        try:
            print(f"测试损失函数: {loss_name}")
            loss_fn = create_loss(loss_name)

            pred = torch.randn(2, 91, 91, 2)
            target = torch.randn(2, 91, 91, 2)

            loss_dict = loss_fn(pred, target)
            print(f"  损失组件: {list(loss_dict.keys())}")
            print(f"  总损失: {loss_dict['total'].item():.4f}")
            print()

        except Exception as e:
            print(f"  错误: {e}")
            print()