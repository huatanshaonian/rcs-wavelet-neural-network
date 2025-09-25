"""
增强版RCS神经网络架构
解决当前loss=0.7收敛问题的根本方案

主要改进：
1. 增强编码器：9→1024维，添加几何分支
2. 多尺度特征提取：三尺度并行+融合网络
3. 渐进式解码器：逐步上采样减少跨度
4. 改进损失函数：HuberLoss+物理约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


class EnhancedParameterEncoder(nn.Module):
    """增强的参数编码器 - 解决特征瓶颈问题"""

    def __init__(self, input_dim: int = 9, output_dim: int = 1024):
        super().__init__()

        # 主分支：渐进式特征扩展
        self.main_branch = nn.Sequential(
            # Stage 1: 基础特征提取
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),

            # Stage 2: 特征扩展
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            # Stage 3: 高维特征学习
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 几何特征分支：专门处理几何参数
        self.geometry_branch = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 512)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, x):
        main_feat = self.main_branch(x)
        geom_feat = self.geometry_branch(x)

        # 特征拼接和融合
        combined = torch.cat([main_feat, geom_feat], dim=1)
        return self.fusion(combined)


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器 - 替代单纯的小波变换"""

    def __init__(self, input_dim: int = 1024):
        super().__init__()

        # 将1D特征映射到多个2D特征图
        self.to_feature_maps = nn.ModuleDict({
            'high_res': nn.Sequential(
                nn.Linear(input_dim, 64*64*8),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            'mid_res': nn.Sequential(
                nn.Linear(input_dim, 32*32*16),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            'low_res': nn.Sequential(
                nn.Linear(input_dim, 16*16*32),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        })

        # 多尺度卷积块
        self.conv_blocks = nn.ModuleDict({
            'high_res': self._make_conv_block(8, 32, 32),
            'mid_res': self._make_conv_block(16, 32, 32),
            'low_res': self._make_conv_block(32, 32, 16)
        })

        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Conv2d(80, 64, 3, padding=1),  # 32+32+16 channels
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

    def _make_conv_block(self, in_channels: int, mid_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # 生成多尺度特征图
        high_feat = self.to_feature_maps['high_res'](x).view(-1, 8, 64, 64)
        mid_feat = self.to_feature_maps['mid_res'](x).view(-1, 16, 32, 32)
        low_feat = self.to_feature_maps['low_res'](x).view(-1, 32, 16, 16)

        # 多尺度卷积处理
        high_conv = self.conv_blocks['high_res'](high_feat)  # [B, 32, 64, 64]
        mid_conv = self.conv_blocks['mid_res'](mid_feat)     # [B, 32, 32, 32]
        low_conv = self.conv_blocks['low_res'](low_feat)    # [B, 16, 16, 16]

        # 上采样到统一尺寸 (64x64)
        mid_up = F.interpolate(mid_conv, size=64, mode='bilinear', align_corners=False)
        low_up = F.interpolate(low_conv, size=64, mode='bilinear', align_corners=False)

        # 特征拼接和融合
        combined = torch.cat([high_conv, mid_up, low_up], dim=1)
        fused = self.fusion_network(combined)

        return fused


class ProgressiveDecoder(nn.Module):
    """渐进式解码器 - 解决上采样跨度过大问题"""

    def __init__(self, input_channels: int = 64, use_log_output: bool = False):
        super().__init__()

        # Stage 1: 64x64 → 72x72
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU()
        )

        # Stage 2: 72x72 → 84x84
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(48, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # Stage 3: 84x84 → 91x91 (最终尺寸)
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        # 最终输出层
        self.final_conv = nn.Conv2d(16, 1, 1)

        # 根据输出类型选择激活函数
        if use_log_output:
            # 对数域输出，不需要激活函数
            self.output_activation = nn.Identity()
        else:
            # 线性域输出，使用Softplus确保正值
            self.output_activation = nn.Softplus()

    def forward(self, x):
        # 渐进式上采样，每次增加约8-12像素
        x = self.stage1(x)  # 64x64
        x = F.interpolate(x, size=72, mode='bilinear', align_corners=False)

        x = self.stage2(x)  # 72x72
        x = F.interpolate(x, size=84, mode='bilinear', align_corners=False)

        x = self.stage3(x)  # 84x84
        x = F.interpolate(x, size=91, mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        return self.output_activation(x)


class EnhancedFrequencyInteraction(nn.Module):
    """增强的频率交互模块"""

    def __init__(self, feature_dim: int = 64):
        super().__init__()

        # 交互特征提取
        self.interaction = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.GELU(),
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 交互特征
        enhanced = self.interaction(x)

        # 注意力加权
        attention_weights = self.attention(enhanced)
        attended = enhanced * attention_weights

        # 残差连接
        return x + attended


class EnhancedTriDimensionalRCSNet(nn.Module):
    """增强版三维RCS神经网络"""

    def __init__(self, input_dim: int = 9, use_log_output: bool = False):
        super().__init__()

        # 增强的参数编码器
        self.encoder = EnhancedParameterEncoder(input_dim, 1024)

        # 多尺度特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(1024)

        # 增强的频率交互模块
        self.freq_interaction = EnhancedFrequencyInteraction(64)

        # 双频分支解码器
        self.freq1_decoder = ProgressiveDecoder(64, use_log_output)  # 1.5GHz
        self.freq2_decoder = ProgressiveDecoder(64, use_log_output)  # 3GHz

        # 物理约束层
        self.use_log_output = use_log_output

    def forward(self, x):
        # 参数编码
        encoded = self.encoder(x)  # [B, 1024]

        # 多尺度特征提取
        features = self.feature_extractor(encoded)  # [B, 64, 64, 64]

        # 频率交互
        enhanced_features = self.freq_interaction(features)  # [B, 64, 64, 64]

        # 双频分支解码
        rcs_1_5g = self.freq1_decoder(enhanced_features)  # [B, 1, 91, 91]
        rcs_3g = self.freq2_decoder(enhanced_features)    # [B, 1, 91, 91]

        # 合并输出
        output = torch.cat([rcs_1_5g, rcs_3g], dim=1)  # [B, 2, 91, 91]
        return output.permute(0, 2, 3, 1)  # [B, 91, 91, 2]


class ImprovedRCSLoss(nn.Module):
    """改进的损失函数 - 解决收敛问题"""

    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()

        self.loss_weights = loss_weights or {
            'main': 0.7,           # Huber Loss主损失
            'aux': 0.25,           # MSE辅助损失
            'freq_consistency': 0.02,  # 频率一致性 (降低权重)
            'continuity': 0.02,    # 空间连续性 (降低权重)
            'symmetry': 0.01       # 对称性约束 (降低权重)
        }

        self.huber_loss = nn.HuberLoss(delta=0.1)  # 对异常值更鲁棒
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 主要损失：Huber Loss（更鲁棒）
        main_loss = self.huber_loss(pred, target)

        # 辅助损失：MSE Loss（精细调优）
        aux_loss = self.mse_loss(pred, target)

        # 频率一致性损失
        freq_consistency = self._frequency_consistency_loss(pred, target)

        # 空间连续性损失
        continuity = self._continuity_loss(pred, target)

        # 对称性损失（改进版）
        symmetry = self._symmetry_loss(pred, target)

        # 计算加权总损失
        total_loss = (
            self.loss_weights['main'] * main_loss +
            self.loss_weights['aux'] * aux_loss +
            self.loss_weights['freq_consistency'] * freq_consistency +
            self.loss_weights['continuity'] * continuity +
            self.loss_weights['symmetry'] * symmetry
        )

        return {
            'total': total_loss,
            'main': main_loss,
            'aux': aux_loss,
            'freq_consistency': freq_consistency,
            'continuity': continuity,
            'symmetry': symmetry
        }

    def _frequency_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """频率间的一致性约束 - 简化版本避免数值问题"""
        pred_1_5g = pred[:, :, :, 0]
        pred_3g = pred[:, :, :, 1]
        target_1_5g = target[:, :, :, 0]
        target_3g = target[:, :, :, 1]

        # 简单的差值一致性而非比值
        pred_diff = pred_3g - pred_1_5g
        target_diff = target_3g - target_1_5g

        return F.mse_loss(pred_diff, target_diff)

    def _continuity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """空间连续性约束"""
        # 计算梯度差异
        pred_grad_x = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_grad_x = target[:, 1:, :, :] - target[:, :-1, :, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)

        return grad_loss_x + grad_loss_y

    def _symmetry_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """改进的对称性损失"""
        # φ=0°在第2维的索引45
        center_phi = 45
        symmetry_loss = 0.0
        count = 0

        for i in range(1, min(center_phi + 1, 20)):  # 限制对称性检查范围
            left_idx = center_phi - i
            right_idx = center_phi + i

            if right_idx < 91:
                # 计算预测和目标的对称性差异
                pred_sym_diff = pred[:, :, left_idx, :] - pred[:, :, right_idx, :]
                target_sym_diff = target[:, :, left_idx, :] - target[:, :, right_idx, :]

                # 确保预测的对称性与目标一致
                symmetry_loss += F.mse_loss(pred_sym_diff, target_sym_diff)
                count += 1

        return symmetry_loss / max(count, 1)


# 创建增强模型的工厂函数
def create_enhanced_model(input_dim: int = 9, use_log_output: bool = False) -> tuple:
    """
    创建增强版模型和损失函数

    返回:
        model: 增强版RCS网络
        loss_fn: 改进的损失函数
    """
    model = EnhancedTriDimensionalRCSNet(input_dim, use_log_output)
    loss_fn = ImprovedRCSLoss()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"增强版网络参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  相比原网络增加约: {(total_params / 100000):.1f}x")

    return model, loss_fn


# 测试代码
if __name__ == "__main__":
    # 测试网络
    model, loss_fn = create_enhanced_model()

    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 9)
    target = torch.randn(batch_size, 91, 91, 2)

    # 前向传播
    print("\n前向传播测试:")
    with torch.no_grad():
        output = model(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")

        # 损失计算
        loss_dict = loss_fn(output, target)
        print(f"\n损失函数测试:")
        for loss_name, loss_value in loss_dict.items():
            print(f"  {loss_name}: {loss_value:.4f}")