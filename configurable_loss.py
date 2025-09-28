"""
可配置损失函数模块
支持动态组合不同损失函数组件，包含高级优化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np


class ConfigurableLoss(nn.Module):
    """
    可配置的损失函数类
    支持动态启用/禁用不同损失组件，并集成高级优化策略
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Huber Loss (如果启用)
        if config.get('use_huber', False):
            self.huber_loss = nn.HuberLoss(delta=config.get('huber_delta', 0.1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            pred: 预测RCS [B, 91, 91, 2]
            target: 真实RCS [B, 91, 91, 2]

        Returns:
            损失字典
        """
        losses = {}

        # 基础损失函数
        if self.config.get('use_mse', False):
            losses['mse'] = self.mse_loss(pred, target)

        if self.config.get('use_huber', False):
            losses['huber'] = self.huber_loss(pred, target)

        if self.config.get('use_l1', False):
            losses['l1'] = self.l1_loss(pred, target)

        # 物理约束损失
        if self.config.get('use_symmetry', False):
            losses['symmetry'] = self._symmetry_loss(pred, target)

        if self.config.get('use_freq_consistency', False):
            freq_type = self.config.get('freq_consistency_type', 'diff')
            losses['freq_consistency'] = self._frequency_consistency_loss(pred, target, freq_type)

        if self.config.get('use_continuity', False):
            continuity_type = self.config.get('continuity_type', 'standard')
            losses['continuity'] = self._continuity_loss(pred, target, continuity_type)

        if self.config.get('use_multiscale', False):
            losses['multiscale'] = self._multiscale_loss(pred, target)

        # 计算加权总损失
        total_loss = 0
        for loss_name, loss_value in losses.items():
            weight_key = f'{loss_name}_weight'
            weight = self.config.get(weight_key, 0)
            total_loss += weight * loss_value

        losses['total'] = total_loss
        return losses

    def _symmetry_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        对称性损失 (φ=0°平面对称性)
        """
        center_phi = 45
        symmetry_loss = 0.0
        count = 0

        for i in range(1, center_phi + 1):
            left_idx = center_phi - i
            right_idx = center_phi + i

            if right_idx < 91:
                # 预测的对称性误差
                pred_left = pred[:, :, left_idx, :]
                pred_right = pred[:, :, right_idx, :]
                pred_sym_diff = pred_left - pred_right

                # 目标的对称性误差
                target_left = target[:, :, left_idx, :]
                target_right = target[:, :, right_idx, :]
                target_sym_diff = target_left - target_right

                # 计算对称性损失
                symmetry_loss += F.mse_loss(pred_sym_diff, target_sym_diff)
                count += 1

        return symmetry_loss / count if count > 0 else torch.tensor(0.0, device=pred.device)

    def _frequency_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor, freq_type: str) -> torch.Tensor:
        """
        频率一致性损失
        """
        pred_1_5g = pred[:, :, :, 0]
        pred_3g = pred[:, :, :, 1]
        target_1_5g = target[:, :, :, 0]
        target_3g = target[:, :, :, 1]

        if freq_type == 'diff':
            # 标准差值一致性
            pred_diff = pred_3g - pred_1_5g
            target_diff = target_3g - target_1_5g
            return F.mse_loss(pred_diff, target_diff)

        elif freq_type == 'correlation':
            # 基于相关性的频率约束
            pred_corr = F.cosine_similarity(
                pred_1_5g.flatten(1), pred_3g.flatten(1), dim=1
            )
            target_corr = F.cosine_similarity(
                target_1_5g.flatten(1), target_3g.flatten(1), dim=1
            )
            return F.mse_loss(pred_corr, target_corr)

        elif freq_type == 'local':
            # 局部窗口频率约束
            return self._local_frequency_consistency(pred, target, window_size=5)

        else:
            return torch.tensor(0.0, device=pred.device)

    def _local_frequency_consistency(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """局部窗口频率一致性约束"""
        # 创建平均池化核
        kernel = torch.ones(1, 1, window_size, window_size) / (window_size**2)
        kernel = kernel.to(pred.device)

        # 对每个频率进行局部平滑
        pred_1_5g_smooth = F.conv2d(
            pred[:, :, :, 0:1].permute(0,3,1,2),
            kernel,
            padding=window_size//2
        )
        pred_3g_smooth = F.conv2d(
            pred[:, :, :, 1:2].permute(0,3,1,2),
            kernel,
            padding=window_size//2
        )

        target_1_5g_smooth = F.conv2d(
            target[:, :, :, 0:1].permute(0,3,1,2),
            kernel,
            padding=window_size//2
        )
        target_3g_smooth = F.conv2d(
            target[:, :, :, 1:2].permute(0,3,1,2),
            kernel,
            padding=window_size//2
        )

        # 在平滑后的数据上进行频率约束
        pred_diff_smooth = pred_3g_smooth - pred_1_5g_smooth
        target_diff_smooth = target_3g_smooth - target_1_5g_smooth

        return F.mse_loss(pred_diff_smooth, target_diff_smooth)

    def _continuity_loss(self, pred: torch.Tensor, target: torch.Tensor, continuity_type: str) -> torch.Tensor:
        """
        空间连续性损失
        """
        if continuity_type == 'standard':
            return self._standard_continuity_loss(pred, target)
        elif continuity_type == 'adaptive':
            return self._adaptive_continuity_loss(pred, target)
        elif continuity_type == 'regional':
            return self._regional_continuity_loss(pred, target)
        else:
            return torch.tensor(0.0, device=pred.device)

    def _standard_continuity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """标准连续性损失"""
        # θ方向梯度
        pred_grad_x = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_grad_x = target[:, 1:, :, :] - target[:, :-1, :, :]

        # φ方向梯度
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)

        return grad_loss_x + grad_loss_y

    def _adaptive_continuity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """自适应连续性损失 - 基于目标梯度强度调整权重"""
        # 计算目标的梯度强度
        target_grad_x = target[:, 1:, :, :] - target[:, :-1, :, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        target_grad_magnitude_x = torch.sqrt(target_grad_x**2 + 1e-8)
        target_grad_magnitude_y = torch.sqrt(target_grad_y**2 + 1e-8)

        # 自适应权重：梯度大的地方约束较弱
        adaptive_weight_x = torch.sigmoid(-5 * target_grad_magnitude_x + 2.5)
        adaptive_weight_y = torch.sigmoid(-5 * target_grad_magnitude_y + 2.5)

        # 预测梯度
        pred_grad_x = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # 加权连续性损失
        loss_x = torch.mean(adaptive_weight_x * (pred_grad_x - target_grad_x)**2)
        loss_y = torch.mean(adaptive_weight_y * (pred_grad_y - target_grad_y)**2)

        return loss_x + loss_y

    def _regional_continuity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """分区域连续性损失 - 基于RCS强度分区域约束"""
        # 识别高RCS区域（可能有高频特征）
        high_rcs_mask = target > torch.quantile(target, 0.8)  # 前20%高值区域

        # 低RCS区域：强连续性约束；高RCS区域：弱连续性约束
        mask_weight = torch.where(high_rcs_mask, 0.1, 1.0)  # 高RCS区域权重降低10倍

        pred_grad_x = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_grad_x = target[:, 1:, :, :] - target[:, :-1, :, :]

        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        weighted_loss_x = torch.mean(mask_weight[:, 1:, :, :] * (pred_grad_x - target_grad_x)**2)
        weighted_loss_y = torch.mean(mask_weight[:, :, 1:, :] * (pred_grad_y - target_grad_y)**2)

        return weighted_loss_x + weighted_loss_y

    def _multiscale_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        多尺度损失
        """
        scales = [1, 2, 4]  # 不同下采样尺度
        total_loss = 0.0

        for scale in scales:
            if scale == 1:
                pred_scale = pred
                target_scale = target
            else:
                # 下采样
                pred_scale = F.avg_pool2d(
                    pred.permute(0, 3, 1, 2),
                    kernel_size=scale, stride=scale
                ).permute(0, 2, 3, 1)

                target_scale = F.avg_pool2d(
                    target.permute(0, 3, 1, 2),
                    kernel_size=scale, stride=scale
                ).permute(0, 2, 3, 1)

            scale_loss = F.mse_loss(pred_scale, target_scale)
            total_loss += scale_loss / scale  # 高分辨率权重更大

        return total_loss / len(scales)


def create_loss_function(config: Dict[str, Any]) -> ConfigurableLoss:
    """
    创建配置化损失函数

    Args:
        config: 损失函数配置字典

    Returns:
        ConfigurableLoss实例
    """
    return ConfigurableLoss(config)


# 预设配置
PRESET_CONFIGS = {
    'original': {
        'use_mse': True, 'mse_weight': 1.0,
        'use_huber': False, 'use_l1': False,
        'use_symmetry': True, 'symmetry_weight': 0.02,
        'use_freq_consistency': False, 'use_continuity': False,
        'use_multiscale': True, 'multiscale_weight': 0.1
    },

    'enhanced': {
        'use_mse': False, 'use_huber': True, 'huber_weight': 0.7, 'huber_delta': 0.1,
        'use_l1': False,
        'use_symmetry': True, 'symmetry_weight': 0.01,
        'use_freq_consistency': True, 'freq_consistency_weight': 0.02, 'freq_consistency_type': 'diff',
        'use_continuity': True, 'continuity_weight': 0.02, 'continuity_type': 'standard',
        'use_multiscale': False
    },

    'robust': {
        'use_mse': False, 'use_huber': True, 'huber_weight': 0.8, 'huber_delta': 0.2,
        'use_l1': True, 'l1_weight': 0.1,
        'use_symmetry': True, 'symmetry_weight': 0.005,
        'use_freq_consistency': True, 'freq_consistency_weight': 0.01, 'freq_consistency_type': 'correlation',
        'use_continuity': False, 'use_multiscale': False
    },

    'high_freq': {
        'use_mse': True, 'mse_weight': 0.9,
        'use_huber': False, 'use_l1': False,
        'use_symmetry': True, 'symmetry_weight': 0.005,
        'use_freq_consistency': True, 'freq_consistency_weight': 0.005, 'freq_consistency_type': 'local',
        'use_continuity': True, 'continuity_weight': 0.005, 'continuity_type': 'adaptive',
        'use_multiscale': False
    },

    'smooth': {
        'use_mse': True, 'mse_weight': 0.6,
        'use_huber': False, 'use_l1': False,
        'use_symmetry': True, 'symmetry_weight': 0.02,
        'use_freq_consistency': True, 'freq_consistency_weight': 0.05, 'freq_consistency_type': 'diff',
        'use_continuity': True, 'continuity_weight': 0.05, 'continuity_type': 'standard',
        'use_multiscale': True, 'multiscale_weight': 0.1
    }
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """获取预设配置"""
    return PRESET_CONFIGS.get(preset_name, PRESET_CONFIGS['original']).copy()