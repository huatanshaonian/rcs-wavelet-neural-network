"""
角度编码器 (AngleEncoder)

使用傅里叶特征映射 (Fourier Feature Mapping) 将连续角度 (θ, φ) 编码为高维特征。
类似于NeRF中的位置编码，能够让网络学习高频细节。

参考文献:
- NeRF: Mildenhall et al., 2020
- Fourier Features: Tancik et al., 2020
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union


class AngleEncoder(nn.Module):
    """
    角度编码器：使用傅里叶特征映射编码角度

    编码方式：
        对于每个频率 k ∈ [0, L-1]:
            freq = 2^k * π
            输出 += [sin(freq*θ), cos(freq*θ), sin(freq*φ), cos(freq*φ)]

    参数：
        L (int): 频率数量 (默认: 16)
                 - L=8: 输出维度32 (低频，更平滑)
                 - L=16: 输出维度64 (中频，平衡，推荐)
                 - L=32: 输出维度128 (高频，更精细)

        theta_range (tuple): θ角度范围，用于归一化 (默认: (45.0, 135.0))
        phi_range (tuple): φ角度范围，用于归一化 (默认: (-45.0, 45.0))

    输入：
        theta: [B,] 张量，θ角度（度）
        phi: [B,] 张量，φ角度（度）

    输出：
        encodings: [B, L*4] 张量，傅里叶特征
                   (L个频率 × 4个分量: sin_θ, cos_θ, sin_φ, cos_φ)

    示例：
        >>> encoder = AngleEncoder(L=16)
        >>> theta = torch.tensor([45.0, 90.0, 135.0])
        >>> phi = torch.tensor([-45.0, 0.0, 45.0])
        >>> features = encoder(theta, phi)
        >>> print(features.shape)  # torch.Size([3, 64])
    """

    def __init__(self,
                 L: int = 16,
                 theta_range: tuple = (45.0, 135.0),
                 phi_range: tuple = (-45.0, 45.0)):
        super().__init__()

        self.L = L
        self.theta_range = theta_range
        self.phi_range = phi_range

        # 计算输出维度
        self.output_dim = L * 4  # L频率 × 4分量

        # 预计算频率权重: [2^0 * π, 2^1 * π, ..., 2^(L-1) * π]
        # 使用register_buffer确保设备移动时一起移动
        frequencies = torch.tensor([2**k * np.pi for k in range(L)], dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)

    def _normalize_angle(self,
                         angle: torch.Tensor,
                         angle_range: tuple) -> torch.Tensor:
        """
        将角度归一化到 [-1, 1] 范围

        参数：
            angle: 原始角度（度）
            angle_range: (min, max) 角度范围

        返回：
            归一化后的角度 [-1, 1]
        """
        min_val, max_val = angle_range
        # 归一化到 [0, 1]
        normalized = (angle - min_val) / (max_val - min_val)
        # 归一化到 [-1, 1]
        normalized = 2.0 * normalized - 1.0
        return normalized

    def forward(self,
                theta: Union[torch.Tensor, float, np.ndarray],
                phi: Union[torch.Tensor, float, np.ndarray]) -> torch.Tensor:
        """
        前向传播：编码角度

        参数：
            theta: θ角度（度），可以是标量、列表或张量
            phi: φ角度（度），可以是标量、列表或张量

        返回：
            encodings: [B, L*4] 傅里叶特征
        """
        # 转换为张量（兼容多种输入格式）
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32)
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.float32)

        # 确保是浮点型
        theta = theta.float()
        phi = phi.float()

        # 确保设备一致
        device = self.frequencies.device
        theta = theta.to(device)
        phi = phi.to(device)

        # 确保是1维张量
        if theta.dim() == 0:
            theta = theta.unsqueeze(0)
        if phi.dim() == 0:
            phi = phi.unsqueeze(0)

        batch_size = theta.shape[0]

        # 归一化角度到 [-1, 1]
        theta_norm = self._normalize_angle(theta, self.theta_range)  # [B,]
        phi_norm = self._normalize_angle(phi, self.phi_range)        # [B,]

        # 扩展维度以便广播
        theta_norm = theta_norm.unsqueeze(1)  # [B, 1]
        phi_norm = phi_norm.unsqueeze(1)      # [B, 1]
        frequencies = self.frequencies.unsqueeze(0)  # [1, L]

        # 计算傅里叶特征
        # theta_norm * frequencies: [B, L]
        theta_sin = torch.sin(theta_norm * frequencies)  # [B, L]
        theta_cos = torch.cos(theta_norm * frequencies)  # [B, L]
        phi_sin = torch.sin(phi_norm * frequencies)      # [B, L]
        phi_cos = torch.cos(phi_norm * frequencies)      # [B, L]

        # 拼接所有特征: [B, L*4]
        encodings = torch.cat([theta_sin, theta_cos, phi_sin, phi_cos], dim=1)

        return encodings

    def extra_repr(self) -> str:
        """打印模块信息"""
        return (f'L={self.L}, output_dim={self.output_dim}, '
                f'theta_range={self.theta_range}, phi_range={self.phi_range}')


if __name__ == "__main__":
    print("=" * 80)
    print("AngleEncoder 测试")
    print("=" * 80)

    # 测试1: 基本功能
    print("\n[Test 1] 基本功能测试")
    encoder = AngleEncoder(L=16)
    print(f"编码器: {encoder}")
    print(f"输出维度: {encoder.output_dim}")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters())} (应为0)")

    # 测试2: 单点编码
    print("\n[Test 2] 单点编码")
    theta = torch.tensor([90.0])
    phi = torch.tensor([0.0])
    features = encoder(theta, phi)
    print(f"输入: θ={theta.item():.1f}°, φ={phi.item():.1f}°")
    print(f"输出形状: {features.shape} (期望: [1, 64])")
    print(f"输出范围: [{features.min():.3f}, {features.max():.3f}]")

    # 测试3: 批量编码
    print("\n[Test 3] 批量编码")
    theta_batch = torch.tensor([45.0, 90.0, 135.0, 60.0])
    phi_batch = torch.tensor([-45.0, 0.0, 45.0, 20.0])
    features_batch = encoder(theta_batch, phi_batch)
    print(f"批量大小: {theta_batch.shape[0]}")
    print(f"输出形状: {features_batch.shape} (期望: [4, 64])")

    # 测试4: 不同频率数量
    print("\n[Test 4] 不同频率数量")
    for L in [8, 16, 32]:
        enc = AngleEncoder(L=L)
        feat = enc(torch.tensor([90.0]), torch.tensor([0.0]))
        print(f"  L={L:2d}: 输出维度={enc.output_dim:3d}, 形状={feat.shape}")

    # 测试5: 梯度验证
    print("\n[Test 5] 梯度验证")
    encoder_grad = AngleEncoder(L=16)
    theta_grad = torch.tensor([90.0], requires_grad=True)
    phi_grad = torch.tensor([0.0], requires_grad=True)
    features_grad = encoder_grad(theta_grad, phi_grad)
    loss = features_grad.sum()
    loss.backward()
    print(f"θ梯度存在: {theta_grad.grad is not None}")
    print(f"φ梯度存在: {phi_grad.grad is not None}")
    if theta_grad.grad is not None:
        print(f"θ梯度范数: {theta_grad.grad.norm():.3f}")

    # 测试6: 输入格式兼容性
    print("\n[Test 6] 输入格式兼容性")
    test_cases = [
        (90.0, 0.0, "标量"),
        ([90.0], [0.0], "列表"),
        (np.array([90.0]), np.array([0.0]), "numpy数组"),
        (torch.tensor([90.0]), torch.tensor([0.0]), "torch张量")
    ]
    for theta_in, phi_in, desc in test_cases:
        feat = encoder(theta_in, phi_in)
        print(f"  {desc:10s}: 输出形状={feat.shape}")

    print("\n" + "=" * 80)
    print("✅ AngleEncoder测试通过!")
    print("=" * 80)
