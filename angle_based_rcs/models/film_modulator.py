"""
FiLM调制层 (FiLMModulator)

Feature-wise Linear Modulation (FiLM) 机制，用参数+频率嵌入调制角度特征。

FiLM机制：
    γ = Linear_γ(param_freq_embed)  # 缩放参数
    β = Linear_β(param_freq_embed)  # 平移参数
    output = γ ⊙ angle_features + β

优势：
- 参数化调制：不同参数/频率组合产生不同的调制效果
- 保留角度结构：角度特征经过仿射变换，保持相对关系
- 高效融合：线性复杂度，易于训练

参考文献:
- FiLM: Perez et al., "FiLM: Visual Reasoning with a General
        Conditioning Layer", AAAI 2018
"""

import torch
import torch.nn as nn


class FiLMModulator(nn.Module):
    """
    FiLM调制层：用参数嵌入调制角度特征

    机制：
        output = γ * angle_features + β
        其中 γ, β 从参数嵌入线性投影得到

    参数：
        angle_feature_dim (int): 角度特征维度 (默认: 64)
        param_embed_dim (int): 参数嵌入维度 (默认: 128)

    输入：
        angle_features: [B, angle_feature_dim] 张量，角度编码特征
        param_embed: [B, param_embed_dim] 张量，参数+频率嵌入

    输出：
        modulated_features: [B, angle_feature_dim] 张量，调制后的特征

    示例：
        >>> modulator = FiLMModulator(angle_feature_dim=64, param_embed_dim=128)
        >>> angle_feat = torch.randn(4, 64)
        >>> param_emb = torch.randn(4, 128)
        >>> modulated = modulator(angle_feat, param_emb)
        >>> print(modulated.shape)  # torch.Size([4, 64])
    """

    def __init__(self,
                 angle_feature_dim: int = 64,
                 param_embed_dim: int = 128):
        super().__init__()

        self.angle_feature_dim = angle_feature_dim
        self.param_embed_dim = param_embed_dim

        # γ生成器：[128] → [64]
        self.gamma_proj = nn.Linear(param_embed_dim, angle_feature_dim)

        # β生成器：[128] → [64]
        self.beta_proj = nn.Linear(param_embed_dim, angle_feature_dim)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化

        策略：
        - γ初始化接近1.0（接近恒等变换，稳定训练）
        - β初始化接近0.0（初始无偏移）
        """
        # γ: 均值1.0，小方差（保持接近恒等变换）
        nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=0.01)
        if self.gamma_proj.bias is not None:
            nn.init.constant_(self.gamma_proj.bias, 1.0)

        # β: 均值0.0，小方差（初始无偏移）
        nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.01)
        if self.beta_proj.bias is not None:
            nn.init.constant_(self.beta_proj.bias, 0.0)

    def forward(self,
                angle_features: torch.Tensor,
                param_embed: torch.Tensor) -> torch.Tensor:
        """
        前向传播：FiLM调制

        参数：
            angle_features: [B, angle_feature_dim] 角度特征
            param_embed: [B, param_embed_dim] 参数+频率嵌入

        返回：
            modulated_features: [B, angle_feature_dim] 调制后的特征
        """
        # 验证输入形状
        if angle_features.shape[-1] != self.angle_feature_dim:
            raise ValueError(
                f"角度特征维度不匹配: 期望{self.angle_feature_dim}，"
                f"实际{angle_features.shape[-1]}"
            )
        if param_embed.shape[-1] != self.param_embed_dim:
            raise ValueError(
                f"参数嵌入维度不匹配: 期望{self.param_embed_dim}，"
                f"实际{param_embed.shape[-1]}"
            )

        # 生成调制参数
        gamma = self.gamma_proj(param_embed)  # [B, angle_feature_dim]
        beta = self.beta_proj(param_embed)    # [B, angle_feature_dim]

        # FiLM调制：γ * x + β
        modulated = gamma * angle_features + beta

        return modulated

    def get_modulation_stats(self,
                            param_embed: torch.Tensor) -> dict:
        """
        获取调制统计信息（用于调试和可视化）

        参数：
            param_embed: [B, param_embed_dim] 参数嵌入

        返回：
            stats: 包含γ和β统计信息的字典
        """
        with torch.no_grad():
            gamma = self.gamma_proj(param_embed)
            beta = self.beta_proj(param_embed)

            stats = {
                'gamma_mean': gamma.mean().item(),
                'gamma_std': gamma.std().item(),
                'gamma_min': gamma.min().item(),
                'gamma_max': gamma.max().item(),
                'beta_mean': beta.mean().item(),
                'beta_std': beta.std().item(),
                'beta_min': beta.min().item(),
                'beta_max': beta.max().item()
            }

        return stats

    def extra_repr(self) -> str:
        """打印模块信息"""
        return (f'angle_feature_dim={self.angle_feature_dim}, '
                f'param_embed_dim={self.param_embed_dim}')


if __name__ == "__main__":
    print("=" * 80)
    print("FiLMModulator 测试")
    print("=" * 80)

    # 测试1: 基本功能
    print("\n[Test 1] 基本功能测试")
    modulator = FiLMModulator(angle_feature_dim=64, param_embed_dim=128)
    print(f"调制器: {modulator}")
    total_params = sum(p.numel() for p in modulator.parameters())
    print(f"参数量: {total_params:,}")
    print(f"理论参数量: (128*64 + 64) * 2 = {(128*64+64)*2:,}")

    # 测试2: 前向传播
    print("\n[Test 2] 前向传播")
    batch_size = 4
    angle_feat = torch.randn(batch_size, 64)
    param_emb = torch.randn(batch_size, 128)

    modulated = modulator(angle_feat, param_emb)
    print(f"角度特征形状: {angle_feat.shape}")
    print(f"参数嵌入形状: {param_emb.shape}")
    print(f"调制后形状: {modulated.shape} (期望: [4, 64])")
    print(f"调制前范围: [{angle_feat.min():.3f}, {angle_feat.max():.3f}]")
    print(f"调制后范围: [{modulated.min():.3f}, {modulated.max():.3f}]")

    # 测试3: 调制统计信息
    print("\n[Test 3] 调制统计信息")
    stats = modulator.get_modulation_stats(param_emb)
    print("γ统计:")
    print(f"  均值={stats['gamma_mean']:.3f}, 标准差={stats['gamma_std']:.3f}")
    print(f"  范围=[{stats['gamma_min']:.3f}, {stats['gamma_max']:.3f}]")
    print("β统计:")
    print(f"  均值={stats['beta_mean']:.3f}, 标准差={stats['beta_std']:.3f}")
    print(f"  范围=[{stats['beta_min']:.3f}, {stats['beta_max']:.3f}]")

    # 测试4: 初始化验证
    print("\n[Test 4] 初始化验证")
    modulator_init = FiLMModulator()
    param_test = torch.randn(10, 128)
    gamma_init = modulator_init.gamma_proj(param_test)
    beta_init = modulator_init.beta_proj(param_test)

    print(f"γ初始均值: {gamma_init.mean():.3f} (期望接近1.0)")
    print(f"γ初始标准差: {gamma_init.std():.3f} (期望较小)")
    print(f"β初始均值: {beta_init.mean():.3f} (期望接近0.0)")
    print(f"β初始标准差: {beta_init.std():.3f} (期望较小)")

    # 测试5: 梯度验证
    print("\n[Test 5] 梯度验证")
    modulator_grad = FiLMModulator()
    angle_grad = torch.randn(2, 64, requires_grad=True)
    param_grad = torch.randn(2, 128, requires_grad=True)
    mod_grad = modulator_grad(angle_grad, param_grad)
    loss = mod_grad.sum()
    loss.backward()

    print(f"角度特征梯度存在: {angle_grad.grad is not None}")
    print(f"参数嵌入梯度存在: {param_grad.grad is not None}")
    print(f"γ权重梯度存在: {modulator_grad.gamma_proj.weight.grad is not None}")
    print(f"β权重梯度存在: {modulator_grad.beta_proj.weight.grad is not None}")

    if angle_grad.grad is not None:
        print(f"角度特征梯度范数: {angle_grad.grad.norm():.3f}")
    if param_grad.grad is not None:
        print(f"参数嵌入梯度范数: {param_grad.grad.norm():.3f}")

    # 测试6: 不同参数产生不同调制
    print("\n[Test 6] 不同参数产生不同调制")
    angle_same = torch.randn(1, 64).repeat(3, 1)  # 相同角度特征
    param_diff = torch.randn(3, 128)  # 不同参数嵌入

    mod1 = modulator(angle_same[0:1], param_diff[0:1])
    mod2 = modulator(angle_same[1:2], param_diff[1:2])
    mod3 = modulator(angle_same[2:3], param_diff[2:3])

    diff_12 = (mod1 - mod2).abs().mean()
    diff_13 = (mod1 - mod3).abs().mean()
    diff_23 = (mod2 - mod3).abs().mean()

    print(f"相同角度，不同参数:")
    print(f"  调制1 vs 调制2差异: {diff_12:.3f}")
    print(f"  调制1 vs 调制3差异: {diff_13:.3f}")
    print(f"  调制2 vs 调制3差异: {diff_23:.3f}")
    print(f"  (差异应显著>0，表明参数影响调制)")

    # 测试7: 错误处理
    print("\n[Test 7] 错误处理")
    try:
        wrong_angle = torch.randn(4, 32)  # 错误维度
        modulator(wrong_angle, param_emb)
        print("  ❌ 应该抛出ValueError")
    except ValueError as e:
        print(f"  ✅ 正确捕获错误: {str(e)[:50]}...")

    try:
        wrong_param = torch.randn(4, 64)  # 错误维度
        modulator(angle_feat, wrong_param)
        print("  ❌ 应该抛出ValueError")
    except ValueError as e:
        print(f"  ✅ 正确捕获错误: {str(e)[:50]}...")

    # 测试8: FiLM效果可视化
    print("\n[Test 8] FiLM调制效果")
    # 创建简单角度特征
    angle_simple = torch.ones(1, 64)
    param_zero = torch.zeros(1, 128)

    # 零参数应该产生接近恒等变换（γ≈1, β≈0）
    mod_zero = modulator(angle_simple, param_zero)
    identity_error = (mod_zero - angle_simple).abs().mean()
    print(f"零参数输入，调制误差: {identity_error:.4f} (期望接近0)")

    print("\n" + "=" * 80)
    print("✅ FiLMModulator测试通过!")
    print("=" * 80)
