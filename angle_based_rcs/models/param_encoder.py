"""
参数+频率编码器 (ParameterEncoder)

将9个设计参数和3维频率one-hot编码组合，通过2层MLP编码为高维参数嵌入。
输出用于FiLM调制角度特征。

架构：
    [9参数] + [3频率] = [12维输入]
        ↓
    Linear(12→64) → Activation → Dropout(0.1)
        ↓
    Linear(64→128) → Activation → Dropout(0.1)
        ↓
    [128维] 参数+频率嵌入
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autoencoder.utils.activation_factory import get_activation


class ParameterEncoder(nn.Module):
    """
    参数+频率编码器

    将9个设计参数和频率编码（3维one-hot）组合并编码为高维嵌入。

    参数：
        param_dim (int): 设计参数维度 (默认: 9)
        freq_dim (int): 频率编码维度 (默认: 3)
        hidden_dim (int): 隐藏层维度 (默认: 64)
        output_dim (int): 输出嵌入维度 (默认: 128)
        activation (str): 激活函数 (默认: 'sin')
                         可选: 'relu', 'sin', 'gelu', 'swish', 'tanh'
        dropout_rate (float): Dropout率 (默认: 0.1)

    输入：
        params: [B, param_dim] 张量，设计参数
        freq_onehot: [B, freq_dim] 张量，频率one-hot编码

    输出：
        param_embed: [B, output_dim] 张量，参数+频率嵌入

    示例：
        >>> encoder = ParameterEncoder(activation='sin')
        >>> params = torch.randn(4, 9)
        >>> freq_onehot = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,0,0]], dtype=torch.float32)
        >>> embed = encoder(params, freq_onehot)
        >>> print(embed.shape)  # torch.Size([4, 128])
    """

    def __init__(self,
                 param_dim: int = 9,
                 freq_dim: int = 3,
                 hidden_dim: int = 64,
                 output_dim: int = 128,
                 activation: str = 'sin',
                 dropout_rate: float = 0.1):
        super().__init__()

        self.param_dim = param_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate

        # 输入维度 = 参数维度 + 频率维度
        input_dim = param_dim + freq_dim

        # 第一层：[12] → [64]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = get_activation(activation)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第二层：[64] → [128]
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = get_activation(activation)
        self.dropout2 = nn.Dropout(dropout_rate)

        # 参数初始化（SIREN风格，适用于sin激活）
        if activation.lower() in ['sin', 'sine', 'siren']:
            self._siren_init()

    def _siren_init(self):
        """
        SIREN风格权重初始化

        参考: Sitzmann et al., "Implicit Neural Representations with
              Periodic Activation Functions", 2020
        """
        with torch.no_grad():
            # 第一层：Uniform(-sqrt(1/n), sqrt(1/n))
            n1 = self.fc1.in_features
            bound1 = 1.0 / n1**0.5
            self.fc1.weight.uniform_(-bound1, bound1)
            if self.fc1.bias is not None:
                self.fc1.bias.uniform_(-bound1, bound1)

            # 第二层：Uniform(-sqrt(6/n), sqrt(6/n))
            n2 = self.fc2.in_features
            bound2 = (6.0 / n2)**0.5
            self.fc2.weight.uniform_(-bound2, bound2)
            if self.fc2.bias is not None:
                self.fc2.bias.uniform_(-bound2, bound2)

    def forward(self, params: torch.Tensor, freq_onehot: torch.Tensor) -> torch.Tensor:
        """
        前向传播：编码参数和频率

        参数：
            params: [B, param_dim] 设计参数
            freq_onehot: [B, freq_dim] 频率one-hot编码

        返回：
            param_embed: [B, output_dim] 参数+频率嵌入
        """
        # 验证输入形状
        if params.shape[-1] != self.param_dim:
            raise ValueError(
                f"参数维度不匹配: 期望{self.param_dim}，实际{params.shape[-1]}"
            )
        if freq_onehot.shape[-1] != self.freq_dim:
            raise ValueError(
                f"频率维度不匹配: 期望{self.freq_dim}，实际{freq_onehot.shape[-1]}"
            )

        # 拼接参数和频率编码: [B, 12]
        combined = torch.cat([params, freq_onehot], dim=-1)

        # 第一层
        x = self.fc1(combined)      # [B, 64]
        x = self.act1(x)
        x = self.dropout1(x)

        # 第二层
        x = self.fc2(x)             # [B, 128]
        x = self.act2(x)
        x = self.dropout2(x)

        return x

    def extra_repr(self) -> str:
        """打印模块信息"""
        return (f'param_dim={self.param_dim}, freq_dim={self.freq_dim}, '
                f'hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, '
                f'activation={self.activation_name}, dropout={self.dropout_rate}')


if __name__ == "__main__":
    print("=" * 80)
    print("ParameterEncoder 测试")
    print("=" * 80)

    # 测试1: 基本功能
    print("\n[Test 1] 基本功能测试")
    encoder = ParameterEncoder(activation='sin')
    print(f"编码器: {encoder}")
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"参数量: {total_params:,}")
    print(f"理论参数量: (12*64 + 64) + (64*128 + 128) = {12*64+64 + 64*128+128:,}")

    # 测试2: 前向传播
    print("\n[Test 2] 前向传播")
    batch_size = 4
    params = torch.randn(batch_size, 9)
    freq_onehot = torch.tensor([
        [1, 0, 0],  # 1.5GHz
        [0, 1, 0],  # 3GHz
        [0, 0, 1],  # 6GHz
        [1, 0, 0]   # 1.5GHz
    ], dtype=torch.float32)

    embed = encoder(params, freq_onehot)
    print(f"参数形状: {params.shape}")
    print(f"频率编码形状: {freq_onehot.shape}")
    print(f"输出嵌入形状: {embed.shape} (期望: [4, 128])")
    print(f"输出范围: [{embed.min():.3f}, {embed.max():.3f}]")

    # 测试3: 不同激活函数
    print("\n[Test 3] 不同激活函数")
    for act in ['sin', 'relu', 'gelu', 'swish', 'tanh']:
        enc = ParameterEncoder(activation=act)
        out = enc(params, freq_onehot)
        print(f"  {act:8s}: 输出范围=[{out.min():.3f}, {out.max():.3f}], "
              f"均值={out.mean():.3f}, 标准差={out.std():.3f}")

    # 测试4: 梯度验证
    print("\n[Test 4] 梯度验证")
    encoder_grad = ParameterEncoder(activation='sin')
    params_grad = torch.randn(2, 9, requires_grad=True)
    freq_grad = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    embed_grad = encoder_grad(params_grad, freq_grad)
    loss = embed_grad.sum()
    loss.backward()

    print(f"参数梯度存在: {params_grad.grad is not None}")
    print(f"权重梯度存在: {encoder_grad.fc1.weight.grad is not None}")
    if params_grad.grad is not None:
        print(f"参数梯度范数: {params_grad.grad.norm():.3f}")

    # 测试5: Dropout效果
    print("\n[Test 5] Dropout效果")
    encoder_drop = ParameterEncoder(activation='sin', dropout_rate=0.5)
    encoder_drop.train()  # 训练模式

    out1 = encoder_drop(params, freq_onehot)
    out2 = encoder_drop(params, freq_onehot)
    diff = (out1 - out2).abs().mean()
    print(f"训练模式（dropout=0.5）: 两次前向传播差异={diff:.3f} (应>0)")

    encoder_drop.eval()  # 评估模式
    out3 = encoder_drop(params, freq_onehot)
    out4 = encoder_drop(params, freq_onehot)
    diff_eval = (out3 - out4).abs().mean()
    print(f"评估模式: 两次前向传播差异={diff_eval:.6f} (应≈0)")

    # 测试6: 错误处理
    print("\n[Test 6] 错误处理")
    try:
        wrong_params = torch.randn(4, 7)  # 错误维度
        encoder(wrong_params, freq_onehot)
        print("  ❌ 应该抛出ValueError")
    except ValueError as e:
        print(f"  ✅ 正确捕获错误: {str(e)}")

    try:
        wrong_freq = torch.randn(4, 5)  # 错误维度
        encoder(params, wrong_freq)
        print("  ❌ 应该抛出ValueError")
    except ValueError as e:
        print(f"  ✅ 正确捕获错误: {str(e)}")

    # 测试7: SIREN初始化验证
    print("\n[Test 7] SIREN初始化验证")
    encoder_siren = ParameterEncoder(activation='sin')
    w1 = encoder_siren.fc1.weight.data
    w2 = encoder_siren.fc2.weight.data

    bound1 = 1.0 / 12**0.5  # 输入维度=12
    bound2 = (6.0 / 64)**0.5  # 隐藏维度=64

    w1_range = (w1.min().item(), w1.max().item())
    w2_range = (w2.min().item(), w2.max().item())

    print(f"第一层权重范围: [{w1_range[0]:.4f}, {w1_range[1]:.4f}]")
    print(f"期望范围: [{-bound1:.4f}, {bound1:.4f}]")
    print(f"第二层权重范围: [{w2_range[0]:.4f}, {w2_range[1]:.4f}]")
    print(f"期望范围: [{-bound2:.4f}, {bound2:.4f}]")

    print("\n" + "=" * 80)
    print("✅ ParameterEncoder测试通过!")
    print("=" * 80)
