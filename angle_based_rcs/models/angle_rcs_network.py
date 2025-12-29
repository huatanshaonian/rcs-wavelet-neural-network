"""
完整角度RCS预测网络 (AngleRCSNetwork)

集成所有模块，实现从角度+参数到单点RCS值的端到端预测。

网络架构：
    输入: (θ, φ) + [9参数] + 频率索引
        ↓
    AngleEncoder (傅里叶特征映射)
    (θ,φ) → [64维] 角度嵌入
        ↓
    FrequencyEncoder (one-hot编码)
    频率索引 → [3维] one-hot向量
        ↓
    ParamEncoder (2层MLP)
    [9参数] + [3频率] → [128维] 参数嵌入
        ↓
    FiLMModulator
    角度嵌入 × γ + β → [64维] 调制特征
        ↓
    RCSPredictor (4层MLP)
    [64维] → [1维] 单点RCS值
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Union

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .angle_encoder import AngleEncoder
from .frequency_encoder import FrequencyEncoder
from .param_encoder import ParameterEncoder
from .film_modulator import FiLMModulator
from autoencoder.utils.activation_factory import get_activation


class RCSPredictor(nn.Module):
    """
    RCS预测头：4层MLP

    架构：
        [64] → Linear(64→128) → Act → Dropout
             → Linear(128→128) → Act → Dropout
             → Linear(128→64) → Act → Dropout
             → Linear(64→1)
             → [1] 单点RCS值
    """

    def __init__(self,
                 input_dim: int = 64,
                 hidden_dims: list = [128, 128, 64],
                 activation: str = 'sin',
                 dropout_rate: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout_rate

        layers = []
        dims = [input_dim] + hidden_dims

        # 隐藏层
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))

        # 输出层（无激活函数，输出标量RCS值）
        layers.append(nn.Linear(dims[-1], 1))

        self.network = nn.Sequential(*layers)

        # SIREN初始化
        if activation.lower() in ['sin', 'sine', 'siren']:
            self._siren_init()

    def _siren_init(self):
        """SIREN风格权重初始化"""
        for m in self.network:
            if isinstance(m, nn.Linear):
                n = m.in_features
                bound = (6.0 / n)**0.5
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AngleRCSNetwork(nn.Module):
    """
    完整的角度RCS预测网络

    参数：
        num_frequencies (int): 频率数量 (2 or 3, 默认: 3)
        angle_L (int): 角度编码频率数量 (默认: 16)
        param_dim (int): 设计参数维度 (默认: 9)
        param_hidden_dim (int): 参数编码器隐藏维度 (默认: 64)
        param_embed_dim (int): 参数嵌入维度 (默认: 128)
        rcs_hidden_dims (list): RCS预测器隐藏层维度 (默认: [128, 128, 64])
        activation (str): 激活函数 (默认: 'sin')
        dropout_rate (float): Dropout率 (默认: 0.1)
        theta_range (tuple): θ角度范围 (默认: (45.0, 135.0))
        phi_range (tuple): φ角度范围 (默认: (-45.0, 45.0))

    输入：
        theta: 角度θ (度)
        phi: 角度φ (度)
        params: [B, 9] 设计参数
        freq_idx: [B,] 频率索引 (0/1/2)

    输出：
        rcs_pred: [B, 1] 单点RCS值

    示例：
        >>> model = AngleRCSNetwork(num_frequencies=3)
        >>> theta = torch.tensor([90.0, 45.0])
        >>> phi = torch.tensor([0.0, -45.0])
        >>> params = torch.randn(2, 9)
        >>> freq_idx = torch.tensor([0, 1])
        >>> rcs = model(theta, phi, params, freq_idx)
        >>> print(rcs.shape)  # torch.Size([2, 1])
    """

    def __init__(self,
                 num_frequencies: int = 3,
                 angle_L: int = 16,
                 param_dim: int = 9,
                 param_hidden_dim: int = 64,
                 param_embed_dim: int = 128,
                 rcs_hidden_dims: list = None,
                 activation: str = 'sin',
                 dropout_rate: float = 0.1,
                 theta_range: tuple = (45.0, 135.0),
                 phi_range: tuple = (-45.0, 45.0)):
        super().__init__()

        if rcs_hidden_dims is None:
            rcs_hidden_dims = [128, 128, 64]

        self.num_frequencies = num_frequencies
        self.angle_L = angle_L
        self.param_dim = param_dim
        self.activation_name = activation

        # 计算各模块输出维度
        self.angle_feature_dim = angle_L * 4  # 64 (16频率 × 4分量)
        self.freq_dim = num_frequencies       # 3
        self.param_embed_dim = param_embed_dim  # 128

        # 1. 角度编码器
        self.angle_encoder = AngleEncoder(
            L=angle_L,
            theta_range=theta_range,
            phi_range=phi_range
        )

        # 2. 频率编码器
        self.frequency_encoder = FrequencyEncoder(
            num_frequencies=num_frequencies
        )

        # 3. 参数编码器
        self.param_encoder = ParameterEncoder(
            param_dim=param_dim,
            freq_dim=self.freq_dim,
            hidden_dim=param_hidden_dim,
            output_dim=param_embed_dim,
            activation=activation,
            dropout_rate=dropout_rate
        )

        # 4. FiLM调制器
        self.film_modulator = FiLMModulator(
            angle_feature_dim=self.angle_feature_dim,
            param_embed_dim=self.param_embed_dim
        )

        # 5. RCS预测器
        self.rcs_predictor = RCSPredictor(
            input_dim=self.angle_feature_dim,
            hidden_dims=rcs_hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate
        )

    def forward(self,
                theta: Union[torch.Tensor, float],
                phi: Union[torch.Tensor, float],
                params: torch.Tensor,
                freq_idx: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        前向传播

        参数：
            theta: θ角度（度）
            phi: φ角度（度）
            params: [B, param_dim] 设计参数
            freq_idx: [B,] 频率索引

        返回：
            rcs_pred: [B, 1] 单点RCS预测值
        """
        # 1. 角度编码
        angle_features = self.angle_encoder(theta, phi)  # [B, 64]

        # 2. 频率编码
        freq_onehot = self.frequency_encoder(freq_idx)   # [B, 3]

        # 3. 参数编码
        param_embed = self.param_encoder(params, freq_onehot)  # [B, 128]

        # 4. FiLM调制
        modulated_features = self.film_modulator(angle_features, param_embed)  # [B, 64]

        # 5. RCS预测
        rcs_pred = self.rcs_predictor(modulated_features)  # [B, 1]

        return rcs_pred

    def count_parameters(self) -> dict:
        """统计各模块参数量"""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        stats = {
            'angle_encoder': _count(self.angle_encoder),
            'frequency_encoder': _count(self.frequency_encoder),
            'param_encoder': _count(self.param_encoder),
            'film_modulator': _count(self.film_modulator),
            'rcs_predictor': _count(self.rcs_predictor),
        }
        stats['total'] = sum(stats.values())

        return stats

    def extra_repr(self) -> str:
        """打印模块信息"""
        return (f'num_frequencies={self.num_frequencies}, '
                f'angle_L={self.angle_L}, '
                f'param_dim={self.param_dim}, '
                f'activation={self.activation_name}')


if __name__ == "__main__":
    print("=" * 80)
    print("AngleRCSNetwork 完整测试")
    print("=" * 80)

    # 测试1: 创建网络
    print("\n[Test 1] 创建网络")
    model = AngleRCSNetwork(num_frequencies=3, angle_L=16, activation='sin')
    print(f"模型: {model}")

    param_stats = model.count_parameters()
    print("\n参数量统计:")
    for name, count in param_stats.items():
        print(f"  {name:20s}: {count:>8,}")

    # 测试2: 前向传播（批量）
    print("\n[Test 2] 前向传播（批量）")
    batch_size = 4
    theta = torch.tensor([45.0, 90.0, 135.0, 60.0])
    phi = torch.tensor([-45.0, 0.0, 45.0, 20.0])
    params = torch.randn(batch_size, 9)
    freq_idx = torch.tensor([0, 1, 2, 1])  # 混合频率

    rcs_pred = model(theta, phi, params, freq_idx)
    print(f"输入:")
    print(f"  θ: {theta.tolist()}")
    print(f"  φ: {phi.tolist()}")
    print(f"  参数: {params.shape}")
    print(f"  频率索引: {freq_idx.tolist()}")
    print(f"输出RCS预测: {rcs_pred.shape} (期望: [4, 1])")
    print(f"RCS值范围: [{rcs_pred.min():.3f}, {rcs_pred.max():.3f}]")

    # 测试3: 前向传播（单点）
    print("\n[Test 3] 前向传播（单点）")
    theta_single = 90.0
    phi_single = 0.0
    params_single = torch.randn(1, 9)
    freq_single = 1  # 3GHz

    rcs_single = model(theta_single, phi_single, params_single, freq_single)
    print(f"输入: θ={theta_single}°, φ={phi_single}°, 频率=3GHz")
    print(f"输出RCS: {rcs_single.item():.6f}")

    # 测试4: 梯度验证
    print("\n[Test 4] 梯度验证")
    model_grad = AngleRCSNetwork(num_frequencies=3)
    theta_grad = torch.tensor([90.0], requires_grad=True)
    phi_grad = torch.tensor([0.0], requires_grad=True)
    params_grad = torch.randn(1, 9, requires_grad=True)
    freq_grad = torch.tensor([0])

    rcs_grad = model_grad(theta_grad, phi_grad, params_grad, freq_grad)
    loss = rcs_grad.sum()
    loss.backward()

    print(f"θ梯度存在: {theta_grad.grad is not None}")
    print(f"φ梯度存在: {phi_grad.grad is not None}")
    print(f"参数梯度存在: {params_grad.grad is not None}")

    if theta_grad.grad is not None:
        print(f"θ梯度: {theta_grad.grad.item():.6f}")
    if phi_grad.grad is not None:
        print(f"φ梯度: {phi_grad.grad.item():.6f}")
    if params_grad.grad is not None:
        print(f"参数梯度范数: {params_grad.grad.norm():.6f}")

    # 测试5: 不同频率对比
    print("\n[Test 5] 不同频率对比（相同角度和参数）")
    theta_test = torch.tensor([90.0, 90.0, 90.0])
    phi_test = torch.tensor([0.0, 0.0, 0.0])
    params_test = torch.randn(1, 9).repeat(3, 1)  # 相同参数
    freq_test = torch.tensor([0, 1, 2])  # 三个频率

    rcs_test = model(theta_test, phi_test, params_test, freq_test)
    print(f"相同角度(θ=90°, φ=0°)和参数，不同频率:")
    for i, freq in enumerate([0, 1, 2]):
        freq_label = model.frequency_encoder.get_frequency_label(freq)
        print(f"  {freq_label}: RCS={rcs_test[i].item():.6f}")

    # 测试6: 2频率配置
    print("\n[Test 6] 2频率配置")
    model_2freq = AngleRCSNetwork(num_frequencies=2)
    params_2freq = model_2freq.count_parameters()
    print(f"2频率模型总参数: {params_2freq['total']:,}")

    theta_2f = torch.tensor([90.0, 90.0])
    phi_2f = torch.tensor([0.0, 0.0])
    params_2f = torch.randn(2, 9)
    freq_2f = torch.tensor([0, 1])

    rcs_2freq = model_2freq(theta_2f, phi_2f, params_2f, freq_2f)
    print(f"2频率输出形状: {rcs_2freq.shape}")

    # 测试7: 不同激活函数
    print("\n[Test 7] 不同激活函数")
    for act in ['sin', 'relu', 'gelu', 'swish']:
        model_act = AngleRCSNetwork(activation=act, num_frequencies=3)
        rcs_act = model_act(theta_single, phi_single, params_single, freq_single)
        total_params = model_act.count_parameters()['total']
        print(f"  {act:8s}: 参数量={total_params:>7,}, RCS={rcs_act.item():.6f}")

    # 测试8: 批量推理性能测试
    print("\n[Test 8] 批量推理性能测试")
    import time

    model.eval()
    large_batch = 1000
    theta_large = torch.randn(large_batch) * 90 + 90  # [45, 135]
    phi_large = torch.randn(large_batch) * 90  # [-45, 45]
    params_large = torch.randn(large_batch, 9)
    freq_large = torch.randint(0, 3, (large_batch,))

    with torch.no_grad():
        start_time = time.time()
        rcs_large = model(theta_large, phi_large, params_large, freq_large)
        end_time = time.time()

    elapsed = (end_time - start_time) * 1000  # 转换为毫秒
    per_point = elapsed / large_batch
    print(f"批量大小: {large_batch}")
    print(f"总耗时: {elapsed:.2f}ms")
    print(f"单点耗时: {per_point:.4f}ms")
    print(f"吞吐量: {1000/elapsed*large_batch:.0f} 点/秒")

    print("\n" + "=" * 80)
    print("✅ AngleRCSNetwork完整测试通过!")
    print("=" * 80)
    print("\n网络就绪，可以开始训练！")
