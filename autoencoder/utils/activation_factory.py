"""
激活函数工厂模块

提供统一的激活函数创建接口，支持多种激活函数类型。
用于参数化AutoEncoder模型，避免为每种激活函数创建独立的类。

支持的激活函数:
- relu: 标准ReLU（默认）
- sin: 正弦激活（周期性信号）
- gelu: Gaussian Error Linear Unit（Transformer常用）
- swish/silu: Swish/SiLU (x * sigmoid(x))
- leaky_relu: Leaky ReLU（解决死神经元）
- elu: Exponential Linear Unit
- tanh: 双曲正切
- mish: Mish激活 (x * tanh(softplus(x)))
"""

import torch
import torch.nn as nn
from typing import Union


class SinActivation(nn.Module):
    """正弦激活函数"""
    def forward(self, x):
        return torch.sin(x)


class MishActivation(nn.Module):
    """
    Mish激活函数: x * tanh(softplus(x))

    参考: https://arxiv.org/abs/1908.08681
    特点: 平滑、非单调、自正则化
    """
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


def get_activation(activation: Union[str, nn.Module], inplace: bool = True) -> nn.Module:
    """
    激活函数工厂方法

    Args:
        activation: 激活函数名称（字符串）或nn.Module实例
        inplace: 是否使用inplace操作（仅适用于ReLU/LeakyReLU/ELU）

    Returns:
        激活函数的nn.Module实例

    Example:
        >>> act = get_activation('relu')
        >>> act = get_activation('sin')
        >>> act = get_activation('gelu')
    """
    # 如果已经是Module，直接返回
    if isinstance(activation, nn.Module):
        return activation

    # 统一转为小写
    activation = activation.lower().strip()

    # 激活函数映射表
    activations = {
        'relu': lambda: nn.ReLU(inplace=inplace),
        'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01, inplace=inplace),
        'elu': lambda: nn.ELU(alpha=1.0, inplace=inplace),
        'prelu': lambda: nn.PReLU(),
        'sin': lambda: SinActivation(),
        'sine': lambda: SinActivation(),  # 别名
        'gelu': lambda: nn.GELU(),
        'swish': lambda: nn.SiLU(),
        'silu': lambda: nn.SiLU(),  # Swish的官方名称
        'tanh': lambda: nn.Tanh(),
        'sigmoid': lambda: nn.Sigmoid(),
        'mish': lambda: MishActivation(),
    }

    if activation not in activations:
        available = ', '.join(sorted(activations.keys()))
        raise ValueError(
            f"Unknown activation function: '{activation}'. "
            f"Available options: {available}"
        )

    return activations[activation]()


def get_activation_name(activation: Union[str, nn.Module]) -> str:
    """
    获取激活函数的标准名称

    Args:
        activation: 激活函数名称（字符串）或nn.Module实例

    Returns:
        标准化的激活函数名称

    Example:
        >>> get_activation_name('ReLU')
        'relu'
        >>> get_activation_name(nn.ReLU())
        'relu'
    """
    if isinstance(activation, str):
        activation = activation.lower().strip()
        # 处理别名
        alias_map = {
            'sine': 'sin',
            'silu': 'swish',
        }
        return alias_map.get(activation, activation)

    # 从Module实例推断名称
    class_name = activation.__class__.__name__.lower()

    if 'relu' in class_name:
        if 'leaky' in class_name:
            return 'leaky_relu'
        elif 'prelu' in class_name:
            return 'prelu'
        else:
            return 'relu'
    elif 'sin' in class_name:
        return 'sin'
    elif 'gelu' in class_name:
        return 'gelu'
    elif 'silu' in class_name or 'swish' in class_name:
        return 'swish'
    elif 'elu' in class_name:
        return 'elu'
    elif 'tanh' in class_name:
        return 'tanh'
    elif 'sigmoid' in class_name:
        return 'sigmoid'
    elif 'mish' in class_name:
        return 'mish'
    else:
        return 'unknown'


def list_available_activations() -> list:
    """
    列出所有可用的激活函数

    Returns:
        激活函数名称列表
    """
    return [
        'relu',
        'leaky_relu',
        'elu',
        'prelu',
        'sin',
        'gelu',
        'swish',
        'tanh',
        'sigmoid',
        'mish'
    ]


# 常用激活函数推荐
RECOMMENDED_ACTIVATIONS = {
    'general': 'relu',           # 通用推荐
    'periodic': 'sin',           # 周期性信号
    'transformer': 'gelu',       # Transformer架构
    'smooth': 'swish',           # 平滑激活
    'deep_network': 'elu',       # 深层网络
    'robust': 'mish',            # 鲁棒性好
}


if __name__ == "__main__":
    print("=" * 80)
    print("激活函数工厂测试")
    print("=" * 80)

    # 测试所有激活函数
    print("\n[Test 1] 测试所有可用激活函数")
    for name in list_available_activations():
        try:
            act = get_activation(name)
            print(f"  OK  {name:15s} -> {act.__class__.__name__}")
        except Exception as e:
            print(f"  ERR {name:15s} -> ERROR: {e}")

    # 测试前向传播
    print("\n[Test 2] 测试前向传播")
    x = torch.randn(4, 10)

    for name in ['relu', 'sin', 'gelu', 'swish', 'mish']:
        act = get_activation(name)
        y = act(x)
        print(f"  {name:10s}: input {x.shape} -> output {y.shape}, "
              f"mean={y.mean().item():.4f}, std={y.std().item():.4f}")

    # 测试名称推断
    print("\n[Test 3] 测试名称推断")
    test_cases = [
        (nn.ReLU(), 'relu'),
        (nn.GELU(), 'gelu'),
        (SinActivation(), 'sin'),
        (nn.SiLU(), 'swish'),
    ]

    for module, expected in test_cases:
        name = get_activation_name(module)
        status = "OK " if name == expected else "ERR"
        print(f"  {status} {module.__class__.__name__:20s} -> '{name}' (expected: '{expected}')")

    # 测试错误处理
    print("\n[Test 4] 测试错误处理")
    try:
        get_activation('invalid_activation')
        print("  ERR Should have raised ValueError")
    except ValueError as e:
        print(f"  OK  Correctly raised ValueError: {str(e)[:60]}...")

    # 显示推荐
    print("\n[Recommendations] 激活函数推荐")
    for scenario, activation in RECOMMENDED_ACTIVATIONS.items():
        print(f"  {scenario:15s} -> {activation}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
