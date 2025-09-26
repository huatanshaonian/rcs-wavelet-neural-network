"""
现代化的网络创建接口
集成插件化架构到现有训练系统，保持向后兼容
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import warnings

# 导入新的插件化架构
try:
    from network_registry import (
        NetworkRegistry, NetworkConfig, LossConfig,
        create_network as registry_create_network,
        create_loss as registry_create_loss,
        list_available_networks
    )
    # 自动发现并注册网络
    NetworkRegistry.auto_discover_networks(['networks.example_networks'])
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    warnings.warn("插件化架构未可用，使用传统创建方式")

# 导入原有的网络
try:
    from wavelet_network import TriDimensionalRCSNet, TriDimensionalRCSLoss
    from enhanced_network import EnhancedTriDimensionalRCSNet, ImprovedRCSLoss
    LEGACY_NETWORKS_AVAILABLE = True
except ImportError:
    LEGACY_NETWORKS_AVAILABLE = False
    warnings.warn("原有网络导入失败")

def create_model(
    input_dim: int = 9,
    model_type: str = 'original',
    use_log_output: bool = False,
    output_shape: tuple = (91, 91, 2),
    **kwargs
) -> nn.Module:
    """
    统一网络创建接口

    支持两种模式：
    1. 传统模式: model_type = 'original' | 'enhanced'
    2. 插件模式: model_type = 任何注册的网络名称

    Args:
        input_dim: 输入维度
        model_type: 网络类型
        use_log_output: 是否使用对数输出
        output_shape: 输出形状
        **kwargs: 其他网络参数

    Returns:
        创建的网络模型
    """

    print(f"创建网络: {model_type}")

    # 1. 尝试插件化架构
    if REGISTRY_AVAILABLE:
        try:
            model = registry_create_network(
                model_type,
                input_dim=input_dim,
                output_shape=output_shape,
                use_log_output=use_log_output
            )
            print(f"[OK] 使用插件化架构创建: {model.get_name()}")
            return model
        except (ValueError, TypeError) as e:
            # 插件架构中没有找到或参数错误，尝试传统方式
            print(f"插件化架构创建失败: {e}")
            pass

    # 2. 传统网络创建方式 (向后兼容)
    if not LEGACY_NETWORKS_AVAILABLE:
        raise ImportError("无法导入传统网络，且插件化架构不可用")

    if model_type == 'original':
        print("[OK] 使用传统RCS网络架构")
        return TriDimensionalRCSNet(input_dim=input_dim, use_log_output=use_log_output)
    elif model_type == 'enhanced':
        print("[OK] 使用增强版RCS网络架构")
        return EnhancedTriDimensionalRCSNet(input_dim=input_dim, use_log_output=use_log_output)
    else:
        available_networks = []
        if REGISTRY_AVAILABLE:
            available_networks.extend(NetworkRegistry.list_networks().keys())
        available_networks.extend(['original', 'enhanced'])

        raise ValueError(
            f"未知的网络类型: {model_type}。"
            f"可用类型: {available_networks}"
        )

def create_loss_function(
    loss_type: str = 'original',
    loss_weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> nn.Module:
    """
    统一损失函数创建接口

    Args:
        loss_type: 损失函数类型
        loss_weights: 损失权重
        **kwargs: 其他损失函数参数

    Returns:
        创建的损失函数
    """

    print(f"创建损失函数: {loss_type}")

    # 1. 尝试插件化架构
    if REGISTRY_AVAILABLE:
        try:
            config = LossConfig(
                loss_type=loss_type,
                loss_weights=loss_weights
            )
            loss_fn = registry_create_loss(loss_type, config)
            print(f"[OK] 使用插件化损失函数")
            return loss_fn
        except ValueError:
            # 插件架构中没有找到，尝试传统方式
            pass

    # 2. 传统损失函数创建方式
    if not LEGACY_NETWORKS_AVAILABLE:
        raise ImportError("无法导入传统损失函数，且插件化架构不可用")

    if loss_type == 'original':
        print("[OK] 使用传统RCS损失函数")
        return TriDimensionalRCSLoss()
    elif loss_type in ['improved', 'enhanced']:
        print("[OK] 使用改进版损失函数")
        return ImprovedRCSLoss()
    else:
        available_losses = []
        if REGISTRY_AVAILABLE:
            available_losses.extend(NetworkRegistry.list_losses())
        available_losses.extend(['original', 'improved'])

        raise ValueError(
            f"未知的损失函数类型: {loss_type}。"
            f"可用类型: {available_losses}"
        )

def get_available_networks() -> Dict[str, str]:
    """获取所有可用的网络类型"""
    networks = {}

    # 传统网络
    if LEGACY_NETWORKS_AVAILABLE:
        networks['original'] = '传统小波RCS网络'
        networks['enhanced'] = '增强版小波RCS网络'

    # 插件化网络
    if REGISTRY_AVAILABLE:
        plugin_networks = NetworkRegistry.list_networks()
        networks.update(plugin_networks)

    return networks

def get_available_losses() -> Dict[str, str]:
    """获取所有可用的损失函数类型"""
    losses = {}

    # 传统损失函数
    if LEGACY_NETWORKS_AVAILABLE:
        losses['original'] = '传统RCS损失函数'
        losses['improved'] = '改进版RCS损失函数'

    # 插件化损失函数
    if REGISTRY_AVAILABLE:
        plugin_losses = NetworkRegistry.list_losses()
        for loss_name in plugin_losses:
            losses[loss_name] = f'插件损失函数: {loss_name}'

    return losses

def get_network_info(model_type: str) -> Dict[str, Any]:
    """获取网络详细信息"""

    # 先尝试插件化架构
    if REGISTRY_AVAILABLE:
        try:
            return NetworkRegistry.get_network_info(model_type)
        except ValueError:
            pass

    # 传统网络信息
    if model_type == 'original' and LEGACY_NETWORKS_AVAILABLE:
        model = TriDimensionalRCSNet()
        total_params = sum(p.numel() for p in model.parameters())
        return {
            'name': 'original',
            'description': '传统小波RCS网络',
            'parameters': {'total': total_params},
            'output_shape': (91, 91, 2)
        }
    elif model_type == 'enhanced' and LEGACY_NETWORKS_AVAILABLE:
        model = EnhancedTriDimensionalRCSNet()
        total_params = sum(p.numel() for p in model.parameters())
        return {
            'name': 'enhanced',
            'description': '增强版小波RCS网络',
            'parameters': {'total': total_params},
            'output_shape': (91, 91, 2)
        }

    raise ValueError(f"无法获取网络信息: {model_type}")

def compare_networks(network_names: list = None) -> None:
    """比较多个网络架构"""
    if network_names is None:
        network_names = list(get_available_networks().keys())

    print("网络架构对比")
    print("=" * 80)
    print(f"{'网络名称':<20} {'参数量':<15} {'描述'}")
    print("-" * 80)

    for name in network_names:
        try:
            info = get_network_info(name)
            params = info['parameters']['total']
            desc = info['description']
            print(f"{name:<20} {params:<15,} {desc}")
        except Exception as e:
            print(f"{name:<20} {'错误':<15} {str(e)}")

# 便捷函数：智能网络选择
def recommend_network(
    dataset_size: int,
    complexity_preference: str = 'medium',
    performance_priority: str = 'accuracy'
) -> str:
    """
    根据数据集大小和需求推荐合适的网络

    Args:
        dataset_size: 数据集大小
        complexity_preference: 'simple' | 'medium' | 'complex'
        performance_priority: 'speed' | 'accuracy' | 'memory'

    Returns:
        推荐的网络名称
    """

    available = list(get_available_networks().keys())

    # 基于数据集大小的基本推荐
    if dataset_size < 50:
        candidates = [name for name in available if 'simple' in name or name == 'original']
    elif dataset_size < 200:
        candidates = [name for name in available if name in ['enhanced', 'original', 'wavelet_rcs']]
    else:
        candidates = available

    # 基于复杂度偏好过滤
    if complexity_preference == 'simple':
        candidates = [name for name in candidates if 'simple' in name or name == 'original']
    elif complexity_preference == 'complex':
        candidates = [name for name in candidates if name in ['enhanced', 'resnet_rcs', 'wavelet_rcs']]

    # 默认推荐
    if not candidates:
        candidates = ['original', 'enhanced']

    # 基于性能优先级选择
    if performance_priority == 'speed' and 'simple_fc' in candidates:
        return 'simple_fc'
    elif performance_priority == 'accuracy' and 'enhanced' in candidates:
        return 'enhanced'
    else:
        return candidates[0]

if __name__ == "__main__":
    # 演示使用
    print("=== 现代化网络架构接口演示 ===\n")

    # 显示可用网络
    print("可用网络:")
    networks = get_available_networks()
    for name, desc in networks.items():
        print(f"  {name}: {desc}")
    print()

    # 显示可用损失函数
    print("可用损失函数:")
    losses = get_available_losses()
    for name, desc in losses.items():
        print(f"  {name}: {desc}")
    print()

    # 网络对比
    compare_networks()
    print()

    # 智能推荐
    print("智能推荐:")
    rec1 = recommend_network(30, 'simple', 'speed')
    rec2 = recommend_network(150, 'medium', 'accuracy')
    rec3 = recommend_network(500, 'complex', 'accuracy')
    print(f"  小数据集(30样本) + 简单 + 速度优先: {rec1}")
    print(f"  中等数据集(150样本) + 中等 + 精度优先: {rec2}")
    print(f"  大数据集(500样本) + 复杂 + 精度优先: {rec3}")
    print()

    # 创建和测试网络
    test_networks = ['original', 'enhanced']
    if REGISTRY_AVAILABLE:
        test_networks.extend(['simple_fc', 'wavelet_rcs'])

    for net_name in test_networks[:2]:  # 只测试前两个避免输出过长
        try:
            print(f"测试网络: {net_name}")
            model = create_model(
                model_type=net_name,
                input_dim=9,
                output_shape=(91, 91, 2),
                use_log_output=False
            )

            # 测试前向传播
            x = torch.randn(2, 9)
            with torch.no_grad():
                output = model(x)
            print(f"  输入形状: {x.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
            print()

        except Exception as e:
            print(f"  错误: {e}")
            print()