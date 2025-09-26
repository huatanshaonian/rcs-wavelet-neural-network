"""
可扩展的网络架构注册系统
支持轻松添加新的网络结构而无需修改核心代码
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import importlib
import inspect

@dataclass
class NetworkConfig:
    """网络配置基类"""
    input_dim: int = 9
    output_shape: Tuple[int, ...] = (91, 91, 2)
    use_log_output: bool = False

    def __post_init__(self):
        """验证配置"""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if len(self.output_shape) < 2:
            raise ValueError("output_shape must have at least 2 dimensions")

@dataclass
class LossConfig:
    """损失函数配置"""
    loss_type: str = 'mse'
    loss_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {'total': 1.0}

class BaseNetwork(nn.Module, ABC):
    """网络基类 - 所有网络都必须继承此类"""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_shape = config.output_shape
        self.use_log_output = config.use_log_output

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 必须实现"""
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """返回网络名称"""
        pass

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """返回网络描述"""
        pass

    @classmethod
    def get_default_config(cls) -> NetworkConfig:
        """返回默认配置"""
        return NetworkConfig()

    @classmethod
    def validate_config(cls, config: NetworkConfig) -> bool:
        """验证配置是否兼容此网络"""
        return True

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }

    def get_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        return {
            'name': self.get_name(),
            'description': self.get_description(),
            'config': self.config.__dict__,
            'parameters': self.get_parameter_count(),
            'output_shape': self.output_shape
        }

class BaseLoss(nn.Module, ABC):
    """损失函数基类"""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.loss_weights = config.loss_weights.copy()

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失 - 必须返回包含'total'键的字典"""
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """返回损失函数名称"""
        pass

    @classmethod
    def get_default_config(cls) -> LossConfig:
        """返回默认配置"""
        return LossConfig()

    def get_loss_components(self) -> List[str]:
        """获取损失组件名称列表"""
        return list(self.loss_weights.keys())

class NetworkRegistry:
    """网络架构注册中心"""

    _networks: Dict[str, type] = {}
    _losses: Dict[str, type] = {}

    @classmethod
    def register_network(cls, network_class: type):
        """注册网络类"""
        if not issubclass(network_class, BaseNetwork):
            raise ValueError(f"{network_class.__name__} must inherit from BaseNetwork")

        name = network_class.get_name()
        if name in cls._networks:
            print(f"Warning: Overriding existing network '{name}'")

        cls._networks[name] = network_class
        print(f"Registered network: {name} - {network_class.get_description()}")
        return network_class

    @classmethod
    def register_loss(cls, loss_class: type):
        """注册损失函数类"""
        if not issubclass(loss_class, BaseLoss):
            raise ValueError(f"{loss_class.__name__} must inherit from BaseLoss")

        name = loss_class.get_name()
        if name in cls._losses:
            print(f"Warning: Overriding existing loss '{name}'")

        cls._losses[name] = loss_class
        print(f"Registered loss: {name}")
        return loss_class

    @classmethod
    def create_network(cls, network_name: str, config: Optional[NetworkConfig] = None) -> BaseNetwork:
        """创建网络实例"""
        if network_name not in cls._networks:
            available = list(cls._networks.keys())
            raise ValueError(f"Unknown network '{network_name}'. Available: {available}")

        network_class = cls._networks[network_name]

        if config is None:
            config = network_class.get_default_config()

        if not network_class.validate_config(config):
            raise ValueError(f"Invalid config for network '{network_name}'")

        return network_class(config)

    @classmethod
    def create_loss(cls, loss_name: str, config: Optional[LossConfig] = None) -> BaseLoss:
        """创建损失函数实例"""
        if loss_name not in cls._losses:
            available = list(cls._losses.keys())
            raise ValueError(f"Unknown loss '{loss_name}'. Available: {available}")

        loss_class = cls._losses[loss_name]

        if config is None:
            config = loss_class.get_default_config()

        return loss_class(config)

    @classmethod
    def list_networks(cls) -> Dict[str, str]:
        """列出所有可用网络"""
        return {name: cls_type.get_description() for name, cls_type in cls._networks.items()}

    @classmethod
    def list_losses(cls) -> List[str]:
        """列出所有可用损失函数"""
        return list(cls._losses.keys())

    @classmethod
    def get_network_info(cls, network_name: str) -> Dict[str, Any]:
        """获取网络详细信息"""
        if network_name not in cls._networks:
            raise ValueError(f"Unknown network '{network_name}'")

        network_class = cls._networks[network_name]
        config = network_class.get_default_config()
        temp_network = network_class(config)

        return temp_network.get_info()

    @classmethod
    def auto_discover_networks(cls, module_names: List[str]):
        """自动发现并注册网络"""
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)

                # 查找所有BaseNetwork的子类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseNetwork) and
                        obj != BaseNetwork and
                        not inspect.isabstract(obj)):
                        cls.register_network(obj)

                    if (issubclass(obj, BaseLoss) and
                        obj != BaseLoss and
                        not inspect.isabstract(obj)):
                        cls.register_loss(obj)

            except ImportError as e:
                print(f"Warning: Could not import module '{module_name}': {e}")

# 装饰器用于简化注册
def register_network(cls):
    """装饰器：注册网络类"""
    return NetworkRegistry.register_network(cls)

def register_loss(cls):
    """装饰器：注册损失函数类"""
    return NetworkRegistry.register_loss(cls)

# 便捷函数
def create_network(network_name: str, **config_kwargs) -> BaseNetwork:
    """便捷创建网络函数"""
    config = NetworkConfig(**config_kwargs) if config_kwargs else None
    return NetworkRegistry.create_network(network_name, config)

def create_loss(loss_name: str, **config_kwargs) -> BaseLoss:
    """便捷创建损失函数"""
    config = LossConfig(**config_kwargs) if config_kwargs else None
    return NetworkRegistry.create_loss(loss_name, config)

def list_available_networks():
    """列出可用网络"""
    networks = NetworkRegistry.list_networks()
    print("Available Networks:")
    print("=" * 50)
    for name, desc in networks.items():
        print(f"{name:20} - {desc}")

def list_available_losses():
    """列出可用损失函数"""
    losses = NetworkRegistry.list_losses()
    print("Available Loss Functions:")
    print("=" * 30)
    for loss in losses:
        print(f"  - {loss}")