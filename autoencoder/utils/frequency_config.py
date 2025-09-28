"""
频率配置工具模块
提供创建不同频率配置的AutoEncoder系统的便捷方法
"""

import torch
from typing import Dict, Tuple, Any

# 使用绝对导入避免相对导入问题
import sys
import os

# 导入Unicode修复工具，支持可爱的Unicode字符 ✨
try:
    from unicode_fix import fix_unicode_output
    fix_unicode_output()
except ImportError:
    # 如果导入失败，使用简单的编码修复
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.cnn_autoencoder import WaveletAutoEncoder, ParameterMapper
from utils.wavelet_transform import WaveletTransform
from utils.data_adapters import RCS_DataAdapter


class FrequencyConfig:
    """频率配置类，定义不同频率设置的参数"""

    # 预定义配置
    CONFIGS = {
        '2freq': {
            'num_frequencies': 2,
            'frequency_labels': ['1.5GHz', '3GHz'],
            'description': '当前标准配置 (1.5GHz + 3GHz)'
        },
        '3freq': {
            'num_frequencies': 3,
            'frequency_labels': ['1.5GHz', '3GHz', '6GHz'],
            'description': '6GHz扩展配置 (1.5GHz + 3GHz + 6GHz)'
        }
    }

    def __init__(self, config_name: str = '2freq'):
        """
        初始化频率配置

        Args:
            config_name: 配置名称 ('2freq' 或 '3freq')
        """
        if config_name not in self.CONFIGS:
            raise ValueError(f"不支持的配置: {config_name}. 支持的配置: {list(self.CONFIGS.keys())}")

        self.config_name = config_name
        self.config = self.CONFIGS[config_name].copy()

        # 计算派生参数
        self.config['input_channels'] = self.config['num_frequencies'] * 4  # 4个小波频带
        self.config['wavelet_bands'] = 4

        print(f"初始化频率配置: {config_name}")
        print(f"描述: {self.config['description']}")
        print(f"频率数量: {self.config['num_frequencies']}")
        print(f"频率标签: {self.config['frequency_labels']}")
        print(f"输入通道数: {self.config['input_channels']}")

    def get_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return self.config.copy()

    def create_autoencoder(self,
                          latent_dim: int = 256,
                          dropout_rate: float = 0.2) -> WaveletAutoEncoder:
        """
        创建对应配置的AutoEncoder

        Args:
            latent_dim: 隐空间维度
            dropout_rate: Dropout比率

        Returns:
            WaveletAutoEncoder实例
        """
        ae = WaveletAutoEncoder(
            latent_dim=latent_dim,
            num_frequencies=self.config['num_frequencies'],
            wavelet_bands=self.config['wavelet_bands'],
            dropout_rate=dropout_rate
        )

        print(f"创建{self.config_name}配置的AutoEncoder:")
        print(f"  - 隐空间维度: {latent_dim}")
        print(f"  - 输入通道数: {self.config['input_channels']}")
        print(f"  - 参数量: {ae.get_parameter_count()['total']:,}")

        return ae

    def create_wavelet_transform(self,
                               wavelet: str = 'db4',
                               mode: str = 'symmetric') -> WaveletTransform:
        """
        创建对应配置的小波变换器

        Args:
            wavelet: 小波基函数
            mode: 边界处理模式

        Returns:
            WaveletTransform实例
        """
        wt = WaveletTransform(
            wavelet=wavelet,
            mode=mode,
            num_frequencies=self.config['num_frequencies']
        )

        print(f"创建{self.config_name}配置的小波变换器:")
        print(f"  - 小波类型: {wavelet}")
        print(f"  - 边界模式: {mode}")
        print(f"  - 频率数量: {self.config['num_frequencies']}")

        return wt

    def create_data_adapter(self,
                          normalize: bool = True,
                          log_transform: bool = False) -> RCS_DataAdapter:
        """
        创建对应配置的数据适配器

        Args:
            normalize: 是否标准化
            log_transform: 是否对数变换

        Returns:
            RCS_DataAdapter实例
        """
        adapter = RCS_DataAdapter(
            normalize=normalize,
            log_transform=log_transform,
            expected_frequencies=self.config['num_frequencies']
        )

        print(f"创建{self.config_name}配置的数据适配器:")
        print(f"  - 标准化: {normalize}")
        print(f"  - 对数变换: {log_transform}")
        print(f"  - 预期频率数: {self.config['num_frequencies']}")

        return adapter

    def create_parameter_mapper(self,
                              param_dim: int = 9,
                              latent_dim: int = 256,
                              hidden_dims: list = [128, 256, 512],
                              dropout_rate: float = 0.2) -> ParameterMapper:
        """
        创建参数映射器 (与频率配置无关，但为了API一致性提供)

        Args:
            param_dim: 参数维度
            latent_dim: 隐空间维度
            hidden_dims: 隐藏层维度
            dropout_rate: Dropout比率

        Returns:
            ParameterMapper实例
        """
        mapper = ParameterMapper(
            param_dim=param_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

        print(f"创建参数映射器:")
        print(f"  - 参数维度: {param_dim}")
        print(f"  - 隐空间维度: {latent_dim}")
        print(f"  - 隐藏层: {hidden_dims}")
        print(f"  - 参数量: {mapper.get_parameter_count():,}")

        return mapper


def create_autoencoder_system(config_name: str = '2freq',
                            latent_dim: int = 256,
                            dropout_rate: float = 0.2,
                            wavelet: str = 'db4',
                            normalize: bool = True) -> Dict[str, Any]:
    """
    一键创建完整的AutoEncoder系统

    Args:
        config_name: 频率配置名称
        latent_dim: 隐空间维度
        dropout_rate: Dropout比率
        wavelet: 小波类型
        normalize: 是否标准化数据

    Returns:
        包含所有组件的字典
    """
    print(f"=== 创建{config_name}配置的AutoEncoder系统 ===")

    # 创建配置
    freq_config = FrequencyConfig(config_name)

    # 创建所有组件
    autoencoder = freq_config.create_autoencoder(latent_dim, dropout_rate)
    wavelet_transform = freq_config.create_wavelet_transform(wavelet)
    data_adapter = freq_config.create_data_adapter(normalize)
    parameter_mapper = freq_config.create_parameter_mapper(latent_dim=latent_dim)

    system = {
        'config': freq_config,
        'autoencoder': autoencoder,
        'wavelet_transform': wavelet_transform,
        'data_adapter': data_adapter,
        'parameter_mapper': parameter_mapper,
        'config_info': freq_config.get_info()
    }

    print("✅ AutoEncoder系统创建完成!")
    print(f"配置信息: {system['config_info']}")

    return system


def test_frequency_configs():
    """测试不同频率配置"""
    print("=== 频率配置测试 ===")

    # 测试2频率配置
    print("\n--- 测试2频率配置 ---")
    system_2freq = create_autoencoder_system('2freq')

    # 创建测试数据
    batch_size = 4
    rcs_2freq = torch.randn(batch_size, 91, 91, 2)
    params = torch.randn(batch_size, 9)

    # 测试数据流
    wt_2freq = system_2freq['wavelet_transform']
    ae_2freq = system_2freq['autoencoder']

    wavelet_coeffs_2freq = wt_2freq.forward_transform(rcs_2freq)
    recon_coeffs_2freq, latent_2freq = ae_2freq(wavelet_coeffs_2freq)
    recon_rcs_2freq = wt_2freq.inverse_transform(recon_coeffs_2freq)

    print(f"2频率数据流测试:")
    print(f"  RCS {rcs_2freq.shape} → 小波系数 {wavelet_coeffs_2freq.shape}")
    print(f"  → 隐空间 {latent_2freq.shape} → 重建 {recon_rcs_2freq.shape}")

    # 测试3频率配置
    print("\n--- 测试3频率配置 ---")
    system_3freq = create_autoencoder_system('3freq')

    # 创建测试数据
    rcs_3freq = torch.randn(batch_size, 91, 91, 3)

    # 测试数据流
    wt_3freq = system_3freq['wavelet_transform']
    ae_3freq = system_3freq['autoencoder']

    wavelet_coeffs_3freq = wt_3freq.forward_transform(rcs_3freq)
    recon_coeffs_3freq, latent_3freq = ae_3freq(wavelet_coeffs_3freq)
    recon_rcs_3freq = wt_3freq.inverse_transform(recon_coeffs_3freq)

    print(f"3频率数据流测试:")
    print(f"  RCS {rcs_3freq.shape} → 小波系数 {wavelet_coeffs_3freq.shape}")
    print(f"  → 隐空间 {latent_3freq.shape} → 重建 {recon_rcs_3freq.shape}")

    # 验证隐空间维度一致性
    print(f"\n--- 兼容性验证 ---")
    print(f"隐空间维度一致性: {latent_2freq.shape[1] == latent_3freq.shape[1]}")
    print("✅ 不同频率配置可以使用相同的参数映射器")

    # 测试参数映射
    mapper = system_2freq['parameter_mapper']  # 可以复用
    mapped_latent_2freq = mapper(params)
    mapped_latent_3freq = mapper(params)  # 同样的参数映射器

    pred_rcs_2freq = wt_2freq.inverse_transform(ae_2freq.decode(mapped_latent_2freq))
    pred_rcs_3freq = wt_3freq.inverse_transform(ae_3freq.decode(mapped_latent_3freq))

    print(f"端到端预测测试:")
    print(f"  参数 {params.shape} → 2频率预测 {pred_rcs_2freq.shape}")
    print(f"  参数 {params.shape} → 3频率预测 {pred_rcs_3freq.shape}")

    return True


if __name__ == "__main__":
    test_frequency_configs()