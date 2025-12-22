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

from models.cnn_autoencoder import WaveletAutoEncoder
from models.parameter_mapper import MLPMapper as ParameterMapper
from models.direct_autoencoder import DirectAutoEncoder
from models.mlp_autoencoder import WaveletMLPAutoEncoder, DirectMLPAutoEncoder
from models.enhanced_cnn_autoencoder import EnhancedWaveletAutoEncoder, EnhancedDirectAutoEncoder
from models.deep_autoencoder import DeepWaveletAutoEncoder, DeepDirectAutoEncoder
from models.differentiable_wavelet_autoencoder import (
    DifferentiableWaveletAutoEncoder,
    DifferentiableWaveletMLPAutoEncoder
)
from models.dual_branch_autoencoder import (
    DualBranchWaveletAutoEncoder,
    DualBranchWaveletMLPAutoEncoder
)
from models.dual_branch_differentiable_autoencoder import (
    DualBranchDifferentiableWaveletAutoEncoder,
    DualBranchDifferentiableWaveletMLPAutoEncoder
)
from models.dual_branch_differentiable_autoencoder_v2 import (
    DualBranchDifferentiableWaveletAutoEncoderV2,
    DualBranchDifferentiableWaveletMLPAutoEncoderV2
)
from models.additive_dual_branch_autoencoder import (
    AdditiveDualBranchWaveletAutoEncoder,
    AdditiveDualBranchDirectAutoEncoder
)
from models.additive_dual_branch_mlp import (
    AdditiveDualBranchWaveletMLPAutoEncoder,
    AdditiveDualBranchDirectMLPAutoEncoder
)
from utils.correct_wavelet_transform import CorrectWaveletTransform as WaveletTransform
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
        self.config['wavelet_input_channels'] = self.config['num_frequencies'] * 4  # Wavelet模式: 频率×4个小波频带
        self.config['direct_input_channels'] = self.config['num_frequencies']       # Direct模式: 等于频率数
        self.config['wavelet_bands'] = 4

        print(f"初始化频率配置: {config_name}")
        print(f"描述: {self.config['description']}")
        print(f"频率数量: {self.config['num_frequencies']}")
        print(f"频率标签: {self.config['frequency_labels']}")
        # 不再显示通用的input_channels，因为它依赖于mode

    def get_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return self.config.copy()

    def create_autoencoder(self,
                          latent_dim: int = 32,
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
            dropout_rate=dropout_rate,
            input_size=49  # db4小波变换后的尺寸
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
                          mode: str = 'direct',
                          db_transform: bool = False,
                          normalization_method: str = 'zscore') -> RCS_DataAdapter:
        """
        创建对应配置的数据适配器

        Args:
            normalize: 是否标准化（向后兼容）
            mode: 数据模式 ('direct' or 'wavelet')
            db_transform: 是否使用dB变换（手动控制）
            normalization_method: 标准化方法 ('none', 'zscore', 'minmax')

        Returns:
            RCS_DataAdapter实例
        """
        adapter = RCS_DataAdapter(
            normalize=normalize,
            mode=mode,
            expected_frequencies=self.config['num_frequencies'],
            db_transform=db_transform,
            normalization_method=normalization_method
        )

        print(f"创建{self.config_name}配置的数据适配器:")
        print(f"  - 标准化方法: {normalization_method}")
        print(f"  - dB变换: {db_transform}")
        print(f"  - 模式: {mode}")
        print(f"  - 预期频率数: {self.config['num_frequencies']}")

        return adapter

    def create_parameter_mapper(self,
                              param_dim: int = 9,
                              latent_dim: int = 32,
                              hidden_dims: list = None,
                              dropout_rate: float = 0.3,
                              activation: str = 'relu',
                              use_adaptive: bool = False,
                              max_ratio: int = 4) -> ParameterMapper:
        """
        创建参数映射器 (与频率配置无关，但为了API一致性提供)

        Args:
            param_dim: 参数维度
            latent_dim: 隐空间维度
            hidden_dims: 隐藏层维度列表（如果use_adaptive=True则忽略）
            dropout_rate: Dropout比率
            activation: 激活函数类型（relu/gelu/sin/swish/mish/tanh）
            use_adaptive: 是否使用自适应隐藏层维度
            max_ratio: 自适应模式下每级最大压缩比

        Returns:
            ParameterMapper实例
        """
        mapper = ParameterMapper(
            param_dim=param_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            use_adaptive=use_adaptive,
            max_ratio=max_ratio
        )

        info = mapper.get_model_info()
        print(f"创建参数映射器:")
        print(f"  - 参数维度: {param_dim}")
        print(f"  - 隐空间维度: {latent_dim}")
        print(f"  - 网络结构: {info['structure']}")
        print(f"  - 激活函数: {activation}")
        print(f"  - 自适应层: {'是' if use_adaptive else '否'}")
        print(f"  - 参数量: {mapper.get_parameter_count():,}")

        return mapper


def create_autoencoder_system(config_name: str = '2freq',
                            latent_dim: int = 32,
                            dropout_rate: float = 0.2,
                            wavelet: str = 'db4',
                            normalize: bool = True,
                            mode: str = 'wavelet',
                            architecture: str = 'cnn',
                            use_channel_attention: bool = False,
                            activation: str = 'relu',
                            db_transform: bool = False,
                            normalization_method: str = 'zscore',
                            mapper_activation: str = None,
                            mapper_use_adaptive: bool = False,
                            mapper_hidden_dims: list = None,
                            # 叠加型双分支专用参数
                            activation_encoder: str = None,
                            activation_high: str = None,
                            activation_smooth: str = None,
                            learnable_weights: bool = False,
                            alpha_high: float = 0.5,
                            alpha_smooth: float = 0.5) -> Dict[str, Any]:
    """
    一键创建完整的AutoEncoder系统

    Args:
        config_name: 频率配置名称
        latent_dim: 隐空间维度
        dropout_rate: Dropout比率
        wavelet: 小波类型
        normalize: 是否标准化数据（向后兼容）
        mode: 'wavelet' 或 'direct' 模式
        architecture: 'cnn' 或 'mlp' 架构
        use_channel_attention: 是否在输入层使用通道注意力机制 (默认: False)
        activation: 激活函数类型 ('relu', 'sin', 'gelu', 'swish'等，默认: 'relu')
        db_transform: 是否使用dB变换（手动控制，默认: False）
        normalization_method: 标准化方法 ('none', 'zscore', 'minmax'，默认: 'zscore')
        mapper_activation: 参数映射器激活函数（默认None，使用AutoEncoder相同的activation）
        mapper_use_adaptive: 参数映射器是否使用自适应隐藏层（默认False）
        mapper_hidden_dims: 参数映射器固定隐藏层维度（仅当mapper_use_adaptive=False时生效）
        activation_encoder: 叠加型双分支Encoder激活函数（仅additive_dual_branch架构，默认None使用activation）
        activation_high: 叠加型双分支高频Decoder激活函数（仅additive_dual_branch架构，默认None使用'sin'）
        activation_smooth: 叠加型双分支低频Decoder激活函数（仅additive_dual_branch架构，默认None使用'tanh'）
        learnable_weights: 叠加型双分支是否学习权重（仅additive_dual_branch架构，默认False）
        alpha_high: 叠加型双分支高频权重（仅additive_dual_branch架构，默认0.5）
        alpha_smooth: 叠加型双分支低频权重（仅additive_dual_branch架构，默认0.5）

    Returns:
        包含所有组件的字典
    """
    mode_desc = "小波增强" if mode in ['wavelet', 'differentiable_wavelet'] else "直接处理"
    arch_desc = architecture.upper()
    print(f"=== 创建{config_name}配置的AutoEncoder系统 ({mode_desc}模式, {arch_desc}架构) ===")

    # 创建配置
    freq_config = FrequencyConfig(config_name)

    # 根据mode和architecture组合创建对应的AutoEncoder
    if mode == 'wavelet':
        # 小波模式
        wavelet_transform = freq_config.create_wavelet_transform(wavelet)

        # 获取实际的小波变换后尺寸（自适应不同小波基）
        wavelet_size = wavelet_transform.wavelet_size[0]  # 取高度（正方形）
        print(f"小波变换后尺寸: {wavelet_size}x{wavelet_size} (小波类型: {wavelet})")

        if architecture.lower() == 'mlp':
            # 小波 + MLP
            autoencoder = WaveletMLPAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                input_size=wavelet_size,  # 自适应小波尺寸
                activation=activation
            )
            print(f"使用 WaveletMLPAutoEncoder (激活函数: {activation})")
        elif architecture.lower() == 'enhanced_cnn':
            # 小波 + Enhanced CNN
            autoencoder = EnhancedWaveletAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                input_size=wavelet_size,  # 自适应小波尺寸
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "启用输入层注意力" if use_channel_attention else "仅中间层注意力"
            print(f"使用 EnhancedWaveletAutoEncoder (增强感受野CNN, {attention_status}, 激活函数: {activation})")
        elif architecture.lower() == 'deep_cnn':
            # 小波 + Deep CNN
            autoencoder = DeepWaveletAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                use_attention=True,
                input_size=wavelet_size,  # 自适应小波尺寸
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "输入层+中间层双注意力" if use_channel_attention else "仅中间层注意力"
            print(f"使用 DeepWaveletAutoEncoder (深度CNN, {attention_status}, 激活函数: {activation})")
        elif architecture.lower() in ('dual_branch_cnn', 'dual_branch'):
            # 小波 + 双分支CNN
            autoencoder = DualBranchWaveletAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                ll_ratio=0.7,  # LL分支占70%隐空间
                activation=activation
            )
            print(f"使用 DualBranchWaveletAutoEncoder (双分支CNN, LL和高频分离处理, 激活函数: {activation})")
        elif architecture.lower() == 'dual_branch_mlp':
            # 小波 + 双分支MLP
            autoencoder = DualBranchWaveletMLPAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                ll_ratio=0.7,  # LL分支占70%隐空间
                activation=activation
            )
            print(f"使用 DualBranchWaveletMLPAutoEncoder (双分支MLP, LL和高频分离处理, 激活函数: {activation})")
        elif architecture.lower() in ('additive_dual_branch_cnn', 'additive_dual_branch'):
            # 小波 + 叠加型双分支CNN
            _activation_encoder = activation_encoder if activation_encoder else activation
            _activation_high = activation_high if activation_high else 'sin'
            _activation_smooth = activation_smooth if activation_smooth else 'tanh'

            autoencoder = AdditiveDualBranchWaveletAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                input_size=wavelet_size,
                activation_encoder=_activation_encoder,
                activation_high=_activation_high,
                activation_smooth=_activation_smooth,
                learnable_weights=learnable_weights,
                alpha_high=alpha_high,
                alpha_smooth=alpha_smooth
            )
            weight_mode = "可学习权重" if learnable_weights else f"固定权重(α={alpha_high},β={alpha_smooth})"
            print(f"使用 AdditiveDualBranchWaveletAutoEncoder (叠加型双分支CNN, {weight_mode})")
            print(f"  Encoder激活: {_activation_encoder}, 高频Decoder: {_activation_high}, 低频Decoder: {_activation_smooth}")
        elif architecture.lower() == 'additive_dual_branch_mlp':
            # 小波 + 叠加型双分支MLP
            _activation_encoder = activation_encoder if activation_encoder else activation
            _activation_high = activation_high if activation_high else 'sin'
            _activation_smooth = activation_smooth if activation_smooth else 'relu'

            autoencoder = AdditiveDualBranchWaveletMLPAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                input_size=wavelet_size,
                activation_encoder=_activation_encoder,
                activation_high=_activation_high,
                activation_smooth=_activation_smooth,
                learnable_weights=learnable_weights,
                alpha_high=alpha_high,
                alpha_smooth=alpha_smooth
            )
            weight_mode = "可学习权重" if learnable_weights else f"固定权重(α={alpha_high},β={alpha_smooth})"
            print(f"使用 AdditiveDualBranchWaveletMLPAutoEncoder (叠加型双分支MLP, {weight_mode})")
            print(f"  Encoder激活: {_activation_encoder}, 高频Decoder: {_activation_high}, 低频Decoder: {_activation_smooth}")
        else:
            # 小波 + CNN (默认)
            autoencoder = WaveletAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                input_size=wavelet_size,  # 自适应小波尺寸
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "启用输入层通道注意力" if use_channel_attention else "关闭输入层通道注意力"
            print(f"使用 WaveletAutoEncoder (标准CNN, {attention_status}, 激活函数: {activation})")
    elif mode == 'direct':
        # 直接模式
        wavelet_transform = None

        if architecture.lower() == 'mlp':
            # 直接 + MLP
            autoencoder = DirectMLPAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                activation=activation
            )
            print(f"使用 DirectMLPAutoEncoder (激活函数: {activation})")
        elif architecture.lower() == 'enhanced_cnn':
            # 直接 + Enhanced CNN
            autoencoder = EnhancedDirectAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "启用输入层注意力" if use_channel_attention else "仅中间层注意力"
            print(f"使用 EnhancedDirectAutoEncoder (增强感受野CNN, {attention_status}, 激活函数: {activation})")
        elif architecture.lower() == 'deep_cnn':
            # 直接 + Deep CNN
            autoencoder = DeepDirectAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                use_attention=True,
                input_size=91,
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "输入层+中间层双注意力" if use_channel_attention else "仅中间层注意力"
            print(f"使用 DeepDirectAutoEncoder (深度CNN, {attention_status}, 激活函数: {activation})")
        elif architecture.lower() in ('additive_dual_branch_cnn', 'additive_dual_branch'):
            # 直接 + 叠加型双分支
            _activation_encoder = activation_encoder if activation_encoder else activation
            _activation_high = activation_high if activation_high else 'sin'
            _activation_smooth = activation_smooth if activation_smooth else 'tanh'

            autoencoder = AdditiveDualBranchDirectAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                input_size=91,
                activation_encoder=_activation_encoder,
                activation_high=_activation_high,
                activation_smooth=_activation_smooth,
                learnable_weights=learnable_weights,
                alpha_high=alpha_high,
                alpha_smooth=alpha_smooth
            )
            weight_mode = "可学习权重" if learnable_weights else f"固定权重(α={alpha_high},β={alpha_smooth})"
            print(f"使用 AdditiveDualBranchDirectAutoEncoder (叠加型双分支CNN, {weight_mode})")
            print(f"  Encoder激活: {_activation_encoder}, 高频Decoder: {_activation_high}, 低频Decoder: {_activation_smooth}")
        elif architecture.lower() == 'additive_dual_branch_mlp':
            # 直接 + 叠加型双分支MLP
            _activation_encoder = activation_encoder if activation_encoder else activation
            _activation_high = activation_high if activation_high else 'sin'
            _activation_smooth = activation_smooth if activation_smooth else 'relu'

            autoencoder = AdditiveDualBranchDirectMLPAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                input_size=91,
                activation_encoder=_activation_encoder,
                activation_high=_activation_high,
                activation_smooth=_activation_smooth,
                learnable_weights=learnable_weights,
                alpha_high=alpha_high,
                alpha_smooth=alpha_smooth
            )
            weight_mode = "可学习权重" if learnable_weights else f"固定权重(α={alpha_high},β={alpha_smooth})"
            print(f"使用 AdditiveDualBranchDirectMLPAutoEncoder (叠加型双分支MLP, {weight_mode})")
            print(f"  Encoder激活: {_activation_encoder}, 高频Decoder: {_activation_high}, 低频Decoder: {_activation_smooth}")
        else:
            # 直接 + CNN (默认)
            autoencoder = DirectAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "启用输入层通道注意力" if use_channel_attention else "关闭输入层通道注意力"
            print(f"使用 DirectAutoEncoder (标准CNN, {attention_status}, 激活函数: {activation})")
    elif mode == 'differentiable_wavelet':
        # 可微分小波模式（第三种模式）
        # 特点：小波变换集成在模型中，损失在RCS空间计算
        wavelet_transform = None  # differentiable模式不需要单独的小波变换

        # 获取小波尺寸（用于模型初始化）
        temp_wt = freq_config.create_wavelet_transform(wavelet)
        wavelet_size = temp_wt.wavelet_size[0]
        print(f"可微分小波尺寸: {wavelet_size}x{wavelet_size} (小波类型: {wavelet})")

        if architecture.lower() == 'mlp':
            # 可微分小波 + MLP
            autoencoder = DifferentiableWaveletMLPAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "启用输入层通道注意力" if use_channel_attention else "关闭输入层通道注意力"
            print(f"使用 DifferentiableWaveletMLPAutoEncoder (端到端训练, {attention_status}, 激活函数: {activation})")
        elif architecture.lower() in ('dual_branch_cnn', 'dual_branch'):
            # 可微分小波 + 双分支CNN V2 (默认使用V2正确实现)
            autoencoder = DualBranchDifferentiableWaveletAutoEncoderV2(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                ll_ratio=0.7,
                activation=activation
            )
            print(f"使用 DualBranchDifferentiableWaveletAutoEncoderV2 (双分支CNN V2 + 可微分小波, 正确对称架构, 激活函数: {activation})")
        elif architecture.lower() == 'dual_branch_mlp':
            # 可微分小波 + 双分支MLP V2 (默认使用V2正确实现)
            autoencoder = DualBranchDifferentiableWaveletMLPAutoEncoderV2(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                ll_ratio=0.7,
                activation=activation
            )
            print(f"使用 DualBranchDifferentiableWaveletMLPAutoEncoderV2 (双分支MLP V2 + 可微分小波, 正确对称架构, 激活函数: {activation})")
        elif architecture.lower() in ('dual_branch_cnn_v1', 'dual_branch_v1'):
            # 可微分小波 + 双分支CNN V1 (旧版，仅向后兼容)
            autoencoder = DualBranchDifferentiableWaveletAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                ll_ratio=0.7,
                activation=activation
            )
            print(f"⚠️ 使用 DualBranchDifferentiableWaveletAutoEncoder V1 (旧版，有架构缺陷，建议使用V2)")
        elif architecture.lower() == 'dual_branch_mlp_v1':
            # 可微分小波 + 双分支MLP V1 (旧版，仅向后兼容)
            autoencoder = DualBranchDifferentiableWaveletMLPAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                ll_ratio=0.7,
                activation=activation
            )
            print(f"⚠️ 使用 DualBranchDifferentiableWaveletMLPAutoEncoder V1 (旧版，有架构缺陷，建议使用V2)")
        else:
            # 可微分小波 + CNN (默认)
            autoencoder = DifferentiableWaveletAutoEncoder(
                latent_dim=latent_dim,
                num_frequencies=freq_config.config['num_frequencies'],
                wavelet_bands=freq_config.config['wavelet_bands'],
                dropout_rate=dropout_rate,
                wavelet_type=wavelet,
                input_size=wavelet_size,
                use_channel_attention=use_channel_attention,
                activation=activation
            )
            attention_status = "启用输入层通道注意力" if use_channel_attention else "关闭输入层通道注意力"
            print(f"使用 DifferentiableWaveletAutoEncoder (标准CNN, 端到端训练, {attention_status}, 激活函数: {activation})")
    else:
        raise ValueError(f"未知的模式: {mode}. 支持的模式: 'wavelet', 'direct', 'differentiable_wavelet'")

    data_adapter = freq_config.create_data_adapter(normalize, mode=mode if mode != 'differentiable_wavelet' else 'wavelet', db_transform=db_transform, normalization_method=normalization_method)

    # 创建参数映射器（支持独立配置）
    if mapper_activation is None:
        mapper_activation = activation  # 默认使用AutoEncoder相同的激活函数

    parameter_mapper = freq_config.create_parameter_mapper(
        latent_dim=latent_dim,
        activation=mapper_activation,
        use_adaptive=mapper_use_adaptive,
        hidden_dims=mapper_hidden_dims
    )

    # 打印模型参数信息
    ae_params = autoencoder.get_parameter_count()
    model_info = autoencoder.get_model_info()

    # 根据mode确定实际输入通道数
    if mode == 'wavelet':
        actual_input_channels = freq_config.config['wavelet_input_channels']
        input_shape_desc = f"[B, 49, 49, {actual_input_channels}]"
    elif mode == 'differentiable_wavelet':
        # differentiable模式输入/输出都是RCS
        actual_input_channels = freq_config.config['direct_input_channels']
        input_shape_desc = f"[B, 91, 91, {actual_input_channels}] (RCS空间)"
    else:
        actual_input_channels = freq_config.config['direct_input_channels']
        input_shape_desc = f"[B, 91, 91, {actual_input_channels}]"

    print(f"\n【模型配置】")
    print(f"  - AutoEncoder参数量: {ae_params['total']:,}")
    print(f"  - 隐空间维度: {latent_dim}")
    print(f"  - 输入通道数: {actual_input_channels} ({mode}模式)")
    print(f"  - 输入形状: {input_shape_desc}")

    # 显示网络结构（如果有）
    if 'fc_structure' in model_info:
        print(f"\n【网络结构】")
        print(f"  - FC层结构: {model_info['fc_structure']}")
        print(f"  - FC层数: {model_info['num_fc_layers']}")
    elif 'mlp_structure' in model_info:
        print(f"\n【网络结构】")
        print(f"  - MLP结构: {model_info['mlp_structure']}")
        print(f"  - MLP层数: {model_info['num_mlp_layers']}")

    if 'compression_ratio' in model_info:
        print(f"  - 总压缩比: {model_info['compression_ratio']}")

    # 构建完整的config_info（包含所有参数）
    config_info = freq_config.get_info()
    config_info.update({
        'mode': mode,
        'architecture': architecture,
        'activation': activation,
        'latent_dim': latent_dim,
        'dropout_rate': dropout_rate,
        'wavelet': wavelet,
        'normalize': normalize,
        'db_transform': db_transform,
        'normalization_method': normalization_method,
        'use_channel_attention': use_channel_attention
    })

    system = {
        'config': freq_config,
        'autoencoder': autoencoder,
        'wavelet_transform': wavelet_transform,
        'data_adapter': data_adapter,
        'parameter_mapper': parameter_mapper,
        'config_info': config_info,
        'mode': mode,
        'architecture': architecture
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
