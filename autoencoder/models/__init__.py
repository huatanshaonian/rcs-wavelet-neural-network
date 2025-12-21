"""
AutoEncoder模型定义
"""

from .cnn_autoencoder import WaveletAutoEncoder
from .direct_autoencoder import DirectAutoEncoder
from .mlp_autoencoder import WaveletMLPAutoEncoder, DirectMLPAutoEncoder
from .enhanced_cnn_autoencoder import EnhancedWaveletAutoEncoder, EnhancedDirectAutoEncoder
from .deep_autoencoder import DeepWaveletAutoEncoder, DeepDirectAutoEncoder
from .parameter_mapper import ParameterMapperFactory, MLPMapper as ParameterMapper
from .differentiable_wavelet_autoencoder import (
    DifferentiableWaveletAutoEncoder,
    DifferentiableWaveletMLPAutoEncoder
)
from .dual_branch_autoencoder import (
    DualBranchWaveletAutoEncoder,
    DualBranchWaveletMLPAutoEncoder
)
from .dual_branch_differentiable_autoencoder import (
    DualBranchDifferentiableWaveletAutoEncoder,
    DualBranchDifferentiableWaveletMLPAutoEncoder
)
from .dual_branch_differentiable_autoencoder_v2 import (
    DualBranchDifferentiableWaveletAutoEncoderV2,
    DualBranchDifferentiableWaveletMLPAutoEncoderV2
)
from .additive_dual_branch_autoencoder import (
    AdditiveDualBranchWaveletAutoEncoder,
    AdditiveDualBranchDirectAutoEncoder
)
from .additive_dual_branch_mlp import (
    AdditiveDualBranchWaveletMLPAutoEncoder,
    AdditiveDualBranchDirectMLPAutoEncoder
)

__all__ = [
    'WaveletAutoEncoder',
    'DirectAutoEncoder',
    'WaveletMLPAutoEncoder',
    'DirectMLPAutoEncoder',
    'EnhancedWaveletAutoEncoder',
    'EnhancedDirectAutoEncoder',
    'DeepWaveletAutoEncoder',
    'DeepDirectAutoEncoder',
    'ParameterMapper',
    'ParameterMapperFactory',
    # Differentiable模式
    'DifferentiableWaveletAutoEncoder',
    'DifferentiableWaveletMLPAutoEncoder',
    # 双分支模式
    'DualBranchWaveletAutoEncoder',
    'DualBranchWaveletMLPAutoEncoder',
    # 双分支可微分小波模式
    'DualBranchDifferentiableWaveletAutoEncoder',
    'DualBranchDifferentiableWaveletMLPAutoEncoder',
    # 双分支可微分小波模式 V2 (正确实现)
    'DualBranchDifferentiableWaveletAutoEncoderV2',
    'DualBranchDifferentiableWaveletMLPAutoEncoderV2',
    # 叠加型双分支模式 (输出叠加：高频+低频)
    'AdditiveDualBranchWaveletAutoEncoder',
    'AdditiveDualBranchDirectAutoEncoder',
    'AdditiveDualBranchWaveletMLPAutoEncoder',
    'AdditiveDualBranchDirectMLPAutoEncoder'
]