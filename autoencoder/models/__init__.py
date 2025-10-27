"""
AutoEncoder模型定义
"""

from .cnn_autoencoder import WaveletAutoEncoder, ParameterMapper
from .direct_autoencoder import DirectAutoEncoder
from .mlp_autoencoder import WaveletMLPAutoEncoder, DirectMLPAutoEncoder
from .enhanced_cnn_autoencoder import EnhancedWaveletAutoEncoder, EnhancedDirectAutoEncoder
from .deep_autoencoder import DeepWaveletAutoEncoder, DeepDirectAutoEncoder
from .sine_cnn_autoencoder import SinWaveletAutoEncoder, SinDirectAutoEncoder
from .sine_mlp_autoencoder import SinWaveletMLPAutoEncoder, SinDirectMLPAutoEncoder
from .parameter_mapper import ParameterMapperFactory
from .differentiable_wavelet_autoencoder import (
    DifferentiableWaveletAutoEncoder,
    DifferentiableWaveletMLPAutoEncoder,
    DifferentiableSineWaveletMLPAutoEncoder
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
    'SinWaveletAutoEncoder',
    'SinDirectAutoEncoder',
    'SinWaveletMLPAutoEncoder',
    'SinDirectMLPAutoEncoder',
    'ParameterMapper',
    'ParameterMapperFactory',
    # Differentiable模式
    'DifferentiableWaveletAutoEncoder',
    'DifferentiableWaveletMLPAutoEncoder',
    'DifferentiableSineWaveletMLPAutoEncoder'
]