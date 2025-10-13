"""
AutoEncoder模型定义
"""

from .cnn_autoencoder import WaveletAutoEncoder, ParameterMapper
from .direct_autoencoder import DirectAutoEncoder
from .mlp_autoencoder import WaveletMLPAutoEncoder, DirectMLPAutoEncoder
from .enhanced_cnn_autoencoder import EnhancedWaveletAutoEncoder, EnhancedDirectAutoEncoder
from .deep_autoencoder import DeepWaveletAutoEncoder, DeepDirectAutoEncoder
from .parameter_mapper import ParameterMapperFactory

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
    'ParameterMapperFactory'
]