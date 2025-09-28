"""
AutoEncoder-Wavelet RCS重建系统
实现RCS矩阵→隐空间→参数重建的完整流程
"""

__version__ = "0.1.0"
__author__ = "RCS Neural Network Team"

# 导入主要组件
from .models.cnn_autoencoder import WaveletAutoEncoder
from .utils.wavelet_transform import WaveletTransform
from .training.ae_trainer import AE_Trainer

__all__ = [
    'WaveletAutoEncoder',
    'WaveletTransform',
    'AE_Trainer'
]