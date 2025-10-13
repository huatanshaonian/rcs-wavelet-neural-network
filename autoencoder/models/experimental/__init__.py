"""
实验性AutoEncoder模型

这些模型是在开发过程中的实验性版本，目前未被主系统使用。
保留它们以供参考和未来可能的改进。

如需使用这些模型，请：
1. 重命名以符合命名规范（见 ../MODEL_INVENTORY.md）
2. 创建对应的Direct版本（如果只有Wavelet版本）
3. 在 frequency_config.py 中添加调用逻辑
4. 充分测试后再集成到主系统
"""

from .correct_cnn_autoencoder import CorrectCNNAutoEncoder
from .deep_cnn_autoencoder import DeepCNNAutoEncoder
from .efficient_cnn_autoencoder import EfficientCNNAutoEncoder
from .micro_latent_autoencoder import MicroLatentAutoEncoder

__all__ = [
    'CorrectCNNAutoEncoder',
    'DeepCNNAutoEncoder',
    'EfficientCNNAutoEncoder',
    'MicroLatentAutoEncoder'
]
