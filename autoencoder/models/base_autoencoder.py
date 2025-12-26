"""
BaseAutoEncoder - 所有 AutoEncoder 的基类

提供物理约束层的统一管理
"""

import torch.nn as nn


class BaseAutoEncoder(nn.Module):
    """
    AutoEncoder 基类

    提供统一的输出激活层管理（用于物理约束）

    Args:
        output_activation: 输出激活函数类型
            - None: 不使用输出激活（默认）
            - 'softplus': 使用 Softplus 激活（RCS 非负约束）

    使用示例:
        >>> class MyAutoEncoder(BaseAutoEncoder):
        ...     def __init__(self, latent_dim=256, output_activation=None):
        ...         super().__init__(output_activation=output_activation)
        ...         # ... 定义 encoder/decoder ...
        ...
        ...     def decode(self, latent):
        ...         x = self.decoder(latent)
        ...         x = self.apply_output_activation(x)  # ← 应用输出激活
        ...         return x
    """

    def __init__(self, output_activation=None):
        """
        初始化基类

        Args:
            output_activation: 输出激活类型 (None 或 'softplus')
        """
        super().__init__()

        # ✅ 只在需要时创建激活层（关闭时不创建任何额外属性）
        if output_activation == 'softplus':
            self.output_activation = nn.Softplus(beta=1)
            print(f"  [BaseAutoEncoder] 创建 Softplus 输出激活层 (beta=1)")
        elif output_activation is not None:
            raise ValueError(f"不支持的 output_activation: {output_activation}，目前仅支持 None 或 'softplus'")

    def apply_output_activation(self, x):
        """
        应用输出激活层（如果存在）

        Args:
            x: 输入张量（通常是 decoder 的原始输出）

        Returns:
            应用激活后的张量（如果没有激活层，返回原始输入）

        注意：
            - 如果模型创建时 output_activation=None，此方法是恒等映射
            - 代码路径与未使用基类时完全一致（无额外开销）
        """
        if hasattr(self, 'output_activation'):
            return self.output_activation(x)
        return x

    def has_output_activation(self):
        """
        检查是否启用了输出激活层

        Returns:
            bool: True 表示启用，False 表示未启用
        """
        return hasattr(self, 'output_activation')

    def get_output_activation_type(self):
        """
        获取输出激活层类型

        Returns:
            str 或 None: 激活类型（'softplus' 或 None）
        """
        if hasattr(self, 'output_activation'):
            if isinstance(self.output_activation, nn.Softplus):
                return 'softplus'
        return None
