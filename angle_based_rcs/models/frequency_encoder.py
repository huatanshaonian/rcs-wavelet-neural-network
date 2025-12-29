"""
频率编码器 (FrequencyEncoder)

使用one-hot编码将频率索引编码为离散向量，避免网络学习不合理的频率插值。
同时保留学习通用RCS物理规律的能力（跨频率共享参数）。

设计原理：
- 频率作为离散状态（避免学习频率间的平滑插值）
- 0: 1.5GHz → [1, 0, 0]
- 1: 3.0GHz → [0, 1, 0]
- 2: 6.0GHz → [0, 0, 1]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union


class FrequencyEncoder(nn.Module):
    """
    频率编码器：将频率索引编码为one-hot向量

    参数：
        num_frequencies (int): 频率数量 (默认: 3)
                               - 2: 1.5GHz + 3GHz
                               - 3: 1.5GHz + 3GHz + 6GHz

    输入：
        freq_idx: [B,] 张量，频率索引 (0/1/2)
                  - 0: 1.5GHz
                  - 1: 3.0GHz
                  - 2: 6.0GHz (如果num_frequencies=3)

    输出：
        freq_onehot: [B, num_frequencies] 张量，one-hot编码

    示例：
        >>> encoder = FrequencyEncoder(num_frequencies=3)
        >>> freq_idx = torch.tensor([0, 1, 2, 1])
        >>> onehot = encoder(freq_idx)
        >>> print(onehot)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]])
    """

    def __init__(self, num_frequencies: int = 3):
        super().__init__()

        if num_frequencies not in [2, 3]:
            raise ValueError(f"num_frequencies必须是2或3，当前为{num_frequencies}")

        self.num_frequencies = num_frequencies
        self.output_dim = num_frequencies

        # 频率映射（用于可视化和调试）
        if num_frequencies == 2:
            self.frequency_map = {0: '1.5GHz', 1: '3.0GHz'}
        else:  # num_frequencies == 3
            self.frequency_map = {0: '1.5GHz', 1: '3.0GHz', 2: '6.0GHz'}

    def forward(self,
                freq_idx: Union[torch.Tensor, int, list, np.ndarray]) -> torch.Tensor:
        """
        前向传播：编码频率索引

        参数：
            freq_idx: 频率索引，可以是标量、列表或张量
                     值必须在 [0, num_frequencies-1] 范围内

        返回：
            freq_onehot: [B, num_frequencies] one-hot编码
        """
        # 转换为张量（兼容多种输入格式）
        if not isinstance(freq_idx, torch.Tensor):
            if isinstance(freq_idx, (int, float)):
                freq_idx = torch.tensor([freq_idx], dtype=torch.long)
            else:
                freq_idx = torch.tensor(freq_idx, dtype=torch.long)
        else:
            freq_idx = freq_idx.long()

        # 确保是1维张量
        if freq_idx.dim() == 0:
            freq_idx = freq_idx.unsqueeze(0)

        # 验证索引范围
        if torch.any(freq_idx < 0) or torch.any(freq_idx >= self.num_frequencies):
            raise ValueError(
                f"频率索引必须在 [0, {self.num_frequencies-1}] 范围内，"
                f"当前范围: [{freq_idx.min()}, {freq_idx.max()}]"
            )

        batch_size = freq_idx.shape[0]

        # One-hot编码
        freq_onehot = torch.zeros(batch_size, self.num_frequencies,
                                  dtype=torch.float32, device=freq_idx.device)
        freq_onehot.scatter_(1, freq_idx.unsqueeze(1), 1.0)

        return freq_onehot

    def get_frequency_label(self, freq_idx: int) -> str:
        """获取频率标签（用于可视化）"""
        return self.frequency_map.get(freq_idx, f'Unknown({freq_idx})')

    def extra_repr(self) -> str:
        """打印模块信息"""
        freq_labels = ', '.join([f'{k}:{v}' for k, v in self.frequency_map.items()])
        return f'num_frequencies={self.num_frequencies}, map={{freq_labels}}'


if __name__ == "__main__":
    print("=" * 80)
    print("FrequencyEncoder 测试")
    print("=" * 80)

    # 测试1: 基本功能 (3频率)
    print("\n[Test 1] 基本功能测试 (3频率)")
    encoder = FrequencyEncoder(num_frequencies=3)
    print(f"编码器: {encoder}")
    print(f"输出维度: {encoder.output_dim}")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters())} (应为0)")
    print(f"频率映射: {encoder.frequency_map}")

    # 测试2: 单点编码
    print("\n[Test 2] 单点编码")
    for idx in [0, 1, 2]:
        freq_idx = torch.tensor([idx])
        onehot = encoder(freq_idx)
        label = encoder.get_frequency_label(idx)
        print(f"  索引{idx} ({label}): {onehot.squeeze().tolist()}")

    # 测试3: 批量编码
    print("\n[Test 3] 批量编码")
    freq_idx_batch = torch.tensor([0, 1, 2, 1, 0])
    onehot_batch = encoder(freq_idx_batch)
    print(f"输入索引: {freq_idx_batch.tolist()}")
    print(f"输出形状: {onehot_batch.shape} (期望: [5, 3])")
    print(f"One-hot编码:")
    for i, (idx, oh) in enumerate(zip(freq_idx_batch, onehot_batch)):
        label = encoder.get_frequency_label(idx.item())
        print(f"  样本{i}: 索引{idx.item()} ({label}) → {oh.tolist()}")

    # 测试4: 2频率配置
    print("\n[Test 4] 2频率配置")
    encoder_2freq = FrequencyEncoder(num_frequencies=2)
    print(f"频率映射: {encoder_2freq.frequency_map}")
    freq_idx_2 = torch.tensor([0, 1, 0, 1])
    onehot_2 = encoder_2freq(freq_idx_2)
    print(f"输入索引: {freq_idx_2.tolist()}")
    print(f"输出形状: {onehot_2.shape} (期望: [4, 2])")
    print(f"One-hot编码:")
    for idx, oh in zip(freq_idx_2, onehot_2):
        label = encoder_2freq.get_frequency_label(idx.item())
        print(f"  索引{idx.item()} ({label}) → {oh.tolist()}")

    # 测试5: 输入格式兼容性
    print("\n[Test 5] 输入格式兼容性")
    test_cases = [
        (0, "标量"),
        ([0, 1, 2], "列表"),
        (np.array([0, 1, 2]), "numpy数组"),
        (torch.tensor([0, 1, 2]), "torch张量")
    ]
    for freq_in, desc in test_cases:
        onehot = encoder(freq_in)
        print(f"  {desc:10s}: 输出形状={onehot.shape}")

    # 测试6: 错误处理
    print("\n[Test 6] 错误处理")
    try:
        invalid_idx = torch.tensor([3])  # 超出范围
        encoder(invalid_idx)
        print("  ❌ 应该抛出ValueError")
    except ValueError as e:
        print(f"  ✅ 正确捕获错误: {str(e)[:60]}...")

    try:
        invalid_idx = torch.tensor([-1])  # 负数索引
        encoder(invalid_idx)
        print("  ❌ 应该抛出ValueError")
    except ValueError as e:
        print(f"  ✅ 正确捕获错误: {str(e)[:60]}...")

    # 测试7: 验证one-hot正交性
    print("\n[Test 7] 验证one-hot正交性")
    freq_idx_all = torch.arange(3)
    onehot_all = encoder(freq_idx_all)
    # 计算点积矩阵（应该是单位矩阵）
    dot_product = torch.mm(onehot_all, onehot_all.T)
    is_orthogonal = torch.allclose(dot_product, torch.eye(3))
    print(f"点积矩阵（应为单位矩阵）:")
    print(dot_product)
    print(f"正交性验证: {'✅ 通过' if is_orthogonal else '❌ 失败'}")

    print("\n" + "=" * 80)
    print("✅ FrequencyEncoder测试通过!")
    print("=" * 80)
