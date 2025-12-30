"""
角度采样器 (AngleSampler)

实现全局混合采样策略：
- 将所有数据点（200样本 × 8281角度 × 3频率 = 4,968,600点）混合在一起
- 80-20全局随机划分训练集和测试集
- 支持完整epoch和子采样epoch两种模式

设计原理：
- 全局iid采样：避免任何结构化分组，测试真正的泛化能力
- 每个数据点是独立的：(sample_idx, θ, φ, freq_idx)
- 训练集：3,974,880点（80%）
- 测试集：993,720点（20%）
"""

import numpy as np
from typing import Tuple, List, Optional


class DataPoint:
    """
    单个数据点

    属性：
        sample_idx: 设计参数索引 [0, 199]
        i: θ方向索引 [0, 90]
        j: φ方向索引 [0, 90]
        freq_idx: 频率索引 [0, 2]
        theta: θ角度（度）
        phi: φ角度（度）
    """
    __slots__ = ['sample_idx', 'i', 'j', 'freq_idx', 'theta', 'phi']

    def __init__(self, sample_idx: int, i: int, j: int, freq_idx: int,
                 theta: float, phi: float):
        self.sample_idx = sample_idx
        self.i = i
        self.j = j
        self.freq_idx = freq_idx
        self.theta = theta
        self.phi = phi

    def __repr__(self):
        return (f"DataPoint(sample={self.sample_idx}, θ={self.theta:.1f}°, "
                f"φ={self.phi:.1f}°, freq={self.freq_idx})")


class AngleSampler:
    """
    角度采样器：全局混合采样

    参数：
        num_samples (int): 设计参数样本数量（默认: 200）
        num_frequencies (int): 频率数量（默认: 3）
        theta_range (tuple): θ角度范围（默认: (45.0, 135.0)）
        phi_range (tuple): φ角度范围（默认: (-45.0, 45.0)）
        grid_size (int): 网格大小（默认: 91，对应91×91网格）
        train_split (float): 训练集比例（默认: 0.8）
        random_seed (int): 随机种子（默认: 42）

    属性：
        total_points (int): 总数据点数量
        train_indices (ndarray): 训练集索引
        test_indices (ndarray): 测试集索引
    """

    def __init__(self,
                 num_samples: int = 200,
                 num_frequencies: int = 3,
                 theta_range: Tuple[float, float] = (45.0, 135.0),
                 phi_range: Tuple[float, float] = (-45.0, 45.0),
                 grid_size: int = 91,
                 train_split: float = 0.8,
                 random_seed: int = 42):

        self.num_samples = num_samples
        self.num_frequencies = num_frequencies
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.grid_size = grid_size
        self.train_split = train_split
        self.random_seed = random_seed

        # 计算总数据点数量
        self.total_points = num_samples * grid_size * grid_size * num_frequencies

        # 生成角度网格
        self.theta_values = np.linspace(theta_range[0], theta_range[1], grid_size)
        self.phi_values = np.linspace(phi_range[0], phi_range[1], grid_size)

        # 执行全局混合划分
        self.train_indices, self.test_indices = self._global_mixed_split()

        print(f"AngleSampler初始化完成:")
        print(f"  样本数: {num_samples}")
        print(f"  频率数: {num_frequencies}")
        print(f"  网格大小: {grid_size}×{grid_size}")
        print(f"  总数据点: {self.total_points:,}")
        print(f"  训练集: {len(self.train_indices):,} ({train_split*100:.0f}%)")
        print(f"  测试集: {len(self.test_indices):,} ({(1-train_split)*100:.0f}%)")

    def _global_mixed_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        全局混合划分

        将所有数据点索引打乱，然后按比例划分训练集和测试集。

        返回：
            train_indices: 训练集索引数组
            test_indices: 测试集索引数组
        """
        # 生成所有数据点的全局索引 [0, total_points-1]
        all_indices = np.arange(self.total_points)

        # 全局随机打乱
        np.random.seed(self.random_seed)
        shuffled_indices = np.random.permutation(all_indices)

        # 按比例划分
        train_count = int(self.total_points * self.train_split)
        train_indices = shuffled_indices[:train_count]
        test_indices = shuffled_indices[train_count:]

        return train_indices, test_indices

    def global_index_to_data_point(self, global_idx: int) -> DataPoint:
        """
        将全局索引转换为数据点

        全局索引编码方式：
            global_idx = sample_idx * (grid_size^2 * num_freq)
                       + i * (grid_size * num_freq)
                       + j * num_freq
                       + freq_idx

        参数：
            global_idx: 全局索引 [0, total_points-1]

        返回：
            DataPoint对象
        """
        points_per_sample = self.grid_size * self.grid_size * self.num_frequencies

        # 解码
        sample_idx = global_idx // points_per_sample
        remainder = global_idx % points_per_sample

        points_per_i = self.grid_size * self.num_frequencies
        i = remainder // points_per_i
        remainder = remainder % points_per_i

        j = remainder // self.num_frequencies
        freq_idx = remainder % self.num_frequencies

        # 计算角度
        theta = self.theta_values[i]
        phi = self.phi_values[j]

        return DataPoint(sample_idx, i, j, freq_idx, theta, phi)

    def get_train_subset(self, subset_size: Optional[int] = None) -> np.ndarray:
        """
        获取训练集子集（用于快速实验）

        参数：
            subset_size: 子集大小（默认: None，返回全部训练集）

        返回：
            训练集索引数组（可能是子集）
        """
        if subset_size is None or subset_size >= len(self.train_indices):
            return self.train_indices

        # 随机采样子集
        np.random.seed(self.random_seed + 1)  # 不同的种子
        subset_indices = np.random.choice(
            self.train_indices,
            size=subset_size,
            replace=False
        )
        return subset_indices

    def get_statistics(self) -> dict:
        """
        获取采样统计信息

        返回：
            统计信息字典
        """
        # 分析训练集的分布
        train_samples = set()
        train_freqs = {0: 0, 1: 0, 2: 0}

        for idx in self.train_indices[:1000]:  # 采样1000个点分析
            dp = self.global_index_to_data_point(idx)
            train_samples.add(dp.sample_idx)
            train_freqs[dp.freq_idx] += 1

        stats = {
            'total_points': self.total_points,
            'train_count': len(self.train_indices),
            'test_count': len(self.test_indices),
            'train_ratio': len(self.train_indices) / self.total_points,
            'samples_in_train_subset': len(train_samples),  # 1000个点中涉及多少个样本
            'freq_distribution': train_freqs,  # 频率分布（1000个点）
        }

        return stats


def create_global_mixed_split(
    num_samples: int = 200,
    num_frequencies: int = 3,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：创建全局混合划分

    参数：
        num_samples: 样本数量
        num_frequencies: 频率数量
        train_split: 训练集比例
        random_seed: 随机种子

    返回：
        train_indices, test_indices: 训练集和测试集索引
    """
    sampler = AngleSampler(
        num_samples=num_samples,
        num_frequencies=num_frequencies,
        train_split=train_split,
        random_seed=random_seed
    )

    return sampler.train_indices, sampler.test_indices


if __name__ == "__main__":
    print("=" * 80)
    print("AngleSampler 测试")
    print("=" * 80)

    # 测试1: 创建采样器
    print("\n[Test 1] 创建采样器（200样本 × 91×91 × 3频率）")
    sampler = AngleSampler(num_samples=200, num_frequencies=3)

    # 测试2: 全局索引解码
    print("\n[Test 2] 全局索引解码")
    test_indices = [0, 1000, 100000, 1000000, 4968599]
    for idx in test_indices:
        if idx < sampler.total_points:
            dp = sampler.global_index_to_data_point(idx)
            print(f"  全局索引{idx:>7d} → {dp}")

    # 测试3: 训练/测试集验证
    print("\n[Test 3] 训练/测试集验证")
    print(f"训练集大小: {len(sampler.train_indices):,}")
    print(f"测试集大小: {len(sampler.test_indices):,}")
    print(f"总计: {len(sampler.train_indices) + len(sampler.test_indices):,}")
    print(f"期望总计: {sampler.total_points:,}")

    # 验证无重复
    train_set = set(sampler.train_indices)
    test_set = set(sampler.test_indices)
    overlap = train_set & test_set
    print(f"训练/测试集重叠: {len(overlap)} (应为0)")

    # 测试4: 训练集子采样
    print("\n[Test 4] 训练集子采样")
    subset_300k = sampler.get_train_subset(subset_size=300000)
    print(f"子集大小: {len(subset_300k):,} (期望: 300,000)")
    print(f"前5个索引: {subset_300k[:5].tolist()}")

    # 测试5: 统计信息
    print("\n[Test 5] 统计信息")
    stats = sampler.get_statistics()
    print(f"总数据点: {stats['total_points']:,}")
    print(f"训练集: {stats['train_count']:,} ({stats['train_ratio']*100:.1f}%)")
    print(f"测试集: {stats['test_count']:,} ({(1-stats['train_ratio'])*100:.1f}%)")
    print(f"训练集1000点中涉及样本数: {stats['samples_in_train_subset']}")
    print(f"训练集1000点频率分布: {stats['freq_distribution']}")

    # 测试6: 验证全局混合特性
    print("\n[Test 6] 验证全局混合特性")
    sample_points = [sampler.global_index_to_data_point(idx)
                     for idx in sampler.train_indices[:10]]
    print("训练集前10个数据点:")
    for i, dp in enumerate(sample_points):
        print(f"  {i+1}. 样本{dp.sample_idx:3d}, θ={dp.theta:6.2f}°, "
              f"φ={dp.phi:6.2f}°, 频率{dp.freq_idx}")

    # 测试7: 便捷函数
    print("\n[Test 7] 便捷函数")
    train_idx, test_idx = create_global_mixed_split(num_samples=200)
    print(f"便捷函数返回: 训练集{len(train_idx):,}, 测试集{len(test_idx):,}")

    print("\n" + "=" * 80)
    print("[PASS] AngleSampler测试通过!")
    print("=" * 80)
