"""
角度RCS数据集 (AngleRCSDataset)

PyTorch Dataset封装，用于角度-based RCS预测训练。

数据格式：
- 输入: (θ, φ, params, freq_idx)
- 输出: target_rcs (单点RCS值)

数据流：
1. 全局索引 → (sample_idx, i, j, freq_idx)
2. 查找 params[sample_idx] 和 rcs_data[sample_idx, i, j, freq_idx]
3. 返回训练样本
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from .angle_sampler import AngleSampler
    from .data_cache import AngleRCSDataCache
except ImportError:
    from angle_sampler import AngleSampler
    from data_cache import AngleRCSDataCache

from autoencoder.utils.data_adapters import RCS_DataAdapter


class AngleRCSDataset(Dataset):
    """
    角度RCS数据集

    参数：
        rcs_data (ndarray): RCS数据 [N, 91, 91, num_freq]，线性域
        param_data (ndarray): 参数数据 [N, 9]
        sampler (AngleSampler): 角度采样器
        indices (ndarray): 数据索引（训练集或测试集）
        data_adapter (RCS_DataAdapter, optional): 数据适配器（用于标准化）
        normalize_params (bool): 是否标准化参数（默认: True）

    返回：
        字典，包含：
            'theta': 角度θ（度）
            'phi': 角度φ（度）
            'params': [9,] 设计参数（可能标准化）
            'freq_idx': 频率索引 (0/1/2)
            'target_rcs': 目标RCS值（线性域）
    """

    def __init__(self,
                 rcs_data: np.ndarray,
                 param_data: np.ndarray,
                 sampler: AngleSampler,
                 indices: np.ndarray,
                 data_adapter: Optional[RCS_DataAdapter] = None,
                 normalize_params: bool = True,
                 preload_to_gpu: bool = False,
                 # 缓存相关参数（用于计算缓存键）
                 train_split: float = 0.8,
                 random_seed: int = 42,
                 train_subset_size: Optional[int] = None,
                 is_train: bool = True):

        self.sampler = sampler
        self.indices = indices  # 全局索引数组
        self.data_adapter = data_adapter
        self.normalize_params = normalize_params
        self.preload_to_gpu = preload_to_gpu

        # 验证数据形状
        assert rcs_data.shape[0] == param_data.shape[0], \
            f"样本数不匹配: RCS {rcs_data.shape[0]} vs 参数 {param_data.shape[0]}"
        assert rcs_data.shape[1] == rcs_data.shape[2] == sampler.grid_size, \
            f"网格大小不匹配: RCS {rcs_data.shape[1:3]} vs 采样器 {sampler.grid_size}"
        assert rcs_data.shape[3] == sampler.num_frequencies, \
            f"频率数量不匹配: RCS {rcs_data.shape[3]} vs 采样器 {sampler.num_frequencies}"

        # 参数标准化器（如果需要）
        if normalize_params:
            self.param_mean = np.mean(param_data, axis=0, keepdims=True)
            self.param_std = np.std(param_data, axis=0, keepdims=True)
            self.param_std = np.where(self.param_std == 0, 1.0, self.param_std)
        else:
            self.param_mean = None
            self.param_std = None

        # GPU预加载（如果启用）
        if preload_to_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            num_points = len(indices)

            # 初始化缓存管理器
            cache_manager = AngleRCSDataCache()

            # 计算缓存键
            cache_key = cache_manager._compute_cache_key(
                rcs_shape=rcs_data.shape,
                param_shape=param_data.shape,
                num_frequencies=sampler.num_frequencies,
                train_split=train_split,
                random_seed=random_seed,
                train_subset_size=train_subset_size,
                normalize_params=normalize_params,
                is_train=is_train
            )
            split = 'train' if is_train else 'val'

            # 检查缓存是否存在
            if cache_manager.has_cache(cache_key, split):
                print(f"[AngleRCSDataset] 发现缓存，直接加载... ({split})")

                # 从缓存加载所有数据
                cached_data = cache_manager.load_cache(cache_key, split, device='cuda')

                self.theta_array = cached_data['theta_array']
                self.phi_array = cached_data['phi_array']
                self.sample_idx_array = cached_data['sample_idx_array']
                self.i_array = cached_data['i_array']
                self.j_array = cached_data['j_array']
                self.freq_idx_array = cached_data['freq_idx_array']
                self.rcs_data = cached_data['rcs_data']
                self.param_data = cached_data['param_data']

                # 恢复标准化参数（如果有）
                if 'param_mean' in cached_data:
                    self.param_mean = cached_data['param_mean']
                    self.param_std = cached_data['param_std']

                print(f"[AngleRCSDataset] 缓存加载完成: {num_points:,}个数据点")
                print(f"[AngleRCSDataset] GPU显存占用: ~{self._estimate_gpu_memory():.1f} MB")
            else:
                print(f"[AngleRCSDataset] 未发现缓存，开始预加载... ({split})")

                # 转换RCS数据到GPU [N, 91, 91, num_freq]
                self.rcs_data = torch.from_numpy(rcs_data).float().to(device)

                # 转换参数数据到GPU [N, 9]
                if normalize_params:
                    params_norm = (param_data - self.param_mean) / self.param_std
                    self.param_data = torch.from_numpy(params_norm).float().to(device)
                else:
                    self.param_data = torch.from_numpy(param_data).float().to(device)

                # 预计算所有数据点（提前解码，避免__getitem__中重复计算）
                print(f"[AngleRCSDataset] 预计算索引: {num_points:,}个数据点...")

                self.theta_array = torch.zeros(num_points, dtype=torch.float32, device=device)
                self.phi_array = torch.zeros(num_points, dtype=torch.float32, device=device)
                self.sample_idx_array = torch.zeros(num_points, dtype=torch.long, device=device)
                self.i_array = torch.zeros(num_points, dtype=torch.long, device=device)
                self.j_array = torch.zeros(num_points, dtype=torch.long, device=device)
                self.freq_idx_array = torch.zeros(num_points, dtype=torch.long, device=device)

                # 批量解码（显示进度）
                progress_interval = max(1, num_points // 10)  # 每10%输出一次
                for idx, global_idx in enumerate(indices):
                    data_point = sampler.global_index_to_data_point(global_idx)
                    self.theta_array[idx] = data_point.theta
                    self.phi_array[idx] = data_point.phi
                    self.sample_idx_array[idx] = data_point.sample_idx
                    self.i_array[idx] = data_point.i
                    self.j_array[idx] = data_point.j
                    self.freq_idx_array[idx] = data_point.freq_idx

                    # 显示进度
                    if (idx + 1) % progress_interval == 0 or (idx + 1) == num_points:
                        progress = 100.0 * (idx + 1) / num_points
                        print(f"[AngleRCSDataset] 进度: {idx+1:,}/{num_points:,} ({progress:.0f}%)")

                print(f"[AngleRCSDataset] GPU预加载完成: {num_points:,}个数据点")
                print(f"[AngleRCSDataset] GPU显存占用: ~{self._estimate_gpu_memory():.1f} MB")

                # 保存到缓存
                print("[AngleRCSDataset] 保存缓存...")
                cache_data = {
                    'theta_array': self.theta_array,
                    'phi_array': self.phi_array,
                    'sample_idx_array': self.sample_idx_array,
                    'i_array': self.i_array,
                    'j_array': self.j_array,
                    'freq_idx_array': self.freq_idx_array,
                    'rcs_data': self.rcs_data,
                    'param_data': self.param_data,
                }

                # 保存标准化参数（如果有）
                if normalize_params:
                    cache_data['param_mean'] = self.param_mean
                    cache_data['param_std'] = self.param_std

                cache_manager.save_cache(cache_key, split, cache_data)
        else:
            # 常规模式：保持numpy数组
            self.rcs_data = rcs_data  # [N, 91, 91, num_freq]
            self.param_data = param_data  # [N, 9]

    def _estimate_gpu_memory(self) -> float:
        """估算GPU显存占用（MB）"""
        if not self.preload_to_gpu:
            return 0.0

        num_points = len(self.indices)

        # RCS数据 [N, 91, 91, num_freq] × 4 bytes
        rcs_size = self.rcs_data.element_size() * self.rcs_data.numel()

        # 参数数据 [N, 9] × 4 bytes
        param_size = self.param_data.element_size() * self.param_data.numel()

        # 索引数组 [num_points] × 6 × 4 bytes (theta, phi, sample_idx, i, j, freq_idx)
        index_size = num_points * 6 * 4

        total_bytes = rcs_size + param_size + index_size
        return total_bytes / (1024 * 1024)  # 转换为MB

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据点

        参数：
            idx: 数据集内部索引 [0, len(indices)-1]

        返回：
            数据字典
        """
        if self.preload_to_gpu and torch.cuda.is_available():
            # GPU预加载模式：直接索引GPU tensor（零拷贝）
            sample_idx = self.sample_idx_array[idx]
            i = self.i_array[idx]
            j = self.j_array[idx]
            freq_idx = self.freq_idx_array[idx]

            return {
                'theta': self.theta_array[idx],
                'phi': self.phi_array[idx],
                'params': self.param_data[sample_idx],  # 已标准化
                'freq_idx': freq_idx,
                'target_rcs': self.rcs_data[sample_idx, i, j, freq_idx]
            }
        else:
            # 常规模式：CPU numpy数组
            # 获取全局索引
            global_idx = self.indices[idx]

            # 解码为数据点
            data_point = self.sampler.global_index_to_data_point(global_idx)

            # 提取参数
            params = self.param_data[data_point.sample_idx].copy()  # [9,]

            # 参数标准化
            if self.normalize_params:
                params = (params - self.param_mean.squeeze()) / self.param_std.squeeze()

            # 提取目标RCS值（线性域）
            target_rcs = self.rcs_data[
                data_point.sample_idx,
                data_point.i,
                data_point.j,
                data_point.freq_idx
            ]

            # 返回字典
            return {
                'theta': torch.tensor(data_point.theta, dtype=torch.float32),
                'phi': torch.tensor(data_point.phi, dtype=torch.float32),
                'params': torch.tensor(params, dtype=torch.float32),
                'freq_idx': torch.tensor(data_point.freq_idx, dtype=torch.long),
                'target_rcs': torch.tensor(target_rcs, dtype=torch.float32)
            }


def create_dataloaders(
    rcs_data: np.ndarray,
    param_data: np.ndarray,
    batch_size: int = 256,
    num_frequencies: int = 3,
    train_split: float = 0.8,
    random_seed: int = 42,
    train_subset_size: Optional[int] = None,
    normalize_params: bool = True,
    num_workers: int = 0,
    preload_to_gpu: bool = False
) -> Tuple[DataLoader, DataLoader, AngleSampler]:
    """
    创建训练和测试DataLoader

    参数：
        rcs_data: [N, 91, 91, num_freq] RCS数据（线性域）
        param_data: [N, 9] 参数数据
        batch_size: 批次大小（默认: 256）
        num_frequencies: 频率数量（默认: 3）
        train_split: 训练集比例（默认: 0.8）
        random_seed: 随机种子（默认: 42）
        train_subset_size: 训练集子采样大小（默认: None，使用全部）
        normalize_params: 是否标准化参数（默认: True）
        num_workers: DataLoader工作线程数（默认: 0，GPU预加载时自动设为0）
        preload_to_gpu: 是否预加载全部数据到GPU（默认: False）

    返回：
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
        sampler: AngleSampler对象
    """
    # 创建采样器
    num_samples = rcs_data.shape[0]
    sampler = AngleSampler(
        num_samples=num_samples,
        num_frequencies=num_frequencies,
        train_split=train_split,
        random_seed=random_seed
    )

    # 获取训练集索引（可能子采样）
    if train_subset_size is not None:
        train_indices = sampler.get_train_subset(subset_size=train_subset_size)
        print(f"使用训练集子集: {len(train_indices):,} / {len(sampler.train_indices):,}")
    else:
        train_indices = sampler.train_indices

    # GPU预加载时自动调整参数
    if preload_to_gpu and torch.cuda.is_available():
        actual_num_workers = 0  # GPU预加载不需要多进程
        actual_pin_memory = False  # 数据已在GPU，不需要pin_memory
        print(f"[create_dataloaders] 启用GPU预加载模式")
        print(f"[create_dataloaders] 自动设置: num_workers=0, pin_memory=False")

        # GPU-aware collate_fn: 避免CPU-GPU传输
        def gpu_collate_fn(batch):
            """
            GPU-aware collate函数

            当数据已在GPU时，直接在GPU上合并batch，避免默认collate_fn的CPU-GPU传输。

            默认collate_fn问题：
            - 在CPU上执行torch.stack() → GPU数据会被拷贝到CPU
            - 然后trainer._train_epoch再调用.to(device) → 又拷贝回GPU
            - 每个batch都有2次CPU-GPU传输！

            GPU collate解决方案：
            - 数据已在GPU → 直接在GPU上stack → 返回GPU tensor
            - 训练循环无需.to(device)调用
            """
            # 提取所有键
            keys = batch[0].keys()

            # 对每个键进行stack（在GPU上）
            collated = {}
            for key in keys:
                values = [item[key] for item in batch]
                # 直接stack，数据已在GPU，不会触发CPU传输
                collated[key] = torch.stack(values, dim=0)

            return collated

        actual_collate_fn = gpu_collate_fn
    else:
        actual_num_workers = num_workers
        actual_pin_memory = True if torch.cuda.is_available() else False
        actual_collate_fn = None  # 使用默认collate_fn

    # 创建数据集
    train_dataset = AngleRCSDataset(
        rcs_data=rcs_data,
        param_data=param_data,
        sampler=sampler,
        indices=train_indices,
        normalize_params=normalize_params,
        preload_to_gpu=preload_to_gpu,
        # 缓存相关参数
        train_split=train_split,
        random_seed=random_seed,
        train_subset_size=train_subset_size,
        is_train=True
    )

    test_dataset = AngleRCSDataset(
        rcs_data=rcs_data,
        param_data=param_data,
        sampler=sampler,
        indices=sampler.test_indices,
        normalize_params=normalize_params,
        preload_to_gpu=preload_to_gpu,
        # 缓存相关参数
        train_split=train_split,
        random_seed=random_seed,
        train_subset_size=train_subset_size,
        is_train=False
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_num_workers,
        pin_memory=actual_pin_memory,
        collate_fn=actual_collate_fn,  # 使用GPU-aware collate
        drop_last=True  # 丢弃最后不完整的batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_num_workers,
        pin_memory=actual_pin_memory,
        collate_fn=actual_collate_fn,  # 使用GPU-aware collate
        drop_last=False  # 测试集保留所有数据
    )

    return train_loader, test_loader, sampler


if __name__ == "__main__":
    print("=" * 80)
    print("AngleRCSDataset 测试")
    print("=" * 80)

    # 生成模拟数据
    print("\n[Test 1] 生成模拟数据")
    num_samples = 200
    num_freq = 3
    grid_size = 91

    rcs_data = np.random.rand(num_samples, grid_size, grid_size, num_freq) * 0.5
    param_data = np.random.randn(num_samples, 9)

    print(f"RCS数据: {rcs_data.shape}")
    print(f"参数数据: {param_data.shape}")
    print(f"RCS范围: [{rcs_data.min():.6f}, {rcs_data.max():.6f}]")

    # 测试2: 创建采样器
    print("\n[Test 2] 创建采样器")
    sampler = AngleSampler(
        num_samples=num_samples,
        num_frequencies=num_freq,
        train_split=0.8,
        random_seed=42
    )

    # 测试3: 创建数据集
    print("\n[Test 3] 创建数据集")
    train_dataset = AngleRCSDataset(
        rcs_data=rcs_data,
        param_data=param_data,
        sampler=sampler,
        indices=sampler.train_indices[:10000],  # 使用前10000个训练点测试
        normalize_params=True
    )

    print(f"数据集大小: {len(train_dataset)}")

    # 测试4: 获取单个样本
    print("\n[Test 4] 获取单个样本")
    sample = train_dataset[0]
    print("样本内容:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  {key:12s}: {value.item():.6f}")
            else:
                print(f"  {key:12s}: shape={value.shape}, 范围=[{value.min():.3f}, {value.max():.3f}]")

    # 测试5: 创建DataLoader
    print("\n[Test 5] 创建DataLoader")
    train_loader, test_loader, _ = create_dataloaders(
        rcs_data=rcs_data,
        param_data=param_data,
        batch_size=256,
        num_frequencies=num_freq,
        train_subset_size=100000,  # 使用10万个训练点
        normalize_params=True
    )

    print(f"训练DataLoader: {len(train_loader)} 批次")
    print(f"测试DataLoader: {len(test_loader)} 批次")

    # 测试6: 迭代批次
    print("\n[Test 6] 迭代批次")
    for i, batch in enumerate(train_loader):
        if i == 0:
            print("第一个批次:")
            for key, value in batch.items():
                print(f"  {key:12s}: {value.shape}")
            print(f"  theta范围: [{batch['theta'].min():.1f}, {batch['theta'].max():.1f}]")
            print(f"  phi范围: [{batch['phi'].min():.1f}, {batch['phi'].max():.1f}]")
            print(f"  freq_idx唯一值: {torch.unique(batch['freq_idx']).tolist()}")
            print(f"  target_rcs范围: [{batch['target_rcs'].min():.6f}, {batch['target_rcs'].max():.6f}]")
        if i >= 2:  # 只测试前3个批次
            break

    print(f"成功迭代 {i+1} 个批次")

    # 测试7: 参数标准化验证
    print("\n[Test 7] 参数标准化验证")
    normalized_params = []
    for i in range(min(1000, len(train_dataset))):
        sample = train_dataset[i]
        normalized_params.append(sample['params'].numpy())

    normalized_params = np.array(normalized_params)
    print(f"标准化后参数统计:")
    print(f"  均值: {normalized_params.mean(axis=0)[:3]} ... (前3维)")
    print(f"  标准差: {normalized_params.std(axis=0)[:3]} ... (前3维)")
    print(f"  期望: 均值≈0, 标准差≈1")

    # 测试8: 完整epoch吞吐量
    print("\n[Test 8] 完整epoch吞吐量估算")
    import time

    small_loader, _, _ = create_dataloaders(
        rcs_data=rcs_data,
        param_data=param_data,
        batch_size=256,
        train_subset_size=10000,  # 小数据集测试
        num_workers=0
    )

    start_time = time.time()
    for i, batch in enumerate(small_loader):
        if i >= 10:  # 测试10个批次
            break
    elapsed = time.time() - start_time

    batches_per_second = 10 / elapsed
    full_epoch_batches = len(train_loader)
    estimated_time = full_epoch_batches / batches_per_second

    print(f"10批次耗时: {elapsed:.3f}秒")
    print(f"吞吐量: {batches_per_second:.1f} 批次/秒")
    print(f"完整epoch估计耗时: {estimated_time:.1f}秒 ({estimated_time/60:.1f}分钟)")

    print("\n" + "=" * 80)
    print("[PASS] AngleRCSDataset测试通过!")
    print("=" * 80)
    print("\nPhase 2 完成！数据流就绪，可以开始Phase 3（训练器实现）")
