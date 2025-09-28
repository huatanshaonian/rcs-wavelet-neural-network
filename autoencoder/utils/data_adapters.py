"""
数据适配器模块
将现有数据格式适配到AutoEncoder所需格式
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader


class RCS_DataAdapter:
    """
    RCS数据适配器
    将现有的RCS数据格式转换为AutoEncoder所需格式
    """

    def __init__(self,
                 normalize: bool = True,
                 log_transform: bool = False,
                 expected_frequencies: int = 2):
        """
        初始化数据适配器

        Args:
            normalize: 是否标准化数据
            log_transform: 是否进行对数变换
            expected_frequencies: 预期频率数量 (2 for 1.5GHz+3GHz, 3 for +6GHz)
        """
        self.normalize = normalize
        self.log_transform = log_transform
        self.expected_frequencies = expected_frequencies

        # 数据统计信息
        self.data_stats = {}

    def adapt_rcs_data(self,
                      rcs_data: np.ndarray) -> torch.Tensor:
        """
        适配RCS数据

        Args:
            rcs_data: [N, 91, 91, num_freq] 原始RCS数据

        Returns:
            adapted_data: [N, 91, 91, num_freq] 适配后的数据
        """
        # 确保数据格式正确并检测频率数量
        if len(rcs_data.shape) != 4:
            raise ValueError(f"RCS数据应为4维，实际为{len(rcs_data.shape)}维")

        actual_frequencies = rcs_data.shape[3]
        if actual_frequencies != self.expected_frequencies:
            print(f"⚠️ 检测到{actual_frequencies}个频率，预期{self.expected_frequencies}个")
            self.expected_frequencies = actual_frequencies  # 自动适配

        expected_shape = (91, 91, self.expected_frequencies)
        if rcs_data.shape[1:] != expected_shape:
            raise ValueError(f"RCS数据形状应为 [N, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}]，实际为 {rcs_data.shape}")

        data = rcs_data.copy()

        # 对数变换
        if self.log_transform:
            # 避免对负值取对数
            data = np.sign(data) * np.log(np.abs(data) + 1e-8)

        # 标准化
        if self.normalize:
            # 计算统计信息
            mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
            std = np.std(data, axis=(0, 1, 2), keepdims=True)
            std = np.where(std == 0, 1, std)  # 避免除零

            # 保存统计信息
            self.data_stats = {
                'mean': mean,
                'std': std,
                'log_transform': self.log_transform
            }

            # 标准化
            data = (data - mean) / std

        return torch.FloatTensor(data)

    def inverse_adapt(self,
                     adapted_data: torch.Tensor) -> np.ndarray:
        """
        逆适配：将处理后的数据转回原始格式

        Args:
            adapted_data: [N, 91, 91, num_freq] 适配后的数据

        Returns:
            original_data: [N, 91, 91, num_freq] 原始格式数据
        """
        data = adapted_data.detach().cpu().numpy()

        # 逆标准化
        if self.normalize and 'mean' in self.data_stats:
            mean = self.data_stats['mean']
            std = self.data_stats['std']
            data = data * std + mean

        # 逆对数变换
        if self.log_transform:
            data = np.sign(data) * (np.exp(np.abs(data)) - 1e-8)

        return data


class AutoEncoderDataset(Dataset):
    """
    AutoEncoder专用数据集
    """

    def __init__(self,
                 rcs_data: np.ndarray,
                 params_data: Optional[np.ndarray] = None,
                 adapter: Optional[RCS_DataAdapter] = None):
        """
        初始化数据集

        Args:
            rcs_data: [N, 91, 91, num_freq] RCS数据
            params_data: [N, 9] 参数数据（可选）
            adapter: 数据适配器
        """
        self.adapter = adapter or RCS_DataAdapter()

        # 适配RCS数据
        self.rcs_data = self.adapter.adapt_rcs_data(rcs_data)

        # 参数数据
        self.params_data = None
        if params_data is not None:
            self.params_data = torch.FloatTensor(params_data)

        self.has_params = params_data is not None

    def __len__(self):
        return len(self.rcs_data)

    def __getitem__(self, idx):
        rcs = self.rcs_data[idx]

        if self.has_params:
            params = self.params_data[idx]
            return rcs, params
        else:
            return rcs

    def get_rcs_only_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """获取仅RCS数据的DataLoader（用于AE训练）"""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def get_paired_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """获取RCS-参数配对的DataLoader（用于映射训练）"""
        if not self.has_params:
            raise ValueError("数据集不包含参数数据，无法创建配对加载器")
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def create_ae_dataloaders(rcs_data: np.ndarray,
                         params_data: Optional[np.ndarray] = None,
                         validation_split: float = 0.2,
                         batch_size: int = 32,
                         normalize: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    创建AutoEncoder训练用的数据加载器

    Args:
        rcs_data: [N, 91, 91, num_freq] RCS数据
        params_data: [N, 9] 参数数据（可选）
        validation_split: 验证集比例
        batch_size: 批次大小
        normalize: 是否标准化

    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 分割数据
    n_samples = len(rcs_data)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # 创建适配器
    adapter = RCS_DataAdapter(normalize=normalize)

    # 创建训练集
    train_rcs = rcs_data[train_indices]
    train_params = params_data[train_indices] if params_data is not None else None

    train_dataset = AutoEncoderDataset(train_rcs, train_params, adapter)
    train_loader = train_dataset.get_rcs_only_loader(batch_size=batch_size, shuffle=True)

    # 创建验证集（使用相同的适配器统计信息）
    val_rcs = rcs_data[val_indices]
    val_params = params_data[val_indices] if params_data is not None else None

    val_dataset = AutoEncoderDataset(val_rcs, val_params, adapter)
    val_loader = val_dataset.get_rcs_only_loader(batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def test_data_adapters():
    """测试数据适配器"""
    print("=== 数据适配器测试 ===")

    # 测试2频率配置
    print("=== 测试2频率配置 ===")
    n_samples = 10
    rcs_data_2freq = np.random.randn(n_samples, 91, 91, 2) * 10
    params_data = np.random.randn(n_samples, 9)

    print(f"2频率原始数据: RCS {rcs_data_2freq.shape}, 参数 {params_data.shape}")

    adapter_2freq = RCS_DataAdapter(normalize=True, log_transform=False, expected_frequencies=2)

    # 适配2频率数据
    adapted_rcs_2freq = adapter_2freq.adapt_rcs_data(rcs_data_2freq)
    print(f"2频率适配后RCS形状: {adapted_rcs_2freq.shape}")
    print(f"2频率适配后数值范围: [{adapted_rcs_2freq.min():.3f}, {adapted_rcs_2freq.max():.3f}]")

    # 逆适配
    recovered_rcs_2freq = adapter_2freq.inverse_adapt(adapted_rcs_2freq)
    print(f"2频率恢复后RCS形状: {recovered_rcs_2freq.shape}")

    # 计算恢复误差
    recovery_error_2freq = np.mean((rcs_data_2freq - recovered_rcs_2freq) ** 2)
    print(f"2频率恢复MSE误差: {recovery_error_2freq:.6f}")

    # 测试3频率配置
    print("\n=== 测试3频率配置 ===")
    rcs_data_3freq = np.random.randn(n_samples, 91, 91, 3) * 10
    print(f"3频率原始数据: RCS {rcs_data_3freq.shape}")

    adapter_3freq = RCS_DataAdapter(normalize=True, log_transform=False, expected_frequencies=3)
    adapted_rcs_3freq = adapter_3freq.adapt_rcs_data(rcs_data_3freq)
    print(f"3频率适配后RCS形状: {adapted_rcs_3freq.shape}")

    recovered_rcs_3freq = adapter_3freq.inverse_adapt(adapted_rcs_3freq)
    recovery_error_3freq = np.mean((rcs_data_3freq - recovered_rcs_3freq) ** 2)
    print(f"3频率恢复MSE误差: {recovery_error_3freq:.6f}")

    # 测试数据集
    print("\n=== 测试数据集创建 ===")
    dataset_2freq = AutoEncoderDataset(rcs_data_2freq, params_data, adapter_2freq)
    dataset_3freq = AutoEncoderDataset(rcs_data_3freq, params_data, adapter_3freq)
    print(f"2频率数据集大小: {len(dataset_2freq)}")
    print(f"3频率数据集大小: {len(dataset_3freq)}")

    # 测试数据加载器
    train_loader_2freq, val_loader_2freq = create_ae_dataloaders(
        rcs_data_2freq, params_data, validation_split=0.3, batch_size=4
    )
    train_loader_3freq, val_loader_3freq = create_ae_dataloaders(
        rcs_data_3freq, params_data, validation_split=0.3, batch_size=4
    )

    print(f"2频率训练加载器批次数: {len(train_loader_2freq)}")
    print(f"3频率训练加载器批次数: {len(train_loader_3freq)}")

    # 测试批次数据
    for batch_rcs in train_loader_2freq:
        print(f"2频率批次RCS形状: {batch_rcs.shape}")
        break

    for batch_rcs in train_loader_3freq:
        print(f"3频率批次RCS形状: {batch_rcs.shape}")
        break

    # 验证自动适配功能
    print("\n=== 测试自动适配功能 ===")
    adapter_auto = RCS_DataAdapter(expected_frequencies=2)  # 预期2频率
    adapted_auto = adapter_auto.adapt_rcs_data(rcs_data_3freq)  # 但给3频率数据
    print(f"自动适配后频率数: {adapter_auto.expected_frequencies}")

    return (recovery_error_2freq < 1e-10 and recovery_error_3freq < 1e-10)


if __name__ == "__main__":
    test_data_adapters()