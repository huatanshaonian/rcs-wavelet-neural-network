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
                 log_transform: bool = False):
        """
        初始化数据适配器

        Args:
            normalize: 是否标准化数据
            log_transform: 是否进行对数变换
        """
        self.normalize = normalize
        self.log_transform = log_transform

        # 数据统计信息
        self.data_stats = {}

    def adapt_rcs_data(self,
                      rcs_data: np.ndarray) -> torch.Tensor:
        """
        适配RCS数据

        Args:
            rcs_data: [N, 91, 91, 2] 原始RCS数据

        Returns:
            adapted_data: [N, 91, 91, 2] 适配后的数据
        """
        # 确保数据格式正确
        if len(rcs_data.shape) != 4 or rcs_data.shape[1:] != (91, 91, 2):
            raise ValueError(f"RCS数据形状应为 [N, 91, 91, 2]，实际为 {rcs_data.shape}")

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
            adapted_data: [N, 91, 91, 2] 适配后的数据

        Returns:
            original_data: [N, 91, 91, 2] 原始格式数据
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
            rcs_data: [N, 91, 91, 2] RCS数据
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
        rcs_data: [N, 91, 91, 2] RCS数据
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

    # 创建测试数据
    n_samples = 10
    rcs_data = np.random.randn(n_samples, 91, 91, 2) * 10
    params_data = np.random.randn(n_samples, 9)

    print(f"原始数据: RCS {rcs_data.shape}, 参数 {params_data.shape}")

    # 测试适配器
    adapter = RCS_DataAdapter(normalize=True, log_transform=False)

    # 适配数据
    adapted_rcs = adapter.adapt_rcs_data(rcs_data)
    print(f"适配后RCS形状: {adapted_rcs.shape}")
    print(f"适配后数值范围: [{adapted_rcs.min():.3f}, {adapted_rcs.max():.3f}]")

    # 逆适配
    recovered_rcs = adapter.inverse_adapt(adapted_rcs)
    print(f"恢复后RCS形状: {recovered_rcs.shape}")

    # 计算恢复误差
    recovery_error = np.mean((rcs_data - recovered_rcs) ** 2)
    print(f"恢复MSE误差: {recovery_error:.6f}")

    # 测试数据集
    dataset = AutoEncoderDataset(rcs_data, params_data, adapter)
    print(f"数据集大小: {len(dataset)}")

    # 测试数据加载器
    train_loader, val_loader = create_ae_dataloaders(
        rcs_data, params_data, validation_split=0.3, batch_size=4
    )

    print(f"训练加载器批次数: {len(train_loader)}")
    print(f"验证加载器批次数: {len(val_loader)}")

    # 测试一个批次
    for batch_rcs in train_loader:
        print(f"批次RCS形状: {batch_rcs.shape}")
        break

    return recovery_error < 1e-10  # 恢复应该是精确的


if __name__ == "__main__":
    test_data_adapters()