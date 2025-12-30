"""
Angle-based RCS数据集GPU预加载缓存系统

避免每次启动训练都要重新预计算3,974,880个数据点的索引
"""

import os
import hashlib
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class AngleRCSDataCache:
    """GPU预加载数据缓存管理器"""

    def __init__(self, cache_dir: str = 'angle_rcs_cache'):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径（相对于项目根目录）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _compute_cache_key(self,
                          rcs_shape: tuple,
                          param_shape: tuple,
                          num_frequencies: int,
                          train_split: float,
                          random_seed: int,
                          train_subset_size: Optional[int],
                          normalize_params: bool,
                          is_train: bool) -> str:
        """
        计算缓存键（基于数据集配置的唯一标识）

        Args:
            rcs_shape: RCS数据形状 (N, 91, 91, num_freq)
            param_shape: 参数数据形状 (N, 9)
            num_frequencies: 频率数量
            train_split: 训练集比例
            random_seed: 随机种子
            train_subset_size: 训练集子集大小（None表示全量）
            normalize_params: 是否标准化参数
            is_train: 是否为训练集

        Returns:
            cache_key: MD5哈希值（32字符）
        """
        config = {
            'rcs_shape': rcs_shape,
            'param_shape': param_shape,
            'num_frequencies': num_frequencies,
            'train_split': train_split,
            'random_seed': random_seed,
            'train_subset_size': train_subset_size,
            'normalize_params': normalize_params,
            'is_train': is_train
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_cache_path(self, cache_key: str, split: str) -> Path:
        """
        获取缓存文件路径

        Args:
            cache_key: 缓存键
            split: 'train' 或 'val'

        Returns:
            cache_path: 缓存文件完整路径
        """
        return self.cache_dir / f"{cache_key}_{split}.pt"

    def has_cache(self, cache_key: str, split: str) -> bool:
        """
        检查缓存是否存在

        Args:
            cache_key: 缓存键
            split: 'train' 或 'val'

        Returns:
            exists: 缓存文件是否存在
        """
        cache_path = self.get_cache_path(cache_key, split)
        return cache_path.exists()

    def save_cache(self, cache_key: str, split: str, data_dict: Dict[str, torch.Tensor]):
        """
        保存缓存到磁盘

        Args:
            cache_key: 缓存键
            split: 'train' 或 'val'
            data_dict: 包含所有GPU tensor的字典
                - 'theta_array': [N] theta角度
                - 'phi_array': [N] phi角度
                - 'sample_idx_array': [N] 样本索引
                - 'i_array': [N] i坐标
                - 'j_array': [N] j坐标
                - 'freq_idx_array': [N] 频率索引
                - 'rcs_data': [N_samples, 91, 91, num_freq] RCS数据
                - 'param_data': [N_samples, 9] 参数数据
                - 'param_mean': [9] 参数均值（如果标准化）
                - 'param_std': [9] 参数标准差（如果标准化）
        """
        cache_path = self.get_cache_path(cache_key, split)

        # 保存为CPU tensor（减小文件大小，加载时再转到GPU）
        cpu_data_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in data_dict.items()}

        torch.save(cpu_data_dict, cache_path)

        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"[AngleRCSDataCache] 缓存已保存: {cache_path.name} ({file_size_mb:.1f} MB)")

    def load_cache(self, cache_key: str, split: str, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        从磁盘加载缓存

        Args:
            cache_key: 缓存键
            split: 'train' 或 'val'
            device: 目标设备 ('cuda' 或 'cpu')

        Returns:
            data_dict: 包含所有GPU tensor的字典
        """
        cache_path = self.get_cache_path(cache_key, split)

        # 加载并转移到目标设备
        # weights_only=False: 允许加载包含numpy数组的缓存文件（本地生成，可信任）
        data_dict = torch.load(cache_path, map_location=device, weights_only=False)

        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"[AngleRCSDataCache] 缓存已加载: {cache_path.name} ({file_size_mb:.1f} MB)")

        return data_dict

    def clear_cache(self):
        """清空所有缓存文件"""
        cache_files = list(self.cache_dir.glob("*.pt"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

        for cache_file in cache_files:
            cache_file.unlink()

        print(f"[AngleRCSDataCache] 已清空缓存: {len(cache_files)}个文件, {total_size_mb:.1f} MB")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            info: 缓存信息字典
                - 'num_files': 缓存文件数量
                - 'total_size_mb': 总大小（MB）
                - 'files': 文件列表及大小
        """
        cache_files = list(self.cache_dir.glob("*.pt"))

        files_info = []
        total_size = 0
        for cache_file in cache_files:
            size = cache_file.stat().st_size
            total_size += size
            files_info.append({
                'name': cache_file.name,
                'size_mb': size / (1024 * 1024)
            })

        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'files': files_info
        }
