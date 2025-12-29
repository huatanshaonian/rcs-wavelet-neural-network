"""
Angle-based RCS重建工具函数
用于重建完整的91×91 RCS网格
"""

import torch
import numpy as np
from typing import Tuple


def reconstruct_rcs_grid(
    model,
    sample_idx: int,
    freq_idx: int,
    param_data: np.ndarray,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    theta_range: Tuple[float, float] = (45, 135),
    phi_range: Tuple[float, float] = (-45, 45),
    grid_size: int = 91
) -> np.ndarray:
    """
    重建单个样本、单个频率的完整91×91 RCS网格

    Args:
        model: AngleRCSNetwork模型
        sample_idx: 设计参数样本索引（0-199）
        freq_idx: 频率索引（0=1.5GHz, 1=3GHz, 2=6GHz）
        param_data: 设计参数数据 [N, 9]
        device: 计算设备
        theta_range: θ角度范围 (min, max)
        phi_range: φ角度范围 (min, max)
        grid_size: 网格大小（默认91）

    Returns:
        reconstructed_rcs: 重建的RCS网格 [91, 91]，线性值

    示例:
        >>> model = AngleRCSNetwork(...)
        >>> rcs_grid = reconstruct_rcs_grid(model, sample_idx=0, freq_idx=0, param_data=params)
        >>> print(rcs_grid.shape)  # (91, 91)
    """
    model.eval()
    model.to(device)

    # 生成91×91网格的角度坐标
    theta_values = np.linspace(theta_range[0], theta_range[1], grid_size)  # [91]
    phi_values = np.linspace(phi_range[0], phi_range[1], grid_size)        # [91]

    # 创建网格（meshgrid生成所有组合）
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing='ij')  # [91, 91]

    # 展平为[8281]（91×91=8281）
    theta_flat = theta_grid.flatten()  # [8281]
    phi_flat = phi_grid.flatten()      # [8281]

    # 获取对应样本的设计参数 [9]
    params = param_data[sample_idx]  # [9]

    # 复制为[8281, 9]（每个角度点使用相同的设计参数）
    params_repeat = np.tile(params, (grid_size * grid_size, 1))  # [8281, 9]

    # 创建频率索引数组 [8281]
    freq_repeat = np.full(grid_size * grid_size, freq_idx, dtype=np.int64)  # [8281]

    # 转换为torch tensor
    theta_tensor = torch.FloatTensor(theta_flat).to(device)      # [8281]
    phi_tensor = torch.FloatTensor(phi_flat).to(device)          # [8281]
    params_tensor = torch.FloatTensor(params_repeat).to(device)  # [8281, 9]
    freq_tensor = torch.LongTensor(freq_repeat).to(device)       # [8281]

    # 批量推理（一次性处理全部8281个点）
    with torch.no_grad():
        rcs_pred = model(theta_tensor, phi_tensor, params_tensor, freq_tensor).squeeze()  # [8281]

    # 转换回numpy并reshape为[91, 91]
    rcs_grid = rcs_pred.cpu().numpy().reshape(grid_size, grid_size)

    return rcs_grid


def reconstruct_rcs_multi_freq(
    model,
    sample_idx: int,
    param_data: np.ndarray,
    num_frequencies: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    theta_range: Tuple[float, float] = (45, 135),
    phi_range: Tuple[float, float] = (-45, 45),
    grid_size: int = 91
) -> np.ndarray:
    """
    重建单个样本的所有频率的RCS网格

    Args:
        model: AngleRCSNetwork模型
        sample_idx: 设计参数样本索引（0-199）
        param_data: 设计参数数据 [N, 9]
        num_frequencies: 频率数量（2或3）
        device: 计算设备
        theta_range: θ角度范围
        phi_range: φ角度范围
        grid_size: 网格大小

    Returns:
        reconstructed_rcs: 重建的RCS网格 [91, 91, num_freq]，线性值

    示例:
        >>> rcs_all = reconstruct_rcs_multi_freq(model, sample_idx=0, param_data=params, num_frequencies=3)
        >>> print(rcs_all.shape)  # (91, 91, 3)
    """
    rcs_grids = []

    for freq_idx in range(num_frequencies):
        rcs_grid = reconstruct_rcs_grid(
            model=model,
            sample_idx=sample_idx,
            freq_idx=freq_idx,
            param_data=param_data,
            device=device,
            theta_range=theta_range,
            phi_range=phi_range,
            grid_size=grid_size
        )
        rcs_grids.append(rcs_grid)

    # 堆叠为[91, 91, num_freq]
    rcs_multi_freq = np.stack(rcs_grids, axis=-1)

    return rcs_multi_freq
