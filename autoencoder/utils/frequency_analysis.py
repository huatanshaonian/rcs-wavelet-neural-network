"""
频域分析工具
用于分析Additive Dual-Branch AutoEncoder的高频和低频分支输出的频率特征
"""

import numpy as np
import torch
from typing import Tuple, Dict


def compute_2d_fft_spectrum(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算2D FFT频谱

    Args:
        data: [H, W] 2D数组

    Returns:
        freq_magnitude: [H, W] 频谱幅度（已中心化）
        freq_phase: [H, W] 频谱相位（已中心化）
    """
    # 2D FFT
    fft_result = np.fft.fft2(data)

    # 中心化（将零频率分量移到中心）
    fft_shifted = np.fft.fftshift(fft_result)

    # 幅度和相位
    freq_magnitude = np.abs(fft_shifted)
    freq_phase = np.angle(fft_shifted)

    return freq_magnitude, freq_phase


def compute_radial_frequency_profile(freq_magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算径向频率分布（从中心向外的频率能量分布）

    Args:
        freq_magnitude: [H, W] 频谱幅度（中心化）

    Returns:
        radial_distances: [N] 径向距离
        radial_spectrum: [N] 对应距离的平均频谱能量
    """
    h, w = freq_magnitude.shape
    center_y, center_x = h // 2, w // 2

    # 创建径向距离矩阵
    y, x = np.ogrid[:h, :w]
    radial_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

    # 计算每个径向距离的平均能量
    max_radius = int(np.sqrt(center_x**2 + center_y**2))
    radial_spectrum = np.zeros(max_radius + 1)

    for r in range(max_radius + 1):
        mask = (radial_dist == r)
        if np.sum(mask) > 0:
            radial_spectrum[r] = np.mean(freq_magnitude[mask])

    radial_distances = np.arange(max_radius + 1)

    return radial_distances, radial_spectrum


def compute_frequency_energy_ratio(freq_magnitude: np.ndarray, cutoff_ratio: float = 0.3) -> Dict[str, float]:
    """
    计算高频和低频能量比例

    Args:
        freq_magnitude: [H, W] 频谱幅度（中心化）
        cutoff_ratio: 低频截止比例（0-1），默认0.3表示中心30%半径内为低频

    Returns:
        {
            'low_freq_energy': 低频能量占比,
            'high_freq_energy': 高频能量占比,
            'total_energy': 总能量
        }
    """
    h, w = freq_magnitude.shape
    center_y, center_x = h // 2, w // 2

    # 创建径向距离矩阵
    y, x = np.ogrid[:h, :w]
    radial_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # 定义低频区域（中心附近）
    max_radius = np.sqrt(center_x**2 + center_y**2)
    low_freq_mask = radial_dist <= (cutoff_ratio * max_radius)

    # 计算能量（幅度平方）
    energy = freq_magnitude ** 2
    total_energy = np.sum(energy)
    low_freq_energy = np.sum(energy[low_freq_mask])
    high_freq_energy = total_energy - low_freq_energy

    return {
        'low_freq_energy': low_freq_energy / total_energy if total_energy > 0 else 0,
        'high_freq_energy': high_freq_energy / total_energy if total_energy > 0 else 0,
        'total_energy': total_energy
    }


def analyze_branch_frequencies(branch_output: np.ndarray,
                               freq_idx: int = 0,
                               cutoff_ratio: float = 0.3) -> Dict[str, any]:
    """
    分析分支输出的频率特征（单个频率通道）

    Args:
        branch_output: [H, W, num_freq] 分支输出
        freq_idx: 要分析的频率索引
        cutoff_ratio: 低频截止比例

    Returns:
        {
            'freq_magnitude': 频谱幅度,
            'freq_phase': 频谱相位,
            'radial_distances': 径向距离,
            'radial_spectrum': 径向频谱,
            'energy_ratio': 能量比例字典
        }
    """
    # 提取指定频率通道
    data_2d = branch_output[:, :, freq_idx]

    # 计算FFT频谱
    freq_magnitude, freq_phase = compute_2d_fft_spectrum(data_2d)

    # 计算径向分布
    radial_distances, radial_spectrum = compute_radial_frequency_profile(freq_magnitude)

    # 计算能量比例
    energy_ratio = compute_frequency_energy_ratio(freq_magnitude, cutoff_ratio)

    return {
        'freq_magnitude': freq_magnitude,
        'freq_phase': freq_phase,
        'radial_distances': radial_distances,
        'radial_spectrum': radial_spectrum,
        'energy_ratio': energy_ratio
    }


def compare_branch_frequency_characteristics(high_branch: np.ndarray,
                                            smooth_branch: np.ndarray,
                                            freq_idx: int = 0,
                                            cutoff_ratio: float = 0.3) -> Dict[str, any]:
    """
    对比高频和低频分支的频率特征

    Args:
        high_branch: [H, W, num_freq] 高频分支输出
        smooth_branch: [H, W, num_freq] 低频分支输出
        freq_idx: 要分析的频率索引
        cutoff_ratio: 低频截止比例

    Returns:
        {
            'high': 高频分支分析结果,
            'smooth': 低频分支分析结果,
            'comparison': {
                'high_dominance': 高频分支的高频能量占比,
                'smooth_dominance': 低频分支的低频能量占比
            }
        }
    """
    # 分析两个分支
    high_analysis = analyze_branch_frequencies(high_branch, freq_idx, cutoff_ratio)
    smooth_analysis = analyze_branch_frequencies(smooth_branch, freq_idx, cutoff_ratio)

    # 对比分析
    comparison = {
        'high_dominance': high_analysis['energy_ratio']['high_freq_energy'],
        'smooth_dominance': smooth_analysis['energy_ratio']['low_freq_energy']
    }

    return {
        'high': high_analysis,
        'smooth': smooth_analysis,
        'comparison': comparison
    }


def batch_analyze_branches(high_branches: torch.Tensor,
                           smooth_branches: torch.Tensor,
                           sample_idx: int = 0,
                           freq_idx: int = 0) -> Dict[str, any]:
    """
    批量分析分支输出（便捷函数）

    Args:
        high_branches: [B, H, W, C] 高频分支批量输出
        smooth_branches: [B, H, W, C] 低频分支批量输出
        sample_idx: 要分析的样本索引
        freq_idx: 要分析的频率索引

    Returns:
        对比分析结果
    """
    # 转换为numpy并提取单个样本
    high_np = high_branches[sample_idx].cpu().numpy() if isinstance(high_branches, torch.Tensor) else high_branches[sample_idx]
    smooth_np = smooth_branches[sample_idx].cpu().numpy() if isinstance(smooth_branches, torch.Tensor) else smooth_branches[sample_idx]

    return compare_branch_frequency_characteristics(high_np, smooth_np, freq_idx)
