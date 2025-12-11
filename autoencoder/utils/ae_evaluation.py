"""
AutoEncoder评估功能

提供纯函数式的评估指标计算，供GUI和批量实验复用。

核心功能：
1. compute_reconstruction_metrics() - 计算重建质量指标
2. compute_prediction_metrics() - 计算预测质量指标
3. format_metrics_string() - 格式化指标字符串

作者：Claude Code
日期：2025-01-10
"""

import numpy as np
from typing import Dict, Optional, Tuple


def compute_reconstruction_metrics(
    true_rcs: np.ndarray,
    pred_rcs: np.ndarray,
    per_frequency: bool = False
) -> Dict[str, float]:
    """
    计算重建质量指标（纯函数）

    Args:
        true_rcs: 真实RCS数据 [N, H, W, num_freq]
        pred_rcs: 预测RCS数据 [N, H, W, num_freq]
        per_frequency: 是否分频率计算指标（默认False，计算全局指标）

    Returns:
        指标字典:
            - 'mse': 均方误差
            - 'rmse': 均方根误差
            - 'mae': 平均绝对误差
            - 'max_error': 最大误差
            - 'relative_error': 相对误差（百分比）
            - 如果per_frequency=True，还包含:
                - 'mse_per_freq': list[float]
                - 'rmse_per_freq': list[float]
                - 'mae_per_freq': list[float]

    示例:
        >>> metrics = compute_reconstruction_metrics(true_rcs, pred_rcs)
        >>> print(f"RMSE: {metrics['rmse']:.6f}")
    """
    # 验证输入形状
    if true_rcs.shape != pred_rcs.shape:
        raise ValueError(
            f"形状不匹配: true_rcs {true_rcs.shape} vs pred_rcs {pred_rcs.shape}"
        )

    # 计算全局指标
    diff = pred_rcs - true_rcs
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))

    # 计算相对误差（避免除零）
    true_rms = np.sqrt(np.mean(true_rcs ** 2))
    relative_error = (rmse / true_rms * 100) if true_rms > 1e-10 else 0.0

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'max_error': float(max_error),
        'relative_error': float(relative_error)
    }

    # 如果需要分频率计算
    if per_frequency and true_rcs.ndim == 4:
        num_freq = true_rcs.shape[-1]
        mse_per_freq = []
        rmse_per_freq = []
        mae_per_freq = []

        for f in range(num_freq):
            diff_f = pred_rcs[..., f] - true_rcs[..., f]
            mse_f = np.mean(diff_f ** 2)
            rmse_f = np.sqrt(mse_f)
            mae_f = np.mean(np.abs(diff_f))

            mse_per_freq.append(float(mse_f))
            rmse_per_freq.append(float(rmse_f))
            mae_per_freq.append(float(mae_f))

        metrics['mse_per_freq'] = mse_per_freq
        metrics['rmse_per_freq'] = rmse_per_freq
        metrics['mae_per_freq'] = mae_per_freq

    return metrics


def compute_prediction_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray
) -> Dict[str, float]:
    """
    计算预测质量指标（通用版本）

    Args:
        true_values: 真实值 [N, ...]
        pred_values: 预测值 [N, ...]

    Returns:
        指标字典（同 compute_reconstruction_metrics）

    注意：
        这是 compute_reconstruction_metrics 的别名，
        语义上更适合用于参数预测等场景。
    """
    return compute_reconstruction_metrics(true_values, pred_values, per_frequency=False)


def compute_latent_space_metrics(
    latents1: np.ndarray,
    latents2: np.ndarray
) -> Dict[str, float]:
    """
    计算隐空间表示的距离指标

    Args:
        latents1: 第一组隐空间表示 [N, latent_dim]
        latents2: 第二组隐空间表示 [N, latent_dim]

    Returns:
        指标字典:
            - 'euclidean_distance': 欧氏距离
            - 'cosine_similarity': 余弦相似度
            - 'mse': 均方误差
    """
    if latents1.shape != latents2.shape:
        raise ValueError(
            f"形状不匹配: latents1 {latents1.shape} vs latents2 {latents2.shape}"
        )

    # 欧氏距离
    euclidean_dist = np.mean(np.linalg.norm(latents1 - latents2, axis=1))

    # 余弦相似度（平均）
    norm1 = np.linalg.norm(latents1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(latents2, axis=1, keepdims=True)
    cosine_sim = np.mean(
        np.sum(latents1 * latents2, axis=1) / (norm1.flatten() * norm2.flatten() + 1e-10)
    )

    # MSE
    mse = np.mean((latents1 - latents2) ** 2)

    return {
        'euclidean_distance': float(euclidean_dist),
        'cosine_similarity': float(cosine_sim),
        'mse': float(mse)
    }


def format_metrics_string(
    metrics: Dict[str, float],
    include_keys: Optional[list] = None,
    precision: int = 6
) -> str:
    """
    格式化指标字典为可读字符串

    Args:
        metrics: 指标字典
        include_keys: 要包含的键列表（None表示全部）
        precision: 浮点数精度

    Returns:
        格式化的字符串

    示例:
        >>> metrics = {'mse': 0.001234, 'rmse': 0.035, 'mae': 0.028}
        >>> print(format_metrics_string(metrics))
        MSE: 0.001234, RMSE: 0.035000, MAE: 0.028000
    """
    if include_keys is None:
        include_keys = list(metrics.keys())

    parts = []
    for key in include_keys:
        if key in metrics:
            value = metrics[key]
            # 特殊处理：relative_error显示百分号
            if key == 'relative_error':
                parts.append(f"{key.upper()}: {value:.{precision}f}%")
            else:
                parts.append(f"{key.upper()}: {value:.{precision}f}")

    return ", ".join(parts)


def compare_models(
    true_rcs: np.ndarray,
    pred_rcs_dict: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    对比多个模型的预测结果

    Args:
        true_rcs: 真实RCS数据 [N, H, W, num_freq]
        pred_rcs_dict: 模型名称 → 预测RCS的字典

    Returns:
        模型名称 → 指标字典 的字典

    示例:
        >>> results = compare_models(true_rcs, {
        ...     'model_A': pred_rcs_a,
        ...     'model_B': pred_rcs_b
        ... })
        >>> print(f"Model A RMSE: {results['model_A']['rmse']}")
    """
    comparison = {}
    for model_name, pred_rcs in pred_rcs_dict.items():
        metrics = compute_reconstruction_metrics(true_rcs, pred_rcs)
        comparison[model_name] = metrics

    return comparison


def get_best_model(
    comparison: Dict[str, Dict[str, float]],
    metric: str = 'rmse'
) -> Tuple[str, float]:
    """
    从对比结果中找出最佳模型

    Args:
        comparison: 来自 compare_models() 的结果
        metric: 评价指标（默认'rmse'，越小越好）

    Returns:
        (best_model_name, best_metric_value)

    示例:
        >>> comparison = compare_models(...)
        >>> best_name, best_rmse = get_best_model(comparison, 'rmse')
        >>> print(f"Best model: {best_name} (RMSE={best_rmse})")
    """
    best_model = None
    best_value = float('inf')

    for model_name, metrics in comparison.items():
        if metric in metrics:
            value = metrics[metric]
            if value < best_value:
                best_value = value
                best_model = model_name

    if best_model is None:
        raise ValueError(f"指标 '{metric}' 在任何模型中都未找到")

    return best_model, best_value


# ========== 辅助函数 ==========

def compute_frequency_wise_correlation(
    true_rcs: np.ndarray,
    pred_rcs: np.ndarray
) -> np.ndarray:
    """
    计算每个频率的相关系数

    Args:
        true_rcs: [N, H, W, num_freq]
        pred_rcs: [N, H, W, num_freq]

    Returns:
        [num_freq] 相关系数数组
    """
    if true_rcs.ndim != 4 or pred_rcs.ndim != 4:
        raise ValueError("输入必须是4D数组 [N, H, W, num_freq]")

    num_freq = true_rcs.shape[-1]
    correlations = []

    for f in range(num_freq):
        true_flat = true_rcs[..., f].flatten()
        pred_flat = pred_rcs[..., f].flatten()

        # Pearson相关系数
        corr = np.corrcoef(true_flat, pred_flat)[0, 1]
        correlations.append(corr)

    return np.array(correlations)


def compute_spatial_error_map(
    true_rcs: np.ndarray,
    pred_rcs: np.ndarray,
    error_type: str = 'absolute'
) -> np.ndarray:
    """
    计算空间误差分布图

    Args:
        true_rcs: [N, H, W, num_freq]
        pred_rcs: [N, H, W, num_freq]
        error_type: 'absolute' 或 'squared'

    Returns:
        [H, W, num_freq] 平均误差图
    """
    if error_type == 'absolute':
        error = np.abs(pred_rcs - true_rcs)
    elif error_type == 'squared':
        error = (pred_rcs - true_rcs) ** 2
    else:
        raise ValueError(f"未知的error_type: {error_type}")

    # 对样本维度求平均
    error_map = np.mean(error, axis=0)

    return error_map
