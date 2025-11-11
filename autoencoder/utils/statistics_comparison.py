"""
统计对比分析工具

提供模型预测的全局统计对比分析功能，生成多维度的对比图表。
适用于GUI和批量实验。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
from typing import Dict, Optional, Tuple, List
from scipy import stats


def compute_global_statistics(
    ae_system: Dict,
    rcs_data: np.ndarray,
    param_data: Optional[np.ndarray] = None,
    device: str = 'cpu'
) -> Dict:
    """
    计算全局统计指标（核心计算逻辑）

    Args:
        ae_system: AutoEncoder系统字典
        rcs_data: RCS数据 [N, 91, 91, num_freq]
        param_data: 参数数据 [N, num_params]（Three-Stage模式需要）
        device: 计算设备

    Returns:
        统计结果字典，包含：
        - model_stats: 每个模型的统计指标列表
        - all_actual_1_5g: 所有真实RCS数据点（1.5GHz）
        - all_predicted_1_5g: 所有预测RCS数据点（1.5GHz）
        - all_actual_3g: 所有真实RCS数据点（3GHz）
        - all_predicted_3g: 所有预测RCS数据点（3GHz）
        - evaluation_mode: 评估模式标识
    """
    # 获取AutoEncoder组件
    autoencoder = ae_system['autoencoder']
    parameter_mapper = ae_system.get('parameter_mapper', None)
    wavelet_transform = ae_system.get('wavelet_transform', None)
    data_adapter = ae_system.get('data_adapter', None)
    training_mode = ae_system.get('training_mode', 'three_stage')

    # 确定评估模式
    if training_mode == 'stage1_only':
        evaluation_mode = "Stage1-Only RCS重建评估"
    else:
        evaluation_mode = "Three-Stage 参数预测RCS评估"

    # 设置模型为评估模式
    autoencoder.to(device).eval()
    if parameter_mapper is not None:
        parameter_mapper.to(device).eval()

    # 初始化数据收集列表
    all_actual_1_5g = []
    all_actual_3g = []
    all_predicted_1_5g = []
    all_predicted_3g = []
    model_stats = []

    print("="*60)
    print(f"统计分析模式: {evaluation_mode}")
    print(f"数据集大小: {len(rcs_data)} 个样本")
    print("="*60)

    with torch.no_grad():
        if training_mode == 'stage1_only':
            # ===== Stage1-Only模式：RCS重建评估 =====
            print("执行RCS重建评估...")

            # 准备输入数据：RCS → 标准化（+小波变换）
            if wavelet_transform is not None:
                # Wavelet模式：RCS → 小波变换 → 标准化小波系数
                rcs_tensor = torch.FloatTensor(rcs_data).to(device)
                wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)

                if data_adapter:
                    wavelet_coeffs_np = wavelet_coeffs.cpu().numpy()
                    input_adapted = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
                    input_tensor = torch.FloatTensor(input_adapted).to(device)
                else:
                    input_tensor = wavelet_coeffs
            else:
                # Direct模式：RCS → 标准化
                if data_adapter:
                    input_adapted = data_adapter.adapt_rcs_data(rcs_data)
                    input_tensor = torch.FloatTensor(input_adapted).to(device)
                else:
                    input_tensor = torch.FloatTensor(rcs_data).to(device)

            # AutoEncoder重建：Encode → Decode
            latents = autoencoder.encode(input_tensor)
            reconstructed_output = autoencoder.decode(latents)

            # 逆变换到RCS空间
            if wavelet_transform is not None:
                # 小波模式：逆标准化 → 逆小波变换 → RCS
                if data_adapter:
                    reconstructed_coeffs_np = data_adapter.inverse_adapt(reconstructed_output)
                    reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
                else:
                    reconstructed_coeffs = reconstructed_output

                predicted_batch = wavelet_transform.inverse_transform(reconstructed_coeffs).cpu().numpy()
            else:
                # 直接模式：逆标准化 → RCS
                if data_adapter:
                    predicted_batch = data_adapter.inverse_adapt(reconstructed_output)
                else:
                    predicted_batch = reconstructed_output.cpu().numpy()

        else:
            # ===== Three-Stage模式：参数预测RCS =====
            print("执行参数预测评估...")

            if param_data is None:
                raise ValueError("Three-Stage模式需要提供param_data参数")
            if parameter_mapper is None:
                raise ValueError("Three-Stage模式需要parameter_mapper")

            # 批量预测所有模型
            params_tensor = torch.FloatTensor(param_data).to(device)
            predicted_latents = parameter_mapper(params_tensor)
            predicted_output = autoencoder.decode(predicted_latents)

            # 根据模式处理输出
            if wavelet_transform is not None:
                # 小波模式：标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
                if data_adapter:
                    predicted_coeffs_np = data_adapter.inverse_adapt(predicted_output)
                    predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
                else:
                    predicted_coeffs = predicted_output

                predicted_batch = wavelet_transform.inverse_transform(predicted_coeffs).cpu().numpy()
            else:
                # 直接模式：标准化RCS → 逆标准化 → RCS
                if data_adapter:
                    predicted_batch = data_adapter.inverse_adapt(predicted_output)
                else:
                    predicted_batch = predicted_output.cpu().numpy()

    # 收集所有数据和统计信息
    for i in range(len(rcs_data)):
        model_id = f"{i+1:03d}"
        current_rcs_data = rcs_data[i]

        # ===== 真实数据 =====
        actual_1_5g = current_rcs_data[:, :, 0].flatten()
        actual_3g = current_rcs_data[:, :, 1].flatten()

        # ===== 预测数据 =====
        pred_1_5g = predicted_batch[i, :, :, 0].flatten()
        pred_3g = predicted_batch[i, :, :, 1].flatten()

        # 确保线性域数值为正
        pred_1_5g = np.maximum(pred_1_5g, 1e-12)
        pred_3g = np.maximum(pred_3g, 1e-12)

        # 计算每个模型的统计指标
        stats_1_5g = {
            'model_id': model_id,
            'freq': '1.5GHz',
            'actual_mean': np.mean(actual_1_5g),
            'actual_max': np.max(actual_1_5g),
            'actual_min': np.min(actual_1_5g),
            'predicted_mean': np.mean(pred_1_5g),
            'predicted_max': np.max(pred_1_5g),
            'predicted_min': np.min(pred_1_5g),
            'correlation': np.corrcoef(actual_1_5g, pred_1_5g)[0,1],
            'rmse': np.sqrt(np.mean((actual_1_5g - pred_1_5g)**2))
        }

        stats_3g = {
            'model_id': model_id,
            'freq': '3GHz',
            'actual_mean': np.mean(actual_3g),
            'actual_max': np.max(actual_3g),
            'actual_min': np.min(actual_3g),
            'predicted_mean': np.mean(pred_3g),
            'predicted_max': np.max(pred_3g),
            'predicted_min': np.min(pred_3g),
            'correlation': np.corrcoef(actual_3g, pred_3g)[0,1],
            'rmse': np.sqrt(np.mean((actual_3g - pred_3g)**2))
        }

        model_stats.extend([stats_1_5g, stats_3g])

        # 收集所有数据点用于散点图
        all_actual_1_5g.extend(actual_1_5g)
        all_actual_3g.extend(actual_3g)
        all_predicted_1_5g.extend(pred_1_5g)
        all_predicted_3g.extend(pred_3g)

    # 转换为numpy数组
    all_actual_1_5g = np.array(all_actual_1_5g)
    all_actual_3g = np.array(all_actual_3g)
    all_predicted_1_5g = np.array(all_predicted_1_5g)
    all_predicted_3g = np.array(all_predicted_3g)

    print(f"统计计算完成: 处理了 {len(rcs_data)} 个模型")

    return {
        'model_stats': model_stats,
        'all_actual_1_5g': all_actual_1_5g,
        'all_predicted_1_5g': all_predicted_1_5g,
        'all_actual_3g': all_actual_3g,
        'all_predicted_3g': all_predicted_3g,
        'evaluation_mode': evaluation_mode
    }


def plot_statistics_comparison(
    stats_result: Dict,
    save_dir: str,
    figsize: Tuple[int, int] = (15, 10),
    fontsize_scale: float = 1.0,
    return_fig: bool = False
) -> Optional[Figure]:
    """
    绘制统计对比图表

    Args:
        stats_result: compute_global_statistics返回的统计结果
        save_dir: 保存目录
        figsize: 图表尺寸
        fontsize_scale: 字号缩放因子
        return_fig: 是否返回Figure对象（GUI使用）

    Returns:
        如果return_fig=True，返回Figure对象；否则返回None
    """
    # 提取数据
    model_stats = stats_result['model_stats']
    all_actual_1_5g = stats_result['all_actual_1_5g']
    all_predicted_1_5g = stats_result['all_predicted_1_5g']
    all_actual_3g = stats_result['all_actual_3g']
    all_predicted_3g = stats_result['all_predicted_3g']
    evaluation_mode = stats_result['evaluation_mode']

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建主图表
    fig = plt.figure(figsize=figsize)

    # ===== 子图1: 1.5GHz 模型均值对比图 (dBsm单位) =====
    ax1 = fig.add_subplot(2, 3, 1)
    stats_1_5_list = [s for s in model_stats if s['freq'] == '1.5GHz']

    # 提取各个模型的均值，转换为dBsm
    model_ids = [s['model_id'] for s in stats_1_5_list]
    actual_means_linear = [s['actual_mean'] for s in stats_1_5_list]
    predicted_means_linear = [s['predicted_mean'] for s in stats_1_5_list]

    # 转换为dBsm: RCS_dBsm = 10*log10(RCS_linear)
    actual_means_dbsm = [10 * np.log10(max(val, 1e-12)) for val in actual_means_linear]
    predicted_means_dbsm = [10 * np.log10(max(val, 1e-12)) for val in predicted_means_linear]

    ax1.scatter(actual_means_dbsm, predicted_means_dbsm, alpha=0.8, s=50, color='blue')
    # 添加理想预测线
    min_val = min(min(actual_means_dbsm), min(predicted_means_dbsm))
    max_val = max(max(actual_means_dbsm), max(predicted_means_dbsm))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测线')

    # 计算均值的相关系数
    r1 = np.corrcoef(actual_means_dbsm, predicted_means_dbsm)[0,1]
    ax1.set_xlabel('真实均值 (dBsm)', fontsize=int(12*fontsize_scale))
    ax1.set_ylabel('预测均值 (dBsm)', fontsize=int(12*fontsize_scale))
    ax1.set_title(f'1.5GHz 模型均值对比\nR = {r1:.4f}, 模型数: {len(model_ids)}\n[{evaluation_mode}]',
                  fontsize=int(12*fontsize_scale))
    ax1.legend(fontsize=int(10*fontsize_scale))
    ax1.grid(True, alpha=0.3)

    # ===== 子图2: 3GHz 模型均值对比图 (dBsm单位) =====
    ax2 = fig.add_subplot(2, 3, 2)
    stats_3_list = [s for s in model_stats if s['freq'] == '3GHz']

    # 提取各个模型的均值，转换为dBsm
    actual_means_linear_3g = [s['actual_mean'] for s in stats_3_list]
    predicted_means_linear_3g = [s['predicted_mean'] for s in stats_3_list]

    actual_means_dbsm_3g = [10 * np.log10(max(val, 1e-12)) for val in actual_means_linear_3g]
    predicted_means_dbsm_3g = [10 * np.log10(max(val, 1e-12)) for val in predicted_means_linear_3g]

    ax2.scatter(actual_means_dbsm_3g, predicted_means_dbsm_3g, alpha=0.8, s=50, color='red')
    # 添加理想预测线
    min_val_3g = min(min(actual_means_dbsm_3g), min(predicted_means_dbsm_3g))
    max_val_3g = max(max(actual_means_dbsm_3g), max(predicted_means_dbsm_3g))
    ax2.plot([min_val_3g, max_val_3g], [min_val_3g, max_val_3g], 'r--', alpha=0.8, linewidth=2, label='理想预测线')

    # 计算均值的相关系数
    r2 = np.corrcoef(actual_means_dbsm_3g, predicted_means_dbsm_3g)[0,1]
    ax2.set_xlabel('真实均值 (dBsm)', fontsize=int(12*fontsize_scale))
    ax2.set_ylabel('预测均值 (dBsm)', fontsize=int(12*fontsize_scale))
    ax2.set_title(f'3GHz 模型均值对比\nR = {r2:.4f}, 模型数: {len(model_ids)}\n[{evaluation_mode}]',
                  fontsize=int(12*fontsize_scale))
    ax2.legend(fontsize=int(10*fontsize_scale))
    ax2.grid(True, alpha=0.3)

    # ===== 子图3: 统计指标对比图（均值、最大值、最小值） =====
    ax3 = fig.add_subplot(2, 3, 3)

    metrics = ['均值', '最大值', '最小值']
    actual_1_5_means = [np.mean([s['actual_mean'] for s in stats_1_5_list]),
                       np.mean([s['actual_max'] for s in stats_1_5_list]),
                       np.mean([s['actual_min'] for s in stats_1_5_list])]
    predicted_1_5_means = [np.mean([s['predicted_mean'] for s in stats_1_5_list]),
                          np.mean([s['predicted_max'] for s in stats_1_5_list]),
                          np.mean([s['predicted_min'] for s in stats_1_5_list])]
    actual_3_means = [np.mean([s['actual_mean'] for s in stats_3_list]),
                     np.mean([s['actual_max'] for s in stats_3_list]),
                     np.mean([s['actual_min'] for s in stats_3_list])]
    predicted_3_means = [np.mean([s['predicted_mean'] for s in stats_3_list]),
                        np.mean([s['predicted_max'] for s in stats_3_list]),
                        np.mean([s['predicted_min'] for s in stats_3_list])]

    x = np.arange(len(metrics))
    width = 0.2
    ax3.bar(x - 1.5*width, actual_1_5_means, width, label='1.5GHz 真实', color='lightblue', alpha=0.8)
    ax3.bar(x - 0.5*width, predicted_1_5_means, width, label='1.5GHz 预测', color='blue', alpha=0.8)
    ax3.bar(x + 0.5*width, actual_3_means, width, label='3GHz 真实', color='lightcoral', alpha=0.8)
    ax3.bar(x + 1.5*width, predicted_3_means, width, label='3GHz 预测', color='red', alpha=0.8)
    ax3.set_xlabel('统计指标', fontsize=int(12*fontsize_scale))
    ax3.set_ylabel('RCS值 (线性)', fontsize=int(12*fontsize_scale))
    ax3.set_title('统计指标对比', fontsize=int(12*fontsize_scale))
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=int(10*fontsize_scale))
    ax3.legend(fontsize=int(8*fontsize_scale))
    ax3.grid(True, alpha=0.3)

    # ===== 子图4: 性能指标 - 相关系数 =====
    ax4 = fig.add_subplot(2, 3, 4)
    models = [s['model_id'] for s in stats_1_5_list]
    corr_1_5 = [s['correlation'] for s in stats_1_5_list]
    corr_3 = [s['correlation'] for s in stats_3_list]
    x = np.arange(len(models))
    ax4.bar(x - 0.2, corr_1_5, 0.4, label='1.5GHz', alpha=0.7)
    ax4.bar(x + 0.2, corr_3, 0.4, label='3GHz', alpha=0.7)
    ax4.set_xlabel('模型ID', fontsize=int(12*fontsize_scale))
    ax4.set_ylabel('相关系数', fontsize=int(12*fontsize_scale))
    ax4.set_title('预测相关性对比', fontsize=int(12*fontsize_scale))
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontsize=int(8*fontsize_scale))
    ax4.legend(fontsize=int(10*fontsize_scale))
    ax4.grid(True, alpha=0.3)

    # ===== 子图5: RMSE对比 =====
    ax5 = fig.add_subplot(2, 3, 5)
    rmse_1_5 = [s['rmse'] for s in stats_1_5_list]
    rmse_3 = [s['rmse'] for s in stats_3_list]
    ax5.bar(x - 0.2, rmse_1_5, 0.4, label='1.5GHz', alpha=0.7)
    ax5.bar(x + 0.2, rmse_3, 0.4, label='3GHz', alpha=0.7)
    ax5.set_xlabel('模型ID', fontsize=int(12*fontsize_scale))
    ax5.set_ylabel('RMSE', fontsize=int(12*fontsize_scale))
    ax5.set_title('预测误差对比', fontsize=int(12*fontsize_scale))
    ax5.set_xticks(x)
    ax5.set_xticklabels(models, fontsize=int(8*fontsize_scale))
    ax5.legend(fontsize=int(10*fontsize_scale))
    ax5.grid(True, alpha=0.3)

    # ===== 子图6: 整体性能汇总 =====
    ax6 = fig.add_subplot(2, 3, 6)
    avg_r1 = np.mean(corr_1_5)
    avg_r2 = np.mean(corr_3)
    avg_rmse1 = np.mean(rmse_1_5)
    avg_rmse2 = np.mean(rmse_3)

    summary_text = f"""整体性能统计：

1.5GHz:
  平均相关系数: {avg_r1:.4f}
  平均RMSE: {avg_rmse1:.3e}
  模型数量: {len(stats_1_5_list)}

3GHz:
  平均相关系数: {avg_r2:.4f}
  平均RMSE: {avg_rmse2:.3e}
  模型数量: {len(stats_3_list)}

总体:
  总模型数: {len(model_stats)//2}
  数据点数: {len(all_actual_1_5g) + len(all_actual_3g)}"""

    ax6.text(0.1, 0.1, summary_text, fontsize=int(10*fontsize_scale),
            verticalalignment='bottom', transform=ax6.transAxes)
    ax6.axis('off')
    ax6.set_title('性能汇总统计', fontsize=int(12*fontsize_scale))

    fig.tight_layout()

    # 保存主图表
    main_plot_path = os.path.join(save_dir, 'statistics_comparison.png')
    fig.savefig(main_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 统计对比图已保存: {main_plot_path}")

    # 保存散点图
    _save_scatter_plots(all_actual_1_5g, all_predicted_1_5g, all_actual_3g, all_predicted_3g,
                       save_dir, fontsize_scale)

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None


def _save_scatter_plots(
    all_actual_1_5g: np.ndarray,
    all_predicted_1_5g: np.ndarray,
    all_actual_3g: np.ndarray,
    all_predicted_3g: np.ndarray,
    save_dir: str,
    fontsize_scale: float = 1.0
):
    """保存散点图到文件（内部函数）"""
    try:
        # 创建散点图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 1.5GHz 散点图
        sample_size = min(5000, len(all_actual_1_5g))
        indices = np.random.choice(len(all_actual_1_5g), sample_size, replace=False)

        # 转换为dBsm单位
        x1_linear = all_actual_1_5g[indices]
        y1_linear = all_predicted_1_5g[indices]
        x1_linear = np.maximum(x1_linear, 1e-12)
        y1_linear = np.maximum(y1_linear, 1e-12)
        x1_dbsm = 10 * np.log10(x1_linear)
        y1_dbsm = 10 * np.log10(y1_linear)

        ax1.scatter(x1_dbsm, y1_dbsm, alpha=0.3, s=1, color='blue', rasterized=True)
        min_val, max_val = min(x1_dbsm.min(), y1_dbsm.min()), max(x1_dbsm.max(), y1_dbsm.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测线')
        r1 = np.corrcoef(x1_dbsm, y1_dbsm)[0,1]
        ax1.set_xlabel('真实值 (dBsm)', fontsize=int(12*fontsize_scale))
        ax1.set_ylabel('预测值 (dBsm)', fontsize=int(12*fontsize_scale))
        ax1.set_title(f'1.5GHz 全数据点散点图\nR = {r1:.4f}, 点数: {len(x1_dbsm)}',
                     fontsize=int(12*fontsize_scale))
        ax1.legend(fontsize=int(10*fontsize_scale))
        ax1.grid(True, alpha=0.3)

        # 3GHz 散点图
        indices = np.random.choice(len(all_actual_3g), sample_size, replace=False)
        x2_linear = all_actual_3g[indices]
        y2_linear = all_predicted_3g[indices]
        x2_linear = np.maximum(x2_linear, 1e-12)
        y2_linear = np.maximum(y2_linear, 1e-12)
        x2_dbsm = 10 * np.log10(x2_linear)
        y2_dbsm = 10 * np.log10(y2_linear)

        ax2.scatter(x2_dbsm, y2_dbsm, alpha=0.3, s=1, color='red', rasterized=True)
        min_val, max_val = min(x2_dbsm.min(), y2_dbsm.min()), max(x2_dbsm.max(), y2_dbsm.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测线')
        r2 = np.corrcoef(x2_dbsm, y2_dbsm)[0,1]
        ax2.set_xlabel('真实值 (dBsm)', fontsize=int(12*fontsize_scale))
        ax2.set_ylabel('预测值 (dBsm)', fontsize=int(12*fontsize_scale))
        ax2.set_title(f'3GHz 全数据点散点图\nR = {r2:.4f}, 点数: {len(x2_dbsm)}',
                     fontsize=int(12*fontsize_scale))
        ax2.legend(fontsize=int(10*fontsize_scale))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_plot_path = os.path.join(save_dir, 'scatter_plots.png')
        plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 散点图已保存: {scatter_plot_path}")

    except Exception as e:
        print(f"⚠ 保存散点图失败: {e}")


def generate_statistics_comparison(
    ae_system: Dict,
    rcs_data: np.ndarray,
    param_data: Optional[np.ndarray],
    save_dir: str,
    device: str = 'cpu',
    figsize: Tuple[int, int] = (15, 10),
    fontsize_scale: float = 1.0
) -> Dict:
    """
    生成统计对比分析（一站式接口）

    Args:
        ae_system: AutoEncoder系统
        rcs_data: RCS数据
        param_data: 参数数据（Three-Stage模式需要）
        save_dir: 保存目录
        device: 计算设备
        figsize: 图表尺寸
        fontsize_scale: 字号缩放因子

    Returns:
        统计结果字典
    """
    # 计算统计指标
    stats_result = compute_global_statistics(ae_system, rcs_data, param_data, device)

    # 绘制并保存图表
    plot_statistics_comparison(stats_result, save_dir, figsize, fontsize_scale, return_fig=False)

    return stats_result
