"""
AutoEncoder可视化绘图功能

提供纯函数式的绘图接口，供GUI和批量实验复用。

核心功能：
1. plot_rcs_comparison() - 绘制RCS对比图（真实 vs 预测 vs 残差）
2. plot_rcs_heatmap() - 绘制单个RCS热图
3. plot_latent_space_2d() - 绘制隐空间2D投影
4. plot_training_curves() - 绘制训练曲线（简化版）
5. plot_ae_training_progress() - 绘制AE三阶段训练进度（完整版，支持最佳epoch标记）
6. plot_wavelet_coefficients_comparison() - 绘制小波系数对比（原始 vs 重建 vs 残差，4通道）

设计原则：
- 纯函数，接受数据作为参数
- 返回matplotlib.figure.Figure对象或直接保存到文件
- GUI可以在Canvas上显示，批量实验可以保存到文件
- 不依赖GUI组件，不弹窗，不读取文件

作者：Claude Code
日期：2025-01-10
更新：
  - 2025-01-10: 添加plot_ae_training_progress()统一绘制AE训练进度
  - 2025-01-10: 添加plot_wavelet_coefficients_comparison()统一绘制小波系数对比
  - 2025-01-11: 配置科研图片标准字体（Arial/Helvetica，国际顶刊标准）
  - 2025-01-11: 修复Unicode减号报错（\u2212），提升图片出版质量
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Dict

# ============================================================================
# 科研图片字体配置（国际顶刊标准）
# ============================================================================
# 推荐字体：Arial（Nature, Science, Cell等顶刊常用）
# 备用字体：Helvetica, DejaVu Sans（开源，支持Unicode）
#
# 解决Unicode减号报错：
# - 方案1: 使用支持Unicode的字体（Arial, DejaVu Sans）
# - 方案2: 禁用Unicode减号（使用普通连字符）
# ============================================================================

# 强制刷新字体管理器（解决中文字体缓存问题）
import matplotlib.font_manager as fm
# 重建字体缓存（如果有中文显示问题）
try:
    fm._load_fontmanager(try_read_cache=False)
except:
    pass  # 如果失败，继续使用缓存

# 配置matplotlib全局参数（应用于所有图片）
# 重要：中文字体必须放在前面，matplotlib不会自动fallback到后面的字体
# 解决方案：使用Microsoft YaHei作为主字体（同时支持中英文）
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 禁用Unicode减号，避免字体不支持时出现dummy symbol

# 注意：
# - Microsoft YaHei (微软雅黑)：同时支持中英文，优先使用
# - SimHei (黑体)：中文备用字体
# - Arial：英文专用字体（国际顶刊标准），但不支持中文
# - 英文字符也会使用Microsoft YaHei显示，字体清晰专业
# - 如果需要纯英文+Arial，可以在特定函数中单独设置

# 同时设置plt.rcParams（确保兼容性）
plt.rcParams.update(matplotlib.rcParams)
plt.rcParams['font.size'] = 10  # 基础字号（科研图片标准）
plt.rcParams['axes.labelsize'] = 11  # 坐标轴标签
plt.rcParams['axes.titlesize'] = 12  # 标题
plt.rcParams['xtick.labelsize'] = 10  # x轴刻度
plt.rcParams['ytick.labelsize'] = 10  # y轴刻度
plt.rcParams['legend.fontsize'] = 10  # 图例
plt.rcParams['figure.titlesize'] = 13  # 图形标题

# 高质量输出配置
plt.rcParams['figure.dpi'] = 100  # 显示分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存分辨率（出版质量）
plt.rcParams['savefig.bbox'] = 'tight'  # 紧凑布局
plt.rcParams['savefig.pad_inches'] = 0.1  # 边距

# 线条和网格配置
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_rcs_comparison(
    true_rcs: np.ndarray,
    pred_rcs: np.ndarray,
    freq_label: str = "Unknown",
    model_id: str = "Unknown",
    phi_range: Tuple[float, float] = (-45, 45),
    theta_range: Tuple[float, float] = (45, 135),
    figsize: Tuple[int, int] = (15, 5),
    fontsize_scale: float = 1.0,
    save_path: Optional[str] = None,
    fig: Optional[Figure] = None
) -> Figure:
    """
    绘制RCS对比图：真实 vs 预测 vs 残差

    Args:
        true_rcs: 真实RCS数据 [H, W]，线性值
        pred_rcs: 预测RCS数据 [H, W]，线性值
        freq_label: 频率标签（如 "1.5 GHz"）
        model_id: 模型ID（如 "001"）
        phi_range: 方位角范围 (min, max)
        theta_range: 俯仰角范围 (min, max)
        figsize: 图像尺寸
        fontsize_scale: 字号缩放因子
        save_path: 保存路径（None表示不保存）
        fig: 复用的Figure对象（None表示创建新的）

    Returns:
        matplotlib.figure.Figure对象

    示例:
        >>> fig = plot_rcs_comparison(true_rcs, pred_rcs, "1.5 GHz", "001")
        >>> plt.show()  # GUI显示

        >>> plot_rcs_comparison(true_rcs, pred_rcs, "1.5 GHz", "001",
        ...                     save_path="comparison.png")  # 批量实验保存
    """
    # 创建或复用figure
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()

    # 转换为dB
    true_rcs_db = 10 * np.log10(true_rcs + 1e-10)
    pred_rcs_db = 10 * np.log10(pred_rcs + 1e-10)
    residual_db = true_rcs_db - pred_rcs_db

    # 计算指标
    mse_linear = np.mean((true_rcs - pred_rcs) ** 2)
    residual_finite = residual_db[np.isfinite(residual_db)]
    mae_db = np.mean(np.abs(residual_finite)) if len(residual_finite) > 0 else 0

    # 设置extent（正确显示角度范围）
    extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

    # 子图1: 真实RCS
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(true_rcs_db, cmap='jet', aspect='equal', extent=extent)
    ax1.set_title(f'真实RCS\n模型{model_id} @ {freq_label}',
                  fontsize=int(20*fontsize_scale), fontweight='bold')
    ax1.set_xlabel('φ (方位角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax1.set_ylabel('θ (俯仰角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
    cbar1.ax.tick_params(labelsize=int(16*fontsize_scale))

    # 获取colorbar范围
    vmin, vmax = im1.get_clim()

    # 子图2: 预测RCS（使用相同colorbar范围）
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(pred_rcs_db, cmap='jet', aspect='equal',
                     vmin=vmin, vmax=vmax, extent=extent)
    ax2.set_title(f'预测RCS\nMSE={mse_linear:.4e}',
                  fontsize=int(20*fontsize_scale), fontweight='bold')
    ax2.set_xlabel('φ (方位角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax2.set_ylabel('θ (俯仰角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
    cbar2.ax.tick_params(labelsize=int(16*fontsize_scale))

    # 子图3: 残差图（对称colorbar）
    ax3 = fig.add_subplot(1, 3, 3)
    residual_abs_max = np.percentile(np.abs(residual_finite), 95) if len(residual_finite) > 0 else 10
    im3 = ax3.imshow(residual_db, cmap='RdBu_r', aspect='equal',
                     vmin=-residual_abs_max, vmax=residual_abs_max, extent=extent)
    ax3.set_title(f'残差（真实-预测）\nMAE={mae_db:.2f} dB',
                  fontsize=int(20*fontsize_scale), fontweight='bold')
    ax3.set_xlabel('φ (方位角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax3.set_ylabel('θ (俯仰角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax3)
    cbar3.set_label('残差 (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
    cbar3.ax.tick_params(labelsize=int(16*fontsize_scale))

    fig.suptitle(f'AutoEncoder对比分析 - 模型{model_id} @ {freq_label}',
                 fontsize=int(24*fontsize_scale), fontweight='bold')
    fig.tight_layout()

    # 保存到文件（如果提供了路径）
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_rcs_heatmap(
    rcs_data: np.ndarray,
    title: str = "RCS Distribution",
    phi_range: Tuple[float, float] = (-45, 45),
    theta_range: Tuple[float, float] = (45, 135),
    figsize: Tuple[int, int] = (8, 6),
    fontsize_scale: float = 1.0,
    save_path: Optional[str] = None,
    fig: Optional[Figure] = None
) -> Figure:
    """
    绘制单个RCS热图

    Args:
        rcs_data: RCS数据 [H, W]，线性值
        title: 图表标题
        phi_range: 方位角范围
        theta_range: 俯仰角范围
        figsize: 图像尺寸
        fontsize_scale: 字号缩放因子
        save_path: 保存路径
        fig: 复用的Figure对象

    Returns:
        matplotlib.figure.Figure对象
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()

    # 转换为dB
    rcs_db = 10 * np.log10(rcs_data + 1e-10)

    # 设置extent
    extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(rcs_db, cmap='jet', aspect='equal', extent=extent)
    ax.set_title(title, fontsize=int(24*fontsize_scale), fontweight='bold')
    ax.set_xlabel('φ (方位角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax.set_ylabel('θ (俯仰角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
    ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
    cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_latent_space_2d(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'pca',
    title: str = "Latent Space Visualization",
    figsize: Tuple[int, int] = (10, 8),
    fontsize_scale: float = 1.0,
    save_path: Optional[str] = None,
    fig: Optional[Figure] = None
) -> Figure:
    """
    绘制隐空间2D投影

    Args:
        latents: 隐空间表示 [N, latent_dim]
        labels: 样本标签 [N]（可选，用于着色）
        method: 降维方法 ('pca' 或 'tsne')
        title: 图表标题
        figsize: 图像尺寸
        fontsize_scale: 字号缩放因子
        save_path: 保存路径
        fig: 复用的Figure对象

    Returns:
        matplotlib.figure.Figure对象
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()

    # 降维到2D
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        latents_2d = reducer.fit_transform(latents)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"PCA (解释方差: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        latents_2d = reducer.fit_transform(latents)
        subtitle = "t-SNE"
    else:
        raise ValueError(f"未知的降维方法: {method}")

    ax = fig.add_subplot(1, 1, 1)

    if labels is not None:
        scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                            c=labels, cmap='viridis', s=50, alpha=0.6)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('样本标签', fontsize=int(16*fontsize_scale))
    else:
        ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                  c='blue', s=50, alpha=0.6)

    ax.set_title(f'{title}\n{subtitle}',
                fontsize=int(20*fontsize_scale), fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=int(16*fontsize_scale))
    ax.set_ylabel('Component 2', fontsize=int(16*fontsize_scale))
    ax.tick_params(axis='both', labelsize=int(14*fontsize_scale))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    training_history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4),
    fontsize_scale: float = 1.0,
    save_path: Optional[str] = None,
    fig: Optional[Figure] = None
) -> Figure:
    """
    绘制训练曲线

    Args:
        training_history: 训练历史字典，包含:
            - 'stage1_train_loss': List[float]
            - 'stage1_val_loss': List[float]
            - 'stage2_train_loss': List[float] (可选)
            - 'stage2_val_loss': List[float] (可选)
            - 'stage3_train_loss': List[float] (可选)
            - 'stage3_val_loss': List[float] (可选)
        figsize: 图像尺寸
        fontsize_scale: 字号缩放因子
        save_path: 保存路径
        fig: 复用的Figure对象

    Returns:
        matplotlib.figure.Figure对象
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()

    # 确定有几个阶段
    stages = []
    if 'stage1_train_loss' in training_history:
        stages.append('stage1')
    if 'stage2_train_loss' in training_history:
        stages.append('stage2')
    if 'stage3_train_loss' in training_history:
        stages.append('stage3')

    n_stages = len(stages)
    if n_stages == 0:
        # 没有训练历史
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, '无训练历史数据', ha='center', va='center',
               fontsize=int(20*fontsize_scale))
        return fig

    # 为每个阶段创建子图
    for idx, stage in enumerate(stages, 1):
        ax = fig.add_subplot(1, n_stages, idx)

        train_key = f'{stage}_train_loss'
        val_key = f'{stage}_val_loss'

        train_loss = training_history.get(train_key, [])
        val_loss = training_history.get(val_key, [])

        if train_loss:
            epochs = range(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)

        if val_loss:
            epochs = range(1, len(val_loss) + 1)
            ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)

        stage_names = {
            'stage1': 'Stage 1 (AE预训练)',
            'stage2': 'Stage 2 (参数映射)',
            'stage3': 'Stage 3 (端到端)'
        }

        ax.set_title(stage_names.get(stage, stage),
                    fontsize=int(18*fontsize_scale), fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=int(14*fontsize_scale))
        ax.set_ylabel('Loss', fontsize=int(14*fontsize_scale))
        ax.legend(fontsize=int(12*fontsize_scale))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=int(12*fontsize_scale))

    fig.suptitle('训练曲线', fontsize=int(20*fontsize_scale), fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ========== 工具函数 ==========

def create_figure(figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    创建新的Figure对象

    Args:
        figsize: 图像尺寸

    Returns:
        matplotlib.figure.Figure对象

    示例:
        >>> fig = create_figure()
        >>> plot_rcs_comparison(..., fig=fig)
    """
    return plt.figure(figsize=figsize)


def close_figure(fig: Figure) -> None:
    """
    关闭Figure对象，释放内存

    Args:
        fig: matplotlib.figure.Figure对象
    """
    plt.close(fig)


def plot_wavelet_coefficients_comparison(
    original_coeffs: np.ndarray,
    reconstructed_coeffs: np.ndarray,
    freq_idx: int = 0,
    freq_label: str = "1.5 GHz",
    model_id: str = "Unknown",
    figsize: Tuple[int, int] = (15, 20),
    fontsize_scale: float = 1.0,
    save_path: Optional[str] = None,
    fig: Optional[Figure] = None
) -> Figure:
    """
    绘制小波系数对比图（原始 vs 重建 vs 残差）

    供GUI和批量实验复用，仅适用于Wavelet模式。

    Args:
        original_coeffs: 原始小波系数 [H, W, num_freq*4]
        reconstructed_coeffs: 重建小波系数 [H, W, num_freq*4]
        freq_idx: 频率索引 (0=1.5GHz, 1=3GHz, 2=6GHz)
        freq_label: 频率标签（如 "1.5 GHz"）
        model_id: 模型ID（如 "001"）
        figsize: 图像尺寸
        fontsize_scale: 字号缩放因子
        save_path: 保存路径（批量实验使用）
        fig: 复用的Figure对象（GUI使用）

    Returns:
        matplotlib.figure.Figure对象

    示例:
        # GUI使用（复用Figure）
        >>> plot_wavelet_coefficients_comparison(orig, recon, freq_idx=0, fig=gui.vis_fig)

        # 批量实验使用（保存到文件）
        >>> plot_wavelet_coefficients_comparison(orig, recon, freq_idx=0, save_path='wavelet_comp.png')
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()

    # 提取该频率的4个小波通道
    # 小波系数格式: [H, W, num_freq*4]
    # 对于2频率: 通道0-3是1.5GHz的LL/LH/HL/HH, 通道4-7是3GHz的LL/LH/HL/HH
    base_idx = freq_idx * 4

    channel_names = ['LL (低频近似)', 'LH (水平边缘)', 'HL (垂直边缘)', 'HH (对角边缘)']

    # 创建4行3列的图表
    for ch_idx in range(4):
        coeff_idx = base_idx + ch_idx

        # 提取该通道的原始和重建系数
        orig_ch = original_coeffs[:, :, coeff_idx]  # [H, W]
        recon_ch = reconstructed_coeffs[:, :, coeff_idx]  # [H, W]
        residual_ch = orig_ch - recon_ch

        # 计算MSE和MAE
        mse = np.mean((orig_ch - recon_ch)**2)
        residual_finite = residual_ch[np.isfinite(residual_ch)]
        mae = np.mean(np.abs(residual_finite)) if len(residual_finite) > 0 else 0

        # 第一列: 原始系数
        ax1 = fig.add_subplot(4, 3, ch_idx*3 + 1)
        vmin, vmax = orig_ch.min(), orig_ch.max()
        im1 = ax1.imshow(orig_ch, cmap='viridis', aspect='equal')
        ax1.set_title(f'{channel_names[ch_idx]}\n原始系数',
                      fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.set_ylabel(f'通道{ch_idx+1}', fontsize=int(20*fontsize_scale), fontweight='bold')
        if ch_idx == 3:
            ax1.set_xlabel('像素', fontsize=int(20*fontsize_scale), fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=int(16*fontsize_scale))
        cbar1.set_label('系数值', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 第二列: 重建系数 (使用相同的colorbar范围)
        ax2 = fig.add_subplot(4, 3, ch_idx*3 + 2)
        im2 = ax2.imshow(recon_ch, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
        ax2.set_title(f'重建系数\nMSE={mse:.4e}',
                      fontsize=int(20*fontsize_scale), fontweight='bold')
        if ch_idx == 3:
            ax2.set_xlabel('像素', fontsize=int(20*fontsize_scale), fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=int(16*fontsize_scale))
        cbar2.set_label('系数值', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 第三列: 残差
        ax3 = fig.add_subplot(4, 3, ch_idx*3 + 3)
        if len(residual_finite) > 0:
            residual_abs_max = np.percentile(np.abs(residual_finite), 95)
        else:
            residual_abs_max = 1
        im3 = ax3.imshow(residual_ch, cmap='RdBu_r', aspect='equal',
                       vmin=-residual_abs_max, vmax=residual_abs_max)
        ax3.set_title(f'残差\nMAE={mae:.4e}',
                      fontsize=int(20*fontsize_scale), fontweight='bold')
        if ch_idx == 3:
            ax3.set_xlabel('像素', fontsize=int(20*fontsize_scale), fontweight='bold')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=int(16*fontsize_scale))
        cbar3.set_label('残差值', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))

    fig.suptitle(f'小波系数对比分析 - 模型{model_id} @ {freq_label}',
                 fontsize=int(24*fontsize_scale), fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_ae_training_progress(
    ae_training_history: Dict,
    figsize: Tuple[int, int] = (18, 12),
    fontsize_scale: float = 1.0,
    save_path: Optional[str] = None,
    fig: Optional[Figure] = None,
    use_log_scale: bool = True,
    show_best_epoch: bool = True,
    show_gradient: bool = True
) -> Figure:
    """
    绘制AutoEncoder训练进度（支持三阶段训练 + 梯度监控）

    供GUI和批量实验复用，统一绘制AE训练进度可视化。

    Args:
        ae_training_history: AE训练历史字典，包含:
            {
                'stage_histories': {
                    'stage1': {
                        'train_losses': List[float],
                        'val_losses': List[float],
                        'best_val_loss': float (可选),
                        'best_epoch': int (可选),
                        'gradient_history': {  # 梯度历史（可选）
                            'epochs': List[int],
                            'grad_norm': List[float],
                            'grad_mean': List[float],
                            'grad_std': List[float],
                            'grad_max': List[float],
                            'grad_min': List[float]
                        }
                    },
                    'stage2': {...},
                    'stage3': {...}
                }
            }
        figsize: 图像尺寸
        fontsize_scale: 字号缩放因子
        save_path: 保存路径（批量实验使用）
        fig: 复用的Figure对象（GUI使用）
        use_log_scale: 是否使用对数坐标
        show_best_epoch: 是否显示最佳epoch标记
        show_gradient: 是否显示梯度变化图（默认True）

    Returns:
        matplotlib.figure.Figure对象

    示例:
        # GUI使用（复用Figure）
        >>> plot_ae_training_progress(history, fig=gui.vis_fig)

        # 批量实验使用（保存到文件）
        >>> plot_ae_training_progress(history, save_path='progress.png')
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()

    # 检查数据完整性
    if not ae_training_history or 'stage_histories' not in ae_training_history:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'AutoEncoder训练进度可视化\n(暂无训练历史数据)',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=int(20*fontsize_scale), fontweight='bold')
        ax.set_title('AutoEncoder训练进度',
                     fontsize=int(24*fontsize_scale), fontweight='bold')
        ax.axis('off')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # 获取训练历史数据
    stage_histories = ae_training_history['stage_histories']
    num_stages = len(stage_histories)

    if num_stages == 0:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'AutoEncoder训练进度可视化\n(暂无阶段历史数据)',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=int(20*fontsize_scale), fontweight='bold')
        ax.set_title('AutoEncoder训练进度',
                     fontsize=int(24*fontsize_scale), fontweight='bold')
        ax.axis('off')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # 检查是否有梯度数据
    has_gradient = any(
        'gradient_history' in stage_data and
        stage_data['gradient_history'] and
        len(stage_data['gradient_history'].get('grad_norm', [])) > 0
        for stage_data in stage_histories.values()
    )

    # 确定布局：如果有梯度数据且需要显示，使用2行布局；否则使用单行布局
    if show_gradient and has_gradient:
        rows = 2
        cols = min(num_stages, 3)  # 最多3列
    else:
        # 原始布局（两列）
        rows = (num_stages + 1) // 2
        cols = 2 if num_stages > 1 else 1

    stage_colors = ['blue', 'green', 'red', 'orange']
    stage_names = ['阶段1(AE预训练)', '阶段2(参数映射)', '阶段3(端到端)', '完整训练']

    for i, (stage_name, stage_data) in enumerate(stage_histories.items()):
        if 'train_losses' not in stage_data or 'val_losses' not in stage_data:
            continue

        # 计算子图位置
        if show_gradient and has_gradient:
            # 2行布局：第1行显示训练曲线
            ax_loss = fig.add_subplot(rows, cols, i + 1)
        else:
            # 原始布局
            if num_stages == 1:
                ax_loss = fig.add_subplot(1, 1, 1)
            else:
                row = i // cols
                col = i % cols
                ax_loss = fig.add_subplot(rows, cols, i + 1)

        # 绘制训练和验证损失
        epochs = range(1, len(stage_data['train_losses']) + 1)
        color = stage_colors[i % len(stage_colors)]

        ax_loss.plot(epochs, stage_data['train_losses'], color=color, linestyle='-',
                    label='训练损失', linewidth=2)
        ax_loss.plot(epochs, stage_data['val_losses'], color=color, linestyle='--',
                    label='验证损失', linewidth=2)

        ax_loss.set_xlabel('训练轮数', fontsize=int(16*fontsize_scale), fontweight='bold')
        ax_loss.set_ylabel('损失值', fontsize=int(16*fontsize_scale), fontweight='bold')
        ax_loss.set_title(f'{stage_names[i] if i < len(stage_names) else stage_name}',
                         fontsize=int(18*fontsize_scale), fontweight='bold')
        ax_loss.legend(fontsize=int(12*fontsize_scale), loc='best')
        ax_loss.grid(True, alpha=0.3)

        # 使用对数坐标
        if use_log_scale:
            ax_loss.set_yscale('log')

        ax_loss.tick_params(axis='both', labelsize=int(14*fontsize_scale))

        # 添加最佳损失标记
        if show_best_epoch and 'best_val_loss' in stage_data:
            best_epoch = stage_data.get('best_epoch', len(stage_data['val_losses']))
            ax_loss.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7,
                           label=f'最佳: Epoch {best_epoch}')
            ax_loss.legend(fontsize=int(12*fontsize_scale), loc='best')

        # 绘制梯度变化图（第2行）
        if show_gradient and has_gradient and 'gradient_history' in stage_data:
            grad_hist = stage_data['gradient_history']
            if grad_hist and len(grad_hist.get('grad_norm', [])) > 0:
                ax_grad = fig.add_subplot(rows, cols, cols + i + 1)

                grad_epochs = grad_hist['epochs']

                # 主要显示梯度范数（最重要的指标）
                ax_grad.plot(grad_epochs, grad_hist['grad_norm'],
                            color='purple', linestyle='-', marker='o', markersize=4,
                            label='梯度范数 (L2)', linewidth=2)

                # 添加警告阈值线（如果超出阈值会有视觉提示）
                warn_high = 10.0
                warn_low = 1e-5
                ax_grad.axhline(y=warn_high, color='red', linestyle='--', alpha=0.5,
                               linewidth=1.5, label=f'爆炸阈值 ({warn_high})')
                ax_grad.axhline(y=warn_low, color='orange', linestyle='--', alpha=0.5,
                               linewidth=1.5, label=f'消失阈值 ({warn_low})')

                ax_grad.set_xlabel('训练轮数', fontsize=int(16*fontsize_scale), fontweight='bold')
                ax_grad.set_ylabel('梯度范数', fontsize=int(16*fontsize_scale), fontweight='bold')
                ax_grad.set_title(f'{stage_names[i]} - 梯度监控',
                                 fontsize=int(18*fontsize_scale), fontweight='bold')
                ax_grad.legend(fontsize=int(10*fontsize_scale), loc='best')
                ax_grad.grid(True, alpha=0.3)
                ax_grad.set_yscale('log')  # 梯度范数通常需要对数坐标
                ax_grad.tick_params(axis='both', labelsize=int(14*fontsize_scale))

    if show_gradient and has_gradient:
        fig.suptitle('AutoEncoder训练进度 + 梯度监控',
                     fontsize=int(24*fontsize_scale), fontweight='bold', y=0.98)
    else:
        fig.suptitle('AutoEncoder训练进度',
                     fontsize=int(24*fontsize_scale), fontweight='bold')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
