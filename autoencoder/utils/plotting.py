"""
AutoEncoder可视化绘图功能

提供纯函数式的绘图接口，供GUI和批量实验复用。

核心功能：
1. plot_rcs_comparison() - 绘制RCS对比图（真实 vs 预测 vs 残差）
2. plot_rcs_heatmap() - 绘制单个RCS热图
3. plot_latent_space_2d() - 绘制隐空间2D投影
4. plot_training_curves() - 绘制训练曲线

设计原则：
- 纯函数，接受数据作为参数
- 返回matplotlib.figure.Figure对象或直接保存到文件
- GUI可以在Canvas上显示，批量实验可以保存到文件
- 不依赖GUI组件，不弹窗，不读取文件

作者：Claude Code
日期：2025-01-10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Dict
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
