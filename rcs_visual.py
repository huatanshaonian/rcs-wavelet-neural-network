"""
RCS数据可视化统一模块
整合所有RCS可视化功能，包括2D热图、3D图、球坐标图和矩阵数据接口

主要功能:
1. 2D热图可视化（线性值+分贝值）
2. 3D表面图可视化
3. 球坐标3D图可视化
4. 直接矩阵数据接口
5. 数据保存和加载
6. 多模型对比功能

作者: 基于simple_rcs_plot.py和rcs_visual.py整合
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 核心数据接口函数
# ============================================================================

def get_rcs_matrix(model_id="001", freq_suffix="1.5G", data_dir=r"F:\data\parameter\csv_output"):
    """
    直接输出RCS矩阵数据接口 - 核心数据获取函数

    参数:
    model_id (str): 模型编号，默认"001"
    freq_suffix (str): 频率后缀，默认"1.5G"
    data_dir (str): 数据目录路径

    返回:
    dict: 包含所有矩阵数据和元信息的字典
    """
    data_file = os.path.join(data_dir, f"{model_id}_{freq_suffix}.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    print(f"正在加载数据: {data_file}")

    # 读取数据
    df = pd.read_csv(data_file, encoding='utf-8')

    # 提取角度和RCS数据
    theta_values = np.sort(df['Theta'].unique())
    phi_values = np.sort(df['Phi'].unique())

    # 创建RCS矩阵
    rcs_matrix = np.full((len(theta_values), len(phi_values)), np.nan)

    # 填充数据
    for _, row in df.iterrows():
        theta_idx = np.where(theta_values == row['Theta'])[0][0]
        phi_idx = np.where(phi_values == row['Phi'])[0][0]
        rcs_matrix[theta_idx, phi_idx] = row['RCS(Total)']

    # 转换为分贝
    rcs_db_matrix = 10 * np.log10(np.maximum(rcs_matrix, 1e-10))

    # 创建网格
    phi_grid, theta_grid = np.meshgrid(phi_values, theta_values)

    # 统计信息
    data_info = {
        'total_points': len(df),
        'valid_points': np.sum(~np.isnan(rcs_matrix)),
        'theta_range': (theta_values.min(), theta_values.max()),
        'phi_range': (phi_values.min(), phi_values.max()),
        'rcs_linear_range': (np.nanmin(rcs_matrix), np.nanmax(rcs_matrix)),
        'rcs_db_range': (np.nanmin(rcs_db_matrix), np.nanmax(rcs_db_matrix)),
        'rcs_linear_mean': np.nanmean(rcs_matrix),
        'rcs_db_mean': np.nanmean(rcs_db_matrix),
        'matrix_shape': rcs_matrix.shape
    }

    print(f"数据加载完成:")
    print(f"  - 矩阵形状: {data_info['matrix_shape']}")
    print(f"  - 有效数据点: {data_info['valid_points']}")
    print(f"  - RCS线性值范围: {data_info['rcs_linear_range'][0]:.6e} - {data_info['rcs_linear_range'][1]:.6e}")
    print(f"  - RCS分贝值范围: {data_info['rcs_db_range'][0]:.1f} - {data_info['rcs_db_range'][1]:.1f} dB")

    return {
        'rcs_linear': rcs_matrix,
        'rcs_db': rcs_db_matrix,
        'theta_values': theta_values,
        'phi_values': phi_values,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid,
        'data_info': data_info
    }


def save_rcs_matrix(matrix_data, save_path="rcs_matrix_data.npz"):
    """保存RCS矩阵数据到文件"""
    np.savez_compressed(save_path,
                       rcs_linear=matrix_data['rcs_linear'],
                       rcs_db=matrix_data['rcs_db'],
                       theta_values=matrix_data['theta_values'],
                       phi_values=matrix_data['phi_values'],
                       theta_grid=matrix_data['theta_grid'],
                       phi_grid=matrix_data['phi_grid'])
    print(f"矩阵数据已保存到: {save_path}")


def load_rcs_matrix(file_path="rcs_matrix_data.npz"):
    """从文件加载RCS矩阵数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = np.load(file_path)
    return {
        'rcs_linear': data['rcs_linear'],
        'rcs_db': data['rcs_db'],
        'theta_values': data['theta_values'],
        'phi_values': data['phi_values'],
        'theta_grid': data['theta_grid'],
        'phi_grid': data['phi_grid']
    }


# ============================================================================
# 2D可视化函数
# ============================================================================

def plot_2d_heatmap(model_id="001", freq_suffix="1.5G", data_dir=None,
                   db_vmin=None, db_vmax=None, linear_vmin=None, linear_vmax=None,
                   colormap='jet', figsize=(16, 6), save_path=None, show_plot=True):
    """
    绘制2D热图（线性值+分贝值）

    参数:
    model_id: 模型编号
    freq_suffix: 频率后缀
    data_dir: 数据目录
    db_vmin, db_vmax: 分贝值colorbar范围
    linear_vmin, linear_vmax: 线性值colorbar范围
    colormap: 颜色映射
    figsize: 图像尺寸
    save_path: 保存路径
    show_plot: 是否显示图像

    返回:
    fig, (ax1, ax2), data: 图形对象、轴对象、数据字典
    """
    if data_dir is None:
        data_dir = r"F:\data\parameter\csv_output"

    # 获取数据
    data = get_rcs_matrix(model_id, freq_suffix, data_dir)
    rcs_matrix = data['rcs_linear']
    rcs_db_matrix = data['rcs_db']
    theta_values = data['theta_values']
    phi_values = data['phi_values']

    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 线性值热图
    im1 = ax1.imshow(rcs_matrix,
                    extent=[phi_values.min(), phi_values.max(),
                           theta_values.max(), theta_values.min()],
                    aspect='auto', origin='upper', cmap=colormap,
                    vmin=linear_vmin, vmax=linear_vmax)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('RCS Linear Value', fontsize=12)
    ax1.set_xlabel('Phi - Azimuth (degrees)', fontsize=12)
    ax1.set_ylabel('Theta - Elevation (degrees)', fontsize=12)
    ax1.set_title(f'RCS Linear Value - Model {model_id} ({freq_suffix})', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 分贝值热图
    im2 = ax2.imshow(rcs_db_matrix,
                    extent=[phi_values.min(), phi_values.max(),
                           theta_values.max(), theta_values.min()],
                    aspect='auto', origin='upper', cmap=colormap,
                    vmin=db_vmin, vmax=db_vmax)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('RCS (dB)', fontsize=12)
    ax2.set_xlabel('Phi - Azimuth (degrees)', fontsize=12)
    ax2.set_ylabel('Theta - Elevation (degrees)', fontsize=12)
    ax2.set_title(f'RCS dB Value - Model {model_id} ({freq_suffix})', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D热图已保存到: {save_path}")

    if show_plot:
        plt.show()

    return fig, (ax1, ax2), data


# ============================================================================
# 3D可视化函数
# ============================================================================

def plot_3d_surface(model_id="001", freq_suffix="1.5G", data_dir=None,
                   db_vmin=None, db_vmax=None, colormap='jet',
                   figsize=(12, 8), save_path=None, show_plot=True):
    """
    绘制3D表面图

    参数:
    model_id: 模型编号
    freq_suffix: 频率后缀
    data_dir: 数据目录
    db_vmin, db_vmax: 分贝值范围
    colormap: 颜色映射
    figsize: 图像尺寸
    save_path: 保存路径
    show_plot: 是否显示图像

    返回:
    fig, ax, data: 图形对象、轴对象、数据字典
    """
    if data_dir is None:
        data_dir = r"F:\data\parameter\csv_output"

    # 获取数据
    data = get_rcs_matrix(model_id, freq_suffix, data_dir)
    rcs_db_matrix = data['rcs_db']
    theta_grid = data['theta_grid']
    phi_grid = data['phi_grid']

    # 创建3D图
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面
    surf = ax.plot_surface(phi_grid, theta_grid, rcs_db_matrix,
                          cmap=colormap, alpha=0.8, linewidth=0, antialiased=True,
                          vmin=db_vmin, vmax=db_vmax)

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('RCS (dB)', fontsize=12)

    # 设置标签
    ax.set_xlabel('Phi - Azimuth (degrees)', fontsize=12)
    ax.set_ylabel('Theta - Elevation (degrees)', fontsize=12)
    ax.set_zlabel('RCS (dB)', fontsize=12)
    ax.set_title(f'RCS 3D Surface - Model {model_id} ({freq_suffix})', fontsize=14)

    # 设置视角
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D表面图已保存到: {save_path}")

    if show_plot:
        plt.show()

    return fig, ax, data


def plot_spherical_3d(model_id="001", freq_suffix="1.5G", data_dir=None,
                     use_db=True, colormap='jet', radius_scale=1.0,
                     figsize=(12, 9), save_path=None, show_plot=True):
    """
    绘制球坐标3D图

    参数:
    model_id: 模型编号
    freq_suffix: 频率后缀
    data_dir: 数据目录
    use_db: 是否使用分贝值
    colormap: 颜色映射
    radius_scale: 半径缩放因子
    figsize: 图像尺寸
    save_path: 保存路径
    show_plot: 是否显示图像

    返回:
    fig, ax, data: 图形对象、轴对象、数据字典
    """
    if data_dir is None:
        data_dir = r"F:\data\parameter\csv_output"

    # 获取数据
    data = get_rcs_matrix(model_id, freq_suffix, data_dir)

    rcs_data = data['rcs_db'] if use_db else data['rcs_linear']
    theta_values = data['theta_values']
    phi_values = data['phi_values']
    unit = "dB" if use_db else "m²"

    # 转换角度为弧度
    theta_rad = np.radians(theta_values)
    phi_rad = np.radians(phi_values)

    # 创建完整的角度网格
    theta_grid_rad, phi_grid_rad = np.meshgrid(theta_rad, phi_rad)

    # 计算球坐标
    if use_db:
        r_base = 1.0
        rcs_normalized = (rcs_data - np.min(rcs_data)) / (np.max(rcs_data) - np.min(rcs_data))
        radius = r_base + radius_scale * rcs_normalized
    else:
        rcs_normalized = rcs_data / np.max(rcs_data)
        radius = radius_scale * rcs_normalized

    # 转换为笛卡尔坐标
    x = radius.T * np.sin(theta_grid_rad) * np.cos(phi_grid_rad)
    y = radius.T * np.sin(theta_grid_rad) * np.sin(phi_grid_rad)
    z = radius.T * np.cos(theta_grid_rad)

    # 创建3D图
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 颜色映射基于RCS值
    colors = rcs_data.T

    # 绘制3D表面
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.get_cmap(colormap)(
        (colors - np.min(colors)) / (np.max(colors) - np.min(colors))),
        alpha=0.8, linewidth=0, antialiased=True)

    # 添加颜色条
    norm = plt.Normalize(vmin=np.min(rcs_data), vmax=np.max(rcs_data))
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label(f'RCS ({unit})', fontsize=12)

    # 设置标签和标题
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'RCS Spherical 3D - Model {model_id} ({freq_suffix})', fontsize=14)

    # 设置等比例坐标轴
    max_range = np.max([np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # 优化视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"球坐标3D图已保存到: {save_path}")

    if show_plot:
        plt.show()

    return fig, ax, data


# ============================================================================
# 综合可视化函数
# ============================================================================

def plot_all_views(model_id="001", freq_suffix="1.5G", data_dir=None,
                  save_dir=None, show_plots=True):
    """
    生成所有类型的可视化图

    参数:
    model_id: 模型编号
    freq_suffix: 频率后缀
    data_dir: 数据目录
    save_dir: 保存目录
    show_plots: 是否显示图像

    返回:
    figs, axes, data: 图形对象列表、轴对象列表、数据字典
    """
    if data_dir is None:
        data_dir = r"F:\data\parameter\csv_output"

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2D热图
    save_path_2d = os.path.join(save_dir, f"rcs_2d_{model_id}_{freq_suffix}.png") if save_dir else None
    fig1, (ax1, ax2), data = plot_2d_heatmap(model_id, freq_suffix, data_dir,
                                             save_path=save_path_2d, show_plot=show_plots)

    # 3D表面图
    save_path_3d = os.path.join(save_dir, f"rcs_3d_{model_id}_{freq_suffix}.png") if save_dir else None
    fig2, ax3, _ = plot_3d_surface(model_id, freq_suffix, data_dir,
                                  save_path=save_path_3d, show_plot=show_plots)

    # 球坐标3D图
    save_path_sphere = os.path.join(save_dir, f"rcs_sphere_{model_id}_{freq_suffix}.png") if save_dir else None
    fig3, ax4, _ = plot_spherical_3d(model_id, freq_suffix, data_dir,
                                    save_path=save_path_sphere, show_plot=show_plots)

    print(f"\n所有可视化图已生成完成！")
    if save_dir:
        print(f"图像保存在目录: {save_dir}")

    return [fig1, fig2, fig3], [ax1, ax2, ax3, ax4], data


def compare_models(model_ids, freq_suffix="1.5G", data_dir=None,
                  figsize=(15, 10), save_path=None, show_plot=True):
    """
    多模型对比可视化

    参数:
    model_ids: 模型编号列表
    freq_suffix: 频率后缀
    data_dir: 数据目录
    figsize: 图像尺寸
    save_path: 保存路径
    show_plot: 是否显示图像

    返回:
    fig, axes, data_list: 图形对象、轴对象、数据列表
    """
    if data_dir is None:
        data_dir = r"F:\data\parameter\csv_output"

    n_models = len(model_ids)
    fig, axes = plt.subplots(2, n_models, figsize=figsize)
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    data_list = []

    for i, model_id in enumerate(model_ids):
        data = get_rcs_matrix(model_id, freq_suffix, data_dir)
        data_list.append(data)

        rcs_db = data['rcs_db']
        theta_values = data['theta_values']
        phi_values = data['phi_values']

        # 2D热图
        im1 = axes[0, i].imshow(rcs_db,
                               extent=[phi_values.min(), phi_values.max(),
                                      theta_values.max(), theta_values.min()],
                               aspect='auto', origin='upper', cmap='jet')
        axes[0, i].set_title(f'Model {model_id} - 2D Heatmap')
        axes[0, i].set_xlabel('Phi (degrees)')
        axes[0, i].set_ylabel('Theta (degrees)')
        plt.colorbar(im1, ax=axes[0, i])

        # 统计直方图
        axes[1, i].hist(rcs_db.flatten(), bins=50, alpha=0.7,
                       label=f'Model {model_id}', color=f'C{i}')
        axes[1, i].set_title(f'Model {model_id} - RCS Distribution')
        axes[1, i].set_xlabel('RCS (dB)')
        axes[1, i].set_ylabel('Count')
        axes[1, i].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")

    if show_plot:
        plt.show()

    return fig, axes, data_list


# ============================================================================
# 主程序和示例
# ============================================================================

def main():
    """主程序示例"""
    print("=" * 60)
    print("RCS统一可视化模块 - 主程序")
    print("=" * 60)

    try:
        # 示例1: 基本2D可视化
        print("\n1. 生成2D热图...")
        fig_2d, axes_2d, data = plot_2d_heatmap("001", "1.5G")

        # 示例2: 3D表面图
        print("\n2. 生成3D表面图...")
        fig_3d, ax_3d, _ = plot_3d_surface("001", "1.5G")

        # 示例3: 球坐标图
        print("\n3. 生成球坐标3D图...")
        fig_sphere, ax_sphere, _ = plot_spherical_3d("001", "1.5G")

        # 示例4: 矩阵数据操作
        print("\n4. 矩阵数据分析...")
        matrix_data = get_rcs_matrix("001", "1.5G")
        rcs_linear = matrix_data['rcs_linear']
        max_pos = np.unravel_index(np.nanargmax(rcs_linear), rcs_linear.shape)
        max_theta = matrix_data['theta_values'][max_pos[0]]
        max_phi = matrix_data['phi_values'][max_pos[1]]
        print(f"   最大RCS位置: Theta={max_theta:.1f}°, Phi={max_phi:.1f}°")
        print(f"   最大RCS值: {np.nanmax(rcs_linear):.6e}")

        print(f"\n✓ 所有功能演示完成！")

    except Exception as e:
        print(f"出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()