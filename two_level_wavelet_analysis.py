#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二级小波分解分析程序
对LL通道进行二次分解，可视化所有通道
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

# 导入RCS数据读取器
import rcs_data_reader as rdr


def two_level_wavelet_decomposition(data, wavelet='haar'):
    """
    二级小波分解

    Args:
        data: [91, 91] RCS数据
        wavelet: 小波类型

    Returns:
        dict: 包含所有分解系数的字典
    """
    # 第一级分解
    coeffs_level1 = pywt.dwt2(data, wavelet)
    cA1, (cH1, cV1, cD1) = coeffs_level1

    # 第二级分解（对LL1进行分解）
    coeffs_level2 = pywt.dwt2(cA1, wavelet)
    cA2, (cH2, cV2, cD2) = coeffs_level2

    return {
        'level1': {
            'LL': cA1,  # 一级低频（会被二级分解）
            'LH': cH1,  # 一级水平
            'HL': cV1,  # 一级垂直
            'HH': cD1   # 一级对角
        },
        'level2': {
            'LL': cA2,  # 二级低频
            'LH': cH2,  # 二级水平
            'HL': cV2,  # 二级垂直
            'HH': cD2   # 二级对角
        }
    }


def plot_two_level_wavelet(original_data, coeffs, model_id='001', frequency='1.5G',
                           wavelet='haar', data_type='dB'):
    """
    绘制二级小波分解结果

    Args:
        original_data: 原始RCS数据（线性域）
        coeffs: 二级分解系数字典
        model_id: 模型ID
        frequency: 频率
        wavelet: 小波类型
        data_type: 显示单位 ('dB' 或 'linear')
    """
    # 转换为dB显示
    epsilon = 1e-10
    if data_type == 'dB':
        original_dB = 10 * np.log10(np.maximum(original_data, epsilon))

        # 一级系数转dB
        level1_dB = {
            'LL': 10 * np.log10(np.maximum(np.abs(coeffs['level1']['LL']), epsilon)),
            'LH': 10 * np.log10(np.maximum(np.abs(coeffs['level1']['LH']), epsilon)),
            'HL': 10 * np.log10(np.maximum(np.abs(coeffs['level1']['HL']), epsilon)),
            'HH': 10 * np.log10(np.maximum(np.abs(coeffs['level1']['HH']), epsilon))
        }

        # 二级系数转dB
        level2_dB = {
            'LL': 10 * np.log10(np.maximum(np.abs(coeffs['level2']['LL']), epsilon)),
            'LH': 10 * np.log10(np.maximum(np.abs(coeffs['level2']['LH']), epsilon)),
            'HL': 10 * np.log10(np.maximum(np.abs(coeffs['level2']['HL']), epsilon)),
            'HH': 10 * np.log10(np.maximum(np.abs(coeffs['level2']['HH']), epsilon))
        }
    else:
        original_dB = original_data
        level1_dB = coeffs['level1']
        level2_dB = coeffs['level2']

    # 创建图形：3行3列
    fig = plt.figure(figsize=(15, 12))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算colorbar范围（原图的范围）
    vmin_orig = np.min(original_dB)
    vmax_orig = np.max(original_dB)

    # 角度范围
    phi_range = np.linspace(0, 360, original_data.shape[1])
    theta_range = np.linspace(0, 180, original_data.shape[0])
    extent = [phi_range[0], phi_range[-1], theta_range[-1], theta_range[0]]

    # === 第一行：原图 + 一级LL + 二级LL ===

    # 原始数据
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(original_dB, cmap='jet', aspect='equal', extent=extent,
                     vmin=vmin_orig, vmax=vmax_orig)
    ax1.set_title(f'原始RCS\n模型{model_id} @ {frequency}', fontsize=11, fontweight='bold')
    ax1.set_xlabel('φ (度)')
    ax1.set_ylabel('θ (度)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 一级LL（会被二次分解）
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(level1_dB['LL'], cmap='jet', aspect='equal', extent=extent,
                     vmin=vmin_orig, vmax=vmax_orig)
    ax2.set_title(f'一级LL (近似)\n[{coeffs["level1"]["LL"].shape[0]}×{coeffs["level1"]["LL"].shape[1]}]',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('φ (度)')
    ax2.set_ylabel('θ (度)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 二级LL（最粗尺度）
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(level2_dB['LL'], cmap='jet', aspect='equal', extent=extent,
                     vmin=vmin_orig, vmax=vmax_orig)
    ax3.set_title(f'二级LL (最粗近似)\n[{coeffs["level2"]["LL"].shape[0]}×{coeffs["level2"]["LL"].shape[1]}]',
                  fontsize=11, fontweight='bold', color='red')
    ax3.set_xlabel('φ (度)')
    ax3.set_ylabel('θ (度)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # === 第二行：一级细节系数 (LH, HL, HH) ===

    level1_comps = [
        ('LH', level1_dB['LH'], 'LH (水平)'),
        ('HL', level1_dB['HL'], 'HL (垂直)'),
        ('HH', level1_dB['HH'], 'HH (对角)')
    ]

    for i, (key, comp_data, title) in enumerate(level1_comps):
        ax = plt.subplot(3, 3, 4 + i)
        im = ax.imshow(comp_data, cmap='RdBu_r', aspect='equal', extent=extent)
        shape = coeffs['level1'][key].shape
        ax.set_title(f'一级{title}\n[{shape[0]}×{shape[1]}]', fontsize=11)
        ax.set_xlabel('φ (度)')
        ax.set_ylabel('θ (度)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # === 第三行：二级细节系数 (LH, HL, HH) ===

    level2_comps = [
        ('LH', level2_dB['LH'], 'LH (水平)'),
        ('HL', level2_dB['HL'], 'HL (垂直)'),
        ('HH', level2_dB['HH'], 'HH (对角)')
    ]

    for i, (key, comp_data, title) in enumerate(level2_comps):
        ax = plt.subplot(3, 3, 7 + i)
        im = ax.imshow(comp_data, cmap='RdBu_r', aspect='equal', extent=extent)
        shape = coeffs['level2'][key].shape
        ax.set_title(f'二级{title}\n[{shape[0]}×{shape[1]}]', fontsize=11, color='red')
        ax.set_xlabel('φ (度)')
        ax.set_ylabel('θ (度)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 总标题
    fig.suptitle(f'二级小波分解分析 - 模型{model_id} @ {frequency} (小波: {wavelet}, 单位: {data_type})',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


def analyze_energy_distribution(coeffs):
    """
    分析能量分布

    Args:
        coeffs: 二级分解系数字典

    Returns:
        dict: 能量分布统计
    """
    # 计算各分量能量
    energies = {}

    # 一级能量
    energies['level1_LL'] = np.sum(coeffs['level1']['LL']**2)
    energies['level1_LH'] = np.sum(coeffs['level1']['LH']**2)
    energies['level1_HL'] = np.sum(coeffs['level1']['HL']**2)
    energies['level1_HH'] = np.sum(coeffs['level1']['HH']**2)

    # 二级能量
    energies['level2_LL'] = np.sum(coeffs['level2']['LL']**2)
    energies['level2_LH'] = np.sum(coeffs['level2']['LH']**2)
    energies['level2_HL'] = np.sum(coeffs['level2']['HL']**2)
    energies['level2_HH'] = np.sum(coeffs['level2']['HH']**2)

    # 总能量
    total_energy = sum(energies.values())

    # 能量占比
    energy_ratios = {k: v/total_energy*100 for k, v in energies.items()}

    return {
        'energies': energies,
        'energy_ratios': energy_ratios,
        'total_energy': total_energy
    }


def print_statistics(coeffs, energy_info):
    """打印统计信息"""
    print("\n" + "="*60)
    print("二级小波分解统计信息")
    print("="*60)

    print("\n【尺寸信息】")
    print(f"  一级 LL: {coeffs['level1']['LL'].shape}")
    print(f"  一级 LH: {coeffs['level1']['LH'].shape}")
    print(f"  一级 HL: {coeffs['level1']['HL'].shape}")
    print(f"  一级 HH: {coeffs['level1']['HH'].shape}")
    print(f"  二级 LL: {coeffs['level2']['LL'].shape}")
    print(f"  二级 LH: {coeffs['level2']['LH'].shape}")
    print(f"  二级 HL: {coeffs['level2']['HL'].shape}")
    print(f"  二级 HH: {coeffs['level2']['HH'].shape}")

    print(f"\n【能量分布】(总能量: {energy_info['total_energy']:.2e})")
    print("  一级分量:")
    print(f"    LL: {energy_info['energy_ratios']['level1_LL']:.2f}%")
    print(f"    LH: {energy_info['energy_ratios']['level1_LH']:.2f}%")
    print(f"    HL: {energy_info['energy_ratios']['level1_HL']:.2f}%")
    print(f"    HH: {energy_info['energy_ratios']['level1_HH']:.2f}%")

    print("  二级分量:")
    print(f"    LL: {energy_info['energy_ratios']['level2_LL']:.2f}%")
    print(f"    LH: {energy_info['energy_ratios']['level2_LH']:.2f}%")
    print(f"    HL: {energy_info['energy_ratios']['level2_HL']:.2f}%")
    print(f"    HH: {energy_info['energy_ratios']['level2_HH']:.2f}%")

    # 分层汇总
    level1_total = sum([energy_info['energy_ratios'][f'level1_{k}']
                       for k in ['LL', 'LH', 'HL', 'HH']])
    level2_total = sum([energy_info['energy_ratios'][f'level2_{k}']
                       for k in ['LL', 'LH', 'HL', 'HH']])

    print(f"\n  一级总能量占比: {level1_total:.2f}%")
    print(f"  二级总能量占比: {level2_total:.2f}%")

    print("\n" + "="*60)


def main():
    """主函数"""
    # 参数配置
    model_id = '001'
    frequency = '1.5G'
    wavelet_type = 'haar'
    rcs_data_dir = '../parameter/csv_output'

    print(f"正在读取模型 {model_id} @ {frequency} 的RCS数据...")

    # 读取RCS数据
    rcs_result = rdr.get_rcs_matrix(model_id, frequency, rcs_data_dir)

    if rcs_result is None:
        print(f"错误: 无法读取数据")
        return

    # 获取线性RCS数据
    rcs_linear = rcs_result['rcs_linear']
    print(f"数据形状: {rcs_linear.shape}")
    print(f"数据范围: [{np.min(rcs_linear):.2e}, {np.max(rcs_linear):.2e}]")

    # 二级小波分解
    print(f"\n执行二级小波分解 (小波类型: {wavelet_type})...")
    coeffs = two_level_wavelet_decomposition(rcs_linear, wavelet=wavelet_type)

    # 能量分析
    energy_info = analyze_energy_distribution(coeffs)

    # 打印统计信息
    print_statistics(coeffs, energy_info)

    # 绘制结果
    print("\n生成可视化图表...")
    fig = plot_two_level_wavelet(rcs_linear, coeffs, model_id=model_id,
                                 frequency=frequency, wavelet=wavelet_type,
                                 data_type='dB')

    # 保存图片
    output_path = f'two_level_wavelet_{model_id}_{frequency}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}")

    # 显示图表
    plt.show()

    print("\n分析完成！")


if __name__ == "__main__":
    main()
