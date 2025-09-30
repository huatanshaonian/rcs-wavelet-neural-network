#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小波GUI助手模块
提供小波分析相关的独立功能函数
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt

def simple_wavelet_analysis(data, wavelet='db4', data_type='dB'):
    """
    简化的小波分析函数
    独立于其他模块，可直接在GUI中使用

    重要原理：
    - 小波变换始终在原始（线性）RCS数据上进行
    - data_type参数只影响最终的可视化显示
    - 这样确保小波变换的数学性质最优

    Args:
        data: 2D numpy数组（如果data_type='dB'，则为分贝值；否则为线性值）
        wavelet: 小波类型
        data_type: 显示类型 ('dB' 或 'linear') - 仅影响可视化

    Returns:
        dict: 包含分析结果的字典
    """

    # 根据输入数据类型确定原始线性数据
    if data_type == 'dB':
        # 输入是分贝值，转换为线性值用于小波分析
        epsilon = 1e-10
        linear_data = np.power(10, data / 10.0)  # dB转线性：10^(dB/10)
        analysis_data = linear_data  # 小波分析用线性数据
        display_original = data      # 显示用分贝数据
    else:
        # 输入已是线性值
        analysis_data = data         # 小波分析用原始线性数据
        display_original = data      # 显示也用线性数据

    # 对线性数据进行小波分解（确保最佳数学性质）
    coeffs = pywt.dwt2(analysis_data, wavelet)
    cA, (cH, cV, cD) = coeffs

    # 重建（在线性域）
    reconstructed_linear = pywt.idwt2(coeffs, wavelet)
    if reconstructed_linear.shape != analysis_data.shape:
        reconstructed_linear = reconstructed_linear[:analysis_data.shape[0], :analysis_data.shape[1]]

    # 准备显示数据
    if data_type == 'dB':
        # 将线性重建结果转换为分贝用于显示对比
        epsilon = 1e-10
        display_reconstructed = 10 * np.log10(np.maximum(reconstructed_linear, epsilon))
    else:
        # 线性模式直接显示
        display_reconstructed = reconstructed_linear

    # 小波分量显示：直接使用原始尺寸，不进行填充
    # 这样可以正确显示小波分量的实际内容，而不是填充后的左上角部分
    cA_display = cA
    cH_display = cH
    cV_display = cV
    cD_display = cD

    # 计算误差（用于显示的数据）
    error = np.abs(display_original - display_reconstructed)
    mse = np.mean((display_original - display_reconstructed)**2)
    psnr = 20 * np.log10(np.max(np.abs(display_original)) / np.sqrt(mse)) if mse > 0 else float('inf')

    # 计算统计信息（基于线性域的小波系数）
    components_orig = [cA, cH, cV, cD]
    components_display = [cA_display, cH_display, cV_display, cD_display]
    component_names = ['LL (Approximation)', 'LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)']

    # 能量分析（在线性域进行）
    energies = [np.sum(comp**2) for comp in components_orig]
    total_energy = sum(energies)
    energy_ratios = [e/total_energy for e in energies]

    # 稀疏度分析（在线性域进行）
    sparsities = []
    for comp in components_orig:
        threshold = np.std(comp) * 0.1
        sparsity = np.sum(np.abs(comp) < threshold) / comp.size
        sparsities.append(sparsity)

    # 动态范围（在线性域）
    max_values = [np.max(np.abs(comp)) for comp in components_orig]

    stats = {
        'energy_ratios': energy_ratios,
        'sparsities': sparsities,
        'max_values': max_values,
        'mse': mse,
        'psnr': psnr,
        'max_error': np.max(error),
        'analysis_domain': 'linear',  # 标明小波分析在线性域进行
        'display_domain': data_type   # 标明显示域
    }

    return {
        'original': display_original,
        'reconstructed': display_reconstructed,
        'error': error,
        'components': components_display,
        'component_names': component_names,
        'stats': stats
    }

def create_wavelet_plot(analysis_result, data_type='dB', model_name='001'):
    """
    创建小波分析图表

    Args:
        analysis_result: simple_wavelet_analysis的返回结果
        data_type: 数据类型 ('dB' 或 'linear')
        model_name: 模型名称

    Returns:
        matplotlib.figure.Figure: 图表对象
    """
    # 设置matplotlib中文字体支持和编码
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 确保支持Unicode
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取数据
    original = analysis_result['original']
    reconstructed = analysis_result['reconstructed']
    error = analysis_result['error']
    components = analysis_result['components']
    component_names = analysis_result['component_names']
    stats = analysis_result['stats']

    # 角度范围定义
    phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
    theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°
    extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'模型 {model_name} 小波分析 ({data_type})', fontsize=16, fontweight='bold')

    # 确定单位标签
    unit_label = 'dB' if data_type == 'dB' else 'Linear'

    # 第一行：原图、重建、误差、统计
    # 原始图像
    im1 = axes[0, 0].imshow(original, cmap='jet', aspect='auto', extent=extent)
    axes[0, 0].set_title(f'原始数据 ({unit_label})')
    axes[0, 0].set_xlabel('φ (度)')
    axes[0, 0].set_ylabel('θ (度)')
    cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    cbar1.set_label(unit_label, rotation=270, labelpad=15)

    # 重建图像
    im2 = axes[0, 1].imshow(reconstructed, cmap='jet', aspect='auto', extent=extent)
    axes[0, 1].set_title(f'重建数据 ({unit_label})')
    axes[0, 1].set_xlabel('φ (度)')
    axes[0, 1].set_ylabel('θ (度)')
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    cbar2.set_label(unit_label, rotation=270, labelpad=15)

    # 误差图
    im3 = axes[0, 2].imshow(error, cmap='Reds', aspect='auto', extent=extent)
    axes[0, 2].set_title('重建误差')
    axes[0, 2].set_xlabel('φ (度)')
    axes[0, 2].set_ylabel('θ (度)')
    cbar3 = plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    cbar3.set_label('误差', rotation=270, labelpad=15)

    # 统计信息
    stats_text = f'''分析信息:
小波域: 线性
显示数据: {unit_label}

重建质量:
MSE: {stats['mse']:.2e}
PSNR: {stats['psnr']:.1f} dB

能量分布:
LL: {stats['energy_ratios'][0]:.1%}
LH: {stats['energy_ratios'][1]:.1%}
HL: {stats['energy_ratios'][2]:.1%}
HH: {stats['energy_ratios'][3]:.1%}

平均稀疏度:
{np.mean(stats['sparsities']):.1%}'''

    axes[0, 3].text(0.05, 0.95, stats_text, transform=axes[0, 3].transAxes,
                   fontsize=10, verticalalignment='top')
    axes[0, 3].set_title('统计信息')
    axes[0, 3].axis('off')

    # 第二行：四个小波分量
    component_short_names = ['LL', 'LH', 'HL', 'HH']
    component_names_cn = ['LL (近似)', 'LH (水平)', 'HL (垂直)', 'HH (对角)']
    cmaps = ['jet', 'RdBu_r', 'RdBu_r', 'RdBu_r']

    for i, (comp_data, name, short_name, name_cn) in enumerate(zip(components, component_names, component_short_names, component_names_cn)):
        ax = axes[1, i]

        # 小波分量显示其原始尺寸的完整角度范围
        # 每个小波分量都对应完整的角度空间，只是分辨率是原图的一半
        im = ax.imshow(comp_data, cmap=cmaps[i], aspect='auto', extent=extent)

        # 添加统计信息到标题
        energy_pct = stats['energy_ratios'][i] * 100
        sparsity_pct = stats['sparsities'][i] * 100

        title = f'{name_cn}\\n能量: {energy_pct:.1f}%, 稀疏度: {sparsity_pct:.1f}%'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('φ (度)')
        ax.set_ylabel('θ (度)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig

def simple_performance_comparison(rcs_data, param_data, wavelet_system, direct_system, batch_size=10):
    """
    简化的性能对比函数

    Args:
        rcs_data: RCS数据
        param_data: 参数数据
        wavelet_system: 小波系统
        direct_system: 直接系统
        batch_size: 批次大小

    Returns:
        dict: 对比结果
    """
    import torch
    import time

    # 准备数据
    test_size = min(len(rcs_data), batch_size)
    test_rcs = rcs_data[:test_size]
    test_params = param_data[:test_size]

    rcs_tensor = torch.tensor(test_rcs, dtype=torch.float32)
    param_tensor = torch.tensor(test_params, dtype=torch.float32)

    results = {}

    # 测试小波系统
    print("测试小波增强系统...")
    wavelet_ae = wavelet_system['autoencoder']
    wavelet_mapper = wavelet_system['parameter_mapper']
    wavelet_transform = wavelet_system['wavelet_transform']

    wavelet_ae.eval()
    wavelet_mapper.eval()

    with torch.no_grad():
        start_time = time.time()

        # 小波模式推理
        wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
        recon_coeffs, latent = wavelet_ae(wavelet_coeffs)
        recon_rcs = wavelet_transform.inverse_transform(recon_coeffs)

        # 参数映射
        mapped_latent = wavelet_mapper(param_tensor)
        pred_coeffs = wavelet_ae.decode(mapped_latent)
        pred_rcs = wavelet_transform.inverse_transform(pred_coeffs)

        wavelet_time = time.time() - start_time

        # 计算误差
        wavelet_mse = torch.mean((rcs_tensor - recon_rcs)**2).item()

    results['wavelet'] = {
        'mse': wavelet_mse,
        'time': wavelet_time,
        'params': sum(p.numel() for p in wavelet_ae.parameters())
    }

    # 测试直接系统
    print("测试直接模式系统...")
    direct_ae = direct_system['autoencoder']
    direct_mapper = direct_system['parameter_mapper']

    direct_ae.eval()
    direct_mapper.eval()

    with torch.no_grad():
        start_time = time.time()

        # 直接模式推理
        recon_rcs, latent = direct_ae(rcs_tensor)

        # 参数映射
        mapped_latent = direct_mapper(param_tensor)
        pred_rcs = direct_ae.decode(mapped_latent)

        direct_time = time.time() - start_time

        # 计算误差
        direct_mse = torch.mean((rcs_tensor - recon_rcs)**2).item()

    results['direct'] = {
        'mse': direct_mse,
        'time': direct_time,
        'params': sum(p.numel() for p in direct_ae.parameters())
    }

    # 计算对比指标
    results['comparison'] = {
        'accuracy_improvement': (results['direct']['mse'] - results['wavelet']['mse']) / results['direct']['mse'] * 100,
        'speed_improvement': (results['wavelet']['time'] - results['direct']['time']) / results['wavelet']['time'] * 100,
        'param_reduction': (results['wavelet']['params'] - results['direct']['params']) / results['wavelet']['params'] * 100
    }

    return results

def create_comparison_plot(comparison_results):
    """
    创建对比分析图表

    Args:
        comparison_results: simple_performance_comparison的返回结果

    Returns:
        matplotlib.figure.Figure: 图表对象
    """
    # 设置matplotlib中文字体支持和编码
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('AutoEncoder Mode Performance Comparison', fontsize=16, fontweight='bold')

    # 1. MSE对比
    modes = ['Wavelet Enhanced', 'Direct Mode']
    mse_values = [comparison_results['wavelet']['mse'], comparison_results['direct']['mse']]

    bars1 = axes[0].bar(modes, mse_values, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0].set_ylabel('Reconstruction MSE')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_yscale('log')

    for bar, val in zip(bars1, mse_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom')

    # 2. 推理时间对比
    time_values = [comparison_results['wavelet']['time'], comparison_results['direct']['time']]

    bars2 = axes[1].bar(modes, time_values, color=['lightgreen', 'orange'], alpha=0.8)
    axes[1].set_ylabel('推理时间 (秒)')
    axes[1].set_title('推理速度对比')

    for bar, val in zip(bars2, time_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}s', ha='center', va='bottom')

    # 3. 参数数量对比
    param_values = [comparison_results['wavelet']['params'], comparison_results['direct']['params']]

    bars3 = axes[2].bar(modes, param_values, color=['purple', 'pink'], alpha=0.8)
    axes[2].set_ylabel('参数数量')
    axes[2].set_title('模型复杂度对比')

    for bar, val in zip(bars3, param_values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val/1e6:.1f}M', ha='center', va='bottom')

    plt.tight_layout()
    return fig