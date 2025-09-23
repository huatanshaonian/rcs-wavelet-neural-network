#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角度范围诊断脚本

验证实际的RCS数据中的角度定义和范围，
找出网络结构和可视化中的角度问题
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_angle_ranges():
    """诊断实际的角度范围"""
    print("=" * 80)
    print("RCS数据角度范围诊断")
    print("=" * 80)

    try:
        # 导入数据读取模块
        import rcs_visual as rv
        from training import create_data_config

        # 获取数据配置
        data_config = create_data_config()
        rcs_dir = data_config['rcs_data_dir']

        print(f"RCS数据目录: {rcs_dir}")

        if not os.path.exists(rcs_dir):
            print(f"错误: RCS数据目录不存在: {rcs_dir}")
            return

        # 查找第一个可用的模型进行分析
        test_model = None
        for file in os.listdir(rcs_dir):
            if file.endswith('_1.5G.csv'):
                test_model = file.split('_')[0]
                break

        if not test_model:
            print("错误: 未找到RCS数据文件")
            return

        print(f"使用测试模型: {test_model}")

        # 分别分析两个频率的数据
        frequencies = ['1.5G', '3G']
        angle_info = {}

        for freq in frequencies:
            print(f"\n分析 {freq} 频率数据...")

            try:
                # 获取RCS矩阵数据
                matrix_data = rv.get_rcs_matrix(test_model, freq, rcs_dir)

                print(f"  数据形状: {matrix_data['rcs_linear'].shape}")
                print(f"  RCS线性值范围: {matrix_data['data_info']['rcs_linear_range']}")
                print(f"  RCS分贝值范围: {matrix_data['data_info']['rcs_db_range']}")
                print(f"  角度信息:")
                print(f"    Theta范围: {matrix_data['data_info']['theta_range']}")
                print(f"    Phi范围: {matrix_data['data_info']['phi_range']}")
                print(f"    Theta值数量: {len(matrix_data['theta_values'])}")
                print(f"    Phi值数量: {len(matrix_data['phi_values'])}")

                # 详细角度分析
                theta_values = matrix_data['theta_values']
                phi_values = matrix_data['phi_values']

                print(f"  详细角度分析:")
                print(f"    Theta: min={theta_values.min():.1f}°, max={theta_values.max():.1f}°, 步长={np.diff(theta_values).mean():.1f}°")
                print(f"    Phi: min={phi_values.min():.1f}°, max={phi_values.max():.1f}°, 步长={np.diff(phi_values).mean():.1f}°")

                # 检查角度网格的规律性
                theta_steps = np.diff(theta_values)
                phi_steps = np.diff(phi_values)

                print(f"    Theta步长统计: 均值={theta_steps.mean():.2f}°, 标准差={theta_steps.std():.4f}°")
                print(f"    Phi步长统计: 均值={phi_steps.mean():.2f}°, 标准差={phi_steps.std():.4f}°")

                # 保存角度信息
                angle_info[freq] = {
                    'theta_range': matrix_data['data_info']['theta_range'],
                    'phi_range': matrix_data['data_info']['phi_range'],
                    'theta_values': theta_values,
                    'phi_values': phi_values,
                    'shape': matrix_data['rcs_linear'].shape
                }

            except Exception as e:
                print(f"  错误: 无法读取{freq}数据: {e}")

        # 比较两个频率的角度网格
        if '1.5G' in angle_info and '3G' in angle_info:
            print(f"\n角度网格一致性检查:")

            theta_1_5g = angle_info['1.5G']['theta_values']
            theta_3g = angle_info['3G']['theta_values']
            phi_1_5g = angle_info['1.5G']['phi_values']
            phi_3g = angle_info['3G']['phi_values']

            theta_match = np.allclose(theta_1_5g, theta_3g)
            phi_match = np.allclose(phi_1_5g, phi_3g)

            print(f"  Theta网格一致: {theta_match}")
            print(f"  Phi网格一致: {phi_match}")

            if not theta_match:
                print(f"    1.5G Theta范围: {theta_1_5g.min():.1f}° - {theta_1_5g.max():.1f}°")
                print(f"    3G Theta范围: {theta_3g.min():.1f}° - {theta_3g.max():.1f}°")

            if not phi_match:
                print(f"    1.5G Phi范围: {phi_1_5g.min():.1f}° - {phi_1_5g.max():.1f}°")
                print(f"    3G Phi范围: {phi_3g.min():.1f}° - {phi_3g.max():.1f}°")

        # 分析网络假设与实际数据的差异
        print(f"\n网络假设vs实际数据分析:")
        if angle_info:
            first_freq = list(angle_info.keys())[0]
            actual_shape = angle_info[first_freq]['shape']
            actual_theta_range = angle_info[first_freq]['theta_range']
            actual_phi_range = angle_info[first_freq]['phi_range']

            print(f"  实际数据形状: {actual_shape}")
            print(f"  网络假设形状: (91, 91)")
            print(f"  形状匹配: {actual_shape == (91, 91)}")

            print(f"\n  实际角度范围:")
            print(f"    Theta: {actual_theta_range[0]:.1f}° - {actual_theta_range[1]:.1f}°")
            print(f"    Phi: {actual_phi_range[0]:.1f}° - {actual_phi_range[1]:.1f}°")

            print(f"\n  网络假设的角度范围 (基于对称性约束代码):")
            print(f"    Phi: -45° - +45° (基于center_phi=45的假设)")

            # 检查对称性约束的正确性
            phi_values = angle_info[first_freq]['phi_values']
            center_phi_value = 0.0  # 实际的φ=0°值

            # 找到最接近0°的索引
            center_phi_idx = np.argmin(np.abs(phi_values - center_phi_value))
            actual_center_phi = phi_values[center_phi_idx]

            print(f"\n  对称性约束分析:")
            print(f"    网络假设的φ=0°索引: 45")
            print(f"    实际φ=0°最接近的索引: {center_phi_idx}")
            print(f"    实际φ=0°对应的值: {actual_center_phi:.1f}°")
            print(f"    网络假设错误程度: {abs(45 - center_phi_idx)} 个索引位置")

        # 创建可视化图表
        if angle_info:
            print(f"\n创建角度网格可视化...")

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('RCS数据角度网格分析', fontsize=14)

            for i, freq in enumerate(['1.5G', '3G']):
                if freq in angle_info:
                    theta_vals = angle_info[freq]['theta_values']
                    phi_vals = angle_info[freq]['phi_values']

                    # Theta分布
                    axes[0, i].plot(theta_vals, 'bo-', markersize=3)
                    axes[0, i].set_title(f'{freq} Theta角度分布')
                    axes[0, i].set_xlabel('索引')
                    axes[0, i].set_ylabel('Theta (度)')
                    axes[0, i].grid(True, alpha=0.3)

                    # Phi分布
                    axes[1, i].plot(phi_vals, 'ro-', markersize=3)
                    axes[1, i].set_title(f'{freq} Phi角度分布')
                    axes[1, i].set_xlabel('索引')
                    axes[1, i].set_ylabel('Phi (度)')
                    axes[1, i].grid(True, alpha=0.3)

                    # 标记φ=0°位置
                    center_idx = np.argmin(np.abs(phi_vals - 0.0))
                    axes[1, i].axvline(center_idx, color='green', linestyle='--',
                                     label=f'φ=0° (索引{center_idx})')
                    axes[1, i].axvline(45, color='red', linestyle='--',
                                     label='网络假设(索引45)')
                    axes[1, i].legend()

            plt.tight_layout()
            plt.savefig('angle_grid_analysis.png', dpi=150, bbox_inches='tight')
            print(f"  角度网格分析图已保存: angle_grid_analysis.png")

        # 输出修复建议
        print(f"\n" + "=" * 80)
        print("修复建议:")
        print("=" * 80)

        if angle_info:
            first_freq = list(angle_info.keys())[0]
            actual_shape = angle_info[first_freq]['shape']
            phi_values = angle_info[first_freq]['phi_values']
            center_phi_idx = np.argmin(np.abs(phi_values - 0.0))

            print(f"1. 网络结构修复:")
            print(f"   - 当前假设: 91x91网格，φ=0°在索引45")
            print(f"   - 实际情况: {actual_shape[0]}x{actual_shape[1]}网格，φ=0°在索引{center_phi_idx}")
            print(f"   - 建议: 动态获取角度网格信息，修正对称性约束")

            print(f"\n2. 对称性约束修复:")
            print(f"   - 修改 wavelet_network.py 中的 _apply_phi_symmetry 函数")
            print(f"   - 将 center_phi = 45 改为 center_phi = {center_phi_idx}")
            print(f"   - 或者动态计算center_phi索引")

            print(f"\n3. 可视化角度范围修复:")
            print(f"   - 确保可视化使用正确的角度标签")
            print(f"   - 验证extent参数设置正确")

            print(f"\n4. 训练数据流检查:")
            print(f"   - 验证数据加载时的角度信息传递")
            print(f"   - 确保网络接收正确的角度上下文")

    except Exception as e:
        print(f"诊断过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_angle_ranges()