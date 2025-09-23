"""
诊断RCS数据维度和对称性问题
"""
import numpy as np
import torch
from rcs_data_reader import get_adaptive_rcs_matrix
from training import RCSDataLoader

print("=" * 80)
print("诊断RCS数据维度和对称性约束问题")
print("=" * 80)

data_dir = r"..\parameter\csv_output"
model_id = "001"

print("\n1. 从CSV文件读取原始数据:")
print("-" * 80)
rcs_15 = get_adaptive_rcs_matrix(model_id, "1.5G", data_dir, verbose=True)
print(f"\nTheta values: 范围 {rcs_15['theta_values'].min()} - {rcs_15['theta_values'].max()}")
print(f"Phi values: 范围 {rcs_15['phi_values'].min()} - {rcs_15['phi_values'].max()}")
print(f"RCS矩阵形状: {rcs_15['rcs_linear'].shape}")
print(f"注意: 矩阵形状是 (theta, phi) = ({len(rcs_15['theta_values'])}, {len(rcs_15['phi_values'])})")

print("\n2. RCSDataLoader如何处理数据:")
print("-" * 80)
loader = RCSDataLoader(data_dir)
rcs_array, angle_range = loader.load_rcs_data(num_models=1, frequencies=["1.5G", "3G"])
print(f"RCS数组形状: {rcs_array.shape}")
print(f"解释: [num_models, num_phi, num_theta, num_freq]")
print(f"      [{rcs_array.shape[0]}, {rcs_array.shape[1]}, {rcs_array.shape[2]}, {rcs_array.shape[3]}]")

print("\n3. 神经网络输出格式:")
print("-" * 80)
print("网络输出: [batch, 91, 91, 2]")
print("当前代码假设: [batch, phi, theta, freq]")
print("这意味着:")
print("  - 第1维(91): phi (偏航角, 方位角)")
print("  - 第2维(91): theta (俯仰角, 仰角)")

print("\n4. 对称性约束检查:")
print("-" * 80)
print("当前对称性约束代码:")
print("  center_phi = 45  # 假设phi=0°在索引45")
print("  对第1维应用对称: rcs[:, left_idx, :, :] <-> rcs[:, right_idx, :, :]")
print("\n问题:")
print("  - 如果phi范围是0-90°,中心不在45")
print("  - 需要确认phi=0°对应哪个索引")

print(f"\nPhi值详情:")
phi_vals = rcs_15['phi_values']
print(f"  范围: {phi_vals.min():.1f}° - {phi_vals.max():.1f}°")
print(f"  数量: {len(phi_vals)}")
print(f"  步长: {np.mean(np.diff(phi_vals)):.1f}°")
if 0 in phi_vals:
    phi_0_idx = np.where(phi_vals == 0)[0][0]
    print(f"  phi=0°的索引: {phi_0_idx}")
else:
    closest_idx = np.argmin(np.abs(phi_vals))
    print(f"  最接近0°的phi值: {phi_vals[closest_idx]:.1f}°, 索引: {closest_idx}")

print(f"\nTheta值详情:")
theta_vals = rcs_15['theta_values']
print(f"  范围: {theta_vals.min():.1f}° - {theta_vals.max():.1f}°")
print(f"  数量: {len(theta_vals)}")
print(f"  步长: {np.mean(np.diff(theta_vals)):.1f}°")
if 90 in theta_vals:
    theta_90_idx = np.where(theta_vals == 90)[0][0]
    print(f"  theta=90°的索引: {theta_90_idx}")

print("\n5. 损失函数与评估指标的尺度问题:")
print("-" * 80)
print("损失函数(MSE)计算的是:")
print("  - 如果use_log_output=True: 对数域MSE (标准化后的dB值)")
print("  - 如果use_log_output=False: 线性域MSE (原始RCS值)")

print("\n评估指标(RMSE, R²)计算的是:")
print("  - 总是在线性域计算 (需要反向转换)")
print("\n如果损失在对数域,但评估在线性域,会导致:")
print("  - 低loss (如0.01)对应的对数域误差很小")
print("  - 但反转换到线性域后RMSE会很大")
print("  - R²可能为负(预测比均值还差)")

print("\n6. 数据形状总结:")
print("-" * 80)
print("CSV原始数据:")
print(f"  Theta: {len(rcs_15['theta_values'])} 个值")
print(f"  Phi: {len(rcs_15['phi_values'])} 个值")
print(f"  矩阵: ({len(rcs_15['theta_values'])}, {len(rcs_15['phi_values'])})")

print("\nRCSDataLoader加载后:")
print(f"  形状: {rcs_array.shape}")
print(f"  维度顺序: [models, phi, theta, freq]")

print("\n网络输出:")
print("  形状: [batch, 91, 91, 2]")
print("  当前假设: [batch, phi, theta, freq]")

print("\n对称性物理约束:")
print("  phi=0°平面对称: σ(phi, theta) = σ(-phi, theta)")
print("  即关于phi=0°对称,theta保持不变")
print("  当前代码对第1维(假设是phi)应用对称")

print("\n" + "=" * 80)
print("关键发现:")
print("=" * 80)
print("\n可能的问题:")
print("1. 对称性约束的中心索引可能不对 (当前假设45,但phi=0°可能不在45)")
print("2. 损失函数在对数域,但RMSE在线性域,导致指标不匹配")
print("3. 可视化中看到的对称轴如果是theta=90°,说明维度可能反了")
print("\n需要修复:")
print("1. 确认phi=0°的实际索引")
print("2. 统一损失函数和评估指标的域")
print("3. 检查可视化代码的维度假设")