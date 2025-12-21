"""
Stage 1 重建诊断工具

用于诊断Stage 1训练时，小波系数完整但RCS重建出现空白的问题
"""

import numpy as np
import matplotlib.pyplot as plt


def diagnose_reconstruction_data(original_rcs, reconstructed_rcs,
                                 original_coeffs=None, reconstructed_coeffs=None):
    """
    诊断重建数据，查找空白区域的原因

    Args:
        original_rcs: 原始RCS数据 [N, H, W, C]
        reconstructed_rcs: 重建的RCS数据 [N, H, W, C]
        original_coeffs: 原始小波系数 [N, 49, 49, 8] (可选)
        reconstructed_coeffs: 重建小波系数 [N, 49, 49, 8] (可选)
    """
    print("="*80)
    print("Stage 1 重建数据诊断")
    print("="*80)

    # 1. 原始RCS数据分析
    print("\n【1. 原始RCS数据分析】")
    print(f"形状: {original_rcs.shape}")
    print(f"数值范围: [{original_rcs.min():.6e}, {original_rcs.max():.6e}]")
    print(f"均值: {original_rcs.mean():.6e}")
    print(f"标准差: {original_rcs.std():.6e}")

    # 检查零值和极小值
    zero_ratio = (original_rcs == 0).sum() / original_rcs.size
    tiny_ratio = (np.abs(original_rcs) < 1e-10).sum() / original_rcs.size
    print(f"零值占比: {zero_ratio*100:.2f}%")
    print(f"极小值(<1e-10)占比: {tiny_ratio*100:.2f}%")

    # 2. 重建RCS数据分析
    print("\n【2. 重建RCS数据分析】")
    print(f"形状: {reconstructed_rcs.shape}")
    print(f"数值范围: [{reconstructed_rcs.min():.6e}, {reconstructed_rcs.max():.6e}]")
    print(f"均值: {reconstructed_rcs.mean():.6e}")
    print(f"标准差: {reconstructed_rcs.std():.6e}")

    # 检查零值和极小值
    zero_ratio_recon = (reconstructed_rcs == 0).sum() / reconstructed_rcs.size
    tiny_ratio_recon = (np.abs(reconstructed_rcs) < 1e-10).sum() / reconstructed_rcs.size
    print(f"零值占比: {zero_ratio_recon*100:.2f}%")
    print(f"极小值(<1e-10)占比: {tiny_ratio_recon*100:.2f}%")

    # 检查负值
    negative_ratio = (reconstructed_rcs < 0).sum() / reconstructed_rcs.size
    print(f"负值占比: {negative_ratio*100:.2f}%")
    if negative_ratio > 0:
        print(f"  ⚠️ 警告: 重建RCS包含负值！物理上RCS应该≥0")
        print(f"  负值范围: [{reconstructed_rcs[reconstructed_rcs<0].min():.6e}, 0]")

    # 3. 空白区域分析
    print("\n【3. 空白区域分析】")

    # 定义"空白"阈值（根据可视化设置）
    blank_threshold = 1e-10  # 可以调整

    original_blank = original_rcs < blank_threshold
    reconstructed_blank = reconstructed_rcs < blank_threshold

    # 新增的空白（原本有数据，重建后变空白）
    new_blank = (~original_blank) & reconstructed_blank
    new_blank_ratio = new_blank.sum() / original_rcs.size

    print(f"原始数据空白区域: {original_blank.sum() / original_rcs.size * 100:.2f}%")
    print(f"重建数据空白区域: {reconstructed_blank.sum() / reconstructed_rcs.size * 100:.2f}%")
    print(f"新增空白区域: {new_blank_ratio * 100:.2f}%")

    if new_blank_ratio > 0.1:  # 超过10%
        print(f"  ⚠️ 警告: 重建后新增了 {new_blank_ratio*100:.2f}% 的空白区域！")
        print(f"  这可能是导致可视化出现白色区域的主要原因。")

    # 4. 重建误差分析
    print("\n【4. 重建误差分析】")

    # 避免除零
    epsilon = 1e-10

    # 绝对误差
    abs_error = np.abs(reconstructed_rcs - original_rcs)
    print(f"绝对误差 (MAE): {abs_error.mean():.6e}")
    print(f"绝对误差范围: [{abs_error.min():.6e}, {abs_error.max():.6e}]")

    # 相对误差（只在非零位置计算）
    non_zero_mask = original_rcs > epsilon
    if non_zero_mask.sum() > 0:
        rel_error = abs_error[non_zero_mask] / (original_rcs[non_zero_mask] + epsilon)
        print(f"相对误差 (非零区域): {rel_error.mean()*100:.2f}%")
        print(f"相对误差中位数: {np.median(rel_error)*100:.2f}%")

    # MSE
    mse = np.mean((reconstructed_rcs - original_rcs) ** 2)
    print(f"均方误差 (MSE): {mse:.6e}")
    print(f"均方根误差 (RMSE): {np.sqrt(mse):.6e}")

    # 5. 小波系数分析（如果提供）
    if original_coeffs is not None and reconstructed_coeffs is not None:
        print("\n【5. 小波系数分析】")
        print(f"原始小波系数形状: {original_coeffs.shape}")
        print(f"原始小波系数范围: [{original_coeffs.min():.6f}, {original_coeffs.max():.6f}]")
        print(f"重建小波系数形状: {reconstructed_coeffs.shape}")
        print(f"重建小波系数范围: [{reconstructed_coeffs.min():.6f}, {reconstructed_coeffs.max():.6f}]")

        # 小波系数的重建误差
        coeffs_mse = np.mean((reconstructed_coeffs - original_coeffs) ** 2)
        print(f"小波系数 MSE: {coeffs_mse:.6e}")

        # 检查小波系数是否有异常
        coeffs_zero_ratio = (reconstructed_coeffs == 0).sum() / reconstructed_coeffs.size
        print(f"重建小波系数零值占比: {coeffs_zero_ratio*100:.2f}%")

    # 6. 可视化建议
    print("\n【6. 诊断结论与建议】")

    if new_blank_ratio > 0.1:
        print("⚠️ 问题确认: 重建后确实产生了大量新的空白区域")
        print("\n可能原因：")
        print("  1. 模型训练不足 - Stage 1刚开始训练，模型还没学会如何重建")
        print("  2. 数值范围问题 - 重建值过小或出现负值被裁剪")
        print("  3. 逆小波变换问题 - 小波系数重建质量差，导致逆变换失败")

        if negative_ratio > 0:
            print("\n❌ 发现负值: 重建的RCS包含负值，这是不合理的！")
            print("  建议: 检查decoder输出是否应该加激活函数（如ReLU或Softplus）")

        if original_coeffs is not None and coeffs_mse > 1.0:
            print("\n⚠️ 小波系数重建误差较大")
            print("  建议: 增加训练epoch数，或调整学习率")
    else:
        print("✅ 重建质量正常: 新增空白区域较少")
        if zero_ratio > 0.3:
            print("\n📌 注意: 原始RCS数据本身就有较多零值区域")
            print(f"  原始数据零值占比: {zero_ratio*100:.2f}%")
            print("  → 这种情况下看到空白区域是正常的！")

    # 7. 可视化对比（保存为图片）
    print("\n【7. 生成对比图】")

    # 选择第一个样本和第一个频率
    sample_idx = 0
    freq_idx = 0

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始RCS
    im1 = axes[0, 0].imshow(original_rcs[sample_idx, :, :, freq_idx], cmap='jet')
    axes[0, 0].set_title(f'Original RCS (Freq {freq_idx})')
    axes[0, 0].set_xlabel('Phi')
    axes[0, 0].set_ylabel('Theta')
    plt.colorbar(im1, ax=axes[0, 0])

    # 重建RCS
    im2 = axes[0, 1].imshow(reconstructed_rcs[sample_idx, :, :, freq_idx], cmap='jet')
    axes[0, 1].set_title(f'Reconstructed RCS (Freq {freq_idx})')
    axes[0, 1].set_xlabel('Phi')
    axes[0, 1].set_ylabel('Theta')
    plt.colorbar(im2, ax=axes[0, 1])

    # 误差图
    error_map = np.abs(reconstructed_rcs[sample_idx, :, :, freq_idx] -
                       original_rcs[sample_idx, :, :, freq_idx])
    im3 = axes[0, 2].imshow(error_map, cmap='hot')
    axes[0, 2].set_title('Absolute Error')
    axes[0, 2].set_xlabel('Phi')
    axes[0, 2].set_ylabel('Theta')
    plt.colorbar(im3, ax=axes[0, 2])

    # dB尺度对比
    original_db = 10 * np.log10(np.maximum(original_rcs[sample_idx, :, :, freq_idx], 1e-10))
    reconstructed_db = 10 * np.log10(np.maximum(reconstructed_rcs[sample_idx, :, :, freq_idx], 1e-10))

    im4 = axes[1, 0].imshow(original_db, cmap='jet', vmin=-50, vmax=10)
    axes[1, 0].set_title('Original RCS (dB)')
    axes[1, 0].set_xlabel('Phi')
    axes[1, 0].set_ylabel('Theta')
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(reconstructed_db, cmap='jet', vmin=-50, vmax=10)
    axes[1, 1].set_title('Reconstructed RCS (dB)')
    axes[1, 1].set_xlabel('Phi')
    axes[1, 1].set_ylabel('Theta')
    plt.colorbar(im5, ax=axes[1, 1])

    # 空白区域标记
    blank_mask = reconstructed_rcs[sample_idx, :, :, freq_idx] < blank_threshold
    axes[1, 2].imshow(blank_mask, cmap='gray')
    axes[1, 2].set_title(f'Blank Regions (< {blank_threshold:.0e})')
    axes[1, 2].set_xlabel('Phi')
    axes[1, 2].set_ylabel('Theta')

    plt.tight_layout()
    plt.savefig('stage1_reconstruction_diagnosis.png', dpi=150, bbox_inches='tight')
    print("对比图已保存: stage1_reconstruction_diagnosis.png")

    print("\n" + "="*80)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("""
使用方法：

1. 在GUI中训练Stage 1后，在Python控制台运行：

```python
import sys
sys.path.append('G:/feko_data/wavelet')
from diagnose_stage1_reconstruction import diagnose_reconstruction_data

# 获取测试数据
test_rcs = rcs_data[-20:]  # 后20个样本

# 使用AutoEncoder重建
result = gui._reconstruct_rcs(
    input_data=test_rcs,
    input_type='rcs',
    return_wavelet_coeffs=True  # 获取小波系数
)

reconstructed_rcs = result['reconstructed_rcs']
original_coeffs = result.get('original_wavelet_coeffs')
reconstructed_coeffs = result.get('reconstructed_wavelet_coeffs')

# 运行诊断
diagnose_reconstruction_data(
    original_rcs=test_rcs,
    reconstructed_rcs=reconstructed_rcs,
    original_coeffs=original_coeffs,
    reconstructed_coeffs=reconstructed_coeffs
)
```

2. 查看输出的诊断信息和生成的对比图 stage1_reconstruction_diagnosis.png
    """)
