"""
小波变换尺寸计算的完整解释
解答：为什么91x91 → 49x49而不是45.5x45.5
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt

def explain_wavelet_sizing():
    """
    完整解释小波变换的尺寸计算机制
    """
    print("=== 小波变换尺寸计算完整解释 ===")
    print()

    # 关键问题：为什么不是严格的减半？
    print("🔍 核心问题：为什么91 → 49 而不是 91/2 = 45.5？")
    print()

    print("📊 答案：小波变换的尺寸计算公式")
    print("   output_size = floor((input_size + padding - 1) / decimation_factor) + 1")
    print("   其中：")
    print("   - padding取决于小波类型和边界处理模式")
    print("   - decimation_factor = 2 (下采样因子)")
    print("   - 实际计算还有PyWavelets的内部优化")
    print()

    # 详细分析db4小波
    wavelet = pywt.Wavelet('db4')
    filter_length = len(wavelet.dec_lo)

    print("🔬 db4小波的具体分析：")
    print(f"   滤波器长度: {filter_length}")
    print(f"   滤波器系数: {len(wavelet.dec_lo)}个")
    print(f"   边界处理: symmetric模式")
    print()

    # 尺寸计算步骤
    input_size = 91
    print("📐 逐步计算过程：")
    print(f"   1. 原始输入: {input_size} × {input_size}")

    # Padding计算
    if True:  # symmetric模式
        pad = filter_length - 1
        padded_size = input_size + pad
        print(f"   2. 添加padding: {input_size} + {pad} = {padded_size}")

    # 下采样
    downsampled = padded_size // 2
    print(f"   3. 下采样(÷2): {padded_size} // 2 = {downsampled}")

    # 实际结果
    test_data = np.random.randn(input_size, input_size)
    coeffs = pywt.dwt2(test_data, 'db4', mode='symmetric')
    cA, (cH, cV, cD) = coeffs
    actual_size = cA.shape[0]

    print(f"   4. PyWavelets输出: {actual_size} × {actual_size}")
    print(f"   5. 减少因子: {input_size/actual_size:.3f} (不是严格的2.0)")
    print()

    # 对比不同小波
    print("🌊 不同小波类型的对比：")
    wavelets = ['haar', 'db1', 'db4', 'db8', 'bior2.2']

    for wav in wavelets:
        if wav in pywt.wavelist():
            test_coeffs = pywt.dwt2(test_data, wav, mode='symmetric')
            wav_cA = test_coeffs[0]
            wav_filter_len = len(pywt.Wavelet(wav).dec_lo)
            reduction = input_size / wav_cA.shape[0]

            print(f"   {wav:8s}: {input_size}×{input_size} → {wav_cA.shape[0]}×{wav_cA.shape[1]} "
                  f"(滤波器长度:{wav_filter_len}, 减少{reduction:.2f}倍)")

    print()

    # 边界处理模式的影响
    print("🔧 边界处理模式的影响：")
    modes = ['symmetric', 'periodization', 'zero', 'constant']

    for mode in modes:
        try:
            mode_coeffs = pywt.dwt2(test_data, 'db4', mode=mode)
            mode_cA = mode_coeffs[0]
            reduction = input_size / mode_cA.shape[0]
            print(f"   {mode:12s}: {input_size}×{input_size} → {mode_cA.shape[0]}×{mode_cA.shape[1]} "
                  f"(减少{reduction:.2f}倍)")
        except:
            print(f"   {mode:12s}: 不支持")

    print()

    # 总结
    print("📋 总结：")
    print("   ✅ 小波变换确实是'减半'，但不是严格的数学除法")
    print("   ✅ 91 → 49 是正确的，因为：")
    print("      • db4滤波器需要7个填充像素")
    print("      • (91 + 7) ÷ 2 = 49")
    print("      • 这保证了完美重建能力")
    print("   ✅ 不同小波类型产生不同的输出尺寸")
    print("   ✅ 这是小波变换的标准行为，不是错误")
    print()

    print("🎯 对您项目的意义：")
    print("   • CNN架构设计正确：[B, 8, 49, 49]")
    print("   • 数据压缩比3.45倍是合理的")
    print("   • 3层CNN足够处理49×49的输入")

def demonstrate_size_patterns():
    """
    展示不同输入尺寸的模式
    """
    print("\n=== 输入尺寸模式规律 ===")

    # 测试一系列尺寸
    test_sizes = range(32, 129, 4)
    results = []

    for size in test_sizes:
        test_data = np.random.randn(size, size)
        coeffs = pywt.dwt2(test_data, 'db4', mode='symmetric')
        cA = coeffs[0]
        output_size = cA.shape[0]
        reduction = size / output_size

        results.append((size, output_size, reduction))

        if size <= 100:  # 只显示部分结果
            print(f"   {size:3d} → {output_size:2d} (减少{reduction:.2f}倍)")

    print("   ...")
    print(f"   {results[-1][0]:3d} → {results[-1][1]:2d} (减少{results[-1][2]:.2f}倍)")

    # 分析规律
    avg_reduction = np.mean([r[2] for r in results])
    print(f"\n   平均减少因子: {avg_reduction:.3f}")
    print(f"   标准差: {np.std([r[2] for r in results]):.3f}")
    print(f"   接近但不等于2.0，这是正常的！")

if __name__ == "__main__":
    explain_wavelet_sizing()
    demonstrate_size_patterns()