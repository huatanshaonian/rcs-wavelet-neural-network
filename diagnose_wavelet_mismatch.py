"""
诊断可微分小波CNN的尺寸不匹配问题
"""
import torch
import ptwt
import pywt

def diagnose_wavelet_size(wavelet_type='db4', input_size=91):
    """诊断给定小波基的输出尺寸"""
    print(f"\n{'='*60}")
    print(f"诊断小波基: {wavelet_type}")
    print(f"{'='*60}")

    # 1. pywt理论计算
    w = pywt.Wavelet(wavelet_type)
    pywt_size = pywt.dwt_coeff_len(input_size, w.dec_len, mode='symmetric')
    print(f"1. pywt理论计算: {input_size} → {pywt_size}")

    # 2. ptwt实际测试
    test_input = torch.randn(1, 1, input_size, input_size)
    coeffs = ptwt.wavedec2(test_input, wavelet_type, level=1, mode='symmetric')
    ptwt_size = coeffs[0].shape[2]
    print(f"2. ptwt实际输出: {input_size} → {ptwt_size}")

    # 3. 计算CNN flatten后的维度
    def calc_cnn_output(input_h):
        """计算经过4层stride=2卷积后的尺寸"""
        h = input_h
        # Layer 1: stride=2, padding=1, kernel=3
        h = (h + 2*1 - 3) // 2 + 1
        # Layer 2
        h = (h + 2*1 - 3) // 2 + 1
        # Layer 3
        h = (h + 2*1 - 3) // 2 + 1
        # Layer 4
        h = (h + 2*1 - 3) // 2 + 1
        return h

    cnn_h = calc_cnn_output(ptwt_size)
    flatten_dim = cnn_h * cnn_h * 256

    print(f"3. CNN输出尺寸: {ptwt_size}×{ptwt_size} → {cnn_h}×{cnn_h}×256 = {flatten_dim}")

    # 4. 检查是否匹配预期的4096
    expected = 4096
    if flatten_dim == expected:
        print(f"✓ 匹配预期维度 {expected}")
    else:
        print(f"✗ 不匹配！预期 {expected}，实际 {flatten_dim}")
        print(f"   差异: {expected - flatten_dim}")

    return ptwt_size, flatten_dim

if __name__ == "__main__":
    print("\n" + "="*60)
    print("可微分小波CNN尺寸诊断工具")
    print("="*60)

    # 测试常用小波基
    wavelets = ['db4', 'haar', 'coif1', 'bior1.3', 'sym4']

    print(f"\n{'小波基':<10} | {'小波输出':<10} | {'CNN flatten':<12} | {'匹配4096':<10}")
    print("-" * 60)

    for wavelet in wavelets:
        w = pywt.Wavelet(wavelet)
        pywt_size = pywt.dwt_coeff_len(91, w.dec_len, mode='symmetric')

        test_input = torch.randn(1, 1, 91, 91)
        coeffs = ptwt.wavedec2(test_input, wavelet, level=1, mode='symmetric')
        ptwt_size = coeffs[0].shape[2]

        def calc_cnn_output(input_h):
            h = input_h
            for _ in range(4):
                h = (h + 2*1 - 3) // 2 + 1
            return h

        cnn_h = calc_cnn_output(ptwt_size)
        flatten_dim = cnn_h * cnn_h * 256

        match = 'YES' if flatten_dim == 4096 else 'NO'
        print(f"{wavelet:<10} | {ptwt_size:<10} | {flatten_dim:<12} | {match:<10}")

    print("\n" + "="*60)
    print("建议:")
    print("="*60)
    print("1. 如果您的错误是 2304 vs 4096，说明使用了输出48×48的小波基")
    print("   (coif1 或 bior1.3)")
    print("2. 请检查训练配置中的小波基设置")
    print("3. 建议使用 db4 或 sym4 (输出49×49，匹配模型期望)")
    print("="*60)
