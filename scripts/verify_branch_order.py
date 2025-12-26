"""
验证Additive Dual-Branch模型的分支输出顺序

检查decoder_high (SIN激活) 和 decoder_smooth (LEAKY_RELU激活) 的输出
是否与forward_with_branches()的返回顺序一致
"""

import torch
import numpy as np
from autoencoder.models.additive_dual_branch_mlp import AdditiveDualBranchDirectMLPAutoEncoder

# 加载模型
print("===== 加载模型 =====")
checkpoint = torch.load('G:/feko_data/wavelet/models/direct_additive_dual_branch_mlp_relu_raw_20251224_153337.pth',
                        map_location='cpu')

model = AdditiveDualBranchDirectMLPAutoEncoder(
    latent_dim=64,
    num_frequencies=2,
    dropout_rate=0.2,
    input_size=91,
    activation_encoder='relu',
    activation_high='sin',
    activation_smooth='leaky_relu',
    learnable_weights=False,
    alpha_high=1.0,
    alpha_smooth=1.0
)

model.load_state_dict(checkpoint['autoencoder'])
model.eval()

print(f"模型激活函数配置:")
print(f"  activation_high_type: {model.activation_high_type}")
print(f"  activation_smooth_type: {model.activation_smooth_type}")
print()

# 创建随机输入
print("===== 创建测试输入 =====")
x = torch.randn(1, 91, 91, 2)  # [B, H, W, C]
print(f"输入shape: {x.shape}")
print()

# 方法1：直接调用decoder
print("===== 方法1：直接调用decoder =====")
with torch.no_grad():
    latent = model.encode(x)
    out_high_direct = model.decoder_high(latent)
    out_smooth_direct = model.decoder_smooth(latent)

out_high_direct_2d = out_high_direct.reshape(1, 91, 91, -1)[0, :, :, 0].numpy()
out_smooth_direct_2d = out_smooth_direct.reshape(1, 91, 91, -1)[0, :, :, 0].numpy()

print(f"decoder_high输出统计 (应该来自SIN激活):")
print(f"  Mean: {out_high_direct_2d.mean():.6f}")
print(f"  Std:  {out_high_direct_2d.std():.6f}")
print(f"  Min:  {out_high_direct_2d.min():.6f}")
print(f"  Max:  {out_high_direct_2d.max():.6f}")
print()

print(f"decoder_smooth输出统计 (应该来自LEAKY_RELU激活):")
print(f"  Mean: {out_smooth_direct_2d.mean():.6f}")
print(f"  Std:  {out_smooth_direct_2d.std():.6f}")
print(f"  Min:  {out_smooth_direct_2d.min():.6f}")
print(f"  Max:  {out_smooth_direct_2d.max():.6f}")
print()

# 方法2：通过forward_with_branches
print("===== 方法2：通过forward_with_branches =====")
with torch.no_grad():
    recon, latent_out, recon_high, recon_smooth = model.forward_with_branches(x)

recon_high_2d = recon_high[0, :, :, 0].numpy()
recon_smooth_2d = recon_smooth[0, :, :, 0].numpy()

print(f"recon_high输出统计 (forward_with_branches第3个返回值):")
print(f"  Mean: {recon_high_2d.mean():.6f}")
print(f"  Std:  {recon_high_2d.std():.6f}")
print(f"  Min:  {recon_high_2d.min():.6f}")
print(f"  Max:  {recon_high_2d.max():.6f}")
print()

print(f"recon_smooth输出统计 (forward_with_branches第4个返回值):")
print(f"  Mean: {recon_smooth_2d.mean():.6f}")
print(f"  Std:  {recon_smooth_2d.std():.6f}")
print(f"  Min:  {recon_smooth_2d.min():.6f}")
print(f"  Max:  {recon_smooth_2d.max():.6f}")
print()

# 验证一致性
print("===== 验证一致性 =====")
diff_high = np.abs(out_high_direct_2d - recon_high_2d).max()
diff_smooth = np.abs(out_smooth_direct_2d - recon_smooth_2d).max()

print(f"decoder_high vs recon_high 最大差异: {diff_high:.10f}")
print(f"decoder_smooth vs recon_smooth 最大差异: {diff_smooth:.10f}")
print()

if diff_high < 1e-6 and diff_smooth < 1e-6:
    print("✅ 一致性验证通过！")
    print("✅ recon_high确实来自decoder_high (SIN激活)")
    print("✅ recon_smooth确实来自decoder_smooth (LEAKY_RELU激活)")
else:
    print("❌ 发现不一致！可能存在交换！")

# 频谱分析
print()
print("===== 频谱特征分析 =====")
from scipy.fft import fft2, fftshift

def analyze_frequency_content(data_2d, name):
    """简单的频率内容分析"""
    # 2D FFT
    fft_result = fftshift(fft2(data_2d))
    magnitude = np.abs(fft_result)

    # 计算高频能量占比（外围30%）
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius_threshold = int(min(h, w) * 0.35)  # 35%半径作为分界

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - center_h)**2 + (x - center_w)**2)

    low_freq_mask = distance <= radius_threshold
    high_freq_mask = distance > radius_threshold

    low_freq_energy = magnitude[low_freq_mask].sum()
    high_freq_energy = magnitude[high_freq_mask].sum()
    total_energy = low_freq_energy + high_freq_energy

    high_ratio = high_freq_energy / total_energy
    low_ratio = low_freq_energy / total_energy

    print(f"{name}:")
    print(f"  高频能量占比: {high_ratio*100:.1f}%")
    print(f"  低频能量占比: {low_ratio*100:.1f}%")

    if high_ratio > 0.5:
        print(f"  → 高频主导 (符合SIN激活特征)")
    else:
        print(f"  → 低频主导 (符合RELU激活特征)")
    print()

    return high_ratio

high_ratio_high = analyze_frequency_content(recon_high_2d, "recon_high (应为SIN)")
high_ratio_smooth = analyze_frequency_content(recon_smooth_2d, "recon_smooth (应为RELU)")

print("===== 最终结论 =====")
if high_ratio_high > 0.5:
    print("✅ recon_high显示高频特征，符合SIN激活预期")
else:
    print("⚠️ recon_high显示低频特征，不符合SIN激活预期！")

if high_ratio_smooth < 0.5:
    print("✅ recon_smooth显示低频特征，符合RELU激活预期")
else:
    print("⚠️ recon_smooth显示高频特征，不符合RELU激活预期！")
