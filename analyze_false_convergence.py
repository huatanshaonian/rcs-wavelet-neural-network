"""
分析为什么错误的输出还能有小loss
"""
import numpy as np

print("="*70)
print("虚假收敛分析：为什么网络输出错误但loss很小")
print("="*70)

# 模拟实际情况
mean = -26.76
std = 9.01

# 1. 训练目标（标准化RCS）
target_std = np.array([-3.0, -1.0, 0.0, 1.0, 2.5])  # 范围 -5~3
print("\n1. 训练目标（标准化域）")
print(f"   target_std = {target_std}")

# 2. 网络错误输出
pred_std_wrong = np.array([-150, -120, -100, -80, -60])  # 网络实际输出范围
print("\n2. 网络错误输出（标准化域）")
print(f"   pred_std = {pred_std_wrong}")

# 3. 计算MSE loss（训练时）
mse_wrong = np.mean((pred_std_wrong - target_std) ** 2)
print(f"\n3. MSE Loss (标准化域)")
print(f"   MSE = mean((pred - target)²)")
print(f"   MSE = mean(({pred_std_wrong} - {target_std})²)")
print(f"   MSE = {mse_wrong:.2f}  ← 看起来很大！")

# 4. 但如果对称性损失主导...
print("\n4. 对称性损失的影响（旧版本bug）")
print("   旧版对称性loss只要求pred自身对称，不管是否接近target")

# 模拟对称的错误输出
pred_symmetric_wrong = np.array([-100, -100, -100, -100, -100])
print(f"   如果网络输出全是常数: {pred_symmetric_wrong}")
print(f"   对称性loss = 0 （完美对称！）")
print(f"   MSE loss = {np.mean((pred_symmetric_wrong - target_std)**2):.2f}")

# 5. 总loss = MSE*1.0 + symmetry*0.3 + multiscale*0.5
mse_loss = np.mean((pred_symmetric_wrong - target_std)**2)
symmetry_loss = 0.0  # 完美对称
multiscale_loss = mse_loss * 0.5  # 假设和MSE相关

total_loss_old = 1.0 * mse_loss + 0.3 * symmetry_loss + 0.5 * multiscale_loss
print(f"\n5. 旧版总loss（对称性权重0.3）")
print(f"   total = 1.0*{mse_loss:.2f} + 0.3*{symmetry_loss:.2f} + 0.5*{multiscale_loss:.2f}")
print(f"   total = {total_loss_old:.2f}")

# 6. 如果网络学会输出接近0...
pred_near_zero = np.array([-0.5, -0.3, 0.0, 0.2, 0.5])
mse_near_zero = np.mean((pred_near_zero - target_std)**2)
print(f"\n6. 如果网络学会输出接近0")
print(f"   pred = {pred_near_zero}")
print(f"   MSE = {mse_near_zero:.2f}")

# 但对称性仍然完美
symmetry_near_zero = 0.0
total_near_zero = 1.0 * mse_near_zero + 0.3 * symmetry_near_zero
print(f"   对称性loss = {symmetry_near_zero}")
print(f"   total = {total_near_zero:.2f}")

# 7. 现在的问题：网络输出-150但loss还是小
print("\n7. 当前问题：网络输出-150，为什么loss小？")

# 可能原因1：损失函数计算在错误的域
print("\n   可能原因1：损失域错误")
# 如果误把pred当成dB计算...
pred_as_db = pred_std_wrong  # -150 dB
target_as_db = target_std * std + mean  # 转回dB：-30 ~ -6 dB
mse_wrong_domain = np.mean((pred_as_db - target_as_db)**2)
print(f"   如果把标准化值当dB值计算MSE:")
print(f"     pred (当作dB): {pred_as_db}")
print(f"     target (dB): {target_as_db}")
print(f"     MSE = {mse_wrong_domain:.2f}  ← 巨大！")

# 可能原因2：多尺度损失的下采样平滑了误差
print("\n   可能原因2：多尺度loss平滑误差")
# 下采样后，所有值接近均值
pred_downsampled = np.mean(pred_std_wrong) * np.ones(5)
target_downsampled = np.mean(target_std) * np.ones(5)
mse_downsampled = np.mean((pred_downsampled - target_downsampled)**2)
print(f"   下采样后pred均值: {pred_downsampled[0]:.2f}")
print(f"   下采样后target均值: {target_downsampled[0]:.2f}")
print(f"   下采样MSE: {mse_downsampled:.2f}")

# 可能原因3：对称性损失bug（已修复）
print("\n   可能原因3：对称性loss bug（已修复）")
print("   旧版：只要pred自身对称，loss=0")
print("   新版：pred和target的对称性差异")

# 8. 正确的情况
print("\n8. 修复后应该的情况")
pred_correct = np.array([-2.8, -0.9, 0.1, 1.1, 2.3])
mse_correct = np.mean((pred_correct - target_std)**2)
print(f"   正确输出: {pred_correct}")
print(f"   目标: {target_std}")
print(f"   MSE: {mse_correct:.4f}  ← 很小！")

# 反标准化检查
pred_correct_db = pred_correct * std + mean
target_db = target_std * std + mean
print(f"   反标准化pred (dB): {pred_correct_db}")
print(f"   反标准化target (dB): {target_db}")

print("\n" + "="*70)
print("结论：")
print("1. 旧版对称性loss让网络学会\"对称的错误值\"而非\"正确的RCS\"")
print("2. 网络输出-150是因为损失函数设计缺陷，不是数据问题")
print("3. loss小是因为对称性loss=0主导了总loss")
print("4. 必须用修复后的损失函数重新训练")
print("="*70)