# -*- coding: utf-8 -*-
"""
Verify loss function calculation
Test perfect prediction scenario
"""
import numpy as np
import torch
from wavelet_network import create_loss_function

print("="*70)
print("LOSS FUNCTION VERIFICATION")
print("="*70)

# Preprocessing stats (from training)
mean_db = -26.76
std_db = 9.01

# 1. Simulate original dB data
print("\n[1] Simulate original RCS data (dB)")
np.random.seed(42)
original_db = np.random.uniform(-60, 0, (91, 91, 2))
print(f"Original RCS (dB): {original_db.min():.2f} ~ {original_db.max():.2f} dB")

# 2. Standardize (training preprocessing)
print("\n[2] Standardize (simulate training preprocessing)")
target_std = (original_db - mean_db) / std_db
print(f"Standardized: {target_std.min():.3f} ~ {target_std.max():.3f}")
print(f"  (expected range: -5 ~ 3)")

# 3. Perfect prediction: pred = target
print("\n[3] Perfect prediction: pred = target")
pred_std = target_std.copy()

pred_tensor = torch.FloatTensor(pred_std).unsqueeze(0)  # [1, 91, 91, 2]
target_tensor = torch.FloatTensor(target_std).unsqueeze(0)

# 4. Calculate training loss (standardized domain)
print("\n[4] Calculate training loss (standardized domain)")
loss_fn = create_loss_function()

with torch.no_grad():
    losses = loss_fn(pred_tensor, target_tensor)

print(f"Perfect prediction loss:")
print(f"  MSE: {losses['mse'].item():.10f}")
print(f"  Symmetry: {losses['symmetry'].item():.10f}")
print(f"  Multiscale: {losses['multiscale'].item():.10f}")
print(f"  Total: {losses['total'].item():.10f}")

if losses['total'].item() < 1e-6:
    print("  [OK] Perfect prediction loss is near 0")
else:
    print(f"  [WARNING] Perfect prediction loss = {losses['total'].item():.6f}")

# 5. De-standardize
print("\n[5] De-standardize to dB domain")
pred_db = pred_std * std_db + mean_db
diff_db = np.abs(pred_db - original_db)
print(f"pred (dB): {pred_db.min():.2f} ~ {pred_db.max():.2f} dB")
print(f"Diff from original: max {diff_db.max():.6f} dB")

if diff_db.max() < 1e-10:
    print("  [OK] De-standardization correct")

# 6. Convert to linear
print("\n[6] Convert to linear domain")
original_linear = np.power(10, original_db / 10)
pred_linear = np.power(10, pred_db / 10)

print(f"Original (linear): {original_linear.min():.6e} ~ {original_linear.max():.6e}")
print(f"pred (linear): {pred_linear.min():.6e} ~ {pred_linear.max():.6e}")

# 7. Calculate evaluation metrics (linear domain)
print("\n[7] Calculate evaluation metrics (linear domain)")
rmse = np.sqrt(np.mean((pred_linear - original_linear)**2))
ss_res = np.sum((original_linear - pred_linear)**2)
ss_tot = np.sum((original_linear - original_linear.mean())**2)
r2 = 1 - ss_res / ss_tot

print(f"RMSE (linear): {rmse:.6e}")
print(f"R2: {r2:.10f}")

if r2 > 0.9999:
    print("  [OK] Perfect prediction R2 is near 1")

# 8. Test with noise
print("\n" + "="*70)
print("[8] Test with noise (std=0.1 in standardized domain)")
print("="*70)

np.random.seed(123)
noise_std = np.random.randn(91, 91, 2) * 0.1
pred_noisy_std = target_std + noise_std

pred_noisy_tensor = torch.FloatTensor(pred_noisy_std).unsqueeze(0)

# Calculate noisy loss
with torch.no_grad():
    losses_noisy = loss_fn(pred_noisy_tensor, target_tensor)

print(f"\nStandardized domain loss:")
print(f"  MSE: {losses_noisy['mse'].item():.6f}")
print(f"  Total: {losses_noisy['total'].item():.6f}")

# De-standardize and evaluate
pred_noisy_db = pred_noisy_std * std_db + mean_db
pred_noisy_linear = np.power(10, pred_noisy_db / 10)

rmse_noisy = np.sqrt(np.mean((pred_noisy_linear - original_linear)**2))
ss_res_noisy = np.sum((original_linear - pred_noisy_linear)**2)
r2_noisy = 1 - ss_res_noisy / ss_tot

print(f"\nLinear domain evaluation:")
print(f"  RMSE: {rmse_noisy:.6e}")
print(f"  R2: {r2_noisy:.6f}")

# 9. Analyze relationship
print("\n" + "="*70)
print("ANALYSIS: Standardized loss vs Linear evaluation")
print("="*70)

print(f"\nStandardized domain:")
print(f"  Noise std: 0.1")
print(f"  MSE loss: {losses_noisy['mse'].item():.6f}")
print(f"  (theoretical value is 0.1^2 = 0.01)")

print(f"\ndB domain:")
print(f"  Noise std: 0.1 * {std_db:.2f} = {0.1*std_db:.2f} dB")

print(f"\nLinear domain:")
print(f"  RMSE: {rmse_noisy:.6e}")
print(f"  R2: {r2_noisy:.6f}")

# 10. Larger noise test
print("\n" + "="*70)
print("[10] Test with larger noise (std=0.5)")
print("="*70)

noise_large = np.random.randn(91, 91, 2) * 0.5
pred_large_std = target_std + noise_large
pred_large_tensor = torch.FloatTensor(pred_large_std).unsqueeze(0)

with torch.no_grad():
    losses_large = loss_fn(pred_large_tensor, target_tensor)

pred_large_db = pred_large_std * std_db + mean_db
pred_large_linear = np.power(10, pred_large_db / 10)

rmse_large = np.sqrt(np.mean((pred_large_linear - original_linear)**2))
r2_large = 1 - np.sum((original_linear - pred_large_linear)**2) / ss_tot

print(f"Standardized MSE: {losses_large['mse'].item():.6f}")
print(f"Linear RMSE: {rmse_large:.6e}")
print(f"Linear R2: {r2_large:.6f}")

print("\n" + "="*70)
print("SUMMARY:")
print(f"1. Std noise=0.1 -> MSE={losses_noisy['mse'].item():.4f}, R2={r2_noisy:.4f}")
print(f"2. Std noise=0.5 -> MSE={losses_large['mse'].item():.4f}, R2={r2_large:.4f}")
print("3. Loss computed in standardized domain")
print("4. Evaluation (RMSE, R2) computed in linear domain")
print("5. Small error in standardized domain can be amplified in linear domain")
print("="*70)