import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from autoencoder.utils.frequency_config import create_autoencoder_system
from autoencoder.utils.activation_factory import SinActivation

def check_activation(module_seq):
    activations = []
    for layer in module_seq:
        if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.SiLU, nn.GELU, SinActivation)):
            activations.append(layer.__class__.__name__)
    return activations

print("=== 验证模型激活函数配置 ===")

# 创建系统: High=SIN, Smooth=RELU
print("\n1. 创建系统 (High=SIN, Smooth=RELU)...")
system = create_autoencoder_system(
    config_name='2freq',
    mode='wavelet',
    architecture='additive_dual_branch_cnn',
    activation_high='sin',
    activation_smooth='relu',
    latent_dim=32
)

ae = system['autoencoder']

# 检查 High 分支 (应该全是 SinActivation)
print("\n[检查 High 分支 (预期: SIN)]")
high_fc_acts = check_activation(ae.decoder_high_fc)
high_conv_acts = check_activation(ae.decoder_high_conv)
print(f"  FC层激活: {high_fc_acts}")
print(f"  Conv层激活: {high_conv_acts}")

is_high_correct = all('Sin' in name for name in high_fc_acts + high_conv_acts)
print(f"  -> 结果: {'✅ 正确 (全是SIN)' if is_high_correct else '❌ 错误'}")

# 检查 Smooth 分支 (应该全是 ReLU)
print("\n[检查 Smooth 分支 (预期: RELU)]")
smooth_fc_acts = check_activation(ae.decoder_smooth_fc)
smooth_conv_acts = check_activation(ae.decoder_smooth_conv)
print(f"  FC层激活: {smooth_fc_acts}")
print(f"  Conv层激活: {smooth_conv_acts}")

is_smooth_correct = all('ReLU' in name for name in smooth_fc_acts + smooth_conv_acts)
print(f"  -> 结果: {'✅ 正确 (全是ReLU)' if is_smooth_correct else '❌ 错误'}")

# 验证前向传播输出
print("\n[验证前向传播输出]")
dummy_input = torch.randn(1, 49, 49, 8) # 2 freq * 4 bands
recon, latent, recon_high, recon_smooth = ae.forward_with_branches(dummy_input)

print(f"  recon_high shape: {recon_high.shape}")
print(f"  recon_smooth shape: {recon_smooth.shape}")
print("  -> forward_with_branches 调用成功")
