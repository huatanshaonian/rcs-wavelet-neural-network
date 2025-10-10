"""
调试AutoEncoder训练的脚本
验证小波变换时间和loss计算准确性
"""

import torch
import torch.nn as nn
import time
import numpy as np
from autoencoder.utils.frequency_config import create_autoencoder_system

def debug_ae_training():
    """调试AutoEncoder训练过程"""
    print("=== AutoEncoder训练调试 ===")

    # 创建测试数据
    batch_size = 16
    rcs_data = torch.randn(100, 91, 91, 2) * 0.1  # 模拟真实RCS数据范围
    print(f"模拟RCS数据: {rcs_data.shape}, 范围: [{rcs_data.min():.4f}, {rcs_data.max():.4f}]")

    # 创建AutoEncoder系统
    system = create_autoencoder_system('2freq', latent_dim=256)
    autoencoder = system['autoencoder']
    wavelet_transform = system['wavelet_transform']

    # 测试小波变换时间
    print("\n=== 小波变换性能测试 ===")
    start_time = time.time()
    wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
    transform_time = time.time() - start_time

    print(f"小波变换时间: {transform_time:.3f}s")
    print(f"变换后形状: {wavelet_coeffs.shape}")
    print(f"小波系数范围: [{wavelet_coeffs.min():.4f}, {wavelet_coeffs.max():.4f}]")

    # 测试AutoEncoder重建
    print("\n=== AutoEncoder重建测试 ===")
    autoencoder.eval()
    with torch.no_grad():
        reconstructed, latent = autoencoder(wavelet_coeffs[:16])  # 取前16个样本
        print(f"隐空间形状: {latent.shape}")
        print(f"重建系数形状: {reconstructed.shape}")

        # 计算重建损失
        mse_loss = nn.MSELoss()
        recon_loss = mse_loss(reconstructed, wavelet_coeffs[:16])
        print(f"重建MSE损失: {recon_loss.item():.6f}")

        # 测试逆变换
        reconstructed_rcs = wavelet_transform.inverse_transform(reconstructed)
        print(f"重建RCS形状: {reconstructed_rcs.shape}")

        # 端到端损失
        end_to_end_loss = mse_loss(reconstructed_rcs, rcs_data[:16])
        print(f"端到端MSE损失: {end_to_end_loss.item():.6f}")

    # 模拟训练过程
    print("\n=== 模拟训练过程 ===")
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(wavelet_coeffs)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0
    num_batches = 0

    for batch_idx, (batch_coeffs,) in enumerate(train_loader):
        if batch_idx >= 5:  # 只测试前5个批次
            break

        start_time = time.time()

        # 前向传播
        reconstructed, latent = autoencoder(batch_coeffs)
        loss = criterion(reconstructed, batch_coeffs)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        total_loss += loss.item()
        num_batches += 1

        print(f"批次 {batch_idx+1}: Loss={loss.item():.6f}, 时间={batch_time:.3f}s")

    avg_loss = total_loss / num_batches
    print(f"平均训练损失: {avg_loss:.6f}")

    # 验证数据流程正确性
    print("\n=== 数据流程验证 ===")
    test_batch = wavelet_coeffs[:4]
    print(f"输入小波系数: {test_batch.shape}, 范围: [{test_batch.min():.4f}, {test_batch.max():.4f}]")

    with torch.no_grad():
        recon_coeffs, _ = autoencoder(test_batch)
        print(f"重建小波系数: {recon_coeffs.shape}, 范围: [{recon_coeffs.min():.4f}, {recon_coeffs.max():.4f}]")

        # 检查数值稳定性
        diff = torch.abs(recon_coeffs - test_batch)
        print(f"重建误差统计: 均值={diff.mean():.6f}, 最大={diff.max():.6f}")

if __name__ == "__main__":
    debug_ae_training()