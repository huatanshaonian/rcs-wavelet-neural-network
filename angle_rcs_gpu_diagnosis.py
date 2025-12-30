"""
Angle-RCS GPU训练瓶颈诊断脚本
"""
import torch
import time
import numpy as np
from angle_based_rcs.data.angle_dataset import create_dataloaders
from angle_based_rcs.models.angle_rcs_network import AngleRCSNetwork

print("="*60)
print("Angle-RCS GPU瓶颈诊断")
print("="*60)

if not torch.cuda.is_available():
    print("❌ CUDA不可用！")
    exit(1)

device = torch.device('cuda')
print(f"✅ 使用设备: {device}")

# 创建模拟数据
print("\n[1] 创建模拟数据...")
num_samples = 200
rcs_data = np.random.rand(num_samples, 91, 91, 3) * 0.5
param_data = np.random.randn(num_samples, 9)
print(f"  RCS数据: {rcs_data.shape}")
print(f"  参数数据: {param_data.shape}")

# 测试不同batch size
for batch_size in [256, 512, 1024, 2048]:
    print(f"\n{'='*60}")
    print(f"[测试] Batch Size = {batch_size}")
    print(f"{'='*60}")
    
    # 创建DataLoader（启用GPU预加载）
    print("\n[2] 创建DataLoader（GPU预加载模式）...")
    train_loader, val_loader, sampler = create_dataloaders(
        rcs_data=rcs_data,
        param_data=param_data,
        batch_size=batch_size,
        num_frequencies=3,
        train_split=0.8,
        random_seed=42,
        train_subset_size=10000,  # 小数据集测试
        normalize_params=True,
        num_workers=0,
        preload_to_gpu=True
    )
    
    print(f"  训练集: {len(train_loader)} batches")
    
    # 检查数据是否在GPU
    print("\n[3] 检查数据位置...")
    first_batch = next(iter(train_loader))
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:12s}: device={value.device}, shape={value.shape}")
    
    gpu_data = first_batch['theta'].device.type == 'cuda'
    if gpu_data:
        print("  ✅ 数据在GPU上")
    else:
        print("  ❌ 数据在CPU上！")
    
    # 创建模型
    print("\n[4] 创建模型...")
    model = AngleRCSNetwork(num_frequencies=3, angle_L=16, activation='sin').to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 测试训练速度
    print("\n[5] 测试训练速度（10个batch）...")
    model.train()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        
        # 检查是否需要.to(device)
        if batch['theta'].device != device:
            print(f"  ⚠️ Batch {i}: 数据在CPU，需要移动到GPU")
            theta = batch['theta'].to(device)
            phi = batch['phi'].to(device)
            params = batch['params'].to(device)
            freq_idx = batch['freq_idx'].to(device)
            target_rcs = batch['target_rcs'].to(device)
        else:
            theta = batch['theta']
            phi = batch['phi']
            params = batch['params']
            freq_idx = batch['freq_idx']
            target_rcs = batch['target_rcs']
        
        optimizer.zero_grad()
        pred = model(theta, phi, params, freq_idx).squeeze()
        loss = criterion(pred, target_rcs)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    batches_per_sec = 10 / elapsed
    throughput = batches_per_sec * batch_size
    
    print(f"\n  结果:")
    print(f"    10个batch耗时: {elapsed:.3f}秒")
    print(f"    吞吐量: {batches_per_sec:.1f} batches/秒")
    print(f"    样本吞吐: {throughput:.0f} samples/秒")
    print(f"    每batch平均: {elapsed/10*1000:.1f}ms")
    
    # 估算GPU利用率
    # 假设理想情况下batch_size=256需要3ms
    ideal_time = 0.003 * (batch_size / 256)
    actual_time = elapsed / 10
    efficiency = ideal_time / actual_time * 100
    
    print(f"\n  性能分析:")
    print(f"    理想时间/batch: {ideal_time*1000:.1f}ms")
    print(f"    实际时间/batch: {actual_time*1000:.1f}ms")
    print(f"    估算效率: {efficiency:.1f}%")
    
    if efficiency < 30:
        print(f"    ❌ 效率极低！存在严重瓶颈")
    elif efficiency < 60:
        print(f"    ⚠️ 效率偏低，有优化空间")
    else:
        print(f"    ✅ 效率正常")

print("\n" + "="*60)
print("诊断完成")
print("="*60)
