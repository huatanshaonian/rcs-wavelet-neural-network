"""
诊断网络输出范围问题
"""
import torch
import numpy as np
from training import RCSDataLoader
from wavelet_network import create_model
import glob

print("="*60)
print("诊断网络输出范围")
print("="*60)

# 1. 加载数据查看训练目标范围
print("\n1. 加载训练数据")
data_config = {
    'params_file': '../parameter/parameters_sorted.csv',
    'rcs_data_dir': '../parameter/csv_output',
    'model_ids': [f"{i:03d}" for i in range(1, 101)],
    'frequencies': ['1.5G', '3G'],
    'preprocessing': {
        'use_log': True,
        'log_epsilon': 1e-10,
        'normalize_after_log': True
    }
}

data_loader = RCSDataLoader(data_config)
params, rcs_data = data_loader.load_data()
stats = data_loader.preprocessing_stats

print(f"训练目标（标准化RCS）:")
print(f"  范围: {rcs_data.min():.3f} ~ {rcs_data.max():.3f}")
print(f"  均值: {rcs_data.mean():.3f}")
print(f"  std统计: mean={stats['mean']:.2f} dB, std={stats['std']:.2f} dB")

# 2. 查找checkpoint
checkpoint_files = glob.glob("checkpoints/best_model_fold_*.pth")
if not checkpoint_files:
    checkpoint_files = glob.glob("models/*.pth")

if not checkpoint_files:
    print("\nERROR: 未找到checkpoint文件")
    exit(1)

checkpoint_path = checkpoint_files[0]
print(f"\n2. 加载checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 3. 检查checkpoint内容
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    print("OK 新格式checkpoint")
    saved_stats = checkpoint.get('preprocessing_stats')
    if saved_stats:
        print(f"  保存的stats: mean={saved_stats['mean']:.2f}, std={saved_stats['std']:.2f}")
    else:
        print("  WARNING: 无preprocessing_stats")
        saved_stats = stats  # 使用当前计算的

    use_log = checkpoint.get('use_log_output', True)
    print(f"  use_log_output: {use_log}")

    # 4. 创建模型
    print("\n3. 创建模型并加载权重")
    model = create_model(input_dim=9, use_log_output=use_log)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. 测试几个样本
    print("\n4. 测试网络输出")
    device = 'cpu'
    model.to(device)

    test_indices = [0, 25, 50, 75, 99]

    with torch.no_grad():
        for idx in test_indices:
            param_tensor = torch.FloatTensor(params[idx:idx+1]).to(device)
            target = rcs_data[idx]
            output = model(param_tensor).cpu().numpy().squeeze()

            print(f"\n  样本 {idx+1}:")
            print(f"    网络输出: 范围 {output.min():.3f} ~ {output.max():.3f}")
            print(f"    目标值:   范围 {target.min():.3f} ~ {target.max():.3f}")

            # 反标准化
            output_db = output * saved_stats['std'] + saved_stats['mean']
            print(f"    反标准化: 范围 {output_db.min():.2f} ~ {output_db.max():.2f} dB")

            # 检查异常
            if output.min() < -10 or output.max() > 10:
                print(f"    WARNING: 网络输出超出预期范围！")

                # 检查最后一层权重
                final_conv_weight = None
                for name, param in model.named_parameters():
                    if 'final_conv' in name and 'weight' in name:
                        final_conv_weight = param.data.numpy()
                        print(f"    final_conv权重: 范围 {final_conv_weight.min():.6f} ~ {final_conv_weight.max():.6f}")
                        print(f"    final_conv权重均值: {final_conv_weight.mean():.6f}")

    # 6. 检查网络结构
    print("\n5. 检查网络配置")
    print(f"  use_log_output: {model.use_log_output}")
    print(f"  输出激活函数: {model.freq1_decoder.output_activation}")

else:
    print("WARNING: 旧格式checkpoint")

print("\n" + "="*60)