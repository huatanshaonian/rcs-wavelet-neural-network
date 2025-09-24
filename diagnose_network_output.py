"""
诊断网络输出范围问题
检查网络是否输出了常数或异常值
"""
import torch
import numpy as np
from wavelet_network import create_model
from training import RCSDataModule
import os

def diagnose_network_output(model_path):
    """诊断网络输出"""

    print("="*60)
    print("网络输出诊断")
    print("="*60)

    # 加载数据
    data_config = {
        'params_file': '../parameter/parameters_sorted.csv',
        'rcs_data_dir': '../parameter/csv_output',
        'model_ids': [f"{i:03d}" for i in range(1, 101)],
        'frequencies': ['1.5G', '3G'],
        'use_log_preprocessing': True
    }

    data_module = RCSDataModule(
        data_config=data_config,
        use_log_preprocessing=True,
        normalize_after_log=True
    )

    params, rcs_data = data_module.load_data()
    print(f"\n数据加载完成:")
    print(f"  参数形状: {params.shape}")
    print(f"  RCS形状: {rcs_data.shape}")
    print(f"  预处理统计: mean={data_module.preprocessing_stats['mean']:.2f} dB, std={data_module.preprocessing_stats['std']:.2f} dB")

    # 加载模型
    if not os.path.exists(model_path):
        print(f"\n错误: 模型文件不存在: {model_path}")
        return

    model = create_model(input_dim=9, use_log_output=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n模型加载完成: {model_path}")

    # 测试几个样本
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    test_indices = [0, 10, 50, 90]

    with torch.no_grad():
        for idx in test_indices:
            param_tensor = torch.FloatTensor(params[idx:idx+1]).to(device)
            target = rcs_data[idx]

            output = model(param_tensor).cpu().numpy().squeeze()

            print(f"\n样本 {idx+1}:")
            print(f"  网络输出 (标准化):")
            print(f"    范围: {output.min():.6f} ~ {output.max():.6f}")
            print(f"    均值: {output.mean():.6f}, 标准差: {output.std():.6f}")
            print(f"    唯一值数量: {len(np.unique(output))}")

            # 反标准化到dB
            mean = data_module.preprocessing_stats['mean']
            std = data_module.preprocessing_stats['std']
            output_db = output * std + mean

            print(f"  反标准化到dB:")
            print(f"    范围: {output_db.min():.2f} ~ {output_db.max():.2f} dB")
            print(f"    均值: {output_db.mean():.2f} dB")

            # 检查是否全是同一个值
            if output.std() < 1e-6:
                print(f"  ⚠️ 警告: 输出几乎是常数！")

            # 目标值统计
            print(f"  目标值 (标准化):")
            print(f"    范围: {target.min():.6f} ~ {target.max():.6f}")
            print(f"    均值: {target.mean():.6f}, 标准差: {target.std():.6f}")

            # MSE
            mse = np.mean((output - target) ** 2)
            print(f"  MSE (标准化域): {mse:.6f}")

    # 检查所有样本的输出分布
    print("\n" + "="*60)
    print("全数据集输出分布检查")
    print("="*60)

    all_outputs = []
    with torch.no_grad():
        for i in range(0, len(params), 10):
            param_tensor = torch.FloatTensor(params[i:i+1]).to(device)
            output = model(param_tensor).cpu().numpy()
            all_outputs.append(output)

    all_outputs = np.concatenate(all_outputs, axis=0)

    print(f"\n采样输出统计 (标准化域):")
    print(f"  形状: {all_outputs.shape}")
    print(f"  范围: {all_outputs.min():.6f} ~ {all_outputs.max():.6f}")
    print(f"  均值: {all_outputs.mean():.6f}")
    print(f"  标准差: {all_outputs.std():.6f}")

    # 反标准化
    all_outputs_db = all_outputs * std + mean
    print(f"\n采样输出统计 (dB):")
    print(f"  范围: {all_outputs_db.min():.2f} ~ {all_outputs_db.max():.2f} dB")
    print(f"  均值: {all_outputs_db.mean():.2f} dB")

    # 检查是否塌缩到某个值
    if all_outputs.std() < 0.1:
        print(f"\n❌ 严重问题: 网络输出塌缩到几乎常数!")
        print(f"   所有输出都在 {all_outputs.mean():.4f} ± {all_outputs.std():.4f} 范围内")

    print("\n" + "="*60)

if __name__ == "__main__":
    # 查找最新的模型
    import glob
    model_files = glob.glob("models/best_model_fold*.pth")

    if model_files:
        model_path = model_files[0]
        print(f"使用模型: {model_path}\n")
        diagnose_network_output(model_path)
    else:
        print("未找到模型文件，请先训练模型")