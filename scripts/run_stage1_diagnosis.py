"""
Stage 1 重建诊断 - 独立运行脚本

训练完Stage 1后，直接运行此脚本进行诊断

使用方法：
    python run_stage1_diagnosis.py [model_path] [data_path]

参数：
    model_path: 模型文件路径（可选，默认: ae_checkpoints/best_autoencoder.pth）
    data_path: 数据文件路径（可选，默认: data/rcs_data.npy）

示例：
    python run_stage1_diagnosis.py
    python run_stage1_diagnosis.py ae_checkpoints/my_model.pth
    python run_stage1_diagnosis.py ae_checkpoints/my_model.pth data/my_data.npy
"""

import torch
import numpy as np
import sys
import os
import argparse

# 确保路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagnose_stage1_reconstruction import diagnose_reconstruction_data


def main(model_path=None, data_path=None):
    print("="*80)
    print("Stage 1 AutoEncoder 重建诊断工具")
    print("="*80)
    print()

    # ========== 1. 加载已保存的模型 ==========
    print("[Step 1] Loading saved AutoEncoder model...")

    # 使用默认路径或命令行参数
    if not model_path:
        model_path = "ae_checkpoints/best_autoencoder.pth"

    print(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("\nAvailable model files:")
        if os.path.exists("ae_checkpoints"):
            for f in os.listdir("ae_checkpoints"):
                if f.endswith('.pth'):
                    print(f"  - ae_checkpoints/{f}")
        else:
            print("  (ae_checkpoints directory not found)")
        return

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"[OK] Model loaded: {model_path}")

    # 提取模型配置信息（从'config'字段）
    config = checkpoint.get('config', {})
    mode = config.get('mode', checkpoint.get('mode', 'wavelet'))
    architecture = config.get('architecture', checkpoint.get('architecture', 'cnn'))
    latent_dim = config.get('latent_dim', checkpoint.get('latent_dim', 256))
    num_frequencies = config.get('num_frequencies', checkpoint.get('num_frequencies', 2))
    training_mode = checkpoint.get('training_mode', 'three_stage')

    print(f"\nModel configuration:")
    print(f"  Mode: {mode}")
    print(f"  Architecture: {architecture}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Num frequencies: {num_frequencies}")
    print(f"  Training mode: {training_mode}")

    # 检查频率配置
    if num_frequencies != 2:
        print(f"\n[ERROR] This diagnosis script only supports 2-frequency models")
        print(f"Current model has {num_frequencies} frequencies")
        print("Please use a 2freq model for diagnosis")
        return

    if training_mode != 'stage1_only':
        print(f"\n[WARNING] Training mode is '{training_mode}', but diagnosis tool is designed for 'stage1_only'")
        print("Continuing anyway...")

    # ========== 2. 重建AutoEncoder系统 ==========
    print("\n[Step 2] Rebuilding AutoEncoder system...")

    from autoencoder.utils.frequency_config import create_autoencoder_system

    ae_system = create_autoencoder_system(
        config_name='2freq' if num_frequencies == 2 else '3freq',
        latent_dim=latent_dim,
        mode=mode,
        architecture=architecture
    )

    # 加载模型权重（checkpoint中的key是'autoencoder'而不是'autoencoder_state_dict'）
    ae_system['autoencoder'].load_state_dict(checkpoint['autoencoder'])
    ae_system['autoencoder'].eval()

    print("[OK] AutoEncoder system rebuilt")

    # ========== 3. 加载测试数据 ==========
    print("\n[Step 3] Loading RCS test data...")

    rcs_data = None

    # 方法1：从命令行指定的路径加载
    if data_path and os.path.exists(data_path):
        print(f"Loading from specified path: {data_path}")
        rcs_data = np.load(data_path)
        print(f"[OK] Data loaded: {rcs_data.shape}")

    # 方法2：从默认npy路径加载
    elif not data_path and os.path.exists("data/rcs_data.npy"):
        print("Loading from default path: data/rcs_data.npy")
        rcs_data = np.load("data/rcs_data.npy")
        print(f"[OK] Data loaded: {rcs_data.shape}")

    # 方法3：从缓存加载（优先）
    else:
        print("No .npy file specified, searching for cache files...")
        cache_dir = "cache"

        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir)
                          if f.startswith('rcs_data_') and f.endswith('.pkl')]

            if cache_files:
                print(f"Found {len(cache_files)} cache file(s)")
                print("Searching for 2-frequency cache...")

                # 查找所有2freq缓存文件
                import pickle
                matching_caches = []

                for cache_file in cache_files:
                    cache_path = os.path.join(cache_dir, cache_file)
                    try:
                        with open(cache_path, 'rb') as f:
                            cache_data = pickle.load(f)

                        data_shape = cache_data['rcs_data'].shape
                        data_num_freq = data_shape[-1]

                        print(f"  {cache_file}: {data_shape} ({data_num_freq} freq)")

                        if data_num_freq == 2:
                            matching_caches.append((cache_file, cache_path, cache_data))
                    except Exception as e:
                        print(f"  {cache_file}: [ERROR] {e}")

                if matching_caches:
                    # 如果有多个2freq缓存，选择最新的
                    matching_caches.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
                    selected_file, selected_path, cache_data = matching_caches[0]

                    rcs_data = cache_data['rcs_data']
                    cache_info = cache_data['info']

                    print(f"\n[OK] Found matching 2-frequency cache: {selected_file}")
                    print(f"  Data shape: {rcs_data.shape}")
                    print(f"  Created at: {cache_info['created_at']}")
                    print(f"  Cache file: {selected_path}")
                else:
                    print("\n[ERROR] No 2-frequency cache found!")
                    rcs_data = None
            else:
                print("[WARNING] No cache files found in 'cache/' directory")
        else:
            print("[WARNING] Cache directory 'cache/' not found")

    # 如果所有方法都失败
    if rcs_data is None:
        print("\n[ERROR] Could not load RCS data!")
        print("\nPlease do one of the following:")
        print("  1. Load data in GUI first (it will create cache files)")
        print("  2. Specify data file path: python run_stage1_diagnosis.py model.pth data.npy")
        print("  3. Save RCS data as data/rcs_data.npy")
        return

    # 验证数据频率配置
    data_num_freq = rcs_data.shape[-1]
    if data_num_freq != 2:
        print(f"\n[ERROR] Data frequency mismatch!")
        print(f"  Model expects: 2 frequencies")
        print(f"  Data has: {data_num_freq} frequencies")
        print(f"  Data shape: {rcs_data.shape}")
        print("\nThis diagnosis script only works with 2-frequency data")
        print("Please load 2freq data in GUI and try again")
        return

    # 使用后20个样本
    test_size = min(20, len(rcs_data))
    test_rcs = rcs_data[-test_size:]
    print(f"Using {test_size} test samples (2 frequencies)")

    # ========== 4. 执行重建 ==========
    print("\n[Step 4] Performing RCS reconstruction...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    autoencoder = ae_system['autoencoder'].to(device)
    wavelet_transform = ae_system.get('wavelet_transform')
    data_adapter = ae_system.get('data_adapter')

    with torch.no_grad():
        # 预处理
        if data_adapter and mode == 'wavelet':
            # 小波模式：先小波变换再标准化
            rcs_tensor = torch.FloatTensor(test_rcs).to(device)
            wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
            original_coeffs_np = wavelet_coeffs.cpu().numpy()

            adapted_input = data_adapter.adapt_rcs_data(original_coeffs_np)
            adapted_input = torch.FloatTensor(adapted_input).to(device)
        elif data_adapter and mode == 'direct':
            # Direct模式：直接标准化
            adapted_input = data_adapter.adapt_rcs_data(test_rcs)
            adapted_input = torch.FloatTensor(adapted_input).to(device)
            original_coeffs_np = None
        else:
            # 无标准化
            if mode == 'wavelet':
                rcs_tensor = torch.FloatTensor(test_rcs).to(device)
                adapted_input = wavelet_transform.forward_transform(rcs_tensor)
                original_coeffs_np = adapted_input.cpu().numpy()
            else:
                adapted_input = torch.FloatTensor(test_rcs).to(device)
                original_coeffs_np = None

        # Encoder → Decoder
        reconstructed_output, _ = autoencoder(adapted_input)

        # 逆预处理
        if mode == 'wavelet':
            # 逆标准化小波系数
            if data_adapter:
                # inverse_adapt期望tensor输入，返回numpy数组
                reconstructed_coeffs_np = data_adapter.inverse_adapt(reconstructed_output)
                reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
            else:
                reconstructed_coeffs = reconstructed_output
                reconstructed_coeffs_np = reconstructed_coeffs.cpu().numpy()

            # 逆小波变换
            reconstructed_rcs_tensor = wavelet_transform.inverse_transform(reconstructed_coeffs)
            reconstructed_rcs = reconstructed_rcs_tensor.cpu().numpy()
        else:
            # Direct模式：逆标准化
            if data_adapter:
                # inverse_adapt期望tensor输入，返回numpy数组
                reconstructed_rcs = data_adapter.inverse_adapt(reconstructed_output)
            else:
                reconstructed_rcs = reconstructed_output.cpu().numpy()
            reconstructed_coeffs_np = None

    print("[OK] Reconstruction completed")

    # ========== 5. 运行诊断 ==========
    print("\n[Step 5] Running diagnostic analysis...")
    print("="*80)
    print()

    diagnose_reconstruction_data(
        original_rcs=test_rcs,
        reconstructed_rcs=reconstructed_rcs,
        original_coeffs=original_coeffs_np,
        reconstructed_coeffs=reconstructed_coeffs_np
    )

    print("\n[DONE] Diagnosis completed! Please check:")
    print("  1. Diagnostic output above")
    print("  2. Generated comparison plot: stage1_reconstruction_diagnosis.png")
    print()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stage 1 AutoEncoder Reconstruction Diagnosis')
    parser.add_argument('model_path', nargs='?', default=None,
                       help='Path to model file (default: ae_checkpoints/best_autoencoder.pth)')
    parser.add_argument('data_path', nargs='?', default=None,
                       help='Path to data file (default: data/rcs_data.npy)')

    args = parser.parse_args()

    try:
        main(model_path=args.model_path, data_path=args.data_path)
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
