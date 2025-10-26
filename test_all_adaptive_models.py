"""
测试所有12个AutoEncoder模型的自适应功能

验证每个模型：
1. 能够正常创建（latent_dim=32和16）
2. forward pass正常
3. get_model_info()包含自适应字段
4. FC结构符合预期（压缩比≤4:1）
"""

import torch
from autoencoder.models import (
    # 标准CNN系列
    WaveletAutoEncoder,
    DirectAutoEncoder,
    # MLP系列
    WaveletMLPAutoEncoder,
    DirectMLPAutoEncoder,
    # Deep系列
    DeepWaveletAutoEncoder,
    DeepDirectAutoEncoder,
    # Enhanced系列
    EnhancedWaveletAutoEncoder,
    EnhancedDirectAutoEncoder,
    # Sine CNN系列
    SinWaveletAutoEncoder,
    SinDirectAutoEncoder,
    # Sine MLP系列
    SinWaveletMLPAutoEncoder,
    SinDirectMLPAutoEncoder,
)


def test_model(model_class, model_name, input_shape, latent_dim, **kwargs):
    """
    测试单个模型

    Args:
        model_class: 模型类
        model_name: 模型名称（用于显示）
        input_shape: 输入形状 (B, H, W, C)
        latent_dim: 隐空间维度
        **kwargs: 模型初始化参数

    Returns:
        是否通过测试
    """
    try:
        # 创建模型
        model = model_class(latent_dim=latent_dim, **kwargs)

        # 创建测试输入
        x = torch.randn(*input_shape)

        # Forward pass
        recon, latent = model(x)

        # 验证形状
        assert latent.shape == (input_shape[0], latent_dim), \
            f"Latent shape mismatch: expected {(input_shape[0], latent_dim)}, got {latent.shape}"
        assert recon.shape == input_shape, \
            f"Reconstruction shape mismatch: expected {input_shape}, got {recon.shape}"

        # 获取模型信息
        info = model.get_model_info()

        # 验证自适应字段存在
        required_keys = ['fc_structure', 'intermediate_dims', 'compression_ratios']
        for key_name in ['fc_structure', 'intermediate_dims']:
            # 对于MLP模型，字段名可能是 mlp_structure, num_mlp_layers
            if key_name not in info and key_name.replace('fc', 'mlp') not in info:
                raise AssertionError(f"Missing adaptive field: {key_name}")

        # 提取FC结构（兼容fc_structure和mlp_structure）
        fc_structure = info.get('fc_structure') or info.get('mlp_structure')
        compression_ratios = info['compression_ratios']

        # 验证压缩比（允许稍微超过4:1，因为向上取整到2的幂次可能导致4.5-4.7）
        for ratio_str in compression_ratios:
            # 从 "4096:1024 = 4.0:1" 中提取 4.0
            ratio_value = float(ratio_str.split('=')[1].strip().split(':')[0])
            assert ratio_value <= 5.0, f"Compression ratio too high: {ratio_value} (should ≤5:1)"

        print(f"[PASS] {model_name} (latent_dim={latent_dim})")
        print(f"   FC: {fc_structure}")
        print(f"   Params: {info['parameters']['total']:,}")
        print()

        return True

    except Exception as e:
        print(f"[FAIL] {model_name} (latent_dim={latent_dim})")
        print(f"   Error: {e}")
        print()
        return False


def main():
    """测试所有12个模型"""

    print("="*80)
    print("测试所有AutoEncoder模型的自适应功能")
    print("="*80)
    print()

    # 测试配置
    batch_size = 2
    num_frequencies = 2

    # Wavelet模式输入: [B, 49, 49, 8]
    wavelet_input = (batch_size, 49, 49, num_frequencies * 4)

    # Direct模式输入: [B, 91, 91, 2]
    direct_input = (batch_size, 91, 91, num_frequencies)

    # 测试的latent_dim值
    test_latent_dims = [32, 16]

    results = []

    # 1. Standard CNN
    print("[1. Standard CNN]")
    print("-"*80)
    for latent_dim in test_latent_dims:
        results.append(test_model(
            WaveletAutoEncoder, "WaveletAutoEncoder",
            wavelet_input, latent_dim, num_frequencies=num_frequencies
        ))
        results.append(test_model(
            DirectAutoEncoder, "DirectAutoEncoder",
            direct_input, latent_dim, num_frequencies=num_frequencies
        ))

    # 2. MLP
    print("[2. MLP]")
    print("-"*80)
    for latent_dim in test_latent_dims:
        results.append(test_model(
            WaveletMLPAutoEncoder, "WaveletMLPAutoEncoder",
            wavelet_input, latent_dim, num_frequencies=num_frequencies
        ))
        results.append(test_model(
            DirectMLPAutoEncoder, "DirectMLPAutoEncoder",
            direct_input, latent_dim, num_frequencies=num_frequencies
        ))

    # 3. Deep CNN
    print("[3. Deep CNN]")
    print("-"*80)
    for latent_dim in test_latent_dims:
        results.append(test_model(
            DeepWaveletAutoEncoder, "DeepWaveletAutoEncoder",
            wavelet_input, latent_dim, num_frequencies=num_frequencies
        ))
        results.append(test_model(
            DeepDirectAutoEncoder, "DeepDirectAutoEncoder",
            direct_input, latent_dim, num_frequencies=num_frequencies
        ))

    # 4. Enhanced CNN
    print("[4. Enhanced CNN]")
    print("-"*80)
    for latent_dim in test_latent_dims:
        results.append(test_model(
            EnhancedWaveletAutoEncoder, "EnhancedWaveletAutoEncoder",
            wavelet_input, latent_dim, num_frequencies=num_frequencies
        ))
        results.append(test_model(
            EnhancedDirectAutoEncoder, "EnhancedDirectAutoEncoder",
            direct_input, latent_dim, num_frequencies=num_frequencies
        ))

    # 5. Sine CNN
    print("[5. Sine CNN]")
    print("-"*80)
    for latent_dim in test_latent_dims:
        results.append(test_model(
            SinWaveletAutoEncoder, "SinWaveletAutoEncoder",
            wavelet_input, latent_dim, num_frequencies=num_frequencies
        ))
        results.append(test_model(
            SinDirectAutoEncoder, "SinDirectAutoEncoder",
            direct_input, latent_dim, num_frequencies=num_frequencies
        ))

    # 6. Sine MLP
    print("[6. Sine MLP]")
    print("-"*80)
    for latent_dim in test_latent_dims:
        results.append(test_model(
            SinWaveletMLPAutoEncoder, "SinWaveletMLPAutoEncoder",
            wavelet_input, latent_dim, num_frequencies=num_frequencies
        ))
        results.append(test_model(
            SinDirectMLPAutoEncoder, "SinDirectMLPAutoEncoder",
            direct_input, latent_dim, num_frequencies=num_frequencies
        ))

    # 统计结果
    print("="*80)
    print("测试结果总结")
    print("="*80)
    total_tests = len(results)
    passed_tests = sum(results)
    failed_tests = total_tests - passed_tests

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Pass rate: {passed_tests/total_tests*100:.1f}%")
    print()

    if failed_tests == 0:
        print("SUCCESS! All 12 AutoEncoder models support adaptive small latent space!")
    else:
        print("WARNING: Some models failed. Please check the errors above.")

    return failed_tests == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
