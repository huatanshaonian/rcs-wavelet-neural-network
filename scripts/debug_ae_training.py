"""
测试MLP架构集成 - 验证4种组合的创建和前向传播
"""
import torch
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'autoencoder'))

from autoencoder.utils.frequency_config import create_autoencoder_system

def test_architecture_combination(mode, architecture, config_name='2freq'):
    """测试特定的架构组合"""
    print(f"\n{'='*60}")
    print(f"测试组合: {mode.upper()} + {architecture.upper()} ({config_name})")
    print(f"{'='*60}")

    try:
        # 创建系统
        system = create_autoencoder_system(
            config_name=config_name,
            latent_dim=32,
            dropout_rate=0.2,
            wavelet='db4',
            normalize=True,
            mode=mode,
            architecture=architecture
        )

        autoencoder = system['autoencoder']
        wavelet_transform = system['wavelet_transform']

        # 打印模型信息
        param_count = autoencoder.get_parameter_count()
        print(f"\n模型参数量: {param_count['total']:,}")

        # 创建测试数据
        batch_size = 2
        num_freq = 2 if config_name == '2freq' else 3
        rcs_data = torch.randn(batch_size, 91, 91, num_freq)

        # 测试前向传播
        if mode == 'wavelet':
            # 小波模式: RCS → 小波系数 → AutoEncoder → 重建
            wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
            print(f"小波系数形状: {wavelet_coeffs.shape}")

            recon_coeffs, latent = autoencoder(wavelet_coeffs)
            print(f"隐空间形状: {latent.shape}")
            print(f"重建小波系数形状: {recon_coeffs.shape}")

            recon_rcs = wavelet_transform.inverse_transform(recon_coeffs)
            print(f"重建RCS形状: {recon_rcs.shape}")

            # 计算重建误差
            mse = torch.nn.functional.mse_loss(recon_rcs, rcs_data)
            print(f"重建MSE: {mse.item():.6f}")

        else:
            # 直接模式: RCS → AutoEncoder → 重建
            recon_rcs, latent = autoencoder(rcs_data)
            print(f"隐空间形状: {latent.shape}")
            print(f"重建RCS形状: {recon_rcs.shape}")

            # 计算重建误差
            mse = torch.nn.functional.mse_loss(recon_rcs, rcs_data)
            print(f"重建MSE: {mse.item():.6f}")

        # 测试编码器和解码器单独调用
        if mode == 'wavelet':
            input_data = wavelet_coeffs
        else:
            input_data = rcs_data

        latent_only = autoencoder.encode(input_data)
        recon_only = autoencoder.decode(latent_only)
        print(f"\n编码器输出形状: {latent_only.shape}")
        print(f"解码器输出形状: {recon_only.shape}")

        print(f"\n✅ {mode.upper()} + {architecture.upper()} 测试通过!")
        return True

    except Exception as e:
        print(f"\n❌ {mode.upper()} + {architecture.upper()} 测试失败!")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """测试所有4种组合"""
    print("="*60)
    print("MLP架构集成测试 - 测试所有架构组合")
    print("="*60)

    # 定义测试组合
    test_cases = [
        ('wavelet', 'cnn'),   # 小波 + CNN
        ('wavelet', 'mlp'),   # 小波 + MLP
        ('direct', 'cnn'),    # 直接 + CNN
        ('direct', 'mlp'),    # 直接 + MLP
    ]

    results = {}

    # 测试2频率配置
    print("\n\n" + "="*60)
    print("第一阶段: 2频率配置测试")
    print("="*60)
    for mode, arch in test_cases:
        key = f"{mode}+{arch} (2freq)"
        results[key] = test_architecture_combination(mode, arch, '2freq')

    # 测试3频率配置
    print("\n\n" + "="*60)
    print("第二阶段: 3频率配置测试")
    print("="*60)
    for mode, arch in test_cases:
        key = f"{mode}+{arch} (3freq)"
        results[key] = test_architecture_combination(mode, arch, '3freq')

    # 汇总结果
    print("\n\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for key, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{key:30s} : {status}")

    # 统计
    total = len(results)
    passed = sum(results.values())
    print(f"\n总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")

    if passed == total:
        print("\n🎉 所有测试通过! MLP架构集成成功!")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
