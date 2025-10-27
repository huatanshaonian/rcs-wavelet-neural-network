"""
测试differentiable_wavelet模式的三种网络

验证：
1. 系统能否成功创建
2. 前向传播是否正常
3. 梯度是否可以回传
4. 输入/输出尺寸是否正确
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autoencoder.utils.frequency_config import create_autoencoder_system


def test_differentiable_mode(architecture, latent_dim=32):
    """测试单个架构"""
    print("\n" + "="*80)
    print(f"测试 differentiable_wavelet + {architecture.upper()}")
    print("="*80)

    try:
        # 创建系统
        system = create_autoencoder_system(
            config_name='2freq',
            latent_dim=latent_dim,
            mode='differentiable_wavelet',
            architecture=architecture,
            wavelet='db4'
        )

        autoencoder = system['autoencoder']
        print(f"\n[OK] 系统创建成功")
        print(f"模型类型: {type(autoencoder).__name__}")

        # 创建测试数据（RCS格式，不是小波系数！）
        batch_size = 4
        rcs_data = torch.randn(batch_size, 91, 91, 2, requires_grad=True)
        print(f"\n测试数据形状: {rcs_data.shape}")

        # 前向传播
        print("\n[测试前向传播...]")
        reconstructed_rcs, latent = autoencoder(rcs_data)
        print(f"输入RCS形状: {rcs_data.shape}")
        print(f"隐空间形状: {latent.shape}")
        print(f"重建RCS形状: {reconstructed_rcs.shape}")

        # 验证形状
        assert rcs_data.shape == reconstructed_rcs.shape, \
            f"形状不匹配: {rcs_data.shape} vs {reconstructed_rcs.shape}"
        assert latent.shape == (batch_size, latent_dim), \
            f"隐空间形状不匹配: {latent.shape} vs {(batch_size, latent_dim)}"

        print(f"[OK] 形状验证通过")

        # 测试梯度反向传播
        print("\n[测试梯度反向传播...]")
        loss = torch.mean((reconstructed_rcs - rcs_data) ** 2)
        loss.backward()

        if rcs_data.grad is not None:
            print(f"[OK] 梯度成功回传!")
            print(f"  输入梯度范数: {torch.norm(rcs_data.grad).item():.6e}")
            print(f"  损失值: {loss.item():.6e}")
        else:
            print("[FAIL] 梯度回传失败!")
            return False

        # 计算重建误差
        mse = torch.mean((rcs_data - reconstructed_rcs) ** 2).item()
        print(f"\n重建MSE: {mse:.6e}")

        print("\n" + "="*80)
        print(f"[SUCCESS] {architecture.upper()} 测试通过!")
        print("="*80)
        return True

    except Exception as e:
        print("\n" + "="*80)
        print(f"[FAIL] {architecture.upper()} 测试失败")
        print(f"错误: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


def main():
    """测试所有三种架构"""
    print("="*80)
    print("Differentiable Wavelet 模式测试")
    print("="*80)

    architectures = ['cnn', 'mlp', 'sine_mlp']
    results = {}

    for arch in architectures:
        results[arch] = test_differentiable_mode(arch, latent_dim=32)

    # 统计结果
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)

    for arch, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {arch.upper()}")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\n总计: {total} 个测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"通过率: {passed/total*100:.1f}%")

    if failed == 0:
        print("\n[SUCCESS] 所有测试通过!")
        print("differentiable_wavelet模式已成功集成!")
    else:
        print(f"\n[WARNING] {failed} 个测试失败")

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
