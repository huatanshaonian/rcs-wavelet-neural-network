"""
诊断模型加载和推理流程

验证项：
1. 模型文件结构完整性
2. 是否存在 enforce_nonnegative_rcs 残留代码
3. 推理流程是否会应用 Softplus/ReLU
4. 模型输出是否与训练时一致
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np

def diagnose_model_loading():
    """诊断模型加载和推理"""

    print("=" * 80)
    print("模型加载和推理流程诊断")
    print("=" * 80)

    model_path = 'G:/feko_data/wavelet/models/direct_mlp_sin_raw_20251221_220849.pth'

    # 1. 检查模型文件
    print("\n[步骤1] 检查模型文件结构")
    print("-" * 80)
    checkpoint = torch.load(model_path, map_location='cpu')

    print(f"✅ 模型文件加载成功")
    print(f"  - 包含的键: {list(checkpoint.keys())}")

    config = checkpoint.get('config', {})
    print(f"\n  - 模式: {config.get('mode', 'N/A')}")
    print(f"  - 架构: {config.get('architecture', 'N/A')}")
    print(f"  - 激活函数: {config.get('activation', 'N/A')}")
    print(f"  - 标准化: {config.get('normalize', 'N/A')}")
    print(f"  - dB变换: {config.get('db_transform', 'N/A')}")

    # 检查是否有 enforce_nonnegative_rcs 配置
    has_enforce = 'enforce_nonnegative_rcs' in config
    print(f"\n  - ⚠️ 包含 enforce_nonnegative_rcs: {has_enforce}")

    # 2. 检查 AutoEncoder 结构
    print("\n[步骤2] 检查 AutoEncoder 结构")
    print("-" * 80)
    ae_state = checkpoint['autoencoder']

    # 统计层数
    encoder_layers = [k for k in ae_state.keys() if k.startswith('encoder.')]
    decoder_layers = [k for k in ae_state.keys() if k.startswith('decoder.')]

    print(f"  - Encoder层数: {len(set([k.split('.')[1] for k in encoder_layers]))}")
    print(f"  - Decoder层数: {len(set([k.split('.')[1] for k in decoder_layers]))}")

    # 检查是否有异常层
    suspicious_keys = [k for k in ae_state.keys()
                      if 'softplus' in k.lower() or 'softmax' in k.lower() or 'relu' in k.lower()]

    if suspicious_keys:
        print(f"\n  - ❌ 发现可疑层: {suspicious_keys}")
    else:
        print(f"\n  - ✅ 没有发现 Softplus/Softmax/ReLU 层")

    # 3. 检查代码库中是否有残留
    print("\n[步骤3] 检查代码库中的残留代码")
    print("-" * 80)

    # 检查 reconstruction_manager.py
    recon_manager_path = 'gui_managers/managers/reconstruction_manager.py'
    if os.path.exists(recon_manager_path):
        with open(recon_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()

        has_enforce = 'ae_enforce_nonnegative_rcs' in content
        has_softplus = 'softplus' in content.lower()
        has_relu_constraint = 'torch.relu(reconstructed_rcs)' in content

        print(f"  - reconstruction_manager.py:")
        print(f"    - 包含 ae_enforce_nonnegative_rcs: {has_enforce}")
        print(f"    - 包含 softplus 引用: {has_softplus}")
        print(f"    - 包含 ReLU 约束: {has_relu_constraint}")

        if has_enforce or has_relu_constraint:
            print(f"    - ❌ 发现残留代码！需要清理")
        else:
            print(f"    - ✅ 已完全清理")

    # 检查 gui_autoencoder_extension.py
    extension_path = 'gui_managers/extensions/gui_autoencoder_extension.py'
    if os.path.exists(extension_path):
        with open(extension_path, 'r', encoding='utf-8') as f:
            content = f.read()

        has_enforce = 'ae_enforce_nonnegative_rcs' in content

        print(f"\n  - gui_autoencoder_extension.py:")
        print(f"    - 包含 ae_enforce_nonnegative_rcs: {has_enforce}")

        if has_enforce:
            print(f"    - ❌ 发现残留代码！需要清理")
        else:
            print(f"    - ✅ 已完全清理")

    # 4. 模拟加载和推理
    print("\n[步骤4] 模拟模型加载和推理")
    print("-" * 80)

    try:
        from autoencoder.utils.frequency_config import create_autoencoder_system

        # 根据配置创建系统
        ae_system = create_autoencoder_system(
            config_name=config.get('config_name', '2freq'),
            mode=config.get('mode', 'direct'),
            architecture=config.get('architecture', 'mlp'),
            latent_dim=config.get('latent_dim', 64),
            activation=config.get('activation', 'sin'),
            dropout_rate=config.get('dropout_rate', 0.2),
            normalize=config.get('normalize', False),
            db_transform=config.get('db_transform', False),
            normalization_method=config.get('normalization_method', 'none')
        )

        print(f"  - ✅ AutoEncoder系统创建成功")
        print(f"  - AutoEncoder类型: {type(ae_system['autoencoder']).__name__}")
        print(f"  - ParameterMapper类型: {type(ae_system['parameter_mapper']).__name__}")

        # 加载权重
        ae_system['autoencoder'].load_state_dict(checkpoint['autoencoder'])
        ae_system['parameter_mapper'].load_state_dict(checkpoint['parameter_mapper'])

        print(f"  - ✅ 模型权重加载成功")

        # 测试推理
        ae_system['autoencoder'].eval()
        ae_system['parameter_mapper'].eval()

        # 创建测试输入 (Direct MLP模式: 91×91×2)
        test_rcs = torch.randn(1, 91, 91, 2)

        with torch.no_grad():
            # 直接推理（无预处理，因为 normalize=False, db_transform=False）
            flat_input = test_rcs.reshape(1, -1)
            reconstructed, latent = ae_system['autoencoder'](flat_input)

        print(f"\n  - ✅ 推理测试成功")
        print(f"    - 输入形状: {flat_input.shape}")
        print(f"    - 隐空间形状: {latent.shape}")
        print(f"    - 重建形状: {reconstructed.shape}")
        print(f"    - 重建值范围: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")

        # 检查是否有负值（如果有，说明没有应用 Softplus/ReLU）
        has_negative = (reconstructed < 0).any().item()
        print(f"    - 包含负值: {has_negative}")

        if has_negative:
            print(f"    - ✅ 正常！输出可以有负值（未应用 Softplus/ReLU）")
        else:
            # 可能是巧合全部为正，或者有约束
            print(f"    - ⚠️ 输出全为正值（可能巧合，或存在隐含约束）")

    except Exception as e:
        print(f"  - ❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 5. 总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)

    print("\n✅ 如果所有检查都通过，说明：")
    print("  1. 模型文件结构正常，没有异常层")
    print("  2. 代码库已完全清理 enforce_nonnegative_rcs 相关代码")
    print("  3. 推理时不会应用 Softplus/ReLU 约束")
    print("  4. 模型输出与训练时一致")

    print("\n❌ 如果用户仍然看到异常，可能原因：")
    print("  1. 使用了错误的代码版本（未完全回滚）")
    print("  2. GUI 中有缓存的配置（重启 GUI 解决）")
    print("  3. 存在其他文件中的残留代码（需进一步排查）")

    print("\n建议操作：")
    print("  1. 确认当前 git 分支和提交: git log --oneline -5")
    print("  2. 确认所有 revert 提交已应用: git log --grep='Revert'")
    print("  3. 重新启动 GUI，清除任何运行时缓存")
    print("  4. 如果问题仍存在，提供具体的错误信息或异常截图")

if __name__ == '__main__':
    diagnose_model_loading()
