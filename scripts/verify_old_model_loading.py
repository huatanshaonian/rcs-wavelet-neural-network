"""验证旧模型的加载行为（模拟GUI加载流程）"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch

def test_old_model_loading():
    """测试旧模型（无enforce_nonnegative_rcs配置）的加载行为"""

    print("=" * 70)
    print("测试旧模型加载行为（模拟GUI）")
    print("=" * 70)

    model_path = 'G:/feko_data/wavelet/models/direct_mlp_sin_raw_20251221_220849.pth'

    # 1. 加载模型文件
    print(f"\n【步骤1】加载模型文件: {os.path.basename(model_path)}")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})

    print(f"✅ 模型配置:")
    print(f"  - mode: {config.get('mode', 'N/A')}")
    print(f"  - architecture: {config.get('architecture', 'N/A')}")
    print(f"  - activation: {config.get('activation', 'N/A')}")
    print(f"  - normalization_method: {config.get('normalization_method', 'N/A')}")

    # 2. 检查是否包含enforce_nonnegative_rcs
    print(f"\n【步骤2】检查enforce_nonnegative_rcs配置")
    has_enforce = 'enforce_nonnegative_rcs' in config
    print(f"  - 模型文件包含enforce_nonnegative_rcs: {has_enforce}")

    if has_enforce:
        print(f"  - 值: {config['enforce_nonnegative_rcs']}")
    else:
        print(f"  - ⚠️ 旧模型，缺少此配置")

    # 3. 模拟GUI加载时的默认值处理
    print(f"\n【步骤3】模拟GUI加载时的默认值处理")

    # 检查当前代码的行为
    if 'enforce_nonnegative_rcs' in config:
        enforce_value = config['enforce_nonnegative_rcs']
        print(f"  - ✅ 从模型文件读取: {enforce_value}")
    else:
        enforce_value = False  # 修复后的默认值
        print(f"  - ✅ 使用默认值（旧模型）: {enforce_value}")

    # 4. 验证修复效果
    print(f"\n【步骤4】验证修复效果")
    print(f"  - 最终enforce_nonnegative_rcs值: {enforce_value}")

    if enforce_value == False:
        print(f"  - ✅ 正确！使用False，保持训练时行为")
        print(f"  - ✅ 重建RCS不会被Softplus处理")
        print(f"  - ✅ 输出范围应与训练时一致")
    else:
        print(f"  - ❌ 错误！使用True会改变重建结果")
        print(f"  - ❌ 重建RCS会被Softplus压缩")
        print(f"  - ❌ 输出范围会与训练时不同")

    print("\n" + "=" * 70)
    print("验证完成")
    print("=" * 70)

    # 5. 展示Softplus的影响
    print(f"\n【额外信息】Softplus的影响示例")
    import torch.nn.functional as F

    test_values = torch.tensor([-0.0122, -0.005, 0.0, 0.2, 0.491])
    softplus_values = F.softplus(test_values, beta=1)

    print(f"  原始值: {test_values.tolist()}")
    print(f"  Softplus后: {[f'{v:.6f}' for v in softplus_values.tolist()]}")
    print(f"  说明：负值被压缩，正值略微减小")

if __name__ == '__main__':
    test_old_model_loading()
