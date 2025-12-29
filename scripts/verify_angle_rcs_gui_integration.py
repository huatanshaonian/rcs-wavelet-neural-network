#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证Angle-based RCS GUI集成

检查所有组件是否正确安装和集成：
1. 核心网络模块（AngleRCSNetwork）
2. 数据处理模块（AngleSampler, AngleRCSDataset）
3. 训练器模块（AngleRCSTrainer）
4. GUI扩展模块（AngleRCSExtension）
5. GUI集成（gui.py中的集成代码）
"""

# 设置UTF-8编码（Windows兼容）
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os

# 添加项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """测试所有模块导入"""
    print("=" * 80)
    print("Angle-based RCS GUI集成验证")
    print("=" * 80)

    tests_passed = 0
    tests_total = 6

    # Test 1: 网络模块
    print("\n[Test 1/6] 导入AngleRCSNetwork...")
    try:
        from angle_based_rcs.models.angle_rcs_network import AngleRCSNetwork
        print("  ✓ AngleRCSNetwork导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # Test 2: 数据处理模块
    print("\n[Test 2/6] 导入数据处理模块...")
    try:
        from angle_based_rcs.data.angle_sampler import AngleSampler
        from angle_based_rcs.data.angle_dataset import AngleRCSDataset, create_dataloaders
        print("  ✓ AngleSampler, AngleRCSDataset, create_dataloaders导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # Test 3: 训练器模块
    print("\n[Test 3/6] 导入AngleRCSTrainer...")
    try:
        from angle_based_rcs.training.angle_trainer import AngleRCSTrainer
        print("  ✓ AngleRCSTrainer导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # Test 4: GUI扩展模块
    print("\n[Test 4/6] 导入AngleRCSExtension...")
    try:
        from gui_managers.extensions.gui_angle_rcs_extension import AngleRCSExtension
        print("  ✓ AngleRCSExtension导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # Test 5: 检查GUI集成代码
    print("\n[Test 5/6] 检查gui.py集成代码...")
    try:
        gui_path = os.path.join(project_root, 'gui.py')
        with open(gui_path, 'r', encoding='utf-8') as f:
            gui_content = f.read()

        # 检查集成代码
        if 'from gui_managers.extensions.gui_angle_rcs_extension import AngleRCSExtension' in gui_content:
            if 'app.angle_rcs_extension = AngleRCSExtension(app)' in gui_content:
                if 'app.angle_rcs_extension.create_angle_rcs_tab()' in gui_content:
                    print("  ✓ gui.py已正确集成AngleRCSExtension")
                    tests_passed += 1
                else:
                    print("  ✗ 缺少create_angle_rcs_tab()调用")
            else:
                print("  ✗ 缺少AngleRCSExtension实例化")
        else:
            print("  ✗ 缺少AngleRCSExtension导入")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # Test 6: 创建测试实例
    print("\n[Test 6/6] 创建测试网络实例...")
    try:
        import torch
        from angle_based_rcs.models.angle_rcs_network import AngleRCSNetwork

        model = AngleRCSNetwork(
            num_frequencies=3,
            angle_L=16,
            param_dim=9,
            param_embed_dim=128,
            activation='sin',
            dropout_rate=0.1
        )

        # 测试前向传播
        theta = torch.randn(4)
        phi = torch.randn(4)
        params = torch.randn(4, 9)
        freq_idx = torch.tensor([0, 1, 2, 1])

        rcs_pred = model(theta, phi, params, freq_idx)

        if rcs_pred.shape == (4, 1):
            print(f"  ✓ 网络实例化成功，输出形状正确: {rcs_pred.shape}")
            tests_passed += 1
        else:
            print(f"  ✗ 输出形状错误: {rcs_pred.shape}, 期望: (4, 1)")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # 总结
    print("\n" + "=" * 80)
    print(f"测试结果: {tests_passed}/{tests_total} 通过")
    print("=" * 80)

    if tests_passed == tests_total:
        print("\n✓ 所有测试通过！Angle-based RCS GUI集成完成。")
        print("\n使用方法:")
        print("  1. 运行GUI: python gui.py")
        print("  2. 在GUI中找到'Angle-based RCS'标签页")
        print("  3. 配置模型参数和训练设置")
        print("  4. 点击'开始训练'按钮")
        return True
    else:
        print(f"\n✗ 有{tests_total - tests_passed}个测试失败，请检查安装。")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
