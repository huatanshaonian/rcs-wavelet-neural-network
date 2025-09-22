#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
除零错误调试脚本

诊断训练过程中的 division by zero 错误
"""

import os
import sys
import torch
import numpy as np
import traceback
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_division_by_zero():
    """调试除零错误"""
    print("=" * 60)
    print("除零错误调试")
    print("=" * 60)

    try:
        # 导入模块
        from wavelet_network import create_model, create_loss_function
        from training import RCSDataset, create_data_config, create_training_config, CrossValidationTrainer
        from data_cache import create_cache_manager

        print("1. 模块导入成功")

        # 创建测试数据
        data_config = create_data_config()
        training_config = create_training_config()

        print(f"2. 训练配置: {training_config}")

        # 加载少量数据用于测试
        test_model_ids = [f"{i:03d}" for i in range(1, 11)]  # 10个模型

        cache_manager = create_cache_manager()
        param_data, rcs_data = cache_manager.load_data_with_cache(
            params_file=data_config['params_file'],
            rcs_data_dir=data_config['rcs_data_dir'],
            model_ids=test_model_ids,
            frequencies=data_config['frequencies']
        )

        print(f"3. 数据形状: 参数{param_data.shape}, RCS{rcs_data.shape}")

        # 检查数据中是否有零值或异常值
        print("4. 数据完整性检查:")
        print(f"   参数数据范围: [{param_data.min():.6f}, {param_data.max():.6f}]")
        print(f"   RCS数据范围: [{rcs_data.min():.6f}, {rcs_data.max():.6f}]")
        print(f"   参数数据中的零值: {np.sum(param_data == 0)}")
        print(f"   RCS数据中的零值: {np.sum(rcs_data == 0)}")
        print(f"   参数数据中的NaN: {np.sum(np.isnan(param_data))}")
        print(f"   RCS数据中的NaN: {np.sum(np.isnan(rcs_data))}")

        # 创建数据集
        dataset = RCSDataset(param_data, rcs_data)
        print(f"5. 数据集大小: {len(dataset)}")

        # 测试损失函数计算
        print("6. 测试损失函数...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 创建模型
        model = create_model(input_dim=9, hidden_dims=[64, 128])
        model.to(device)

        # 创建损失函数
        loss_fn = create_loss_function()

        # 测试单个样本
        sample_params, sample_rcs = dataset[0]
        test_params = sample_params.unsqueeze(0).to(device)
        test_rcs = sample_rcs.unsqueeze(0).to(device)

        print(f"   测试参数形状: {test_params.shape}")
        print(f"   测试RCS形状: {test_rcs.shape}")

        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(test_params)
            print(f"   模型输出形状: {output.shape}")
            print(f"   模型输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")

            # 检查输出中的零值和异常值
            print(f"   输出中的零值: {torch.sum(output == 0).item()}")
            print(f"   输出中的NaN: {torch.sum(torch.isnan(output)).item()}")
            print(f"   输出中的Inf: {torch.sum(torch.isinf(output)).item()}")

            try:
                # 尝试计算损失
                print("7. 计算损失...")
                losses = loss_fn(output, test_rcs)
                print(f"   损失计算成功: {losses}")

            except Exception as loss_error:
                print(f"   损失计算失败: {loss_error}")
                print("   详细错误信息:")
                traceback.print_exc()

                # 分别测试每个损失组件
                print("\n8. 分别测试损失组件:")

                try:
                    # MSE损失
                    mse_loss = torch.nn.functional.mse_loss(output, test_rcs)
                    print(f"   MSE损失: {mse_loss.item()}")
                except Exception as e:
                    print(f"   MSE损失失败: {e}")

                try:
                    # 对称性损失测试
                    print("   测试对称性损失组件...")
                    # 模拟对称性损失计算
                    # output: [B, 91, 91, 2]
                    batch_size = output.shape[0]

                    symmetry_loss = torch.tensor(0.0, device=output.device)

                    for b in range(batch_size):
                        for freq in range(2):  # 两个频率
                            rcs_matrix = output[b, :, :, freq]  # [91, 91]

                            # 检查矩阵中是否有异常值
                            if torch.isnan(rcs_matrix).any() or torch.isinf(rcs_matrix).any():
                                print(f"     频率 {freq} 矩阵包含NaN或Inf值")

                            # 计算对称性
                            diff = rcs_matrix - rcs_matrix.T
                            sym_loss = torch.mean(diff ** 2)

                            print(f"     批次{b}, 频率{freq}: 对称性损失 = {sym_loss.item()}")

                            symmetry_loss += sym_loss

                    if batch_size > 0:
                        symmetry_loss /= (batch_size * 2)

                    print(f"   对称性损失总计: {symmetry_loss.item()}")

                except Exception as e:
                    print(f"   对称性损失测试失败: {e}")
                    traceback.print_exc()

        # 测试交叉验证设置
        print("\n9. 测试交叉验证配置...")
        try:
            trainer = CrossValidationTrainer(
                model_params={'input_dim': 9, 'hidden_dims': [64, 128]},
                device=device,
                n_folds=5
            )

            print(f"   交叉验证器创建成功")
            print(f"   数据集大小: {len(dataset)}")
            print(f"   折数: 5")

            # 检查是否数据集太小导致某些fold为空
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            fold_sizes = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(param_data)):
                train_size = len(train_idx)
                val_size = len(val_idx)
                fold_sizes.append((train_size, val_size))
                print(f"   Fold {fold+1}: 训练集{train_size}, 验证集{val_size}")

                if train_size == 0 or val_size == 0:
                    print(f"   警告: Fold {fold+1} 中有空集合！")

            # 检查批次大小配置
            batch_size = training_config['batch_size']
            min_fold_size = min([min(train, val) for train, val in fold_sizes])

            print(f"   配置的批次大小: {batch_size}")
            print(f"   最小fold大小: {min_fold_size}")

            if batch_size > min_fold_size:
                print(f"   警告: 批次大小({batch_size}) > 最小fold大小({min_fold_size})")
                print("   这可能导致某些fold无法创建有效的数据加载器!")

        except Exception as e:
            print(f"   交叉验证测试失败: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"调试过程出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_division_by_zero()