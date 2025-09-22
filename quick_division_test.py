#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速除零错误测试

只测试一个简单的训练循环以验证修复
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """快速测试除零错误修复"""
    print("=" * 40)
    print("快速除零错误修复测试")
    print("=" * 40)

    try:
        from training import CrossValidationTrainer, RCSDataset, create_training_config
        from data_cache import create_cache_manager
        from training import create_data_config

        # 创建配置
        config = create_training_config()
        data_config = create_data_config()

        print(f"配置的批次大小: {config['batch_size']}")

        # 创建非常小的数据集
        test_model_ids = [f"{i:03d}" for i in range(1, 4)]  # 只用3个模型

        cache_manager = create_cache_manager()
        param_data, rcs_data = cache_manager.load_data_with_cache(
            params_file=data_config['params_file'],
            rcs_data_dir=data_config['rcs_data_dir'],
            model_ids=test_model_ids,
            frequencies=data_config['frequencies']
        )

        dataset = RCSDataset(param_data, rcs_data)
        print(f"测试数据集大小: {len(dataset)}")

        # 创建交叉验证训练器
        trainer = CrossValidationTrainer(
            model_params={'input_dim': 9, 'hidden_dims': [64, 128]},
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_folds=3  # 减少折数以加速测试
        )

        # 快速测试，只运行1个epoch
        config['epochs'] = 1
        config['batch_size'] = 1  # 使用最小批次大小

        print("开始快速交叉验证测试...")
        results = trainer.cross_validate(dataset, config)

        print("快速测试成功!")
        print(f"结果: 平均分数 = {results['mean_score']:.6f}")

        # 检查是否还有inf值
        if any(score == float('inf') for score in results['fold_scores']):
            print("警告: 仍然存在inf损失值")
        else:
            print("✓ 所有损失值均为有限值，除零错误已修复!")

    except Exception as e:
        print(f"快速测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()