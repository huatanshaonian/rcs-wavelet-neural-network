#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
转换旧sine_mlp模型到新的参数化激活函数架构
"""

import torch
import json
import sys

def convert_sine_mlp_model(input_path, output_path):
    """
    转换sine_mlp模型到新架构

    旧模型: SinDirectMLPAutoEncoder (硬编码sin激活)
    新模型: DirectMLPAutoEncoder(activation='sin') (参数化激活)
    """
    print("=" * 80)
    print("转换 Sine MLP 模型到新架构")
    print("=" * 80)

    # 1. 加载旧模型
    print(f"\n[1/4] 加载旧模型: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    print("\n模型文件包含的键:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # 2. 检查和修正config
    print("\n[2/4] 分析模型配置...")

    # 旧模型可能没有config字段，需要从state_dict推断
    if 'config' in checkpoint:
        old_config = checkpoint['config']
        print(f"找到config: {old_config}")
    else:
        print("WARNING: 没有config字段，从state_dict推断配置")
        old_config = {}

    # 从autoencoder state_dict推断配置
    if 'autoencoder' in checkpoint:
        ae_state = checkpoint['autoencoder']

        # 检查输入维度判断频率数
        if 'encoder.1.weight' in ae_state:
            input_dim = ae_state['encoder.1.weight'].shape[1]

            if input_dim == 16562:  # 91*91*2
                num_frequencies = 2
            elif input_dim == 24843:  # 91*91*3
                num_frequencies = 3
            else:
                print(f"ERROR: 未知的输入维度 {input_dim}")
                return False

            print(f"  检测到输入维度: {input_dim}")
            print(f"  推断频率数: {num_frequencies}")
        else:
            print("ERROR: 无法找到encoder.1.weight")
            return False

        # 推断latent_dim
        # 查找encoder的最后一个Linear层
        encoder_layers = [k for k in ae_state.keys() if k.startswith('encoder.')]
        last_encoder_layer = max([k for k in encoder_layers if '.weight' in k and 'Linear' not in k or k.endswith('.weight')],
                                 key=lambda x: int(x.split('.')[1]))
        if last_encoder_layer:
            latent_dim = ae_state[last_encoder_layer].shape[0]
            print(f"  推断latent_dim: {latent_dim}")
        else:
            latent_dim = 256  # 默认值
            print(f"  使用默认latent_dim: {latent_dim}")
    else:
        print("ERROR: checkpoint中没有autoencoder字段")
        return False

    # 3. 创建新配置
    print("\n[3/4] 创建新配置...")

    new_config = {
        'mode': 'direct',
        'architecture': 'mlp',  # <-- 关键：改为mlp而非sine_mlp
        'activation': 'sin',     # <-- 关键：添加activation参数
        'latent_dim': old_config.get('latent_dim', latent_dim),
        'num_frequencies': old_config.get('num_frequencies', num_frequencies),
        'dropout_rate': old_config.get('dropout_rate', 0.2),
        'wavelet_type': old_config.get('wavelet_type', 'db4'),
    }

    print("新配置:")
    for k, v in new_config.items():
        print(f"  {k}: {v}")

    # 4. 更新checkpoint
    print("\n[4/4] 保存转换后的模型...")

    # 保留所有原始数据，只更新config
    checkpoint['config'] = new_config

    # 如果有training_history，也保留
    # 如果有adapter_stats，也保留
    # 权重完全不需要修改，因为网络结构相同

    # 保存
    torch.save(checkpoint, output_path)
    print(f"[OK] 转换后的模型已保存: {output_path}")

    # 同时保存配置JSON
    config_path = output_path.replace('.pth', '_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    print(f"[OK] 配置文件已保存: {config_path}")

    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print("\n使用方法:")
    print("  1. 在GUI中选择:")
    print("     - AE模式: Direct")
    print("     - 架构类型: MLP")
    print(f"     - 频率配置: {'2freq' if num_frequencies == 2 else '3freq'}")
    print(f"     - 激活函数: Sin")
    print("  2. 点击'创建当前模式系统'")
    print(f"  3. 加载转换后的模型: {output_path}")
    print("\n注意:")
    print("  - 权重完全兼容，无需重新训练")
    print("  - 新架构支持动态切换激活函数")
    print("  - 保留了所有训练历史和统计信息")

    return True


if __name__ == '__main__':
    input_model = 'models/mlp10000.pth'
    output_model = 'models/mlp10000_converted.pth'

    print("\n输入文件:", input_model)
    print("输出文件:", output_model)
    print()

    success = convert_sine_mlp_model(input_model, output_model)

    if success:
        print("\n[SUCCESS] 模型转换成功！")
        sys.exit(0)
    else:
        print("\n[FAILED] 模型转换失败")
        sys.exit(1)
