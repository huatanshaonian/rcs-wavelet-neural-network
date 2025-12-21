#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复ae112.pth的config信息
基于日志文件 logs\rcs_wavelet_20251011_165553.log 的分析结果
"""

import torch
import os

def fix_ae112_config():
    """修复ae112模型的config信息"""

    print("="*70)
    print("修复 ae112.pth 的 Config 信息")
    print("="*70)
    print()

    # 1. 加载原始模型
    model_path = 'models/ae112.pth'
    if not os.path.exists(model_path):
        print(f"Error: File not found {model_path}")
        return

    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 2. 显示原始config
    print("\n原始Config:")
    for key, value in checkpoint['config'].items():
        print(f"  {key}: {value}")

    # 3. 从日志分析得到的配置信息
    # 日志: logs\rcs_wavelet_20251011_165553.log
    # - Direct模式 + MLP架构
    # - 2freq配置
    # - 隐空间维度: 256 (从日志)
    # - 输入维度: 16562 = 91*91*2 (Direct模式，无小波变换)
    # - ParameterMapper输出: 128 (从权重推断，这个不太对，应该是256)

    # 从AutoEncoder的encoder最后一层推断latent_dim
    # encoder.21.bias的形状就是latent_dim
    ae_latent_dim = checkpoint['autoencoder']['encoder.21.bias'].shape[0]

    print(f"\nInferred latent_dim from AutoEncoder weights: {ae_latent_dim}")

    # 4. 补全config
    updated_config = {
        'config_name': '2freq',
        'latent_dim': ae_latent_dim,  # 使用从AutoEncoder推断的值
        'dropout_rate': 0.2,  # 标准值
        'wavelet': 'db4',  # 虽然是direct模式，但保留这个字段
        'normalize': True,
        'mode': 'direct',  # 从日志确认: Direct模式
        'architecture': 'mlp'  # 从日志确认: MLP架构
    }

    checkpoint['config'].update(updated_config)

    # 5. 显示更新后的config
    print("\nUpdated Config:")
    for key, value in sorted(checkpoint['config'].items()):
        print(f"  {key}: {value}")

    # 6. 保存为新文件
    output_path = 'models/ae112_fixed.pth'
    torch.save(checkpoint, output_path)

    file_size_mb = os.path.getsize(output_path) / (1024*1024)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print()
    print("="*70)
    print("Fix completed!")
    print("="*70)
    print()
    print("You can now load ae112_fixed.pth using GUI 'Load Model' button")
    print()

    # 7. 验证
    print("Verifying...")
    verify_checkpoint = torch.load(output_path, map_location='cpu', weights_only=False)
    required_fields = ['config_name', 'latent_dim', 'dropout_rate', 'wavelet',
                       'normalize', 'mode', 'architecture']
    missing = [f for f in required_fields if f not in verify_checkpoint['config']]

    if missing:
        print(f"Warning: Still missing fields: {missing}")
    else:
        print("OK: All required fields present")
    print()

if __name__ == "__main__":
    fix_ae112_config()
