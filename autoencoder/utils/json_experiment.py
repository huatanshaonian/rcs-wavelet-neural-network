#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON参数空间批量实验工具

支持从JSON文件定义实验参数空间，自动生成所有实验配置。

核心特性：
1. 自动可扩展：JSON中任何字段都会传递给training_config，无需硬编码
2. 支持单值（固定参数）和列表（搜索空间）
3. 智能ID生成：限制长度，避免文件名过长
4. 完整验证：检查必需参数和合法性

JSON格式示例：
{
  "experiment_name": "latent_dim_search",
  "description": "探索隐空间维度对性能的影响",
  "base_config": {
    "mode": "wavelet",
    "architecture": "cnn",
    "activation": "relu",
    "normalization_method": "zscore",
    "db_transform": true,
    "wavelet_type": "db4",
    "training_mode": "three_stage",
    "batch_size": 8,
    "learning_rate": 0.001,
    "epochs": {"stage1": 100, "stage2": 50, "stage3": 30}
  },
  "parameter_grid": {
    "latent_dim": [64, 128, 256, 512],
    "dropout_rate": [0.1, 0.2, 0.3]
  }
}

使用方法：
    from autoencoder.utils.json_experiment import load_experiments_from_json

    exp_config = load_experiments_from_json('experiments/my_experiment.json')
    print(f"实验名称: {exp_config['experiment_name']}")
    print(f"实验数量: {len(exp_config['experiments'])}")

    for config in exp_config['experiments']:
        print(f"实验ID: {config['experiment_id']}")
        # config包含所有训练超参数，可直接传给训练函数
"""

import json
import os
from itertools import product
from typing import Dict, List, Any, Union
from datetime import datetime


# 必需的基础参数（确保最小可运行配置）
REQUIRED_BASE_PARAMS = {
    'mode': 'wavelet',
    'architecture': 'cnn',
    'activation': 'relu',
    'latent_dim': 256,
    'batch_size': 8,
    'learning_rate': 0.001,
    'training_mode': 'three_stage',
    'epochs': {'stage1': 100, 'stage2': 50, 'stage3': 30}
}

# 可选参数默认值（可扩展，不影响现有逻辑）
DEFAULT_OPTIONAL_PARAMS = {
    'normalization_method': 'zscore',
    'db_transform': True,
    'wavelet_type': 'db4',
    'dropout_rate': 0.2,
    'lr_scheduler': 'plateau',
    'optimizer_type': 'adam',
    'momentum': 0.9,
    'gradient_monitoring': False,
    'channel_attention': False,
    'loss_normalization': False,
    'ssim_weight': 0.0,
    'gradient_penalty_weight': 0.0,
    'num_lr_stages': 3,
    'lr_decay_factor': 0.1
}


def load_experiments_from_json(json_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    从JSON文件加载实验配置并生成所有实验组合

    Args:
        json_path: JSON配置文件路径
        validate: 是否验证配置合法性

    Returns:
        {
            'experiment_name': str,
            'description': str,
            'base_config': dict,
            'parameter_grid': dict,
            'total_experiments': int,
            'experiments': List[dict]  # 所有实验配置
        }

    Raises:
        FileNotFoundError: JSON文件不存在
        ValueError: JSON格式错误或缺少必需参数
    """
    # 读取JSON文件
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON配置文件不存在: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {e}")

    # 验证顶层结构
    if 'base_config' not in config:
        raise ValueError("JSON配置缺少'base_config'字段")

    if 'parameter_grid' not in config:
        raise ValueError("JSON配置缺少'parameter_grid'字段")

    base_config = config['base_config']
    parameter_grid = config['parameter_grid']
    experiment_name = config.get('experiment_name', 'unnamed_experiment')
    description = config.get('description', '')

    # 验证base_config（可选，默认宽松处理）
    if validate:
        _validate_base_config(base_config)

    # 补充默认值（只补充base_config中未设置的可选参数）
    complete_base_config = _fill_default_values(base_config)

    # 生成所有实验配置
    experiments = _generate_experiment_configs(complete_base_config, parameter_grid)

    return {
        'experiment_name': experiment_name,
        'description': description,
        'base_config': complete_base_config,
        'parameter_grid': parameter_grid,
        'total_experiments': len(experiments),
        'experiments': experiments,
        'created_time': datetime.now().isoformat(),
        'json_path': os.path.abspath(json_path)
    }


def _validate_base_config(base_config: Dict[str, Any]):
    """
    验证基础配置的合法性

    Args:
        base_config: 基础配置字典

    Raises:
        ValueError: 配置不合法
    """
    # 检查必需参数
    missing_params = []
    for param in REQUIRED_BASE_PARAMS.keys():
        if param not in base_config:
            missing_params.append(param)

    if missing_params:
        raise ValueError(
            f"base_config缺少必需参数: {missing_params}\n"
            f"必需参数列表: {list(REQUIRED_BASE_PARAMS.keys())}"
        )

    # 验证epochs格式（如果存在）
    if 'epochs' in base_config:
        epochs = base_config['epochs']
        if not isinstance(epochs, dict):
            raise ValueError(f"epochs必须是字典格式，当前类型: {type(epochs)}")

        required_stages = ['stage1', 'stage2', 'stage3']
        for stage in required_stages:
            if stage not in epochs:
                raise ValueError(f"epochs缺少{stage}配置")
            if not isinstance(epochs[stage], int) or epochs[stage] <= 0:
                raise ValueError(f"epochs[{stage}]必须是正整数，当前值: {epochs[stage]}")

    # 验证mode合法性
    valid_modes = ['wavelet', 'direct', 'differentiable_wavelet']
    if base_config['mode'] not in valid_modes:
        raise ValueError(f"mode必须是{valid_modes}之一，当前值: {base_config['mode']}")

    # 验证architecture合法性
    valid_architectures = ['cnn', 'mlp', 'enhanced_cnn', 'deep_cnn',
                          'dual_branch_cnn', 'dual_branch_mlp',
                          'dual_branch_cnn_v2', 'dual_branch_mlp_v2']
    if base_config['architecture'] not in valid_architectures:
        raise ValueError(
            f"architecture必须是{valid_architectures}之一，当前值: {base_config['architecture']}"
        )


def _fill_default_values(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    用默认值补充缺失的可选参数

    ⚠️ 关键设计：只补充未设置的参数，不覆盖用户配置
    这确保了完全的可扩展性——用户可以添加任何新参数而不影响此函数

    Args:
        base_config: 原始配置

    Returns:
        补充默认值后的完整配置
    """
    complete_config = base_config.copy()

    # 只补充未设置的可选参数
    for param, default_value in DEFAULT_OPTIONAL_PARAMS.items():
        if param not in complete_config:
            complete_config[param] = default_value

    # 特殊处理：如果用户没设置epochs，使用默认值
    if 'epochs' not in complete_config:
        complete_config['epochs'] = REQUIRED_BASE_PARAMS['epochs'].copy()

    return complete_config


def _generate_experiment_configs(base_config: Dict[str, Any],
                                 parameter_grid: Dict[str, Union[List, Any]]) -> List[Dict[str, Any]]:
    """
    生成所有实验配置（笛卡尔积）

    ⚠️ 核心可扩展性设计：
    - parameter_grid中的任何参数都会自动覆盖base_config
    - 不需要硬编码参数列表
    - 新增参数时无需修改此函数

    Args:
        base_config: 基础配置
        parameter_grid: 参数搜索空间
            - 值为列表：该参数的搜索空间（会生成多个实验）
            - 值为单值：固定参数（所有实验共享）

    Returns:
        实验配置列表
    """
    # 分离固定参数和搜索参数
    fixed_params = {}
    search_params = {}

    for param_name, param_value in parameter_grid.items():
        if isinstance(param_value, list):
            if len(param_value) > 0:
                search_params[param_name] = param_value
            else:
                # 空列表：跳过该参数
                print(f"⚠️ 警告: parameter_grid['{param_name}']是空列表，已跳过")
        else:
            # 单值：作为固定参数
            fixed_params[param_name] = param_value

    # 如果没有搜索参数，返回单个实验（base + fixed）
    if not search_params:
        config = base_config.copy()
        config.update(fixed_params)
        config['experiment_id'] = 'exp000_single'
        return [config]

    # 生成笛卡尔积
    param_names = list(search_params.keys())
    param_values = list(search_params.values())

    experiments = []
    for idx, combination in enumerate(product(*param_values)):
        # 复制base_config
        exp_config = base_config.copy()

        # 应用固定参数
        exp_config.update(fixed_params)

        # 应用当前搜索参数组合
        for param_name, param_value in zip(param_names, combination):
            exp_config[param_name] = param_value

        # 生成实验ID（智能限制长度）
        exp_config['experiment_id'] = _generate_experiment_id(idx, param_names, combination)

        experiments.append(exp_config)

    return experiments


def _generate_experiment_id(index: int, param_names: List[str], param_values: List[Any]) -> str:
    """
    生成简洁的实验ID（避免文件名过长）

    策略：
    1. 最多显示前3个参数（最重要的）
    2. 参数值格式化为简洁字符串
    3. 总长度限制在80字符内

    Args:
        index: 实验序号
        param_names: 参数名列表
        param_values: 参数值列表

    Returns:
        实验ID字符串
    """
    id_parts = [f"exp{index:03d}"]

    # 最多显示前3个参数
    max_params = min(3, len(param_names))

    for i in range(max_params):
        param_name = param_names[i]
        param_value = param_values[i]

        # 简化参数名（只取前3个字符）
        short_name = _abbreviate_param_name(param_name)

        # 格式化参数值
        value_str = _format_param_value(param_value)

        id_parts.append(f"{short_name}{value_str}")

    # 如果还有更多参数，添加省略号
    if len(param_names) > max_params:
        id_parts.append(f"etc{len(param_names) - max_params}")

    experiment_id = "_".join(id_parts)

    # 截断过长ID
    if len(experiment_id) > 80:
        experiment_id = experiment_id[:77] + "..."

    return experiment_id


def _abbreviate_param_name(param_name: str) -> str:
    """
    简化参数名为简洁缩写

    Args:
        param_name: 原始参数名

    Returns:
        简化后的参数名
    """
    # 特殊缩写映射
    abbreviations = {
        'latent_dim': 'ld',
        'learning_rate': 'lr',
        'batch_size': 'bs',
        'dropout_rate': 'dr',
        'activation': 'act',
        'architecture': 'arch',
        'lr_scheduler': 'sch',
        'optimizer_type': 'opt',
        'normalization_method': 'norm',
        'db_transform': 'db',
        'wavelet_type': 'wv',
        'num_lr_stages': 'nstg',
        'lr_decay_factor': 'decay'
    }

    return abbreviations.get(param_name, param_name[:3])


def _format_param_value(value: Any) -> str:
    """
    格式化参数值为简洁字符串

    Args:
        value: 参数值

    Returns:
        格式化后的字符串
    """
    if isinstance(value, bool):
        return 't' if value else 'f'

    elif isinstance(value, float):
        # 科学计数法
        if value < 0.01 or value > 1000:
            return f"{value:.0e}".replace('-0', '-')  # 1e-03 -> 1e-3
        else:
            return f"{value:.3g}"

    elif isinstance(value, int):
        return str(value)

    elif isinstance(value, dict):
        # 字典：只显示关键信息
        # epochs: {"stage1": 100, "stage2": 50} -> "s1-100_s2-50"
        parts = []
        for k, v in list(value.items())[:2]:  # 最多2个键
            short_k = k.replace('stage', 's')
            parts.append(f"{short_k}{v}")
        return "_".join(parts)

    elif isinstance(value, str):
        # 字符串：最多10个字符
        return value[:10]

    else:
        return str(value)[:10]


def save_experiment_template(save_path: str = 'experiments/template.json'):
    """
    保存一个实验配置模板文件，方便用户快速开始

    Args:
        save_path: 保存路径
    """
    template = {
        "_comment": "这是AutoEncoder批量实验的JSON配置模板",
        "_usage": "1. 修改base_config设置固定参数; 2. 在parameter_grid中定义搜索空间; 3. 运行批量实验",
        "_扩展性说明": "parameter_grid中可以包含任何训练超参数，无需修改代码",

        "experiment_name": "example_experiment",
        "description": "示例实验：探索隐空间维度和学习率的影响",

        "base_config": {
            "# 模型配置": "-----",
            "mode": "wavelet",
            "architecture": "cnn",
            "activation": "relu",
            "latent_dim": 256,

            "# 数据预处理": "-----",
            "normalization_method": "zscore",
            "db_transform": True,
            "wavelet_type": "db4",

            "# 训练配置": "-----",
            "training_mode": "three_stage",
            "batch_size": 8,
            "learning_rate": 0.001,
            "epochs": {
                "stage1": 100,
                "stage2": 50,
                "stage3": 30
            },

            "# 优化器配置": "-----",
            "lr_scheduler": "plateau",
            "optimizer_type": "adam",
            "momentum": 0.9,

            "# 正则化": "-----",
            "dropout_rate": 0.2,

            "# 高级功能（可选）": "-----",
            "gradient_monitoring": False,
            "channel_attention": False,
            "loss_normalization": False,
            "ssim_weight": 0.0,
            "gradient_penalty_weight": 0.0
        },

        "parameter_grid": {
            "_comment_说明": "列表值会生成笛卡尔积，单值是固定参数",

            "latent_dim": [128, 256, 512],
            "learning_rate": [0.001, 0.0001],
            "dropout_rate": [0.1, 0.2, 0.3]
        },

        "_comment_更多可选参数示例": {
            "activation": ["relu", "sin", "gelu"],
            "architecture": ["cnn", "mlp", "enhanced_cnn"],
            "batch_size": [8, 16, 32],
            "lr_scheduler": ["cosine", "plateau", "multi_stage"],
            "optimizer_type": ["adam", "adamw", "sgd", "lbfgs"],
            "num_lr_stages": [2, 3, 4],
            "lr_decay_factor": [0.1, 0.5]
        }
    }

    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"[OK] 实验配置模板已保存: {save_path}")
    return save_path


def print_experiment_summary(exp_config: Dict[str, Any]):
    """
    打印实验配置摘要

    Args:
        exp_config: load_experiments_from_json的返回值
    """
    print("=" * 80)
    print(f"实验名称: {exp_config['experiment_name']}")
    if exp_config['description']:
        print(f"实验描述: {exp_config['description']}")
    print(f"实验数量: {exp_config['total_experiments']}")
    print(f"配置文件: {exp_config['json_path']}")
    print(f"创建时间: {exp_config['created_time']}")
    print("=" * 80)

    print("\n📋 基础配置:")
    for key, value in exp_config['base_config'].items():
        if not key.startswith('#'):
            print(f"  {key}: {value}")

    print("\n🔍 参数搜索空间:")
    for key, value in exp_config['parameter_grid'].items():
        if isinstance(value, list):
            print(f"  {key}: {value} ({len(value)}个选项)")
        else:
            print(f"  {key}: {value} (固定)")

    print("\n📊 实验列表预览 (前5个):")
    for i, config in enumerate(exp_config['experiments'][:5], 1):
        print(f"  {i}. {config['experiment_id']}")

    if exp_config['total_experiments'] > 5:
        print(f"  ... 还有 {exp_config['total_experiments'] - 5} 个实验")

    print("=" * 80)


if __name__ == "__main__":
    """测试JSON实验配置加载器"""
    print("JSON实验配置工具测试\n")

    # 1. 生成模板
    print("1️⃣ 生成配置模板...")
    template_path = save_experiment_template('experiments/template.json')

    # 2. 加载模板
    print("\n2️⃣ 加载配置...")
    try:
        exp_config = load_experiments_from_json(template_path, validate=False)
        print_experiment_summary(exp_config)

        print("\n3️⃣ 实验配置示例:")
        if exp_config['experiments']:
            example = exp_config['experiments'][0]
            print(f"实验ID: {example['experiment_id']}")
            print("完整配置:")
            for key, value in example.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ 测试完成！")
