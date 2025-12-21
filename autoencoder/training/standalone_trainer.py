#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立AutoEncoder训练函数 - 无GUI依赖

提供可在任何环境下运行的训练接口：
- CLI脚本
- Jupyter Notebook
- 批量实验
- 远程服务器

核心特性：
1. 完全独立：不依赖GUI，最小化依赖
2. 灵活回调：支持自定义日志和进度回调
3. 完整兼容：复用现有AETrainer的所有逻辑
4. 自动管理：自动创建ae_system、初始化data_adapter

使用示例：
    from autoencoder.training.standalone_trainer import train_autoencoder_standalone

    # 最简单的用法
    result = train_autoencoder_standalone(
        rcs_data=rcs_data,
        param_data=param_data,
        training_config={
            'mode': 'wavelet',
            'latent_dim': 256,
            'learning_rate': 0.001,
            # ... 其他配置
        }
    )

    # 自定义日志
    result = train_autoencoder_standalone(
        rcs_data=rcs_data,
        param_data=param_data,
        training_config=config,
        log_callback=lambda msg: print(f"[LOG] {msg}"),
        progress_callback=lambda stage, epoch, total: print(f"{stage}: {epoch}/{total}")
    )
"""

import numpy as np
import torch
from typing import Dict, Any, Callable, Optional
from io import StringIO
import sys


def train_autoencoder_standalone(
    rcs_data: np.ndarray,
    param_data: np.ndarray,
    training_config: Dict[str, Any],
    ae_system: Optional[Dict] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    silent: bool = False
) -> Dict[str, Any]:
    """
    独立的AutoEncoder训练函数（无GUI依赖）

    Args:
        rcs_data: RCS数据 [N, 91, 91, num_freq]
        param_data: 参数数据 [N, num_params]
        training_config: 完整训练配置字典，包含所有超参数
            必需参数:
                - mode: 'wavelet' | 'direct' | 'differentiable_wavelet'
                - architecture: 'cnn' | 'mlp' | 'enhanced_cnn' | ...
                - latent_dim: int (隐空间维度)
                - batch_size: int
                - learning_rate: float
                - epochs: {'stage1': int, 'stage2': int, 'stage3': int}
                - training_mode: 'three_stage' | 'stage1_only'
            可选参数（会使用默认值）:
                - activation: str (默认'relu')
                - normalization_method: str (默认'zscore')
                - db_transform: bool (默认True)
                - wavelet_type: str (默认'db4')
                - dropout_rate: float (默认0.2)
                - lr_scheduler: str (默认'plateau')
                - optimizer_type: str (默认'adam')
                - ... (任何其他参数都会传递给训练器)

        ae_system: 可选，外部提供的AutoEncoder系统
            如果不提供，会根据training_config自动创建

        log_callback: 可选，日志回调函数
            签名: log_callback(message: str) -> None
            默认: print

        progress_callback: 可选，进度回调函数
            签名: progress_callback(stage_name: str, current_epoch: int, total_epochs: int) -> None
            默认: None

        silent: 是否静默模式（完全不输出日志）

    Returns:
        {
            'ae_system': Dict,  # 训练后的AutoEncoder系统
            'training_history': Dict,  # 训练历史
            'final_metrics': Dict,  # 最终指标
            'success': bool,  # 训练是否成功
            'error': str  # 错误信息（如果失败）
        }

    Raises:
        ValueError: 配置参数不合法
        RuntimeError: 训练过程中出错
    """
    # ===== 1. 参数验证 =====
    _validate_inputs(rcs_data, param_data, training_config)

    # ===== 2. 初始化日志系统 =====
    log_fn = _create_logger(log_callback, silent)

    try:
        log_fn("🚀 初始化独立训练环境...")

        # ===== 3. 创建或使用AutoEncoder系统 =====
        if ae_system is None:
            log_fn("📦 根据配置创建AutoEncoder系统...")
            ae_system = _create_ae_system_from_config(training_config)
            log_fn(f"  ✓ AutoEncoder系统已创建")
            log_fn(f"    - 模式: {training_config['mode']}")
            log_fn(f"    - 架构: {training_config['architecture']}")
            log_fn(f"    - 隐空间维度: {training_config['latent_dim']}")
        else:
            log_fn("📦 使用外部提供的AutoEncoder系统")

        # ===== 4. 初始化data_adapter统计信息 =====
        log_fn("📊 初始化数据预处理统计...")
        _initialize_data_adapter(ae_system, rcs_data)

        # ===== 5. 创建最小化GUI对象 =====
        log_fn("🔧 创建训练环境...")
        gui = _create_minimal_gui(ae_system, log_fn, progress_callback)

        # ===== 6. 执行训练 =====
        log_fn("=" * 80)
        log_fn("开始训练...")
        log_fn("=" * 80)

        from gui_managers.trainers.ae_trainer import AETrainer
        trainer = AETrainer(gui)

        # 执行三阶段训练（或Stage1 Only）
        training_history = trainer.run_three_stage_training(
            rcs_data=rcs_data,
            param_data=param_data,
            training_config=training_config
        )

        log_fn("=" * 80)
        log_fn("✅ 训练完成！")
        log_fn("=" * 80)

        # ===== 7. 提取最终指标 =====
        final_metrics = _extract_final_metrics(training_history)

        log_fn("\n📊 最终指标:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                log_fn(f"  {key}: {value:.6f}")
            else:
                log_fn(f"  {key}: {value}")

        # ===== 8. 返回结果 =====
        return {
            'ae_system': ae_system,
            'training_history': training_history,
            'final_metrics': final_metrics,
            'success': True,
            'error': ''
        }

    except Exception as e:
        error_msg = f"训练失败: {str(e)}"
        log_fn(f"❌ {error_msg}")

        import traceback
        if not silent:
            traceback.print_exc()

        return {
            'ae_system': ae_system if 'ae_system' in locals() else None,
            'training_history': None,
            'final_metrics': {},
            'success': False,
            'error': error_msg
        }


def _validate_inputs(rcs_data: np.ndarray, param_data: np.ndarray, config: Dict[str, Any]):
    """验证输入参数"""
    # 检查数据类型
    if not isinstance(rcs_data, np.ndarray):
        raise ValueError(f"rcs_data必须是numpy数组，当前类型: {type(rcs_data)}")

    if not isinstance(param_data, np.ndarray):
        raise ValueError(f"param_data必须是numpy数组，当前类型: {type(param_data)}")

    # 检查数据形状
    if len(rcs_data) != len(param_data):
        raise ValueError(
            f"rcs_data和param_data样本数量不一致: "
            f"rcs={len(rcs_data)}, param={len(param_data)}"
        )

    if len(rcs_data) == 0:
        raise ValueError("数据集为空")

    # 检查必需配置参数
    required_params = ['mode', 'architecture', 'latent_dim', 'batch_size',
                      'learning_rate', 'epochs', 'training_mode']

    missing = [p for p in required_params if p not in config]
    if missing:
        raise ValueError(f"training_config缺少必需参数: {missing}")

    # 检查epochs格式
    if not isinstance(config['epochs'], dict):
        raise ValueError(f"epochs必须是字典格式，当前类型: {type(config['epochs'])}")


def _create_logger(log_callback: Optional[Callable], silent: bool) -> Callable:
    """创建日志函数"""
    if silent:
        return lambda msg: None
    elif log_callback is not None:
        return log_callback
    else:
        return print


def _create_ae_system_from_config(config: Dict[str, Any]) -> Dict:
    """
    根据配置创建AutoEncoder系统

    ⚠️ 核心可扩展性：所有参数都通过**kwargs传递，无需硬编码
    """
    from autoencoder.utils.frequency_config import create_autoencoder_system

    # 提取create_autoencoder_system的参数
    ae_params = {
        'config_name': '2freq',  # 固定使用2freq
        'mode': config['mode'],
        'architecture': config['architecture'],
        'latent_dim': config['latent_dim'],
        'dropout_rate': config.get('dropout_rate', 0.2),
        'activation': config.get('activation', 'relu'),
        'normalization_method': config.get('normalization_method', 'zscore'),
        'db_transform': config.get('db_transform', True),
        'wavelet': config.get('wavelet_type', 'db4'),  # 注意：参数名是wavelet而非wavelet_type
    }

    # 可选参数（如果config中有，则传递）
    optional_params = [
        'channel_attention', 'll_ratio', 'hf_ratio',
        'use_batch_norm', 'use_layer_norm'
    ]

    for param in optional_params:
        if param in config:
            ae_params[param] = config[param]

    # 创建系统
    ae_system = create_autoencoder_system(**ae_params)

    # 添加训练模式标记
    ae_system['training_mode'] = config.get('training_mode', 'three_stage')

    return ae_system


def _initialize_data_adapter(ae_system: Dict, rcs_data: np.ndarray):
    """
    初始化data_adapter的统计信息

    ⚠️ 重要：RCS_DataAdapter通过adapt_rcs_data自动计算统计信息
    这里需要根据模式预处理数据，触发统计计算
    """
    data_adapter = ae_system.get('data_adapter', None)
    if data_adapter is None:
        return

    mode = ae_system.get('mode', 'direct')

    # 根据模式决定预处理的数据
    if mode == 'wavelet':
        # Wavelet模式: 先小波变换，再调用adapt_rcs_data计算统计
        wavelet_transform = ae_system.get('wavelet_transform', None)
        if wavelet_transform:
            import torch
            rcs_tensor = torch.FloatTensor(rcs_data)
            wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
            # 调用adapt_rcs_data会自动计算并保存统计信息到data_stats
            _ = data_adapter.adapt_rcs_data(wavelet_coeffs.numpy())
        else:
            # 如果没有wavelet_transform，直接处理RCS
            _ = data_adapter.adapt_rcs_data(rcs_data)
    else:
        # Direct/Differentiable模式: 直接处理RCS数据，自动计算统计
        _ = data_adapter.adapt_rcs_data(rcs_data)


def _create_minimal_gui(ae_system: Dict, log_fn: Callable,
                       progress_callback: Optional[Callable]) -> object:
    """
    创建最小化GUI对象（满足AETrainer的接口要求）

    AETrainer需要的GUI接口:
    - ae_system: AutoEncoder系统
    - ae_training_history: 训练历史
    - ae_log(message): 日志函数
    - batch_experiment_mode: 批量模式标志（禁用弹窗）
    """
    class MinimalGUI:
        def __init__(self):
            self.ae_system = ae_system
            self.ae_training_history = {}
            self.batch_experiment_mode = True  # 禁用messagebox弹窗
            self._log_fn = log_fn
            self._progress_callback = progress_callback

        def ae_log(self, message: str):
            """日志输出"""
            self._log_fn(message)

        def update_progress(self, stage: str, epoch: int, total: int):
            """进度更新（可选）"""
            if self._progress_callback:
                self._progress_callback(stage, epoch, total)

    return MinimalGUI()


def _extract_final_metrics(training_history: Dict) -> Dict[str, Any]:
    """
    从训练历史中提取最终指标

    Returns:
        {
            'best_stage1_val_loss': float,
            'best_stage2_val_loss': float,
            'best_stage3_val_loss': float,
            'stage1_epochs': int,
            'stage2_epochs': int,
            'stage3_epochs': int,
            'training_mode': str
        }
    """
    if not training_history:
        return {}

    metrics = {}
    metrics['training_mode'] = training_history.get('training_mode', 'unknown')

    stage_histories = training_history.get('stage_histories', {})

    # 提取各阶段最佳损失
    for stage_name in ['stage1', 'stage2', 'stage3']:
        if stage_name in stage_histories:
            stage_hist = stage_histories[stage_name]
            metrics[f'best_{stage_name}_val_loss'] = stage_hist.get('best_val_loss', float('inf'))
            metrics[f'{stage_name}_epochs'] = len(stage_hist.get('train_losses', []))
            metrics[f'best_{stage_name}_epoch'] = stage_hist.get('best_epoch', -1)

    return metrics


# ===== 便捷函数 =====

def train_from_json(
    json_path: str,
    rcs_data: np.ndarray,
    param_data: np.ndarray,
    experiment_index: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    从JSON配置文件训练单个实验

    Args:
        json_path: JSON配置文件路径
        rcs_data: RCS数据
        param_data: 参数数据
        experiment_index: 使用JSON中的第几个实验配置（默认第一个）
        **kwargs: 传递给train_autoencoder_standalone的其他参数

    Returns:
        训练结果字典
    """
    from autoencoder.utils.json_experiment import load_experiments_from_json

    exp_config = load_experiments_from_json(json_path)

    if experiment_index >= len(exp_config['experiments']):
        raise IndexError(
            f"experiment_index={experiment_index}超出范围，"
            f"总共{len(exp_config['experiments'])}个实验"
        )

    config = exp_config['experiments'][experiment_index]

    print(f"从JSON加载实验配置: {config['experiment_id']}")

    return train_autoencoder_standalone(
        rcs_data=rcs_data,
        param_data=param_data,
        training_config=config,
        **kwargs
    )


def batch_train_from_json(
    json_path: str,
    rcs_data: np.ndarray,
    param_data: np.ndarray,
    save_dir: str = 'batch_experiments',
    log_callback: Optional[Callable] = None,
    **kwargs
) -> list:
    """
    从JSON配置文件批量训练所有实验

    Args:
        json_path: JSON配置文件路径
        rcs_data: RCS数据
        param_data: 参数数据
        save_dir: 结果保存目录
        log_callback: 日志回调
        **kwargs: 传递给train_autoencoder_standalone的其他参数

    Returns:
        所有实验的结果列表
    """
    from autoencoder.utils.json_experiment import load_experiments_from_json, print_experiment_summary
    import os
    from datetime import datetime

    # 加载实验配置
    exp_config = load_experiments_from_json(json_path)
    print_experiment_summary(exp_config)

    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"{exp_config['experiment_name']}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"\n实验目录: {experiment_dir}\n")

    # 批量训练
    results = []
    total = len(exp_config['experiments'])

    for idx, config in enumerate(exp_config['experiments'], 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{total}] 实验: {config['experiment_id']}")
        print(f"{'='*80}")

        result = train_autoencoder_standalone(
            rcs_data=rcs_data,
            param_data=param_data,
            training_config=config,
            log_callback=log_callback,
            **kwargs
        )

        results.append(result)

        # 保存模型
        if result['success']:
            model_path = os.path.join(experiment_dir, f"{config['experiment_id']}.pth")
            torch.save(result['ae_system'], model_path)
            print(f"✓ 模型已保存: {model_path}")

    print(f"\n{'='*80}")
    print(f"批量训练完成！")
    print(f"成功: {sum(1 for r in results if r['success'])}/{total}")
    print(f"失败: {sum(1 for r in results if not r['success'])}/{total}")
    print(f"结果目录: {experiment_dir}")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    """测试独立训练函数"""
    print("独立训练函数测试\n")

    # 生成测试数据
    print("生成测试数据...")
    rcs_data = np.random.randn(100, 91, 91, 2)
    param_data = np.random.randn(100, 5)

    # 测试配置
    test_config = {
        'mode': 'wavelet',
        'architecture': 'cnn',
        'activation': 'relu',
        'latent_dim': 64,
        'batch_size': 8,
        'learning_rate': 0.001,
        'training_mode': 'stage1_only',
        'epochs': {'stage1': 2, 'stage2': 1, 'stage3': 1},  # 快速测试
        'normalization_method': 'zscore',
        'db_transform': True,
        'wavelet_type': 'db4',
        'dropout_rate': 0.2
    }

    print("开始训练...")
    result = train_autoencoder_standalone(
        rcs_data=rcs_data,
        param_data=param_data,
        training_config=test_config,
        log_callback=lambda msg: print(f"  {msg}")
    )

    print(f"\n训练结果:")
    print(f"  成功: {result['success']}")
    if result['success']:
        print(f"  最终指标: {result['final_metrics']}")
    else:
        print(f"  错误: {result['error']}")

    print("\n✅ 测试完成！")
