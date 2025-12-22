"""
AutoEncoder模型的保存和加载功能

提供纯函数式的模型I/O接口，供GUI和批量实验复用，保证保存格式的一致性。

核心功能：
1. save_ae_model_to_file() - 保存模型到.pth文件
2. load_ae_model_from_file() - 从.pth文件加载模型
3. validate_model_config() - 验证模型配置
4. restore_data_adapter_stats() - 恢复data_adapter统计信息

设计原则：
- 纯函数设计，无副作用
- 统一的保存格式（GUI和批量实验使用相同格式）
- 完整的错误处理和验证

作者：Claude Code
日期：2025-01-10
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from datetime import datetime


def create_model_state_dict(
    ae_system: Dict,
    training_history: Optional[Dict] = None,
    training_mode: str = 'three_stage'
) -> Dict[str, Any]:
    """
    创建模型状态字典（统一格式）

    Args:
        ae_system: AutoEncoder系统字典，必须包含:
            - 'autoencoder': AutoEncoder模型实例
            - 'parameter_mapper': 参数映射器实例
            - 'config_info': 配置信息字典（可选）
            - 'data_adapter': 数据适配器实例（可选）
        training_history: 训练历史字典（可选），包含训练损失曲线等
        training_mode: 训练模式 ('three_stage' 或 'stage1_only')

    Returns:
        模型状态字典，包含以下键：
            - 'autoencoder': AutoEncoder的state_dict
            - 'parameter_mapper': ParameterMapper的state_dict
            - 'config': 配置信息字典
            - 'training_mode': 训练模式
            - 'adapter_stats': data_adapter统计信息（如果存在）
            - 'training_history': 训练历史（如果提供）

    注意：
        - 统一使用 'autoencoder' 和 'parameter_mapper' 作为键名
        - 与GUI的save_ae_model()格式完全一致
        - numpy数组会自动转换为列表以便JSON序列化
    """
    # 基本模型权重（统一键名）
    model_state = {
        'autoencoder': ae_system['autoencoder'].state_dict(),
        'parameter_mapper': ae_system['parameter_mapper'].state_dict(),
        'config': ae_system.get('config_info', {}),
        'training_mode': training_mode
    }

    # 保存data_adapter统计信息
    data_adapter = ae_system.get('data_adapter', None)
    if data_adapter and hasattr(data_adapter, 'data_stats') and data_adapter.data_stats:
        # 将numpy数组转换为列表以便JSON序列化
        adapter_stats = {}
        for key, value in data_adapter.data_stats.items():
            if hasattr(value, 'tolist'):
                adapter_stats[key] = value.tolist()
            else:
                adapter_stats[key] = value
        model_state['adapter_stats'] = adapter_stats

    # 保存param_scaler统计信息
    param_scaler = ae_system.get('param_scaler', None)
    if param_scaler is not None:
        # 保存StandardScaler的统计信息（mean_和scale_）
        param_scaler_stats = {
            'mean': param_scaler.mean_.tolist() if hasattr(param_scaler, 'mean_') else None,
            'scale': param_scaler.scale_.tolist() if hasattr(param_scaler, 'scale_') else None,
            'n_features_in': int(param_scaler.n_features_in_) if hasattr(param_scaler, 'n_features_in_') else None
        }
        model_state['param_scaler_stats'] = param_scaler_stats

    # 保存训练历史
    if training_history:
        model_state['training_history'] = training_history

    # 保存Loss归一化系数（如果存在）
    loss_normalization_factor = ae_system.get('loss_normalization_factor', None)
    if loss_normalization_factor is not None:
        model_state['loss_normalization_factor'] = float(loss_normalization_factor)

    # 保存ParameterMapper配置（关键！用于正确重建mapper）
    parameter_mapper = ae_system.get('parameter_mapper', None)
    if parameter_mapper is not None and hasattr(parameter_mapper, 'hidden_dims'):
        mapper_config = {
            'param_dim': getattr(parameter_mapper, 'param_dim', 9),
            'latent_dim': getattr(parameter_mapper, 'latent_dim', 256),
            'hidden_dims': getattr(parameter_mapper, 'hidden_dims', None),
            'dropout_rate': getattr(parameter_mapper, 'dropout_rate', 0.3),
            'activation': getattr(parameter_mapper, 'activation_name', 'relu'),
            'use_adaptive': getattr(parameter_mapper, 'use_adaptive', False)
        }
        model_state['config']['mapper_config'] = mapper_config

    return model_state


def save_ae_model_to_file(
    ae_system: Dict,
    model_path: str,
    training_history: Optional[Dict] = None,
    training_mode: str = 'three_stage',
    save_json_config: bool = True,
    config_override: Optional[Dict] = None
) -> None:
    """
    保存AutoEncoder模型到文件（统一保存函数）

    Args:
        ae_system: AutoEncoder系统字典
        model_path: 模型保存路径（.pth文件）
        training_history: 训练历史（可选）
        training_mode: 训练模式 ('three_stage' 或 'stage1_only')
        save_json_config: 是否同时保存JSON配置文件
        config_override: 额外的配置信息，会合并到config中（可选）

    Raises:
        ValueError: 如果ae_system缺少必需字段
        IOError: 如果文件保存失败

    示例:
        >>> save_ae_model_to_file(
        ...     ae_system=my_ae_system,
        ...     model_path='model.pth',
        ...     training_history=history,
        ...     config_override={'latent_dim': 256}
        ... )
    """
    # 验证必需字段
    required_keys = ['autoencoder', 'parameter_mapper']
    for key in required_keys:
        if key not in ae_system:
            raise ValueError(f"ae_system缺少必需字段: '{key}'")

    # 创建模型状态字典
    model_state = create_model_state_dict(ae_system, training_history, training_mode)

    # 覆盖配置（如果提供）
    if config_override:
        model_state['config'].update(config_override)

    # 保存.pth文件
    try:
        torch.save(model_state, model_path)
    except Exception as e:
        raise IOError(f"保存模型文件失败: {e}")

    # 保存JSON配置文件
    if save_json_config:
        config_path = os.path.splitext(model_path)[0] + "_config.json"
        try:
            save_model_config_json(model_state, config_path)
        except Exception as e:
            # JSON保存失败不影响主流程，只记录警告
            print(f"警告：保存JSON配置文件失败: {e}")


def save_model_config_json(
    model_state: Dict,
    json_path: str
) -> None:
    """
    保存模型配置到JSON文件（人类可读格式）

    Args:
        model_state: 模型状态字典（来自create_model_state_dict）
        json_path: JSON文件保存路径

    注意：
        - 只保存配置信息，不保存模型权重
        - 用于快速查看模型配置，无需加载.pth文件
    """
    # 构建可读的配置信息（不包含state_dict等二进制数据）
    config = model_state.get('config', {})

    # 提取并简化训练历史（仅保留关键指标，避免文件过大）
    full_history = model_state.get('training_history', {})
    history_summary = {}

    if full_history:
        # 保留训练配置
        if 'training_config' in full_history:
            history_summary['training_config'] = full_history['training_config']

        # 简化各阶段历史（只保留最佳值和最终值）
        if 'stage_histories' in full_history:
            history_summary['stage_histories'] = {}
            for stage_name, stage_data in full_history['stage_histories'].items():
                summary = {
                    'best_val_loss': stage_data.get('best_val_loss', 'N/A'),
                    'best_epoch': stage_data.get('best_epoch', 'N/A')
                }
                # 提取最终loss
                if 'loss_history' in stage_data and stage_data['loss_history']:
                    summary['final_loss'] = stage_data['loss_history'][-1]
                if 'val_loss_history' in stage_data and stage_data['val_loss_history']:
                    summary['final_val_loss'] = stage_data['val_loss_history'][-1]
                
                history_summary['stage_histories'][stage_name] = summary

    json_data = {
        'model_info': {
            'mode': config.get('mode', 'N/A'),
            'architecture': config.get('architecture', 'N/A'),
            'latent_dim': config.get('latent_dim', 'N/A'),
            'dropout_rate': config.get('dropout_rate', 0.2),
            'activation': config.get('activation', 'relu')
        },
        'frequency_config': {
            'num_frequencies': config.get('num_frequencies', 'N/A'),
            'frequency_labels': config.get('frequency_labels', []),
            'frequencies_ghz': config.get('frequencies_ghz', [])
        },
        'data_preprocessing': {
            'wavelet_type': config.get('wavelet', 'N/A'),
            'normalize': config.get('normalize', False),
            'db_transform': config.get('db_transform', False)
        },
        'training_info': {
            'training_mode': model_state.get('training_mode', 'N/A'),
            'training_history_summary': history_summary,
            'loss_normalization_factor': model_state.get('loss_normalization_factor', 'N/A')
        },
        'save_info': {
            'save_time': datetime.now().isoformat(),
            'file_path': json_path
        }
    }

    # 添加data_adapter统计信息
    if 'adapter_stats' in model_state:
        json_data['data_preprocessing']['adapter_stats'] = model_state['adapter_stats']

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def load_ae_model_from_file(
    model_path: str,
    device: str = 'cpu'
) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    从文件加载AutoEncoder模型状态（统一加载函数）

    Args:
        model_path: 模型文件路径（.pth文件）
        device: 设备 ('cpu' 或 'cuda')

    Returns:
        (checkpoint, config, training_history) 三元组:
            - checkpoint: 原始checkpoint字典，包含 'autoencoder' 和 'parameter_mapper' state_dict
            - config: 配置信息字典
            - training_history: 训练历史（可能为None）

    Raises:
        FileNotFoundError: 如果文件不存在
        KeyError: 如果文件缺少必需字段
        RuntimeError: 如果文件损坏或格式错误

    示例:
        >>> checkpoint, config, history = load_ae_model_from_file('model.pth')
        >>> ae_system['autoencoder'].load_state_dict(checkpoint['autoencoder'])
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"加载模型文件失败: {e}")

    # 验证必需字段
    required_keys = ['autoencoder', 'parameter_mapper']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"模型文件缺少必需字段: '{key}'。"
                f"文件中的键: {list(checkpoint.keys())}"
            )

    # 提取配置信息
    config = checkpoint.get('config', {})

    # 提取训练历史
    training_history = checkpoint.get('training_history', None)

    return checkpoint, config, training_history


def validate_model_config(
    config: Dict,
    expected_config: Optional[Dict] = None
) -> Tuple[bool, str]:
    """
    验证模型配置的完整性和一致性

    Args:
        config: 待验证的配置字典
        expected_config: 期望的配置（可选，用于一致性检查）

    Returns:
        (is_valid, error_message) 二元组:
            - is_valid: True表示验证通过，False表示验证失败
            - error_message: 如果验证失败，包含错误信息；否则为空字符串

    示例:
        >>> is_valid, msg = validate_model_config(config)
        >>> if not is_valid:
        ...     print(f"配置无效: {msg}")
    """
    # 检查必需字段
    required_fields = ['mode', 'architecture', 'latent_dim', 'num_frequencies']
    for field in required_fields:
        if field not in config:
            return False, f"配置缺少必需字段: '{field}'"

    # 验证字段值的合法性
    valid_modes = ['wavelet', 'direct']
    if config['mode'] not in valid_modes:
        return False, f"无效的mode: {config['mode']}（有效值: {valid_modes}）"

    if config['latent_dim'] <= 0:
        return False, f"无效的latent_dim: {config['latent_dim']}（必须>0）"

    if config['num_frequencies'] <= 0:
        return False, f"无效的num_frequencies: {config['num_frequencies']}（必须>0）"

    # 如果提供了expected_config，进行一致性检查
    if expected_config:
        critical_fields = ['mode', 'num_frequencies', 'latent_dim']
        for field in critical_fields:
            config_value = config.get(field)
            expected_value = expected_config.get(field)
            if config_value != expected_value:
                return False, (
                    f"配置不匹配: {field} "
                    f"(模型: {config_value}, 期望: {expected_value})"
                )

    return True, ""


def restore_data_adapter_stats(
    data_adapter,
    adapter_stats: Dict
) -> None:
    """
    从保存的统计信息恢复data_adapter状态

    Args:
        data_adapter: RCS_DataAdapter实例
        adapter_stats: 保存的统计信息字典（来自checkpoint['adapter_stats']）

    注意：
        - 会将列表转换回numpy数组
        - 直接修改data_adapter.data_stats
        - 恢复后data_adapter可以正确执行inverse_adapt()

    示例:
        >>> checkpoint, config, _ = load_ae_model_from_file('model.pth')
        >>> if 'adapter_stats' in checkpoint:
        ...     restore_data_adapter_stats(data_adapter, checkpoint['adapter_stats'])
    """
    if not hasattr(data_adapter, 'data_stats'):
        data_adapter.data_stats = {}

    # 将列表转换回numpy数组
    for key, value in adapter_stats.items():
        if isinstance(value, list):
            data_adapter.data_stats[key] = np.array(value)
        else:
            data_adapter.data_stats[key] = value


def get_model_info_summary(model_path: str) -> str:
    """
    获取模型信息摘要（无需完全加载模型）

    Args:
        model_path: 模型文件路径

    Returns:
        模型信息摘要字符串

    示例:
        >>> print(get_model_info_summary('model.pth'))
        模式: wavelet | 架构: cnn | 隐空间: 256 | 频率: 2
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config', {})

        mode = config.get('mode', 'N/A')
        arch = config.get('architecture', 'N/A')
        latent = config.get('latent_dim', 'N/A')
        nfreq = config.get('num_frequencies', 'N/A')
        training_mode = checkpoint.get('training_mode', 'N/A')

        return (
            f"模式: {mode} | 架构: {arch} | 隐空间: {latent} | "
            f"频率: {nfreq} | 训练模式: {training_mode}"
        )
    except Exception as e:
        return f"无法读取模型信息: {e}"


# ========== 实用工具函数 ==========

def is_compatible_model(
    model_path: str,
    expected_mode: str,
    expected_num_frequencies: int
) -> bool:
    """
    检查模型是否与当前数据兼容

    Args:
        model_path: 模型文件路径
        expected_mode: 期望的模式 ('wavelet' 或 'direct')
        expected_num_frequencies: 期望的频率数量

    Returns:
        True表示兼容，False表示不兼容

    示例:
        >>> if is_compatible_model('model.pth', 'wavelet', 2):
        ...     print("模型与当前数据兼容")
    """
    try:
        _, config, _ = load_ae_model_from_file(model_path, device='cpu')
        return (
            config.get('mode') == expected_mode and
            config.get('num_frequencies') == expected_num_frequencies
        )
    except Exception:
        return False
