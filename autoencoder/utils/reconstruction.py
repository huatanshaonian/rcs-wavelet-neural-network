"""
AutoEncoder重建功能

提供纯函数式的RCS重建接口，供批量实验复用。

核心功能：
1. reconstruct_from_params() - 从参数预测RCS（Three-Stage模式）
2. reconstruct_from_rcs() - 从RCS重建RCS（Stage1-Only模式）

注意：
    这个模块提供简化的重建接口，专门用于批量实验等场景。
    GUI的完整重建功能在 reconstruction_manager.py 中。

作者：Claude Code
日期：2025-01-10
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def reconstruct_from_params(
    ae_system: Dict,
    params: np.ndarray,
    device: str = 'cpu'
) -> np.ndarray:
    """
    从参数预测RCS（Three-Stage模式）

    Args:
        ae_system: AutoEncoder系统字典，包含:
            - 'autoencoder': AutoEncoder模型
            - 'parameter_mapper': 参数映射器
            - 'data_adapter': 数据适配器（可选）
            - 'wavelet_transform': 小波变换器（Wavelet模式）
            - 'mode': 'wavelet' 或 'direct'
        params: 参数数据 [N, param_dim]
        device: 'cpu' 或 'cuda'

    Returns:
        predicted_rcs: 预测的RCS数据 [N, 91, 91, num_freq]（线性域）

    流程：
        参数 → ParameterMapper → Latent → Decoder → 标准化输出
        → 逆标准化 → (逆小波变换) → RCS

    示例:
        >>> pred_rcs = reconstruct_from_params(ae_system, test_params)
        >>> print(f"预测RCS形状: {pred_rcs.shape}")
    """
    # 获取系统组件
    autoencoder = ae_system['autoencoder']
    parameter_mapper = ae_system['parameter_mapper']
    data_adapter = ae_system.get('data_adapter', None)
    wavelet_transform = ae_system.get('wavelet_transform', None)
    mode = ae_system.get('mode', 'wavelet')
    param_scaler = ae_system.get('param_scaler', None)

    # 设置为评估模式
    autoencoder.to(device).eval()
    parameter_mapper.to(device).eval()

    with torch.no_grad():
        # Step 0: 标准化参数（如果有scaler）
        if param_scaler is not None:
            params_normalized = param_scaler.transform(params)
        else:
            # 向后兼容：旧模型没有param_scaler
            params_normalized = params

        # Step 1: 参数 → ParameterMapper → Latent
        param_tensor = torch.FloatTensor(params_normalized).to(device)
        predicted_latents = parameter_mapper(param_tensor)

        # Step 2: Latent → Decoder → 标准化输出
        decoder_output = autoencoder.decode(predicted_latents)

        # Step 3: 逆预处理到RCS空间
        if mode == 'wavelet':
            # 小波模式：标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
            if data_adapter:
                # 逆标准化（逆dB + 逆Z-score）
                predicted_coeffs_np = data_adapter.inverse_adapt(decoder_output)
                predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
            else:
                predicted_coeffs = decoder_output

            # 逆小波变换
            if wavelet_transform is None:
                raise ValueError("Wavelet模式需要wavelet_transform")
            predicted_rcs = wavelet_transform.inverse_transform(predicted_coeffs)

        else:
            # Direct模式：标准化RCS → 逆标准化 → RCS
            if data_adapter:
                predicted_rcs_np = data_adapter.inverse_adapt(decoder_output)
                predicted_rcs = torch.FloatTensor(predicted_rcs_np).to(device)
            else:
                predicted_rcs = decoder_output

        # 转换为numpy
        predicted_rcs_np = predicted_rcs.cpu().numpy()

    return predicted_rcs_np


def reconstruct_from_rcs(
    ae_system: Dict,
    rcs_data: np.ndarray,
    device: str = 'cpu'
) -> np.ndarray:
    """
    从RCS重建RCS（Stage1-Only模式）

    Args:
        ae_system: AutoEncoder系统字典
        rcs_data: RCS数据 [N, 91, 91, num_freq]
        device: 'cpu' 或 'cuda'

    Returns:
        reconstructed_rcs: 重建的RCS数据 [N, 91, 91, num_freq]

    流程：
        RCS → (小波变换) → 标准化 → Encoder → Latent → Decoder
        → 逆标准化 → (逆小波变换) → RCS

    示例:
        >>> recon_rcs = reconstruct_from_rcs(ae_system, test_rcs)
        >>> print(f"重建RCS形状: {recon_rcs.shape}")
    """
    # 获取系统组件
    autoencoder = ae_system['autoencoder']
    data_adapter = ae_system.get('data_adapter', None)
    wavelet_transform = ae_system.get('wavelet_transform', None)
    mode = ae_system.get('mode', 'wavelet')

    # 设置为评估模式
    autoencoder.to(device).eval()

    with torch.no_grad():
        # Step 1: 预处理输入
        if mode == 'wavelet':
            # RCS → 小波变换 → 标准化
            rcs_tensor = torch.FloatTensor(rcs_data).to(device)
            if wavelet_transform is None:
                raise ValueError("Wavelet模式需要wavelet_transform")
            wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)

            if data_adapter:
                input_data = data_adapter.adapt_rcs_data(wavelet_coeffs.cpu().numpy())
                input_tensor = torch.FloatTensor(input_data).to(device)
            else:
                input_tensor = wavelet_coeffs

        else:
            # Direct模式：RCS → 标准化
            if data_adapter:
                input_data = data_adapter.adapt_rcs_data(rcs_data)
                input_tensor = torch.FloatTensor(input_data).to(device)
            else:
                input_tensor = torch.FloatTensor(rcs_data).to(device)

        # Step 2: Encoder → Latent → Decoder
        latents = autoencoder.encode(input_tensor)
        decoder_output = autoencoder.decode(latents)

        # Step 3: 逆预处理到RCS空间
        if mode == 'wavelet':
            # 逆标准化 → 逆小波变换
            if data_adapter:
                reconstructed_coeffs_np = data_adapter.inverse_adapt(decoder_output)
                reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
            else:
                reconstructed_coeffs = decoder_output

            reconstructed_rcs = wavelet_transform.inverse_transform(reconstructed_coeffs)

        else:
            # Direct模式：逆标准化
            if data_adapter:
                reconstructed_rcs_np = data_adapter.inverse_adapt(decoder_output)
                reconstructed_rcs = torch.FloatTensor(reconstructed_rcs_np).to(device)
            else:
                reconstructed_rcs = decoder_output

        # 转换为numpy
        reconstructed_rcs_np = reconstructed_rcs.cpu().numpy()

    return reconstructed_rcs_np


def reconstruct_with_latents(
    ae_system: Dict,
    params: np.ndarray,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从参数预测RCS，同时返回隐空间表示

    Args:
        ae_system: AutoEncoder系统字典
        params: 参数数据 [N, param_dim]
        device: 'cpu' 或 'cuda'

    Returns:
        (predicted_rcs, latents):
            - predicted_rcs: [N, 91, 91, num_freq]
            - latents: [N, latent_dim]

    示例:
        >>> pred_rcs, latents = reconstruct_with_latents(ae_system, params)
        >>> print(f"隐空间维度: {latents.shape[1]}")
    """
    # 获取系统组件
    autoencoder = ae_system['autoencoder']
    parameter_mapper = ae_system['parameter_mapper']
    data_adapter = ae_system.get('data_adapter', None)
    wavelet_transform = ae_system.get('wavelet_transform', None)
    mode = ae_system.get('mode', 'wavelet')
    param_scaler = ae_system.get('param_scaler', None)

    # 设置为评估模式
    autoencoder.to(device).eval()
    parameter_mapper.to(device).eval()

    with torch.no_grad():
        # Step 0: 标准化参数（如果有scaler）
        if param_scaler is not None:
            params_normalized = param_scaler.transform(params)
        else:
            # 向后兼容：旧模型没有param_scaler
            params_normalized = params

        # Step 1: 参数 → Latent
        param_tensor = torch.FloatTensor(params_normalized).to(device)
        predicted_latents = parameter_mapper(param_tensor)
        latents_np = predicted_latents.cpu().numpy()

        # Step 2: Latent → RCS（复用 reconstruct_from_params 的逻辑）
        decoder_output = autoencoder.decode(predicted_latents)

        # Step 3: 逆预处理
        if mode == 'wavelet':
            if data_adapter:
                predicted_coeffs_np = data_adapter.inverse_adapt(decoder_output)
                predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
            else:
                predicted_coeffs = decoder_output

            if wavelet_transform is None:
                raise ValueError("Wavelet模式需要wavelet_transform")
            predicted_rcs = wavelet_transform.inverse_transform(predicted_coeffs)
        else:
            if data_adapter:
                predicted_rcs_np = data_adapter.inverse_adapt(decoder_output)
                predicted_rcs = torch.FloatTensor(predicted_rcs_np).to(device)
            else:
                predicted_rcs = decoder_output

        predicted_rcs_np = predicted_rcs.cpu().numpy()

    return predicted_rcs_np, latents_np


# ========== 辅助函数 ==========

def get_device_from_model(ae_system: Dict) -> str:
    """
    从模型推断设备

    Args:
        ae_system: AutoEncoder系统字典

    Returns:
        'cpu' 或 'cuda'
    """
    autoencoder = ae_system['autoencoder']
    device = next(autoencoder.parameters()).device
    return str(device)


def validate_ae_system(ae_system: Dict, training_mode: str = 'three_stage') -> None:
    """
    验证ae_system是否包含必要组件

    Args:
        ae_system: AutoEncoder系统字典
        training_mode: 训练模式

    Raises:
        ValueError: 如果缺少必要组件
    """
    required_keys = ['autoencoder', 'mode']

    if training_mode in ('three_stage', 'joint_training'):
        required_keys.append('parameter_mapper')

    for key in required_keys:
        if key not in ae_system:
            raise ValueError(f"ae_system缺少必需组件: '{key}'")

    mode = ae_system['mode']
    if mode == 'wavelet' and 'wavelet_transform' not in ae_system:
        raise ValueError("Wavelet模式需要wavelet_transform")
