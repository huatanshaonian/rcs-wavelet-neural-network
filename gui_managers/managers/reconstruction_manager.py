"""
重建管理器
处理RCS重建相关功能
"""

import torch
import numpy as np


class ReconstructionManager:
    """重建管理器 - 负责RCS重建功能"""

    def __init__(self, parent_gui):
        """
        初始化重建管理器

        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui

    def _reconstruct_rcs(self, input_data=None, input_type='auto', model_ids=None, return_latents=False, return_wavelet_coeffs=False):
        """
        统一的RCS重建函数 - 支持多种输入方式

        根据训练模式自动选择重建路径：
        - Three-Stage模式: 参数 → ParameterMapper → Decoder → RCS
        - Stage1-Only模式: RCS → Encoder → Decoder → RCS

        Args:
            input_data: 输入数据（可选）
                - 如果input_type='params': np.ndarray [N, 9] 设计参数
                - 如果input_type='rcs': np.ndarray [N, H, W, C] RCS数据
                - 如果input_type='auto': 根据training_mode自动推断
            input_type: 输入类型
                - 'auto': 根据training_mode自动判断（默认）
                - 'params': 从参数重建（需要Three-Stage模式）
                - 'rcs': 从RCS重建（Stage1-Only模式）
                - 'model_ids': 从模型ID列表重建
            model_ids: 模型ID列表（当input_type='model_ids'时使用）
                - 可以是字符串列表 ['001', '002'] 或整数列表 [0, 1]
            return_latents: 是否返回隐空间表示（默认False）
            return_wavelet_coeffs: 是否返回小波系数（仅Wavelet模式，默认False）

        Returns:
            dict: {
                'reconstructed_rcs': np.ndarray [N, 91, 91, num_freq] 重建的RCS（线性域）
                'latents': np.ndarray [N, latent_dim] 隐空间表示（如果return_latents=True）
                'original_wavelet_coeffs': np.ndarray [N, 49, 49, 8] 原始小波系数（如果return_wavelet_coeffs=True且mode='wavelet'）
                'reconstructed_wavelet_coeffs': np.ndarray [N, 49, 49, 8] 重建小波系数（如果return_wavelet_coeffs=True且mode='wavelet'）
                'input_type_used': str 实际使用的输入类型
                'training_mode': str 训练模式
            }
        """
        import torch
        import numpy as np

        # 1. 获取系统组件
        autoencoder = self.gui.ae_system['autoencoder']
        parameter_mapper = self.gui.ae_system['parameter_mapper']
        wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)
        data_adapter = self.gui.ae_system.get('data_adapter', None)
        mode = self.gui.ae_system.get('mode', 'wavelet')
        training_mode = self.gui.ae_system.get('training_mode', 'three_stage')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        autoencoder.to(device).eval()
        if training_mode in ('three_stage', 'joint_training'):
            parameter_mapper.to(device).eval()

        # 初始化小波系数变量
        original_wavelet_coeffs_np = None
        reconstructed_wavelet_coeffs_np = None

        # 2. 确定输入类型
        if input_type == 'auto':
            if training_mode == 'stage1_only':
                input_type = 'rcs'
            elif training_mode in ('three_stage', 'joint_training'):
                input_type = 'params'
            else:
                input_type = 'params'  # 默认从参数重建

        # 3. 处理model_ids输入（转换为实际数据）
        if input_type == 'model_ids':
            if model_ids is None:
                raise ValueError("input_type='model_ids' 需要提供model_ids参数")

            # 转换model_ids为索引
            indices = []
            for mid in model_ids:
                if isinstance(mid, str):
                    indices.append(int(mid) - 1)  # "001" → 0
                else:
                    indices.append(int(mid))

            # 根据training_mode获取对应数据
            if training_mode in ('three_stage', 'joint_training'):
                # Three-Stage/Joint-Training: 从参数重建
                param_data = self.gui.ae_system.get('param_data', None)
                if param_data is None:
                    param_data = self.gui.ae_system.get('parameter_data', None)
                if param_data is None and hasattr(self.gui, 'param_data') and self.gui.param_data is not None:
                    self.gui.ae_system['param_data'] = self.gui.param_data
                    param_data = self.gui.param_data
                if param_data is None:
                    raise KeyError("param_data 缺失：请先加载参数数据或在数据管理页创建系统")
                input_data = param_data[indices]
                input_type = 'params'
            else:
                # Stage1-Only: 从RCS重建
                rcs_data = self.gui.ae_system.get('rcs_data', None)
                if rcs_data is None and hasattr(self.gui, 'rcs_data') and self.gui.rcs_data is not None:
                    self.gui.ae_system['rcs_data'] = self.gui.rcs_data
                    rcs_data = self.gui.rcs_data
                if rcs_data is None:
                    raise KeyError("rcs_data 缺失：请先加载RCS数据或在数据管理页创建系统")
                input_data = rcs_data[indices]
                input_type = 'rcs'

        # 4. 验证输入数据
        if input_data is None:
            raise ValueError("必须提供input_data或model_ids")

        # 确保是numpy数组
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        # 5. 执行重建
        with torch.no_grad():
            if input_type == 'params':
                # ========== Three-Stage/Joint-Training模式：从参数重建 ==========
                if training_mode not in ('three_stage', 'joint_training'):
                    raise ValueError(f"从参数重建需要Three-Stage或Joint-Training训练模式，当前模式: {training_mode}")

                # ✅ 使用param_scaler标准化参数（与训练时保持一致）
                param_scaler = self.gui.ae_system.get('param_scaler', None)
                if param_scaler is not None:
                    # 标准化参数（训练时使用了StandardScaler）
                    params_normalized = param_scaler.transform(input_data)
                    param_tensor = torch.FloatTensor(params_normalized).to(device)
                else:
                    # 如果没有scaler（旧模型），使用原始参数并警告
                    print("⚠️ 未找到param_scaler，使用原始参数（可能导致预测不准确）")
                    param_tensor = torch.FloatTensor(input_data).to(device)

                # 参数 → ParameterMapper → Latent
                latents = parameter_mapper(param_tensor)

                # Latent → Decoder → 标准化输出
                decoder_output = autoencoder.decode(latents)

                # 逆预处理
                if mode == 'wavelet':
                    # 小波模式：标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
                    if data_adapter:
                        # 逆标准化（逆dB + 逆Z-score）
                        predicted_coeffs_np = data_adapter.inverse_adapt(decoder_output)
                        predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
                    else:
                        predicted_coeffs = decoder_output
                        predicted_coeffs_np = predicted_coeffs.cpu().numpy()

                    # 保存重建的小波系数（如果需要）
                    if return_wavelet_coeffs:
                        reconstructed_wavelet_coeffs_np = predicted_coeffs_np.copy()
                        # 获取原始RCS并计算原始小波系数（用于对比）
                        if model_ids is not None:
                            # 从数据集获取真实RCS
                            indices = []
                            for mid in model_ids:
                                if isinstance(mid, str):
                                    indices.append(int(mid) - 1)
                                else:
                                    indices.append(int(mid))
                            rcs_data_safe = self.gui.ae_system.get('rcs_data', None)
                            if rcs_data_safe is None and hasattr(self.gui, 'rcs_data') and self.gui.rcs_data is not None:
                                self.gui.ae_system['rcs_data'] = self.gui.rcs_data
                                rcs_data_safe = self.gui.rcs_data
                            if rcs_data_safe is None:
                                raise KeyError("rcs_data 缺失：无法计算原始小波系数")
                            original_rcs = rcs_data_safe[indices]
                            # 计算原始小波系数
                            original_rcs_tensor = torch.FloatTensor(original_rcs).to(device)
                            original_coeffs = wavelet_transform.forward_transform(original_rcs_tensor)
                            original_wavelet_coeffs_np = original_coeffs.cpu().numpy()

                    # 逆小波变换
                    reconstructed_rcs = wavelet_transform.inverse_transform(predicted_coeffs)
                elif mode == 'differentiable_wavelet':
                    # Differentiable小波模式：decoder输出已经是RCS，但需要获取小波系数
                    reconstructed_rcs = decoder_output  # 已经是RCS

                    # 如果需要小波系数，从AutoEncoder内部提取
                    if return_wavelet_coeffs:
                        # 获取原始RCS
                        if model_ids is not None:
                            indices = []
                            for mid in model_ids:
                                if isinstance(mid, str):
                                    indices.append(int(mid) - 1)
                                else:
                                    indices.append(int(mid))
                            rcs_data_safe = self.gui.ae_system.get('rcs_data', None)
                            if rcs_data_safe is None and hasattr(self.gui, 'rcs_data') and self.gui.rcs_data is not None:
                                self.gui.ae_system['rcs_data'] = self.gui.rcs_data
                                rcs_data_safe = self.gui.rcs_data
                            if rcs_data_safe is None:
                                raise KeyError("rcs_data 缺失：无法计算原始小波系数")
                            original_rcs = rcs_data_safe[indices]
                            original_rcs_tensor = torch.FloatTensor(original_rcs).to(device)

                            # 使用AutoEncoder内部的wavelet_transform获取原始小波系数
                            original_coeffs = autoencoder.wavelet_transform.forward_transform(original_rcs_tensor)
                            original_wavelet_coeffs_np = original_coeffs.cpu().numpy()

                            # 使用AutoEncoder内部的wavelet_transform获取重建小波系数
                            reconstructed_coeffs = autoencoder.wavelet_transform.forward_transform(reconstructed_rcs)
                            reconstructed_wavelet_coeffs_np = reconstructed_coeffs.cpu().numpy()
                else:
                    # Direct模式：标准化RCS → 逆标准化 → RCS
                    if data_adapter:
                        reconstructed_rcs_np = data_adapter.inverse_adapt(decoder_output)
                        reconstructed_rcs = torch.FloatTensor(reconstructed_rcs_np).to(device)
                    else:
                        reconstructed_rcs = decoder_output

            elif input_type == 'rcs':
                # ========== Stage1-Only模式：从RCS重建 ==========
                # RCS → 预处理 → (小波变换) → Encoder → Latent → Decoder → 逆预处理 → RCS

                # 预处理RCS数据
                if data_adapter:
                    if mode == 'wavelet':
                        # 小波模式：先小波变换再标准化
                        rcs_tensor = torch.FloatTensor(input_data).to(device)
                        wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
                        # 保存原始小波系数（如果需要）
                        if return_wavelet_coeffs:
                            original_wavelet_coeffs_np = wavelet_coeffs.cpu().numpy()
                        adapted_input = data_adapter.adapt_rcs_data(wavelet_coeffs.cpu().numpy())
                        adapted_input = torch.FloatTensor(adapted_input).to(device)
                    elif mode == 'differentiable_wavelet':
                        # Differentiable小波模式：直接输入RCS（小波变换在AutoEncoder内部）
                        rcs_tensor = torch.FloatTensor(input_data).to(device)
                        # 如果需要小波系数，先计算原始小波系数
                        if return_wavelet_coeffs:
                            original_coeffs = autoencoder.wavelet_transform.forward_transform(rcs_tensor)
                            original_wavelet_coeffs_np = original_coeffs.cpu().numpy()
                        adapted_input = rcs_tensor  # 直接使用RCS
                    else:
                        # Direct模式：直接标准化RCS
                        adapted_input = data_adapter.adapt_rcs_data(input_data)
                        adapted_input = torch.FloatTensor(adapted_input).to(device)
                else:
                    if mode == 'wavelet':
                        rcs_tensor = torch.FloatTensor(input_data).to(device)
                        adapted_input = wavelet_transform.forward_transform(rcs_tensor)
                        # 保存原始小波系数（如果需要）
                        if return_wavelet_coeffs:
                            original_wavelet_coeffs_np = adapted_input.cpu().numpy()
                    elif mode == 'differentiable_wavelet':
                        # Differentiable小波模式：直接输入RCS
                        rcs_tensor = torch.FloatTensor(input_data).to(device)
                        if return_wavelet_coeffs:
                            original_coeffs = autoencoder.wavelet_transform.forward_transform(rcs_tensor)
                            original_wavelet_coeffs_np = original_coeffs.cpu().numpy()
                        adapted_input = rcs_tensor
                    else:
                        adapted_input = torch.FloatTensor(input_data).to(device)

                # Encoder → Latent → Decoder
                reconstructed_output, latents = autoencoder(adapted_input)

                # 逆预处理
                if mode == 'wavelet':
                    # 逆标准化小波系数
                    if data_adapter:
                        reconstructed_coeffs_np = data_adapter.inverse_adapt(reconstructed_output)
                        reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
                    else:
                        reconstructed_coeffs = reconstructed_output
                        reconstructed_coeffs_np = reconstructed_coeffs.cpu().numpy()

                    # 保存重建的小波系数（如果需要）
                    if return_wavelet_coeffs:
                        reconstructed_wavelet_coeffs_np = reconstructed_coeffs_np.copy()

                    # 逆小波变换
                    reconstructed_rcs = wavelet_transform.inverse_transform(reconstructed_coeffs)
                elif mode == 'differentiable_wavelet':
                    # Differentiable小波模式：输出已经是RCS
                    reconstructed_rcs = reconstructed_output

                    # 如果需要小波系数，从AutoEncoder内部提取
                    if return_wavelet_coeffs:
                        reconstructed_coeffs = autoencoder.wavelet_transform.forward_transform(reconstructed_rcs)
                        reconstructed_wavelet_coeffs_np = reconstructed_coeffs.cpu().numpy()
                else:
                    # Direct模式：逆标准化
                    if data_adapter:
                        reconstructed_rcs_np = data_adapter.inverse_adapt(reconstructed_output)
                        reconstructed_rcs = torch.FloatTensor(reconstructed_rcs_np).to(device)
                    else:
                        reconstructed_rcs = reconstructed_output
            else:
                raise ValueError(f"不支持的input_type: {input_type}")

        # 6. 转换为numpy并返回
        reconstructed_rcs_np = reconstructed_rcs.cpu().numpy()
        latents_np = latents.cpu().numpy() if return_latents else None

        result = {
            'reconstructed_rcs': reconstructed_rcs_np,
            'input_type_used': input_type,
            'training_mode': training_mode
        }

        if return_latents:
            result['latents'] = latents_np

        if return_wavelet_coeffs and mode in ('wavelet', 'differentiable_wavelet'):
            result['original_wavelet_coeffs'] = original_wavelet_coeffs_np
            result['reconstructed_wavelet_coeffs'] = reconstructed_wavelet_coeffs_np

        return result
