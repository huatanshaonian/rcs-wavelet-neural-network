"""
统计分析管理器
处理全局统计对比分析相关功能
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
import os
from datetime import datetime
from scipy import stats
from tkinter import messagebox


class StatisticsManager:
    """统计分析管理器 - 负责全局统计对比分析"""
    
    def __init__(self, parent_gui):
        """
        初始化统计管理器
        
        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui
    def plot_global_statistics_comparison(self, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None, model=None):
        """改进的全局统计对比分析 - 保存到results文件夹"""
        
        # 参数处理
        target_fig = fig if fig is not None else getattr(self.gui, 'vis_fig', None)
        target_canvas = canvas if canvas is not None else getattr(self.gui, 'vis_canvas', None)
        log_msg = log_callback if log_callback is not None else getattr(self.gui, 'log_message', print)
        
        # 数据回退
        current_rcs_data = rcs_data if rcs_data is not None else getattr(self.gui, 'rcs_data', None)
        current_param_data = param_data if param_data is not None else getattr(self.gui, 'param_data', None)
        current_model = model if model is not None else getattr(self.gui, 'current_model', None)

        try:
            import numpy as np
            from matplotlib import pyplot as plt
            import pandas as pd
            import os
            from datetime import datetime
            from scipy import stats

            print("生成改进的全局统计对比分析...")

            # 创建结果保存目录（使用会话时间戳）
            timestamp = self.gui.get_ae_session_timestamp() if hasattr(self.gui, 'ae_session_timestamp') else datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join("results", f"statistics_comparison_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)

            # 优化1: 优先使用缓存数据进行批量预测
            all_actual_1_5g = []
            all_actual_3g = []
            all_predicted_1_5g = []
            all_predicted_3g = []
            model_stats = []

            # 检查是否有训练好的模型和缓存数据
            # 优先检查AutoEncoder系统
            has_ae_system = hasattr(self.gui, 'ae_system') and self.gui.ae_system is not None
            has_model = hasattr(self.gui, 'current_model') and self.gui.current_model is not None

            # 初始化AutoEncoder相关变量（避免UnboundLocalError）
            autoencoder = None
            parameter_mapper = None
            wavelet_transform = None

            # 检查数据来源：优先使用ae_system中的数据，否则使用self.gui.rcs_data/param_data
            if has_ae_system and 'rcs_data' in self.gui.ae_system and 'param_data' in self.gui.ae_system:
                # 情况1: ae_system中有数据缓存（训练后立即统计）
                print("检测到AutoEncoder系统缓存数据")
                has_param_data = True
                has_rcs_data = True
                param_data = self.gui.ae_system['param_data']
                rcs_data = self.gui.ae_system['rcs_data']
                # 获取AutoEncoder组件用于预测
                autoencoder = self.gui.ae_system.get('autoencoder')
                parameter_mapper = self.gui.ae_system.get('parameter_mapper')
                wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)
                use_ae_system = True
                print(f"AutoEncoder数据形状: 参数={param_data.shape}, RCS={rcs_data.shape}")
            else:
                # 情况2/3: 使用数据管理页面加载的数据 (优先使用传入的参数)
                has_param_data = current_param_data is not None
                has_rcs_data = current_rcs_data is not None
                if has_param_data and has_rcs_data:
                    param_data = current_param_data
                    rcs_data = current_rcs_data

                    # ⚠️ 关键修复：如果有AE系统，从ae_system获取模型组件
                    if has_ae_system:
                        # 情况2: 加载了模型 + 加载了数据（数据管理页面）
                        print("使用数据管理页面的数据 + AutoEncoder模型")
                        autoencoder = self.gui.ae_system.get('autoencoder')
                        parameter_mapper = self.gui.ae_system.get('parameter_mapper')
                        wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)
                        use_ae_system = True
                        print(f"AutoEncoder数据形状: 参数={param_data.shape}, RCS={rcs_data.shape}")
                    else:
                        # 情况3: 只有传统模型
                        print(f"传统模型缓存数据形状: 参数={param_data.shape}, RCS={rcs_data.shape}")
                        use_ae_system = False

            print(f"缓存数据检查: AE系统={has_ae_system}, 传统模型={has_model}, 参数数据={has_param_data}, RCS数据={has_rcs_data}")

            # ⚠️ 修复Stage1-Only缓存可用性：只需要autoencoder，不强制要求parameter_mapper
            # Stage1-Only会在后续检测并给出友好提示
            if (has_model or (has_ae_system and autoencoder is not None)) and has_param_data and has_rcs_data:

                print("使用缓存数据进行快速统计计算...")

                # 使用缓存的参数和RCS数据进行批量预测
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # ⚠️ 确定评估模式标识（用于图表显示）- 修复：直接检查training_mode
                if use_ae_system:
                    training_mode = self.gui.ae_system.get('training_mode', 'three_stage')
                    if training_mode == 'stage1_only':
                        evaluation_mode = "Stage1-Only RCS重建评估"
                    else:
                        evaluation_mode = "Three-Stage 参数预测RCS评估"
                else:
                    training_mode = 'three_stage'  # 传统模型默认为three_stage
                    evaluation_mode = "传统模型 参数预测RCS评估"

                # 根据系统类型选择预测方式
                if use_ae_system:
                    autoencoder.to(device).eval()
                    data_adapter = self.gui.ae_system.get('data_adapter', None)

                    # ⚠️ 区分两种评估模式 - 修复：直接检查training_mode而非parameter_mapper
                    if training_mode == 'stage1_only':
                        # ===== Stage1-Only模式：RCS重建评估 =====
                        print("="*60)
                        print("使用AutoEncoder系统进行重建评估（Stage1-Only模式）")
                        print("评估目标：AutoEncoder的RCS重建能力")
                        print("流程：真实RCS → Encoder → Decoder → 重建RCS")
                        print("="*60)

                        with torch.no_grad():
                            # 准备输入数据：RCS → 标准化（+小波变换）
                            if wavelet_transform is not None:
                                # Wavelet模式：RCS → 小波变换 → 标准化小波系数
                                print("Wavelet模式：先进行小波变换...")
                                rcs_tensor = torch.FloatTensor(rcs_data).to(device)
                                wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)

                                if data_adapter:
                                    # 标准化小波系数
                                    wavelet_coeffs_np = wavelet_coeffs.cpu().numpy()
                                    input_adapted = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
                                    input_tensor = torch.FloatTensor(input_adapted).to(device)
                                else:
                                    input_tensor = wavelet_coeffs
                            else:
                                # Direct模式：RCS → 标准化
                                print("Direct模式：直接标准化RCS...")
                                if data_adapter:
                                    input_adapted = data_adapter.adapt_rcs_data(rcs_data)
                                    input_tensor = torch.FloatTensor(input_adapted).to(device)
                                else:
                                    input_tensor = torch.FloatTensor(rcs_data).to(device)

                            print(f"输入数据形状: {input_tensor.shape}")

                            # AutoEncoder重建：Encode → Decode
                            latents = autoencoder.encode(input_tensor)
                            reconstructed_output = autoencoder.decode(latents)

                            print(f"Latent形状: {latents.shape}")
                            print(f"重建输出形状: {reconstructed_output.shape}")

                            # 逆变换到RCS空间
                            if wavelet_transform is not None:
                                # 小波模式：逆标准化 → 逆小波变换 → RCS
                                if data_adapter:
                                    reconstructed_coeffs_np = data_adapter.inverse_adapt(reconstructed_output)
                                    reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
                                else:
                                    reconstructed_coeffs = reconstructed_output

                                predicted_batch = wavelet_transform.inverse_transform(reconstructed_coeffs).cpu().numpy()
                            else:
                                # 直接模式：逆标准化 → RCS
                                if data_adapter:
                                    predicted_batch = data_adapter.inverse_adapt(reconstructed_output)
                                else:
                                    predicted_batch = reconstructed_output.cpu().numpy()

                            print(f"重建RCS形状: {predicted_batch.shape}")
                            print(f"前3个样本的重建RCS均值: {[predicted_batch[i].mean() for i in range(min(3, len(predicted_batch)))]}")

                    else:
                        # ===== Three-Stage模式：参数预测RCS =====
                        print("="*60)
                        print("使用AutoEncoder系统进行参数预测评估（Three-Stage模式）")
                        print("评估目标：从参数预测RCS的能力")
                        print("流程：参数 → ParameterMapper → Decoder → 预测RCS")
                        print("="*60)

                        parameter_mapper.to(device).eval()

                        with torch.no_grad():
                            # ⚠️ 关键修复：使用训练时的param_scaler标准化参数！
                            param_scaler = self.gui.ae_system.get('param_scaler', None)

                            if param_scaler is not None:
                                print("✅ 使用训练时的param_scaler标准化参数")
                                param_normalized = param_scaler.transform(param_data)
                                params_tensor = torch.FloatTensor(param_normalized).to(device)
                            else:
                                print("⚠️ 警告：未找到param_scaler，使用原始参数（可能导致预测不准确）")
                                params_tensor = torch.FloatTensor(param_data).to(device)

                            # 调试：检查输入参数的多样性
                            print(f"输入参数形状: {param_data.shape}")
                            print(f"前3个样本的原始参数: \n{param_data[:3]}")
                            if param_scaler is not None:
                                print(f"前3个样本的标准化参数: \n{param_normalized[:3]}")
                            print(f"原始参数统计: min={param_data.min():.6f}, max={param_data.max():.6f}, mean={param_data.mean():.6f}, std={param_data.std():.6f}")
                            if param_scaler is not None:
                                print(f"标准化参数统计: min={param_normalized.min():.6f}, max={param_normalized.max():.6f}, mean={param_normalized.mean():.6f}, std={param_normalized.std():.6f}")

                            predicted_latents = parameter_mapper(params_tensor)

                            # 调试：检查latent的多样性
                            latent_np = predicted_latents.cpu().numpy()
                            print(f"预测的latent形状: {latent_np.shape}")
                            print(f"前3个样本的latent均值: {latent_np[:3].mean(axis=1)}")
                            print(f"前3个样本的latent标准差: {latent_np[:3].std(axis=1)}")
                            print(f"所有latent的整体统计: min={latent_np.min():.6f}, max={latent_np.max():.6f}, mean={latent_np.mean():.6f}, std={latent_np.std():.6f}")

                            # 检查不同样本间的latent差异
                            if len(latent_np) >= 2:
                                latent_diff = np.abs(latent_np[0] - latent_np[1]).mean()
                                print(f"样本1和样本2的latent平均差异: {latent_diff:.6f}")

                            predicted_output = autoencoder.decode(predicted_latents)

                            # 根据模式处理输出
                            if wavelet_transform is not None:
                                # 小波模式：标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
                                if data_adapter:
                                    # inverse_adapt期望tensor输入，返回numpy
                                    predicted_coeffs_np = data_adapter.inverse_adapt(predicted_output)
                                    predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
                                else:
                                    predicted_coeffs = predicted_output

                                predicted_batch = wavelet_transform.inverse_transform(predicted_coeffs).cpu().numpy()
                            else:
                                # 直接模式：标准化RCS → 逆标准化（逆dB + 逆Z-score） → RCS
                                if data_adapter:
                                    # inverse_adapt期望tensor输入，返回numpy
                                    predicted_batch = data_adapter.inverse_adapt(predicted_output)
                                else:
                                    predicted_batch = predicted_output.cpu().numpy()

                            # 调试：检查预测输出的多样性
                            print(f"预测输出形状: {predicted_batch.shape}")
                            print(f"前3个样本的预测RCS均值: {[predicted_batch[i].mean() for i in range(min(3, len(predicted_batch)))]}")
                            print(f"前3个样本的预测RCS标准差: {[predicted_batch[i].std() for i in range(min(3, len(predicted_batch)))]}")
                else:
                    # 传统模型预测
                    print("使用传统神经网络模型进行预测...")
                    self.gui.current_model.to(device).eval()

                    with torch.no_grad():
                        # 批量预测所有模型（速度更快）
                        params_tensor = torch.FloatTensor(param_data).to(device)
                        predicted_batch = self.gui.current_model(params_tensor).cpu().numpy()

                # 收集所有数据和统计信息
                for i, current_rcs_data in enumerate(rcs_data):
                    model_id = f"{i+1:03d}"

                    # ===== 真实数据 =====
                    # current_rcs_data来自self.gui.ae_system['rcs_data'] 或 self.gui.rcs_data
                    # 由cache_manager.load_data_with_cache()加载
                    # 使用rcs_result['rcs_linear'] → 确保是线性域 ✅
                    actual_1_5g = current_rcs_data[:, :, 0].flatten()
                    actual_3g = current_rcs_data[:, :, 1].flatten()

                    # ===== 预测数据 =====
                    # predicted_batch已经通过inverse_adapt逆变换到线性域
                    # Wavelet模式: inverse_adapt(decoder输出) → 逆小波变换 → 线性RCS
                    # Direct模式: inverse_adapt(decoder输出) → 线性RCS
                    pred_raw_1_5g = predicted_batch[i, :, :, 0].flatten()
                    pred_raw_3g = predicted_batch[i, :, :, 1].flatten()

                    # ⚠️ 后处理分贝转换（仅在线性训练模型时可用）
                    use_postprocess_abs_db = self.gui.ae_postprocess_abs_db.get()

                    # 安全检查（只在第一个模型时执行）
                    if i == 0 and use_postprocess_abs_db and use_ae_system:
                        data_adapter_check = self.gui.ae_system.get('data_adapter', None)
                        if data_adapter_check and data_adapter_check.db_transform:
                            raise ValueError(
                                "错误：模型训练时已使用分贝处理，不应在后处理中再次转换分贝！\n"
                                "请取消勾选'后处理绝对值+分贝'选项。"
                            )
                        if i == 0:
                            print("🔄 统计分析将应用后处理：绝对值 + 分贝转换")

                    # 根据系统类型进行域转换
                    if use_ae_system:
                        # AutoEncoder系统：predicted_batch已经是线性域RCS
                        # （已经过inverse_adapt处理）
                        pred_1_5g = pred_raw_1_5g
                        pred_3g = pred_raw_3g

                        # ⚠️ 应用后处理（如果启用）
                        if use_postprocess_abs_db:
                            # 取绝对值（消除负值影响）
                            actual_1_5g = np.abs(actual_1_5g)
                            actual_3g = np.abs(actual_3g)
                            pred_1_5g = np.abs(pred_1_5g)
                            pred_3g = np.abs(pred_3g)

                        # ⚠️ 详细调试：验证数据域一致性
                        if i < 3:  # 只打印前3个模型
                            print(f"\n模型{model_id} 数据域验证:")
                            print(f"  真实RCS [1.5GHz]: min={actual_1_5g.min():.6e}, max={actual_1_5g.max():.6e}, mean={actual_1_5g.mean():.6e}")
                            print(f"  预测RCS [1.5GHz]: min={pred_1_5g.min():.6e}, max={pred_1_5g.max():.6e}, mean={pred_1_5g.mean():.6e}")
                            print(f"  真实RCS [3GHz]:   min={actual_3g.min():.6e}, max={actual_3g.max():.6e}, mean={actual_3g.mean():.6e}")
                            print(f"  预测RCS [3GHz]:   min={pred_3g.min():.6e}, max={pred_3g.max():.6e}, mean={pred_3g.mean():.6e}")
                            if use_postprocess_abs_db:
                                print(f"  数据域: 线性RCS（已取绝对值）")
                            else:
                                print(f"  数据域检查: 真实和预测的数量级应该相近（线性域RCS通常在1e-6 ~ 1e-1范围）")
                    elif hasattr(self.gui, 'preprocessing_stats') and self.gui.preprocessing_stats:
                        # 传统模型 + 预处理：网络输出是标准化的dB值
                        mean = self.gui.preprocessing_stats['mean']
                        std = self.gui.preprocessing_stats['std']
                        # 反标准化到dB域
                        pred_db_1_5g = pred_raw_1_5g * std + mean
                        pred_db_3g = pred_raw_3g * std + mean
                        # 从 dB 转换到线性域： RCS = 10^(dB/10)
                        pred_1_5g = np.power(10, pred_db_1_5g / 10.0)
                        pred_3g = np.power(10, pred_db_3g / 10.0)
                        print(f"模型{model_id}: 传统模型（dB域 → 线性域）")
                    else:
                        # 传统模型无预处理：假设输出已经是线性域
                        pred_1_5g = pred_raw_1_5g
                        pred_3g = pred_raw_3g
                        print(f"模型{model_id}: 传统模型输出（线性域）")

                    # 确保线性域数值为正
                    pred_1_5g = np.maximum(pred_1_5g, 1e-12)  # 避免负值和零值
                    pred_3g = np.maximum(pred_3g, 1e-12)

                    # 计算每个模型的统计指标
                    stats_1_5g = {
                        'model_id': model_id,
                        'freq': '1.5GHz',
                        'actual_mean': np.mean(actual_1_5g),
                        'actual_max': np.max(actual_1_5g),
                        'actual_min': np.min(actual_1_5g),
                        'predicted_mean': np.mean(pred_1_5g),
                        'predicted_max': np.max(pred_1_5g),
                        'predicted_min': np.min(pred_1_5g),
                        'correlation': np.corrcoef(actual_1_5g, pred_1_5g)[0,1],
                        'rmse': np.sqrt(np.mean((actual_1_5g - pred_1_5g)**2))
                    }

                    stats_3g = {
                        'model_id': model_id,
                        'freq': '3GHz',
                        'actual_mean': np.mean(actual_3g),
                        'actual_max': np.max(actual_3g),
                        'actual_min': np.min(actual_3g),
                        'predicted_mean': np.mean(pred_3g),
                        'predicted_max': np.max(pred_3g),
                        'predicted_min': np.min(pred_3g),
                        'correlation': np.corrcoef(actual_3g, pred_3g)[0,1],
                        'rmse': np.sqrt(np.mean((actual_3g - pred_3g)**2))
                    }

                    model_stats.extend([stats_1_5g, stats_3g])

                    # 收集所有数据点用于散点图
                    all_actual_1_5g.extend(actual_1_5g)
                    all_actual_3g.extend(actual_3g)
                    all_predicted_1_5g.extend(pred_1_5g)
                    all_predicted_3g.extend(pred_3g)

                print(f"使用缓存数据处理了 {len(rcs_data)} 个模型")

            else:
                # 降级方案：使用文件读取 + 模型预测
                print("缓存数据不可用，使用文件读取方式...")

                # ⚠️ 关键检查：必须有模型才能生成预测统计
                if not (has_model or (has_ae_system and autoencoder is not None)):
                    messagebox.showerror(
                        "错误",
                        "无法生成统计对比：\n\n"
                        "缓存数据不可用，且未检测到可用的模型。\n\n"
                        "请先：\n"
                        "1. 加载数据（建立缓存）\n"
                        "2. 或加载已训练的模型\n\n"
                        "统计对比需要使用模型进行真实预测，不能使用模拟数据。"
                    )
                    return

                import rcs_visual as rv
                import rcs_data_reader as rdr

                rcs_dir = self.gui.data_config['rcs_data_dir']
                params_file = self.gui.data_config['params_file']

                # 检查参数文件是否存在
                if not os.path.exists(params_file):
                    messagebox.showerror("错误", f"参数文件不存在: {params_file}")
                    return

                # 读取参数数据
                print(f"从文件读取参数数据: {params_file}")
                param_data_full, param_names = rdr.load_parameters(params_file, verbose=False)

                if param_data_full is None or len(param_data_full) == 0:
                    messagebox.showerror("错误", "无法读取参数数据")
                    return

                print(f"成功读取参数数据: {param_data_full.shape}")

                # 获取所有可用模型ID
                available_models = []
                if os.path.exists(rcs_dir):
                    for file in os.listdir(rcs_dir):
                        if file.endswith('_1.5G.csv'):
                            model_id = file.split('_')[0]
                            if model_id.isdigit():
                                available_models.append(model_id)

                available_models = sorted(available_models)  # 使用所有可用模型

                if not available_models:
                    messagebox.showwarning("警告", "未找到RCS数据文件")
                    return

                print(f"找到 {len(available_models)} 个RCS数据文件")

                # 准备设备
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # ⚠️ 区分两种评估模式（文件读取路径）- 修复：直接检查training_mode
                if use_ae_system:
                    training_mode = self.gui.ae_system.get('training_mode', 'three_stage')
                    is_stage1_only = (training_mode == 'stage1_only')
                else:
                    is_stage1_only = False
                    training_mode = 'three_stage'

                # 确定评估模式标识（用于图表显示）
                if is_stage1_only:
                    evaluation_mode = "Stage1-Only RCS重建评估"
                elif use_ae_system:
                    evaluation_mode = "Three-Stage 参数预测RCS评估"
                else:
                    evaluation_mode = "传统模型 参数预测RCS评估"

                if is_stage1_only:
                    # Stage1-Only模式：RCS重建评估
                    print("="*60)
                    print("文件读取模式 - Stage1-Only RCS重建评估")
                    print("评估目标：AutoEncoder的RCS重建能力")
                    print("="*60)
                    autoencoder.to(device).eval()
                    data_adapter = self.gui.ae_system.get('data_adapter', None)
                else:
                    # Three-Stage或传统模型：参数预测评估
                    print("="*60)
                    print("文件读取模式 - 参数预测RCS评估")
                    print("="*60)
                    if use_ae_system:
                        autoencoder.to(device).eval()
                        parameter_mapper.to(device).eval()
                        data_adapter = self.gui.ae_system.get('data_adapter', None)
                    elif has_model:
                        self.gui.current_model.to(device).eval()

                # 遍历每个模型进行预测和统计
                for model_id in available_models:
                    try:
                        # 读取真实RCS数据
                        data_1_5g = rv.get_rcs_matrix(model_id, "1.5G", rcs_dir)
                        data_3g = rv.get_rcs_matrix(model_id, "3G", rcs_dir)

                        actual_1_5g = data_1_5g['rcs_linear'].flatten()
                        actual_3g = data_3g['rcs_linear'].flatten()

                        # 组合为完整RCS数据 [1, 91, 91, 2]
                        rcs_matrix = np.stack([data_1_5g['rcs_linear'], data_3g['rcs_linear']], axis=-1)
                        rcs_input = rcs_matrix[np.newaxis, ...]  # [1, 91, 91, 2]

                        with torch.no_grad():
                            if is_stage1_only:
                                # ===== Stage1-Only模式：RCS → 重建RCS =====
                                # 准备输入
                                if wavelet_transform is not None:
                                    # Wavelet模式：RCS → 小波变换 → 标准化
                                    rcs_tensor = torch.FloatTensor(rcs_input).to(device)
                                    wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)

                                    if data_adapter:
                                        wavelet_coeffs_np = wavelet_coeffs.cpu().numpy()
                                        input_adapted = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
                                        input_tensor = torch.FloatTensor(input_adapted).to(device)
                                    else:
                                        input_tensor = wavelet_coeffs
                                else:
                                    # Direct模式：RCS → 标准化
                                    if data_adapter:
                                        input_adapted = data_adapter.adapt_rcs_data(rcs_input)
                                        input_tensor = torch.FloatTensor(input_adapted).to(device)
                                    else:
                                        input_tensor = torch.FloatTensor(rcs_input).to(device)

                                # 重建
                                latents = autoencoder.encode(input_tensor)
                                reconstructed_output = autoencoder.decode(latents)

                                # 逆变换
                                if wavelet_transform is not None:
                                    if data_adapter:
                                        reconstructed_coeffs_np = data_adapter.inverse_adapt(reconstructed_output)
                                        reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
                                    else:
                                        reconstructed_coeffs = reconstructed_output

                                    predicted_rcs = wavelet_transform.inverse_transform(reconstructed_coeffs).cpu().numpy()
                                else:
                                    if data_adapter:
                                        predicted_rcs = data_adapter.inverse_adapt(reconstructed_output)
                                    else:
                                        predicted_rcs = reconstructed_output.cpu().numpy()

                            else:
                                # ===== Three-Stage或传统模型：参数 → 预测RCS =====
                                model_idx = int(model_id) - 1

                                if model_idx >= len(param_data_full):
                                    print(f"跳过模型 {model_id}: 参数数据索引越界")
                                    continue

                                model_params = param_data_full[model_idx:model_idx+1]  # [1, num_params]

                                # ⚠️ 关键修复：使用训练时的param_scaler标准化参数！
                                if use_ae_system:
                                    param_scaler = self.gui.ae_system.get('param_scaler', None)
                                    if param_scaler is not None:
                                        model_params = param_scaler.transform(model_params)

                                params_tensor = torch.FloatTensor(model_params).to(device)

                                if use_ae_system:
                                    # Three-Stage模式
                                    predicted_latents = parameter_mapper(params_tensor)
                                    predicted_output = autoencoder.decode(predicted_latents)

                                    # 逆变换
                                    if wavelet_transform is not None:
                                        if data_adapter:
                                            predicted_coeffs_np = data_adapter.inverse_adapt(predicted_output)
                                            predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
                                        else:
                                            predicted_coeffs = predicted_output

                                        predicted_rcs = wavelet_transform.inverse_transform(predicted_coeffs).cpu().numpy()
                                    else:
                                        if data_adapter:
                                            predicted_rcs = data_adapter.inverse_adapt(predicted_output)
                                        else:
                                            predicted_rcs = predicted_output.cpu().numpy()
                                else:
                                    # 传统模型
                                    predicted_rcs = self.gui.current_model(params_tensor).cpu().numpy()

                            # 提取预测的两个频率 [1, 91, 91, 2/3]
                            pred_rcs_1d = predicted_rcs[0]  # [91, 91, 2/3]
                            pred_1_5g = pred_rcs_1d[:, :, 0].flatten()
                            pred_3g = pred_rcs_1d[:, :, 1].flatten()

                            # 确保线性域数值为正
                            pred_1_5g = np.maximum(pred_1_5g, 1e-12)
                            pred_3g = np.maximum(pred_3g, 1e-12)

                        # 计算统计指标
                        stats_1_5g = {
                            'model_id': model_id,
                            'freq': '1.5GHz',
                            'actual_mean': np.mean(actual_1_5g),
                            'actual_max': np.max(actual_1_5g),
                            'actual_min': np.min(actual_1_5g),
                            'predicted_mean': np.mean(pred_1_5g),
                            'predicted_max': np.max(pred_1_5g),
                            'predicted_min': np.min(pred_1_5g),
                            'correlation': np.corrcoef(actual_1_5g, pred_1_5g)[0,1],
                            'rmse': np.sqrt(np.mean((actual_1_5g - pred_1_5g)**2))
                        }

                        stats_3g = {
                            'model_id': model_id,
                            'freq': '3GHz',
                            'actual_mean': np.mean(actual_3g),
                            'actual_max': np.max(actual_3g),
                            'actual_min': np.min(actual_3g),
                            'predicted_mean': np.mean(pred_3g),
                            'predicted_max': np.max(pred_3g),
                            'predicted_min': np.min(pred_3g),
                            'correlation': np.corrcoef(actual_3g, pred_3g)[0,1],
                            'rmse': np.sqrt(np.mean((actual_3g - pred_3g)**2))
                        }

                        model_stats.extend([stats_1_5g, stats_3g])

                        # 收集部分数据点用于散点图（降采样以提高速度）
                        sample_indices = np.random.choice(len(actual_1_5g), min(1000, len(actual_1_5g)), replace=False)
                        all_actual_1_5g.extend(actual_1_5g[sample_indices])
                        all_actual_3g.extend(actual_3g[sample_indices])
                        all_predicted_1_5g.extend(pred_1_5g[sample_indices])
                        all_predicted_3g.extend(pred_3g[sample_indices])

                    except Exception as e:
                        print(f"跳过模型 {model_id}: {e}")

            if not model_stats:
                messagebox.showwarning("警告", "无法获取有效的统计数据")
                return

            # 转换为numpy数组以提高计算速度
            all_actual_1_5g = np.array(all_actual_1_5g)
            all_actual_3g = np.array(all_actual_3g)
            all_predicted_1_5g = np.array(all_predicted_1_5g)
            all_predicted_3g = np.array(all_predicted_3g)

            # ===== 在GUI中显示统计对比图 =====
            # 清除当前图形
            target_fig.clear()

            # 设置图形尺寸
            target_fig.set_size_inches(15, 10)

            # 首先创建并保存散点图
            self._save_scatter_plots(all_actual_1_5g, all_predicted_1_5g, all_actual_3g, all_predicted_3g, results_dir)

            # 创建并保存分布对比图
            self._save_distribution_plots(all_actual_1_5g, all_predicted_1_5g, all_actual_3g, all_predicted_3g, results_dir)

            # 子图1: 1.5GHz 模型均值对比图 (dBsm单位)
            ax1 = target_fig.add_subplot(2, 3, 1)
            stats_1_5_list = [s for s in model_stats if s['freq'] == '1.5GHz']

            # 提取各个模型的均值，转换为dBsm
            model_ids = [s['model_id'] for s in stats_1_5_list]
            actual_means_linear = [s['actual_mean'] for s in stats_1_5_list]
            predicted_means_linear = [s['predicted_mean'] for s in stats_1_5_list]

            # 转换为dBsm: RCS_dBsm = 10*log10(RCS_linear)
            actual_means_dbsm = [10 * np.log10(max(val, 1e-12)) for val in actual_means_linear]
            predicted_means_dbsm = [10 * np.log10(max(val, 1e-12)) for val in predicted_means_linear]

            ax1.scatter(actual_means_dbsm, predicted_means_dbsm, alpha=0.8, s=50, color='blue')
            # 添加理想预测线
            min_val = min(min(actual_means_dbsm), min(predicted_means_dbsm))
            max_val = max(max(actual_means_dbsm), max(predicted_means_dbsm))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测线')

            # 计算均值的相关系数
            r1 = np.corrcoef(actual_means_dbsm, predicted_means_dbsm)[0,1]
            ax1.set_xlabel('真实均值 (dBsm)')
            ax1.set_ylabel('预测均值 (dBsm)')
            ax1.set_title(f'1.5GHz 模型均值对比\nR = {r1:.4f}, 模型数: {len(model_ids)}\n[{evaluation_mode}]')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 子图2: 3GHz 模型均值对比图 (dBsm单位)
            ax2 = target_fig.add_subplot(2, 3, 2)
            stats_3_list = [s for s in model_stats if s['freq'] == '3GHz']

            # 提取各个模型的均值，转换为dBsm
            actual_means_linear_3g = [s['actual_mean'] for s in stats_3_list]
            predicted_means_linear_3g = [s['predicted_mean'] for s in stats_3_list]

            actual_means_dbsm_3g = [10 * np.log10(max(val, 1e-12)) for val in actual_means_linear_3g]
            predicted_means_dbsm_3g = [10 * np.log10(max(val, 1e-12)) for val in predicted_means_linear_3g]

            ax2.scatter(actual_means_dbsm_3g, predicted_means_dbsm_3g, alpha=0.8, s=50, color='red')
            # 添加理想预测线
            min_val_3g = min(min(actual_means_dbsm_3g), min(predicted_means_dbsm_3g))
            max_val_3g = max(max(actual_means_dbsm_3g), max(predicted_means_dbsm_3g))
            ax2.plot([min_val_3g, max_val_3g], [min_val_3g, max_val_3g], 'r--', alpha=0.8, linewidth=2, label='理想预测线')

            # 计算均值的相关系数
            r2 = np.corrcoef(actual_means_dbsm_3g, predicted_means_dbsm_3g)[0,1]
            ax2.set_xlabel('真实均值 (dBsm)')
            ax2.set_ylabel('预测均值 (dBsm)')
            ax2.set_title(f'3GHz 模型均值对比\nR = {r2:.4f}, 模型数: {len(model_ids)}\n[{evaluation_mode}]')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 子图3: 统计指标对比图（均值、最大值、最小值）
            ax3 = target_fig.add_subplot(2, 3, 3)
            stats_1_5_list = [s for s in model_stats if s['freq'] == '1.5GHz']
            stats_3_list = [s for s in model_stats if s['freq'] == '3GHz']

            metrics = ['均值', '最大值', '最小值']
            actual_1_5_means = [np.mean([s['actual_mean'] for s in stats_1_5_list]),
                               np.mean([s['actual_max'] for s in stats_1_5_list]),
                               np.mean([s['actual_min'] for s in stats_1_5_list])]
            predicted_1_5_means = [np.mean([s['predicted_mean'] for s in stats_1_5_list]),
                                  np.mean([s['predicted_max'] for s in stats_1_5_list]),
                                  np.mean([s['predicted_min'] for s in stats_1_5_list])]
            actual_3_means = [np.mean([s['actual_mean'] for s in stats_3_list]),
                             np.mean([s['actual_max'] for s in stats_3_list]),
                             np.mean([s['actual_min'] for s in stats_3_list])]
            predicted_3_means = [np.mean([s['predicted_mean'] for s in stats_3_list]),
                                np.mean([s['predicted_max'] for s in stats_3_list]),
                                np.mean([s['predicted_min'] for s in stats_3_list])]

            x = np.arange(len(metrics))
            width = 0.2
            ax3.bar(x - 1.5*width, actual_1_5_means, width, label='1.5GHz 真实', color='lightblue', alpha=0.8)
            ax3.bar(x - 0.5*width, predicted_1_5_means, width, label='1.5GHz 预测', color='blue', alpha=0.8)
            ax3.bar(x + 0.5*width, actual_3_means, width, label='3GHz 真实', color='lightcoral', alpha=0.8)
            ax3.bar(x + 1.5*width, predicted_3_means, width, label='3GHz 预测', color='red', alpha=0.8)
            ax3.set_xlabel('统计指标')
            ax3.set_ylabel('RCS值 (线性)')
            ax3.set_title('统计指标对比')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 子图4: 性能指标 - 相关系数
            ax4 = target_fig.add_subplot(2, 3, 4)
            models = [s['model_id'] for s in stats_1_5_list]
            corr_1_5 = [s['correlation'] for s in stats_1_5_list]
            corr_3 = [s['correlation'] for s in stats_3_list]
            x = np.arange(len(models))
            ax4.bar(x - 0.2, corr_1_5, 0.4, label='1.5GHz', alpha=0.7)
            ax4.bar(x + 0.2, corr_3, 0.4, label='3GHz', alpha=0.7)
            ax4.set_xlabel('模型ID')
            ax4.set_ylabel('相关系数')
            ax4.set_title('预测相关性对比')
            ax4.set_xticks(x)
            ax4.set_xticklabels(models)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 子图5: RMSE对比
            ax5 = target_fig.add_subplot(2, 3, 5)
            rmse_1_5 = [s['rmse'] for s in stats_1_5_list]
            rmse_3 = [s['rmse'] for s in stats_3_list]
            ax5.bar(x - 0.2, rmse_1_5, 0.4, label='1.5GHz', alpha=0.7)
            ax5.bar(x + 0.2, rmse_3, 0.4, label='3GHz', alpha=0.7)
            ax5.set_xlabel('模型ID')
            ax5.set_ylabel('RMSE')
            ax5.set_title('预测误差对比')
            ax5.set_xticks(x)
            ax5.set_xticklabels(models)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 子图6: 整体性能汇总
            ax6 = target_fig.add_subplot(2, 3, 6)
            avg_r1 = np.mean(corr_1_5)
            avg_r2 = np.mean(corr_3)
            avg_rmse1 = np.mean(rmse_1_5)
            avg_rmse2 = np.mean(rmse_3)

            summary_text = f"""整体性能统计：

1.5GHz:
  平均相关系数: {avg_r1:.4f}
  平均RMSE: {avg_rmse1:.3e}
  模型数量: {len(stats_1_5_list)}

3GHz:
  平均相关系数: {avg_r2:.4f}
  平均RMSE: {avg_rmse2:.3e}
  模型数量: {len(stats_3_list)}

总体:
  总模型数: {len(model_stats)//2}
  数据点数: {len(all_actual_1_5g) + len(all_actual_3g)}"""

            ax6.text(0.1, 0.1, summary_text, fontsize=10,
                    verticalalignment='bottom', transform=ax6.transAxes)
            ax6.axis('off')
            ax6.set_title('性能汇总统计')

            target_fig.tight_layout()
            target_canvas.draw()

            log_msg(f"改进的全局统计对比分析完成!")
            log_msg(f"结果保存位置: {results_dir}")
            log_msg(f"处理模型数量: {len(stats_1_5_list)}")
            log_msg(f"整体相关系数: 1.5GHz={avg_r1:.4f}, 3GHz={avg_r2:.4f}")

        except Exception as e:
            error_msg = f"改进的全局统计对比分析失败: {str(e)}"
            print(error_msg)
            messagebox.showerror("错误", error_msg)
            import traceback
            traceback.print_exc()

    def _save_scatter_plots(self, all_actual_1_5g, all_predicted_1_5g, all_actual_3g, all_predicted_3g, results_dir):
        """保存散点图到文件"""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import os

            # 创建散点图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 1.5GHz 散点图
            sample_size = min(5000, len(all_actual_1_5g))
            indices = np.random.choice(len(all_actual_1_5g), sample_size, replace=False)

            # 转换为dBsm单位
            x1_linear = all_actual_1_5g[indices]
            y1_linear = all_predicted_1_5g[indices]
            x1_linear = np.maximum(x1_linear, 1e-12)
            y1_linear = np.maximum(y1_linear, 1e-12)
            x1_dbsm = 10 * np.log10(x1_linear)
            y1_dbsm = 10 * np.log10(y1_linear)

            ax1.scatter(x1_dbsm, y1_dbsm, alpha=0.3, s=1, color='blue', rasterized=True)
            min_val, max_val = min(x1_dbsm.min(), y1_dbsm.min()), max(x1_dbsm.max(), y1_dbsm.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测线')
            r1 = np.corrcoef(x1_dbsm, y1_dbsm)[0,1]
            ax1.set_xlabel('真实值 (dBsm)')
            ax1.set_ylabel('预测值 (dBsm)')
            ax1.set_title(f'1.5GHz 全数据点散点图\nR = {r1:.4f}, 点数: {len(x1_dbsm)}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 3GHz 散点图
            indices = np.random.choice(len(all_actual_3g), sample_size, replace=False)
            x2_linear = all_actual_3g[indices]
            y2_linear = all_predicted_3g[indices]
            x2_linear = np.maximum(x2_linear, 1e-12)
            y2_linear = np.maximum(y2_linear, 1e-12)
            x2_dbsm = 10 * np.log10(x2_linear)
            y2_dbsm = 10 * np.log10(y2_linear)

            ax2.scatter(x2_dbsm, y2_dbsm, alpha=0.3, s=1, color='red', rasterized=True)
            min_val, max_val = min(x2_dbsm.min(), y2_dbsm.min()), max(x2_dbsm.max(), y2_dbsm.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测线')
            r2 = np.corrcoef(x2_dbsm, y2_dbsm)[0,1]
            ax2.set_xlabel('真实值 (dBsm)')
            ax2.set_ylabel('预测值 (dBsm)')
            ax2.set_title(f'3GHz 全数据点散点图\nR = {r2:.4f}, 点数: {len(x2_dbsm)}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            scatter_plot_path = os.path.join(results_dir, 'scatter_plots.png')
            plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"散点图已保存到: {scatter_plot_path}")

        except Exception as e:
            print(f"保存散点图失败: {e}")

    def _save_distribution_plots(self, all_actual_1_5g, all_predicted_1_5g, all_actual_3g, all_predicted_3g, results_dir):
        """保存RCS分布对比图到文件"""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import os

            # 创建分布对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # ===== 1.5GHz 分布对比 =====
            # 转换为dBsm单位
            actual_1_5g_linear = np.maximum(all_actual_1_5g, 1e-12)
            predicted_1_5g_linear = np.maximum(all_predicted_1_5g, 1e-12)
            actual_1_5g_dbsm = 10 * np.log10(actual_1_5g_linear)
            predicted_1_5g_dbsm = 10 * np.log10(predicted_1_5g_linear)

            # 计算合理的bins范围（使用99%分位数避免异常值影响）
            data_min = min(np.percentile(actual_1_5g_dbsm, 1), np.percentile(predicted_1_5g_dbsm, 1))
            data_max = max(np.percentile(actual_1_5g_dbsm, 99), np.percentile(predicted_1_5g_dbsm, 99))
            bins = np.linspace(data_min, data_max, 60)  # 60个bins

            # 绘制直方图
            ax1.hist(actual_1_5g_dbsm, bins=bins, alpha=0.6, label='真实值', color='blue',
                    edgecolor='black', linewidth=0.5, density=True)
            ax1.hist(predicted_1_5g_dbsm, bins=bins, alpha=0.6, label='预测值', color='red',
                    edgecolor='black', linewidth=0.5, density=True)

            # 添加统计信息
            actual_mean = np.mean(actual_1_5g_dbsm)
            actual_std = np.std(actual_1_5g_dbsm)
            pred_mean = np.mean(predicted_1_5g_dbsm)
            pred_std = np.std(predicted_1_5g_dbsm)

            # 在图上标注均值线
            ax1.axvline(actual_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'真实均值: {actual_mean:.2f} dBsm')
            ax1.axvline(pred_mean, color='red', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'预测均值: {pred_mean:.2f} dBsm')

            ax1.set_xlabel('RCS值 (dBsm)', fontsize=12)
            ax1.set_ylabel('概率密度', fontsize=12)
            ax1.set_title(f'1.5GHz RCS分布对比\n真实: μ={actual_mean:.2f}, σ={actual_std:.2f}\n预测: μ={pred_mean:.2f}, σ={pred_std:.2f}',
                         fontsize=11, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)

            # ===== 3GHz 分布对比 =====
            # 转换为dBsm单位
            actual_3g_linear = np.maximum(all_actual_3g, 1e-12)
            predicted_3g_linear = np.maximum(all_predicted_3g, 1e-12)
            actual_3g_dbsm = 10 * np.log10(actual_3g_linear)
            predicted_3g_dbsm = 10 * np.log10(predicted_3g_linear)

            # 计算合理的bins范围
            data_min_3g = min(np.percentile(actual_3g_dbsm, 1), np.percentile(predicted_3g_dbsm, 1))
            data_max_3g = max(np.percentile(actual_3g_dbsm, 99), np.percentile(predicted_3g_dbsm, 99))
            bins_3g = np.linspace(data_min_3g, data_max_3g, 60)

            # 绘制直方图
            ax2.hist(actual_3g_dbsm, bins=bins_3g, alpha=0.6, label='真实值', color='blue',
                    edgecolor='black', linewidth=0.5, density=True)
            ax2.hist(predicted_3g_dbsm, bins=bins_3g, alpha=0.6, label='预测值', color='red',
                    edgecolor='black', linewidth=0.5, density=True)

            # 添加统计信息
            actual_mean_3g = np.mean(actual_3g_dbsm)
            actual_std_3g = np.std(actual_3g_dbsm)
            pred_mean_3g = np.mean(predicted_3g_dbsm)
            pred_std_3g = np.std(predicted_3g_dbsm)

            # 在图上标注均值线
            ax2.axvline(actual_mean_3g, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'真实均值: {actual_mean_3g:.2f} dBsm')
            ax2.axvline(pred_mean_3g, color='red', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'预测均值: {pred_mean_3g:.2f} dBsm')

            ax2.set_xlabel('RCS值 (dBsm)', fontsize=12)
            ax2.set_ylabel('概率密度', fontsize=12)
            ax2.set_title(f'3GHz RCS分布对比\n真实: μ={actual_mean_3g:.2f}, σ={actual_std_3g:.2f}\n预测: μ={pred_mean_3g:.2f}, σ={pred_std_3g:.2f}',
                         fontsize=11, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            distribution_plot_path = os.path.join(results_dir, 'distribution_plots.png')
            plt.savefig(distribution_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"RCS分布对比图已保存到: {distribution_plot_path}")

            # 打印分布统计摘要
            print(f"\n=== RCS分布统计摘要 ===")
            print(f"1.5GHz:")
            print(f"  真实值: μ={actual_mean:.2f} dBsm, σ={actual_std:.2f} dBsm, 范围=[{actual_1_5g_dbsm.min():.2f}, {actual_1_5g_dbsm.max():.2f}]")
            print(f"  预测值: μ={pred_mean:.2f} dBsm, σ={pred_std:.2f} dBsm, 范围=[{predicted_1_5g_dbsm.min():.2f}, {predicted_1_5g_dbsm.max():.2f}]")
            print(f"  均值偏差: {abs(pred_mean - actual_mean):.2f} dBsm ({abs(pred_mean - actual_mean)/abs(actual_mean)*100:.2f}%)")
            print(f"3GHz:")
            print(f"  真实值: μ={actual_mean_3g:.2f} dBsm, σ={actual_std_3g:.2f} dBsm, 范围=[{actual_3g_dbsm.min():.2f}, {actual_3g_dbsm.max():.2f}]")
            print(f"  预测值: μ={pred_mean_3g:.2f} dBsm, σ={pred_std_3g:.2f} dBsm, 范围=[{predicted_3g_dbsm.min():.2f}, {predicted_3g_dbsm.max():.2f}]")
            print(f"  均值偏差: {abs(pred_mean_3g - actual_mean_3g):.2f} dBsm ({abs(pred_mean_3g - actual_mean_3g)/abs(actual_mean_3g)*100:.2f}%)")

        except Exception as e:
            print(f"保存RCS分布对比图失败: {e}")
            import traceback
            traceback.print_exc()

