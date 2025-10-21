"""
评估管理器
处理所有模型评估相关功能
"""

import numpy as np
import torch
from tkinter import messagebox, filedialog


class EvaluationManager:
    """评估管理器 - 负责所有模型评估功能"""

    def __init__(self, parent_gui):
        """
        初始化评估管理器

        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui

    def _evaluate_traditional_model(self):
        """评估传统网络模型"""
        # 准备预处理统计信息（使用训练时保存的stats）
        use_log = self.gui.use_log_preprocessing.get()

        # 优先使用checkpoint中保存的preprocessing_stats
        if hasattr(self, 'preprocessing_stats') and self.gui.preprocessing_stats:
            preprocessing_stats = self.gui.preprocessing_stats
            self.gui.log_message(f"使用checkpoint的preprocessing_stats: mean={preprocessing_stats['mean']:.2f}, std={preprocessing_stats['std']:.2f}")
        elif use_log:
            # 尝试使用缓存的preprocessing_stats
            if hasattr(self, '_preprocessing_stats') and self.gui._preprocessing_stats:
                preprocessing_stats = self.gui._preprocessing_stats
                self.gui.log_message(f"使用缓存的stats: mean={preprocessing_stats['mean']:.2f}, std={preprocessing_stats['std']:.2f}")
            else:
                # 如果没有缓存，重新计算预处理统计
                import numpy as np  # 确保numpy可用
                self.gui.log_message("警告: 无checkpoint stats且无缓存，重新计算...")
                epsilon = float(self.gui.log_epsilon_var.get()) if self.gui.log_epsilon_var.get() else 1e-10
                rcs_db = 10 * np.log10(np.maximum(self.gui.rcs_data, epsilon))
                preprocessing_stats = {
                    'mean': np.mean(rcs_db),
                    'std': np.std(rcs_db)
                }
                # 缓存结果
                self.gui._preprocessing_stats = preprocessing_stats
                self.gui.log_message(f"重新计算的stats: mean={preprocessing_stats['mean']:.2f}, std={preprocessing_stats['std']:.2f}")
        else:
            preprocessing_stats = None

        # 创建测试数据集：使用预处理后的数据
        if use_log:
            # 使用缓存的预处理数据用于评估
            if hasattr(self, '_preprocessed_data'):
                params_eval = self.gui._preprocessed_data['params'][-20:]
                rcs_eval = self.gui._preprocessed_data['rcs'][-20:]
                test_dataset = RCSDataset(params_eval, rcs_eval, augment=False)
                self.gui.log_message("使用缓存的预处理数据进行评估")
            else:
                # 如果没有预处理缓存，使用原始数据
                self.gui.log_message("警告: 无预处理缓存，使用原始数据")
                test_dataset = RCSDataset(self.gui.param_data[-20:], self.gui.rcs_data[-20:], augment=False)
        else:
            # 使用原始数据
            test_dataset = RCSDataset(self.gui.param_data[-20:], self.gui.rcs_data[-20:], augment=False)

        # 创建评估器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        evaluator = RCSEvaluator(
            self.gui.current_model,
            device,
            use_log_output=use_log,
            preprocessing_stats=preprocessing_stats
        )

        # 执行评估
        self.gui.evaluation_results = evaluator.evaluate_dataset(test_dataset)

        # 更新评估结果显示
        self.gui._update_evaluation_display()

    def _evaluate_autoencoder_model(self):
        """评估AutoEncoder模型 - 使用统一重建函数"""
        import torch
        import numpy as np

        try:
            # 获取测试数据
            rcs_data = self.gui.ae_system['rcs_data']
            param_data = self.gui.ae_system['param_data']
            training_mode = self.gui.ae_system.get('training_mode', 'three_stage')

            # 使用后20%的数据作为测试集
            test_size = int(len(rcs_data) * 0.2)
            test_rcs = rcs_data[-test_size:]
            test_params = param_data[-test_size:]

            self.gui.log_message(f"📊 AutoEncoder评估配置:")
            self.gui.log_message(f"  训练模式: {training_mode}")
            self.gui.log_message(f"  测试样本数: {test_size}")
            self.gui.log_message(f"  RCS数据: {test_rcs.shape}")
            if training_mode == 'three_stage':
                self.gui.log_message(f"  参数数据: {test_params.shape}")

            if training_mode == 'stage1_only':
                self.gui.log_message("📈 开始AutoEncoder重建评估（Stage 1 Only）...")
                self.gui.log_message("  评估方式: RCS → Encoder → Decoder → 重建RCS")
            else:
                self.gui.log_message("📈 开始AutoEncoder端到端评估（Three Stage）...")
                self.gui.log_message("  评估方式: 参数 → ParameterMapper → Decoder → RCS")

            # 批量重建 - 使用统一重建函数
            batch_size = 10
            num_batches = (test_size + batch_size - 1) // batch_size
            predictions = []
            targets = []
            total_loss = 0.0
            total_samples = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, test_size)
                batch_targets = test_rcs[start_idx:end_idx]

                # 准备输入数据
                if training_mode == 'three_stage':
                    batch_input = test_params[start_idx:end_idx]
                else:
                    batch_input = test_rcs[start_idx:end_idx]

                # 使用统一重建函数
                result = self.gui._reconstruct_rcs(
                    input_data=batch_input,
                    input_type='auto',  # 自动根据training_mode选择
                    return_latents=False
                )

                batch_predictions = result['reconstructed_rcs']

                # 计算损失
                batch_loss = np.mean((batch_predictions - batch_targets) ** 2)
                total_loss += batch_loss * len(batch_predictions)
                total_samples += len(batch_predictions)

                # 收集结果
                predictions.append(batch_predictions)
                targets.append(batch_targets)

                if batch_idx % 5 == 0:
                    self.gui.log_message(f"  批次 {batch_idx+1}/{num_batches}: Loss = {batch_loss:.6f}")

            # 合并所有批次结果
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)

            # 计算整体指标
            avg_loss = total_loss / total_samples
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))

            # 按频率计算指标
            freq_mse = []
            freq_rmse = []
            freq_mae = []

            for freq_idx in range(predictions.shape[-1]):  # 最后一维是频率
                pred_freq = predictions[..., freq_idx]
                target_freq = targets[..., freq_idx]

                freq_mse.append(np.mean((pred_freq - target_freq) ** 2))
                freq_rmse.append(np.sqrt(freq_mse[-1]))
                freq_mae.append(np.mean(np.abs(pred_freq - target_freq)))

            # 创建评估结果
            self.gui.evaluation_results = {
                'overall': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'avg_loss': avg_loss
                },
                'frequencies': {
                    'mse': freq_mse,
                    'rmse': freq_rmse,
                    'mae': freq_mae
                },
                'model_type': 'autoencoder',
                'test_samples': test_size
            }

            self.gui.log_message(f"✅ AutoEncoder评估完成:")
            self.gui.log_message(f"  平均损失: {avg_loss:.6f}")
            self.gui.log_message(f"  整体RMSE: {rmse:.6f}")
            self.gui.log_message(f"  整体MAE: {mae:.6f}")

            # 更新评估结果显示
            self.gui._update_evaluation_display()

        except Exception as e:
            self.gui.log_message(f"❌ AutoEncoder评估失败: {e}")
            raise e
    def _update_evaluation_display(self):
        """更新评估结果显示（支持AutoEncoder和传统网络）"""
        # 清空现有内容
        for item in self.gui.eval_tree.get_children():
            self.gui.eval_tree.delete(item)

        results = self.gui.evaluation_results

        # 根据模型类型显示不同的结果
        if results.get('model_type') == 'autoencoder':
            self.gui._display_autoencoder_results(results)
        else:
            self.gui._display_traditional_results(results)
    def _display_autoencoder_results(self, results):
        """显示AutoEncoder评估结果"""
        # 添加基本信息
        basic_node = self.gui.eval_tree.insert("", "end", text="基本信息")
        self.gui.eval_tree.insert(basic_node, "end", values=("模型类型", "AutoEncoder", "", ""))
        self.gui.eval_tree.insert(basic_node, "end", values=("测试样本数", str(results['test_samples']), "", ""))

        # 添加整体指标
        overall_node = self.gui.eval_tree.insert("", "end", text="整体指标")
        overall = results['overall']
        self.gui.eval_tree.insert(overall_node, "end", values=("MSE", "", "", f"{overall['mse']:.6f}"))
        self.gui.eval_tree.insert(overall_node, "end", values=("RMSE", "", "", f"{overall['rmse']:.6f}"))
        self.gui.eval_tree.insert(overall_node, "end", values=("MAE", "", "", f"{overall['mae']:.6f}"))
        self.gui.eval_tree.insert(overall_node, "end", values=("平均损失", "", "", f"{overall['avg_loss']:.6f}"))

        # 添加频率指标（如果有多个频率）
        freq_metrics = results['frequencies']
        if len(freq_metrics['mse']) > 1:
            freq_node = self.gui.eval_tree.insert("", "end", text="频率指标")
            freq_labels = ['1.5GHz', '3GHz'] if len(freq_metrics['mse']) == 2 else [f'Freq{i+1}' for i in range(len(freq_metrics['mse']))]

            for metric in ['mse', 'rmse', 'mae']:
                values = [f"{val:.6f}" for val in freq_metrics[metric]]
                if len(values) == 2:
                    self.gui.eval_tree.insert(freq_node, "end", values=(metric.upper(), values[0], values[1], ""))
                else:
                    self.gui.eval_tree.insert(freq_node, "end", values=(metric.upper(), str(values), "", ""))
    def _display_traditional_results(self, results):
        """显示传统网络评估结果"""
        # 添加回归指标
        reg_node = self.gui.eval_tree.insert("", "end", text="回归指标")
        metrics = results['regression_metrics']
        self.gui.eval_tree.insert(reg_node, "end", values=("RMSE", "", "", f"{metrics['rmse']:.4f}"))
        self.gui.eval_tree.insert(reg_node, "end", values=("R²", "", "", f"{metrics['r2']:.4f}"))
        self.gui.eval_tree.insert(reg_node, "end", values=("相关系数", "", "", f"{metrics['correlation']:.4f}"))

        # 添加频率指标
        freq_node = self.gui.eval_tree.insert("", "end", text="频率指标")
        freq_metrics = results['frequency_metrics']
        for metric in ['rmse', 'correlation', 'r2']:
            self.gui.eval_tree.insert(freq_node, "end",
                                values=(metric.upper(),
                                       f"{freq_metrics['1.5GHz'][metric]:.4f}",
                                       f"{freq_metrics['3GHz'][metric]:.4f}", ""))

        # 添加物理一致性
        phys_node = self.gui.eval_tree.insert("", "end", text="物理一致性")
        phys_metrics = results['physics_consistency']
        self.gui.eval_tree.insert(phys_node, "end",
                            values=("对称性得分", "", "", f"{phys_metrics['symmetry_score']:.4f}"))

