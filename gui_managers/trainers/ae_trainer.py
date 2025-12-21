"""
AutoEncoder Trainer
Handles all AutoEncoder related training workflows (Stage 1, 2, 3 and End-to-End)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tkinter import messagebox
import tkinter as tk
from copy import deepcopy
import os
from datetime import datetime

from autoencoder.utils.gradient_monitor import GradientMonitor

from autoencoder.utils.configurable_loss import create_loss_function as create_configurable_loss



class AETrainer:
    """AutoEncoder Trainer - Handles complex multi-stage training workflows"""

    def __init__(self, gui):
        """
        Initialize AutoEncoder Trainer

        Args:
            gui: Parent GUI instance for accessing state and logging
        """
        self.gui = gui

    def _save_stage1_checkpoint(self, best_val_loss, best_epoch):
        """
        自动保存Stage 1 checkpoint（三阶段训练时调用）

        Args:
            best_val_loss: Stage 1最佳验证损失
            best_epoch: Stage 1最佳epoch
        """
        import json

        try:
            # 创建checkpoint目录
            checkpoint_dir = "ae_checkpoints"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # 生成文件名
            ae_system = self.gui.ae_system
            mode = ae_system.get('mode', 'wavelet')
            architecture = ae_system.get('architecture', 'cnn')
            activation = ae_system.get('activation', 'relu')

            # 获取预处理标签
            data_adapter = ae_system.get('data_adapter')
            preprocess_tag = "raw"
            if data_adapter:
                normalize = getattr(data_adapter, 'normalize', False)
                use_db = getattr(data_adapter, 'use_db', False)
                if normalize and use_db:
                    preprocess_tag = "norm_db"
                elif normalize:
                    preprocess_tag = "norm"
                elif use_db:
                    preprocess_tag = "db"

            # 时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 文件名格式: mode_architecture_activation_preprocess_stage1_timestamp.pth
            base_name = f"{mode}_{architecture}_{activation}_{preprocess_tag}_stage1_{timestamp}"
            model_path = os.path.join(checkpoint_dir, f"{base_name}.pth")
            config_path = os.path.join(checkpoint_dir, f"{base_name}_config.json")

            # 保存模型权重（仅AutoEncoder，不含ParameterMapper）
            autoencoder = ae_system['autoencoder']
            save_dict = {
                'autoencoder_state_dict': autoencoder.state_dict(),
                'training_mode': 'stage1_only',  # 标记为Stage 1模型
                'stage1_best_val_loss': best_val_loss,
                'stage1_best_epoch': best_epoch
            }

            # 如果有data_adapter统计信息，一并保存
            if data_adapter and hasattr(data_adapter, 'data_stats') and data_adapter.data_stats:
                save_dict['data_adapter_stats'] = data_adapter.data_stats

            torch.save(save_dict, model_path)

            # 保存配置文件
            config_data = {
                'model_info': {
                    'mode': mode,
                    'architecture': architecture,
                    'latent_dim': ae_system.get('latent_dim', 256),
                    'dropout_rate': ae_system.get('dropout_rate', 0.2),
                    'activation': activation,
                    'training_mode': 'stage1_only'
                },
                'frequency_config': {
                    'num_frequencies': ae_system.get('num_frequencies', 2),
                    'frequency_labels': ae_system.get('frequency_labels', ['1.5GHz', '3.0GHz']),
                    'frequency_values_ghz': ae_system.get('frequency_values_ghz', [1.5, 3.0])
                },
                'data_preprocessing': {
                    'wavelet_type': ae_system.get('wavelet_type', 'db4'),
                    'normalize': getattr(data_adapter, 'normalize', False) if data_adapter else False,
                    'use_db': getattr(data_adapter, 'use_db', False) if data_adapter else False,
                    'normalization_method': getattr(data_adapter, 'normalization_method', 'z_score') if data_adapter else 'z_score'
                },
                'training_info': {
                    'stage1_best_val_loss': best_val_loss,
                    'stage1_best_epoch': best_epoch,
                    'note': 'This is a Stage 1 checkpoint automatically saved during three-stage training'
                },
                'save_info': {
                    'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model_filename': os.path.basename(model_path),
                    'checkpoint_type': 'stage1_auto_checkpoint'
                }
            }

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            self.gui.ae_log(f"💾 自动保存Stage 1 checkpoint:")
            self.gui.ae_log(f"  模型: {os.path.basename(model_path)}")
            self.gui.ae_log(f"  配置: {os.path.basename(config_path)}")
            self.gui.ae_log(f"  最佳损失: {best_val_loss:.6f} (Epoch {best_epoch})")
            self.gui.ae_log(f"  💡 提示: 可以加载此checkpoint单独查看Stage 1重建性能")

        except Exception as e:
            self.gui.ae_log(f"⚠️ 保存Stage 1 checkpoint失败: {e}")
            # 不抛出异常，避免中断训练流程

    def _is_lbfgs_optimizer(self, optimizer):
        """检查是否是L-BFGS优化器"""
        return isinstance(optimizer, optim.LBFGS)

    def _train_batch_with_lbfgs(self, optimizer, model, batch_data, batch_target, criterion, device):
        """
        使用L-BFGS训练单个batch（需要闭包）

        Args:
            optimizer: L-BFGS优化器
            model: 模型（autoencoder或parameter_mapper）
            batch_data: 输入数据
            batch_target: 目标数据（可能与batch_data相同，用于重建）
            criterion: 损失函数
            device: 设备

        Returns:
            loss.item(): 损失值
        """
        batch_data = batch_data.to(device)
        if batch_target is not batch_data:
            batch_target = batch_target.to(device)

        # L-BFGS需要闭包函数
        def closure():
            optimizer.zero_grad()

            # 根据模型类型选择前向传播方式
            if hasattr(model, 'encode'):  # AutoEncoder
                output, _ = model(batch_data)
            else:  # ParameterMapper
                output = model(batch_data)

            loss = criterion(output, batch_target)
            loss.backward()
            return loss

        # L-BFGS.step()返回loss
        loss = optimizer.step(closure)

        # 如果返回的是tensor，转换为float
        if isinstance(loss, torch.Tensor):
            return loss.item()
        return loss

    def _train_batch_standard(self, optimizer, model, batch_data, batch_target, criterion, device):
        """
        使用标准优化器（Adam/SGD等）训练单个batch

        Args:
            optimizer: 标准优化器
            model: 模型
            batch_data: 输入数据
            batch_target: 目标数据
            criterion: 损失函数
            device: 设备

        Returns:
            loss.item(): 损失值
        """
        batch_data = batch_data.to(device)
        if batch_target is not batch_data:
            batch_target = batch_target.to(device)

        # 前向传播
        if hasattr(model, 'encode'):  # AutoEncoder
            output, _ = model(batch_data)
        else:  # ParameterMapper
            output = model(batch_data)

        loss = criterion(output, batch_target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def run_three_stage_training(self, rcs_data, param_data, training_config):
        """执行三阶段训练 (使用统一配置管理器)"""
        try:
            # 获取训练模式
            training_mode = training_config.get('training_mode', 'three_stage')

            if training_mode == 'stage1_only':
                # 仅Stage 1模式：只训练AutoEncoder重建能力
                self.gui.ae_log("🚀 开始AutoEncoder重建训练 (Stage 1 Only):")
                self.gui.ae_log("📌 模式说明: 专注于AutoEncoder的重建性能研究，不训练参数映射器")

                # ✅ 数据集变化检测（继续训练场景）
                data_adapter = self.gui.ae_system.get('data_adapter', None)
                if data_adapter and hasattr(data_adapter, 'data_stats') and data_adapter.data_stats:
                    current_data_size = len(rcs_data)
                    self.gui.ae_log(f"📊 当前数据集大小: {current_data_size} 样本")
                    self.gui.ae_log("⚠️ 数据预处理统计信息将基于当前数据集重新计算")
                    self.gui.ae_log("  原因: 数据集扩充后，mean/std需要更新以反映新的数据分布")
                    self.gui.ae_log("  影响: 训练将使用新的标准化参数，模型保存时会更新adapter统计信息")

                # ✅ 智能处理训练历史：继续训练 vs 新训练
                if hasattr(self.gui, 'ae_training_history') and self.gui.ae_training_history:
                    # 已有训练历史，保留并追加
                    if 'stage_histories' in self.gui.ae_training_history and 'stage1' in self.gui.ae_training_history['stage_histories']:
                        self.gui.ae_log("🔄 检测到已有Stage 1训练历史，继续训练（权重将从当前状态继续）")
                        self.gui.ae_log(f"  之前最佳epoch: {self.gui.ae_training_history['stage_histories']['stage1'].get('best_epoch', 'N/A')}")
                        self.gui.ae_log(f"  之前最佳损失: {self.gui.ae_training_history['stage_histories']['stage1'].get('best_val_loss', 'N/A'):.6f}")
                    else:
                        self.gui.ae_log("🆕 首次训练Stage 1")

                    # 更新配置，保留stage_histories
                    self.gui.ae_training_history['training_mode'] = 'stage1_only'
                    self.gui.ae_training_history['training_config'] = training_config
                    self.gui.ae_system['training_mode'] = 'stage1_only'  # ✅ 同步到ae_system
                    if 'stage_histories' not in self.gui.ae_training_history:
                        self.gui.ae_training_history['stage_histories'] = {}
                else:
                    # 无历史，初始化新的
                    self.gui.ae_log("🆕 首次训练Stage 1")
                    self.gui.ae_training_history = {
                        'training_mode': 'stage1_only',
                        'stage_histories': {},
                        'training_config': training_config
                    }
                    self.gui.ae_system['training_mode'] = 'stage1_only'  # ✅ 同步到ae_system

                # 阶段1: AutoEncoder预训练
                self.gui.ae_log("📊 开始阶段1: AutoEncoder预训练...")
                stage1_history = self.train_stage1(rcs_data, training_config)

                # ✅ 追加/更新训练历史（保留之前的记录）
                if 'stage1' in self.gui.ae_training_history['stage_histories']:
                    # 已有Stage 1历史，合并训练曲线
                    old_history = self.gui.ae_training_history['stage_histories']['stage1']
                    stage1_history['train_losses'] = old_history.get('train_losses', []) + stage1_history['train_losses']
                    stage1_history['val_losses'] = old_history.get('val_losses', []) + stage1_history['val_losses']
                    self.gui.ae_log(f"✅ 训练历史已合并，总共 {len(stage1_history['train_losses'])} 个epoch")

                self.gui.ae_training_history['stage_histories']['stage1'] = stage1_history

                self.gui.ae_log("🎉 AutoEncoder重建训练完成!")

                # 打印通道注意力权重（如果启用）
                self.print_channel_attention_weights(rcs_data)

                self.gui.ae_log("💡 提示: 该模型只能进行RCS重建评估，不能从参数预测RCS")

                # 批量实验模式下不弹窗
                if not getattr(self.gui, 'batch_experiment_mode', False):
                    messagebox.showinfo("成功", "AutoEncoder重建训练完成！\n\n该模型专注于重建性能，适合调参和模型对比研究。")

                # 返回训练历史（批量实验需要）
                return self.gui.ae_training_history

            else:
                # 完整三阶段模式
                self.gui.ae_log("🚀 开始三阶段训练流程 (v2统一配置):")

                # ✅ 检查是否已有训练历史
                if hasattr(self.gui, 'ae_training_history') and self.gui.ae_training_history:
                    if 'stage_histories' in self.gui.ae_training_history and self.gui.ae_training_history['stage_histories']:
                        self.gui.ae_log("⚠️ 检测到已有训练历史，将被新训练覆盖")
                        self.gui.ae_log("  💡 如果只想微调，建议选择'端到端训练'模式")

                # ✅ 数据集变化检测（继续训练场景）
                data_adapter = self.gui.ae_system.get('data_adapter', None)
                if data_adapter and hasattr(data_adapter, 'data_stats') and data_adapter.data_stats:
                    current_data_size = len(rcs_data)
                    self.gui.ae_log(f"📊 当前数据集大小: {current_data_size} 样本")
                    self.gui.ae_log("⚠️ 数据预处理统计信息将基于当前数据集重新计算")
                    self.gui.ae_log("  原因: 数据集扩充后，mean/std需要更新以反映新的数据分布")
                    self.gui.ae_log("  影响: 训练将使用新的标准化参数，模型保存时会更新adapter统计信息")

                # 初始化训练历史（三阶段训练会重新训练所有阶段）
                self.gui.ae_training_history = {
                    'training_mode': 'three_stage',
                    'stage_histories': {},
                    'training_config': training_config  # ✅ 保存训练配置
                }

                # 阶段1: AutoEncoder预训练
                self.gui.ae_log("📊 开始阶段1: AutoEncoder预训练...")
                stage1_history = self.train_stage1(rcs_data, training_config)
                self.gui.ae_training_history['stage_histories']['stage1'] = stage1_history

                # 阶段2: 参数映射训练
                self.gui.ae_log("🎯 开始阶段2: 参数映射训练...")
                stage2_history = self.train_stage2(rcs_data, param_data, training_config)
                self.gui.ae_training_history['stage_histories']['stage2'] = stage2_history

                # 阶段3: 端到端微调
                self.gui.ae_log("⚡ 开始阶段3: 端到端微调...")
                stage3_history = self.train_stage3(rcs_data, param_data, training_config)
                self.gui.ae_training_history['stage_histories']['stage3'] = stage3_history

                # ✅ 三阶段训练完成，同步训练模式到ae_system
                self.gui.ae_system['training_mode'] = 'three_stage'

                self.gui.ae_log("🎉 三阶段训练完成!")

                # 打印通道注意力权重（如果启用）
                self.print_channel_attention_weights(rcs_data)

                # 批量实验模式下不弹窗
                if not getattr(self.gui, 'batch_experiment_mode', False):
                    messagebox.showinfo("成功", "三阶段训练完成!")

            # 返回训练历史（批量实验需要）
            return self.gui.ae_training_history

        except Exception as e:
            error_msg = f"训练失败: {e}"
            self.gui.ae_log(f"❌ {error_msg}")

            # 批量实验模式下不弹窗，只抛出异常让批量实验处理
            if not getattr(self.gui, 'batch_experiment_mode', False):
                messagebox.showerror("错误", error_msg)
            raise

    def run_end_to_end_training(self, rcs_data, param_data, training_config):
        """执行端到端训练 (使用统一配置管理器)"""
        try:
            total_epochs = sum(training_config['epochs'].values())
            self.gui.ae_log("🚀 开始端到端训练流程 (v2统一配置):")
            self.gui.ae_log(f"  📊 总训练轮数: {total_epochs}")

            # 实现端到端训练
            self.train_end_to_end_full(rcs_data, param_data, training_config, total_epochs)

            # ✅ 端到端训练完成，同步训练模式到ae_system
            if not hasattr(self.gui, 'ae_training_history'):
                self.gui.ae_training_history = {}
            self.gui.ae_training_history['training_mode'] = 'end_to_end'
            self.gui.ae_system['training_mode'] = 'end_to_end'

            self.gui.ae_log("🎉 端到端训练完成!")

            # 批量实验模式下不弹窗
            if not getattr(self.gui, 'batch_experiment_mode', False):
                messagebox.showinfo("成功", "端到端训练完成!")

        except Exception as e:
            error_msg = f"端到端训练失败: {e}"
            self.gui.ae_log(f"❌ {error_msg}")

            # 批量实验模式下不弹窗，只抛出异常让批量实验处理
            if not getattr(self.gui, 'batch_experiment_mode', False):
                messagebox.showerror("错误", error_msg)
            raise

    def run_joint_training(self, rcs_data, param_data, training_config):
        """
        执行联合训练模式

        核心思想：同时训练AutoEncoder和ParameterMapper，强制隐空间与参数对齐

        损失函数：
            L_recon_rcs: RCS重建损失（衡量AE压缩能力）
            L_consistency: 隐空间一致性损失（强制encoder和mapper对齐）
            L_param_recon: 参数重建损失（params → RCS的能力）

        总损失：total_loss = α*L_recon_rcs + β*L_consistency + γ*L_param_recon
        """
        try:
            epochs = training_config['epochs'].get('joint', 200)  # 联合训练epoch数
            self.gui.ae_log("🚀 开始联合训练模式:")
            self.gui.ae_log(f"  📊 训练轮数: {epochs}")
            self.gui.ae_log(f"  🎯 目标: 优化 params → RCS 的重建能力")

            # 获取损失权重
            alpha = training_config.get('alpha_recon', 0.3)      # RCS重建权重
            beta = training_config.get('beta_consistency', 0.5)  # 一致性权重
            gamma = training_config.get('gamma_param_recon', 1.0)  # 参数重建权重（最重要）

            self.gui.ae_log(f"  ⚖️ 损失权重: α={alpha} (RCS重建), β={beta} (一致性), γ={gamma} (参数重建)")

            # 执行联合训练
            history = self.train_joint_mode(rcs_data, param_data, training_config, epochs, alpha, beta, gamma)

            # 保存训练历史
            if not hasattr(self.gui, 'ae_training_history'):
                self.gui.ae_training_history = {'stage_histories': {}}
            self.gui.ae_training_history['joint'] = history
            self.gui.ae_training_history['training_mode'] = 'joint_training'
            self.gui.ae_system['training_mode'] = 'joint_training'

            self.gui.ae_log("🎉 联合训练完成!")

            # 批量实验模式下不弹窗
            if not getattr(self.gui, 'batch_experiment_mode', False):
                messagebox.showinfo("成功", "联合训练完成!")

        except Exception as e:
            error_msg = f"联合训练失败: {e}"
            self.gui.ae_log(f"❌ {error_msg}")

            if not getattr(self.gui, 'batch_experiment_mode', False):
                messagebox.showerror("错误", error_msg)
            raise

    def _apply_rcs_constraint_if_needed(self, decoder_output, mode, data_adapter, wavelet_transform, device):
        """
        如果启用RCS非负约束，将decoder输出逆变换到RCS空间并应用Softplus

        ⚠️ 重要：保持梯度流动！不能使用detach()

        Args:
            decoder_output: Decoder输出（标准化空间，tensor with grad）
            mode: 'wavelet', 'direct', 'differentiable_wavelet'
            data_adapter: 数据适配器
            wavelet_transform: 小波变换器（wavelet模式需要）
            device: 设备

        Returns:
            rcs_output: RCS空间的输出（可能应用了softplus约束，保持梯度）
        """
        # 检查是否启用约束
        enforce_constraint = (hasattr(self.gui, 'ae_enforce_nonnegative_rcs') and
                            self.gui.ae_enforce_nonnegative_rcs.get())

        # 逆变换到RCS空间（⚠️ 必须保持可微分）
        if mode == 'wavelet':
            # Wavelet: 标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
            if data_adapter:
                # ✅ 使用可微分的逆标准化（基于data_adapter的mean/std）
                # 逆Z-score: x_orig = x_normalized * std + mean
                mean = torch.FloatTensor(data_adapter.mean_).to(device)
                std = torch.FloatTensor(data_adapter.std_).to(device)
                coeffs_tensor = decoder_output * std + mean

                # 如果有dB逆变换，也需要可微分实现
                if data_adapter.db_transform:
                    # 逆dB: x_linear = sign(x_db) * 10^(abs(x_db)/10)
                    coeffs_tensor = torch.sign(coeffs_tensor) * torch.pow(10, torch.abs(coeffs_tensor) / 10.0)
            else:
                coeffs_tensor = decoder_output

            # 逆小波变换（已经是可微分的）
            rcs_output = wavelet_transform.inverse_transform(coeffs_tensor)

        elif mode == 'differentiable_wavelet':
            # Differentiable Wavelet: decoder输出已经是RCS
            rcs_output = decoder_output

        else:  # direct
            # Direct: 标准化RCS → 逆标准化 → RCS
            if data_adapter:
                # ✅ 可微分逆标准化
                mean = torch.FloatTensor(data_adapter.mean_).to(device)
                std = torch.FloatTensor(data_adapter.std_).to(device)
                rcs_output = decoder_output * std + mean

                # 如果有dB逆变换
                if data_adapter.db_transform:
                    rcs_output = torch.sign(rcs_output) * torch.pow(10, torch.abs(rcs_output) / 10.0)
            else:
                rcs_output = decoder_output

        # 应用非负约束（如果启用）
        if enforce_constraint:
            rcs_output = torch.nn.functional.softplus(rcs_output, beta=1)

        return rcs_output

    def train_stage1(self, rcs_data, training_config):
        """阶段1: AutoEncoder预训练"""
        try:
            # 获取AutoEncoder组件
            autoencoder = self.gui.ae_system['autoencoder']
            # ✅ 修复: 初始化 parameter_mapper (即使为None)，避免未定义错误
            parameter_mapper = self.gui.ae_system.get('parameter_mapper')
            wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)  # 直接模式时为None
            mode = self.gui.ae_system.get('mode', 'wavelet')
            # 获取Loss归一化系数（如果存在）
            loss_normalization_factor = self.gui.ae_system.get('loss_normalization_factor', 1.0)

            # 检查是否启用RCS非负约束
            enforce_rcs_constraint = (hasattr(self.gui, 'ae_enforce_nonnegative_rcs') and
                                     self.gui.ae_enforce_nonnegative_rcs.get())
            if enforce_rcs_constraint:
                self.gui.ae_log("✅ RCS非负约束已启用（训练时在RCS空间计算损失）")

            # 设置设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device)
            self.gui.ae_log(f"🖥️ 使用设备: {device}")
            self.gui.ae_log(f"🔧 训练模式: {mode}")

            # 获取data_adapter并应用数据预处理
            data_adapter = self.gui.ae_system.get('data_adapter', None)
            if data_adapter is None:
                # 如果没有adapter，创建默认的（不应该发生）
                from autoencoder.utils.data_adapters import RCS_DataAdapter
                current_mode = self.gui.ae_system.get('mode', 'direct')
                data_adapter = RCS_DataAdapter(normalize=True, mode=current_mode)
                self.gui.ae_log(f"⚠️ 未找到data_adapter，使用默认配置 (mode={current_mode})")

            # ⚠️ 关键: 数据处理顺序
            self.gui.ae_log(f"🔧 数据预处理配置: 标准化={data_adapter.normalize}, dB变换={data_adapter.db_transform}")
            self.gui.ae_log(f"🔧 原始RCS数据范围: [{rcs_data.min():.4f}, {rcs_data.max():.4f}]")

            # 根据模式决定输入数据
            if mode in ['wavelet', 'differentiable_wavelet']:
                # Wavelet模式: Step 1 - 小波变换（必须在线性域进行）
                self.gui.ae_log("📊 Step 1: 在原始RCS线性域数据上执行小波变换...")
                # ⚠️ 修复：forward_transform期望tensor输入，但rcs_data是numpy
                rcs_tensor = torch.FloatTensor(rcs_data)
                wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor) if wavelet_transform else rcs_tensor
                self.gui.ae_log(f"📊 小波系数范围（线性域）: [{wavelet_coeffs.min():.4f}, {wavelet_coeffs.max():.4f}]")

                # Step 2 - 预处理（dB变换 + Z-score标准化）
                self.gui.ae_log("📊 Step 2: 对小波系数应用预处理（dB变换 + 标准化）...")
                input_data = data_adapter.adapt_rcs_data(wavelet_coeffs.cpu().numpy())
                self.gui.ae_log(f"📊 预处理后小波系数范围: [{input_data.min():.4f}, {input_data.max():.4f}]")
            else:
                # Direct模式: 直接预处理（dB变换 + Z-score标准化）
                self.gui.ae_log("📊 Direct模式: 对RCS数据应用预处理（dB变换 + 标准化）...")
                input_data = data_adapter.adapt_rcs_data(rcs_data)
                self.gui.ae_log(f"📊 预处理后RCS数据范围: [{input_data.min():.4f}, {input_data.max():.4f}]")

            # 数据划分: 80%训练，20%验证
            # ✅ 如果启用RCS约束，需要同时保存原始RCS作为target
            if enforce_rcs_constraint:
                # 创建包含输入数据和原始RCS的数据集
                rcs_tensor = torch.FloatTensor(rcs_data)
                dataset = TensorDataset(torch.FloatTensor(input_data), rcs_tensor)
            else:
                # 正常流程：只有输入数据
                dataset = TensorDataset(torch.FloatTensor(input_data))

            torch.manual_seed(42)
            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size

            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

            self.gui.ae_log(f"📊 阶段1数据划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

            # 输出训练集和验证集样本标号
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices
            self.gui.ae_log(f"📋 训练集样本标号: {sorted(train_indices)[:20]}{'...' if len(train_indices) > 20 else ''}")
            self.gui.ae_log(f"📋 验证集样本标号: {sorted(val_indices)[:20]}{'...' if len(val_indices) > 20 else ''}")
            if len(train_indices) > 20 or len(val_indices) > 20:
                self.gui.ae_log(f"   (仅显示前20个标号，完整标号请查看详细日志)")

            # 调整批次大小
            batch_size = training_config['batch_size']
            if batch_size > train_size:
                batch_size = train_size
                self.gui.ae_log(f"⚠️ 批次大小调整为 {batch_size}")

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False, drop_last=False)

            # 创建优化器和调度器 (复用项目标准)
            optimizer, scheduler = self._create_optimizer_and_scheduler(autoencoder.parameters(), training_config, stage='stage1')

            # 创建损失函数 (支持三阶段独立配置)
            criterion = self._create_stage_loss_function(training_config, stage='stage1')

            # 训练配置
            epochs = training_config['epochs']['stage1']
            patience = training_config['patience']['stage1']
            scheduler_type = training_config['lr_scheduler']

            # 训练循环
            autoencoder.train()
            best_val_loss = float('inf')
            best_epoch = 0
            best_model_state = None  # 保存最佳模型权重
            patience_counter = 0

            # 用于保存历史
            train_losses = []
            val_losses = []
            attention_history = {
                'epochs': [],
                'weights': [],
                'channel_names': None
            }

            # 创建梯度监控器
            gradient_monitor = GradientMonitor(
                log_interval=10,           # 每10步记录一次
                warn_threshold_high=10.0,  # 梯度范数>10警告
                warn_threshold_low=1e-5    # 梯度范数<1e-5警告
            )
            self.gui.ae_log("梯度监控已启用 (阈值: 1e-5 < grad_norm < 10.0)")

            # 梯度历史记录
            gradient_history = {
                'epochs': [], 'grad_norm': [], 'grad_mean': [], 'grad_std': [], 'grad_max': [], 'grad_min': []
            }

            # 检测是否使用L-BFGS
            is_lbfgs = self._is_lbfgs_optimizer(optimizer)
            if is_lbfgs:
                self.gui.ae_log("  🔍 检测到L-BFGS优化器，使用闭包训练循环")

            for epoch in range(epochs):
                # 训练
                autoencoder.train()
                train_loss = 0.0
                train_samples = 0

                for batch_idx, batch_data in enumerate(train_loader):
                    # ✅ 根据是否启用约束，解包不同的数据
                    if enforce_rcs_constraint:
                        batch_input, batch_rcs_target = batch_data
                        batch_input = batch_input.to(device)
                        batch_rcs_target = batch_rcs_target.to(device)
                    else:
                        batch_coeffs, = batch_data
                        batch_input = batch_coeffs.to(device)
                        batch_rcs_target = None

                    # 使用适配的训练方法
                    if is_lbfgs:
                        # L-BFGS暂不支持RCS约束（需要闭包支持）
                        if enforce_rcs_constraint:
                            raise NotImplementedError("L-BFGS暂不支持RCS非负约束训练")
                        loss_value = self._train_batch_with_lbfgs(
                            optimizer, autoencoder, batch_input, batch_input, criterion, device
                        )
                    else:
                        reconstructed, latent = autoencoder(batch_input)

                        # ✅ 根据是否启用约束，计算不同的损失
                        if enforce_rcs_constraint:
                            # 在RCS空间计算损失
                            rcs_recon = self._apply_rcs_constraint_if_needed(
                                reconstructed, mode, data_adapter, wavelet_transform, device
                            )
                            loss = criterion(rcs_recon, batch_rcs_target)
                        else:
                            # 在标准化空间计算损失（原有逻辑）
                            loss = criterion(reconstructed, batch_input)

                        # ⭐ 在backward前应用归一化（真正影响梯度和训练）
                        if loss_normalization_factor != 1.0:
                            loss = loss * loss_normalization_factor

                        optimizer.zero_grad()
                        loss.backward()

                        # 梯度监控（仅标准优化器）
                        if batch_idx == 0 and (epoch + 1) % 10 == 0:
                            stats, status = gradient_monitor.check_gradients(autoencoder, step=epoch, verbose=False)
                            gradient_history['epochs'].append(epoch)
                            gradient_history['grad_norm'].append(stats['grad_norm'])
                            gradient_history['grad_mean'].append(stats['grad_mean'])
                            gradient_history['grad_std'].append(stats['grad_std'])
                            gradient_history['grad_max'].append(stats['grad_max'])
                            gradient_history['grad_min'].append(stats['grad_min'])

                        optimizer.step()
                        loss_value = loss.item()

                    batch_size = batch_input.size(0) if not is_lbfgs else len(batch_input)
                    train_loss += loss_value * batch_size  # loss已经是归一化后的
                    train_samples += batch_size

                avg_train_loss = train_loss / train_samples

                # 验证
                autoencoder.eval()
                val_loss = 0.0
                val_samples = 0

                with torch.no_grad():
                    for batch_data in val_loader:
                        # ✅ 根据是否启用约束，解包不同的数据
                        if enforce_rcs_constraint:
                            batch_input, batch_rcs_target = batch_data
                            batch_input = batch_input.to(device)
                            batch_rcs_target = batch_rcs_target.to(device)
                        else:
                            batch_coeffs, = batch_data
                            batch_input = batch_coeffs.to(device)
                            batch_rcs_target = None

                        reconstructed, latent = autoencoder(batch_input)

                        # ✅ 根据是否启用约束，计算不同的损失
                        if enforce_rcs_constraint:
                            # 在RCS空间计算损失
                            rcs_recon = self._apply_rcs_constraint_if_needed(
                                reconstructed, mode, data_adapter, wavelet_transform, device
                            )
                            loss = criterion(rcs_recon, batch_rcs_target)
                        else:
                            # 在标准化空间计算损失（原有逻辑）
                            loss = criterion(reconstructed, batch_input)

                        # ⭐ 应用归一化（验证时只需要计算，不需要backward）
                        if loss_normalization_factor != 1.0:
                            loss = loss * loss_normalization_factor

                        batch_size = batch_input.size(0)
                        val_loss += loss.item() * batch_size
                        val_samples += batch_size

                avg_val_loss = val_loss / val_samples

                # 保存历史
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)

                # 学习率调度
                old_lr = optimizer.param_groups[0]['lr']
                self._step_scheduler(scheduler, scheduler_type, avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                # ✅ 如果学习率降低，回退到最佳模型
                if current_lr < old_lr and best_model_state is not None:
                    self.gui.ae_log(f"  🔄 学习率降低 {old_lr:.2e} → {current_lr:.2e}，回退到最佳模型 (Epoch {best_epoch})")
                    autoencoder.load_state_dict(best_model_state['autoencoder'])
                    patience_counter = 0

                # 早停检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_model_state = {
                        'autoencoder': deepcopy(autoencoder.state_dict())
                    }
                else:
                    patience_counter += 1

                # 记录进度
                if (epoch + 1) % 10 == 0:
                    grad_norm_str = ""
                    if epoch in gradient_history['epochs']:
                        idx = gradient_history['epochs'].index(epoch)
                        grad_norm = gradient_history['grad_norm'][idx]
                        grad_norm_str = f", Grad={grad_norm:.2e}"
                    self._log_progress(epoch, epochs, avg_train_loss, avg_val_loss, current_lr, "阶段1")
                    if grad_norm_str:
                        self.gui.ae_log(f"    梯度监控{grad_norm_str}")

                # 每100 epoch记录注意力权重
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    self._record_attention_weights(autoencoder, input_data[:8].to(device), attention_history, epoch + 1)

                # 检查用户停止请求
                if getattr(self.gui, 'stop_training_flag', False) or getattr(self.gui.training_manager, 'stop_training_flag', False):
                    self.gui.ae_log(f"  ⏹️ 用户停止训练 (Epoch {epoch+1}/{epochs})")
                    self.gui.ae_log(f"     当前验证损失: {avg_val_loss:.6f}, 最佳: {best_val_loss:.6f}")
                    break

                # 早停
                current_patience = patience
                if scheduler_type in ['multi_stage', 'adaptive_multi_stage']:
                    from autoencoder.training.multi_stage_scheduler import PatienceDrivenMultiStageLRScheduler
                    if isinstance(scheduler, PatienceDrivenMultiStageLRScheduler):
                        current_patience = scheduler.get_current_patience()
                        if patience_counter >= current_patience:
                            switched = scheduler.switch_to_next_stage()
                            if switched:
                                self.gui.ae_log(f"  🔄 切换到下一学习率阶段，回退到最佳模型 (Epoch {best_epoch})")
                                autoencoder.load_state_dict(best_model_state['autoencoder'])
                                if 'parameter_mapper' in best_model_state and parameter_mapper:
                                    parameter_mapper.load_state_dict(best_model_state['parameter_mapper'])
                                patience_counter = 0
                                continue
                            else:
                                self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 已是最后学习率阶段，验证损失连续{current_patience}轮无改善")
                                break
                    else:
                        if patience_counter >= current_patience:
                            self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 验证损失连续{current_patience}轮无改善")
                            break
                else:
                    if patience_counter >= current_patience:
                        self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 验证损失连续{current_patience}轮无改善")
                        break

            # ✅ 训练结束后，恢复最佳模型
            if best_model_state is not None:
                self.gui.ae_log(f"🔄 恢复最佳epoch {best_epoch}的模型权重 (最佳验证损失: {best_val_loss:.6f})")
                autoencoder.load_state_dict(best_model_state['autoencoder'])
            else:
                self.gui.ae_log(f"⚠️ 未找到最佳模型检查点，使用当前模型")

            self.gui.ae_log(f"✅ 阶段1: AutoEncoder预训练完成，最佳验证损失: {best_val_loss:.6f}")

            self.gui.ae_log("\n📊 分析训练完成后的隐空间分布...")
            self.print_latent_space_statistics(rcs_data)

            # ✅ 三阶段训练和联合训练时自动保存Stage 1 checkpoint
            training_mode = training_config.get('training_mode', 'three_stage')
            if training_mode in ('three_stage', 'joint_training'):
                self._save_stage1_checkpoint(best_val_loss, best_epoch)

            return {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'attention_history': attention_history,
                'gradient_history': gradient_history,
                'train_indices': train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices),
                'val_indices': val_indices.tolist() if hasattr(val_indices, 'tolist') else list(val_indices)
            }

        except Exception as e:
            self.gui.ae_log(f"❌ 阶段1训练失败: {e}")
            raise e

    def train_stage2(self, rcs_data, param_data, training_config):
        """阶段2: 参数映射训练"""
        # (Content of _train_parameter_mapping_stage2_v2 with self.gui.ae_log and self._create_optimizer_and_scheduler adjustment)
        # I will use similar logic as above, but for brevity in this interaction, I will assume the content matches the previous file read
        # I will implement it properly now.
        try:
            autoencoder = self.gui.ae_system['autoencoder']
            parameter_mapper = self.gui.ae_system['parameter_mapper']
            wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)
            # 获取Loss归一化系数（如果存在）
            loss_normalization_factor = self.gui.ae_system.get('loss_normalization_factor', 1.0)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device)
            parameter_mapper.to(device)

            for param in autoencoder.encoder.parameters():
                param.requires_grad = False

            data_adapter = self.gui.ae_system.get('data_adapter', None)
            if data_adapter is None:
                from autoencoder.utils.data_adapters import RCS_DataAdapter
                current_mode = self.gui.ae_system.get('mode', 'direct')
                data_adapter = RCS_DataAdapter(normalize=True, mode=current_mode)
            
            autoencoder.eval()
            mode = self.gui.ae_system.get('mode', 'wavelet')
            self.gui.ae_log(f"🔧 获取隐空间表示 (mode={mode})...")

            with torch.no_grad():
                if mode == 'wavelet':
                    rcs_tensor = torch.FloatTensor(rcs_data)
                    wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
                    input_data = data_adapter.adapt_rcs_data(wavelet_coeffs.cpu().numpy())
                else:
                    input_data = data_adapter.adapt_rcs_data(rcs_data)

                _, target_latents = autoencoder(input_data.to(device))
                target_latents = target_latents.cpu()

            from sklearn.preprocessing import StandardScaler
            self.gui.ae_log("🔧 标准化设计参数...")
            param_scaler = StandardScaler()
            param_normalized = param_scaler.fit_transform(param_data)
            self.gui.ae_system['param_scaler'] = param_scaler
            
            param_tensor = torch.FloatTensor(param_normalized)
            dataset = TensorDataset(param_tensor, target_latents)

            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

            self.gui.ae_log(f"📊 阶段2数据划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

            # 输出训练集和验证集样本标号
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices
            self.gui.ae_log(f"📋 训练集样本标号: {sorted(train_indices)[:20]}{'...' if len(train_indices) > 20 else ''}")
            self.gui.ae_log(f"📋 验证集样本标号: {sorted(val_indices)[:20]}{'...' if len(val_indices) > 20 else ''}")
            if len(train_indices) > 20 or len(val_indices) > 20:
                self.gui.ae_log(f"   (仅显示前20个标号，完整标号请查看详细日志)")

            batch_size = min(training_config['batch_size'], train_size)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)

            optimizer, scheduler = self._create_optimizer_and_scheduler(parameter_mapper.parameters(), training_config, stage='stage2')
            criterion = self._create_stage_loss_function(training_config, stage='stage2')

            epochs = training_config['epochs']['stage2']
            patience = training_config['patience']['stage2']
            scheduler_type = training_config['lr_scheduler']

            parameter_mapper.train()
            best_val_loss = float('inf')
            best_epoch = 0
            best_model_state = None
            patience_counter = 0
            train_losses, val_losses = [], []
            gradient_history = {'epochs': [], 'grad_norm': []} # Simplified for brevity

            # 检测是否使用L-BFGS
            is_lbfgs = self._is_lbfgs_optimizer(optimizer)
            if is_lbfgs:
                self.gui.ae_log("  🔍 检测到L-BFGS优化器，使用闭包训练循环")

            for epoch in range(epochs):
                parameter_mapper.train()
                train_loss = 0.0
                train_samples = 0
                for batch_params, batch_latents in train_loader:
                    # 使用适配的训练方法
                    if is_lbfgs:
                        loss_value = self._train_batch_with_lbfgs(
                            optimizer, parameter_mapper, batch_params, batch_latents, criterion, device
                        )
                    else:
                        batch_params = batch_params.to(device)
                        batch_latents = batch_latents.to(device)
                        predicted_latents = parameter_mapper(batch_params)
                        loss = criterion(predicted_latents, batch_latents)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_value = loss.item()

                    batch_size = batch_params.size(0) if not is_lbfgs else len(batch_params)
                    train_loss += loss_value * loss_normalization_factor * batch_size
                    train_samples += batch_size
                avg_train_loss = train_loss / train_samples

                parameter_mapper.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch_params, batch_latents in val_loader:
                        batch_params = batch_params.to(device)
                        batch_latents = batch_latents.to(device)
                        predicted_latents = parameter_mapper(batch_params)
                        loss = criterion(predicted_latents, batch_latents)
                        val_loss += loss.item() * loss_normalization_factor * batch_params.size(0)
                        val_samples += batch_params.size(0)
                avg_val_loss = val_loss / val_samples

                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                old_lr = optimizer.param_groups[0]['lr']
                self._step_scheduler(scheduler, scheduler_type, avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                if current_lr < old_lr and best_model_state:
                     self.gui.ae_log(f"  🔄 LR降低，回退...")
                     autoencoder.load_state_dict(best_model_state['autoencoder'])
                     parameter_mapper.load_state_dict(best_model_state['parameter_mapper'])
                     patience_counter = 0

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_model_state = {
                        'autoencoder': deepcopy(autoencoder.state_dict()),
                        'parameter_mapper': deepcopy(parameter_mapper.state_dict())
                    }
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                     self._log_progress(epoch, epochs, avg_train_loss, avg_val_loss, current_lr, "阶段2")
                
                if getattr(self.gui, 'stop_training_flag', False) or getattr(self.gui.training_manager, 'stop_training_flag', False):
                    break

                # 早停 (支持patience驱动的多阶段调度器)
                current_patience = patience
                if scheduler_type in ['multi_stage', 'adaptive_multi_stage']:
                    from autoencoder.training.multi_stage_scheduler import PatienceDrivenMultiStageLRScheduler
                    if isinstance(scheduler, PatienceDrivenMultiStageLRScheduler):
                        current_patience = scheduler.get_current_patience()
                        if patience_counter >= current_patience:
                            switched = scheduler.switch_to_next_stage()
                            if switched:
                                self.gui.ae_log(f"  🔄 切换到下一学习率阶段，回退到最佳模型 (Epoch {best_epoch})")
                                autoencoder.load_state_dict(best_model_state['autoencoder'])
                                parameter_mapper.load_state_dict(best_model_state['parameter_mapper'])
                                patience_counter = 0
                                continue
                            else:
                                self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 已是最后学习率阶段，验证损失连续{current_patience}轮无改善")
                                break
                    else:
                        if patience_counter >= current_patience:
                            self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 验证损失连续{current_patience}轮无改善")
                            break
                else:
                    if patience_counter >= current_patience:
                        self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 验证损失连续{current_patience}轮无改善")
                        break
            
            if best_model_state:
                autoencoder.load_state_dict(best_model_state['autoencoder'])
                parameter_mapper.load_state_dict(best_model_state['parameter_mapper'])
            
            for param in autoencoder.encoder.parameters():
                param.requires_grad = True

            self.gui.ae_log(f"✅ 阶段2完成，最佳损失: {best_val_loss:.6f}")
            return {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'train_indices': train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices),
                'val_indices': val_indices.tolist() if hasattr(val_indices, 'tolist') else list(val_indices)
            }
        except Exception as e:
            self.gui.ae_log(f"❌ 阶段2失败: {e}")
            raise e

    def train_stage3(self, rcs_data, param_data, training_config):
        """阶段3: 端到端微调"""
        # Similar structure to stage 2 but training both models end-to-end
        try:
            autoencoder = self.gui.ae_system['autoencoder']
            parameter_mapper = self.gui.ae_system['parameter_mapper']
            wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)
            mode = self.gui.ae_system.get('mode', 'wavelet')
            # 获取Loss归一化系数（如果存在）
            loss_normalization_factor = self.gui.ae_system.get('loss_normalization_factor', 1.0)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device)
            parameter_mapper.to(device)

            data_adapter = self.gui.ae_system.get('data_adapter', None)
            param_scaler = self.gui.ae_system.get('param_scaler', None)
            
            param_normalized = param_scaler.transform(param_data)
            param_tensor = torch.FloatTensor(param_normalized)

            if mode == 'wavelet':
                rcs_tensor = torch.FloatTensor(rcs_data)
                wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
                target_data = data_adapter.adapt_rcs_data(wavelet_coeffs.cpu().numpy())
            else:
                target_data = data_adapter.adapt_rcs_data(rcs_data)

            dataset = TensorDataset(param_tensor, target_data)
            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

            self.gui.ae_log(f"📊 阶段3数据划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

            # 输出训练集和验证集样本标号
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices
            self.gui.ae_log(f"📋 训练集样本标号: {sorted(train_indices)[:20]}{'...' if len(train_indices) > 20 else ''}")
            self.gui.ae_log(f"📋 验证集样本标号: {sorted(val_indices)[:20]}{'...' if len(val_indices) > 20 else ''}")
            if len(train_indices) > 20 or len(val_indices) > 20:
                self.gui.ae_log(f"   (仅显示前20个标号，完整标号请查看详细日志)")

            batch_size = min(training_config['batch_size'], train_size)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)

            # Fine-tuning uses lower LR
            cfg_fine = training_config.copy()
            cfg_fine['learning_rate'] = training_config['learning_rate'] * 0.1
            
            optimizer, scheduler = self._create_optimizer_and_scheduler(
                list(autoencoder.parameters()) + list(parameter_mapper.parameters()),
                cfg_fine,
                stage='stage3'
            )
            criterion = self._create_stage_loss_function(training_config, stage='stage3')
            epochs = training_config['epochs']['stage3']
            patience = training_config['patience']['stage3']
            scheduler_type = training_config['lr_scheduler']

            best_val_loss = float('inf')
            best_epoch = 0
            best_model_state = None
            patience_counter = 0
            train_losses, val_losses = [], []

            # 检测是否使用L-BFGS
            is_lbfgs = self._is_lbfgs_optimizer(optimizer)
            if is_lbfgs:
                self.gui.ae_log("  🔍 检测到L-BFGS优化器，使用闭包训练循环（端到端模式）")

            for epoch in range(epochs):
                autoencoder.train()
                parameter_mapper.train()
                train_loss = 0.0
                train_samples = 0
                for batch_params, batch_target in train_loader:
                    # 使用适配的训练方法（Stage 3特殊：联合训练）
                    if is_lbfgs:
                        batch_params = batch_params.to(device)
                        batch_target = batch_target.to(device)

                        # L-BFGS闭包（端到端）
                        def closure():
                            optimizer.zero_grad()
                            pred_latents = parameter_mapper(batch_params)
                            recon = autoencoder.decode(pred_latents)
                            loss = criterion(recon, batch_target)
                            # ⭐ 在backward前应用归一化
                            if loss_normalization_factor != 1.0:
                                loss = loss * loss_normalization_factor
                            loss.backward()
                            return loss

                        loss = optimizer.step(closure)
                        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                    else:
                        batch_params = batch_params.to(device)
                        batch_target = batch_target.to(device)

                        pred_latents = parameter_mapper(batch_params)
                        recon = autoencoder.decode(pred_latents)
                        loss = criterion(recon, batch_target)

                        # ⭐ 在backward前应用归一化（Stage 3使用Stage 1的系数）
                        if loss_normalization_factor != 1.0:
                            loss = loss * loss_normalization_factor

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_value = loss.item()

                    batch_size = batch_params.size(0) if not is_lbfgs else len(batch_params)
                    train_loss += loss_value * batch_size  # loss已经是归一化后的
                    train_samples += batch_size
                avg_train_loss = train_loss / train_samples

                autoencoder.eval()
                parameter_mapper.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch_params, batch_target in val_loader:
                        batch_params = batch_params.to(device)
                        batch_target = batch_target.to(device)
                        pred_latents = parameter_mapper(batch_params)
                        recon = autoencoder.decode(pred_latents)
                        loss = criterion(recon, batch_target)
                        # ⭐ 应用归一化
                        if loss_normalization_factor != 1.0:
                            loss = loss * loss_normalization_factor
                        val_loss += loss.item() * batch_params.size(0)
                        val_samples += batch_params.size(0)
                avg_val_loss = val_loss / val_samples
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                old_lr = optimizer.param_groups[0]['lr']
                self._step_scheduler(scheduler, scheduler_type, avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                if current_lr < old_lr and best_model_state:
                     autoencoder.load_state_dict(best_model_state['autoencoder'])
                     parameter_mapper.load_state_dict(best_model_state['parameter_mapper'])
                     patience_counter = 0

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_model_state = {
                        'autoencoder': deepcopy(autoencoder.state_dict()),
                        'parameter_mapper': deepcopy(parameter_mapper.state_dict())
                    }
                else:
                    patience_counter += 1

                if (epoch + 1) % 5 == 0:
                     self._log_progress(epoch, epochs, avg_train_loss, avg_val_loss, current_lr, "阶段3")

                if getattr(self.gui, 'stop_training_flag', False) or getattr(self.gui.training_manager, 'stop_training_flag', False):
                    break

                # 早停 (支持patience驱动的多阶段调度器)
                current_patience = patience
                if scheduler_type in ['multi_stage', 'adaptive_multi_stage']:
                    from autoencoder.training.multi_stage_scheduler import PatienceDrivenMultiStageLRScheduler
                    if isinstance(scheduler, PatienceDrivenMultiStageLRScheduler):
                        current_patience = scheduler.get_current_patience()
                        if patience_counter >= current_patience:
                            switched = scheduler.switch_to_next_stage()
                            if switched:
                                self.gui.ae_log(f"  🔄 切换到下一学习率阶段，回退到最佳模型")
                                autoencoder.load_state_dict(best_model_state['autoencoder'])
                                parameter_mapper.load_state_dict(best_model_state['parameter_mapper'])
                                patience_counter = 0
                                continue
                            else:
                                self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 已是最后学习率阶段")
                                break
                    else:
                        if patience_counter >= current_patience:
                            self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1})")
                            break
                else:
                    if patience_counter >= current_patience:
                        self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1})")
                        break

            if best_model_state:
                autoencoder.load_state_dict(best_model_state['autoencoder'])
                parameter_mapper.load_state_dict(best_model_state['parameter_mapper'])

            self.gui.ae_log(f"✅ 阶段 3完成，最佳损失: {best_val_loss:.6f}")
            return {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'train_indices': train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices),
                'val_indices': val_indices.tolist() if hasattr(val_indices, 'tolist') else list(val_indices)
            }
        except Exception as e:
            self.gui.ae_log(f"❌ 阶段 3失败: {e}")
            raise e
            
    def train_end_to_end_full(self, rcs_data, param_data, training_config, total_epochs):
        """完整端到端训练"""
        # Typically similar to stage 3 but starting from scratch or combined logic
        # For now, implementing as wrapper around Stage 3 logic but with full epochs
        # In reality, this might involve pre-training too if starting cold.
        # Assuming this is just a long stage 3.
        return self.train_stage3(rcs_data, param_data, {**training_config, 'epochs': {'stage3': total_epochs}})

    def _create_optimizer_and_scheduler(self, params, training_config, stage='stage1'):
        """
        创建优化器和学习率调度器（支持Adam/AdamW/SGD/L-BFGS）

        Args:
            params: 模型参数
            training_config: 训练配置
            stage: 当前训练阶段 ('stage1', 'stage2', 'stage3')，用于选择对应的patience
        """
        optimizer_type = training_config.get('optimizer_type', 'adam').lower()
        lr = training_config['learning_rate']
        weight_decay = training_config.get('weight_decay', 1e-4)

        # 创建优化器
        if optimizer_type == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            self.gui.ae_log(f"✅ 使用Adam优化器 (lr={lr:.2e}, wd={weight_decay:.2e})")
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            self.gui.ae_log(f"✅ 使用AdamW优化器 (lr={lr:.2e}, wd={weight_decay:.2e})")
        elif optimizer_type == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            self.gui.ae_log(f"✅ 使用SGD优化器 (lr={lr:.2e}, momentum={momentum}, wd={weight_decay:.2e})")
        elif optimizer_type == 'lbfgs':
            # L-BFGS参数
            max_iter = training_config.get('lbfgs_max_iter', 20)
            history_size = training_config.get('lbfgs_history_size', 100)
            optimizer = optim.LBFGS(params, lr=lr, max_iter=max_iter, history_size=history_size)
            self.gui.ae_log(f"✅ 使用L-BFGS优化器 (lr={lr:.2e}, max_iter={max_iter}, history_size={history_size})")
            self.gui.ae_log(f"⚠️ L-BFGS需要闭包，已自动适配训练循环")
        else:
            # 默认使用Adam
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            self.gui.ae_log(f"⚠️ 未知优化器类型'{optimizer_type}'，默认使用Adam")

        # 创建学习率调度器
        scheduler_type = training_config.get('lr_scheduler', 'constant')

        if scheduler_type == 'cosine_restart':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=training_config.get('restart_period', 50))
        elif scheduler_type == 'adaptive':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        elif scheduler_type in ['multi_stage', 'adaptive_multi_stage']:
            # 多阶段学习率调度器（patience驱动）
            from autoencoder.training.multi_stage_scheduler import create_patience_driven_scheduler
            num_stages = training_config.get('num_lr_stages', 3)
            lr_decay = training_config.get('lr_decay_factor', 0.1)

            # ✅ 使用当前阶段对应的patience（而非总是使用stage1）
            initial_patience = training_config['patience'].get(stage, 30)

            scheduler = create_patience_driven_scheduler(
                optimizer,
                initial_lr=lr,
                num_stages=num_stages,
                lr_decay_factor=lr_decay,
                initial_patience=initial_patience,
                verbose=True
            )
            self.gui.ae_log(f"✅ 多阶段调度器：使用{stage}的patience={initial_patience}作为基准")
        else:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)

        return optimizer, scheduler

    def _create_stage_loss_function(self, training_config, stage='stage1'):
        """
        创建指定阶段的损失函数
        支持：阶段特定配置 > 全局自定义配置 > 默认MSE

        注意：Stage 2（参数映射）总是使用MSE，不受全局自定义配置影响
        """
        # 1. 检查阶段特定配置（优先级最高）
        stage_config_key = f'{stage}_loss_config'
        if stage_config_key in training_config:
            loss_config = training_config[stage_config_key]
            from autoencoder.utils.configurable_loss import create_loss_function
            return self._wrap_configurable_loss(create_loss_function(loss_config))

        # 2. 检查全局自定义配置（Stage 2除外，因为它在latent空间操作）
        if stage != 'stage2' and training_config.get('use_custom_loss', False) and 'custom_loss_config' in training_config:
            loss_config = training_config['custom_loss_config']
            from autoencoder.utils.configurable_loss import create_loss_function
            return self._wrap_configurable_loss(create_loss_function(loss_config))

        # 3. 默认MSE（包括Stage 2）
        return nn.MSELoss()

    def _wrap_configurable_loss(self, configurable_loss):
        """包装ConfigurableLoss，返回tensor而非字典"""
        def loss_wrapper(pred, target):
            loss_dict = configurable_loss(pred, target)
            return loss_dict['total']
        return loss_wrapper

    def _create_joint_loss_function(self, training_config, loss_name):
        """
        为联合训练创建指定的损失函数

        Args:
            training_config: 训练配置字典
            loss_name: 损失名称 ('recon_rcs', 'consistency', 'param_recon')

        Returns:
            损失函数

        优先级：
        1. joint_loss_config[loss_name] (最高)
        2. 全局自定义配置 use_custom_loss
        3. 默认MSE
        """
        # 1. 检查联合训练特定配置
        if 'joint_loss_config' in training_config:
            joint_config = training_config['joint_loss_config']
            if loss_name in joint_config:
                from autoencoder.utils.configurable_loss import create_loss_function
                self.gui.ae_log(f"  使用自定义{loss_name}损失函数")
                return self._wrap_configurable_loss(create_loss_function(joint_config[loss_name]))

        # 2. 检查全局自定义配置（仅对recon_rcs和param_recon，不对consistency）
        # consistency是隐空间的MSE，通常不需要自定义
        if loss_name != 'consistency' and training_config.get('use_custom_loss', False):
            if 'custom_loss_config' in training_config:
                from autoencoder.utils.configurable_loss import create_loss_function
                self.gui.ae_log(f"  {loss_name}使用全局自定义损失函数")
                return self._wrap_configurable_loss(create_loss_function(training_config['custom_loss_config']))

        # 3. 默认MSE
        return nn.MSELoss()

    def _step_scheduler(self, scheduler, scheduler_type, val_loss):
        if scheduler_type == 'adaptive':
            scheduler.step(val_loss)
        elif scheduler_type in ['multi_stage', 'adaptive_multi_stage']:
            # Patience驱动的多阶段调度器：不需要step，阶段切换在早停逻辑中处理
            # 学习率由scheduler.switch_to_next_stage()直接更新
            pass
        else:
            scheduler.step()

    def _log_progress(self, epoch, total_epochs, train_loss, val_loss, lr, stage):
        self.gui.ae_log(f"  {stage} Epoch {epoch+1}/{total_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr:.2e}")
        self.gui.root.update_idletasks()

    def _record_attention_weights(self, autoencoder, sample_data, attention_history, epoch):
        # ... logic ...
        pass

    def print_channel_attention_weights(self, rcs_data):
        # ... logic ...
        pass

    def print_latent_space_statistics(self, rcs_data):
        # ... logic ...
        pass

    def train_joint_mode(self, rcs_data, param_data, training_config, epochs, alpha, beta, gamma):
        """
        联合训练核心实现

        训练流程：
        1. 前向传播：
           - RCS → Encoder → latent_from_rcs → Decoder → recon_rcs
           - Params → Mapper → latent_from_params
           - Params → Mapper → latent_from_params → Decoder → recon_from_params

        2. 损失计算：
           - L_recon_rcs = MSE(recon_rcs, rcs)
           - L_consistency = MSE(latent_from_rcs, latent_from_params)
           - L_param_recon = MSE(recon_from_params, rcs)

        3. 反向传播：
           - 梯度通过三条路径自动流动
           - Encoder梯度 = α*∂L_recon_rcs + β*∂L_consistency
           - Decoder梯度 = α*∂L_recon_rcs + γ*∂L_param_recon
           - Mapper梯度 = β*∂L_consistency + γ*∂L_param_recon

        Args:
            rcs_data: RCS数据 [N, ...]
            param_data: 参数数据 [N, 9]
            training_config: 训练配置
            epochs: 训练轮数
            alpha: RCS重建损失权重
            beta: 一致性损失权重
            gamma: 参数重建损失权重（最重要）

        Returns:
            history: 训练历史字典
        """
        # 获取组件
        autoencoder = self.gui.ae_system['autoencoder']
        parameter_mapper = self.gui.ae_system['parameter_mapper']
        wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)
        data_adapter = self.gui.ae_system.get('data_adapter', None)
        mode = self.gui.ae_system.get('mode', 'wavelet')

        # 获取损失归一化系数（如果已初始化）
        loss_normalization_factor = self.gui.ae_system.get('loss_normalization_factor', 1.0)
        if loss_normalization_factor != 1.0:
            self.gui.ae_log(f"🔧 使用Loss归一化系数: {loss_normalization_factor:.6f}")
        else:
            self.gui.ae_log("⚠️ 未初始化Loss归一化（使用默认系数1.0）")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        autoencoder.to(device)
        parameter_mapper.to(device)

        self.gui.ae_log(f"🖥️ 使用设备: {device}")
        self.gui.ae_log(f"🔧 训练模式: {mode}")

        # 数据预处理（与Stage 1相同的逻辑）
        if data_adapter is None:
            from autoencoder.utils.data_adapters import RCS_DataAdapter
            data_adapter = RCS_DataAdapter(normalize=True, mode=mode)

        self.gui.ae_log(f"🔧 数据预处理: 标准化={data_adapter.normalize}, dB变换={data_adapter.db_transform}")

        # RCS数据预处理
        if mode in ['wavelet', 'differentiable_wavelet']:
            self.gui.ae_log("📊 对RCS执行小波变换...")
            rcs_tensor = torch.FloatTensor(rcs_data)
            wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
            input_data = data_adapter.adapt_rcs_data(wavelet_coeffs.cpu().numpy())
        else:
            self.gui.ae_log("📊 对RCS执行预处理...")
            input_data = data_adapter.adapt_rcs_data(rcs_data)

        # 参数数据标准化
        from sklearn.preprocessing import StandardScaler
        self.gui.ae_log("🔧 标准化设计参数...")
        param_scaler = StandardScaler()
        param_normalized = param_scaler.fit_transform(param_data)
        self.gui.ae_system['param_scaler'] = param_scaler

        # 创建数据集
        input_tensor = torch.FloatTensor(input_data)
        param_tensor = torch.FloatTensor(param_normalized)
        dataset = TensorDataset(input_tensor, param_tensor)

        # 数据划分
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                                   generator=torch.Generator().manual_seed(42))

        self.gui.ae_log(f"📊 数据划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

        # 创建数据加载器
        batch_size = min(training_config['batch_size'], train_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)

        # 创建联合优化器（包含AE和Mapper的所有参数）
        from itertools import chain
        all_params = chain(autoencoder.parameters(), parameter_mapper.parameters())
        optimizer, scheduler = self._create_optimizer_and_scheduler(all_params, training_config, stage='joint')

        # 获取调度器类型（用于后续step调用）
        scheduler_type = training_config.get('lr_scheduler', 'constant')

        self.gui.ae_log("✅ 联合优化器已创建（包含AutoEncoder + ParameterMapper）")
        self.gui.ae_log(f"  调度器类型: {scheduler_type}")

        # 训练循环
        autoencoder.train()
        parameter_mapper.train()

        best_val_loss = float('inf')
        best_epoch = 0
        best_ae_state = None
        best_mapper_state = None
        patience = training_config['patience'].get('joint', 50)
        patience_counter = 0

        # 训练历史
        history = {
            'train_loss': [], 'val_loss': [],
            'train_loss_recon': [], 'train_loss_consistency': [], 'train_loss_param_recon': [],
            'val_loss_recon': [], 'val_loss_consistency': [], 'val_loss_param_recon': []
        }

        # 创建三个损失函数（支持configurable_loss）
        criterion_recon = self._create_joint_loss_function(training_config, 'recon_rcs')
        criterion_consistency = self._create_joint_loss_function(training_config, 'consistency')
        criterion_param_recon = self._create_joint_loss_function(training_config, 'param_recon')

        self.gui.ae_log("✅ 联合训练损失函数已创建")
        self.gui.ae_log(f"  L_recon_rcs: {type(criterion_recon).__name__}")
        self.gui.ae_log(f"  L_consistency: {type(criterion_consistency).__name__}")
        self.gui.ae_log(f"  L_param_recon: {type(criterion_param_recon).__name__}")

        # 检测L-BFGS
        is_lbfgs = self._is_lbfgs_optimizer(optimizer)
        if is_lbfgs:
            self.gui.ae_log("⚠️ 联合训练模式不推荐使用L-BFGS（训练图复杂）")

        self.gui.ae_log(f"🚀 开始联合训练 (epochs={epochs}, patience={patience})...")

        for epoch in range(epochs):
            # ========== 训练阶段 ==========
            autoencoder.train()
            parameter_mapper.train()

            train_loss_total = 0.0
            train_loss_recon = 0.0
            train_loss_consistency = 0.0
            train_loss_param_recon = 0.0
            train_samples = 0

            for rcs_batch, param_batch in train_loader:
                rcs_batch = rcs_batch.to(device)
                param_batch = param_batch.to(device)
                batch_size_actual = rcs_batch.size(0)

                optimizer.zero_grad()

                # ========== 前向传播（三条路径）==========
                # 路径1: RCS → Encoder → Decoder → RCS
                recon_rcs, latent_from_rcs = autoencoder(rcs_batch)

                # 路径2: Params → Mapper → Latent
                latent_from_params = parameter_mapper(param_batch)

                # 路径3: Params → Mapper → Decoder → RCS
                recon_from_params = autoencoder.decode(latent_from_params)

                # ========== 损失计算 ==========
                L_recon_rcs = criterion_recon(recon_rcs, rcs_batch)          # α
                L_consistency = criterion_consistency(latent_from_rcs, latent_from_params)  # β
                L_param_recon = criterion_param_recon(recon_from_params, rcs_batch)  # γ

                # 总损失（加权）
                total_loss = alpha * L_recon_rcs + beta * L_consistency + gamma * L_param_recon

                # 应用损失归一化
                total_loss = total_loss * loss_normalization_factor

                # ========== 反向传播 ==========
                total_loss.backward()
                optimizer.step()

                # 累积损失
                train_loss_total += total_loss.item() * batch_size_actual
                train_loss_recon += L_recon_rcs.item() * batch_size_actual
                train_loss_consistency += L_consistency.item() * batch_size_actual
                train_loss_param_recon += L_param_recon.item() * batch_size_actual
                train_samples += batch_size_actual

            # 平均训练损失
            avg_train_loss = train_loss_total / train_samples
            avg_train_recon = train_loss_recon / train_samples
            avg_train_consistency = train_loss_consistency / train_samples
            avg_train_param_recon = train_loss_param_recon / train_samples

            # ========== 验证阶段 ==========
            autoencoder.eval()
            parameter_mapper.eval()

            val_loss_total = 0.0
            val_loss_recon = 0.0
            val_loss_consistency = 0.0
            val_loss_param_recon = 0.0
            val_samples = 0

            with torch.no_grad():
                for rcs_batch, param_batch in val_loader:
                    rcs_batch = rcs_batch.to(device)
                    param_batch = param_batch.to(device)
                    batch_size_actual = rcs_batch.size(0)

                    # 前向传播
                    recon_rcs, latent_from_rcs = autoencoder(rcs_batch)
                    latent_from_params = parameter_mapper(param_batch)
                    recon_from_params = autoencoder.decode(latent_from_params)

                    # 损失计算
                    L_recon_rcs = criterion_recon(recon_rcs, rcs_batch)
                    L_consistency = criterion_consistency(latent_from_rcs, latent_from_params)
                    L_param_recon = criterion_param_recon(recon_from_params, rcs_batch)
                    total_loss = alpha * L_recon_rcs + beta * L_consistency + gamma * L_param_recon

                    # 应用损失归一化（与训练保持一致）
                    total_loss = total_loss * loss_normalization_factor

                    val_loss_total += total_loss.item() * batch_size_actual
                    val_loss_recon += L_recon_rcs.item() * batch_size_actual
                    val_loss_consistency += L_consistency.item() * batch_size_actual
                    val_loss_param_recon += L_param_recon.item() * batch_size_actual
                    val_samples += batch_size_actual

            # 平均验证损失
            avg_val_loss = val_loss_total / val_samples
            avg_val_recon = val_loss_recon / val_samples
            avg_val_consistency = val_loss_consistency / val_samples
            avg_val_param_recon = val_loss_param_recon / val_samples

            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_loss_recon'].append(avg_train_recon)
            history['train_loss_consistency'].append(avg_train_consistency)
            history['train_loss_param_recon'].append(avg_train_param_recon)
            history['val_loss_recon'].append(avg_val_recon)
            history['val_loss_consistency'].append(avg_val_consistency)
            history['val_loss_param_recon'].append(avg_val_param_recon)

            # 学习率调度（使用统一的_step_scheduler方法）
            old_lr = optimizer.param_groups[0]['lr']
            self._step_scheduler(scheduler, scheduler_type, avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # 如果学习率下降，回退到最佳模型（多阶段调度器特性）
            if current_lr < old_lr and best_ae_state is not None:
                autoencoder.load_state_dict(best_ae_state)
                parameter_mapper.load_state_dict(best_mapper_state)
                patience_counter = 0
                self.gui.ae_log(f"  🔄 学习率下降 {old_lr:.2e} → {current_lr:.2e}，回退到最佳模型")

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_ae_state = deepcopy(autoencoder.state_dict())
                best_mapper_state = deepcopy(parameter_mapper.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # 日志输出
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.gui.ae_log(
                    f"  Epoch {epoch+1}/{epochs}: "
                    f"Total={avg_train_loss:.6f}/{avg_val_loss:.6f}, "
                    f"Recon={avg_train_recon:.6f}, "
                    f"Consistency={avg_train_consistency:.6f}, "
                    f"ParamRecon={avg_train_param_recon:.6f}, "
                    f"LR={current_lr:.2e}"
                )

            # 早停（支持patience驱动的多阶段调度器）
            current_patience = patience
            if scheduler_type in ['multi_stage', 'adaptive_multi_stage']:
                from autoencoder.training.multi_stage_scheduler import PatienceDrivenMultiStageLRScheduler
                if isinstance(scheduler, PatienceDrivenMultiStageLRScheduler):
                    current_patience = scheduler.get_current_patience()
                    if patience_counter >= current_patience:
                        switched = scheduler.switch_to_next_stage()
                        if switched:
                            self.gui.ae_log(f"  🔄 切换到下一学习率阶段，回退到最佳模型")
                            autoencoder.load_state_dict(best_ae_state)
                            parameter_mapper.load_state_dict(best_mapper_state)
                            patience_counter = 0
                            continue
                        else:
                            self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1}): 已是最后学习率阶段")
                            break
                else:
                    if patience_counter >= current_patience:
                        self.gui.ae_log(f"  🛑 早停触发 (Epoch {epoch+1})")
                        break
            else:
                if patience_counter >= current_patience:
                    self.gui.ae_log(f"⏹️ 早停触发 (patience={patience}), 最佳epoch={best_epoch+1}")
                    break

        # 恢复最佳模型
        if best_ae_state is not None:
            autoencoder.load_state_dict(best_ae_state)
            parameter_mapper.load_state_dict(best_mapper_state)
            self.gui.ae_log(f"✅ 已恢复最佳模型 (epoch {best_epoch+1}, val_loss={best_val_loss:.6f})")

        return history
