"""
训练管理器 (Controller)
处理所有训练相关功能，委托具体任务给 AETrainer 和 LegacyTrainer
"""

import numpy as np
import torch
import os
import threading
from datetime import datetime
from tkinter import messagebox
import tkinter as tk

# 导入新的 Trainer
from gui_managers.trainers.legacy_trainer import LegacyTrainer
from gui_managers.trainers.ae_trainer import AETrainer

# 导入训练相关模块 (Config helpers need these)
from training import CrossValidationTrainer, RCSDataset
from wavelet_network import create_model, create_loss_function
from autoencoder.utils.configurable_loss import create_loss_function as create_configurable_loss


class TrainingManager:
    """训练管理器 - 负责协调训练任务"""

    def __init__(self, parent_gui):
        """
        初始化训练管理器

        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui
        self.batch_experiment_mode = False
        self.training_log_buffer = []
        self.stop_training_flag = False
        self.training_thread = None
        
        # 初始化 Trainers
        self.legacy_trainer = LegacyTrainer(parent_gui)
        self.ae_trainer = AETrainer(parent_gui)

    # --- Legacy Training Delegation ---

    def _train_model(self):
        """训练模型（简单/交叉验证）"""
        self.stop_training_flag = False
        self.gui.stop_training_flag = False  # 确保重置GUI层面的停止标志
        # LegacyTrainer directly accesses gui state, so just calling it is enough
        self.legacy_trainer.train_model()

    def _training_finished(self):
        # 此方法被 LegacyTrainer 调用，或者作为 GUI 回调
        self.legacy_trainer._training_finished()

    def _set_random_seeds(self, seed=42):
        self.legacy_trainer._set_random_seeds(seed)

    def _initialize_cuda_safely(self):
        self.legacy_trainer._initialize_cuda_safely()

    # --- AutoEncoder Training Delegation ---

    def start_ae_training(self):
        """开始AutoEncoder训练"""
        try:
            if self.gui.ae_system is None:
                messagebox.showwarning("警告", "请先创建AutoEncoder系统!")
                return

            if not self.gui.data_loaded:
                messagebox.showwarning("警告", "请先加载数据!")
                return

            # 获取训练模式（提前获取用于判断是否重新初始化）
            training_mode = self.gui.ae_training_mode.get()

            # 重新初始化模型权重（但"仅Stage 2"模式除外）
            if self.gui.ae_model_loaded:
                if training_mode == "仅Stage 2":
                    # 仅Stage 2模式：保留AutoEncoder权重，仅重新初始化ParameterMapper
                    self.gui.ae_log("🔄 仅Stage 2模式：保留AutoEncoder权重")

                    import torch.nn as nn
                    def init_weights(m):
                        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)

                    if 'parameter_mapper' in self.gui.ae_system:
                        self.gui.ae_system['parameter_mapper'].apply(init_weights)
                        self.gui.ae_log("✅ 参数映射器权重已重新初始化（随机）")

                    # 不清空训练历史，保留Stage 1的历史记录
                else:
                    # 其他模式：完全重新初始化
                    self.gui.ae_log("🔄 检测到已加载模型，重新初始化权重...")
                    self.gui.ae_log("  💡 提示：使用'继续训练'按钮可从加载的权重继续训练")

                    import torch.nn as nn
                    def init_weights(m):
                        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)

                    self.gui.ae_system['autoencoder'].apply(init_weights)
                    if 'parameter_mapper' in self.gui.ae_system:
                        self.gui.ae_system['parameter_mapper'].apply(init_weights)
                    self.gui.ae_log("✅ 模型权重已重新初始化（随机）")

                    self.gui.ae_training_history = {}
                    self.gui.ae_log("  训练历史已清空，将从头开始训练")

            self.gui.ae_log("🚀 开始AutoEncoder训练...")

            # 创建配置
            training_config = self._create_ae_training_config()

            self.gui.ae_log(f"📊 训练配置:")
            self.gui.ae_log(f"  批次大小: {training_config['batch_size']}")
            self.gui.ae_log(f"  学习率: {training_config['learning_rate']} (min: {training_config['min_lr']})")
            self.gui.ae_log(f"  调度策略: {training_config['lr_scheduler']}")
            self.gui.ae_log(f"  损失函数: {'自定义配置' if training_config['use_custom_loss'] else '标准MSE'}")

            # 检查"仅Stage 2"模式是否加载了Stage 1模型（training_mode已在line 77获取）
            if training_mode == "仅Stage 2":
                if not self.gui.ae_model_loaded:
                    messagebox.showwarning("警告", "仅Stage 2模式需要先加载已训练好的Stage 1模型！\n\n请先：\n1. 加载Stage 1模型\n2. 选择'继续训练'")
                    return

                # 检查加载的模型是否包含训练好的AutoEncoder
                loaded_training_mode = self.gui.ae_training_history.get('training_mode', 'unknown')
                if loaded_training_mode not in ['stage1_only', 'three_stage']:
                    messagebox.showwarning("警告", f"当前加载的模型训练模式为'{loaded_training_mode}'，\n不适合进行Stage 2训练。\n\n建议加载stage1_only或three_stage模式的模型。")
                    return

                self.gui.ae_log("✅ 检测到已加载Stage 1模型，将仅训练参数映射器")
                self.gui.ae_log("  AutoEncoder权重将保持冻结")

            # Log training mode details (simplified)
            self.gui.ae_log(f"  模式: {training_mode}")

            # 获取数据
            if 'rcs_data' not in self.gui.ae_system or 'param_data' not in self.gui.ae_system:
                self.gui.ae_log("❌ 数据未正确集成到AutoEncoder系统")
                messagebox.showerror("错误", "数据未正确集成，请重新创建AutoEncoder系统")
                return

            rcs_data = self.gui.ae_system['rcs_data']
            param_data = self.gui.ae_system['param_data']

            # 启动后台线程
            import threading
            self.training_thread = threading.Thread(
                target=self._run_ae_training_in_background,
                args=(rcs_data, param_data, training_config, training_mode),
                daemon=True
            )
            self.training_thread.start()

        except Exception as e:
            error_msg = f"启动训练失败: {e}"
            self.gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _run_ae_training_in_background(self, rcs_data, param_data, training_config, training_mode):
        """在后台线程运行AutoEncoder训练 (委托给 AETrainer)"""
        try:
            self.stop_training_flag = False
            self.gui.stop_training_flag = False  # 确保重置GUI层面的停止标志
            
            # 使用 AETrainer 的方法
            # 注意：run_three_stage_training 内部会处理 stage1_only
            if training_mode == "三阶段训练":
                self.ae_trainer.run_three_stage_training(rcs_data, param_data, training_config)
            elif training_mode == "仅Stage 1":
                # Ensure config reflects this
                training_config['training_mode'] = 'stage1_only'
                self.ae_trainer.run_three_stage_training(rcs_data, param_data, training_config)
            elif training_mode == "仅Stage 2":
                # 仅训练参数映射器（需要先加载Stage 1模型）
                training_config['training_mode'] = 'stage2_only'
                stage2_history = self.ae_trainer.train_stage2(rcs_data, param_data, training_config)
                # 保存训练历史
                if not hasattr(self.gui, 'ae_training_history'):
                    self.gui.ae_training_history = {'stage_histories': {}}
                self.gui.ae_training_history['stage_histories']['stage2'] = stage2_history
                self.gui.ae_training_history['training_mode'] = 'stage2_only'
                self.gui.ae_system['training_mode'] = 'stage2_only'
                self.gui.ae_log("✅ 参数映射器训练完成！")
            elif training_mode == "联合训练":
                # 联合训练模式：同时训练AE和Mapper，强制隐空间对齐
                training_config['training_mode'] = 'joint_training'
                self.ae_trainer.run_joint_training(rcs_data, param_data, training_config)
            else:
                self.ae_trainer.run_end_to_end_training(rcs_data, param_data, training_config)

            self.gui.root.after(0, self._on_ae_training_completed)

        except Exception as e:
            # 错误已经在 Trainer 中记录，这里主要是确保 UI 更新或异常传递
            import traceback
            traceback.print_exc()
            self.gui.root.after(0, lambda: messagebox.showerror("训练错误", f"{e}"))

    def _on_ae_training_completed(self):
        self.gui.ae_log("✅ 训练流程全部完成!")
        if not self.batch_experiment_mode:
            messagebox.showinfo("成功", "AutoEncoder训练完成!")

    def stop_ae_training(self):
        if not self.training_thread or not self.training_thread.is_alive():
            messagebox.showwarning("警告", "当前没有正在进行的训练")
            return

        self.gui.ae_log("⏹️ 用户请求停止训练...")
        self.stop_training_flag = True
        # Set flag in gui as well since Trainers might check it there
        self.gui.stop_training_flag = True
        
        self.gui.ae_log("⏳ 正在停止训练，请稍候...")
        messagebox.showinfo("提示", "停止信号已发送\n训练将在当前epoch完成后停止")

    def resume_ae_training(self):
        """继续训练AutoEncoder"""
        # Logic is complex, copying simplified version that delegates to AETrainer
        try:
            if self.gui.ae_system is None:
                messagebox.showwarning("警告", "请先创建AutoEncoder系统!")
                return

            if not self.gui.ae_model_loaded:
                messagebox.showwarning("警告", "请先加载模型!")
                return

            self.gui.ae_log("🔄 继续训练...")

            # Restore weights logic (retained here as it interacts with GUI state)
            self.gui.ae_system['autoencoder'].load_state_dict(self.gui.ae_loaded_weights['autoencoder'])
            if 'parameter_mapper' in self.gui.ae_loaded_weights:
                self.gui.ae_system['parameter_mapper'].load_state_dict(self.gui.ae_loaded_weights['parameter_mapper'])

            # ✅ 打印上次训练的最佳loss作为基准
            if hasattr(self.gui, 'ae_training_history') and self.gui.ae_training_history:
                self.gui.ae_log("📊 上次训练的最佳Loss（作为基准）:")
                stage_histories = self.gui.ae_training_history.get('stage_histories', {})
                training_mode = self.gui.ae_training_mode.get()

                if training_mode == '3阶段训练' or training_mode == 'three_stage':
                    # 三阶段训练：打印所有阶段的最佳loss
                    for stage_name in ['stage1', 'stage2', 'stage3']:
                        if stage_name in stage_histories:
                            stage = stage_histories[stage_name]
                            best_loss = stage.get('best_val_loss', 'N/A')
                            if isinstance(best_loss, float):
                                self.gui.ae_log(f"  {stage_name.upper()}: {best_loss:.6f}")
                elif training_mode == '仅Stage 1' or training_mode == 'stage1_only':
                    # Stage 1 Only：只打印stage1
                    if 'stage1' in stage_histories:
                        stage1 = stage_histories['stage1']
                        best_loss = stage1.get('best_val_loss', 'N/A')
                        if isinstance(best_loss, float):
                            self.gui.ae_log(f"  Stage 1: {best_loss:.6f}")
                elif training_mode == '仅Stage 2' or training_mode == 'stage2_only':
                    # Stage 2 Only：打印stage2（如果存在）
                    if 'stage2' in stage_histories:
                        stage2 = stage_histories['stage2']
                        best_loss = stage2.get('best_val_loss', 'N/A')
                        if isinstance(best_loss, float):
                            self.gui.ae_log(f"  Stage 2: {best_loss:.6f}")

                self.gui.ae_log("💡 注意: 继续训练将从best_val_loss=inf开始（每次训练独立评估最佳模型）")
                self.gui.ae_log("  这样可以确保只有真正改进的模型才会被保存")

            training_config = self._create_ae_training_config()
            training_mode = self.gui.ae_training_mode.get()  # 重新获取当前训练模式

            rcs_data = self.gui.ae_system['rcs_data']
            param_data = self.gui.ae_system['param_data']

            import threading
            self.training_thread = threading.Thread(
                target=self._run_ae_training_in_background,
                args=(rcs_data, param_data, training_config, training_mode),
                daemon=True
            )
            self.training_thread.start()

        except Exception as e:
            self.gui.ae_log(f"❌ 继续训练失败: {e}")
            messagebox.showerror("错误", f"{e}")

    def _continue_training_from_stage1(self):
        """从Stage 1模型继续训练Stage 2和Stage 3"""
        try:
            if 'rcs_data' not in self.gui.ae_system:
                return

            rcs_data = self.gui.ae_system['rcs_data']
            param_data = self.gui.ae_system['param_data']
            training_config = self._create_ae_training_config()

            self.gui.ae_log("🚀 从Stage 1模型继续训练...")
            
            # Delegate directly to trainer methods
            stage2_history = self.ae_trainer.train_stage2(rcs_data, param_data, training_config)
            
            if not hasattr(self.gui, 'ae_training_history'):
                self.gui.ae_training_history = {'stage_histories': {}}
            self.gui.ae_training_history['stage_histories']['stage2'] = stage2_history
            
            stage3_history = self.ae_trainer.train_stage3(rcs_data, param_data, training_config)
            self.gui.ae_training_history['stage_histories']['stage3'] = stage3_history

            self.gui.ae_system['training_mode'] = 'three_stage'
            self.gui.ae_log("🎉 完成!")
            
            self.gui.root.after(0, lambda: messagebox.showinfo("成功", "从Stage 1继续训练完成!"))

        except Exception as e:
            self.gui.ae_log(f"❌ 失败: {e}")
            messagebox.showerror("错误", f"{e}")

    # --- Delegated Methods (Compatibility Layer) ---

    def _run_three_stage_training_v2(self, rcs_data, param_data, training_config):
        return self.ae_trainer.run_three_stage_training(rcs_data, param_data, training_config)

    def _run_end_to_end_training_v2(self, rcs_data, param_data, training_config):
        return self.ae_trainer.run_end_to_end_training(rcs_data, param_data, training_config)

    def _train_autoencoder_stage1_v2(self, rcs_data, training_config):
        return self.ae_trainer.train_stage1(rcs_data, training_config)

    def _train_parameter_mapping_stage2_v2(self, rcs_data, param_data, training_config):
        return self.ae_trainer.train_stage2(rcs_data, param_data, training_config)

    def _train_end_to_end_stage3_v2(self, rcs_data, param_data, training_config):
        return self.ae_trainer.train_stage3(rcs_data, param_data, training_config)

    def _train_full_end_to_end_v2(self, rcs_data, param_data, training_config, total_epochs):
        return self.ae_trainer.train_end_to_end_full(rcs_data, param_data, training_config, total_epochs)
        
    def _print_latent_space_statistics(self, rcs_data):
        return self.ae_trainer.print_latent_space_statistics(rcs_data)

    def _create_ae_optimizer_and_scheduler(self, params, config, stage='stage1'):
        return self.ae_trainer._create_optimizer_and_scheduler(params, config, stage)

    def _create_stage_loss_function(self, config, stage='stage1'):
        return self.ae_trainer._create_stage_loss_function(config, stage)

    def _ae_step_scheduler(self, scheduler, type, val_loss):
        return self.ae_trainer._step_scheduler(scheduler, type, val_loss)

    def _ae_log_training_progress(self, *args):
        return self.ae_trainer._log_progress(*args)

    # --- Configuration Methods (Retained) ---
    
    def _create_ae_training_config(self):
        """创建AutoEncoder训练配置"""
        try:
            config = {
                'batch_size': int(self.gui.ae_batch_size.get()),
                'learning_rate': float(self.gui.ae_learning_rate.get()),
                'optimizer_type': self.gui.ae_optimizer_type.get(),
                'weight_decay': float(self.gui.ae_weight_decay.get()),
                'momentum': float(self.gui.ae_momentum.get()),
                'lr_scheduler': self.gui.ae_lr_scheduler.get(),
                'min_lr': float(self.gui.ae_min_lr.get()),
                'restart_period': int(self.gui.ae_restart_period.get()),
                'use_custom_loss': self.gui.ae_use_custom_loss.get()
            }

            config['epochs'] = {
                'stage1': int(self.gui.ae_epochs_stage1.get()),
                'stage2': int(self.gui.ae_epochs_stage2.get()),
                'stage3': int(self.gui.ae_epochs_stage3.get()),
                'joint': int(getattr(self.gui, 'ae_epochs_joint', tk.IntVar(value=200)).get())
            }

            config['patience'] = {
                'stage1': int(self.gui.ae_patience_stage1.get()),
                'stage2': int(self.gui.ae_patience_stage2.get()),
                'stage3': int(self.gui.ae_patience_stage3.get()),
                'e2e': int(self.gui.ae_patience_e2e.get()),
                'joint': int(getattr(self.gui, 'ae_patience_joint', tk.IntVar(value=50)).get())
            }

            config['num_lr_stages'] = int(self.gui.ae_num_lr_stages.get())
            config['lr_decay_factor'] = float(self.gui.ae_lr_decay_factor.get())
            config['patience_multiplier'] = float(self.gui.ae_patience_multiplier.get())

            config['training_mode'] = {
                '三阶段训练': 'three_stage',
                '端到端训练': 'end_to_end',
                '仅Stage 1': 'stage1_only',
                '仅Stage 2': 'stage2_only',
                '联合训练': 'joint_training'
            }.get(self.gui.ae_training_mode.get(), 'three_stage')

            # 联合训练损失权重配置
            config['alpha_recon'] = float(getattr(self.gui, 'ae_alpha_recon', tk.DoubleVar(value=0.3)).get())
            config['beta_consistency'] = float(getattr(self.gui, 'ae_beta_consistency', tk.DoubleVar(value=0.5)).get())
            config['gamma_param_recon'] = float(getattr(self.gui, 'ae_gamma_param_recon', tk.DoubleVar(value=1.0)).get())

            if hasattr(self.gui, 'ae_custom_loss_config'):
                config['custom_loss_config'] = self.gui.ae_custom_loss_config

            if hasattr(self.gui, 'ae_stage_loss_configs'):
                for stage, loss_cfg in self.gui.ae_stage_loss_configs.items():
                    config[f'{stage}_loss_config'] = loss_cfg

            # 联合训练损失配置（三个独立的损失函数）
            if hasattr(self.gui, 'ae_joint_loss_config'):
                config['joint_loss_config'] = self.gui.ae_joint_loss_config

            return config

        except ValueError as e:
            self.gui.ae_log(f"❌ 配置参数错误: {e}")
            messagebox.showerror("配置错误", f"请检查输入参数是否有效: {e}")
            raise e

    def _open_loss_config_for_ae(self):
        """为AutoEncoder打开损失函数配置页面 - 切换到主界面标签页"""
        try:
            # 尝试找到并切换到损失配置标签页
            notebook = self.gui.notebook
            found = False
            for tab_id in notebook.tabs():
                tab_text = notebook.tab(tab_id, "text")
                if "损失函数配置" in tab_text:
                    notebook.select(tab_id)
                    found = True
                    self.gui.ae_log("👉 已切换到损失函数配置标签页")
                    break
            
            if not found:
                messagebox.showinfo("提示", "请在主界面的'损失函数配置'标签页中进行设置。")
                
        except Exception as e:
            self.gui.ae_log(f"❌ 切换标签页失败: {e}")
            messagebox.showinfo("提示", "请在主界面的'损失函数配置'标签页中进行设置。")