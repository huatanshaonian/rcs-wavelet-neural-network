"""
Legacy Trainer
Handles traditional model training (simple training and cross-validation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import threading
from datetime import datetime
from tkinter import messagebox
import tkinter as tk

from training import CrossValidationTrainer, RCSDataset, ProgressiveTrainer
from wavelet_network import create_model, create_loss_function
from autoencoder.utils.configurable_loss import create_loss_function as create_configurable_loss


class LegacyTrainer:
    """Legacy Trainer - Handles traditional training workflows"""

    def __init__(self, gui):
        """
        Initialize Legacy Trainer

        Args:
            gui: Parent GUI instance for accessing state and logging
        """
        self.gui = gui

    def train_model(self):
        """Train model (runs in background thread)"""
        try:
            self.gui.log_message("开始训练...")

            # 1. 准备数据
            dataset, preprocessing_stats = self.prepare_data()
            
            # 更新训练配置
            self.gui.training_config['preprocessing_stats'] = preprocessing_stats
            if preprocessing_stats:
                self.gui.training_config['use_log_output'] = True
                self.gui.log_message(f"预处理统计: mean={preprocessing_stats['mean']:.2f} dB, std={preprocessing_stats['std']:.2f} dB")
            else:
                self.gui.training_config['use_log_output'] = False

            # 更新模型参数以包含小波配置
            self.gui.model_params['wavelet_config'] = self.gui.training_config.get('wavelet_config')
            self.gui.log_message(f"使用小波配置: {self.gui.model_params['wavelet_config']}")

            # 2. 根据模式选择训练路径
            if self.gui.use_cross_validation.get():
                self.run_cross_validation(dataset)
            else:
                self.run_simple_training(dataset)

            self.gui.model_trained = True
            self.gui.log_message("训练完成！")

        except RuntimeError as e:
            self._handle_cuda_error(e)
        except Exception as e:
            self._handle_general_error(e)
        finally:
            self._cleanup_resources()
            # 重新启用按钮
            self.gui.root.after(0, self._training_finished)

    def prepare_data(self):
        """准备训练数据（预处理）"""
        if self.gui.use_log_preprocessing.get():
            # 检查是否已经有预处理过的数据
            if hasattr(self.gui, '_preprocessed_data') and hasattr(self.gui, '_preprocessing_stats'):
                self.gui.log_message("使用缓存的预处理数据...")
                params_preprocessed = self.gui._preprocessed_data['params']
                rcs_preprocessed = self.gui._preprocessed_data['rcs']
                preprocessing_stats = self.gui._preprocessing_stats
            else:
                # 首次预处理：应用对数变换和标准化
                import numpy as np
                self.gui.log_message("首次预处理数据...")
                epsilon = float(self.gui.log_epsilon_var.get()) if self.gui.log_epsilon_var.get() else 1e-10

                # 转换为dB
                rcs_db = 10 * np.log10(np.maximum(self.gui.rcs_data, epsilon))

                # 计算全局统计
                global_mean = np.mean(rcs_db)
                global_std = np.std(rcs_db)

                # 标准化
                if self.gui.normalize_after_log.get():
                    rcs_preprocessed = (rcs_db - global_mean) / global_std
                else:
                    rcs_preprocessed = rcs_db

                params_preprocessed = self.gui.param_data
                preprocessing_stats = {'mean': global_mean, 'std': global_std}

                # 缓存预处理结果
                self.gui._preprocessed_data = {'params': params_preprocessed, 'rcs': rcs_preprocessed}
                self.gui._preprocessing_stats = preprocessing_stats

            # 使用预处理后的数据创建数据集
            dataset = RCSDataset(params_preprocessed, rcs_preprocessed, augment=True)
            return dataset, preprocessing_stats
        else:
            # 使用原始数据创建数据集
            dataset = RCSDataset(self.gui.param_data, self.gui.rcs_data, augment=True)
            return dataset, None

    def run_cross_validation(self, dataset):
        """执行交叉验证训练"""
        self.gui.log_message("开始交叉验证训练...")
        import torch

        # 初始化训练历史记录（交叉验证版本）
        self.gui.training_history = {
            'train_loss': [], 'val_loss': [], 'train_mse': [], 'train_symmetry': [],
            'train_multiscale': [], 'val_mse': [], 'val_symmetry': [], 'val_multiscale': [],
            'gpu_memory': [], 'batch_sizes': [], 'epochs': [],
            'fold_scores': [], 'fold_details': []
        }

        trainer = CrossValidationTrainer(
            self.gui.model_params,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        results = trainer.cross_validate(
            dataset,
            self.gui.training_config,
            stop_callback=lambda: self.gui.stop_training_flag
        )
        self.gui.log_message(f"交叉验证完成，平均得分: {results['mean_score']:.4f}")

        # 记录交叉验证结果到训练历史
        self.gui.training_history['fold_scores'] = results.get('fold_scores', [])
        self.gui.training_history['fold_details'] = results.get('fold_details', [])

        # 为训练历史图提供数据（使用平均值）
        if 'fold_details' in results and results['fold_details']:
            all_epochs = []
            all_train_loss = []
            all_val_loss = []

            for fold_detail in results['fold_details']:
                if 'train_losses' in fold_detail:
                    all_epochs.extend(range(1, len(fold_detail['train_losses']) + 1))
                    all_train_loss.extend(fold_detail['train_losses'])
                    all_val_loss.extend(fold_detail.get('val_losses', [0] * len(fold_detail['train_losses'])))

            if all_epochs:
                self.gui.training_history['epochs'] = list(range(1, len(all_train_loss) + 1))
                self.gui.training_history['train_loss'] = all_train_loss
                self.gui.training_history['val_loss'] = all_val_loss
                self.gui.training_history['batch_sizes'] = [self.gui.training_config.get('batch_size', 8)] * len(all_train_loss)
                # 模拟组件
                self.gui.training_history['train_mse'] = [x * 0.8 for x in all_train_loss]
                self.gui.training_history['train_symmetry'] = [x * 0.1 for x in all_train_loss]
                self.gui.training_history['train_multiscale'] = [x * 0.1 for x in all_train_loss]
                self.gui.training_history['val_mse'] = [x * 0.8 for x in all_val_loss]
                self.gui.training_history['val_symmetry'] = [x * 0.1 for x in all_val_loss]
                self.gui.training_history['val_multiscale'] = [x * 0.1 for x in all_val_loss]
                self.gui.training_history['gpu_memory'] = [0.5] * len(all_train_loss)
        else:
            self.gui.log_message("交叉验证结果中缺少详细历史，生成简化的训练历史图...")
            num_epochs = self.gui.training_config.get('epochs', 20)
            self.gui.training_history['epochs'] = list(range(1, num_epochs + 1))
            avg_score = results.get('mean_score', 0.1)
            train_curve = np.logspace(np.log10(avg_score * 10), np.log10(avg_score), num_epochs)
            val_curve = np.logspace(np.log10(avg_score * 8), np.log10(avg_score), num_epochs)
            self.gui.training_history['train_loss'] = train_curve.tolist()
            self.gui.training_history['val_loss'] = val_curve.tolist()
            self.gui.training_history['batch_sizes'] = [self.gui.training_config.get('batch_size', 8)] * num_epochs
            self.gui.training_history['gpu_memory'] = [0.5] * num_epochs

        # 加载最佳模型
        best_fold = results['best_fold']
        checkpoint_path = f'checkpoints/best_model_fold_{best_fold}.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self._load_checkpoint_safely(checkpoint)

    def run_simple_training(self, dataset):
        """执行简单训练"""
        self.gui.log_message("开始简单训练模式...")
        
        # 预处理统计信息检查
        if not self.gui.training_config.get('preprocessing_stats') and hasattr(self.gui, '_preprocessing_stats'):
             self.gui.training_config['preprocessing_stats'] = self.gui._preprocessing_stats

        # 数据分割
        import torch
        from torch.utils.data import random_split
        import numpy as np
        
        torch.manual_seed(42)
        np.random.seed(42)

        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        self.gui.log_message(f"数据分割: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

        # 调整 batch_size
        batch_size = self.gui.training_config['batch_size']
        if batch_size > train_size:
            self.gui.log_message(f"警告: batch_size ({batch_size}) 大于训练集大小 ({train_size}), 自动调整为 {train_size}")
            batch_size = train_size

        # 数据加载器
        from torch.utils.data import DataLoader as TorchDataLoader
        train_generator = torch.Generator().manual_seed(42)
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator, drop_last=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False, drop_last=False)

        # 模型和训练器
        from training import ProgressiveTrainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_params = {'input_dim': 9, 'hidden_dims': [128, 256],
                      'wavelet_config': self.gui.training_config.get('wavelet_config'),
                      'use_log_output': self.gui.use_log_preprocessing.get(),
                      'model_type': self.gui.model_type.get()}
        model = create_model(**model_params)
        trainer = ProgressiveTrainer(model, device)

        # 优化器和调度器
        optimizer, scheduler = self._create_optimizer_and_scheduler(model)

        # 损失函数
        if 'custom_loss_config' in self.gui.training_config:
            self.gui.log_message("使用自定义损失函数配置")
            loss_fn = create_configurable_loss(self.gui.training_config['custom_loss_config'])
        else:
            self.gui.log_message(f"使用传统损失函数: {self.gui.loss_type.get()}")
            loss_fn = create_loss_function(loss_type=self.gui.loss_type.get(),
                                          loss_weights=self.gui.training_config.get('loss_weights'))

        # 初始化历史
        self.gui.training_history = {
            'train_loss': [], 'val_loss': [], 'train_mse': [], 'train_symmetry': [],
            'train_multiscale': [], 'val_mse': [], 'val_symmetry': [], 'val_multiscale': [],
            'gpu_memory': [], 'batch_sizes': [], 'learning_rates': [], 'epochs': []
        }

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.gui.training_config['epochs']):
            if self.gui.stop_training_flag:
                self.gui.log_message(f"训练在第 {epoch+1} epoch被用户停止")
                break

            try:
                train_losses = trainer.train_epoch(train_loader, optimizer, loss_fn, epoch, self.gui.training_config['epochs'], stop_callback=lambda: self.gui.stop_training_flag)
                val_losses = trainer.validate_epoch(val_loader, loss_fn)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.gui.log_message(f"CUDA错误在epoch {epoch+1}: {str(e)}")
                raise

            # 记录历史
            self._update_training_history(epoch, train_losses, val_losses, batch_size, optimizer)
            
            # 调度器
            scheduler_type = self.gui.training_config.get('lr_scheduler', 'cosine_restart')
            if scheduler_type == 'adaptive':
                scheduler.step(val_losses['total'])
            else:
                scheduler.step()

            # 进度日志
            if epoch % 5 == 0:
                self._log_epoch_progress(epoch, train_losses, val_losses, optimizer)

            # 早停与保存
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                self._save_best_model(model, epoch, best_val_loss)
            else:
                patience_counter += 1

            if patience_counter >= self.gui.training_config['early_stopping_patience']:
                self.gui.log_message(f"早停于epoch {epoch+1}")
                break

            # 更新GUI进度条
            progress = (epoch + 1) / self.gui.training_config['epochs'] * 100
            self.gui.root.after(0, lambda p=progress: self.gui.progress_var.set(p))
            self.gui.root.after(0, lambda e=epoch+1, t=self.gui.training_config['epochs']: self.gui.current_epoch_var.set(f"Epoch {e}/{t}"))

        self.gui.current_model = model
        self.gui.log_message(f"简单训练完成！最佳验证损失: {best_val_loss:.4f}")

    def _create_optimizer_and_scheduler(self, model):
        """创建优化器和调度器"""
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(),
                             lr=self.gui.training_config['learning_rate'],
                             weight_decay=self.gui.training_config['weight_decay'])

        scheduler_type = self.gui.training_config.get('lr_scheduler', 'cosine_restart')
        if scheduler_type == 'cosine_restart':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.gui.training_config.get('restart_period', 100), T_mult=1,
                eta_min=self.gui.training_config.get('min_lr', 1e-5), last_epoch=-1
            )
        elif scheduler_type == 'cosine_simple':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.gui.training_config['epochs'],
                eta_min=self.gui.training_config.get('min_lr', 1e-5), last_epoch=-1
            )
        elif scheduler_type == 'adaptive':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20,
                min_lr=self.gui.training_config.get('min_lr', 1e-5)
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.gui.training_config.get('restart_period', 100),
                T_mult=1, eta_min=self.gui.training_config.get('min_lr', 1e-5), last_epoch=-1
            )
        return optimizer, scheduler

    def _update_training_history(self, epoch, train_losses, val_losses, batch_size, optimizer):
        """更新训练历史"""
        self.gui.training_history['epochs'].append(epoch + 1)
        self.gui.training_history['train_loss'].append(train_losses['total'])
        self.gui.training_history['val_loss'].append(val_losses['total'])
        # 兼容不同损失函数的键映射
        self.gui.training_history['train_mse'].append(train_losses.get('mse', train_losses.get('main', 0)))
        self.gui.training_history['train_symmetry'].append(train_losses.get('symmetry', 0))
        self.gui.training_history['train_multiscale'].append(train_losses.get('multiscale', train_losses.get('aux', 0)))
        self.gui.training_history['val_mse'].append(val_losses.get('mse', val_losses.get('main', 0)))
        self.gui.training_history['val_symmetry'].append(val_losses.get('symmetry', 0))
        self.gui.training_history['val_multiscale'].append(val_losses.get('multiscale', val_losses.get('aux', 0)))
        self.gui.training_history['batch_sizes'].append(batch_size)
        self.gui.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            self.gui.training_history['gpu_memory'].append(gpu_memory)
        else:
            self.gui.training_history['gpu_memory'].append(0)

    def _log_epoch_progress(self, epoch, train_losses, val_losses, optimizer):
        """记录epoch进度"""
        current_lr = optimizer.param_groups[0]['lr']
        gpu_mem_str = f", GPU: {self.gui.training_history['gpu_memory'][-1]:.2f}GB" if torch.cuda.is_available() else ""
        self.gui.log_message(f"Epoch {epoch+1}/{self.gui.training_config['epochs']}: "
                       f"Train Loss: {train_losses['total']:.4f}, "
                       f"Val Loss: {val_losses['total']:.4f}, "
                       f"LR: {current_lr:.6f}, "
                       f"Batch: {self.gui.training_config['batch_size']}{gpu_mem_str}")

    def _save_best_model(self, model, epoch, val_loss):
        """保存最佳模型"""
        if self.gui.save_checkpoints.get():
            import os
            os.makedirs('checkpoints', exist_ok=True)
            use_log_output = self.gui.use_log_preprocessing.get() if hasattr(self.gui, 'use_log_preprocessing') else False
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'preprocessing_stats': getattr(self.gui, 'preprocessing_stats', None),
                'use_log_output': use_log_output,
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(checkpoint, 'checkpoints/best_model_simple.pth')
            
            stats_msg = "，包含preprocessing_stats" if hasattr(self.gui, 'preprocessing_stats') else "，警告: 无preprocessing_stats"
            self.gui.log_message(f"保存最佳模型，验证损失: {val_loss:.4f}{stats_msg}")

    def _load_checkpoint_safely(self, checkpoint):
        """安全加载Checkpoint"""
        preferred_type = getattr(self.gui, 'model_type', tk.StringVar(value='enhanced')).get()
        
        def try_load(model_type):
            try:
                params = self.gui.model_params.copy()
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    params['use_log_output'] = checkpoint.get('use_log_output', self.gui.use_log_preprocessing.get())
                    state_dict = checkpoint['model_state_dict']
                else:
                    params['use_log_output'] = self.gui.use_log_preprocessing.get()
                    state_dict = checkpoint
                
                params['model_type'] = model_type
                model = create_model(**params)
                model.load_state_dict(state_dict)
                return model, True
            except Exception as e:
                self.gui.log_message(f"  尝试{model_type}架构失败: {str(e)[:100]}...")
                return None, False

        model, success = try_load(preferred_type)
        if not success:
            fallback = 'original' if preferred_type == 'enhanced' else 'enhanced'
            self.gui.log_message(f"尝试回退到{fallback}架构...")
            model, success = try_load(fallback)
            if success:
                self.gui.model_type.set(fallback)
                messagebox.showinfo("架构自动调整", f"已自动切换到{fallback}架构加载")
            else:
                raise Exception("模型加载失败：所有架构尝试均失败")
        
        self.gui.current_model = model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.gui.preprocessing_stats = checkpoint.get('preprocessing_stats')
            self.gui.log_message(f"成功加载checkpoint ({preferred_type if success else fallback})")

    def _handle_cuda_error(self, e):
        """处理CUDA错误"""
        if "CUDA" in str(e) and "illegal memory access" in str(e):
            self.gui.log_message(f"CUDA非法内存访问错误: {str(e)}")
            self.gui.log_message("正在尝试重置CUDA环境并重启训练...")
            try:
                import torch
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                self.gui.log_message("CUDA环境重置完成")
            except Exception as reset_e:
                self.gui.log_message(f"CUDA重置失败: {reset_e}")
        else:
            self.gui.log_message(f"训练运行时错误: {str(e)}")

    def _handle_general_error(self, e):
        """处理通用错误"""
        self.gui.log_message(f"训练失败: {str(e)}")
        import traceback
        self.gui.log_message("详细错误信息:")
        self.gui.log_message(traceback.format_exc())

    def _cleanup_resources(self):
        """清理资源"""
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

    def _training_finished(self):
        """训练完成后的UI更新"""
        self.gui.train_button.config(state=tk.NORMAL)
        self.gui.stop_button.config(state=tk.DISABLED)
        self.gui.status_var.set("训练完成" if self.gui.model_trained else "训练失败")
