"""
小波RCS网络训练模块

提供完整的训练功能，包括:
1. 数据加载和预处理
2. 数据增强策略
3. 渐进式训练
4. 交叉验证
5. 模型检查点保存
6. 早停机制
7. 学习率调度

针对小数据集(~100样本)优化的训练策略

作者: RCS Wavelet Network Project
版本: 1.0
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings

# 导入项目模块
import rcs_data_reader as rdr
from wavelet_network import TriDimensionalRCSNet, TriDimensionalRCSLoss, create_model, create_loss_function

warnings.filterwarnings('ignore')


class RCSDataset(Dataset):
    """
    RCS数据集类

    处理9个飞行器参数到双频RCS数据的映射
    """

    def __init__(self, parameters: np.ndarray, rcs_data: np.ndarray,
                 transform=None, augment=False):
        """
        初始化数据集

        参数:
            parameters: 飞行器参数 [N, 9]
            rcs_data: RCS数据 [N, 91, 91, 2]
            transform: 数据变换函数
            augment: 是否启用数据增强
        """
        # 检查并清理异常值
        if np.any(np.isnan(rcs_data)) or np.any(np.isinf(rcs_data)):
            print(f"警告: RCS数据包含NaN或Inf值，进行清理...")
            rcs_data = np.nan_to_num(rcs_data, nan=0.0, posinf=1e10, neginf=-1e10)

        if np.any(np.isnan(parameters)) or np.any(np.isinf(parameters)):
            print(f"警告: 参数数据包含NaN或Inf值，进行清理...")
            parameters = np.nan_to_num(parameters, nan=0.0, posinf=1e10, neginf=-1e10)

        self.transform = transform
        self.augment = augment

        # 数据标准化（只对参数进行）
        self.param_scaler = StandardScaler()
        normalized_params = self.param_scaler.fit_transform(parameters)

        # 转换为tensor（只转换一次）
        self.parameters = torch.tensor(normalized_params, dtype=torch.float32)
        self.rcs_data = torch.tensor(rcs_data, dtype=torch.float32)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        params = self.parameters[idx]
        rcs = self.rcs_data[idx]

        if self.augment:
            params, rcs = self._apply_augmentation(params, rcs)

        if self.transform:
            params, rcs = self.transform(params, rcs)

        return params, rcs

    def _apply_augmentation(self, params: torch.Tensor, rcs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用数据增强

        策略:
        1. 参数空间噪声添加
        2. 角度域旋转变换
        3. 双频一致性保持
        """
        # 参数噪声增强 (±2% 随机噪声)
        param_noise = torch.randn_like(params) * 0.02
        params_aug = params + param_noise

        # 角度域增强 - φ轴镜像 (保持物理约束)
        if torch.rand(1) < 0.3:
            rcs = torch.flip(rcs, dims=[0])  # φ轴翻转

        # 小幅度高斯噪声 (±1% RCS噪声)
        if torch.rand(1) < 0.5:
            rcs_noise = torch.randn_like(rcs) * 0.01 * torch.abs(rcs)
            rcs = rcs + rcs_noise

        return params_aug, rcs


class DataAugmentation:
    """
    数据增强策略集合
    """

    @staticmethod
    def parameter_interpolation(params1: np.ndarray, params2: np.ndarray,
                               rcs1: np.ndarray, rcs2: np.ndarray,
                               num_interpolations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        参数空间插值增强

        在两个样本间进行线性插值生成新样本
        """
        alphas = np.linspace(0.1, 0.9, num_interpolations)

        interp_params = []
        interp_rcs = []

        for alpha in alphas:
            # 参数插值
            params_interp = alpha * params1 + (1 - alpha) * params2

            # RCS插值 (在对数域进行以保持物理意义)
            rcs_log1 = np.log(np.maximum(rcs1, 1e-10))
            rcs_log2 = np.log(np.maximum(rcs2, 1e-10))
            rcs_interp = np.exp(alpha * rcs_log1 + (1 - alpha) * rcs_log2)

            interp_params.append(params_interp)
            interp_rcs.append(rcs_interp)

        return np.array(interp_params), np.array(interp_rcs)

    @staticmethod
    def frequency_consistent_augmentation(rcs_data: np.ndarray) -> np.ndarray:
        """
        频率一致性增强

        确保两个频率间的物理关系合理性
        """
        freq_ratio = 3.0 / 1.5  # 频率比

        # 基于物理模型的轻微调整
        rcs_aug = rcs_data.copy()

        # 在合理范围内调整频率间关系
        adjustment_factor = 1 + 0.05 * (np.random.rand(*rcs_data.shape) - 0.5)
        rcs_aug[:, :, :, 1] *= adjustment_factor[:, :, :, 1]  # 调整3GHz数据

        return rcs_aug


class ProgressiveTrainer:
    """
    渐进式训练器

    实现渐进式训练策略，先训练低分辨率后训练高分辨率
    """

    def __init__(self, model: TriDimensionalRCSNet, device: str = 'cuda'):
        """
        初始化训练器

        参数:
            model: RCS预测模型
            device: 训练设备
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }

    def _create_progressive_loss_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        创建渐进式损失权重

        早期注重低分辨率和物理约束，后期注重高分辨率细节
        """
        progress = epoch / total_epochs

        # 渐进式权重调整
        base_weights = {
            'mse': 1.0,
            'symmetry': 0.05 - 0.03 * progress,  # 显著降低对称性权重，避免主导训练
            'multiscale': 0.3 - 0.2 * progress  # 多尺度损失逐渐减弱
        }

        return base_weights

    def train_epoch(self, train_loader: TorchDataLoader, optimizer: optim.Optimizer,
                   loss_fn: TriDimensionalRCSLoss, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        训练一个epoch
        """
        self.model.train()

        # 更新损失权重
        progressive_weights = self._create_progressive_loss_weights(epoch, total_epochs)
        loss_fn.loss_weights = progressive_weights

        epoch_losses = {'total': 0, 'mse': 0, 'symmetry': 0,
                       'multiscale': 0}
        num_batches = 0

        for batch_idx, (params, targets) in enumerate(train_loader):
            try:
                # 数据验证和清理
                if torch.isnan(params).any() or torch.isinf(params).any():
                    print(f"警告: 批次{batch_idx}参数包含NaN/Inf，跳过")
                    continue

                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    print(f"警告: 批次{batch_idx}目标包含NaN/Inf，跳过")
                    continue

                params, targets = params.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                predictions = self.model(params)

                # 计算损失
                losses = loss_fn(predictions, targets)

                # 反向传播
                losses['total'].backward()

                # 梯度裁剪 (防止梯度爆炸)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                # 累积损失
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                num_batches += 1

            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"\n错误: GPU内存不足或CUDA错误 (批次 {batch_idx})")
                    print(f"错误信息: {str(e)}")
                    if torch.cuda.is_available():
                        print(f"当前GPU显存使用: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                        print(f"峰值GPU显存使用: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                        torch.cuda.empty_cache()
                    raise RuntimeError(f"CUDA错误: 批次大小({params.shape[0]})可能过大，请减小批次大小") from e
                else:
                    raise

        # 平均损失
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        else:
            print("警告: 训练批次数为0，可能是批次大小配置问题")
            # 返回零损失以避免除零错误
            epoch_losses = {key: 0.0 for key in epoch_losses}

        return epoch_losses

    def validate_epoch(self, val_loader: TorchDataLoader,
                      loss_fn: TriDimensionalRCSLoss) -> Dict[str, float]:
        """
        验证一个epoch
        """
        self.model.eval()

        epoch_losses = {'total': 0, 'mse': 0, 'symmetry': 0,
                       'multiscale': 0}
        num_batches = 0

        with torch.no_grad():
            for params, targets in val_loader:
                params, targets = params.to(self.device), targets.to(self.device)

                predictions = self.model(params)
                losses = loss_fn(predictions, targets)

                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                num_batches += 1

        # 平均损失
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        else:
            print("警告: 验证批次数为0，可能是批次大小配置问题")
            # 返回高损失值以避免除零错误
            epoch_losses = {key: float('inf') for key in epoch_losses}

        return epoch_losses


class CrossValidationTrainer:
    """
    交叉验证训练器

    实现5折交叉验证，适应小数据集场景
    """

    def __init__(self, model_params: Dict, device: str = 'cuda', n_folds: int = 5):
        """
        初始化交叉验证训练器

        参数:
            model_params: 模型参数
            device: 训练设备
            n_folds: 交叉验证折数
        """
        self.model_params = model_params
        self.device = device
        self.n_folds = n_folds

        # 交叉验证结果
        self.cv_results = {
            'fold_scores': [],
            'mean_score': 0,
            'std_score': 0,
            'best_fold': 0
        }

    def cross_validate(self, dataset: RCSDataset, training_config: Dict) -> Dict:
        """
        执行交叉验证
        """
        # 应用CUDA优化设置
        self._apply_cuda_optimizations()

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_scores = []
        fold_details = []  # 收集每折的详细历史

        # 获取数据
        all_params = dataset.parameters.numpy()
        all_rcs = dataset.rcs_data.numpy()

        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_params)):
            print(f"\n开始第 {fold+1}/{self.n_folds} 折训练...")

            # 创建数据加载器
            train_params = all_params[train_idx]
            train_rcs = all_rcs[train_idx]
            val_params = all_params[val_idx]
            val_rcs = all_rcs[val_idx]

            train_dataset = RCSDataset(train_params, train_rcs, augment=True)
            val_dataset = RCSDataset(val_params, val_rcs, augment=False)

            # 应用CUDA内存优化的批次大小调整
            memory_config = training_config.get('memory_optimization', {})
            batch_size = self._get_safe_batch_size(training_config['batch_size'],
                                                 len(train_dataset), memory_config)

            train_loader = TorchDataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=memory_config.get('pin_memory', True),
                                    drop_last=True)
            val_loader = TorchDataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=memory_config.get('pin_memory', True),
                                  drop_last=False)  # 保留小批次以确保验证集不为空

            # 清理上一fold的显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 创建模型
            model = create_model(**self.model_params)
            trainer = ProgressiveTrainer(model, self.device)

            # 训练，获取最佳损失和详细历史
            best_val_loss, fold_history = self._train_fold(trainer, train_loader, val_loader,
                                                          training_config, fold)

            fold_scores.append(best_val_loss)
            fold_details.append(fold_history)

            print(f"第 {fold+1} 折完成，最佳验证损失: {best_val_loss:.6f}")

        # 计算交叉验证统计
        self.cv_results['fold_scores'] = fold_scores
        self.cv_results['fold_details'] = fold_details  # 添加详细历史
        self.cv_results['mean_score'] = np.mean(fold_scores)
        self.cv_results['std_score'] = np.std(fold_scores)
        self.cv_results['best_fold'] = np.argmin(fold_scores)

        print(f"\n交叉验证完成:")
        print(f"  平均验证损失: {self.cv_results['mean_score']:.6f} ± {self.cv_results['std_score']:.6f}")
        print(f"  最佳折: {self.cv_results['best_fold'] + 1} (损失: {fold_scores[self.cv_results['best_fold']]:.6f})")

        return self.cv_results

    def _get_safe_batch_size(self, requested_batch_size: int, dataset_size: int,
                           memory_config: Dict) -> int:
        """
        获取安全的批次大小，避免CUDA内存错误和除零错误
        """
        # 确保批次大小不超过数据集大小
        max_possible = min(requested_batch_size, dataset_size)

        # 额外检查：确保批次大小至少为1，且不超过数据集大小
        if max_possible <= 0:
            print(f"警告: 数据集大小为 {dataset_size}，使用最小批次大小 1")
            return 1

        # 对于小数据集，使用更保守的策略
        if dataset_size <= 10:
            # 非常小的数据集，使用最小可能的批次大小
            safe_batch_size = min(2, dataset_size)
            print(f"小数据集检测 (大小={dataset_size})，使用保守批次大小: {safe_batch_size}")
            return safe_batch_size

        # 大批次大小警告和自动限制
        if requested_batch_size > 32:
            print(f"⚠️ 警告: 批次大小{requested_batch_size}较大，可能导致CUDA内存错误")
            # 自动限制最大批次为32，更安全
            safe_batch_size = min(32, dataset_size)
            print(f"  自动限制批次大小为: {safe_batch_size}")
            return safe_batch_size

        # 移除硬编码的批次大小限制，让用户自由设置
        # 只要不超过数据集大小即可
        safe_batch_size = max_possible

        if safe_batch_size != requested_batch_size:
            print(f"批次大小调整: {requested_batch_size} -> {safe_batch_size} (受数据集大小限制)")

        return safe_batch_size

    def _apply_cuda_optimizations(self):
        """
        应用CUDA优化设置
        """
        if torch.cuda.is_available():
            # 启用cuDNN基准测试模式
            torch.backends.cudnn.benchmark = True

            # 清理GPU内存
            torch.cuda.empty_cache()

            # 重置GPU峰值内存统计
            torch.cuda.reset_peak_memory_stats()

            # 限制GPU内存使用
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.85)  # 降低到85%更安全

            print("CUDA优化设置已应用")

    def _train_fold(self, trainer: ProgressiveTrainer, train_loader: TorchDataLoader,
                   val_loader: TorchDataLoader, config: Dict, fold: int) -> Tuple[float, Dict]:
        """
        训练单个fold，返回最佳损失和详细历史

        返回:
            Tuple[float, Dict]: (最佳验证损失, 训练历史详情)
        """
        # 优化器和学习率调度器
        optimizer = optim.Adam(trainer.model.parameters(),
                             lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])

        # 使用余弦退火调度器，支持周期性重启以逃离局部最优
        # CosineAnnealingWarmRestarts: 学习率周期性从高到低，然后重启
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,              # 第一个周期的epoch数
            T_mult=1,            # 每次重启后周期长度的倍数（1表示保持不变）
            eta_min=config.get('min_lr', 2e-5),  # 最小学习率(从配置读取)
            last_epoch=-1
        )

        # 备用：ReduceLROnPlateau作为辅助（可选）
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,          # 更温和的衰减
            patience=30,         # 更大的耐心值
            min_lr=1e-7,
            verbose=False
        )

        # 损失函数
        loss_fn = create_loss_function(loss_weights=config.get('loss_weights'))

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0

        # 详细历史记录
        fold_history = {
            'train_losses': [],
            'val_losses': [],
            'train_mse': [],
            'train_symmetry': [],
            'train_multiscale': [],
            'val_mse': [],
            'val_symmetry': [],
            'val_multiscale': [],
            'epochs': [],
            'learning_rates': [],
            'fold': fold
        }

        for epoch in range(config['epochs']):
            # 定期清理CUDA缓存
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 训练
            train_losses = trainer.train_epoch(train_loader, optimizer, loss_fn,
                                             epoch, config['epochs'])

            # 验证
            val_losses = trainer.validate_epoch(val_loader, loss_fn)

            # 记录详细历史
            fold_history['epochs'].append(epoch + 1)
            fold_history['train_losses'].append(train_losses['total'])
            fold_history['val_losses'].append(val_losses['total'])
            fold_history['train_mse'].append(train_losses.get('mse', 0))
            fold_history['train_symmetry'].append(train_losses.get('symmetry', 0))
            fold_history['train_multiscale'].append(train_losses.get('multiscale', 0))
            fold_history['val_mse'].append(val_losses.get('mse', 0))
            fold_history['val_symmetry'].append(val_losses.get('symmetry', 0))
            fold_history['val_multiscale'].append(val_losses.get('multiscale', 0))
            fold_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # 学习率调度 - CosineAnnealingWarmRestarts每个epoch都要step
            scheduler.step()

            # 可选：如果验证损失长期不改善，使用plateau_scheduler进一步降低学习率
            # plateau_scheduler.step(val_losses['total'])

            # 早停检查
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0

                # 保存最佳模型
                torch.save(trainer.model.state_dict(),
                          f'checkpoints/best_model_fold_{fold}.pth')
            else:
                patience_counter += 1

            if patience_counter >= config['early_stopping_patience']:
                print(f"早停于epoch {epoch+1}")
                break

            # 打印进度
            if epoch % 10 == 0:
                print(f"Fold {fold+1}, Epoch {epoch+1}: "
                      f"Train Loss: {train_losses['total']:.4f}, "
                      f"Val Loss: {val_losses['total']:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        return best_val_loss, fold_history


class RCSDataLoader:
    """
    RCS数据加载器

    负责从CSV文件加载和预处理RCS数据
    """

    def __init__(self, data_config: Dict):
        """
        初始化数据加载器

        参数:
            data_config: 数据配置
        """
        self.data_config = data_config
        self.preprocessing_config = data_config.get('preprocessing', {})
        self.use_log_preprocessing = self.preprocessing_config.get('use_log', False)
        self.log_epsilon = self.preprocessing_config.get('log_epsilon', 1e-10)
        self.normalize_after_log = self.preprocessing_config.get('normalize_after_log', True)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载训练数据

        返回:
            (parameters, rcs_data): 参数和RCS数据
        """
        print("加载训练数据...")

        # 加载参数数据
        params_file = self.data_config['params_file']
        param_data, param_names = rdr.load_parameters(params_file)

        print(f"加载参数: {len(param_names)} 个参数, {param_data.shape[0]} 个样本")

        # 加载RCS数据
        rcs_data = self._load_rcs_data()

        print(f"RCS数据形状: {rcs_data.shape}")

        return param_data, rcs_data

    def _load_rcs_data(self) -> np.ndarray:
        """
        加载双频RCS数据，支持对数预处理
        """
        data_dir = self.data_config['rcs_data_dir']
        model_ids = self.data_config['model_ids']
        frequencies = self.data_config['frequencies']  # ['1.5G', '3G']

        rcs_list = []
        all_rcs_values = []  # 用于计算全局统计

        print(f"加载RCS数据，对数预处理: {'开启' if self.use_log_preprocessing else '关闭'}")

        for model_id in model_ids:
            model_rcs_list = []

            for freq in frequencies:
                try:
                    # 使用现有的数据读取函数
                    data = rdr.get_adaptive_rcs_matrix(model_id, freq, data_dir, verbose=False)
                    rcs_linear = data['rcs_linear']  # [91, 91]

                    # 应用对数预处理 (转换为分贝值)
                    if self.use_log_preprocessing:
                        # 确保没有负值或零值，转换为dB: 10 * log10
                        rcs_processed = np.maximum(rcs_linear, self.log_epsilon)
                        rcs_processed = 10 * np.log10(rcs_processed)
                    else:
                        rcs_processed = rcs_linear

                    model_rcs_list.append(rcs_processed)
                    all_rcs_values.append(rcs_processed.flatten())

                except Exception as e:
                    print(f"警告: 无法加载模型 {model_id} 频率 {freq}: {e}")
                    # 创建零填充数据
                    if self.use_log_preprocessing:
                        # dB域的零值应该是很小的负值 (10 * log10)
                        fill_value = 10 * np.log10(self.log_epsilon)
                    else:
                        fill_value = 0.0
                    model_rcs_list.append(np.full((91, 91), fill_value))

            # 组合双频数据 [91, 91, 2]
            if len(model_rcs_list) == 2:
                model_rcs = np.stack(model_rcs_list, axis=2)
                rcs_list.append(model_rcs)

        rcs_data = np.array(rcs_list)  # [N, 91, 91, 2]

        # 对数预处理后的归一化
        if self.use_log_preprocessing and self.normalize_after_log:
            all_values = np.concatenate(all_rcs_values)
            global_mean = np.mean(all_values)
            global_std = np.std(all_values)

            print(f"对数预处理统计:")
            print(f"  原始RCS范围: {np.min(all_values):.1f} - {np.max(all_values):.1f} dB")
            print(f"  全局均值: {global_mean:.1f} dB, 标准差: {global_std:.1f} dB")

            # 标准化
            rcs_data = (rcs_data - global_mean) / global_std

            print(f"  标准化后范围: {np.min(rcs_data):.3f} - {np.max(rcs_data):.3f}")

            # 保存统计信息用于后续逆变换
            self.preprocessing_stats = {
                'mean': global_mean,
                'std': global_std,
                'min_original': np.min(all_values),
                'max_original': np.max(all_values)
            }
        else:
            # 线性数据的统计信息
            if len(all_rcs_values) > 0:
                all_values = np.concatenate(all_rcs_values)
                print(f"线性RCS数据统计:")
                print(f"  范围: {np.min(all_values):.6e} - {np.max(all_values):.6e}")
                print(f"  均值: {np.mean(all_values):.6e}, 标准差: {np.std(all_values):.6e}")

        return rcs_data


def create_training_config(early_stopping_patience: int = 50) -> Dict:
    """
    创建默认训练配置 - 优化CUDA内存使用

    Args:
        early_stopping_patience: 早停耐心值，连续多少轮验证损失不改善就停止训练
    """
    return {
        'batch_size': 8,  # 适合小数据集的批次大小
        'learning_rate': 3e-3,  # 初始学习率 (推荐范围: 1e-3 到 5e-3)
        'min_lr': 2e-5,  # 最低学习率/eta_min (推荐范围: 1e-5 到 5e-5)
        'weight_decay': 1e-4,
        'epochs': 200,
        'early_stopping_patience': early_stopping_patience,  # 可调节的早停参数
        'loss_weights': {
            'mse': 1.0,
            'symmetry': 0.02,  # 显著降低默认对称性权重
            'multiscale': 0.1
        },
        # 新增：CUDA内存优化配置
        'memory_optimization': {
            'gradient_accumulation': True, # 启用梯度累积
            'mixed_precision': True,       # 启用混合精度训练
            'pin_memory': True,           # 启用内存固定
            'empty_cache_frequency': 10   # 每10个epoch清理缓存
        }
    }


def create_data_config(use_log_preprocessing: bool = False) -> Dict:
    """
    创建默认数据配置

    参数:
        use_log_preprocessing: 是否启用对数预处理
    """
    return {
        'params_file': r"..\parameter\parameters_sorted.csv",
        'rcs_data_dir': r"..\parameter\csv_output",
        'model_ids': [f"{i:03d}" for i in range(1, 101)],  # 001-100
        'frequencies': ['1.5G', '3G'],
        'preprocessing': {
            'use_log': use_log_preprocessing,
            'log_epsilon': 1e-10,  # 防止log(0)的最小值
            'normalize_after_log': True  # 对数变换后是否标准化
        }
    }


def save_training_results(results: Dict, save_path: str):
    """
    保存训练结果
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换numpy类型为python类型以便JSON序列化
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = value.item()
        else:
            serializable_results[key] = value

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def plot_training_history(history: Dict, save_path: str = None):
    """
    绘制训练历史
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue')
    axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 学习率曲线
    axes[0, 1].plot(history['learning_rates'], color='green')
    axes[0, 1].set_title('学习率变化')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('学习率')
    axes[0, 1].grid(True)

    # 损失对比
    if len(history['train_loss']) > 0:
        final_epochs = min(50, len(history['train_loss']))
        axes[1, 0].plot(history['train_loss'][-final_epochs:], label='训练损失')
        axes[1, 0].plot(history['val_loss'][-final_epochs:], label='验证损失')
        axes[1, 0].set_title('最后50个Epoch损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # 训练统计
    axes[1, 1].text(0.1, 0.8, f"最终训练损失: {history['train_loss'][-1]:.4f}")
    axes[1, 1].text(0.1, 0.6, f"最终验证损失: {history['val_loss'][-1]:.4f}")
    axes[1, 1].text(0.1, 0.4, f"总训练轮数: {len(history['train_loss'])}")
    axes[1, 1].set_title('训练统计')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 训练示例
    print("RCS小波网络训练模块测试")

    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建配置
    model_params = {'input_dim': 9, 'hidden_dims': [128, 256]}
    training_config = create_training_config()
    data_config = create_data_config()

    print("配置创建完成")
    print(f"训练配置: {training_config}")
    print(f"数据配置: {data_config}")

    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("训练模块准备就绪")