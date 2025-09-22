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
        self.parameters = torch.tensor(parameters, dtype=torch.float32)
        self.rcs_data = torch.tensor(rcs_data, dtype=torch.float32)
        self.transform = transform
        self.augment = augment

        # 数据标准化
        self.param_scaler = StandardScaler()
        self.parameters = torch.tensor(
            self.param_scaler.fit_transform(parameters),
            dtype=torch.float32
        )

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
            'symmetry': 0.3 - 0.2 * progress,  # 对称性约束逐渐减弱
            'frequency_consistency': 0.1 - 0.05 * progress,
            'multiscale': 0.5 - 0.3 * progress  # 多尺度损失逐渐减弱
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
                       'frequency_consistency': 0, 'multiscale': 0}
        num_batches = 0

        for batch_idx, (params, targets) in enumerate(train_loader):
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

        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate_epoch(self, val_loader: TorchDataLoader,
                      loss_fn: TriDimensionalRCSLoss) -> Dict[str, float]:
        """
        验证一个epoch
        """
        self.model.eval()

        epoch_losses = {'total': 0, 'mse': 0, 'symmetry': 0,
                       'frequency_consistency': 0, 'multiscale': 0}
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
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

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
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_scores = []

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

            train_loader = TorchDataLoader(train_dataset,
                                    batch_size=training_config['batch_size'],
                                    shuffle=True)
            val_loader = TorchDataLoader(val_dataset,
                                  batch_size=training_config['batch_size'],
                                  shuffle=False)

            # 创建模型
            model = create_model(**self.model_params)
            trainer = ProgressiveTrainer(model, self.device)

            # 训练
            best_val_loss = self._train_fold(trainer, train_loader, val_loader,
                                           training_config, fold)

            fold_scores.append(best_val_loss)

        # 计算交叉验证统计
        self.cv_results['fold_scores'] = fold_scores
        self.cv_results['mean_score'] = np.mean(fold_scores)
        self.cv_results['std_score'] = np.std(fold_scores)
        self.cv_results['best_fold'] = np.argmin(fold_scores)

        return self.cv_results

    def _train_fold(self, trainer: ProgressiveTrainer, train_loader: TorchDataLoader,
                   val_loader: TorchDataLoader, config: Dict, fold: int) -> float:
        """
        训练单个fold
        """
        # 优化器和学习率调度器
        optimizer = optim.Adam(trainer.model.parameters(),
                             lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # 损失函数
        loss_fn = create_loss_function(loss_weights=config.get('loss_weights'))

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(config['epochs']):
            # 训练
            train_losses = trainer.train_epoch(train_loader, optimizer, loss_fn,
                                             epoch, config['epochs'])

            # 验证
            val_losses = trainer.validate_epoch(val_loader, loss_fn)

            # 学习率调度
            scheduler.step(val_losses['total'])

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
                print(f"早停于epoch {epoch}")
                break

            # 打印进度
            if epoch % 10 == 0:
                print(f"Fold {fold+1}, Epoch {epoch}: "
                      f"Train Loss: {train_losses['total']:.4f}, "
                      f"Val Loss: {val_losses['total']:.4f}")

        return best_val_loss


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
        加载双频RCS数据
        """
        data_dir = self.data_config['rcs_data_dir']
        model_ids = self.data_config['model_ids']
        frequencies = self.data_config['frequencies']  # ['1.5G', '3G']

        rcs_list = []

        for model_id in model_ids:
            model_rcs_list = []

            for freq in frequencies:
                try:
                    # 使用现有的数据读取函数
                    data = rdr.get_adaptive_rcs_matrix(model_id, freq, data_dir)
                    rcs_linear = data['rcs_linear']  # [91, 91]
                    model_rcs_list.append(rcs_linear)

                except Exception as e:
                    print(f"警告: 无法加载模型 {model_id} 频率 {freq}: {e}")
                    # 创建零填充数据
                    model_rcs_list.append(np.zeros((91, 91)))

            # 组合双频数据 [91, 91, 2]
            if len(model_rcs_list) == 2:
                model_rcs = np.stack(model_rcs_list, axis=2)
                rcs_list.append(model_rcs)

        return np.array(rcs_list)  # [N, 91, 91, 2]


def create_training_config() -> Dict:
    """
    创建默认训练配置
    """
    return {
        'batch_size': 8,  # 小数据集使用小批次
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 200,
        'early_stopping_patience': 20,
        'loss_weights': {
            'mse': 1.0,
            'symmetry': 0.1,
            'frequency_consistency': 0.05,
            'multiscale': 0.2
        }
    }


def create_data_config() -> Dict:
    """
    创建默认数据配置
    """
    return {
        'params_file': r"..\parameter\parameters_sorted.csv",
        'rcs_data_dir': r"..\parameter\csv_output",
        'model_ids': [f"{i:03d}" for i in range(1, 101)],  # 001-100
        'frequencies': ['1.5G', '3G']
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