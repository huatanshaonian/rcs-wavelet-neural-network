"""
角度RCS训练器 (AngleRCSTrainer)

单阶段端到端训练，直接从角度+参数预测单点RCS值。

训练流程：
1. 创建DataLoader（支持完整/子采样）
2. 创建优化器和学习率调度器
3. 训练循环（Early Stopping）
4. Checkpoint保存/加载
5. 评估和可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Optional, Dict, Tuple
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ..models.angle_rcs_network import AngleRCSNetwork
    from ..data.angle_dataset import create_dataloaders
except ImportError:
    from angle_based_rcs.models.angle_rcs_network import AngleRCSNetwork
    from angle_based_rcs.data.angle_dataset import create_dataloaders


class EarlyStopping:
    """
    Early Stopping机制

    参数：
        patience (int): 容忍的epoch数
        min_delta (float): 最小改进阈值
        mode (str): 'min' or 'max'
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        检查是否应该early stop

        参数：
            score: 当前验证指标
            epoch: 当前epoch

        返回：
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = (self.best_score - score) > self.min_delta
        else:
            improved = (score - self.best_score) > self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class AngleRCSTrainer:
    """
    角度RCS训练器

    参数：
        model (AngleRCSNetwork): 模型
        device (str): 设备 ('cuda' or 'cpu')
        checkpoint_dir (str): checkpoint保存目录
    """

    def __init__(self,
                 model: AngleRCSNetwork,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = './checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # 创建checkpoint目录
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def _create_optimizer_and_scheduler(self,
                                        lr: float = 1e-4,
                                        optimizer_type: str = 'adam',
                                        scheduler_type: str = 'cosine',
                                        weight_decay: float = 1e-5,
                                        epochs: int = 200,
                                        log_fn=None,
                                        **kwargs) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        创建优化器和学习率调度器（复用ae_trainer.py的逻辑）

        参数：
            lr: 学习率
            optimizer_type: 优化器类型 ('adam', 'adamw', 'sgd', 'lbfgs')
            scheduler_type: 调度器类型 ('constant', 'cosine', 'cosine_restart', 'adaptive', 'multi_stage', 'adaptive_multi_stage')
            weight_decay: 权重衰减
            epochs: 总epoch数（用于cosine scheduler）
            log_fn: 日志函数（可选）
            **kwargs: 其他参数

        返回：
            optimizer, scheduler
        """
        # 如果没有log_fn，使用print
        if log_fn is None:
            log_fn = print

        params = self.model.parameters()

        # 创建优化器
        if optimizer_type == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            log_fn(f"✅ 优化器: Adam (lr={lr:.2e}, wd={weight_decay:.2e})")
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            log_fn(f"✅ 优化器: AdamW (lr={lr:.2e}, wd={weight_decay:.2e})")
        elif optimizer_type == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            log_fn(f"✅ 优化器: SGD (lr={lr:.2e}, momentum={momentum}, wd={weight_decay:.2e})")
        elif optimizer_type == 'lbfgs':
            max_iter = kwargs.get('lbfgs_max_iter', 20)
            history_size = kwargs.get('lbfgs_history_size', 100)
            optimizer = optim.LBFGS(params, lr=lr, max_iter=max_iter, history_size=history_size)
            log_fn(f"✅ 优化器: L-BFGS (lr={lr:.2e}, max_iter={max_iter}, history_size={history_size})")
            log_fn(f"⚠️ L-BFGS需要闭包，训练循环已自动适配")
        else:
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            log_fn(f"⚠️ 未知优化器 '{optimizer_type}'，默认使用Adam")

        # 创建学习率调度器
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
            log_fn(f"✅ 调度器: CosineAnnealingLR (T_max={epochs})")
        elif scheduler_type == 'cosine_restart':
            T_0 = kwargs.get('restart_period', 50)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0)
            log_fn(f"✅ 调度器: CosineAnnealingWarmRestarts (T_0={T_0})")
        elif scheduler_type == 'adaptive':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            log_fn(f"✅ 调度器: ReduceLROnPlateau (patience=10)")
        elif scheduler_type in ['multi_stage', 'adaptive_multi_stage']:
            # 多阶段学习率调度器（从AE复用）
            from autoencoder.training.multi_stage_scheduler import create_patience_driven_scheduler
            num_stages = kwargs.get('num_lr_stages', 3)
            lr_decay = kwargs.get('lr_decay_factor', 0.1)
            patience = kwargs.get('patience', 50)

            scheduler = create_patience_driven_scheduler(
                optimizer,
                initial_lr=lr,
                num_stages=num_stages,
                lr_decay_factor=lr_decay,
                initial_patience=patience,
                verbose=True
            )
            log_fn(f"✅ 调度器: {scheduler_type} (阶段数={num_stages}, LR衰减={lr_decay}, 初始patience={patience})")
        else:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
            log_fn(f"✅ 调度器: Constant")

        return optimizer, scheduler

    def _train_epoch(self,
                     train_loader: DataLoader,
                     optimizer: optim.Optimizer,
                     criterion: nn.Module,
                     is_lbfgs: bool = False,
                     epoch: int = 0,
                     log_fn=None,
                     print_batch_every: int = 500) -> float:
        """
        训练一个epoch

        参数：
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            is_lbfgs: 是否使用L-BFGS
            epoch: 当前epoch数（用于日志）
            log_fn: 日志函数
            print_batch_every: 每隔多少batch输出一次进度

        返回：
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader, 1):
            # GPU预加载模式：数据已在GPU，直接使用（避免冗余.to()调用）
            # 常规模式：需要移动到device
            theta = batch['theta'] if batch['theta'].device == self.device else batch['theta'].to(self.device)
            phi = batch['phi'] if batch['phi'].device == self.device else batch['phi'].to(self.device)
            params = batch['params'] if batch['params'].device == self.device else batch['params'].to(self.device)
            freq_idx = batch['freq_idx'] if batch['freq_idx'].device == self.device else batch['freq_idx'].to(self.device)
            target_rcs = batch['target_rcs'] if batch['target_rcs'].device == self.device else batch['target_rcs'].to(self.device)

            batch_size = theta.size(0)

            if is_lbfgs:
                # L-BFGS需要闭包
                def closure():
                    optimizer.zero_grad()
                    rcs_pred = self.model(theta, phi, params, freq_idx).squeeze()
                    loss = criterion(rcs_pred, target_rcs)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                total_loss += loss.item() * batch_size
            else:
                # 标准优化器
                optimizer.zero_grad()
                rcs_pred = self.model(theta, phi, params, freq_idx).squeeze()
                loss = criterion(rcs_pred, target_rcs)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size

            total_samples += batch_size

            # 定期输出batch进度（避免长时间无输出）
            if log_fn and print_batch_every > 0 and batch_idx % print_batch_every == 0:
                current_avg_loss = total_loss / total_samples
                progress_pct = 100.0 * batch_idx / num_batches
                log_fn(f"  Epoch {epoch} - Batch [{batch_idx:4d}/{num_batches}] ({progress_pct:5.1f}%) | "
                       f"Avg Loss: {current_avg_loss:.6f}")

        return total_loss / total_samples

    def _validate_epoch(self,
                        val_loader: DataLoader,
                        criterion: nn.Module) -> float:
        """
        验证一个epoch

        参数：
            val_loader: 验证数据加载器
            criterion: 损失函数

        返回：
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                # GPU预加载模式：数据已在GPU，直接使用（避免冗余.to()调用）
                # 常规模式：需要移动到device
                theta = batch['theta'] if batch['theta'].device == self.device else batch['theta'].to(self.device)
                phi = batch['phi'] if batch['phi'].device == self.device else batch['phi'].to(self.device)
                params = batch['params'] if batch['params'].device == self.device else batch['params'].to(self.device)
                freq_idx = batch['freq_idx'] if batch['freq_idx'].device == self.device else batch['freq_idx'].to(self.device)
                target_rcs = batch['target_rcs'] if batch['target_rcs'].device == self.device else batch['target_rcs'].to(self.device)

                batch_size = theta.size(0)

                rcs_pred = self.model(theta, phi, params, freq_idx).squeeze()
                loss = criterion(rcs_pred, target_rcs)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / total_samples

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 200,
              lr: float = 1e-4,
              optimizer_type: str = 'adam',
              scheduler_type: str = 'cosine',
              patience: int = 50,
              weight_decay: float = 1e-5,
              print_every: int = 5,
              log_callback=None,
              **kwargs) -> Dict:
        """
        完整训练流程

        参数：
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练epoch数
            lr: 学习率
            optimizer_type: 优化器类型
            scheduler_type: 调度器类型
            patience: Early Stopping patience
            weight_decay: 权重衰减
            print_every: 打印频率
            log_callback: 日志回调函数（可选）
            **kwargs: 其他参数

        返回：
            训练历史字典
        """
        # 定义打印函数
        def log(message):
            if log_callback:
                log_callback(message)
            else:
                print(message)

        log("=" * 80)
        log("开始训练 AngleRCSNetwork")
        log("=" * 80)

        # 创建优化器和调度器
        optimizer, scheduler = self._create_optimizer_and_scheduler(
            lr=lr,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,  # 传递patience给multi_stage调度器
            log_fn=log,  # 传递log函数
            **kwargs
        )

        # 损失函数
        criterion = nn.MSELoss()

        # Early Stopping
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-6, mode='min')

        # 是否L-BFGS
        is_lbfgs = (optimizer_type == 'lbfgs')

        # 最佳模型保存
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        # 训练循环
        log(f"\n训练配置:")
        log(f"  Epochs: {epochs}")
        log(f"  Train batches: {len(train_loader)}")
        log(f"  Val batches: {len(val_loader)}")
        log(f"  Patience: {patience}")
        log(f"  Device: {self.device}")
        log("")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 训练（传递epoch和log函数用于batch进度输出）
            train_loss = self._train_epoch(train_loader, optimizer, criterion, is_lbfgs,
                                          epoch=epoch, log_fn=log, print_batch_every=500)

            # 验证
            val_loss = self._validate_epoch(val_loader, criterion)

            # 学习率调度
            if scheduler_type == 'adaptive':
                scheduler.step(val_loss)
            elif not is_lbfgs:  # L-BFGS不需要step
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }

            # 打印进度
            if epoch % print_every == 0 or epoch == 1:
                epoch_time = time.time() - epoch_start
                log(f"Epoch [{epoch:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.2f}s")

            # Early Stopping检查
            if early_stopping(val_loss, epoch):
                log(f"\n早停触发! 最佳epoch: {early_stopping.best_epoch}, "
                    f"最佳验证损失: {early_stopping.best_score:.6f}")
                break

        total_time = time.time() - start_time

        # 恢复最佳模型
        if best_model_state is not None:
            log(f"\n恢复最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.6f})")
            self.model.load_state_dict(best_model_state['model_state_dict'])

        log("=" * 80)
        log(f"训练完成!")
        log(f"  总耗时: {total_time/60:.2f}分钟")
        log(f"  最佳Epoch: {best_epoch}/{epoch}")
        log(f"  最佳验证损失: {best_val_loss:.6f}")
        log(f"  最终训练损失: {train_loss:.6f}")
        log("=" * 80)

        # 保存最终checkpoint
        self.save_checkpoint('final_model.pth', best_model_state)

        return {
            'train_losses': self.history['train_loss'],
            'val_losses': self.history['val_loss'],
            'learning_rates': self.history['learning_rate'],
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_time': total_time
        }

    def save_checkpoint(self, filename: str, state_dict: Optional[Dict] = None):
        """保存checkpoint"""
        if state_dict is None:
            state_dict = {
                'model_state_dict': self.model.state_dict(),
                'history': self.history
            }

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state_dict, filepath)
        log_fn(f"✅ Checkpoint保存: {filepath}")

    def load_checkpoint(self, filename: str):
        """加载checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        log_fn(f"✅ Checkpoint加载: {filepath}")
        return checkpoint


if __name__ == "__main__":
    log("=" * 80)
    log_fn("AngleRCSTrainer 测试")
    log("=" * 80)

    # 创建模型
    log_fn("\n[Test 1] 创建模型")
    model = AngleRCSNetwork(num_frequencies=3, angle_L=16, activation='sin')
    param_stats = model.count_parameters()
    log_fn(f"模型参数: {param_stats['total']:,}")

    # 创建训练器
    log_fn("\n[Test 2] 创建训练器")
    trainer = AngleRCSTrainer(model, device='cpu', checkpoint_dir='./test_checkpoints')

    # 创建模拟数据
    log_fn("\n[Test 3] 创建模拟数据")
    rcs_data = np.random.rand(200, 91, 91, 3) * 0.5
    param_data = np.random.randn(200, 9)

    train_loader, val_loader, sampler = create_dataloaders(
        rcs_data=rcs_data,
        param_data=param_data,
        batch_size=256,
        train_subset_size=5000,  # 小数据集测试
        normalize_params=True
    )

    log_fn(f"训练批次: {len(train_loader)}")
    log_fn(f"验证批次: {len(val_loader)}")

    # 训练（少量epoch测试）
    log_fn("\n[Test 4] 训练测试（5 epochs）")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-3,
        optimizer_type='adam',
        scheduler_type='constant',
        patience=50,
        print_every=1
    )

    log_fn(f"\n训练历史:")
    log_fn(f"  Train losses: {[f'{x:.6f}' for x in history['train_losses']]}")
    log_fn(f"  Val losses: {[f'{x:.6f}' for x in history['val_losses']]}")
    log_fn(f"  最佳epoch: {history['best_epoch']}")
    log(f"  最佳验证损失: {history['best_val_loss']:.6f}")

    # 清理测试文件
    import shutil
    if os.path.exists('./test_checkpoints'):
        shutil.rmtree('./test_checkpoints')
        log_fn("\n✅ 测试checkpoint已清理")

    log_fn("\n" + "=" * 80)
    log_fn("[PASS] AngleRCSTrainer测试通过!")
    log("=" * 80)
