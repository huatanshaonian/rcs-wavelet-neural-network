"""
多阶段学习率调度器（Patience驱动版本）

不再按照epoch数划分阶段，而是基于patience自动切换：
- 从高学习率开始训练
- 当patience耗尽时（验证损失连续N轮无改善），自动降低学习率
- 每次降低学习率时，patience翻倍，回退到最佳模型继续训练
- 充分训练每个学习率阶段，直到真正无法改善为止

设计理念：
- 前期高学习率 + 小patience：快速探索
- 后期低学习率 + 大patience：精细微调
- 自适应切换：根据实际训练情况决定何时降低学习率

作者：Claude Code
日期：2025-01-18（重新设计）
"""

import torch
from typing import List, Optional


class PatienceDrivenMultiStageLRScheduler:
    """
    Patience驱动的多阶段学习率调度器

    不预先划分epoch，而是当patience耗尽时自动切换到下一阶段（降低学习率）。
    每次切换时patience增加，确保后期有足够的训练时间。

    Args:
        optimizer: PyTorch优化器
        lr_stages: 学习率序列，例如[1e-3, 1e-4, 1e-5]
        initial_patience: 初始patience（第一阶段）
        patience_multiplier: 每次切换阶段时patience的倍增因子（默认2.0，即翻倍）
        verbose: 是否打印阶段切换信息

    工作流程:
        1. 开始：lr=1e-3, patience=30
        2. 训练直到patience耗尽（验证损失30轮无改善）
        3. 切换：lr=1e-4, patience=60, 回退到最佳模型
        4. 继续训练直到patience耗尽
        5. 切换：lr=1e-5, patience=120, 回退到最佳模型
        6. 最后阶段patience耗尽 → 真正早停

    示例:
        >>> lr_stages = [1e-3, 1e-4, 1e-5]
        >>> scheduler = PatienceDrivenMultiStageLRScheduler(
        ...     optimizer,
        ...     lr_stages=lr_stages,
        ...     initial_patience=30,
        ...     patience_multiplier=2.0
        ... )
        >>>
        >>> # 训练循环中
        >>> for epoch in range(max_epochs):
        ...     train_loss = train(...)
        ...     val_loss = validate(...)
        ...
        ...     # 检查是否需要切换阶段
        ...     if val_loss < best_val_loss:
        ...         best_val_loss = val_loss
        ...         patience_counter = 0
        ...     else:
        ...         patience_counter += 1
        ...
        ...     # 获取当前patience
        ...     current_patience = scheduler.get_current_patience()
        ...
        ...     if patience_counter >= current_patience:
        ...         # 尝试切换到下一阶段
        ...         switched = scheduler.switch_to_next_stage()
        ...         if switched:
        ...             # 切换成功：回退到最佳模型，重置计数器
        ...             model.load_state_dict(best_model_state)
        ...             patience_counter = 0
        ...             best_val_loss = val_loss  # 重置最佳损失
        ...         else:
        ...             # 已是最后阶段，真正早停
        ...             break
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_stages: List[float],
        initial_patience: int = 30,
        patience_multiplier: float = 2.0,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.lr_stages = lr_stages
        self.initial_patience = initial_patience
        self.patience_multiplier = patience_multiplier
        self.verbose = verbose

        # 当前阶段索引
        self.current_stage = 0

        # 设置初始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_stages[0]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PatienceDrivenMultiStageLRScheduler 初始化")
            print(f"{'='*60}")
            print(f"学习率序列: {self.lr_stages}")
            print(f"初始patience: {self.initial_patience}")
            print(f"Patience倍增因子: {self.patience_multiplier}")
            print(f"总阶段数: {len(self.lr_stages)}")
            print(f"")
            for i, lr in enumerate(self.lr_stages):
                patience = self._calculate_patience(i)
                print(f"  Stage {i+1}: LR={lr:.2e}, Patience={patience}")
            print(f"{'='*60}\n")

    def _calculate_patience(self, stage_idx: int) -> int:
        """计算指定阶段的patience"""
        return int(self.initial_patience * (self.patience_multiplier ** stage_idx))

    def get_current_patience(self) -> int:
        """获取当前阶段的patience"""
        return self._calculate_patience(self.current_stage)

    def get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

    def get_current_stage(self) -> int:
        """获取当前阶段索引（0-based）"""
        return self.current_stage

    def get_total_stages(self) -> int:
        """获取总阶段数"""
        return len(self.lr_stages)

    def has_next_stage(self) -> bool:
        """是否还有下一个阶段"""
        return self.current_stage < len(self.lr_stages) - 1

    def switch_to_next_stage(self) -> bool:
        """
        切换到下一个阶段（降低学习率，增加patience）

        Returns:
            bool: 如果成功切换到下一阶段返回True，如果已是最后阶段返回False
        """
        if not self.has_next_stage():
            if self.verbose:
                print(f"\n⚠️ 已是最后阶段 (Stage {self.current_stage + 1})，无法继续切换")
            return False

        # 切换到下一阶段
        old_stage = self.current_stage
        old_lr = self.get_current_lr()
        old_patience = self.get_current_patience()

        self.current_stage += 1
        new_lr = self.lr_stages[self.current_stage]
        new_patience = self.get_current_patience()

        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔄 切换到 Stage {self.current_stage + 1}/{len(self.lr_stages)}")
            print(f"{'='*60}")
            print(f"学习率: {old_lr:.2e} → {new_lr:.2e} (降低{old_lr/new_lr:.1f}x)")
            print(f"Patience: {old_patience} → {new_patience} (增加{self.patience_multiplier}x)")
            print(f"💡 提示：请回退到最佳模型并重置patience计数器")
            print(f"{'='*60}\n")

        return True

    def get_stage_info(self) -> dict:
        """获取当前阶段的详细信息"""
        return {
            'current_stage': self.current_stage + 1,
            'total_stages': len(self.lr_stages),
            'current_lr': self.get_current_lr(),
            'current_patience': self.get_current_patience(),
            'has_next_stage': self.has_next_stage(),
            'next_lr': self.lr_stages[self.current_stage + 1] if self.has_next_stage() else None,
            'next_patience': self._calculate_patience(self.current_stage + 1) if self.has_next_stage() else None
        }

    def __repr__(self):
        info = self.get_stage_info()
        return (
            f"PatienceDrivenMultiStageLRScheduler("
            f"stage={info['current_stage']}/{info['total_stages']}, "
            f"lr={info['current_lr']:.2e}, "
            f"patience={info['current_patience']})"
        )


def create_patience_driven_scheduler(
    optimizer: torch.optim.Optimizer,
    initial_lr: float = 1e-3,
    num_stages: int = 3,
    lr_decay_factor: float = 0.1,
    initial_patience: int = 30,
    patience_multiplier: float = 2.0,
    verbose: bool = True
) -> PatienceDrivenMultiStageLRScheduler:
    """
    便捷函数：创建patience驱动的多阶段学习率调度器

    自动生成学习率序列：initial_lr, initial_lr * decay, initial_lr * decay^2, ...

    Args:
        optimizer: PyTorch优化器
        initial_lr: 初始学习率（默认1e-3）
        num_stages: 阶段数量（默认3）
        lr_decay_factor: 学习率衰减因子（默认0.1，即每阶段降低10倍）
        initial_patience: 初始patience（默认30）
        patience_multiplier: patience倍增因子（默认2.0，即翻倍）
        verbose: 是否打印详细信息

    Returns:
        PatienceDrivenMultiStageLRScheduler实例

    示例:
        >>> # 创建3阶段调度器：1e-3 → 1e-4 → 1e-5, patience: 30 → 60 → 120
        >>> scheduler = create_patience_driven_scheduler(
        ...     optimizer,
        ...     initial_lr=1e-3,
        ...     num_stages=3,
        ...     lr_decay_factor=0.1,
        ...     initial_patience=30
        ... )
    """
    # 生成学习率序列
    lr_stages = []
    current_lr = initial_lr
    for _ in range(num_stages):
        lr_stages.append(current_lr)
        current_lr *= lr_decay_factor

    return PatienceDrivenMultiStageLRScheduler(
        optimizer=optimizer,
        lr_stages=lr_stages,
        initial_patience=initial_patience,
        patience_multiplier=patience_multiplier,
        verbose=verbose
    )


# ==================== 向后兼容：保留旧接口 ====================
# 注意：以下是旧的基于epoch的调度器，已废弃，仅保留向后兼容

from torch.optim.lr_scheduler import _LRScheduler


class MultiStageLRScheduler(_LRScheduler):
    """
    [已废弃] 基于epoch的多阶段学习率调度器

    请使用新的 PatienceDrivenMultiStageLRScheduler 代替。
    """

    def __init__(self, optimizer, stages, verbose=False):
        print("⚠️ 警告：MultiStageLRScheduler已废弃，建议使用PatienceDrivenMultiStageLRScheduler")
        self.stages = stages
        self.verbose = verbose
        self.stage_boundaries = []
        cumulative_epochs = 0
        for stage in stages:
            start = cumulative_epochs
            end = cumulative_epochs + stage['epochs']
            self.stage_boundaries.append({
                'start': start,
                'end': end,
                'lr': stage['lr'],
                'patience': stage.get('patience', 50)
            })
            cumulative_epochs = end
        self.total_epochs = cumulative_epochs
        self.current_stage_idx = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.stages[0]['lr']
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        current_epoch = self.last_epoch
        for idx, stage in enumerate(self.stage_boundaries):
            if stage['start'] <= current_epoch < stage['end']:
                self.current_stage_idx = idx
                return [stage['lr'] for _ in self.optimizer.param_groups]
        return [self.stage_boundaries[-1]['lr'] for _ in self.optimizer.param_groups]

    def get_current_stage(self):
        return self.current_stage_idx

    def get_current_patience(self):
        if self.current_stage_idx < len(self.stage_boundaries):
            return self.stage_boundaries[self.current_stage_idx]['patience']
        return self.stage_boundaries[-1]['patience']


class AdaptiveMultiStageLRScheduler(MultiStageLRScheduler):
    """[已废弃] 自适应多阶段学习率调度器"""

    def __init__(self, optimizer, stages, early_stage_switch=True, switch_patience=None, verbose=False):
        print("⚠️ 警告：AdaptiveMultiStageLRScheduler已废弃，建议使用PatienceDrivenMultiStageLRScheduler")
        super().__init__(optimizer, stages, verbose)
        self.early_stage_switch = early_stage_switch
        self.switch_patience = switch_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0


def create_multi_stage_scheduler(optimizer, total_epochs, initial_lr=1e-3, num_stages=3,
                                   lr_decay_factor=0.1, adaptive=False, verbose=True):
    """[已废弃] 请使用 create_patience_driven_scheduler"""
    print("⚠️ 警告：create_multi_stage_scheduler已废弃，建议使用create_patience_driven_scheduler")
    return create_patience_driven_scheduler(
        optimizer, initial_lr, num_stages, lr_decay_factor,
        initial_patience=30, patience_multiplier=2.0, verbose=verbose
    )
