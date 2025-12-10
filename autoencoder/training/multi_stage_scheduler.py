"""
多阶段学习率调度器

支持在训练过程中分阶段调整学习率和早停patience，实现更精细的训练控制。

用途：
- 前期使用高学习率快速收敛
- 中期降低学习率精细调整
- 后期使用极低学习率微调，提高模型性能

作者：Claude Code
日期：2025-01-18
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Dict, Optional


class MultiStageLRScheduler(_LRScheduler):
    """
    多阶段学习率调度器

    在训练过程中按照预定义的阶段切换学习率，每个阶段可以有不同的：
    - 学习率
    - epoch范围
    - 建议的早停patience

    Args:
        optimizer: PyTorch优化器
        stages: 阶段配置列表，每个阶段包含：
            - 'epochs': 该阶段的epoch数量
            - 'lr': 该阶段的学习率
            - 'patience': 建议的早停patience（可选）
        verbose: 是否打印阶段切换信息

    示例:
        >>> stages = [
        ...     {'epochs': 100, 'lr': 1e-3, 'patience': 30},   # Phase 1: 快速下降
        ...     {'epochs': 100, 'lr': 1e-4, 'patience': 50},   # Phase 2: 精细调整
        ...     {'epochs': 100, 'lr': 1e-5, 'patience': 100},  # Phase 3: 微调
        ... ]
        >>> scheduler = MultiStageLRScheduler(optimizer, stages)
        >>>
        >>> # 训练循环
        >>> for epoch in range(total_epochs):
        ...     train(...)
        ...     val_loss = validate(...)
        ...
        ...     # 获取当前阶段的patience
        ...     current_patience = scheduler.get_current_patience()
        ...
        ...     # 早停检查
        ...     if patience_counter >= current_patience:
        ...         break
        ...
        ...     scheduler.step()  # 更新学习率
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        stages: List[Dict],
        verbose: bool = False
    ):
        self.stages = stages
        self.verbose = verbose

        # 计算每个阶段的起止epoch
        self.stage_boundaries = []
        cumulative_epochs = 0
        for stage in stages:
            start = cumulative_epochs
            end = cumulative_epochs + stage['epochs']
            self.stage_boundaries.append({
                'start': start,
                'end': end,
                'lr': stage['lr'],
                'patience': stage.get('patience', 50)  # 默认patience=50
            })
            cumulative_epochs = end

        self.total_epochs = cumulative_epochs
        self.current_stage_idx = 0

        # 初始化优化器学习率为第一个阶段
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.stages[0]['lr']

        super().__init__(optimizer, last_epoch=-1)

        if self.verbose:
            print(f"MultiStageLRScheduler初始化:")
            for i, stage in enumerate(self.stage_boundaries):
                print(f"  Stage {i+1}: Epochs {stage['start']}-{stage['end']}, "
                      f"LR={stage['lr']:.2e}, Patience={stage['patience']}")

    def get_lr(self):
        """返回当前epoch的学习率"""
        current_epoch = self.last_epoch

        # 找到当前epoch所属的阶段
        for idx, stage in enumerate(self.stage_boundaries):
            if stage['start'] <= current_epoch < stage['end']:
                self.current_stage_idx = idx
                return [stage['lr'] for _ in self.optimizer.param_groups]

        # 如果超出所有阶段，使用最后一个阶段的学习率
        return [self.stage_boundaries[-1]['lr'] for _ in self.optimizer.param_groups]

    def get_current_stage(self) -> int:
        """获取当前阶段索引 (0-based)"""
        return self.current_stage_idx

    def get_current_patience(self) -> int:
        """获取当前阶段建议的早停patience"""
        if self.current_stage_idx < len(self.stage_boundaries):
            return self.stage_boundaries[self.current_stage_idx]['patience']
        return self.stage_boundaries[-1]['patience']

    def get_stage_info(self) -> Dict:
        """获取当前阶段的完整信息"""
        if self.current_stage_idx < len(self.stage_boundaries):
            stage = self.stage_boundaries[self.current_stage_idx]
            return {
                'stage': self.current_stage_idx + 1,
                'total_stages': len(self.stage_boundaries),
                'lr': stage['lr'],
                'patience': stage['patience'],
                'epoch_start': stage['start'],
                'epoch_end': stage['end'],
                'epoch_in_stage': self.last_epoch - stage['start'] + 1,
                'epochs_in_stage': stage['end'] - stage['start']
            }
        return {}

    def step(self, epoch=None):
        """
        更新学习率

        Args:
            epoch: 当前epoch（可选，默认使用内部计数器）
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        # 检查是否切换到新阶段
        old_stage = self.current_stage_idx
        super().step(epoch)
        new_stage = self.current_stage_idx

        if old_stage != new_stage and self.verbose:
            info = self.get_stage_info()
            print(f"\n{'='*60}")
            print(f"🔄 切换到Stage {info['stage']}/{info['total_stages']}")
            print(f"   学习率: {info['lr']:.2e}")
            print(f"   建议Patience: {info['patience']}")
            print(f"   Epoch范围: {info['epoch_start']}-{info['epoch_end']}")
            print(f"{'='*60}\n")


class AdaptiveMultiStageLRScheduler(MultiStageLRScheduler):
    """
    自适应多阶段学习率调度器

    在MultiStageLRScheduler基础上增加自适应功能：
    - 如果当前阶段验证损失不再下降，提前进入下一阶段
    - 支持动态调整patience

    Args:
        optimizer: PyTorch优化器
        stages: 阶段配置列表
        early_stage_switch: 是否启用提前阶段切换
        switch_patience: 触发阶段切换的patience（默认为当前阶段patience的一半）
        verbose: 是否打印详细信息
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        stages: List[Dict],
        early_stage_switch: bool = True,
        switch_patience: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(optimizer, stages, verbose)

        self.early_stage_switch = early_stage_switch
        self.switch_patience = switch_patience

        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.stage_switched = False

    def check_and_switch_stage(self, val_loss: float) -> bool:
        """
        检查验证损失并决定是否提前切换阶段

        Args:
            val_loss: 当前验证损失

        Returns:
            是否切换了阶段
        """
        if not self.early_stage_switch:
            return False

        # 更新最佳损失
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1

        # 计算当前阶段的切换patience
        current_stage_patience = self.stage_boundaries[self.current_stage_idx]['patience']
        switch_patience = self.switch_patience if self.switch_patience else current_stage_patience // 2

        # 如果patience耗尽且不是最后一个阶段，切换到下一阶段
        if self.patience_counter >= switch_patience and self.current_stage_idx < len(self.stages) - 1:
            if self.verbose:
                print(f"\n⚡ 提前切换阶段：验证损失连续{switch_patience}轮无改善")

            # 强制切换到下一阶段
            next_stage = self.current_stage_idx + 1
            self.last_epoch = self.stage_boundaries[next_stage]['start'] - 1
            self.step()  # 这会触发阶段切换

            # 重置计数器
            self.patience_counter = 0
            self.best_val_loss = val_loss

            return True

        return False

    def get_dynamic_patience(self) -> int:
        """
        获取动态调整的patience

        根据当前阶段和训练进度动态调整patience：
        - 阶段初期：使用较小的patience（快速响应）
        - 阶段后期：使用较大的patience（更有耐心）
        """
        base_patience = self.get_current_patience()

        # 计算当前阶段的进度 (0-1)
        stage = self.stage_boundaries[self.current_stage_idx]
        progress = (self.last_epoch - stage['start']) / (stage['end'] - stage['start'])

        # 前50%使用基础patience，后50%线性增长到1.5倍
        if progress < 0.5:
            return base_patience
        else:
            multiplier = 1.0 + (progress - 0.5) * 1.0  # 0.5->1.0: 1.0x->1.5x
            return int(base_patience * multiplier)


def create_multi_stage_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    initial_lr: float = 1e-3,
    num_stages: int = 3,
    lr_decay_factor: float = 0.1,
    adaptive: bool = False,
    verbose: bool = True
) -> MultiStageLRScheduler:
    """
    便捷函数：创建多阶段学习率调度器

    自动将total_epochs分成num_stages个阶段，每个阶段学习率按lr_decay_factor衰减

    Args:
        optimizer: PyTorch优化器
        total_epochs: 总训练轮数
        initial_lr: 初始学习率
        num_stages: 阶段数量（默认3）
        lr_decay_factor: 学习率衰减因子（默认0.1，即每阶段降低10倍）
        adaptive: 是否使用自适应调度器
        verbose: 是否打印信息

    Returns:
        MultiStageLRScheduler或AdaptiveMultiStageLRScheduler实例

    示例:
        >>> # 300 epochs分3个阶段: 1e-3 -> 1e-4 -> 1e-5
        >>> scheduler = create_multi_stage_scheduler(
        ...     optimizer,
        ...     total_epochs=300,
        ...     initial_lr=1e-3,
        ...     num_stages=3
        ... )
    """
    epochs_per_stage = total_epochs // num_stages
    remainder = total_epochs % num_stages

    stages = []
    current_lr = initial_lr

    for i in range(num_stages):
        # 最后一个阶段包含余数epochs
        stage_epochs = epochs_per_stage + (remainder if i == num_stages - 1 else 0)

        # patience随着阶段增加而增加（后期更有耐心）
        base_patience = 30
        stage_patience = base_patience * (i + 1)

        stages.append({
            'epochs': stage_epochs,
            'lr': current_lr,
            'patience': stage_patience
        })

        # 学习率衰减
        current_lr *= lr_decay_factor

    if adaptive:
        return AdaptiveMultiStageLRScheduler(optimizer, stages, verbose=verbose)
    else:
        return MultiStageLRScheduler(optimizer, stages, verbose=verbose)
