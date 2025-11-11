"""
梯度监控工具

用于训练过程中监控梯度统计信息，帮助诊断训练问题。

功能：
1. 计算梯度范数（L2 norm）
2. 统计梯度分布（均值、标准差、最大/最小值）
3. 检测梯度异常（爆炸、消失）
4. 记录梯度历史，用于可视化
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


class GradientMonitor:
    """梯度监控器"""

    def __init__(self,
                 log_interval: int = 10,
                 warn_threshold_high: float = 10.0,
                 warn_threshold_low: float = 1e-5):
        """
        初始化梯度监控器

        Args:
            log_interval: 每隔多少步记录一次梯度统计（默认10）
            warn_threshold_high: 梯度范数过大的警告阈值（默认10.0）
            warn_threshold_low: 梯度范数过小的警告阈值（默认1e-5）
        """
        self.log_interval = log_interval
        self.warn_threshold_high = warn_threshold_high
        self.warn_threshold_low = warn_threshold_low

        # 梯度历史记录
        self.history = {
            'step': [],
            'grad_norm': [],
            'grad_mean': [],
            'grad_std': [],
            'grad_max': [],
            'grad_min': []
        }

        self.step_count = 0

    def compute_gradient_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        计算模型所有参数的梯度统计信息

        Args:
            model: PyTorch模型

        Returns:
            梯度统计字典，包含：
            - grad_norm: L2范数（最重要）
            - grad_mean: 均值
            - grad_std: 标准差
            - grad_max: 最大值
            - grad_min: 最小值
            - num_params: 参数总数
        """
        # 收集所有梯度
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.detach().cpu().flatten())

        if not all_grads:
            return {
                'grad_norm': 0.0,
                'grad_mean': 0.0,
                'grad_std': 0.0,
                'grad_max': 0.0,
                'grad_min': 0.0,
                'num_params': 0
            }

        # 拼接所有梯度
        all_grads_tensor = torch.cat(all_grads)

        # 计算统计信息
        grad_norm = torch.norm(all_grads_tensor, p=2).item()  # L2范数
        grad_mean = torch.mean(all_grads_tensor).item()
        grad_std = torch.std(all_grads_tensor).item()
        grad_max = torch.max(all_grads_tensor).item()
        grad_min = torch.min(all_grads_tensor).item()
        num_params = all_grads_tensor.numel()

        return {
            'grad_norm': grad_norm,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'grad_max': grad_max,
            'grad_min': grad_min,
            'num_params': num_params
        }

    def check_gradients(self,
                       model: torch.nn.Module,
                       step: Optional[int] = None,
                       verbose: bool = True) -> Tuple[Dict[str, float], str]:
        """
        检查梯度并返回统计信息和状态

        Args:
            model: PyTorch模型
            step: 当前训练步数（可选）
            verbose: 是否打印详细信息

        Returns:
            (stats, status)
            - stats: 梯度统计字典
            - status: 'healthy', 'warning', 'exploding', 'vanishing'
        """
        self.step_count += 1
        if step is None:
            step = self.step_count

        # 计算梯度统计
        stats = self.compute_gradient_stats(model)
        grad_norm = stats['grad_norm']

        # 判断梯度状态
        if grad_norm > self.warn_threshold_high:
            status = 'exploding'
            message = f"WARNING: Gradient Exploding! grad_norm={grad_norm:.2e} > {self.warn_threshold_high}"
        elif grad_norm < self.warn_threshold_low:
            status = 'vanishing'
            message = f"WARNING: Gradient Vanishing! grad_norm={grad_norm:.2e} < {self.warn_threshold_low}"
        elif grad_norm > 1.0:
            status = 'warning'
            message = f"CAUTION: Gradient is large: grad_norm={grad_norm:.2e} (recommended: 1e-3 ~ 1e-1)"
        else:
            status = 'healthy'
            message = f"OK: Gradient is healthy: grad_norm={grad_norm:.2e}"

        # 记录历史
        if step % self.log_interval == 0:
            self.history['step'].append(step)
            self.history['grad_norm'].append(grad_norm)
            self.history['grad_mean'].append(stats['grad_mean'])
            self.history['grad_std'].append(stats['grad_std'])
            self.history['grad_max'].append(stats['grad_max'])
            self.history['grad_min'].append(stats['grad_min'])

        # 打印信息
        if verbose and step % self.log_interval == 0:
            print(f"\n[Step {step}] 梯度监控:")
            print(f"  {message}")
            print(f"  梯度统计: mean={stats['grad_mean']:.2e}, std={stats['grad_std']:.2e}")
            print(f"  梯度范围: [{stats['grad_min']:.2e}, {stats['grad_max']:.2e}]")

        return stats, status

    def get_gradient_summary(self) -> str:
        """
        获取梯度监控总结报告

        Returns:
            格式化的总结字符串
        """
        if not self.history['grad_norm']:
            return "暂无梯度记录"

        grad_norms = np.array(self.history['grad_norm'])

        summary = f"""
梯度监控总结报告
{'='*60}
记录步数: {len(grad_norms)}
梯度范数统计:
  均值: {np.mean(grad_norms):.2e}
  中位数: {np.median(grad_norms):.2e}
  标准差: {np.std(grad_norms):.2e}
  最大值: {np.max(grad_norms):.2e}
  最小值: {np.min(grad_norms):.2e}

健康度评估:
  梯度爆炸次数 (>{self.warn_threshold_high}): {np.sum(grad_norms > self.warn_threshold_high)}
  梯度消失次数 (<{self.warn_threshold_low}): {np.sum(grad_norms < self.warn_threshold_low)}
  健康比例: {np.sum((grad_norms >= self.warn_threshold_low) & (grad_norms <= self.warn_threshold_high)) / len(grad_norms) * 100:.1f}%
{'='*60}
"""
        return summary

    def plot_gradient_history(self, save_path: Optional[str] = None):
        """
        绘制梯度历史曲线

        Args:
            save_path: 保存路径（None表示不保存）
        """
        if not self.history['grad_norm']:
            print("暂无梯度记录可绘制")
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        steps = self.history['step']

        # 子图1: 梯度范数
        ax = axes[0, 0]
        ax.plot(steps, self.history['grad_norm'], linewidth=2, color='blue')
        ax.axhline(y=self.warn_threshold_high, color='red', linestyle='--',
                   label=f'爆炸阈值 ({self.warn_threshold_high})')
        ax.axhline(y=self.warn_threshold_low, color='orange', linestyle='--',
                   label=f'消失阈值 ({self.warn_threshold_low})')
        ax.set_yscale('log')
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Norm (L2)', fontsize=12, fontweight='bold')
        ax.set_title('梯度范数历史', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 子图2: 梯度均值
        ax = axes[0, 1]
        ax.plot(steps, self.history['grad_mean'], linewidth=2, color='green')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Mean', fontsize=12, fontweight='bold')
        ax.set_title('梯度均值历史', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 子图3: 梯度标准差
        ax = axes[1, 0]
        ax.plot(steps, self.history['grad_std'], linewidth=2, color='purple')
        ax.set_yscale('log')
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Std', fontsize=12, fontweight='bold')
        ax.set_title('梯度标准差历史', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 子图4: 梯度范围
        ax = axes[1, 1]
        ax.plot(steps, self.history['grad_max'], linewidth=2, color='red', label='Max')
        ax.plot(steps, self.history['grad_min'], linewidth=2, color='blue', label='Min')
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Value', fontsize=12, fontweight='bold')
        ax.set_title('梯度最大/最小值历史', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"梯度历史图已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def reset(self):
        """重置监控历史"""
        self.history = {
            'step': [],
            'grad_norm': [],
            'grad_mean': [],
            'grad_std': [],
            'grad_max': [],
            'grad_min': []
        }
        self.step_count = 0


def clip_gradients(model: torch.nn.Module,
                   max_norm: float = 1.0,
                   norm_type: float = 2.0) -> float:
    """
    梯度裁剪（Gradient Clipping）

    用于防止梯度爆炸，将梯度范数裁剪到指定上限。

    Args:
        model: PyTorch模型
        max_norm: 最大梯度范数（默认1.0）
        norm_type: 范数类型（默认2.0表示L2范数）

    Returns:
        裁剪前的总梯度范数
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """演示梯度监控的使用"""

    print("="*60)
    print("梯度监控工具演示")
    print("="*60)

    # 1. 创建一个简单模型
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )

    # 2. 创建梯度监控器
    monitor = GradientMonitor(
        log_interval=5,           # 每5步记录一次
        warn_threshold_high=10.0,  # 梯度范数>10认为过大
        warn_threshold_low=1e-5    # 梯度范数<1e-5认为过小
    )

    # 3. 模拟训练过程
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n模拟训练过程...")
    for step in range(50):
        # 前向传播
        x = torch.randn(8, 10)
        y = torch.randn(8, 1)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 监控梯度（每5步打印一次）
        stats, status = monitor.check_gradients(model, step=step, verbose=True)

        # 如果梯度爆炸，进行裁剪
        if status == 'exploding':
            total_norm = clip_gradients(model, max_norm=1.0)
            print(f"  Gradient clipped: {total_norm:.2e} -> 1.0")

        # 更新参数
        optimizer.step()

    # 4. 打印总结报告
    print(monitor.get_gradient_summary())

    # 5. 绘制梯度历史（注释掉以避免弹窗）
    # monitor.plot_gradient_history(save_path='gradient_history.png')

    print("\n演示完成！")
