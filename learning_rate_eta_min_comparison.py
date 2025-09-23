"""
学习率调度器中eta_min的影响分析
展示不同最低学习率对训练的影响
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cosine_annealing(initial_lr, eta_min, T_0, epochs):
    """计算余弦退火学习率"""
    lr_history = []
    for epoch in range(epochs):
        t = epoch % T_0
        lr = eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * t / T_0)) / 2
        lr_history.append(lr)
    return lr_history

# 参数设置
initial_lr = 0.003
T_0 = 50
epochs = 200

# 不同的eta_min值
eta_mins = {
    '过低 (1e-6)': 1e-6,    # 旧值，太低
    '推荐 (2e-5)': 2e-5,    # 新默认值
    '保守 (1e-5)': 1e-5,    # 更低，更精细
    '激进 (5e-5)': 5e-5     # 更高，更快
}

# 计算各个eta_min的学习率历史
lr_histories = {}
for name, eta_min in eta_mins.items():
    lr_histories[name] = cosine_annealing(initial_lr, eta_min, T_0, epochs)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 完整对比
ax1 = axes[0, 0]
colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12']
for (name, lr_hist), color in zip(lr_histories.items(), colors):
    ax1.plot(lr_hist, label=name, linewidth=2, alpha=0.8, color=color)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Learning Rate', fontsize=12)
ax1.set_title('学习率调度对比 (初始LR=0.003)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=initial_lr, color='gray', linestyle='--', alpha=0.5)

# 子图2: 对数刻度
ax2 = axes[0, 1]
for (name, lr_hist), color in zip(lr_histories.items(), colors):
    ax2.semilogy(lr_hist, label=name, linewidth=2, alpha=0.8, color=color)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Learning Rate (log scale)', fontsize=12)
ax2.set_title('学习率调度对比 (对数刻度)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)

# 子图3: 单周期详细对比
ax3 = axes[1, 0]
for (name, lr_hist), color in zip(lr_histories.items(), colors):
    ax3.plot(lr_hist[:50], label=name, linewidth=2.5, alpha=0.8, color=color)
ax3.set_xlabel('Epoch (First Cycle)', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('第一个周期详细对比 (0-50 epoch)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
# 标注精细学习阶段
ax3.axvspan(40, 50, alpha=0.2, color='green', label='精细学习阶段')

# 子图4: 统计信息
ax4 = axes[1, 1]
ax4.axis('off')

# 计算统计信息
stats_text = "学习率调度器参数影响分析\n\n"
stats_text += "="*50 + "\n"
stats_text += f"初始学习率: {initial_lr}\n"
stats_text += f"周期长度 (T_0): {T_0} epochs\n"
stats_text += "="*50 + "\n\n"

for name, eta_min in eta_mins.items():
    lr_hist = lr_histories[name]
    avg_lr = np.mean(lr_hist)
    min_lr = np.min(lr_hist)
    max_lr = np.max(lr_hist)

    # 精细学习阶段(每周期最后10 epochs)的平均学习率
    fine_tune_epochs = []
    for i in range(0, epochs, T_0):
        fine_tune_epochs.extend(lr_hist[i+40:i+50])
    avg_fine_tune_lr = np.mean(fine_tune_epochs) if fine_tune_epochs else 0

    stats_text += f"{name}:\n"
    stats_text += f"  eta_min: {eta_min:.2e}\n"
    stats_text += f"  平均LR: {avg_lr:.2e}\n"
    stats_text += f"  精细学习阶段平均LR: {avg_fine_tune_lr:.2e}\n"
    stats_text += f"  动态范围: {max_lr/min_lr:.1f}x\n\n"

stats_text += "="*50 + "\n"
stats_text += "推荐设置:\n"
stats_text += "  • 对数域训练: eta_min = 2e-5\n"
stats_text += "  • 线性域训练: eta_min = 1e-5\n"
stats_text += "  • 快速探索: eta_min = 5e-5\n\n"
stats_text += "关键要点:\n"
stats_text += "  ✓ eta_min太低(1e-6)导致精细学习阶段\n"
stats_text += "    几乎不学习\n"
stats_text += "  ✓ eta_min合理(2e-5)保持有效的精细\n"
stats_text += "    调整能力\n"
stats_text += "  ✓ 初始LR和eta_min是独立参数\n"

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('learning_rate_eta_min_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] 学习率eta_min对比图已保存: learning_rate_eta_min_comparison.png")

# 打印详细分析
print("\n" + "="*60)
print("学习率调度器 eta_min 参数分析")
print("="*60)

for name, eta_min in eta_mins.items():
    lr_hist = lr_histories[name]

    # 精细学习阶段
    fine_tune_epochs = []
    for i in range(0, epochs, T_0):
        fine_tune_epochs.extend(lr_hist[i+40:i+50])

    print(f"\n{name} (eta_min={eta_min:.2e}):")
    print(f"  周期最低LR: {min(lr_hist):.2e}")
    print(f"  精细学习阶段平均LR: {np.mean(fine_tune_epochs):.2e}")
    print(f"  有效学习能力: {'低' if eta_min < 1e-5 else '中' if eta_min < 5e-5 else '高'}")

print("\n" + "="*60)
print("结论:")
print("  1. eta_min=1e-6 (旧值) 太低，精细学习阶段步长过小")
print("  2. eta_min=2e-5 (推荐) 平衡，既能精细学习又不过慢")
print("  3. eta_min是独立参数，不受初始LR影响")
print("  4. 对数域训练需要稍大的eta_min (2e-5)")
print("="*60 + "\n")