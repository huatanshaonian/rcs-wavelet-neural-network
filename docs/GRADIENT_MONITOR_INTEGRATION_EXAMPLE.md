# 梯度监控集成示例

## 📋 概述

本文档展示如何在现有训练循环中集成梯度监控，并在GUI中添加梯度可视化功能。

---

## 1️⃣ 在训练循环中集成梯度监控

### 方法1: 在训练开始时创建监控器

```python
# ===== 在训练函数开始处 =====
from autoencoder.utils.gradient_monitor import GradientMonitor

# 创建梯度监控器
gradient_monitor = GradientMonitor(
    log_interval=10,           # 每10个epoch记录一次
    warn_threshold_high=10.0,  # 梯度>10警告
    warn_threshold_low=1e-5    # 梯度<1e-5警告
)

# 初始化梯度历史存储
gradient_history = {
    'grad_norm': [],    # 梯度范数
    'grad_mean': [],    # 梯度均值
    'grad_std': [],     # 梯度标准差
    'epochs': []        # 对应的epoch
}
```

### 方法2: 在训练循环中监控梯度

```python
for epoch in range(epochs):
    # ===== 训练阶段 =====
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # ⭐⭐⭐ 梯度监控（在optimizer.step()之前）⭐⭐⭐
        if epoch % gradient_monitor.log_interval == 0 and batch_idx == 0:
            # 只在每个监控epoch的第一个batch监控（避免过多输出）
            stats, status = gradient_monitor.check_gradients(
                model,
                step=epoch,
                verbose=False  # 不在控制台打印，我们会自己格式化输出
            )

            # 保存梯度统计到历史记录
            gradient_history['grad_norm'].append(stats['grad_norm'])
            gradient_history['grad_mean'].append(stats['grad_mean'])
            gradient_history['grad_std'].append(stats['grad_std'])
            gradient_history['epochs'].append(epoch)

            # 如果梯度异常，进行裁剪（可选）
            if status == 'exploding':
                from autoencoder.utils.gradient_monitor import clip_gradients
                clip_gradients(model, max_norm=1.0)

        # 更新参数
        optimizer.step()

        train_loss += loss.item()

    # ===== 验证阶段 =====
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    # ⭐⭐⭐ 打印训练进度（包含梯度信息）⭐⭐⭐
    if epoch % 10 == 0:
        # 获取最新的梯度统计
        if epoch in gradient_history['epochs']:
            idx = gradient_history['epochs'].index(epoch)
            grad_norm = gradient_history['grad_norm'][idx]
            grad_info = f", Grad={grad_norm:.2e}"
        else:
            grad_info = ""

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}"
              f"{grad_info}")

# ===== 训练结束后保存梯度历史 =====
return {
    'train_loss': train_loss_history,
    'val_loss': val_loss_history,
    'gradient_history': gradient_history  # ⭐ 添加梯度历史
}
```

### 输出示例

```
Epoch 1/100: Train=0.024567, Val=0.018234, Grad=2.34e-02
Epoch 10/100: Train=0.012345, Val=0.009876, Grad=1.56e-02
Epoch 20/100: Train=0.008765, Val=0.007654, Grad=8.92e-03
Epoch 30/100: Train=0.005432, Val=0.005123, Grad=4.21e-03
...
```

---

## 2️⃣ 在训练历史中保存梯度数据

### 修改训练历史结构

```python
# 原来的训练历史
training_history = {
    'stage1_train_loss': [...],
    'stage1_val_loss': [...],
    'stage2_train_loss': [...],
    'stage2_val_loss': [...],
    'stage3_train_loss': [...],
    'stage3_val_loss': [...]
}

# ⭐ 添加梯度历史
training_history = {
    # 原有的loss历史
    'stage1_train_loss': [...],
    'stage1_val_loss': [...],

    # ⭐ 新增：梯度历史
    'stage1_gradient_history': {
        'grad_norm': [...],
        'grad_mean': [...],
        'grad_std': [...],
        'epochs': [...]
    },

    # Stage 2和Stage 3同理
    'stage2_gradient_history': {...},
    'stage3_gradient_history': {...}
}
```

---

## 3️⃣ 在GUI中添加梯度可视化

### 步骤1: 在AutoEncoder页面添加按钮

```python
# 在 gui_autoencoder_extension.py 中添加
def create_autoencoder_tab(self):
    # ... 现有代码 ...

    # ⭐ 添加梯度监控可视化按钮
    gradient_frame = ttk.LabelFrame(visualization_frame, text="梯度监控", padding=10)
    gradient_frame.pack(fill='x', padx=5, pady=5)

    ttk.Button(
        gradient_frame,
        text="🔍 查看梯度历史",
        command=self._plot_gradient_history,
        width=20
    ).pack(side='left', padx=5)

    ttk.Button(
        gradient_frame,
        text="📊 梯度分析报告",
        command=self._show_gradient_report,
        width=20
    ).pack(side='left', padx=5)
```

### 步骤2: 实现梯度可视化函数

```python
def _plot_gradient_history(self):
    """绘制梯度历史曲线"""
    try:
        # 检查是否有训练历史
        if not hasattr(self, 'ae_training_history') or self.ae_training_history is None:
            messagebox.showwarning("提示", "没有训练历史数据\n请先训练模型")
            return

        # 检查是否有梯度历史
        stage_histories = self.ae_training_history.get('stage_histories', {})
        if not any('gradient_history' in stage for stage in stage_histories.values()):
            messagebox.showinfo(
                "提示",
                "训练历史中没有梯度监控数据\n\n"
                "这可能是因为：\n"
                "1. 模型是用旧版本训练的\n"
                "2. 训练时未启用梯度监控\n\n"
                "请重新训练模型以启用梯度监控"
            )
            return

        # 清除当前图表
        self.vis_fig.clear()

        # 创建子图 (3行2列，显示3个训练阶段)
        fig = self.vis_fig

        stage_names = ['stage1', 'stage2', 'stage3']
        stage_titles = [
            'Stage 1: AutoEncoder预训练',
            'Stage 2: 参数映射器训练',
            'Stage 3: 端到端微调'
        ]

        for idx, (stage_name, stage_title) in enumerate(zip(stage_names, stage_titles)):
            stage_data = stage_histories.get(stage_name, {})
            grad_hist = stage_data.get('gradient_history', None)

            if grad_hist and grad_hist.get('epochs'):
                # 绘制梯度范数
                ax1 = fig.add_subplot(3, 2, idx*2 + 1)
                ax1.plot(grad_hist['epochs'], grad_hist['grad_norm'],
                        linewidth=2, color='blue', marker='o', markersize=3)
                ax1.axhline(y=10.0, color='red', linestyle='--', alpha=0.5,
                           label='爆炸阈值')
                ax1.axhline(y=1e-5, color='orange', linestyle='--', alpha=0.5,
                           label='消失阈值')
                ax1.set_yscale('log')
                ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
                ax1.set_ylabel('梯度范数 (L2)', fontsize=10, fontweight='bold')
                ax1.set_title(f'{stage_title} - 梯度范数', fontsize=11, fontweight='bold')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                # 绘制梯度均值和标准差
                ax2 = fig.add_subplot(3, 2, idx*2 + 2)
                ax2.plot(grad_hist['epochs'], grad_hist['grad_mean'],
                        linewidth=2, color='green', marker='s', markersize=3,
                        label='梯度均值')
                ax2.plot(grad_hist['epochs'], grad_hist['grad_std'],
                        linewidth=2, color='purple', marker='^', markersize=3,
                        label='梯度标准差')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
                ax2.set_ylabel('梯度值', fontsize=10, fontweight='bold')
                ax2.set_title(f'{stage_title} - 梯度分布', fontsize=11, fontweight='bold')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self.vis_canvas.draw()

        self.log_message("✅ 梯度历史可视化完成！")

    except Exception as e:
        error_msg = f"绘制梯度历史失败: {str(e)}"
        self.log_message(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("错误", error_msg)


def _show_gradient_report(self):
    """显示梯度分析报告"""
    try:
        if not hasattr(self, 'ae_training_history') or self.ae_training_history is None:
            messagebox.showwarning("提示", "没有训练历史数据")
            return

        # 生成报告
        report = "="*60 + "\n"
        report += "梯度监控分析报告\n"
        report += "="*60 + "\n\n"

        stage_histories = self.ae_training_history.get('stage_histories', {})

        for stage_name, stage_title in [
            ('stage1', 'Stage 1: AutoEncoder预训练'),
            ('stage2', 'Stage 2: 参数映射器训练'),
            ('stage3', 'Stage 3: 端到端微调')
        ]:
            stage_data = stage_histories.get(stage_name, {})
            grad_hist = stage_data.get('gradient_history', None)

            if grad_hist and grad_hist.get('grad_norm'):
                import numpy as np
                grad_norms = np.array(grad_hist['grad_norm'])

                report += f"{stage_title}\n"
                report += "-"*60 + "\n"
                report += f"  记录点数: {len(grad_norms)}\n"
                report += f"  梯度范数统计:\n"
                report += f"    均值: {np.mean(grad_norms):.2e}\n"
                report += f"    中位数: {np.median(grad_norms):.2e}\n"
                report += f"    最大值: {np.max(grad_norms):.2e}\n"
                report += f"    最小值: {np.min(grad_norms):.2e}\n"

                # 健康度评估
                exploding_count = np.sum(grad_norms > 10.0)
                vanishing_count = np.sum(grad_norms < 1e-5)
                healthy_count = len(grad_norms) - exploding_count - vanishing_count

                report += f"  健康度评估:\n"
                report += f"    健康比例: {healthy_count/len(grad_norms)*100:.1f}%\n"
                report += f"    梯度爆炸次数: {exploding_count}\n"
                report += f"    梯度消失次数: {vanishing_count}\n"
                report += "\n"

        report += "="*60 + "\n"

        # 在新窗口显示报告
        report_window = tk.Toplevel(self.root)
        report_window.title("梯度分析报告")
        report_window.geometry("600x500")

        text_widget = tk.Text(report_window, wrap='word', font=('Consolas', 10))
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget.insert('1.0', report)
        text_widget.config(state='disabled')  # 只读

        # 添加滚动条
        scrollbar = tk.Scrollbar(text_widget)
        scrollbar.pack(side='right', fill='y')
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

    except Exception as e:
        messagebox.showerror("错误", f"生成报告失败: {str(e)}")
```

---

## 4️⃣ 完整的集成清单

### ✅ 训练循环修改

- [ ] 在训练开始时创建 `GradientMonitor`
- [ ] 在每个epoch的第一个batch监控梯度
- [ ] 将梯度统计保存到历史记录
- [ ] 在打印loss时同时打印梯度信息
- [ ] 训练结束时返回包含梯度历史的字典

### ✅ 训练历史修改

- [ ] 在 `training_history` 中添加 `stage1_gradient_history`
- [ ] 在 `training_history` 中添加 `stage2_gradient_history`
- [ ] 在 `training_history` 中添加 `stage3_gradient_history`

### ✅ GUI可视化修改

- [ ] 在AutoEncoder页面添加"查看梯度历史"按钮
- [ ] 实现 `_plot_gradient_history()` 函数
- [ ] 实现 `_show_gradient_report()` 函数
- [ ] 测试新旧训练历史的兼容性

---

## 5️⃣ 实际效果预览

### 训练输出示例

```
Stage 1: AutoEncoder预训练
========================================
  Epoch    1/50: Train=0.024567, Val=0.018234, Grad=2.34e-02  ✅
  Epoch   10/50: Train=0.012345, Val=0.009876, Grad=1.56e-02  ✅
  Epoch   20/50: Train=0.008765, Val=0.007654, Grad=8.92e-03  ✅
  Epoch   30/50: Train=0.005432, Val=0.005123, Grad=4.21e-03  ✅
  Epoch   40/50: Train=0.003876, Val=0.003654, Grad=2.45e-03  ✅
  Epoch   50/50: Train=0.002981, Val=0.002843, Grad=1.89e-03  ✅
✅ Stage 1完成
```

### 梯度历史图表

```
┌─────────────────────────────────┬─────────────────────────────────┐
│ Stage 1 - 梯度范数               │ Stage 1 - 梯度分布               │
│ [梯度范数随epoch变化的曲线图]     │ [梯度均值和标准差的曲线图]        │
├─────────────────────────────────┼─────────────────────────────────┤
│ Stage 2 - 梯度范数               │ Stage 2 - 梯度分布               │
│ [梯度范数随epoch变化的曲线图]     │ [梯度均值和标准差的曲线图]        │
├─────────────────────────────────┼─────────────────────────────────┤
│ Stage 3 - 梯度范数               │ Stage 3 - 梯度分布               │
│ [梯度范数随epoch变化的曲线图]     │ [梯度均值和标准差的曲线图]        │
└─────────────────────────────────┴─────────────────────────────────┘
```

### 梯度分析报告

```
============================================================
梯度监控分析报告
============================================================

Stage 1: AutoEncoder预训练
------------------------------------------------------------
  记录点数: 5
  梯度范数统计:
    均值: 1.21e-02
    中位数: 8.92e-03
    最大值: 2.34e-02
    最小值: 1.89e-03
  健康度评估:
    健康比例: 100.0%
    梯度爆炸次数: 0
    梯度消失次数: 0

Stage 2: 参数映射器训练
------------------------------------------------------------
  记录点数: 3
  梯度范数统计:
    均值: 5.67e-03
    中位数: 5.45e-03
    最大值: 6.78e-03
    最小值: 4.89e-03
  健康度评估:
    健康比例: 100.0%
    梯度爆炸次数: 0
    梯度消失次数: 0

Stage 3: 端到端微调
------------------------------------------------------------
  记录点数: 2
  梯度范数统计:
    均值: 3.12e-03
    中位数: 3.12e-03
    最大值: 3.45e-03
    最小值: 2.78e-03
  健康度评估:
    健康比例: 100.0%
    梯度爆炸次数: 0
    梯度消失次数: 0

============================================================
```

---

## 6️⃣ 注意事项

### ⚠️ 性能影响

- 梯度监控开销：< 5%训练时间
- 建议 `log_interval=10`，每10个epoch监控一次
- 不要每个batch都监控，会显著降低训练速度

### ⚠️ 兼容性

- 新训练的模型：包含梯度历史
- 旧训练的模型：没有梯度历史，GUI应友好提示
- 需要检查 `gradient_history` 是否存在

### ⚠️ 存储空间

- 梯度历史数据量很小（每个epoch几个浮点数）
- 不会显著增加模型文件大小

---

## 7️⃣ 下一步

1. ✅ 完成训练循环集成
2. ✅ 修改训练历史结构
3. ✅ 添加GUI可视化按钮
4. ✅ 测试新旧模型兼容性
5. ✅ 更新用户文档

---

**参考文件**：
- `autoencoder/utils/gradient_monitor.py` - 梯度监控工具
- `docs/GRADIENT_MONITORING_GUIDE.md` - 使用指南
- `gui_managers/managers/training_manager.py` - 训练管理器
- `gui_autoencoder_extension.py` - GUI扩展
