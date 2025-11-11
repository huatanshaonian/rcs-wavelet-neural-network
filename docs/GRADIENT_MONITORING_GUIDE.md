# 梯度监控使用指南

## 🎯 什么是梯度监控？

**梯度（Gradient）** = 神经网络训练时，反向传播计算出的"参数更新方向和大小"

**监控梯度** = 在训练过程中实时观察梯度统计信息，判断训练是否健康

---

## 📊 为什么要监控梯度？

### 问题1: 梯度爆炸（Gradient Exploding）
```
症状: Loss突然变成NaN，训练崩溃
原因: 梯度范数 > 10，参数更新过大
表现: grad_norm = 1523.45  # 远超正常范围
```

### 问题2: 梯度消失（Gradient Vanishing）
```
症状: Loss长时间不下降，训练停滞
原因: 梯度范数 < 1e-5，参数几乎不更新
表现: grad_norm = 3.2e-07  # 太小了
```

### 问题3: 梯度不稳定
```
症状: Loss曲线剧烈震荡
原因: 梯度波动太大
表现: grad_norm在 [0.01, 100] 之间剧烈波动
```

---

## ✅ 健康梯度的标准

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| **梯度范数** | **1e-3 ~ 1e-1** | 最重要！决定训练稳定性 |
| 梯度均值 | 接近0 | 理想情况应该围绕0波动 |
| 梯度标准差 | 适中 | 太大不稳定，太小可能卡住 |

### 直观对比

```python
# ✅ 健康梯度
grad_norm = 0.0234  # 在 1e-3 ~ 1e-1 范围内
grad_mean = -0.0012  # 接近0
→ 训练稳定，Loss平滑下降

# ❌ 梯度爆炸
grad_norm = 145.67  # >> 10
grad_mean = 23.45
→ Loss变NaN，训练崩溃

# ❌ 梯度消失
grad_norm = 3.2e-7  # << 1e-5
grad_mean = -1.2e-8
→ Loss不变，训练停滞
```

---

## 💻 如何使用梯度监控工具？

### 方法1: 在训练循环中使用（推荐）

```python
from autoencoder.utils.gradient_monitor import GradientMonitor

# 1. 创建监控器
monitor = GradientMonitor(
    log_interval=10,           # 每10步记录一次
    warn_threshold_high=10.0,  # 梯度>10警告
    warn_threshold_low=1e-5    # 梯度<1e-5警告
)

# 2. 训练循环
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # ⭐ 监控梯度（在optimizer.step()之前）
        stats, status = monitor.check_gradients(
            model,
            step=epoch * len(train_loader) + batch_idx,
            verbose=True  # 打印信息
        )

        # 如果梯度爆炸，进行裁剪
        if status == 'exploding':
            from autoencoder.utils.gradient_monitor import clip_gradients
            clip_gradients(model, max_norm=1.0)
            print("⚠️ 检测到梯度爆炸，已进行裁剪")

        # 更新参数
        optimizer.step()

# 3. 训练结束，查看总结
print(monitor.get_gradient_summary())

# 4. 绘制梯度历史曲线
monitor.plot_gradient_history(save_path='gradient_history.png')
```

### 方法2: 手动检查单次梯度

```python
from autoencoder.utils.gradient_monitor import GradientMonitor

monitor = GradientMonitor()

# 训练一步后
loss.backward()

# 检查梯度
stats, status = monitor.check_gradients(model, verbose=True)

print(f"梯度范数: {stats['grad_norm']:.2e}")
print(f"梯度状态: {status}")  # 'healthy', 'warning', 'exploding', 'vanishing'
```

### 方法3: 仅计算梯度统计（不记录历史）

```python
from autoencoder.utils.gradient_monitor import GradientMonitor

monitor = GradientMonitor()

# 计算当前梯度统计
stats = monitor.compute_gradient_stats(model)

print(f"梯度范数: {stats['grad_norm']:.2e}")
print(f"梯度均值: {stats['grad_mean']:.2e}")
print(f"梯度标准差: {stats['grad_std']:.2e}")
print(f"梯度范围: [{stats['grad_min']:.2e}, {stats['grad_max']:.2e}]")
```

---

## 🔧 根据梯度监控结果调整训练

### 场景1: 梯度爆炸（grad_norm > 10）

**症状**：
```
[Step 50] 梯度监控:
  ⚠️ 梯度爆炸警告！梯度范数=145.67 > 10.0
  梯度统计: mean=23.45, std=56.78
```

**解决方案**（按优先级）：
1. **立即降低学习率**
   ```python
   # 将学习率减半
   for param_group in optimizer.param_groups:
       param_group['lr'] *= 0.5
   ```

2. **使用梯度裁剪**
   ```python
   from autoencoder.utils.gradient_monitor import clip_gradients

   # 在optimizer.step()之前
   clip_gradients(model, max_norm=1.0)
   ```

3. **检查数据标准化**
   ```python
   # 确保使用了标准化
   config = {
       'normalization_method': 'zscore',
       'db_transform': True
   }
   ```

---

### 场景2: 梯度消失（grad_norm < 1e-5）

**症状**：
```
[Step 200] 梯度监控:
  ⚠️ 梯度消失警告！梯度范数=3.2e-07 < 1e-5
  Loss已经100个epoch没有变化
```

**解决方案**：
1. **提高学习率**
   ```python
   for param_group in optimizer.param_groups:
       param_group['lr'] *= 2.0
   ```

2. **检查标准化设置**
   ```python
   # 确保使用了标准化
   # 如果用了MinMax，尝试Z-score
   ```

3. **检查激活函数**
   ```python
   # 如果用了Sigmoid/Tanh，考虑换成ReLU/GELU
   ```

4. **减小网络深度**（如果网络很深）

---

### 场景3: 梯度不稳定（波动大）

**症状**：
```
Step 100: grad_norm = 0.5
Step 110: grad_norm = 5.2
Step 120: grad_norm = 0.08
Step 130: grad_norm = 12.3
→ 波动过大！
```

**解决方案**：
1. **降低学习率**
2. **增加batch size**
3. **使用更稳定的优化器**（如AdamW）
4. **使用学习率调度器**
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer,
       mode='min',
       factor=0.5,
       patience=10
   )
   ```

---

## 📈 梯度监控可视化

运行 `monitor.plot_gradient_history()` 会生成4个子图：

### 1. 梯度范数历史（最重要）
```
理想情况: 在1e-3 ~ 1e-1之间平稳波动
问题情况:
- 曲线冲出红色虚线 → 梯度爆炸
- 曲线低于橙色虚线 → 梯度消失
- 曲线剧烈震荡 → 训练不稳定
```

### 2. 梯度均值历史
```
理想情况: 在0附近波动
问题情况: 持续偏离0 → 梯度有偏差
```

### 3. 梯度标准差历史
```
理想情况: 相对稳定
问题情况:
- 突然升高 → 训练不稳定
- 持续降低至接近0 → 可能收敛或卡住
```

### 4. 梯度范围（最大/最小值）
```
理想情况: 最大和最小值相对对称
问题情况: 不对称 → 某些参数的梯度异常
```

---

## 🎓 实战案例

### 案例1: 使用原始RCS数据训练

**初始设置**：
```python
config = {
    'normalization_method': None,  # ❌ 没有标准化
    'db_transform': False
}
learning_rate = 1e-3
```

**梯度监控输出**：
```
[Step 10] 梯度监控:
  ⚠️ 梯度爆炸警告！梯度范数=523.45 > 10.0
  梯度统计: mean=145.67, std=234.56

[Step 20] 梯度监控:
  Loss变成NaN，训练崩溃！
```

**修复方案**：
```python
# 1. 添加标准化
config = {
    'normalization_method': 'zscore',
    'db_transform': True
}

# 2. 降低学习率
learning_rate = 1e-5

# 重新训练
[Step 10] 梯度监控:
  ✅ 梯度正常：梯度范数=0.0234
  → 训练稳定！
```

---

### 案例2: 学习率过小

**初始设置**：
```python
config = {
    'normalization_method': 'zscore',
    'db_transform': True
}
learning_rate = 1e-6  # 过小
```

**梯度监控输出**：
```
[Step 100] 梯度监控:
  ✅ 梯度正常：梯度范数=0.0234
  但Loss已经50个epoch没有变化！

→ 梯度正常，但学习率太小导致收敛极慢
```

**修复方案**：
```python
# 提高学习率
learning_rate = 1e-3

# 重新训练
→ Loss快速下降，收敛速度提升50倍！
```

---

## 📋 梯度监控检查清单

训练前检查：
- [ ] 数据已标准化（推荐Z-score）
- [ ] 创建了梯度监控器
- [ ] 设置了合理的警告阈值

训练中监控：
- [ ] 每10-50步检查一次梯度
- [ ] 梯度范数在1e-3 ~ 1e-1之间
- [ ] 梯度均值接近0
- [ ] 没有梯度爆炸/消失警告

训练后分析：
- [ ] 查看梯度总结报告
- [ ] 绘制梯度历史曲线
- [ ] 健康比例 > 90%

---

## 🔍 常见问题

### Q1: 梯度监控会降低训练速度吗？
**A**: 影响很小（<5%）。监控只计算统计信息，不影响参数更新。
建议设置 `log_interval=10`，每10步记录一次。

### Q2: 什么时候应该使用梯度裁剪？
**A**: 当梯度范数持续 > 10 时。但梯度裁剪是"治标不治本"，
最好从源头解决（数据标准化、降低学习率）。

### Q3: 梯度范数的理想值是多少？
**A**: **0.01 ~ 0.1**（1e-2 ~ 1e-1）最理想。
- < 1e-3: 偏小，可能学习慢
- > 1: 偏大，可能不稳定
- > 10: 梯度爆炸

### Q4: 不同阶段的梯度范数会不同吗？
**A**: 是的。通常：
- 训练初期：梯度较大（0.1 ~ 1）
- 训练中期：梯度适中（0.01 ~ 0.1）
- 训练后期：梯度较小（0.001 ~ 0.01）→ 接近收敛

### Q5: 梯度监控对不同优化器有影响吗？
**A**: 监控本身无影响，但不同优化器对梯度的敏感度不同：
- **Adam/AdamW**: 较鲁棒，梯度范围可以宽一些
- **SGD**: 较敏感，需要更严格的梯度控制

---

## 📚 参考资料

- [Understanding the Exploding Gradient Problem](https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11)
- [Gradient Clipping 论文](https://arxiv.org/abs/1211.5063)
- [PyTorch 梯度裁剪文档](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)

---

**总结**：
梯度监控是训练调试的"体检工具"，能及早发现问题，避免浪费训练时间！
