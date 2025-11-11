# 数据预处理方法与学习率敏感度分析

## 1. 数据范围对比

### 原始RCS数据（线性域）
```
范围: 1e-6 ~ 1e-1 (跨越5-6个数量级)
特点: 动态范围极大，数值分布不均匀
问题: 小值被大值主导，梯度不稳定
```

### dB变换
```python
RCS_dB = 10 * log10(RCS_linear)
范围: -60 ~ -10 dBsm
特点: 压缩动态范围，对数尺度
优势: 使不同量级的数据差异更均匀
```

### Z-score标准化
```python
X_normalized = (X - mean) / std
范围: 约 -3 ~ +3 (99.7%数据在±3σ内)
特点: 均值为0，标准差为1
优势: 数据以0为中心，梯度稳定
```

### MinMax标准化
```python
X_normalized = (X - min) / (max - min)
范围: [0, 1]
特点: 线性映射到固定区间
注意: 数据偏向正值，可能导致梯度偏差
```

### 不标准化
```
范围: 保持原始数据范围
问题: 不同特征/通道的数值范围差异大
```

---

## 2. 学习率敏感度分析

### 理论基础

**损失函数**（MSE）：
```
L = (y_pred - y_true)²
```

**梯度大小**：
```
∂L/∂w ∝ (y_pred - y_true) × ∂y_pred/∂w
```

**关键观察**：
- 数据范围大 → 预测误差(y_pred - y_true)大 → 梯度大
- 数据范围小 → 预测误差小 → 梯度小
- 参数更新量 = 学习率 × 梯度

---

## 3. 不同预处理方法的梯度行为

### 场景1: 原始RCS数据（线性域）
```
数据范围: 1e-6 ~ 1e-1
典型误差: 1e-3 ~ 1e-2
梯度大小: 非常大且不稳定
推荐学习率: 1e-5 ~ 1e-6（很小）

问题:
- 梯度爆炸风险高
- 对学习率极度敏感
- 训练不稳定
```

### 场景2: dB变换 + 不标准化
```
数据范围: -60 ~ -10 dBsm
典型误差: 1 ~ 5 dB
梯度大小: 中等
推荐学习率: 1e-4 ~ 1e-3

优势:
- 梯度相对稳定
- 对学习率中度敏感
```

### 场景3: Z-score标准化（±dB变换）
```
数据范围: -3 ~ +3
典型误差: 0.1 ~ 0.5
梯度大小: 适中且稳定
推荐学习率: 1e-3 ~ 1e-2

优势:
- 梯度最稳定
- 对学习率较不敏感
- 训练收敛快
- 不同特征/通道梯度均衡
```

### 场景4: MinMax [0,1]标准化
```
数据范围: [0, 1]
典型误差: 0.01 ~ 0.1
梯度大小: 较小
推荐学习率: 1e-3 ~ 1e-2

注意:
- 数据偏向正值
- 可能导致梯度偏差
- 对学习率中度敏感
```

---

## 4. 学习率敏感度排序（从高到低）

```
原始数据（线性域） >>> dB变换（无标准化） > MinMax标准化 > Z-score标准化
```

**解释**：
1. **原始数据**：动态范围跨越6个数量级，梯度极不稳定，学习率必须非常小
2. **dB变换**：压缩动态范围，但仍有较大数值，中度敏感
3. **MinMax**：固定在[0,1]，但可能有偏差
4. **Z-score**：数据中心化，梯度最稳定，最不敏感

---

## 5. 最佳实践建议

### 推荐组合（按优先级）

#### ⭐ 组合1: dB + Z-score（强烈推荐）
```python
config = {
    'db_transform': True,        # 先dB变换
    'normalization_method': 'zscore'  # 再Z-score标准化
}
learning_rate = 1e-3  # 标准学习率
```

**优势**：
- dB变换：压缩动态范围，使不同量级数据更均匀
- Z-score：中心化数据，梯度稳定
- 最适合Adam优化器
- 训练最稳定、收敛最快

---

#### ⭐ 组合2: Z-score（次推荐）
```python
config = {
    'db_transform': False,
    'normalization_method': 'zscore'
}
learning_rate = 1e-3
```

**适用场景**：
- 不想丢失线性域信息
- 希望保留RCS的物理意义
- 仍然训练稳定

---

#### ⚠️ 组合3: dB + MinMax
```python
config = {
    'db_transform': True,
    'normalization_method': 'minmax'
}
learning_rate = 1e-3 ~ 5e-4
```

**注意**：
- 可能需要调整学习率
- 训练稳定性略低于Z-score

---

#### ❌ 不推荐: 原始数据（无标准化）
```python
config = {
    'db_transform': False,
    'normalization_method': None
}
learning_rate = 1e-5 ~ 1e-6  # 必须很小
```

**问题**：
- 训练极不稳定
- 对学习率极度敏感
- 收敛速度慢
- 容易梯度爆炸或消失

---

## 6. 优化器选择建议

### Adam/AdamW（推荐）
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,  # 标准学习率（Z-score标准化）
    betas=(0.9, 0.999),
    weight_decay=1e-4
)
```

**优势**：
- 自适应调整每个参数的学习率
- 对不同数据范围有一定鲁棒性
- 但仍受初始学习率影响

### SGD with Momentum（不推荐用于原始数据）
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,  # 需要Z-score标准化
    momentum=0.9
)
```

**注意**：
- 对数据范围更敏感
- 需要仔细调整学习率

---

## 7. 学习率调度策略

### ReduceLROnPlateau（推荐）
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # 每次减半
    patience=10,      # 10个epoch无改善则降低
    min_lr=1e-6
)
```

**优势**：
- 自动根据训练进度调整
- 减轻学习率选择压力

### Cosine Annealing
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # 总epoch数
    eta_min=1e-6
)
```

---

## 8. 实验建议：学习率敏感度测试

### 批量实验配置
```python
# 测试不同预处理方法 + 不同学习率的组合
compare_dimensions = {
    'preprocessing': [
        {'normalization_method': 'zscore', 'db_transform': True},    # 推荐
        {'normalization_method': 'zscore', 'db_transform': False},   # 次推荐
        {'normalization_method': 'minmax', 'db_transform': True},    # 备选
        {'normalization_method': None, 'db_transform': False}        # 基线
    ],
    'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5]
}
```

### 评估指标
- 训练稳定性：Loss曲线是否平滑
- 收敛速度：达到目标Loss所需epoch数
- 最终性能：最低Test Loss
- 梯度统计：梯度范数的均值和方差

---

## 9. 常见问题排查

### 问题1: 训练Loss震荡剧烈
```
原因: 学习率过大
解决方案:
- 降低学习率（减半尝试）
- 使用dB + Z-score标准化
- 增加batch size
```

### 问题2: Loss下降极慢
```
原因: 学习率过小 或 数据未标准化
解决方案:
- 提高学习率（翻倍尝试）
- 确保使用标准化
- 使用学习率warmup
```

### 问题3: 梯度爆炸（Loss变为NaN）
```
原因: 学习率过大 + 原始数据
解决方案:
- 立即使用Z-score标准化
- 大幅降低学习率
- 梯度裁剪: torch.nn.utils.clip_grad_norm_()
```

### 问题4: 不同频率收敛速度差异大
```
原因: 不同频率RCS范围差异大
解决方案:
- 使用per-frequency标准化（当前实现）
- 确保data_adapter对每个频率独立标准化
```

---

## 10. 总结和建议

### 核心结论
1. **标准化是必须的**：不标准化会导致训练极不稳定
2. **Z-score最稳定**：对学习率最不敏感，训练最可靠
3. **dB+Z-score最优**：结合对数尺度和标准化的优势
4. **学习率建议**：
   - Z-score标准化：1e-3（标准）
   - 原始数据：1e-5 ~ 1e-6（不推荐）

### 实践流程
```
1. 始终使用标准化（优先Z-score）
2. 从标准学习率1e-3开始
3. 使用ReduceLROnPlateau自动调整
4. 监控梯度范数，确保在合理范围（1e-3 ~ 1e-1）
5. 如有问题，先检查标准化，再调整学习率
```

### 批量实验建议
使用批量实验功能系统性测试：
```python
# 对比维度：预处理方法
compare_dimensions = {
    'preprocessing': [
        {'normalization_method': 'zscore', 'db_transform': True},
        {'normalization_method': 'zscore', 'db_transform': False},
        {'normalization_method': 'minmax', 'db_transform': True}
    ]
}
# 固定参数
base_config = {
    'learning_rate': 1e-3,  # 标准学习率
    'optimizer': 'adamw',
    'batch_size': 8
}
```

通过批量实验可以直观看到：
- 不同预处理方法的训练曲线
- 收敛速度差异
- 最终性能对比
- 训练稳定性差异

---

**参考文献**：
- Ioffe & Szegedy (2015). Batch Normalization
- He et al. (2015). Delving Deep into Rectifiers
- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization
