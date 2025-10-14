# 数据处理流程完整文档

## 1. 数据源特性

### 1.1 RCS数据特性
- **原始格式**: 线性标度 (Linear Scale)
- **数值范围**: RCS数据 > 0 (总是正值)
- **典型范围**: 约 1e-6 到 1e2 (跨越8个数量级)
- **dB转换**: RCS(dB) = 10 * log10(RCS_linear)
- **形状**: [N, 91, 91, num_freq] - N个样本，91×91角度网格，num_freq个频率

### 1.2 参数数据特性
- **形状**: [N, 9] - 9个几何参数
- **范围**: 各参数有不同的物理范围

## 2. 数据处理流程对比

### 2.1 数据管理页面 (Data Management)

**用途**: 数据可视化和预览

**处理流程**:
```
原始RCS数据 (线性, >0)
    ↓
[可选] 对数化: 10 * log10(RCS_linear)  → RCS(dB)
    ↓
[可选] 标准化: (RCS_dB - μ) / σ → 标准化数据
    ↓
显示统计信息
```

**配置选项**:
- `use_log_preprocessing`: 是否转换为dB
- `log_epsilon_var`: epsilon值 (默认1e-10)
- `normalize_after_log`: dB后是否标准化

**重要**: 这些处理仅用于显示，**不影响训练数据**

### 2.2 AutoEncoder页面 (AE Training)

**用途**: 实际训练数据预处理

**❌ 当前流程 (错误)**:
```
原始RCS数据 (线性, >0)
    ↓
RCS_DataAdapter预处理:
  - [可选] 对数变换: sign(x) * log(|x|)
  - [可选] Z-score标准化: (x - μ) / σ
    ↓
预处理后的RCS
    ↓
[小波模式] 小波变换 (WT) ← ❌ 错误: 应该在原始数据上进行!
    ↓
输入神经网络
```

**✅ 正确流程**:
```
原始RCS数据 (线性, >0)
    ↓
分支判断:

【小波模式 (Wavelet)】:
  原始RCS → 小波变换 → 小波系数 → RCS_DataAdapter预处理 → 输入网络

【直接模式 (Direct)】:
  原始RCS → RCS_DataAdapter预处理 → 输入网络
```

### 2.3 关键差异

| 方面 | 数据管理 | AutoEncoder训练 |
|------|----------|----------------|
| **对数方法** | 10*log10 (dB) | sign(x)*log(\|x\|) |
| **目的** | 可视化 | 训练稳定性 |
| **影响训练** | 否 | 是 |
| **小波变换** | 不涉及 | 必须在原始数据上 |

## 3. 为什么小波变换必须在原始数据上？

### 3.1 小波变换的物理意义
小波变换分解信号的多尺度特征，需要保持原始信号的**相对振幅关系**:

```python
# 原始RCS (线性):
RCS_1.5GHz = [0.001, 0.01, 0.1, 1.0]  # 4个数量级差异
RCS_3GHz   = [0.002, 0.02, 0.2, 2.0]

# 标准化后:
Normalized = [-1.2, -0.4, 0.4, 1.2]   # ❌ 原始相对关系被破坏!
```

### 3.2 标准化破坏小波基的正交性
- 小波变换依赖于信号的**局部相关性**
- Z-score标准化改变了信号的统计分布
- 导致小波系数失去物理意义

### 3.3 正确做法
```python
# ✅ 正确: 小波系数的标准化
wavelet_coeffs = WT(RCS_original)  # 保持原始相对关系
normalized_coeffs = (coeffs - μ_coeffs) / σ_coeffs  # 标准化系数
```

## 4. Z-score用于正值数据是否合适？

### 4.1 理论分析
**Z-score假设**: 数据接近正态分布 (均值附近对称)

**RCS线性数据特性**:
- 严格正值: RCS > 0
- 对数正态分布 (Log-Normal Distribution)
- 在线性标度下高度偏斜

### 4.2 实际效果

**场景1: 线性RCS + Z-score** ❌ 不推荐
```python
RCS_linear = [0.001, 0.01, 0.1, 1.0, 10.0]  # 跨5个数量级
μ = 2.222, σ = 4.216
Z-score = [-0.527, -0.525, -0.503, -0.290, 1.845]
# 问题: 大值主导，小值被压缩
```

**场景2: 对数变换 + Z-score** ✅ 推荐
```python
Log_RCS = log([0.001, 0.01, 0.1, 1.0, 10.0]) = [-6.91, -4.61, -2.30, 0, 2.30]
μ = -2.304, σ = 3.457
Z-score = [-1.33, -0.67, -0.01, 0.67, 1.33]
# 优势: 对称分布，训练更稳定
```

### 4.3 建议配置

| 数据类型 | 对数变换 | 标准化 | 原因 |
|---------|---------|--------|------|
| **小波系数** | ❌ 关闭 | ✅ 开启 | 系数可正可负，接近正态分布 |
| **直接RCS** | ✅ 开启 | ✅ 开启 | 对数化后接近正态分布 |
| **原始线性RCS** | ❌ 关闭 | ❌ 关闭 | 仅作为小波变换输入 |

## 5. 两个预处理栏的冲突问题

### 5.1 当前设计
- **数据管理页面**: 仅用于可视化，不保存预处理后的数据
- **AutoEncoder页面**: 实际训练使用的预处理配置

### 5.2 是否冲突？
**不冲突**，因为:
1. 数据管理页面的预处理是**临时的**，仅用于显示统计信息
2. AutoEncoder页面的预处理是**训练时实时应用**的
3. 两者使用不同的对数方法 (dB vs sign*log)

### 5.3 用户混淆风险
**潜在混淆**:
- 用户可能认为数据管理的"对数化"会影响训练
- 实际上训练只受AE页面的配置影响

**建议改进**:
1. 数据管理页面改名为 "数据预览预处理" (仅影响显示)
2. AE页面标注 "训练数据预处理" (实际影响训练)
3. 增加工具提示说明两者独立

## 6. 修复后的完整流程

### 6.1 Stage 1: AutoEncoder预训练
```python
# 读取原始数据
rcs_data = load_data()  # [N, 91, 91, num_freq], 线性标度, >0

if mode == 'wavelet':
    # ✅ 正确: 先小波变换
    wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
    # 再预处理小波系数
    input_data = data_adapter.adapt_rcs_data(wavelet_coeffs)
else:
    # 直接模式: 预处理原始RCS
    input_data = data_adapter.adapt_rcs_data(rcs_data)

# 训练AutoEncoder
autoencoder.train(input_data)
```

### 6.2 Stage 2: 参数映射训练
```python
# 与Stage 1保持一致的预处理
if mode == 'wavelet':
    wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
    input_data = data_adapter.adapt_rcs_data(wavelet_coeffs)
else:
    input_data = data_adapter.adapt_rcs_data(rcs_data)

# 提取隐空间表示
latents = autoencoder.encode(input_data)

# 训练参数映射
parameter_mapper.train(params → latents)
```

### 6.3 Stage 3: 端到端微调
```python
# 与Stage 1/2保持一致的预处理
if mode == 'wavelet':
    wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
    target_data = data_adapter.adapt_rcs_data(wavelet_coeffs)
else:
    target_data = data_adapter.adapt_rcs_data(rcs_data)

# 端到端训练
predicted = autoencoder.decode(parameter_mapper(params))
loss = criterion(predicted, target_data)
```

## 7. 数据统计信息的保存

### 7.1 为什么需要保存？
训练时计算的统计信息 (μ, σ) 必须在推理时复用:
```python
# 训练时:
adapted_data = (data - μ_train) / σ_train

# 推理时必须使用相同的统计:
adapted_data = (new_data - μ_train) / σ_train  # ✅
# 而不是:
adapted_data = (new_data - μ_new) / σ_new      # ❌ 错误!
```

### 7.2 保存位置
模型checkpoint中保存:
```python
checkpoint = {
    'autoencoder': autoencoder.state_dict(),
    'parameter_mapper': parameter_mapper.state_dict(),
    'adapter_stats': {
        'mean': μ_train,
        'std': σ_train,
        'log_transform': True/False
    },
    'config': {...}
}
```

## 8. 推理流程

### 8.1 从参数预测RCS
```python
# 1. 加载模型和统计信息
checkpoint = torch.load('model.pth')
data_adapter.data_stats = checkpoint['adapter_stats']

# 2. 参数 → 隐空间 → 小波系数/RCS
params = torch.tensor([...])  # [1, 9]
latents = parameter_mapper(params)
predicted_data = autoencoder.decode(latents)  # 预处理后的数据

# 3. 逆预处理
if mode == 'wavelet':
    # 先逆标准化
    wavelet_coeffs = data_adapter.inverse_adapt(predicted_data)
    # 再逆小波变换
    rcs_predicted = wavelet_transform.inverse_transform(wavelet_coeffs)
else:
    # 直接逆标准化
    rcs_predicted = data_adapter.inverse_adapt(predicted_data)

# 4. 得到线性标度RCS (>0)
return rcs_predicted  # [1, 91, 91, num_freq]
```

## 9. 关键要点总结

### ✅ 正确做法
1. **小波变换在原始线性数据上运行** (未标准化)
2. **标准化在小波系数上运行** (小波模式) 或直接RCS上 (直接模式)
3. **对数变换 + Z-score组合使用** 用于稳定训练
4. **训练统计信息必须保存** 用于推理
5. **数据管理的预处理不影响训练**

### ❌ 错误做法
1. 在标准化后的数据上做小波变换
2. 对线性RCS直接用Z-score (不先对数化)
3. 推理时重新计算统计信息
4. 混淆数据管理和训练的预处理配置

### ⚠️ 当前bug
**所有三个训练阶段都在标准化后做小波变换** → 需要立即修复!
