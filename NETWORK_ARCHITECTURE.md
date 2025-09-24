# RCS小波神经网络架构详解

## 概述

这是一个基于小波多尺度理论的深度神经网络，用于从9个飞行器参数预测三维RCS（雷达散射截面）数据。

**输入**: 9维参数向量（飞行器几何参数）
**输出**: [91×91×2] 张量（φ角×θ角×频率）

---

## 网络整体架构

```
输入参数 [B, 9]
    ↓
参数编码器 (Linear: 9→128→256)
    ↓
多尺度小波特征提取器 (4个尺度)
    ↓
频率交互模块 (跨频率注意力)
    ↓        ↓
1.5GHz分支  3GHz分支
    ↓        ↓
渐进式解码器 (23×23→46×46→91×91)
    ↓        ↓
    [91×91]  [91×91]
         ↓
    物理对称性约束
         ↓
    输出 [B, 91×91×2]
```

---

## 1. 参数编码器（Parameter Encoder）

**作用**: 将9个飞行器参数编码为高维特征表示

### 结构
```python
Linear(9 → 128)
BatchNorm1d(128)
ReLU()
Dropout(0.2)
    ↓
Linear(128 → 256)
BatchNorm1d(256)
ReLU()
Dropout(0.2)
```

### 输入输出
- **输入**: `[B, 9]` 飞行器参数（长度、宽度、高度等）
- **输出**: `[B, 256]` 高维编码特征

### 关键点
- **BatchNorm**: 稳定训练，加速收敛
- **Dropout**: 防止过拟合（p=0.2）
- **ReLU**: 非线性激活

---

## 2. 多尺度小波特征提取器（MultiScaleWaveletExtractor）

**作用**: 在φ-θ平面进行多分辨率特征提取，捕捉不同尺度的散射特性

### 核心原理

**小波变换数学**:
```
W(a,b) = ∫∫ f(x,y) ψ*((x-b_x)/a, (y-b_y)/a) dx dy
```
- ψ: 母小波
- a: 尺度参数（控制分辨率）
- b: 平移参数（控制位置）

### 2D小波分解（Wavelet2DConv）

每个小波层执行4个子带分解：

```
输入 [B, C, H, W]
    ↓
LL (低频-低频): 平均池化 → [B, C, H/2, W/2]
LH (低频-高频): 水平差分+池化 → [B, C, H/2, W/2]
HL (高频-低频): 垂直差分+池化 → [B, C, H/2, W/2]
HH (高频-高频): 对角差分+池化 → [B, C, H/2, W/2]
    ↓
4个可学习卷积处理各子带
    ↓
拼接 → [B, out_channels, H/2, W/2]
```

**物理意义**:
- **LL**: 平滑区域（主要散射贡献）
- **LH**: 水平边缘（φ方向变化）
- **HL**: 垂直边缘（θ方向变化）
- **HH**: 角点和细节（高频散射）

### 4尺度金字塔结构

**第一步：参数→2D空间映射**
```python
Linear(256 → 23×23×64) → Reshape → [B, 64, 23, 23]
```

**4个尺度的小波分解**:
```
尺度1 (db4):    [B, 64, 23, 23] → [B, 64, 12, 12]  # 低分辨率全局特征
尺度2 (db4):    [B, 64, 12, 12] → [B, 64, 6, 6]    # 中等分辨率
尺度3 (bior2.2): [B, 64, 6, 6]   → [B, 64, 3, 3]    # 高分辨率
尺度4 (bior2.2): [B, 64, 3, 3]   → [B, 64, 2, 2]    # 细节特征
```

**特征融合**:
```
上采样到统一尺寸(12×12) → 拼接 → [B, 256, 12, 12]
    ↓
Conv2d(256→64) + BatchNorm + ReLU
    ↓
AdaptiveAvgPool → [B, 64, 23, 23]
```

### 小波类型说明

**db4 (Daubechies 4)**:
- 优点：光滑、对称性好
- 用于：尺度1和2（粗糙特征）

**bior2.2 (双正交)**:
- 优点：完美重构、紧支撑
- 用于：尺度3和4（细节特征）

---

## 3. 频率交互模块（FrequencyInteractionModule）

**作用**: 建模1.5GHz和3GHz间的物理关系

### 结构

**分频编码**:
```python
共享特征 [B, 64, 23, 23]
    ↓           ↓
频率1编码器   频率2编码器
Conv→BN→ReLU  Conv→BN→ReLU
    ↓           ↓
[B, 64, 23, 23] [B, 64, 23, 23]
```

**跨频率注意力**:
```
Reshape → [B, 529, 64]  (529 = 23×23)
    ↓
MultiheadAttention(8 heads)
freq1_attended ← Attention(freq1, freq2, freq2)
freq2_attended ← Attention(freq2, freq1, freq1)
    ↓
Reshape back → [B, 64, 23, 23]
```

**特征融合**:
```python
freq1_enhanced = Conv(concat[freq1_feat, freq1_attended])
freq2_enhanced = Conv(concat[freq2_feat, freq2_attended])
```

### 物理意义

**频率关系**:
- f2/f1 = 3GHz/1.5GHz = 2
- 散射机制的频率依赖性
- 共振频率可能不同

**注意力机制**捕捉：
- 相同散射源在不同频率的响应
- 频率间的互补信息

---

## 4. 渐进式解码器（ProgressiveDecoder）

**作用**: 逐步上采样生成高分辨率RCS图

### 为什么渐进式？

❌ **直接生成91×91的问题**:
- 参数量巨大（64×91×91 = 531,664）
- 难以收敛
- 容易丢失细节

✅ **渐进式优势**:
- 逐步增加分辨率
- 每个阶段专注不同尺度
- 更稳定的训练

### 三阶段上采样

**阶段1: 23×23 → 46×46**
```python
ConvTranspose2d(64→32, kernel=4, stride=2)  # 转置卷积上采样
BatchNorm2d(32)
ReLU()
Conv2d(32→32, kernel=3)  # 细化特征
BatchNorm2d(32)
ReLU()
```

**阶段2: 46×46 → 91×91**
```python
Interpolate(46×46 → 91×91)  # 双线性插值
Conv2d(32→16, kernel=3)
BatchNorm2d(16)
ReLU()
Conv2d(16→8, kernel=3)
BatchNorm2d(8)
ReLU()
```

**最终输出**:
```python
Conv2d(8→1, kernel=1)  # 点卷积降维
    ↓
激活函数:
  - use_log_output=True: Identity (无激活，对数域)
  - use_log_output=False: Softplus (确保正值，线性域)
```

### 输出激活函数

**对数域（use_log_output=True）**:
```python
Identity()  # 无激活
# 输出范围: (-∞, +∞) 标准化dB值
```

**线性域（use_log_output=False）**:
```python
Softplus(x) = log(1 + exp(x))
# 输出范围: (0, +∞) 确保RCS为正值
```

---

## 5. 物理对称性约束

**物理原理**: φ=0°平面对称性
```
σ(φ, θ, f) = σ(-φ, θ, f)
```

### 实现方式

**数据维度**: `[B, 91_theta, 91_phi, 2_freq]`
- θ: 45°~135°（91个点）
- φ: -45°~+45°（91个点）
- φ=0°对应索引45

**对称约束算法**:
```python
for i in [1, 2, ..., 45]:
    left_idx = 45 - i   # φ < 0
    right_idx = 45 + i  # φ > 0

    # 对称平均
    avg = (RCS[left] + RCS[right]) / 2
    RCS[left] = avg
    RCS[right] = avg
```

**效果**: 强制输出满足物理对称性

---

## 6. 损失函数（TriDimensionalRCSLoss）

### 三个损失项

**1. MSE损失（主导）**
```python
losses['mse'] = MSE(pred_rcs, target_rcs)
权重: 1.0
```
- 训练域：标准化dB值
- 确保预测接近真实RCS

**2. 对称性损失（辅助）**
```python
# 修复后版本
pred_sym_diff = pred_left - pred_right
target_sym_diff = target_left - target_right
losses['symmetry'] = MSE(pred_sym_diff, target_sym_diff)
权重: 0.02（早期）→ 0.02（后期）
```
- 确保预测和目标的对称性一致
- 避免"对称但错误"的输出

**3. 多尺度损失（辅助）**
```python
for scale in [1, 2, 4]:
    pred_downsampled = AvgPool2d(pred, scale)
    target_downsampled = AvgPool2d(target, scale)
    losses['multiscale'] += MSE(pred, target) / scale
权重: 0.1（早期）→ 0.1（后期）
```
- 确保不同分辨率下的一致性

### 渐进式权重调整

```python
progress = epoch / total_epochs
weights = {
    'mse': 1.0,                        # 始终主导
    'symmetry': 0.05 - 0.03*progress,  # 0.05→0.02
    'multiscale': 0.3 - 0.2*progress   # 0.3→0.1
}
```

**总损失**:
```
total_loss = 1.0×MSE + 0.05×symmetry + 0.3×multiscale  (early)
           = 1.0×MSE + 0.02×symmetry + 0.1×multiscale  (late)
```

---

## 7. 数据流详解

### 训练数据流

**原始RCS数据**:
```
CSV文件 [91_theta, 91_phi] 线性值
    ↓
10 * log10(RCS) → dB值 [-57 ~ -5 dB]
    ↓
标准化: (dB - mean) / std → [-3 ~ 3]
    ↓
存储 preprocessing_stats: {mean, std}
```

**训练过程**:
```
输入: 参数 [B, 9]
目标: 标准化RCS [B, 91, 91, 2]
    ↓
前向传播 → 预测RCS [B, 91, 91, 2]
    ↓
损失计算: MSE(pred, target) 在标准化域
    ↓
反向传播 → 更新权重
```

**checkpoint保存**:
```python
{
    'model_state_dict': model.state_dict(),
    'preprocessing_stats': {'mean': -31.2, 'std': 12.5},
    'use_log_output': True,
    'epoch': 150,
    'val_loss': 0.0123
}
```

### 预测数据流

**加载模型**:
```
checkpoint → model.load_state_dict()
           → 恢复 preprocessing_stats
```

**预测过程**:
```
输入参数 [1, 9]
    ↓
网络前向 → 标准化RCS [1, 91, 91, 2]
    ↓
反标准化: pred_db = pred_std * std + mean
    ↓
输出: RCS (dB) [-57 ~ -5 dB]
```

---

## 8. 模型规模

### 参数统计

**参数编码器**: ~33K参数
- Linear(9→128): 1,152
- Linear(128→256): 32,768

**多尺度小波**: ~200K参数
- 4个Wavelet2DConv层
- 特征融合层

**频率交互**: ~80K参数
- 2个频率编码器
- MultiheadAttention(8 heads)

**双频解码器**: ~100K参数
- 2个ProgressiveDecoder

**总参数量**: ~413K参数
**模型大小**: ~1.6 MB (FP32)

### 计算复杂度

**前向传播**:
```
参数编码:     9 → 128 → 256        O(33K)
小波提取:     256 → 64×23×23       O(200K)
频率交互:     跨频率注意力          O(80K)
解码上采样:   23×23 → 91×91       O(100K)
```

**峰值显存**（batch=8）:
- 输入: 8×9 = 72 bytes
- 中间特征: ~50 MB
- 梯度: ~6 MB
- 总计: ~60 MB/样本

---

## 9. 关键设计选择

### 为什么用小波？

**优势**:
1. **多尺度分析**: 自然捕捉不同尺度散射
2. **稀疏表示**: RCS在小波域更稀疏
3. **边缘检测**: 小波天然适合检测突变

**替代方案对比**:
- ❌ 纯卷积：单一尺度，难以捕捉全局-局部关系
- ❌ Transformer：参数量大，小数据集易过拟合
- ✅ 小波多尺度：平衡表达能力和参数效率

### 为什么双频分支？

**物理原因**:
- 不同频率散射机制不同
- 共振频率差异
- 需要频率特异性建模

**网络设计**:
- 共享底层特征（参数编码+小波提取）
- 频率交互建模关联
- 分支解码保持独立性

### 为什么渐进式解码？

**23×23 → 46×46 → 91×91 的理由**:
1. **稳定训练**: 避免直接生成高分辨率
2. **多尺度监督**: 每个阶段有明确目标
3. **参数效率**: 比直接上采样少10倍参数

---

## 10. 训练策略

### 学习率调度

**CosineAnnealingWarmRestarts**:
```python
初始LR: 0.003
最低LR: 0.00002 (eta_min)
周期: T_0=50 epochs
```

**调度曲线**:
```
LR
^
|  /\      /\      /\
| /  \    /  \    /  \
|/    \  /    \  /    \___
+-------------------------> Epoch
0    50   100   150   200
```

### 5折交叉验证

**数据分割**:
```
100样本 → 5折
每折: 80训练 + 20验证
```

**训练流程**:
```
for fold in [1, 2, 3, 4, 5]:
    train on 80 samples
    validate on 20 samples
    save best model

选择验证loss最低的fold
```

### 早停策略

```python
patience = 50 epochs
if val_loss not improve for 50 epochs:
    stop training
```

---

## 11. 关键修复历史

### 修复1: φ=0°对称性维度错误
**问题**: 数据是[θ, φ]但代码假设[φ, θ]
**修复**: 改为正确维度索引`[:, :, phi_idx, :]`

### 修复2: 对称性损失设计缺陷
**问题**: 只要求pred自身对称，导致输出常数
**修复**: 改为`MSE(pred_sym_diff, target_sym_diff)`

### 修复3: 评估域不匹配
**问题**: 训练在标准化域，评估错误地log10
**修复**: 加载preprocessing_stats，正确反标准化

---

## 总结

这是一个**物理驱动**的深度学习架构：

✅ **小波多尺度** → 捕捉不同尺度散射
✅ **频率交互** → 建模双频关系
✅ **渐进式解码** → 稳定高分辨率生成
✅ **对称性约束** → 强制物理一致性
✅ **域正确性** → 训练-评估-预测统一

**核心创新**:
1. 小波金字塔特征提取（4尺度）
2. 跨频率注意力机制
3. 渐进式3阶段解码
4. 物理约束与数据驱动结合