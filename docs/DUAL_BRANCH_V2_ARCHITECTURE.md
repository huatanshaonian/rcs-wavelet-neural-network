# 双分支可微分小波AutoEncoder V2 架构说明

> **版本**: V2 (正确实现)
> **文档目的**: 详细梳理网络结构和前向/反向传播路径，供绘制架构图使用
> **创建日期**: 2025-01-18

---

## 📋 目录

1. [核心设计理念](#核心设计理念)
2. [V1 vs V2 关键区别](#v1-vs-v2-关键区别)
3. [完整架构图（文字描述）](#完整架构图文字描述)
4. [前向传播详细路径](#前向传播详细路径)
5. [反向传播梯度流](#反向传播梯度流)
6. [损失计算](#损失计算)
7. [数据维度变化表](#数据维度变化表)
8. [MLP vs CNN 架构对比](#mlp-vs-cnn-架构对比)
9. [关键代码映射](#关键代码映射)

---

## 核心设计理念

### 为什么需要双分支？

**物理背景**：
- **LL通道（低频）**: 包含90%+的RCS能量，代表目标的整体特征
- **HF通道（高频）**: 包含<10%的能量，代表边缘和细节信息

**设计动机**：
1. LL和HF的重要性不对称 → 分配不同的latent空间维度
2. LL和HF的特征模式不同 → 使用独立的编码器/解码器
3. 避免高频细节被低频主导淹没 → 强制分离处理

### 可微分小波变换

**传统方案（V0）**:
```
RCS ──numpy小波──> 系数 ──AE──> 系数 ──numpy逆小波──> RCS
                   ↑                    ↑
                   损失在此             梯度断开
```

**可微分方案（V1/V2）**:
```
RCS ──torch小波──> 系数 ──AE──> 系数 ──torch逆小波──> RCS
                                                       ↑
                                                    损失在此
                                                 梯度可回传
```

**优势**:
- ✅ 直接优化物理量（RCS）而非中间表示（小波系数）
- ✅ 梯度可贯穿整个网络，理论上学习效果更好
- ✅ 易于添加物理约束（如Softplus保证非负）
- ✅ GPU加速：比numpy快50-100倍

---

## V1 vs V2 关键区别

| 维度 | V1 (有缺陷) | V2 (正确) |
|------|------------|-----------|
| **Decoder架构** | 单分支 (fusion → decoder) | 双分支 (ll_decoder + hf_decoder) |
| **ll_latent_dim** | 计算但未使用（硬编码128） | 真正起作用 |
| **hf_latent_dim** | 计算但未使用（硬编码128） | 真正起作用 |
| **ll_ratio参数** | 无作用 | 实际控制latent分配 |
| **通道组合** | 缺少_combine_channels() | 正确实现 |
| **架构对称性** | Encoder双分支，Decoder单分支 | Encoder和Decoder都是双分支 |

**示例（latent_dim=32, ll_ratio=0.7）**:
- V1: ll_latent=128, hf_latent=128 (固定，忽略ll_ratio)
- V2: ll_latent=22, hf_latent=10 (动态计算，尊重ll_ratio)

---

## 完整架构图（文字描述）

### 高层视图

```
输入: RCS数据 [B, 91, 91, 2]
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                   Differentiable Wavelet Transform             │
│                      (可微分小波变换)                          │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
小波系数 [B, 49, 49, 8]
    │
    ├─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼
  split_channels()
    │
    ├────────────────> LL通道 [B, 49, 49, 2]  (LL0, LL1)
    │
    └────────────────> HF通道 [B, 49, 49, 6]  (LH0,HL0,HH0, LH1,HL1,HH1)
    │                     │
    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│  LL Encoder     │  │  HF Encoder     │
│  (独立处理)     │  │  (独立处理)     │
└─────────────────┘  └─────────────────┘
    │                     │
    ▼                     ▼
ll_latent [B, 22]    hf_latent [B, 10]
    │                     │
    └──────────┬──────────┘
               ▼
      concat [B, 32]  ← Latent Space（隐空间）
               │
               ▼
      split [ll_latent, hf_latent]
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│  LL Decoder     │  │  HF Decoder     │
│  (对称设计)     │  │  (对称设计)     │
└─────────────────┘  └─────────────────┘
    │                     │
    ▼                     ▼
LL重建 [B, 49, 49, 2]  HF重建 [B, 49, 49, 6]
    │                     │
    └──────────┬──────────┘
               ▼
        combine_channels()
               │
               ▼
小波系数重建 [B, 49, 49, 8]
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│              Differentiable Inverse Wavelet Transform          │
│                    (可微分逆小波变换)                          │
└───────────────────────────────────────────────────────────────┘
               │
               ▼
RCS重建 [B, 91, 91, 2]
               │
               ▼
         (可选) Softplus激活
               │
               ▼
输出: RCS预测 [B, 91, 91, 2]
               │
               ▼
         MSE Loss (在RCS空间)
               │
               ▼
         梯度反向传播 (可微分路径)
```

---

## 前向传播详细路径

### MLP版本 (DualBranchDifferentiableWaveletMLPAutoEncoderV2)

#### Encoding阶段

```python
# Step 1: RCS → 小波系数 (可微分)
rcs_data: [B, 91, 91, 2]
    │
    ▼ wavelet_transform.forward_transform()
wavelet_coeffs: [B, 49, 49, 8]
```

**小波变换细节**:
- 使用ptwt库（PyTorch Wavelet Toolbox）
- 对每个频率独立做2D小波变换
- 每个频率产生4个子带：LL, LH, HL, HH
- 小波类型：db4（Daubechies 4）
- 边界模式：symmetric

```python
# Step 2: 分离LL和HF通道
wavelet_coeffs: [B, 49, 49, 8]
    │
    ▼ _split_channels()
    ├─> ll_channels: [B, 49, 49, 2]  # [LL0, LL1]
    └─> hf_channels: [B, 49, 49, 6]  # [LH0,HL0,HH0, LH1,HL1,HH1]
```

**通道顺序说明**:
- **输入**: [f0_LL, f0_LH, f0_HL, f0_HH, f1_LL, f1_LH, f1_HL, f1_HH]
- **LL输出**: [f0_LL, f1_LL]
- **HF输出**: [f0_LH, f0_HL, f0_HH, f1_LH, f1_HL, f1_HH]

```python
# Step 3: Flatten
ll_channels: [B, 49, 49, 2]
    │ reshape
    ▼
ll_flat: [B, 4802]  # 49*49*2

hf_channels: [B, 49, 49, 6]
    │ reshape
    ▼
hf_flat: [B, 14406]  # 49*49*6
```

```python
# Step 4: 双分支MLP编码
ll_flat: [B, 4802]
    │
    ▼ ll_encoder (3层MLP)
    │ Linear(4802 → 512) → ReLU → Dropout
    │ Linear(512 → 256) → ReLU → Dropout
    │ Linear(256 → ll_latent_dim)
    ▼
ll_latent: [B, 22]  # 假设latent_dim=32, ll_ratio=0.7

hf_flat: [B, 14406]
    │
    ▼ hf_encoder (3层MLP)
    │ Linear(14406 → 512) → ReLU → Dropout
    │ Linear(512 → 256) → ReLU → Dropout
    │ Linear(256 → hf_latent_dim)
    ▼
hf_latent: [B, 10]
```

```python
# Step 5: Concat（无额外fusion层）
ll_latent: [B, 22] ─┐
                    ├─> torch.cat(dim=1)
hf_latent: [B, 10] ─┘
    │
    ▼
latent: [B, 32]  ← 隐空间表示
```

#### Decoding阶段

```python
# Step 1: Split latent
latent: [B, 32]
    │
    ├─> latent[:, :22] → ll_latent: [B, 22]
    └─> latent[:, 22:] → hf_latent: [B, 10]
```

```python
# Step 2: 双分支MLP解码
ll_latent: [B, 22]
    │
    ▼ ll_decoder (对称3层MLP)
    │ Linear(22 → 256) → ReLU → Dropout
    │ Linear(256 → 512) → ReLU → Dropout
    │ Linear(512 → 4802)
    ▼
ll_flat: [B, 4802]

hf_latent: [B, 10]
    │
    ▼ hf_decoder (对称3层MLP)
    │ Linear(10 → 256) → ReLU → Dropout
    │ Linear(256 → 512) → ReLU → Dropout
    │ Linear(512 → 14406)
    ▼
hf_flat: [B, 14406]
```

```python
# Step 3: Reshape
ll_flat: [B, 4802]
    │ view
    ▼
ll_channels: [B, 49, 49, 2]

hf_flat: [B, 14406]
    │ view
    ▼
hf_channels: [B, 49, 49, 6]
```

```python
# Step 4: 组合LL和HF（关键！）
ll_channels: [B, 49, 49, 2]  # [f0_LL, f1_LL]
hf_channels: [B, 49, 49, 6]  # [f0_LH, f0_HL, f0_HH, f1_LH, f1_HL, f1_HH]
    │
    ▼ _combine_channels()
    │ 对每个频率: cat([LL, LH, HL, HH])
    ▼
wavelet_coeffs: [B, 49, 49, 8]  # [f0_LL, f0_LH, f0_HL, f0_HH, f1_LL, ...]
```

**_combine_channels() 伪代码**:
```python
for freq_idx in range(num_frequencies):
    ll = ll_channels[:, :, :, freq_idx]           # [B, 49, 49, 1]
    hf = hf_channels[:, :, :, freq_idx*3:(freq_idx+1)*3]  # [B, 49, 49, 3]
    freq_coeffs = concat([ll, hf], dim=3)         # [B, 49, 49, 4]
    all_freq_coeffs.append(freq_coeffs)

wavelet_coeffs = concat(all_freq_coeffs, dim=3)   # [B, 49, 49, 8]
```

```python
# Step 5: 逆小波变换 (可微分)
wavelet_coeffs: [B, 49, 49, 8]
    │
    ▼ wavelet_transform.inverse_transform()
    │ 对每个频率独立执行ptwt.waverec2()
    │ 自动裁剪到原始尺寸(91, 91)
    ▼
rcs_data: [B, 91, 91, 2]
```

```python
# Step 6: 应用输出激活（可选）
rcs_data: [B, 91, 91, 2]
    │
    ▼ apply_output_activation()
    │ if output_activation == 'softplus':
    │     return Softplus(rcs_data)
    │ else:
    │     return rcs_data
    ▼
output: [B, 91, 91, 2]
```

---

### CNN版本 (DualBranchDifferentiableWaveletAutoEncoderV2)

#### Encoding阶段差异

```python
# 前3步同MLP版本（小波变换 + 分离通道）

# Step 4: 双分支CNN编码
ll_channels: [B, 49, 49, 2]
    │ permute → [B, 2, 49, 49]
    ▼ ll_branch (CNN)
    │ Conv2d(2→16, k=7, p=3) → BN → ReLU
    │ Conv2d(16→32, k=3, s=2, p=1) → BN → ReLU → Dropout  # [B,32,25,25]
    │ Conv2d(32→64, k=3, s=2, p=1) → BN → ReLU → Dropout  # [B,64,13,13]
    ▼
ll_feat: [B, 64, 13, 13]

hf_channels: [B, 49, 49, 6]
    │ permute → [B, 6, 49, 49]
    ▼ hf_branch (CNN)
    │ Conv2d(6→16, k=3, p=1) → BN → ReLU
    │ Conv2d(16→32, k=3, s=2, p=1) → BN → ReLU → Dropout  # [B,32,25,25]
    │ Conv2d(32→64, k=3, s=2, p=1) → BN → ReLU → Dropout  # [B,64,13,13]
    ▼
hf_feat: [B, 64, 13, 13]
```

```python
# Step 5: 融合
ll_feat: [B, 64, 13, 13] ─┐
                          ├─> concat(dim=1)
hf_feat: [B, 64, 13, 13] ─┘
    │
    ▼
fused: [B, 128, 13, 13]
    │
    ▼ fusion (CNN)
    │ Conv2d(128→128, k=3, p=1) → BN → ReLU → Dropout
    │ Conv2d(128→128, k=3, s=2, p=1) → BN → ReLU     # [B,128,7,7]
    │ AdaptiveAvgPool2d(1)                           # [B,128,1,1]
    ▼
fused: [B, 128, 1, 1]
```

```python
# Step 6: 双分支FC编码
fused: [B, 128, 1, 1]
    │
    ├─> ll_encoder_fc
    │   Flatten → Linear(128 → ll_latent_dim)
    │   ▼
    │   ll_latent: [B, 22]
    │
    └─> hf_encoder_fc
        Flatten → Linear(128 → hf_latent_dim)
        ▼
        hf_latent: [B, 10]
```

```python
# Step 7: Concat
ll_latent: [B, 22] ─┐
                    ├─> concat
hf_latent: [B, 10] ─┘
    ▼
latent: [B, 32]
```

#### Decoding阶段差异

```python
# Step 1-2: Split + 双分支FC解码（同MLP）
latent: [B, 32] → split → [ll_latent, hf_latent]
    │
    ├─> ll_decoder_fc: [B, 22] → [B, 128]
    └─> hf_decoder_fc: [B, 10] → [B, 128]
```

```python
# Step 3: Reshape
ll_feat: [B, 128] → view → [B, 128, 1, 1]
hf_feat: [B, 128] → view → [B, 128, 1, 1]
```

```python
# Step 4: 双分支CNN解码
ll_feat: [B, 128, 1, 1]
    │
    ▼ ll_decoder_net (CNN)
    │ ConvTranspose2d(128→128, k=7, s=1) → BN → ReLU → Dropout  # [B,128,7,7]
    │ ConvTranspose2d(128→64, k=3, s=2) → BN → ReLU → Dropout   # [B,64,13,13]
    │ ConvTranspose2d(64→32, k=3, s=2) → BN → ReLU → Dropout    # [B,32,25,25]
    │ ConvTranspose2d(32→2, k=3, s=2)                           # [B,2,~49,~49]
    │ AdaptiveAvgPool2d(49, 49)
    ▼
ll_channels: [B, 2, 49, 49]

hf_feat: [B, 128, 1, 1]
    │
    ▼ hf_decoder_net (CNN)
    │ ConvTranspose2d(128→128, k=7, s=1) → BN → ReLU → Dropout  # [B,128,7,7]
    │ ConvTranspose2d(128→64, k=3, s=2) → BN → ReLU → Dropout   # [B,64,13,13]
    │ ConvTranspose2d(64→32, k=3, s=2) → BN → ReLU → Dropout    # [B,32,25,25]
    │ ConvTranspose2d(32→6, k=3, s=2)                           # [B,6,~49,~49]
    │ AdaptiveAvgPool2d(49, 49)
    ▼
hf_channels: [B, 6, 49, 49]
```

```python
# Step 5-7: 组合通道 + 逆小波 + 输出激活（同MLP）
```

---

## 反向传播梯度流

### 完整梯度路径

```
Loss (MSE in RCS space)
    │
    │ ∂L/∂RCS_recon
    ▼
RCS重建 [B, 91, 91, 2]
    │
    │ ∂RCS/∂wavelet (逆小波可微)
    ▼
小波系数重建 [B, 49, 49, 8]
    │
    │ ∂wavelet/∂[LL, HF] (combine_channels可微)
    ▼
┌─────────────────────────────────────────┐
│ LL重建 [B, 49, 49, 2]                   │
│     │                                    │
│     │ ∂LL/∂ll_latent (decoder可微)      │
│     ▼                                    │
│ ll_latent [B, 22] ──┐                   │
│                     │                   │
│ HF重建 [B, 49, 49, 6]                   │
│     │                                    │
│     │ ∂HF/∂hf_latent (decoder可微)      │
│     ▼                                    │
│ hf_latent [B, 10] ──┤                   │
│                     │                   │
│                     ▼                    │
│              latent [B, 32]              │
│                     │                   │
│                     │ (反向split)        │
│                     ▼                    │
│         ┌──────────┴──────────┐         │
│         │                     │         │
│    ∂latent/∂ll_latent    ∂latent/∂hf_latent │
│         │                     │         │
│         ▼                     ▼         │
│    ll_encoder          hf_encoder       │
│         │                     │         │
│         ▼                     ▼         │
│    LL通道               HF通道          │
└─────────────────────────────────────────┘
    │
    │ ∂channels/∂wavelet (split_channels可微)
    ▼
小波系数 [B, 49, 49, 8]
    │
    │ ∂wavelet/∂RCS (前向小波可微)
    ▼
RCS输入 [B, 91, 91, 2]
```

### 关键点

1. **全程可微分**：从RCS输入到RCS输出，整条路径都是PyTorch可微分操作
2. **无梯度断点**：小波变换使用ptwt（PyTorch原生），不经过numpy
3. **双分支独立梯度**：LL和HF的梯度独立计算，互不干扰
4. **物理空间优化**：损失直接在RCS空间计算，梯度直接优化物理量

---

## 损失计算

### 训练时损失函数

```python
# Stage 1: AutoEncoder预训练
def compute_loss(rcs_data_batch):
    # 前向传播
    rcs_recon, latent = model(rcs_data_batch)  # [B, 91, 91, 2]

    # 损失在RCS空间计算
    loss = MSE(rcs_recon, rcs_data_batch)

    # 梯度回传（可微分路径）
    loss.backward()

    return loss

# 梯度流向：
# RCS_recon ← 逆小波 ← [LL, HF] ← [ll_decoder, hf_decoder]
#                                  ← [ll_latent, hf_latent]
#                                  ← [ll_encoder, hf_encoder]
#                                  ← [LL, HF] ← 小波 ← RCS_input
```

### 物理约束（可选）

```python
# 如果启用 output_activation='softplus'
rcs_recon = model.apply_output_activation(rcs_recon)
# 确保 rcs_recon >= 0（物理约束：RCS不能为负）

# Softplus: y = log(1 + exp(x))
# 特性：
# - 平滑、处处可微
# - y >= 0
# - x→∞时，y≈x（线性区域）
# - x→-∞时，y≈0（饱和区域）
```

### 与传统方案对比

| 方案 | 损失空间 | 梯度路径 | 物理意义 |
|------|---------|---------|---------|
| **传统（V0）** | 小波系数空间 | 断开（numpy） | 优化中间表示 |
| **可微分（V1/V2）** | RCS物理空间 | 完整（torch） | 直接优化物理量 |

---

## 数据维度变化表

### MLP版本完整维度流

| 阶段 | 操作 | 输入维度 | 输出维度 |
|------|------|---------|---------|
| **Encoding** |
| 1 | 输入 | - | [B, 91, 91, 2] |
| 2 | 小波变换 | [B, 91, 91, 2] | [B, 49, 49, 8] |
| 3 | 分离通道 | [B, 49, 49, 8] | LL:[B,49,49,2] + HF:[B,49,49,6] |
| 4 | Flatten | LL:[B,49,49,2] | LL:[B,4802] |
| | | HF:[B,49,49,6] | HF:[B,14406] |
| 5 | LL Encoder | [B, 4802] | [B, 512] → [B, 256] → [B, 22] |
| | HF Encoder | [B, 14406] | [B, 512] → [B, 256] → [B, 10] |
| 6 | Concat | LL:[B,22] + HF:[B,10] | [B, 32] |
| **Decoding** |
| 7 | Split | [B, 32] | LL:[B,22] + HF:[B,10] |
| 8 | LL Decoder | [B, 22] | [B, 256] → [B, 512] → [B, 4802] |
| | HF Decoder | [B, 10] | [B, 256] → [B, 512] → [B, 14406] |
| 9 | Reshape | LL:[B,4802] | LL:[B,49,49,2] |
| | | HF:[B,14406] | HF:[B,49,49,6] |
| 10 | 组合通道 | LL:[B,49,49,2] + HF:[B,49,49,6] | [B, 49, 49, 8] |
| 11 | 逆小波 | [B, 49, 49, 8] | [B, 91, 91, 2] |
| 12 | 输出 | [B, 91, 91, 2] | [B, 91, 91, 2] |

### CNN版本关键维度差异

| 阶段 | 操作 | MLP维度 | CNN维度 |
|------|------|---------|---------|
| LL Encoder | 输入 | [B, 4802] | [B, 2, 49, 49] |
| | 输出 | [B, 22] | [B, 64, 13, 13] |
| HF Encoder | 输入 | [B, 14406] | [B, 6, 49, 49] |
| | 输出 | [B, 10] | [B, 64, 13, 13] |
| Fusion | - | 无 | [B, 128, 13, 13] → [B, 128, 1, 1] |
| FC Encoder | 输入 | 无 | [B, 128] |
| | 输出 | 无 | LL:[B,22] + HF:[B,10] |
| FC Decoder | 输入 | LL:[B,22] + HF:[B,10] | LL:[B,22] + HF:[B,10] |
| | 输出 | LL:[B,4802] + HF:[B,14406] | LL:[B,128] + HF:[B,128] |
| LL Decoder Net | 输入 | 无 | [B, 128, 1, 1] |
| | 输出 | 无 | [B, 2, 49, 49] |
| HF Decoder Net | 输入 | 无 | [B, 128, 1, 1] |
| | 输出 | 无 | [B, 6, 49, 49] |

---

## MLP vs CNN 架构对比

### MLP版本

**优点**:
- ✅ 参数量适中（~20M）
- ✅ 结构简单，易于理解
- ✅ 全局感受野（Flatten后所有像素连接）

**缺点**:
- ❌ 忽略空间局部性
- ❌ 参数效率较低
- ❌ 对平移不变性建模较弱

**适用场景**:
- 小数据集（<5000样本）
- 参数敏感性分析
- 快速原型验证

### CNN版本

**优点**:
- ✅ 保留空间结构
- ✅ 参数共享，效率更高
- ✅ 平移不变性
- ✅ 渐进式特征提取

**缺点**:
- ❌ 感受野受限（需要多层才能覆盖全局）
- ❌ 结构更复杂

**适用场景**:
- 大数据集（>5000样本）
- 图像类任务
- 生产环境部署

---

## 关键代码映射

### 文件结构

```
autoencoder/
├── models/
│   ├── dual_branch_differentiable_autoencoder_v2.py  ← V2实现
│   │   ├── DualBranchDifferentiableWaveletMLPAutoEncoderV2  (line 78-418)
│   │   └── DualBranchDifferentiableWaveletAutoEncoderV2     (line 424-861)
│   │
│   ├── dual_branch_differentiable_autoencoder.py  ← V1实现（旧版）
│   └── base_autoencoder.py  ← 基类（物理约束）
│
└── utils/
    ├── differentiable_wavelet_transform.py  ← 可微分小波
    │   ├── DifferentiableWaveletTransform (line 15-175)
    │   │   ├── forward_transform()  (line 67-122)
    │   │   └── inverse_transform()  (line 124-175)
    │
    └── dual_branch_autoencoder.py  ← 工具函数
        └── calculate_branch_latent_dims()  (计算ll/hf latent维度)
```

### 核心方法映射（MLP V2）

| 功能 | 方法名 | 行号 | 关键输入/输出 |
|------|--------|------|--------------|
| 通道分离 | `_split_channels()` | 228-251 | [B,49,49,8] → LL:[B,49,49,2] + HF:[B,49,49,6] |
| 通道组合 | `_combine_channels()` | 253-280 | LL:[B,49,49,2] + HF:[B,49,49,6] → [B,49,49,8] |
| 编码 | `encode()` | 282-311 | [B,91,91,2] → [B,32] |
| 解码 | `decode()` | 313-346 | [B,32] → [B,91,91,2] |
| 前向传播 | `forward()` | 348-361 | [B,91,91,2] → ([B,91,91,2], [B,32]) |

### 小波变换映射

| 功能 | 方法名 | 行号 | 说明 |
|------|--------|------|------|
| 前向小波 | `forward_transform()` | 67-122 | 使用ptwt.wavedec2 |
| 逆小波 | `inverse_transform()` | 124-175 | 使用ptwt.waverec2 |

---

## 供Gemini画图的提示词

### 方案1: 高层架构图

```
请绘制一个深度学习模型架构图，包含以下元素：

1. 输入：RCS数据 [B, 91, 91, 2]

2. 可微分小波变换模块（绿色框）
   - 输出：小波系数 [B, 49, 49, 8]

3. 通道分离（菱形，2个输出分支）
   - 左分支：LL通道 [B, 49, 49, 2]（蓝色）
   - 右分支：HF通道 [B, 49, 49, 6]（橙色）

4. 双分支编码器（并行）
   - LL Encoder（蓝色框）：[B, 49, 49, 2] → [B, 22]
   - HF Encoder（橙色框）：[B, 49, 49, 6] → [B, 10]

5. Latent空间（中心紫色圆）
   - Concat：[B, 32]

6. 双分支解码器（并行，对称）
   - LL Decoder（蓝色框）：[B, 22] → [B, 49, 49, 2]
   - HF Decoder（橙色框）：[B, 10] → [B, 49, 49, 6]

7. 通道组合（菱形，2个输入合并）
   - 输出：小波系数 [B, 49, 49, 8]

8. 逆小波变换模块（绿色框）
   - 输出：RCS重建 [B, 91, 91, 2]

9. 损失函数（红色）
   - MSE Loss在RCS空间

10. 梯度流（虚线箭头）
    - 从Loss反向传播到输入

样式要求：
- 使用流程图风格
- LL分支用蓝色，HF分支用橙色
- 小波变换用绿色，损失用红色
- 标注每个模块的输入输出维度
- 显示前向传播（实线箭头）和反向传播（虚线箭头）
```

### 方案2: 详细数据流图（MLP版本）

```
请绘制一个详细的神经网络数据流图，表示双分支MLP AutoEncoder：

编码路径（从上到下）：
1. 输入层：[B, 91, 91, 2]
2. 小波变换：→ [B, 49, 49, 8]
3. 分离通道：
   - LL: [f0_LL, f1_LL] → [B, 49, 49, 2]
   - HF: [f0_LH, f0_HL, f0_HH, f1_LH, f1_HL, f1_HH] → [B, 49, 49, 6]
4. Flatten：
   - LL: [B, 4802]
   - HF: [B, 14406]
5. LL Encoder MLP：
   - Linear(4802 → 512) + ReLU + Dropout
   - Linear(512 → 256) + ReLU + Dropout
   - Linear(256 → 22)
6. HF Encoder MLP（并行）：
   - Linear(14406 → 512) + ReLU + Dropout
   - Linear(512 → 256) + ReLU + Dropout
   - Linear(256 → 10)
7. Concat: [B, 22] + [B, 10] → [B, 32]

解码路径（从中心向上）：
8. Split: [B, 32] → [B, 22] + [B, 10]
9. LL Decoder MLP：
   - Linear(22 → 256) + ReLU + Dropout
   - Linear(256 → 512) + ReLU + Dropout
   - Linear(512 → 4802)
10. HF Decoder MLP（并行）：
    - Linear(10 → 256) + ReLU + Dropout
    - Linear(256 → 512) + ReLU + Dropout
    - Linear(512 → 14406)
11. Reshape：
    - LL: [B, 4802] → [B, 49, 49, 2]
    - HF: [B, 14406] → [B, 49, 49, 6]
12. 组合通道：[B, 49, 49, 2] + [B, 49, 49, 6] → [B, 49, 49, 8]
13. 逆小波变换：→ [B, 91, 91, 2]
14. 输出：[B, 91, 91, 2]

样式要求：
- 每一层用矩形框表示
- 在框内标注操作名称和输出维度
- LL分支用蓝色，HF分支用橙色
- 用箭头连接各层
- 在箭头旁标注张量维度
```

### 方案3: 对比V1 vs V2

```
请绘制一个对比图，展示V1和V2的架构差异：

左侧：V1架构（有缺陷）
- Encoder：双分支（LL + HF）
- Latent：[128 + 128] = 256（固定，忽略ll_ratio）
- Decoder：单分支（fusion → decoder）
- 标注缺陷：
  - ll_latent_dim和hf_latent_dim未生效
  - 缺少_combine_channels()
  - 架构不对称

右侧：V2架构（正确）
- Encoder：双分支（LL + HF）
- Latent：[22 + 10] = 32（动态，ll_ratio=0.7）
- Decoder：双分支（ll_decoder + hf_decoder）
- 标注改进：
  - ll_latent_dim和hf_latent_dim真正起作用
  - 正确的_combine_channels()
  - 架构对称

用箭头标注关键差异点
```

---

## 附录：参数量统计

### MLP V2 (latent_dim=32, ll_ratio=0.7)

| 模块 | 参数量 |
|------|--------|
| LL Encoder | 2,477,078 |
| HF Encoder | 7,427,850 |
| LL Decoder | 2,477,078 |
| HF Decoder | 7,427,850 |
| Wavelet Transform | 0 (无可训练参数) |
| **Total** | **~20.2M** |

### CNN V2 (latent_dim=32, ll_ratio=0.7)

| 模块 | 参数量 |
|------|--------|
| LL Branch | ~50K |
| HF Branch | ~87K |
| Fusion | ~295K |
| LL Encoder FC | ~2.8K |
| HF Encoder FC | ~1.3K |
| LL Decoder FC | ~2.8K |
| HF Decoder FC | ~1.3K |
| LL Decoder Net | ~148K |
| HF Decoder Net | ~250K |
| Wavelet Transform | 0 |
| **Total** | **~2.2M** |

---

**文档结束**

如需画图，可将上述"供Gemini画图的提示词"部分直接复制给绘图工具。
