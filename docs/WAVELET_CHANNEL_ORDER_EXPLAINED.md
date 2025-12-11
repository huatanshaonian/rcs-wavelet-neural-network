# 小波系数通道顺序详解

## 📊 下标含义

在 `[LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]` 中：

### 下标表示**频率索引**

对于**2频率系统**（2freq）：
- **下标0** = 第1个频率（1.5GHz）
- **下标1** = 第2个频率（3.0GHz）

对于**3频率系统**（3freq）：
- **下标0** = 第1个频率（1.5GHz）
- **下标1** = 第2个频率（3.0GHz）
- **下标2** = 第3个频率（6.0GHz）

---

## 🔍 完整示例：2频率系统

### 输入数据

```python
# RCS数据
rcs_data.shape = [B, 91, 91, 2]
                              ↑
                              2个频率

# 频率配置
frequencies = [1.5GHz, 3.0GHz]
               ↑        ↑
            freq_0   freq_1
```

### 小波变换后

```python
# 小波系数
wavelet_coeffs.shape = [B, 49, 49, 8]
                                   ↑
                                   8个通道

# 通道布局（最后一个维度的8个通道）
通道索引:  0    1    2    3    4    5    6    7
         LL₀  LH₀  HL₀  HH₀  LL₁  LH₁  HL₁  HH₁
          ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
        freq0的4个小波频带  freq1的4个小波频带
```

### 详细说明

**频率0（1.5GHz）的4个小波频带**：
```
通道0: LL₀ = freq0的低频-低频分量（最重要，包含90%+能量）
通道1: LH₀ = freq0的低频-高频分量（水平细节）
通道2: HL₀ = freq0的高频-低频分量（垂直细节）
通道3: HH₀ = freq0的高频-高频分量（对角细节）
```

**频率1（3.0GHz）的4个小波频带**：
```
通道4: LL₁ = freq1的低频-低频分量
通道5: LH₁ = freq1的低频-高频分量
通道6: HL₁ = freq1的高频-低频分量
通道7: HH₁ = freq1的高频-高频分量
```

---

## 📐 具体数值示例

假设某个位置 `[batch=0, i=10, j=10]` 的小波系数值：

```python
wavelet_coeffs[0, 10, 10, :] = [0.8, 0.1, 0.05, 0.03, 0.75, 0.12, 0.06, 0.04]
                                 ↑    ↑    ↑     ↑     ↑     ↑     ↑     ↑
                                LL₀  LH₀  HL₀   HH₀   LL₁   LH₁   HL₁   HH₁
```

**解读**：
- `LL₀ = 0.8` → 1.5GHz的低频分量（主要能量）
- `LH₀ = 0.1` → 1.5GHz的水平细节
- `HL₀ = 0.05` → 1.5GHz的垂直细节
- `HH₀ = 0.03` → 1.5GHz的对角细节
- `LL₁ = 0.75` → 3.0GHz的低频分量
- `LH₁ = 0.12` → 3.0GHz的水平细节
- `HL₁ = 0.06` → 3.0GHz的垂直细节
- `HH₁ = 0.04` → 3.0GHz的对角细节

---

## 🎯 Dual-Branch的分离逻辑

### 目标

将8个通道分成两组：
- **LL组**：`[LL₀, LL₁]` → 2个通道 → flatten成4802维
- **HF组**：`[LH₀, HL₀, HH₀, LH₁, HL₁, HH₁]` → 6个通道 → flatten成14406维

### 代码实现

```python
def _split_channels(self, wavelet_coeffs):
    """
    wavelet_coeffs: [B, 49, 49, 8]
    通道顺序: [LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]
    """
    ll_list = []
    hf_list = []

    for freq_idx in range(self.num_frequencies):  # freq_idx = 0, 1
        base = freq_idx * 4  # base = 0, 4

        # 提取该频率的LL和HF
        ll_list.append(wavelet_coeffs[:, :, :, base:base+1])       # LL
        hf_list.append(wavelet_coeffs[:, :, :, base+1:base+4])     # LH, HL, HH

    # Concat
    ll_channels = torch.cat(ll_list, dim=3)  # [B, 49, 49, 2]
    hf_channels = torch.cat(hf_list, dim=3)  # [B, 49, 49, 6]

    return ll_channels, hf_channels
```

### 分离过程详解

**第一次循环** (`freq_idx=0`, `base=0`):
```python
ll_list.append(wavelet_coeffs[:, :, :, 0:1])    # 提取通道0 → LL₀
hf_list.append(wavelet_coeffs[:, :, :, 1:4])    # 提取通道1,2,3 → LH₀, HL₀, HH₀
```

**第二次循环** (`freq_idx=1`, `base=4`):
```python
ll_list.append(wavelet_coeffs[:, :, :, 4:5])    # 提取通道4 → LL₁
hf_list.append(wavelet_coeffs[:, :, :, 5:8])    # 提取通道5,6,7 → LH₁, HL₁, HH₁
```

**Concat后**:
```python
ll_channels.shape = [B, 49, 49, 2]
# 通道顺序: [LL₀, LL₁]

hf_channels.shape = [B, 49, 49, 6]
# 通道顺序: [LH₀, HL₀, HH₀, LH₁, HL₁, HH₁]
```

---

## 🔄 Decoder的组合逻辑

### 输入

```python
ll_channels: [B, 49, 49, 2]  顺序: [LL₀, LL₁]
hf_channels: [B, 49, 49, 6]  顺序: [LH₀, HL₀, HH₀, LH₁, HL₁, HH₁]
```

### 目标

组合回原始的8通道交错格式：
```
[LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]
```

### 代码实现

```python
def _combine_channels(self, ll_channels, hf_channels):
    """
    ll_channels: [B, 49, 49, 2]  顺序: [LL₀, LL₁]
    hf_channels: [B, 49, 49, 6]  顺序: [LH₀, HL₀, HH₀, LH₁, HL₁, HH₁]
    """
    wavelet_coeffs_list = []

    for freq_idx in range(self.num_frequencies):  # freq_idx = 0, 1
        # 提取该频率的LL和HF
        ll = ll_channels[:, :, :, freq_idx:freq_idx+1]          # [B, 49, 49, 1]
        hf = hf_channels[:, :, :, freq_idx*3:(freq_idx+1)*3]    # [B, 49, 49, 3]

        # 拼接为 [LL, LH, HL, HH]
        freq_coeffs = torch.cat([ll, hf], dim=3)  # [B, 49, 49, 4]
        wavelet_coeffs_list.append(freq_coeffs)

    # 拼接所有频率
    wavelet_coeffs = torch.cat(wavelet_coeffs_list, dim=3)  # [B, 49, 49, 8]
    return wavelet_coeffs
```

### 组合过程详解

**第一次循环** (`freq_idx=0`):
```python
ll = ll_channels[:, :, :, 0:1]     # 提取LL₀ → [B, 49, 49, 1]
hf = hf_channels[:, :, :, 0:3]     # 提取LH₀,HL₀,HH₀ → [B, 49, 49, 3]

freq_coeffs = torch.cat([ll, hf], dim=3)  # [B, 49, 49, 4]
# 通道顺序: [LL₀, LH₀, HL₀, HH₀]
```

**第二次循环** (`freq_idx=1`):
```python
ll = ll_channels[:, :, :, 1:2]     # 提取LL₁ → [B, 49, 49, 1]
hf = hf_channels[:, :, :, 3:6]     # 提取LH₁,HL₁,HH₁ → [B, 49, 49, 3]

freq_coeffs = torch.cat([ll, hf], dim=3)  # [B, 49, 49, 4]
# 通道顺序: [LL₁, LH₁, HL₁, HH₁]
```

**最终Concat**:
```python
wavelet_coeffs = torch.cat([
    [LL₀, LH₀, HL₀, HH₀],  # freq_idx=0的结果
    [LL₁, LH₁, HL₁, HH₁]   # freq_idx=1的结果
], dim=3)

# 最终通道顺序: [LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]
```

---

## 📊 维度变化总结

### 2频率系统 (num_frequencies=2)

```
RCS数据
[B, 91, 91, 2]
    ↓ 小波变换
小波系数 (8通道交错)
[B, 49, 49, 8]: [LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]
    ↓ _split_channels
LL通道: [B, 49, 49, 2]: [LL₀, LL₁]
HF通道: [B, 49, 49, 6]: [LH₀, HL₀, HH₀, LH₁, HL₁, HH₁]
    ↓ flatten
LL: [B, 4802]  (49×49×2)
HF: [B, 14406] (49×49×6)
    ↓ 双分支MLP
LL latent: [B, 22]  (假设ll_ratio=0.7, latent_dim=32)
HF latent: [B, 10]
    ↓ concat
Latent: [B, 32]
    ↓ split
LL latent: [B, 22]
HF latent: [B, 10]
    ↓ 双分支MLP
LL: [B, 4802]
HF: [B, 14406]
    ↓ reshape
LL通道: [B, 49, 49, 2]: [LL₀, LL₁]
HF通道: [B, 49, 49, 6]: [LH₀, HL₀, HH₀, LH₁, HL₁, HH₁]
    ↓ _combine_channels
小波系数 (8通道交错)
[B, 49, 49, 8]: [LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]
    ↓ 逆小波变换
RCS数据
[B, 91, 91, 2]
```

---

## ⚠️ 旧实现的问题

### 问题：缺少_combine_channels

**旧实现**:
```python
def decode(self, latent):
    x = self.decoder_fc(latent)  # [B, 19208]
    wavelet_coeffs = x.view(batch_size, 49, 49, 8)  # ❌ 直接reshape
    # 问题：这8个通道的顺序是网络学到的，不保证是正确顺序！
```

**可能的错误顺序**:
```
情况1: [LL₀, LL₁, LL₀, LL₁, ..., LH₀, HL₀, HH₀, ...] (LL全在前)
情况2: [LH₀, HL₀, HH₀, ..., LL₀, LL₁, ...] (HF全在前)
情况3: 完全随机
```

**期望的顺序**:
```
[LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]
```

**如果顺序错误，逆小波变换会失败！**

---

## ✅ 正确实现

**新实现**:
```python
def decode(self, latent):
    # Split
    ll_latent = latent[:, :self.ll_latent_dim]
    hf_latent = latent[:, self.ll_latent_dim:]

    # 双分支解码
    ll_flat = self.ll_decoder(ll_latent)  # [B, 4802]
    hf_flat = self.hf_decoder(hf_latent)  # [B, 14406]

    # Reshape
    ll_channels = ll_flat.view(B, 49, 49, 2)  # [LL₀, LL₁]
    hf_channels = hf_flat.view(B, 49, 49, 6)  # [LH₀, HL₀, HH₀, LH₁, HL₁, HH₁]

    # ✅ 正确组合
    wavelet_coeffs = self._combine_channels(ll_channels, hf_channels)
    # 保证顺序: [LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]

    # 逆小波变换
    rcs_data = self.wavelet_transform.inverse_transform(wavelet_coeffs)
    return rcs_data
```

---

## 🎓 总结

1. **下标0, 1, 2...** = **频率索引** (frequency index)
   - 0 = 1.5GHz
   - 1 = 3.0GHz
   - 2 = 6.0GHz (仅3freq)

2. **LL, LH, HL, HH** = **小波频带** (wavelet subbands)
   - LL = 低频-低频（主要能量）
   - LH = 低频-高频（水平细节）
   - HL = 高频-低频（垂直细节）
   - HH = 高频-高频（对角细节）

3. **通道顺序至关重要**
   - 正确顺序：`[LL₀, LH₀, HL₀, HH₀, LL₁, LH₁, HL₁, HH₁]`
   - 必须用`_combine_channels()`确保顺序正确
   - 否则逆小波变换会失败或产生错误结果
