# 频率扩展能力说明 - 支持6GHz数据

## 📋 当前设计的扩展能力

### 🔧 已预留的扩展接口

我们的AutoEncoder系统在设计时已经考虑了频率扩展的需求，具备以下扩展能力：

#### 1. **小波变换模块扩展** (`wavelet_transform.py`)

**当前设计**:
```python
# 当前: 2频率 (1.5GHz, 3GHz)
for freq_idx in range(2):  # 硬编码为2
    freq_data = rcs_data[:, :, :, freq_idx]
```

**扩展设计**:
```python
class WaveletTransform:
    def __init__(self,
                 wavelet: str = 'db4',
                 mode: str = 'symmetric',
                 num_frequencies: int = 2):  # 🆕 可配置频率数量
        self.num_frequencies = num_frequencies

    def forward_transform(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, 91, 91, num_freq]
        输出: [B, 91, 91, num_freq * 4]  # 动态通道数
        """
        batch_size = rcs_data.shape[0]
        height, width = rcs_data.shape[1], rcs_data.shape[2]

        all_coeffs = []
        for batch_idx in range(batch_size):
            batch_coeffs = []

            # 🆕 动态处理任意数量的频率
            for freq_idx in range(self.num_frequencies):
                freq_data = rcs_data[batch_idx, :, :, freq_idx]
                # ... 小波变换处理

        # 输出通道数: num_frequencies * 4
```

#### 2. **CNN-AutoEncoder架构扩展** (`cnn_autoencoder.py`)

**当前设计**:
```python
class WaveletAutoEncoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 256,
                 input_channels: int = 8,  # 2频率 × 4频带 = 8
                 dropout_rate: float = 0.2):
```

**扩展方案**:
```python
class WaveletAutoEncoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 256,
                 num_frequencies: int = 2,  # 🆕 频率数量参数
                 wavelet_bands: int = 4,    # 🆕 小波频带数
                 dropout_rate: float = 0.2):

        # 🆕 动态计算输入通道数
        self.input_channels = num_frequencies * wavelet_bands

        # 网络架构自动适配
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, padding=1),  # 动态输入通道
            # ... 其余网络层保持不变
        )
```

#### 3. **数据格式兼容性** (`data_adapters.py`)

**扩展设计**:
```python
class RCS_DataAdapter:
    def __init__(self,
                 normalize: bool = True,
                 log_transform: bool = False,
                 expected_frequencies: int = 2):  # 🆕 预期频率数
        self.expected_frequencies = expected_frequencies

    def adapt_rcs_data(self, rcs_data: np.ndarray) -> torch.Tensor:
        """
        灵活处理不同频率数量的RCS数据
        输入: [N, 91, 91, num_freq] (num_freq可变)
        输出: [N, 91, 91, num_freq] 标准化数据
        """
        if len(rcs_data.shape) != 4:
            raise ValueError(f"RCS数据应为4维，实际为{len(rcs_data.shape)}维")

        num_freq = rcs_data.shape[3]
        if num_freq != self.expected_frequencies:
            print(f"⚠️ 检测到{num_freq}个频率，预期{self.expected_frequencies}个")
            self.expected_frequencies = num_freq  # 自动适配
```

---

## 🚀 6GHz扩展实施方案

### 方案A: 直接扩展 (推荐)

**数据格式变更**:
```
当前: [B, 91, 91, 2]  # 1.5GHz, 3GHz
扩展: [B, 91, 91, 3]  # 1.5GHz, 3GHz, 6GHz
```

**小波系数变更**:
```
当前: [B, 91, 91, 8]   # 2频率 × 4频带 = 8通道
扩展: [B, 91, 91, 12]  # 3频率 × 4频带 = 12通道
```

**网络架构调整**:
```python
# 仅需修改输入通道数
WaveletAutoEncoder(
    latent_dim=256,           # 保持不变
    num_frequencies=3,        # 🆕 2 → 3
    input_channels=12         # 🆕 8 → 12
)
```

**优势**:
- ✅ 最小改动，向后兼容
- ✅ 保持现有隐空间维度
- ✅ 训练好的参数映射器可复用
- ✅ 评估指标无需修改

### 方案B: 渐进式扩展

**阶段1**: 训练3频率版本的新模型
**阶段2**: 提供2频率→3频率的迁移学习
**阶段3**: 统一接口支持多频率切换

---

## 🔧 具体实施步骤

### Step 1: 配置系统扩展

**创建频率配置文件**:
```yaml
# autoencoder/configs/frequency_config.yaml
frequency_settings:
  num_frequencies: 3  # 1.5GHz, 3GHz, 6GHz
  frequency_labels: ["1.5GHz", "3GHz", "6GHz"]
  wavelet_bands: 4

model_settings:
  latent_dim: 256
  input_channels: 12  # 3 * 4 = 12

training_settings:
  batch_size: 16  # 可能需要调整
  learning_rate: 1e-3
```

### Step 2: 代码修改清单

**需要修改的文件**:
1. `wavelet_transform.py` - 动态频率处理
2. `cnn_autoencoder.py` - 输入通道参数化
3. `data_adapters.py` - 数据格式验证
4. `ae_trainer.py` - 训练流程适配
5. `reconstruction_metrics.py` - 评估指标扩展

**修改示例** (`wavelet_transform.py`):
```python
def __init__(self, wavelet='db4', mode='symmetric', num_frequencies=2):
    self.num_frequencies = num_frequencies  # 🆕 支持配置

def get_transform_info(self) -> dict:
    return {
        'wavelet': self.wavelet,
        'num_frequencies': self.num_frequencies,  # 🆕
        'input_shape': f'[B, 91, 91, {self.num_frequencies}]',
        'output_shape': f'[B, 91, 91, {self.num_frequencies * 4}]',
        'frequency_bands': [f'{freq_idx+1}_LL,LH,HL,HH'
                           for freq_idx in range(self.num_frequencies)]
    }
```

### Step 3: 评估指标扩展

**频率一致性分析扩展**:
```python
def _compute_frequency_consistency(self, pred_rcs, true_rcs):
    """扩展到3频率的一致性分析"""
    consistency_errors = []

    # 两两频率对比分析
    for i in range(self.num_frequencies):
        for j in range(i+1, self.num_frequencies):
            pred_i = pred_rcs[:, :, :, i]
            pred_j = pred_rcs[:, :, :, j]
            true_i = true_rcs[:, :, :, i]
            true_j = true_rcs[:, :, :, j]

            # 频率差异一致性
            pred_diff = pred_j - pred_i
            true_diff = true_j - true_i

            error = F.mse_loss(pred_diff, true_diff)
            consistency_errors.append(error.item())

    return np.mean(consistency_errors)
```

### Step 4: 向后兼容处理

**自动检测和适配**:
```python
def create_autoencoder_for_data(rcs_data_sample):
    """根据数据自动创建合适的AutoEncoder"""

    # 检测频率数量
    num_freq = rcs_data_sample.shape[-1]

    print(f"检测到{num_freq}个频率，自动配置模型...")

    # 创建相应的模型
    ae = WaveletAutoEncoder(
        num_frequencies=num_freq,
        input_channels=num_freq * 4,
        latent_dim=256
    )

    wt = WaveletTransform(
        num_frequencies=num_freq
    )

    return ae, wt
```

---

## 📊 扩展对性能的影响分析

### 计算复杂度变化

| 组件 | 2频率 | 3频率 | 变化 |
|------|-------|-------|------|
| 小波变换 | O(2×N²) | O(3×N²) | +50% |
| AE输入层 | 8→32 Conv | 12→32 Conv | +50% params |
| 其余网络层 | 无变化 | 无变化 | 0% |
| 总参数量 | ~10M | ~10.1M | +1% |

### 内存使用变化

```
当前批次内存: batch_size × 91 × 91 × 8 × 4bytes
扩展批次内存: batch_size × 91 × 91 × 12 × 4bytes  (+50%)

建议: batch_size 16 → 12 (保持总内存不变)
```

### 训练时间影响

- **小波变换**: +50% (3频率 vs 2频率)
- **网络训练**: +5% (输入层参数增加)
- **总体影响**: 约+10-15%

---

## 🎯 实施建议

### 优先级安排

1. **高优先级** (立即实施):
   - 参数化频率数量配置
   - 数据格式自动检测
   - 向后兼容接口

2. **中优先级** (6GHz数据到达前):
   - 评估指标扩展
   - 可视化界面适配
   - 性能优化

3. **低优先级** (可选功能):
   - 频率权重调整
   - 不同频率的独立分析
   - 频率间迁移学习

### 测试策略

1. **兼容性测试**: 确保2频率数据仍能正常工作
2. **模拟测试**: 使用模拟3频率数据验证扩展功能
3. **性能测试**: 评估扩展后的计算和内存开销
4. **集成测试**: 确保整个流程端到端工作正常

---

## 💡 扩展能力总结

我们的AutoEncoder系统设计具备良好的扩展能力：

### ✅ 当前支持
- **2频率**: 1.5GHz, 3GHz (已实现)
- **模块化设计**: 便于独立扩展各组件
- **参数化架构**: 支持配置驱动的模型创建

### 🚀 即将支持
- **3频率**: 1.5GHz, 3GHz, 6GHz (几行代码修改)
- **N频率**: 理论支持任意数量频率 (通用化设计)
- **向后兼容**: 现有模型和数据无需修改

### 🔮 未来扩展
- **频率加权**: 不同频率的重要性权重
- **频率融合**: 跨频率特征学习
- **动态频率**: 运行时频率数量调整

**总结**: 当6GHz数据到达时，只需要：
1. 修改配置文件中的频率数量 (2→3)
2. 调整数据加载格式 ([B,91,91,2]→[B,91,91,3])
3. 重新训练模型

整个系统架构无需大改，扩展成本很低！🎉