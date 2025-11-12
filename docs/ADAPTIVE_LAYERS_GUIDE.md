# AutoEncoder动态适配模块使用指南

> **目标**: 让所有AutoEncoder模型支持16-32维小隐空间，避免信息瓶颈
> **核心原理**: 动态生成中间层，保持每级压缩比≤4:1

---

## 📋 模块功能说明

### 位置
`autoencoder/utils/adaptive_layers.py`

### 核心函数

#### 1. `calculate_intermediate_dims(input_dim, latent_dim, max_ratio=4, min_layers=0)`

**功能**: 自动计算中间层维度列表

**参数**:
- `input_dim`: 输入维度（展平后的特征维度）
- `latent_dim`: 目标隐空间维度
- `max_ratio`: 每级最大压缩比（默认4，即4:1）
- `min_layers`: 最少中间层数（MLP通常设为3）

**返回**: 中间层维度列表 `List[int]`

**示例**:
```python
>>> calculate_intermediate_dims(4096, 32, max_ratio=4)
[1024, 256, 64]
# 完整路径: 4096 → 1024 (4:1) → 256 (4:1) → 64 (4:1) → 32 (2:1)
```

**算法逻辑**:
1. 从`input_dim`开始，每次除以`max_ratio`（最多）
2. 将结果向上取整到2的幂次（64, 128, 256, 512...）
3. 重复直到接近`latent_dim`
4. 如果指定了`min_layers`，确保至少有那么多中间层

---

#### 2. `build_adaptive_fc_pair(input_dim, latent_dim, dropout_rate=0.2, max_ratio=4, min_layers=0, arch_type='cnn')`

**功能**: 一键生成自适应的Encoder-Decoder全连接层对

**参数**:
- `input_dim`: 输入维度
- `latent_dim`: 隐空间维度
- `dropout_rate`: Dropout比率
- `max_ratio`: 每级最大压缩比
- `min_layers`: 最少中间层数
- `arch_type`: 架构类型
  - `'cnn'`: CNN架构，自动决定层数
  - `'mlp'`: MLP架构，默认至少3个中间层

**返回**: `(encoder_fc, decoder_fc, intermediate_dims)`
- `encoder_fc`: nn.Sequential，编码器全连接部分
- `decoder_fc`: nn.Sequential，解码器全连接部分
- `intermediate_dims`: List[int]，中间层维度列表

**示例**:
```python
# CNN模式
encoder_fc, decoder_fc, dims = build_adaptive_fc_pair(
    input_dim=4096,
    latent_dim=32,
    dropout_rate=0.2,
    arch_type='cnn'
)
# dims = [1024, 256, 64]
# encoder_fc: 4096 → 1024 → 256 → 64 → 32

# MLP模式（自动至少3个中间层）
encoder_fc, decoder_fc, dims = build_adaptive_fc_pair(
    input_dim=19208,
    latent_dim=32,
    dropout_rate=0.2,
    arch_type='mlp'
)
# dims = [4096, 1024, 256, 64]（至少3层）
```

---

#### 3. `get_structure_info(input_dim, latent_dim, intermediate_dims)`

**功能**: 生成网络结构信息字典（用于`get_model_info()`）

**返回**: 包含以下字段的字典
```python
{
    'fc_structure': '4096 → 1024 → 256 → 64 → 32',
    'intermediate_dims': [1024, 256, 64],
    'num_fc_layers': 4,
    'compression_ratios': ['4096:1024 = 4.0:1', '1024:256 = 4.0:1', ...]
}
```

---

## 🔧 适配方法

### 方法A: 完全替换（推荐用于CNN）

**适用场景**: 模型的全连接部分是固定结构，需要完全替换

**步骤**:

#### Step 1: 导入模块
```python
from autoencoder.utils.adaptive_layers import build_adaptive_fc_pair, get_structure_info
```

#### Step 2: 替换`__init__`中的全连接层定义

**修改前**:
```python
# 固定的全连接层
self.encoder_fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, latent_dim)
)

self.decoder_fc = nn.Sequential(
    nn.Linear(latent_dim, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 4096),
    nn.ReLU()
)
```

**修改后**:
```python
# 动态生成全连接层
self.encoder_fc, self.decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
    input_dim=4096,          # 卷积输出展平后的维度
    latent_dim=latent_dim,
    dropout_rate=dropout_rate,
    arch_type='cnn'          # 或 'mlp'
)
```

#### Step 3: 更新`get_model_info()`方法

**修改前**:
```python
def get_model_info(self):
    return {
        'model_name': 'MyAutoEncoder',
        'latent_dim': self.latent_dim,
        # ...
    }
```

**修改后**:
```python
def get_model_info(self):
    # 获取结构信息
    structure_info = get_structure_info(
        input_dim=4096,  # 或self.flattened_size
        latent_dim=self.latent_dim,
        intermediate_dims=self.intermediate_dims
    )

    return {
        'model_name': 'MyAutoEncoder',
        'architecture': 'CNN',  # 或 'MLP'
        'latent_dim': self.latent_dim,
        'fc_structure': structure_info['fc_structure'],
        'intermediate_dims': structure_info['intermediate_dims'],
        'num_fc_layers': structure_info['num_fc_layers'],
        # ... 其他信息
    }
```

---

### 方法B: 部分替换（用于复杂模型）

**适用场景**: 模型有自定义的encoder/decoder结构，只替换全连接瓶颈部分

#### Step 1: 使用`calculate_intermediate_dims`计算维度
```python
from autoencoder.utils.adaptive_layers import calculate_intermediate_dims

# 在__init__中
self.intermediate_dims = calculate_intermediate_dims(
    input_dim=4096,
    latent_dim=latent_dim,
    max_ratio=4
)
```

#### Step 2: 手动构建encoder_fc
```python
encoder_fc_layers = [nn.Flatten()]
current_dim = 4096

for intermediate_dim in self.intermediate_dims:
    encoder_fc_layers.extend([
        nn.Linear(current_dim, intermediate_dim),
        nn.BatchNorm1d(intermediate_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate)
    ])
    current_dim = intermediate_dim

encoder_fc_layers.append(nn.Linear(current_dim, latent_dim))
self.encoder_fc = nn.Sequential(*encoder_fc_layers)
```

#### Step 3: 对称构建decoder_fc
```python
decoder_fc_layers = []
current_dim = latent_dim

for intermediate_dim in reversed(self.intermediate_dims):
    decoder_fc_layers.extend([
        nn.Linear(current_dim, intermediate_dim),
        nn.BatchNorm1d(intermediate_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate)
    ])
    current_dim = intermediate_dim

decoder_fc_layers.extend([
    nn.Linear(current_dim, 4096),
    nn.ReLU(inplace=True)
])
self.decoder_fc = nn.Sequential(*decoder_fc_layers)
```

---

## 📝 完整示例

### 示例1: CNN模型适配

```python
"""
修改前: 固定256维隐空间的CNN模型
修改后: 支持动态隐空间（16-256维）
"""

import torch.nn as nn
from autoencoder.utils.adaptive_layers import build_adaptive_fc_pair, get_structure_info

class MyCNNAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256, dropout_rate=0.2):
        super().__init__()

        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # 卷积编码器（不变）
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.ReLU(),
            # ... 其他层
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 计算展平维度
        self.flattened_size = 256 * 4 * 4  # 4096

        # 【修改点1】使用动态全连接层
        self.encoder_fc, self.decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
            input_dim=self.flattened_size,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            arch_type='cnn'
        )

        # 卷积解码器（不变）
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            # ... 其他层
        )

    def encode(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.encoder_conv(x)
        latent = self.encoder_fc(features)
        return latent

    def decode(self, latent):
        features = self.decoder_fc(latent)
        x_recon = self.decoder_conv(features)
        # ... 上采样等
        return x_recon

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    # 【修改点2】更新get_model_info
    def get_model_info(self):
        structure_info = get_structure_info(
            self.flattened_size,
            self.latent_dim,
            self.intermediate_dims
        )

        return {
            'model_name': 'MyCNNAutoEncoder',
            'architecture': 'CNN',
            'latent_dim': self.latent_dim,
            'fc_structure': structure_info['fc_structure'],
            'intermediate_dims': structure_info['intermediate_dims'],
            'num_fc_layers': structure_info['num_fc_layers'],
            'dropout_rate': self.dropout_rate,
        }
```

### 示例2: MLP模型适配

```python
from autoencoder.utils.adaptive_layers import build_adaptive_fc_pair, get_structure_info

class MyMLPAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256, dropout_rate=0.2):
        super().__init__()

        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.input_dim = 49 * 49 * 8  # 19208

        # 【修改点】MLP架构，自动至少3个中间层
        self.encoder, self.decoder, self.intermediate_dims = build_adaptive_fc_pair(
            input_dim=self.input_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            arch_type='mlp'  # MLP模式
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def get_model_info(self):
        structure_info = get_structure_info(
            self.input_dim,
            self.latent_dim,
            self.intermediate_dims
        )

        return {
            'model_name': 'MyMLPAutoEncoder',
            'architecture': 'MLP',
            'latent_dim': self.latent_dim,
            'mlp_structure': structure_info['fc_structure'],  # 注意：MLP用mlp_structure
            'intermediate_dims': structure_info['intermediate_dims'],
            'num_mlp_layers': structure_info['num_fc_layers'],
            'dropout_rate': self.dropout_rate,
        }
```

---

## ⚠️ 重要注意事项

### 1. 找到正确的`input_dim`

不同模型的`input_dim`（进入全连接层前的维度）不同：

| 模型类型 | input_dim | 说明 |
|---------|-----------|------|
| WaveletAutoEncoder | 256 × 4 × 4 = 4096 | 卷积后AdaptiveAvgPool2d((4,4)) |
| DirectAutoEncoder | 512 × 1 × 1 = 512 | 卷积后AdaptiveAvgPool2d(1) |
| WaveletMLPAutoEncoder | 49 × 49 × 8 = 19208 | 输入直接展平 |
| DirectMLPAutoEncoder | 91 × 91 × 2 = 16562 | 输入直接展平 |
| EnhancedCNN | 可能不同 | 需要查看代码 |
| DeepCNN | 可能不同 | 需要查看代码 |

**如何确定**: 查看原始代码中`self.encoder`最后一层的输出维度。

### 2. 保留`self.intermediate_dims`

**必须**将`intermediate_dims`保存为实例变量：
```python
self.encoder_fc, self.decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(...)
```

因为`get_model_info()`需要使用它。

### 3. get_model_info的字段名

根据架构类型使用不同的字段名：

**CNN架构**:
```python
{
    'fc_structure': '...',      # ← 用fc_
    'num_fc_layers': 4,
}
```

**MLP架构**:
```python
{
    'mlp_structure': '...',     # ← 用mlp_
    'num_mlp_layers': 5,
}
```

这样`frequency_config.py`才能正确显示。

### 4. 不要忘记更新`__all__`和导入

如果模型文件导出了工厂函数，确保也更新它们的默认参数：
```python
def create_my_autoencoder(latent_dim: int = 32, ...):  # 256 → 32
    # ...
```

---

## 🎯 待适配模型清单

| # | 模型类 | 文件 | 类型 | input_dim |
|---|--------|------|------|-----------|
| 1 | EnhancedWaveletAutoEncoder | enhanced_cnn_autoencoder.py | CNN | 需查看 |
| 2 | EnhancedDirectAutoEncoder | enhanced_cnn_autoencoder.py | CNN | 需查看 |
| 3 | DeepWaveletAutoEncoder | deep_autoencoder.py | CNN | 需查看 |
| 4 | DeepDirectAutoEncoder | deep_autoencoder.py | CNN | 需查看 |

**注**: ~~SinWaveletAutoEncoder, SinDirectAutoEncoder, SinWaveletMLPAutoEncoder, SinDirectMLPAutoEncoder~~ 已移除。
现在通过参数化激活函数实现：
- `WaveletAutoEncoder(activation='sin')` 替代 `SinWaveletAutoEncoder`
- `DirectAutoEncoder(activation='sin')` 替代 `SinDirectAutoEncoder`
- `WaveletMLPAutoEncoder(activation='sin')` 替代 `SinWaveletMLPAutoEncoder`
- `DirectMLPAutoEncoder(activation='sin')` 替代 `SinDirectMLPAutoEncoder`

---

## 📊 验证方法

适配完成后，运行以下代码验证：

```python
from autoencoder.utils.frequency_config import create_autoencoder_system

# 测试小隐空间
system = create_autoencoder_system(
    '2freq',
    latent_dim=32,  # 或16
    mode='wavelet',
    architecture='enhanced_cnn'  # 或其他架构
)

# 检查输出
model_info = system['autoencoder'].get_model_info()
print(f"FC结构: {model_info['fc_structure']}")
print(f"中间层: {model_info['intermediate_dims']}")

# 测试前向传播
import torch
test_input = torch.randn(2, 49, 49, 8)  # 或其他尺寸
recon, latent = system['autoencoder'](test_input)
print(f"隐空间形状: {latent.shape}")  # 应该是 [2, 32]
```

---

## 🔗 参考已完成的模型

可以参考这些已经完成适配的模型：

1. **WaveletAutoEncoder** (`cnn_autoencoder.py` line 120-167)
   - 方法A完全替换示例
   - CNN架构

2. **DirectAutoEncoder** (`direct_autoencoder.py` line 123-168)
   - 方法A完全替换示例
   - CNN架构

3. **WaveletMLPAutoEncoder** (`mlp_autoencoder.py` line 113-156)
   - 方法B部分替换示例
   - MLP架构

4. **DirectMLPAutoEncoder** (`mlp_autoencoder.py` line 302-345)
   - 方法B部分替换示例
   - MLP架构

---

生成时间: 2025-01-18
作者: Claude Code
