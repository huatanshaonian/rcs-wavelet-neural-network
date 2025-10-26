# Codex任务：为剩余8个AutoEncoder模型添加动态隐空间适配

## 📋 任务概述

需要为以下8个AutoEncoder模型添加动态隐空间适配功能，使它们能够支持16-32维的小隐空间。

**已完成的模型**（可作为参考）:
- ✅ WaveletAutoEncoder (`cnn_autoencoder.py`)
- ✅ DirectAutoEncoder (`direct_autoencoder.py`)
- ✅ WaveletMLPAutoEncoder (`mlp_autoencoder.py`)
- ✅ DirectMLPAutoEncoder (`mlp_autoencoder.py`)

**待完成的8个模型**:

| # | 模型类 | 文件路径 | 架构类型 |
|---|--------|---------|---------|
| 1 | EnhancedWaveletAutoEncoder | `autoencoder/models/enhanced_cnn_autoencoder.py` | CNN |
| 2 | EnhancedDirectAutoEncoder | `autoencoder/models/enhanced_cnn_autoencoder.py` | CNN |
| 3 | DeepWaveletAutoEncoder | `autoencoder/models/deep_autoencoder.py` | CNN |
| 4 | DeepDirectAutoEncoder | `autoencoder/models/deep_autoencoder.py` | CNN |
| 5 | SinWaveletAutoEncoder | `autoencoder/models/sine_cnn_autoencoder.py` | CNN |
| 6 | SinDirectAutoEncoder | `autoencoder/models/sine_cnn_autoencoder.py` | CNN |
| 7 | SinWaveletMLPAutoEncoder | `autoencoder/models/sine_mlp_autoencoder.py` | MLP |
| 8 | SinDirectMLPAutoEncoder | `autoencoder/models/sine_mlp_autoencoder.py` | MLP |

---

## 🎯 核心目标

**问题**: 原有模型为256维隐空间设计，直接改为32维会导致信息瓶颈（如4096→32是128:1的极端压缩）

**解决方案**: 使用`autoencoder/utils/adaptive_layers.py`模块，自动生成中间层，保持每级压缩比≤4:1

**示例效果**:
```
原始: 4096 → 256 (16:1压缩，太激进)
优化: 4096 → 1024 → 256 → 64 → 32 (每级4:1，平滑压缩)
```

---

## 📚 必读文档

请先仔细阅读：
1. **ADAPTIVE_LAYERS_GUIDE.md** - 完整的使用指南和示例
2. **已完成的4个模型代码** - 作为参考模板

---

## 🔧 具体修改步骤

### Step 1: 导入adaptive_layers模块

在文件开头添加导入：

```python
from autoencoder.utils.adaptive_layers import build_adaptive_fc_pair, get_structure_info
```

### Step 2: 确定每个模型的`input_dim`

**关键**: 找到全连接层之前的特征维度

**方法**:
1. 查看模型的`__init__`方法
2. 找到encoder的卷积部分最后的输出
3. 通常是 `channels × height × width`

**参考值**（需验证）:
- Wavelet系列（输入49×49）: 可能是 256×4×4 = 4096
- Direct系列（输入91×91）: 可能是 512×1×1 = 512
- MLP系列:
  - Wavelet: 49×49×8 = 19208
  - Direct: 91×91×2 = 16562

**验证方法**: 查看原始代码中类似这样的行：
```python
self.flattened_size = 256 * 4 * 4  # ← 这就是input_dim
```

### Step 3: 替换全连接层定义

#### 对于CNN模型（6个）

**查找原始代码中的全连接层部分**（通常在encoder和decoder中）:

可能是这样的：
```python
# 原始代码（需要替换）
self.encoder_fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_dim, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout_rate),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout_rate),
    nn.Linear(512, latent_dim)
)

self.decoder_fc = nn.Sequential(
    nn.Linear(latent_dim, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout_rate),
    nn.Linear(512, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout_rate),
    nn.Linear(1024, input_dim),
    nn.ReLU(inplace=True)
)
```

**替换为**:
```python
# 动态适配代码
self.encoder_fc, self.decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
    input_dim=self.flattened_size,  # 或具体的数字，如4096
    latent_dim=latent_dim,
    dropout_rate=dropout_rate,
    arch_type='cnn'
)
```

#### 对于MLP模型（2个）

**原始代码**可能有多层全连接：

```python
# 原始代码（需要替换）
self.encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_dim, 8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout_rate),
    nn.Linear(8192, 4096),
    # ... 更多层
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout_rate),
    nn.Linear(512, latent_dim)
)

self.decoder = nn.Sequential(
    # 对称的decoder结构
)
```

**替换为**:
```python
# 动态适配代码
self.encoder, self.decoder, self.intermediate_dims = build_adaptive_fc_pair(
    input_dim=self.input_dim,  # MLP通常是输入直接展平的维度
    latent_dim=latent_dim,
    dropout_rate=dropout_rate,
    arch_type='mlp'  # ← 注意这里是'mlp'
)
```

### Step 4: 更新`get_model_info()`方法

**找到模型的`get_model_info()`方法**，添加结构信息。

#### 对于CNN模型:

```python
def get_model_info(self) -> Dict[str, Any]:
    """获取模型详细信息"""
    param_count = self.get_parameter_count()

    # 【新增】获取结构信息
    structure_info = get_structure_info(
        self.flattened_size,  # 或具体的input_dim
        self.latent_dim,
        self.intermediate_dims
    )

    # 原有的返回值基础上添加这些字段
    return {
        'model_name': 'EnhancedWaveletAutoEncoder',  # 保持原名
        'architecture': 'CNN',  # ← 新增或更新
        'latent_dim': self.latent_dim,
        # ... 其他原有字段 ...

        # 【新增】这些字段
        'fc_structure': structure_info['fc_structure'],
        'intermediate_dims': structure_info['intermediate_dims'],
        'num_fc_layers': structure_info['num_fc_layers'],
        'compression_ratio': f'{input_size}:{self.latent_dim} = {(input_size/self.latent_dim):.1f}:1',

        'parameters': param_count,
        'dropout_rate': self.dropout_rate,
    }
```

#### 对于MLP模型:

```python
def get_model_info(self) -> Dict[str, Any]:
    """获取模型详细信息"""
    param_count = self.get_parameter_count()

    # 【新增】获取结构信息
    structure_info = get_structure_info(
        self.input_dim,
        self.latent_dim,
        self.intermediate_dims
    )

    return {
        'model_name': 'SinWaveletMLPAutoEncoder',  # 保持原名
        'architecture': 'MLP',  # ← 新增或更新
        'latent_dim': self.latent_dim,
        # ... 其他原有字段 ...

        # 【新增】这些字段（注意MLP用mlp_前缀）
        'mlp_structure': structure_info['fc_structure'],  # ← 注意这里
        'intermediate_dims': structure_info['intermediate_dims'],
        'num_mlp_layers': structure_info['num_fc_layers'],  # ← 注意这里
        'compression_ratio': f'{input_size}:{self.latent_dim} = {(input_size/self.latent_dim):.1f}:1',

        'parameters': param_count,
        'dropout_rate': self.dropout_rate,
    }
```

### Step 5: 检查encoder/decoder属性

**重要**: 确保模型有`self.encoder`和`self.decoder`属性（用于训练时的冻结/解冻）

如果模型使用了`build_adaptive_fc_pair`，需要保留原有的`self.encoder`和`self.decoder`引用：

```python
# 对于CNN模型，可能需要这样组织：
self.encoder_conv = nn.Sequential(...)  # 卷积部分
self.encoder_fc = ...  # 全连接部分（来自build_adaptive_fc_pair）

# 保持encoder属性指向卷积部分
self.encoder = self.encoder_conv

# 或者使用ModuleList组合
self.encoder = nn.ModuleList([self.encoder_conv, self.encoder_fc])
```

**验证**: 查看原始代码是否已有`self.encoder`和`self.decoder`的定义。如果有，保持不变；如果没有，参考已完成的模型添加。

---

## ⚠️ 重要注意事项

### 1. 不要改变模型的其他部分

**只修改全连接层部分**，保持以下部分不变：
- ✅ 卷积层结构
- ✅ encode()/decode()/forward()方法的逻辑
- ✅ 权重初始化方法
- ✅ get_parameter_count()方法

### 2. 保留`self.intermediate_dims`

必须保存为实例变量：
```python
self.encoder_fc, self.decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(...)
                                    # ↑ 保存这个变量
```

### 3. 字段命名规范

| 架构 | get_model_info()中的字段名 |
|------|---------------------------|
| CNN  | `'fc_structure'`, `'num_fc_layers'` |
| MLP  | `'mlp_structure'`, `'num_mlp_layers'` |

### 4. 处理特殊激活函数

对于Sine系列模型，它们使用`Sin`激活而非`ReLU`：

**如果需要自定义激活函数**，可能需要使用方法B（手动构建）而非`build_adaptive_fc_pair`。

检查原始代码，如果全连接层使用了自定义激活，请：
1. 使用`calculate_intermediate_dims`获取维度列表
2. 手动构建layers，使用Sin激活
3. 参考`WaveletMLPAutoEncoder`的手动构建方式（line 122-156）

**示例**:
```python
from autoencoder.utils.adaptive_layers import calculate_intermediate_dims

# 计算维度
self.intermediate_dims = calculate_intermediate_dims(
    input_dim=4096,
    latent_dim=latent_dim,
    max_ratio=4
)

# 手动构建encoder（使用Sin激活）
encoder_fc_layers = [nn.Flatten()]
current_dim = 4096

for intermediate_dim in self.intermediate_dims:
    encoder_fc_layers.extend([
        nn.Linear(current_dim, intermediate_dim),
        nn.BatchNorm1d(intermediate_dim),
        Sin(),  # ← 使用自定义激活
        nn.Dropout(dropout_rate)
    ])
    current_dim = intermediate_dim

encoder_fc_layers.append(nn.Linear(current_dim, latent_dim))
self.encoder_fc = nn.Sequential(*encoder_fc_layers)
```

---

## 📖 参考示例

### 参考1: WaveletAutoEncoder (CNN架构)

**文件**: `autoencoder/models/cnn_autoencoder.py`

**关键代码片段**:
```python
# Line 15-57: 导入和辅助函数
from autoencoder.utils.adaptive_layers import (
    calculate_intermediate_dims,  # 如果需要
    build_adaptive_fc_pair,
    get_structure_info
)

# Line 123-146: 替换全连接层
self.encoder_fc, self.decoder_fc, self.intermediate_dims = build_adaptive_fc_pair(
    self.flattened_size, latent_dim, dropout_rate, arch_type='cnn'
)

# Line 302-328: 更新get_model_info
structure_info = get_structure_info(
    self.flattened_size,
    self.latent_dim,
    self.intermediate_dims
)
return {
    'model_name': 'WaveletAutoEncoder',
    'architecture': 'CNN',
    'fc_structure': structure_info['fc_structure'],
    'intermediate_dims': structure_info['intermediate_dims'],
    'num_fc_layers': structure_info['num_fc_layers'],
    # ...
}
```

### 参考2: DirectMLPAutoEncoder (MLP架构)

**文件**: `autoencoder/models/mlp_autoencoder.py`

**关键代码片段**:
```python
# Line 13-73: 导入和辅助函数
from autoencoder.utils.adaptive_layers import calculate_mlp_dims

# Line 302-345: 手动构建（使用calculate_mlp_dims）
self.intermediate_dims = calculate_mlp_dims(
    self.input_dim, latent_dim, max_ratio=4, min_layers=3
)

encoder_layers = [nn.Flatten()]
current_dim = self.input_dim

for intermediate_dim in self.intermediate_dims:
    encoder_layers.extend([
        nn.Linear(current_dim, intermediate_dim),
        nn.BatchNorm1d(intermediate_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate)
    ])
    current_dim = intermediate_dim

encoder_layers.append(nn.Linear(current_dim, latent_dim))
self.encoder = nn.Sequential(*encoder_layers)
# decoder同理（反向）
```

---

## ✅ 验证清单

完成每个模型后，请验证：

### 代码检查
- [ ] 已导入`build_adaptive_fc_pair`和`get_structure_info`
- [ ] 已保存`self.intermediate_dims`变量
- [ ] `get_model_info()`包含新字段：`fc_structure`/`mlp_structure`、`intermediate_dims`、`num_fc_layers`/`num_mlp_layers`
- [ ] 字段命名正确（CNN用`fc_`，MLP用`mlp_`）
- [ ] 模型仍有`self.encoder`和`self.decoder`属性

### 功能测试
每个模型完成后运行：

```python
from autoencoder.utils.frequency_config import create_autoencoder_system

# 测试创建系统
system = create_autoencoder_system(
    '2freq',
    latent_dim=32,
    mode='wavelet',  # 或 'direct'
    architecture='enhanced_cnn'  # 对应的架构名
)

# 检查输出（应该显示网络结构）
# 【模型配置】
#   - AutoEncoder参数量: xxx
#   - 隐空间维度: 32
# 【网络结构】
#   - FC层结构: 4096 → 1024 → 256 → 64 → 32
#   - FC层数: 4

# 测试前向传播
import torch
if mode == 'wavelet':
    test_input = torch.randn(2, 49, 49, 8)
else:
    test_input = torch.randn(2, 91, 91, 2)

model = system['autoencoder']
recon, latent = model(test_input)

print(f"输入形状: {test_input.shape}")
print(f"隐空间形状: {latent.shape}")  # 应该是 [2, 32]
print(f"重建形状: {recon.shape}")  # 应该和输入相同

assert latent.shape[1] == 32, "隐空间维度错误！"
print("✅ 测试通过！")
```

### 边界测试
测试不同的隐空间维度：

```python
# 测试latent_dim=16
system_16 = create_autoencoder_system('2freq', latent_dim=16, mode='wavelet', architecture='enhanced_cnn')
print(f"latent_dim=16: {system_16['autoencoder'].get_model_info()['fc_structure']}")

# 测试latent_dim=64
system_64 = create_autoencoder_system('2freq', latent_dim=64, mode='wavelet', architecture='enhanced_cnn')
print(f"latent_dim=64: {system_64['autoencoder'].get_model_info()['fc_structure']}")

# 测试latent_dim=256（原始）
system_256 = create_autoencoder_system('2freq', latent_dim=256, mode='wavelet', architecture='enhanced_cnn')
print(f"latent_dim=256: {system_256['autoencoder'].get_model_info()['fc_structure']}")
```

预期输出示例：
```
latent_dim=16: 4096 → 1024 → 256 → 64 → 16
latent_dim=64: 4096 → 1024 → 256 → 64
latent_dim=256: 4096 → 1024 → 256
```

---

## 📝 完成后提交

完成所有8个模型后，创建一个commit：

```bash
git add autoencoder/models/enhanced_cnn_autoencoder.py
git add autoencoder/models/deep_autoencoder.py
git add autoencoder/models/sine_cnn_autoencoder.py
git add autoencoder/models/sine_mlp_autoencoder.py

git commit -m "feat(ae-adaptive): 为剩余8个AutoEncoder模型添加动态隐空间适配

完成模型：
- EnhancedWaveletAutoEncoder / EnhancedDirectAutoEncoder
- DeepWaveletAutoEncoder / DeepDirectAutoEncoder
- SinWaveletAutoEncoder / SinDirectAutoEncoder
- SinWaveletMLPAutoEncoder / SinDirectMLPAutoEncoder

所有模型现在支持16-32维小隐空间，自动生成中间层，保持每级压缩比≤4:1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"
```

---

## 🆘 常见问题

### Q1: 找不到`input_dim`怎么办？

**方法1**: 查找原始代码中的`self.flattened_size`或类似变量

**方法2**: 运行测试代码打印：
```python
model = OriginalModel(latent_dim=256)
# 在encoder最后加一行打印
# print(f"Feature shape before FC: {features.shape}")
```

**方法3**: 计算卷积输出
- 如果有`AdaptiveAvgPool2d((4, 4))`且通道数256 → input_dim = 256 × 4 × 4 = 4096
- 如果有`AdaptiveAvgPool2d(1)`且通道数512 → input_dim = 512 × 1 × 1 = 512

### Q2: Sine模型的激活函数怎么处理？

**如果全连接层也使用Sin激活**:
- 使用方法B（手动构建）
- 参考`WaveletMLPAutoEncoder`的手动构建方式
- 将`nn.ReLU()`替换为`Sin()`

**如果只有卷积层使用Sin**:
- 直接使用`build_adaptive_fc_pair`
- 全连接层会自动使用ReLU（通常这样也工作良好）

### Q3: `get_model_info()`原来就有`model_name`等字段，要删除吗？

**不要删除**！保留所有原有字段，只是**添加**新字段：
```python
return {
    # 保留原有的所有字段
    'model_name': '...',
    'total_parameters': ...,
    # ... 其他原字段 ...

    # 添加新字段
    'fc_structure': structure_info['fc_structure'],
    'intermediate_dims': structure_info['intermediate_dims'],
    'num_fc_layers': structure_info['num_fc_layers'],
}
```

### Q4: 模型测试失败怎么办？

**步骤**:
1. 检查`input_dim`是否正确
2. 检查是否保存了`self.intermediate_dims`
3. 检查`get_model_info()`的字段名（CNN用`fc_`，MLP用`mlp_`）
4. 打印`model.get_model_info()`查看输出
5. 运行前向传播测试，查看形状是否匹配

---

## 🎓 学习资源

- **ADAPTIVE_LAYERS_GUIDE.md** - 完整使用指南
- **autoencoder/utils/adaptive_layers.py** - 源代码和注释
- **已完成的4个模型** - 最佳实践参考
- **frequency_config.py** - 系统集成示例

---

**祝顺利完成！有任何问题可以参考已完成的模型代码。**

---

生成时间: 2025-01-18
任务发起: Claude Code
执行者: Codex (待执行)
