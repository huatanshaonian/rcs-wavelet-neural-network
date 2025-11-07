# 参数参考文档

> **用途**: 统一项目中所有参数命名规范，避免参数名不匹配导致的bug
> **最后更新**: 2025-01-18

---

## 📋 目录

1. [核心函数参数](#核心函数参数)
2. [GUI变量命名规范](#gui变量命名规范)
3. [中英文映射表](#中英文映射表)
4. [参数有效值](#参数有效值)
5. [参数依赖关系](#参数依赖关系)

---

## 核心函数参数

### 1. `create_autoencoder_system()` (frequency_config.py)

**完整签名**:
```python
def create_autoencoder_system(
    config_name: str = '2freq',
    latent_dim: int = 32,
    dropout_rate: float = 0.2,
    wavelet: str = 'db4',                    # ⚠️ 注意：是wavelet，不是wavelet_type
    normalize: bool = True,
    mode: str = 'wavelet',
    architecture: str = 'cnn',
    use_channel_attention: bool = False,
    activation: str = 'relu',
    db_transform: bool = False,              # ⚠️ 注意：是db_transform，不是log_transform
    normalization_method: str = 'zscore'
) -> Dict[str, Any]
```

**参数说明**:
| 参数名 | 类型 | 默认值 | 说明 | 有效值 |
|--------|------|--------|------|--------|
| `config_name` | str | '2freq' | 频率配置 | '2freq', '3freq' |
| `latent_dim` | int | 32 | 隐空间维度 | 16, 32, 64, 128, 256, 512 |
| `dropout_rate` | float | 0.2 | Dropout比率 | 0.0-0.5 |
| `wavelet` | str | 'db4' | 小波类型 | 'db4', 'db8', 'haar', 'bior2.2' |
| `normalize` | bool | True | 是否标准化（向后兼容） | True/False |
| `mode` | str | 'wavelet' | AE模式 | 'wavelet', 'direct', 'differentiable_wavelet' |
| `architecture` | str | 'cnn' | 架构类型 | 见[架构类型](#架构类型) |
| `use_channel_attention` | bool | False | 输入层通道注意力 | True/False |
| `activation` | str | 'relu' | 激活函数 | 见[激活函数](#激活函数) |
| `db_transform` | bool | False | dB变换 | True/False |
| `normalization_method` | str | 'zscore' | 标准化方法 | 'none', 'zscore', 'minmax' |

**返回值结构**:
```python
{
    'autoencoder': AutoEncoder实例,
    'parameter_mapper': ParameterMapper实例,
    'wavelet_transform': WaveletTransform实例或None,
    'data_adapter': RCS_DataAdapter实例,
    'config': FrequencyConfig实例
}
```

---

### 2. 三阶段训练函数 (gui.py)

#### `_train_autoencoder_stage1_v2()`
**关键参数**:
- `ae_system`: create_autoencoder_system()返回的字典
- `rcs_data`: RCS数据 [N, 91, 91, num_freq]
- `epochs`: Stage 1训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率

#### `_train_parameter_mapping_stage2_v2()`
**关键参数**:
- `ae_system`: 已训练Stage 1的系统
- `param_data`: 参数数据 [N, param_dim]
- `epochs`: Stage 2训练轮数

#### `_train_end_to_end_stage3_v2()`
**关键参数**:
- `ae_system`: 已训练Stage 1+2的系统
- `rcs_data`, `param_data`: 训练数据
- `epochs`: Stage 3训练轮数

---

### 3. 批量实验相关函数

#### `BatchExperimentManager.__init__()`
```python
def __init__(self,
    base_config: Dict[str, Any],
    compare_dimensions: Dict[str, List[Any]],
    experiment_name: str = "batch_experiment",
    save_dir: str = "batch_experiments"
)
```

**base_config字典结构**:
```python
{
    'mode': str,                    # 'wavelet', 'direct', 'differentiable_wavelet'
    'architecture': str,            # 'cnn', 'mlp', 'enhanced_cnn', 'deep_cnn', ...
    'activation': str,              # 'relu', 'sin', 'gelu', ...
    'wavelet_type': str,            # 'db4', 'db8', 'haar', 'bior2.2'
    'normalization_method': str,    # 'none', 'zscore', 'minmax'
    'db_transform': bool,           # True/False
    'latent_dim': int,              # 隐空间维度
    'batch_size': int,              # 批次大小
    'learning_rate': float,         # 学习率
    'training_mode': str,           # 'three_stage', 'stage1_only'
    'epochs': {
        'stage1': int,
        'stage2': int,
        'stage3': int
    }
}
```

**compare_dimensions字典结构**:
```python
{
    'mode': ['wavelet', 'direct'],              # AE模式对比
    'architecture': ['cnn', 'mlp'],             # 架构对比
    'activation': ['relu', 'sin', 'gelu'],      # 激活函数对比
    'wavelet_type': ['db4', 'db8'],            # 小波类型对比
    'preprocessing': [                          # 预处理对比（特殊格式）
        {'normalization_method': 'zscore', 'db_transform': False},
        {'normalization_method': 'zscore', 'db_transform': True}
    ]
}
```

---

## GUI变量命名规范

### 命名模式
```
ae_{配置项}_{阶段}    # 有阶段区分的配置
ae_{配置项}           # 全局配置
```

### 完整GUI变量列表

#### 全局配置
| GUI变量名 | 类型 | 对应内部参数 | 说明 |
|-----------|------|--------------|------|
| `ae_mode` | StringVar | `mode` | AE模式 |
| `ae_architecture_type` | StringVar | `architecture` | 架构类型（需映射） |
| `ae_activation` | StringVar | `activation` | 激活函数 |
| `ae_wavelet_type` | StringVar | `wavelet` | 小波类型 |
| `ae_normalization_method` | StringVar | `normalization_method` | 标准化方法 |
| `ae_db_transform` | BooleanVar | `db_transform` | dB变换开关 |
| `ae_latent_dim` | StringVar | `latent_dim` | 隐空间维度 |
| `ae_batch_size` | StringVar | `batch_size` | 批次大小 |
| `ae_learning_rate` | StringVar | `learning_rate` | 学习率 |
| `ae_training_mode` | StringVar | `training_mode` | 训练模式（需映射） |

#### 阶段特定配置
| GUI变量名 | 对应内部参数 | 说明 |
|-----------|--------------|------|
| `ae_epochs_stage1` | `epochs['stage1']` | Stage 1轮数 |
| `ae_epochs_stage2` | `epochs['stage2']` | Stage 2轮数 |
| `ae_epochs_stage3` | `epochs['stage3']` | Stage 3轮数 |

⚠️ **注意**: 命名规范是`ae_{config}_{stage}`，不是`ae_{stage}_{config}`！

---

## 中英文映射表

### 1. AE模式 (mode)
| GUI显示（中文） | 内部标识符 | create_autoencoder_system参数 |
|----------------|-----------|-------------------------------|
| Wavelet模式 | 'wavelet' | mode='wavelet' |
| Direct模式 | 'direct' | mode='direct' |
| 可微分小波模式 | 'differentiable_wavelet' | mode='differentiable_wavelet' |

### 2. 架构类型 (architecture)
| GUI显示 | 内部标识符 | 说明 |
|---------|-----------|------|
| CNN | 'cnn' | 标准CNN（4层encoder+decoder） |
| Enhanced_CNN | 'enhanced_cnn' | 增强CNN（多尺度+空洞卷积） |
| Deep_CNN | 'deep_cnn' | 深度CNN（双卷积块） |
| MLP | 'mlp' | 多层感知器（5层全连接） |
| Dual_Branch_CNN | 'dual_branch_cnn' | 双分支CNN（LL+高频分离） |
| Dual_Branch_MLP | 'dual_branch_mlp' | 双分支MLP |

### 3. 激活函数 (activation)
| GUI显示 | 内部标识符 | PyTorch对应 |
|---------|-----------|-------------|
| relu | 'relu' | nn.ReLU() |
| sin | 'sin' | torch.sin() |
| gelu | 'gelu' | nn.GELU() |
| swish | 'swish' | x * torch.sigmoid(x) |
| tanh | 'tanh' | nn.Tanh() |
| mish | 'mish' | x * torch.tanh(F.softplus(x)) |
| elu | 'elu' | nn.ELU() |
| leaky_relu | 'leaky_relu' | nn.LeakyReLU() |
| prelu | 'prelu' | nn.PReLU() |

### 4. 小波类型 (wavelet)
| GUI显示 | 内部标识符 | PyWavelets对应 |
|---------|-----------|----------------|
| db4 | 'db4' | pywt.Wavelet('db4') |
| db8 | 'db8' | pywt.Wavelet('db8') |
| haar | 'haar' | pywt.Wavelet('haar') |
| bior2.2 | 'bior2.2' | pywt.Wavelet('bior2.2') |

### 5. 标准化方法 (normalization_method)
| GUI显示 | 内部标识符 | 说明 |
|---------|-----------|------|
| 无 | 'none' | 不进行标准化 |
| Z-score标准化 | 'zscore' | (x - mean) / std |
| Min-Max标准化 | 'minmax' | (x - min) / (max - min) |

### 6. 预处理组合 (GUI专用)
| GUI显示 | 解析为 |
|---------|--------|
| none | `{'normalization_method': 'none', 'db_transform': False}` |
| zscore | `{'normalization_method': 'zscore', 'db_transform': False}` |
| minmax | `{'normalization_method': 'minmax', 'db_transform': False}` |
| zscore+db | `{'normalization_method': 'zscore', 'db_transform': True}` |
| minmax+db | `{'normalization_method': 'minmax', 'db_transform': True}` |

### 7. 训练模式 (training_mode)
| GUI显示（中文） | 内部标识符 | 说明 |
|----------------|-----------|------|
| 三阶段训练 | 'three_stage' | Stage 1+2+3完整训练 |
| 仅Stage 1 | 'stage1_only' | 只训练AutoEncoder重建 |

---

## 参数有效值

### 架构类型
```python
VALID_ARCHITECTURES = [
    'cnn',              # 标准CNN
    'mlp',              # MLP
    'enhanced_cnn',     # Enhanced CNN
    'deep_cnn',         # Deep CNN
    'dual_branch_cnn',  # 双分支CNN
    'dual_branch_mlp'   # 双分支MLP
]
```

### 激活函数
```python
VALID_ACTIVATIONS = [
    'relu',
    'sin',
    'gelu',
    'swish',
    'tanh',
    'mish',
    'elu',
    'leaky_relu',
    'prelu'
]
```

### 小波类型
```python
VALID_WAVELETS = [
    'db4',      # Daubechies 4
    'db8',      # Daubechies 8
    'haar',     # Haar小波
    'bior2.2'   # Biorthogonal 2.2
]
```

### 隐空间维度（推荐值）
```python
RECOMMENDED_LATENT_DIMS = [16, 32, 64, 128, 256, 512]
```

### 批次大小（推荐值）
```python
RECOMMENDED_BATCH_SIZES = [4, 8, 16, 32, 64]
```

### 学习率（推荐范围）
```python
LEARNING_RATE_RANGE = (1e-5, 1e-2)  # 0.00001 - 0.01
RECOMMENDED_LR = 1e-4               # 默认: 0.0001
```

---

## 参数依赖关系

### 1. 小波类型依赖
```
wavelet_type 仅在以下情况有效：
  - mode == 'wavelet'
  - mode == 'differentiable_wavelet'

如果 mode == 'direct'，wavelet_type参数会被忽略
```

### 2. 架构与模式兼容性
| 架构 | wavelet | direct | differentiable_wavelet |
|------|---------|--------|------------------------|
| cnn | ✅ | ✅ | ✅ |
| mlp | ✅ | ✅ | ✅ |
| enhanced_cnn | ✅ | ✅ | ❌ |
| deep_cnn | ✅ | ✅ | ❌ |
| dual_branch_cnn | ✅ | ❌ | ✅ |
| dual_branch_mlp | ✅ | ❌ | ✅ |

### 3. 预处理依赖
```
db_transform 和 normalization_method 独立：
  - 可以单独使用 normalization_method
  - 可以单独使用 db_transform
  - 可以同时使用（推荐：zscore + db_transform）
```

### 4. 训练模式依赖
```
training_mode == 'stage1_only':
  - 只训练AutoEncoder
  - 不需要param_data（但需要提供占位数据）
  - epochs['stage2'] 和 epochs['stage3'] 会被忽略

training_mode == 'three_stage':
  - 需要完整的rcs_data和param_data
  - 执行完整的三阶段训练
```

### 5. 频率配置依赖
```
config_name 影响：
  - 输入通道数：2freq → 8通道, 3freq → 12通道
  - 模型输入尺寸
  - 数据加载时必须匹配
```

---

## ⚠️ 常见参数错误

### 错误1: 参数名不匹配
```python
# ❌ 错误
create_autoencoder_system(wavelet_type='db4')

# ✅ 正确
create_autoencoder_system(wavelet='db4')
```

### 错误2: GUI变量命名顺序
```python
# ❌ 错误
gui.ae_stage1_epochs.get()

# ✅ 正确
gui.ae_epochs_stage1.get()
```

### 错误3: 预处理参数传递
```python
# ❌ 错误
create_autoencoder_system(log_transform=True)

# ✅ 正确
create_autoencoder_system(db_transform=True)
```

### 错误4: 架构名映射遗漏
```python
# ❌ 错误：GUI直接传递
create_autoencoder_system(architecture=gui.ae_architecture_type.get())  # 'CNN'

# ✅ 正确：需要映射
arch = _map_architecture(gui.ae_architecture_type.get())  # 'cnn'
create_autoencoder_system(architecture=arch)
```

### 错误5: 模式不兼容的架构
```python
# ❌ 错误：direct模式不支持dual_branch
create_autoencoder_system(mode='direct', architecture='dual_branch_cnn')

# ✅ 正确：检查兼容性
if mode == 'direct' and architecture.startswith('dual_branch'):
    raise ValueError("Direct模式不支持双分支架构")
```

---

## 📌 快速查找索引

### 我想...
- **创建AutoEncoder系统** → 见[create_autoencoder_system()](#1-create_autoencoder_system-frequency_configpy)
- **获取GUI配置** → 见[GUI变量命名规范](#gui变量命名规范)
- **映射GUI显示到内部值** → 见[中英文映射表](#中英文映射表)
- **检查参数有效性** → 见[参数有效值](#参数有效值)
- **了解参数限制** → 见[参数依赖关系](#参数依赖关系)
- **排查参数错误** → 见[常见参数错误](#常见参数错误)

---

**维护说明**:
- 添加新参数时，更新本文档对应章节
- 修改参数名时，检查所有使用位置并更新文档
- 添加新的中英文映射时，更新映射表
- 发现参数依赖关系时，记录到依赖关系章节
