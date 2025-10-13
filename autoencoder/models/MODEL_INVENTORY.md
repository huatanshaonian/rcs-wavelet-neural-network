# AutoEncoder模型清单

## 📋 正在使用的模型（6个核心网络）

### 1. 标准CNN系列

#### WaveletAutoEncoder (cnn_autoencoder.py)
- **命名规范**: ✅ 符合
- **模式**: Wavelet模式（小波增强）
- **架构**: 标准CNN（4层encoder + 4层decoder）
- **输入**: [B, 49, 49, 8] 小波系数（2freq×4bands）
- **隐空间**: 256维（默认）
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='cnn')`
- **用途**: 标准的小波增强CNN，平衡性能和速度

#### DirectAutoEncoder (direct_autoencoder.py)
- **命名规范**: ✅ 符合
- **模式**: Direct模式（直接处理）
- **架构**: 标准CNN（4层encoder + 4层decoder）
- **输入**: [B, 91, 91, 2] RCS数据（不经小波变换）
- **隐空间**: 256维（默认）
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='cnn')`
- **用途**: 直接处理RCS数据，无小波变换开销

### 2. MLP系列

#### WaveletMLPAutoEncoder (mlp_autoencoder.py)
- **命名规范**: ✅ 符合
- **模式**: Wavelet模式
- **架构**: 深层MLP（5层encoder + 5层decoder）
- **输入**: [B, 49, 49, 8] 小波系数
- **隐空间**: 256维（默认）
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='mlp')`
- **用途**: 全连接架构，适合参数敏感性分析

#### DirectMLPAutoEncoder (mlp_autoencoder.py)
- **命名规范**: ✅ 符合
- **模式**: Direct模式
- **架构**: 深层MLP（5层encoder + 5层decoder）
- **输入**: [B, 91, 91, 2] RCS数据
- **隐空间**: 256维（默认）
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='mlp')`
- **用途**: 直接MLP处理，无空间偏好

### 3. 增强CNN系列

#### EnhancedWaveletAutoEncoder (enhanced_cnn_autoencoder.py)
- **命名规范**: ✅ 符合
- **模式**: Wavelet模式
- **架构**: 增强感受野CNN（多尺度卷积 + 更大感受野）
- **输入**: [B, 49, 49, 8] 小波系数
- **隐空间**: 256维（默认）
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='enhanced_cnn')`
- **用途**: 更好的全局特征捕捉，适合复杂模式

#### EnhancedDirectAutoEncoder (enhanced_cnn_autoencoder.py)
- **命名规范**: ✅ 符合
- **模式**: Direct模式
- **架构**: 增强感受野CNN
- **输入**: [B, 91, 91, 2] RCS数据
- **隐空间**: 256维（默认）
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='enhanced_cnn')`
- **用途**: 直接处理 + 增强感受野

---

## 🧪 实验性/废弃模型（4个，未使用）

### CorrectCNNAutoEncoder (correct_cnn_autoencoder.py)
- **状态**: ⚠️ 废弃（早期版本）
- **问题**: 输入尺寸假设错误（46×46 vs 实际49×49）
- **原因**: 早期对小波变换尺寸的错误理解
- **是否调用**: ❌ 否（未在__init__.py中导出）
- **建议**: 删除或移至archive/

### DeepCNNAutoEncoder (deep_cnn_autoencoder.py)
- **状态**: ⚠️ 实验性
- **特点**: 5层深度CNN，对比MLP vs CNN性能
- **是否调用**: ❌ 否
- **建议**: 如需使用，需重命名为 DeepWaveletAutoEncoder 并添加Direct版本

### EfficientCNNAutoEncoder (efficient_cnn_autoencoder.py)
- **状态**: ⚠️ 实验性
- **特点**: 轻量3层CNN，针对49×49优化
- **是否调用**: ❌ 否
- **建议**: 重命名为 LightweightWaveletAutoEncoder

### MicroLatentAutoEncoder (micro_latent_autoencoder.py)
- **状态**: ⚠️ 实验性
- **特点**: 支持极小隐空间（10-64维）
- **是否调用**: ❌ 否
- **建议**: 重命名为 MicroLatentWaveletAutoEncoder

---

## 📌 辅助模块

### ParameterMapper (cnn_autoencoder.py + parameter_mapper.py)
- **作用**: 9维参数 → 隐空间映射
- **位置**: 两处定义（重复）
- **调用**: ParameterMapperFactory.create_mapper()
- **建议**: 统一使用 parameter_mapper.py 中的版本

---

## 🎯 命名规范建议

### 当前规范（正在使用的6个）
```
<Mode><Architecture>AutoEncoder

Mode:
- Wavelet: 小波增强模式
- Direct: 直接处理模式

Architecture:
- (空): 标准CNN
- MLP: 全连接网络
- Enhanced: 增强感受野CNN
```

### 示例
- ✅ WaveletAutoEncoder
- ✅ DirectAutoEncoder
- ✅ WaveletMLPAutoEncoder
- ✅ DirectMLPAutoEncoder
- ✅ EnhancedWaveletAutoEncoder
- ✅ EnhancedDirectAutoEncoder

### 不规范的命名（实验性模型）
- ❌ CorrectCNNAutoEncoder → 应为 WaveletAutoEncoder (已被替代)
- ❌ DeepCNNAutoEncoder → 应为 DeepWaveletAutoEncoder + DeepDirectAutoEncoder
- ❌ EfficientCNNAutoEncoder → 应为 LightweightWaveletAutoEncoder + LightweightDirectAutoEncoder
- ❌ MicroLatentAutoEncoder → 应为 MicroLatentWaveletAutoEncoder + MicroLatentDirectAutoEncoder

---

## 🔄 调用流程

```python
# GUI创建系统时
from autoencoder.utils.frequency_config import create_autoencoder_system

# 自动根据mode和architecture选择正确的网络
system = create_autoencoder_system(
    config_name='2freq',
    mode='wavelet',        # 'wavelet' 或 'direct'
    architecture='cnn',    # 'cnn', 'mlp', 或 'enhanced_cnn'
    latent_dim=256
)

# frequency_config.py 内部逻辑:
if mode == 'wavelet':
    if architecture == 'mlp':
        autoencoder = WaveletMLPAutoEncoder(...)
    elif architecture == 'enhanced_cnn':
        autoencoder = EnhancedWaveletAutoEncoder(...)
    else:  # 'cnn'
        autoencoder = WaveletAutoEncoder(...)
else:  # mode == 'direct'
    if architecture == 'mlp':
        autoencoder = DirectMLPAutoEncoder(...)
    elif architecture == 'enhanced_cnn':
        autoencoder = EnhancedDirectAutoEncoder(...)
    else:  # 'cnn'
        autoencoder = DirectAutoEncoder(...)
```

---

## 📊 模型对比

| 模型 | 参数量 | 感受野 | 速度 | 适用场景 |
|------|--------|--------|------|---------|
| WaveletAutoEncoder | 适中 | 标准 | 快 | 通用，平衡性能 |
| DirectAutoEncoder | 较大 | 标准 | 中等 | 无小波开销 |
| WaveletMLPAutoEncoder | 大 | 全局 | 慢 | 参数分析 |
| DirectMLPAutoEncoder | 极大 | 全局 | 很慢 | 实验性 |
| EnhancedWaveletAutoEncoder | 大 | 大 | 中等 | 复杂模式 |
| EnhancedDirectAutoEncoder | 极大 | 大 | 慢 | 最强表达力 |

---

## ✅ 建议操作

1. **保持当前6个核心网络不变**
2. **将4个实验性网络移至 `autoencoder/models/experimental/` 文件夹**
3. **如需使用实验性网络，重命名并创建对应的Direct版本**
4. **统一ParameterMapper到 parameter_mapper.py**
5. **更新文档说明每个网络的用途和性能特点**

---

生成时间: 2025-01-14
维护者: Claude Code
