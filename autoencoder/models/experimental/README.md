# 实验性AutoEncoder模型

## 概述

此文件夹包含在开发过程中创建的实验性AutoEncoder模型。这些模型**目前未被主系统使用**，但保留用于：
- 参考和学习
- 未来可能的改进
- 特定场景的优化

## 模型列表

### 1. CorrectCNNAutoEncoder (correct_cnn_autoencoder.py)
- **创建时间**: 2024年9月30日
- **状态**: 已废弃
- **问题**:
  - 基于错误的小波尺寸假设（46×46 vs 实际49×49）
  - 已被 WaveletAutoEncoder 替代
- **是否可用**: ❌ 否
- **建议**: 可删除

### 2. DeepCNNAutoEncoder (deep_cnn_autoencoder.py)
- **创建时间**: 2024年9月30日
- **状态**: 实验性
- **特点**:
  - 5层深度CNN架构
  - 对比MLP vs CNN的设计思路
  - 参数量：~2M
- **是否可用**: ⚠️ 需要测试
- **如何使用**:
  1. 重命名为 `DeepWaveletAutoEncoder`
  2. 创建对应的 `DeepDirectAutoEncoder`
  3. 在 frequency_config.py 中添加 `architecture='deep_cnn'` 分支

### 3. EfficientCNNAutoEncoder (efficient_cnn_autoencoder.py)
- **创建时间**: 2024年9月30日
- **状态**: 实验性
- **特点**:
  - 轻量3层CNN
  - 针对49×49小波系数优化
  - 参数量：~500K（标准CNN的1/3）
- **是否可用**: ⚠️ 需要测试
- **如何使用**:
  1. 重命名为 `LightweightWaveletAutoEncoder`
  2. 创建 `LightweightDirectAutoEncoder`
  3. 适合嵌入式或快速推理场景

### 4. MicroLatentAutoEncoder (micro_latent_autoencoder.py)
- **创建时间**: 2024年9月30日
- **状态**: 实验性
- **特点**:
  - 支持极小隐空间（10-64维）
  - 专门优化的瓶颈设计
  - 适合极端压缩场景
- **是否可用**: ⚠️ 需要测试
- **如何使用**:
  1. 重命名为 `MicroLatentWaveletAutoEncoder`
  2. 创建 `MicroLatentDirectAutoEncoder`
  3. 适合存储空间极其受限的场景

## 集成指南

如需将这些实验性模型集成到主系统：

### 步骤1: 重命名（符合规范）

```python
# 命名规范: <Mode><Architecture>AutoEncoder
# Mode: Wavelet | Direct
# Architecture: (空)=标准CNN | MLP | Enhanced | Deep | Lightweight | MicroLatent

# 示例
DeepCNNAutoEncoder → DeepWaveletAutoEncoder
```

### 步骤2: 创建对应版本

每个Wavelet版本都需要对应的Direct版本：

```python
# DeepWaveletAutoEncoder 输入: [B, 49, 49, 8] 小波系数
# DeepDirectAutoEncoder 输入: [B, 91, 91, 2] RCS数据
```

### 步骤3: 修改 frequency_config.py

在 `create_autoencoder_system()` 中添加分支：

```python
if mode == 'wavelet':
    # ...existing code...
    elif architecture.lower() == 'deep_cnn':
        autoencoder = DeepWaveletAutoEncoder(...)
        print(f"使用 DeepWaveletAutoEncoder")
else:  # direct
    # ...existing code...
    elif architecture.lower() == 'deep_cnn':
        autoencoder = DeepDirectAutoEncoder(...)
        print(f"使用 DeepDirectAutoEncoder")
```

### 步骤4: 更新 __init__.py

```python
# autoencoder/models/__init__.py
from .deep_autoencoder import DeepWaveletAutoEncoder, DeepDirectAutoEncoder

__all__ = [
    # ...existing...
    'DeepWaveletAutoEncoder',
    'DeepDirectAutoEncoder',
]
```

### 步骤5: 测试

```python
# 创建系统并测试
system = create_autoencoder_system(
    mode='wavelet',
    architecture='deep_cnn',
    latent_dim=256
)

# 训练和评估
# ...
```

## 性能对比

| 模型 | 参数量 | 层数 | 隐空间 | 适用场景 |
|------|--------|------|--------|----------|
| CorrectCNN | - | - | - | 已废弃 |
| DeepCNN | ~2M | 5 | 256 | 需要更强表达力 |
| EfficientCNN | ~500K | 3 | 256 | 需要快速推理 |
| MicroLatent | ~300K | 4 | 10-64 | 极端压缩 |

## 注意事项

⚠️ **使用前请注意**：
1. 这些模型未经过充分测试
2. 性能可能不如主系统的6个核心网络
3. 需要根据具体场景调整超参数
4. 建议先在小数据集上验证效果

## 维护建议

- ✅ **保留**: DeepCNN, EfficientCNN, MicroLatent（有特定用途）
- ❌ **删除**: CorrectCNN（已被替代且有错误）
- 📝 **文档化**: 记录每次实验的结果和经验

---

最后更新: 2025-01-14
