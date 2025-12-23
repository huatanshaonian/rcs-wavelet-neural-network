# AutoEncoder模型清单

> **最后更新**: 2025-12-23
> **维护者**: Claude Code

## 📋 模型架构体系

### 1. 基础架构（8个核心模型）

#### 1.1 标准CNN系列

**WaveletAutoEncoder** (`cnn_autoencoder.py`)
- **模式**: Wavelet（小波增强）
- **架构**: 标准CNN（4层encoder + 4层decoder）
- **输入**: [B, 49, 49, 8] 小波系数（2freq×4bands）
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='cnn')`
- **用途**: 标准的小波增强CNN，平衡性能和速度（推荐默认）
- **参数量**: ~1.5M

**DirectAutoEncoder** (`direct_autoencoder.py`)
- **模式**: Direct（直接处理）
- **架构**: 标准CNN（4层encoder + 4层decoder）
- **输入**: [B, 91, 91, 2] RCS数据（不经小波变换）
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='cnn')`
- **用途**: 直接处理RCS数据，无小波变换开销
- **参数量**: ~2.5M

#### 1.2 MLP系列

**WaveletMLPAutoEncoder** (`mlp_autoencoder.py`)
- **模式**: Wavelet
- **架构**: 深层MLP（5层encoder + 5层decoder）
- **输入**: [B, 49, 49, 8] 小波系数（展平为向量）
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='mlp')`
- **用途**: 全连接架构，适合参数敏感性分析
- **参数量**: ~3M

**DirectMLPAutoEncoder** (`mlp_autoencoder.py`)
- **模式**: Direct
- **架构**: 深层MLP（5层encoder + 5层decoder）
- **输入**: [B, 91, 91, 2] RCS数据（展平为向量）
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='mlp')`
- **用途**: 直接MLP处理，无空间偏好
- **参数量**: ~5M

#### 1.3 Enhanced CNN系列

**EnhancedWaveletAutoEncoder** (`enhanced_cnn_autoencoder.py`)
- **模式**: Wavelet
- **架构**: 增强感受野CNN（多尺度卷积 + 空洞残差 + 通道注意力）
- **输入**: [B, 49, 49, 8] 小波系数
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='enhanced_cnn')`
- **用途**: 更好的全局特征捕捉，适合复杂模式
- **参数量**: ~11M

**EnhancedDirectAutoEncoder** (`enhanced_cnn_autoencoder.py`)
- **模式**: Direct
- **架构**: 增强感受野CNN
- **输入**: [B, 91, 91, 2] RCS数据
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='enhanced_cnn')`
- **用途**: 直接处理 + 增强感受野
- **参数量**: ~25M

#### 1.4 Deep CNN系列

**DeepWaveletAutoEncoder** (`deep_autoencoder.py`)
- **模式**: Wavelet
- **架构**: 深度CNN（4层深度卷积 + 双卷积块 + 通道注意力）
- **输入**: [B, 49, 49, 8] 小波系数
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='deep_cnn')`
- **用途**: Wavelet模式最强表达力，复杂模式学习
- **参数量**: ~29M

**DeepDirectAutoEncoder** (`deep_autoencoder.py`)
- **模式**: Direct
- **架构**: 深度CNN（4层深度卷积 + 双卷积块 + 通道注意力）
- **输入**: [B, 91, 91, 2] RCS数据
- **隐空间**: 256维（默认，可配置）
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='deep_cnn')`
- **用途**: Direct模式最强表达力，计算密集
- **参数量**: ~79M

---

### 2. 可微分小波模式（Differentiable Wavelet）

**核心特点**: 小波变换集成为nn.Module，损失在RCS空间计算，梯度可微分回传

**DifferentiableWaveletAutoEncoder** (`differentiable_wavelet_autoencoder.py`)
- **模式**: Differentiable Wavelet
- **架构**: CNN
- **调用方式**: `create_autoencoder_system(mode='differentiable_wavelet', architecture='cnn')`
- **优势**: 端到端训练，适合物理约束（如RCS非负）的直接应用
- **参数量**: ~1.5M

**DifferentiableWaveletMLPAutoEncoder** (`differentiable_wavelet_autoencoder.py`)
- **模式**: Differentiable Wavelet
- **架构**: MLP
- **调用方式**: `create_autoencoder_system(mode='differentiable_wavelet', architecture='mlp')`
- **参数量**: ~3M

---

### 3. 双分支架构（Dual-Branch）

#### 3.1 分离型双分支（V2推荐）

**用途**: 分别处理LL通道（90%+能量）和HF通道（<10%能量）

**DualBranchDifferentiableWaveletAutoEncoderV2** (`dual_branch_differentiable_autoencoder_v2.py`)
- **架构**: 双分支CNN V2
- **调用方式**: `create_autoencoder_system(mode='differentiable_wavelet', architecture='dual_branch_cnn')`
- **特点**: 正确对称架构，ll_decoder + hf_decoder
- **状态**: ✅ 推荐使用
- **参数量**: ~2.2M

**DualBranchDifferentiableWaveletMLPAutoEncoderV2** (`dual_branch_differentiable_autoencoder_v2.py`)
- **架构**: 双分支MLP V2
- **调用方式**: `create_autoencoder_system(mode='differentiable_wavelet', architecture='dual_branch_mlp')`
- **特点**: 正确对称架构
- **状态**: ✅ 推荐使用
- **参数量**: ~20.2M

**DualBranchDifferentiableWaveletAutoEncoder** (`dual_branch_differentiable_autoencoder.py`)
- **架构**: 双分支CNN V1
- **调用方式**: `create_autoencoder_system(mode='differentiable_wavelet', architecture='dual_branch_cnn_v1')`
- **状态**: ⚠️ 旧版（架构缺陷，仅向后兼容）

**DualBranchDifferentiableWaveletMLPAutoEncoder** (`dual_branch_differentiable_autoencoder.py`)
- **架构**: 双分支MLP V1
- **调用方式**: `create_autoencoder_system(mode='differentiable_wavelet', architecture='dual_branch_mlp_v1')`
- **状态**: ⚠️ 旧版（架构缺陷，仅向后兼容）

#### 3.2 叠加型双分支（Additive Dual-Branch，新架构⭐）

**核心思想**: 双Decoder分别学习高频和低频特征，输出加权叠加

**AdditiveDualBranchWaveletAutoEncoder** (`additive_dual_branch_autoencoder.py`)
- **模式**: Wavelet
- **架构**: Additive Dual-Branch CNN
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='additive_dual_branch_cnn')`
- **优势**: Sin激活学习高频，Tanh/Swish学习低频，输出叠加兼顾两者
- **可配置参数**:
  - `activation_encoder`: Encoder激活函数（默认'relu'）
  - `activation_high`: 高频分支激活函数（默认'sin'）
  - `activation_smooth`: 低频分支激活函数（默认'tanh'）
  - `learnable_weights`: 是否学习权重（默认False）
  - `alpha_high`, `alpha_smooth`: 固定权重（默认0.5）

**AdditiveDualBranchWaveletMLPAutoEncoder** (`additive_dual_branch_mlp.py`)
- **模式**: Wavelet
- **架构**: Additive Dual-Branch MLP
- **调用方式**: `create_autoencoder_system(mode='wavelet', architecture='additive_dual_branch_mlp')`

**AdditiveDualBranchDirectAutoEncoder** (`additive_dual_branch_autoencoder.py`)
- **模式**: Direct
- **架构**: Additive Dual-Branch CNN
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='additive_dual_branch_cnn')`

**AdditiveDualBranchDirectMLPAutoEncoder** (`additive_dual_branch_mlp.py`)
- **模式**: Direct
- **架构**: Additive Dual-Branch MLP
- **调用方式**: `create_autoencoder_system(mode='direct', architecture='additive_dual_branch_mlp')`

---

### 4. 辅助模块

**BaseAutoEncoder** (`base_autoencoder.py`)
- **类型**: 抽象基类
- **作用**: 提供统一的接口和共享方法
- **方法**: `get_parameter_count()`, `get_model_info()`, `_apply_output_activation()`

**ChannelAttention** (`channel_attention.py`)
- **类型**: 模块
- **作用**: 通道注意力机制，可选集成到输入层
- **使用**: `use_channel_attention=True`

**ParameterMapper系列** (`parameter_mapper.py`)
- `MLPMapper`: 标准MLP映射器（默认）
- `RandomForestMapper`: 随机森林映射器（实验性）
- `ParameterMapperFactory`: 工厂类

---

## 📊 架构对比

| 架构类型 | Wavelet模式 | Direct模式 | 参数量 | 适用场景 |
|---------|------------|-----------|--------|----------|
| **标准CNN** | WaveletAutoEncoder | DirectAutoEncoder | 小 | 通用，平衡性能（推荐） |
| **MLP** | WaveletMLPAutoEncoder | DirectMLPAutoEncoder | 中 | 参数敏感性分析 |
| **Enhanced CNN** | EnhancedWaveletAutoEncoder | EnhancedDirectAutoEncoder | 大 | 复杂模式，大感受野 |
| **Deep CNN** | DeepWaveletAutoEncoder | DeepDirectAutoEncoder | 极大 | 最强表达力，计算密集 |
| **Dual-Branch V2** | - | - | 中 | LL/HF分离处理（仅Diff Wavelet） |
| **Additive Dual-Branch** | AdditiveDualBranchWaveletAutoEncoder | AdditiveDualBranchDirectAutoEncoder | 中 | 高频+低频叠加重建 |

---

## 🎯 选择指南

### 按性能需求

- **快速原型**: 标准CNN（WaveletAutoEncoder/DirectAutoEncoder）
- **最佳性能**: Deep CNN（DeepWaveletAutoEncoder/DeepDirectAutoEncoder）
- **平衡选择**: Enhanced CNN（EnhancedWaveletAutoEncoder/EnhancedDirectAutoEncoder）

### 按特征类型

- **全局特征**: MLP系列
- **局部特征**: CNN系列
- **多尺度特征**: Enhanced CNN系列
- **高频+低频**: Additive Dual-Branch系列

### 按模式选择

- **小波增强**: Wavelet模式（所有架构）
- **直接处理**: Direct模式（所有架构）
- **端到端训练**: Differentiable Wavelet模式
- **物理约束**: Differentiable Wavelet + RCS非负约束

---

## 🔄 调用流程

```python
from autoencoder.utils.frequency_config import create_autoencoder_system

# 基础架构
system = create_autoencoder_system(
    config_name='2freq',
    mode='wavelet',        # 'wavelet', 'direct', 'differentiable_wavelet'
    architecture='cnn',    # 'cnn', 'mlp', 'enhanced_cnn', 'deep_cnn'
    latent_dim=256,
    activation='relu'      # 'relu', 'sin', 'gelu', 'swish', 'tanh', 'mish'
)

# 双分支架构
system = create_autoencoder_system(
    config_name='2freq',
    mode='differentiable_wavelet',
    architecture='dual_branch_mlp',  # 或 'dual_branch_cnn'
    latent_dim=32
)

# 叠加型双分支
system = create_autoencoder_system(
    config_name='2freq',
    mode='wavelet',
    architecture='additive_dual_branch_cnn',
    latent_dim=256,
    activation_encoder='relu',
    activation_high='sin',
    activation_smooth='tanh',
    learnable_weights=False,
    alpha_high=0.5,
    alpha_smooth=0.5
)
```

---

## 📈 性能参考

| 模型 | 训练速度 | 推理速度 | 参数量 | 重建精度 |
|------|---------|---------|--------|---------|
| WaveletAutoEncoder | ★★★★☆ | ★★★★★ | ~1.5M | ★★★☆☆ |
| DirectAutoEncoder | ★★★☆☆ | ★★★★☆ | ~2.5M | ★★★☆☆ |
| EnhancedWaveletAutoEncoder | ★★★☆☆ | ★★★☆☆ | ~11M | ★★★★☆ |
| DeepWaveletAutoEncoder | ★★☆☆☆ | ★★☆☆☆ | ~29M | ★★★★★ |
| AdditiveDualBranchWaveletAutoEncoder | ★★★☆☆ | ★★★☆☆ | ~3M | ★★★★☆ |

---

## ✅ 最佳实践

1. **初次使用**: 从标准CNN开始（`WaveletAutoEncoder`）
2. **需要更好性能**: 尝试Enhanced CNN或Deep CNN
3. **特殊需求**:
   - 高频细节：Additive Dual-Branch + Sin激活
   - LL/HF分离：Dual-Branch V2
   - 物理约束：Differentiable Wavelet模式
4. **参数调优**:
   - latent_dim: 32-256（越大越精确，但越慢）
   - dropout_rate: 0.2-0.3（防止过拟合）
   - activation: relu（稳定）, sin（高频）, gelu（平滑）

---

生成时间: 2025-12-23
维护者: Claude Code
