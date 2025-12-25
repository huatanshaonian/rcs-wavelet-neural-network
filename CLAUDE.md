# Claude项目上下文文档

> **最后更新**: 2025-12-25
> **项目**: RCS预测AutoEncoder系统
> **核心技术**: PyTorch + 小波变换 + AutoEncoder

---

## 🎯 项目概述

基于AutoEncoder的雷达散射截面（RCS）预测系统，支持多频率配置（2freq/3freq），使用小波变换增强特征提取，实现参数→RCS的快速预测。

**核心目标**：用AutoEncoder压缩RCS数据，通过参数映射器实现参数空间到隐空间的映射，快速预测任意参数下的RCS。

---

## 📋 核心架构

### 1. AutoEncoder网络架构体系

**命名规范**: `<Mode><Architecture>AutoEncoder`

#### 1.1 基础架构（Wavelet/Direct双模式）

| 模式 (Mode) | 架构 (Architecture) | 类名 | 文件 | GUI选项 |
|------------|---------------------|------|------|---------|
| Wavelet | 标准CNN (默认) | `WaveletAutoEncoder` | `cnn_autoencoder.py` | `cnn` |
| Wavelet | MLP | `WaveletMLPAutoEncoder` | `mlp_autoencoder.py` | `mlp` |
| Wavelet | Enhanced CNN | `EnhancedWaveletAutoEncoder` | `enhanced_cnn_autoencoder.py` | `enhanced_cnn` |
| Wavelet | Deep CNN | `DeepWaveletAutoEncoder` | `deep_autoencoder.py` | `deep_cnn` |
| Direct | 标准CNN (默认) | `DirectAutoEncoder` | `direct_autoencoder.py` | `cnn` |
| Direct | MLP | `DirectMLPAutoEncoder` | `mlp_autoencoder.py` | `mlp` |
| Direct | Enhanced CNN | `EnhancedDirectAutoEncoder` | `enhanced_cnn_autoencoder.py` | `enhanced_cnn` |
| Direct | Deep CNN | `DeepDirectAutoEncoder` | `deep_autoencoder.py` | `deep_cnn` |

**模式说明**:
- **Wavelet模式**: RCS → 小波变换 → [49×49×8] 小波系数 → AutoEncoder
- **Direct模式**: RCS [91×91×2] → 直接输入AutoEncoder（无小波变换）

**架构说明**:
- **标准CNN**: 4层encoder + 4层decoder，平衡性能和速度（推荐默认）
- **MLP**: 5层全连接，适合参数敏感性分析
- **Enhanced CNN**: 多尺度卷积 + 空洞残差 + 通道注意力，更大感受野
- **Deep CNN**: 4层深度卷积 + 双卷积块 + 通道注意力，最强表达力

#### 1.2 可微分小波模式（Differentiable Wavelet）

**核心特点**: 小波变换集成为nn.Module，损失在RCS空间计算，梯度可微分回传
- ✅ 端到端训练，无需单独的小波/逆小波步骤
- ✅ 适合物理约束（如RCS非负）的直接应用

| 架构 | 类名 | 文件 | GUI选项 |
|------|------|------|---------|
| 可微分CNN | `DifferentiableWaveletAutoEncoder` | `differentiable_wavelet_autoencoder.py` | `cnn` (mode=differentiable_wavelet) |
| 可微分MLP | `DifferentiableWaveletMLPAutoEncoder` | `differentiable_wavelet_autoencoder.py` | `mlp` (mode=differentiable_wavelet) |

#### 1.3 双分支架构（Dual-Branch）

**用途**: 分别处理LL通道（90%+能量）和HF通道（<10%能量），实现更精细的特征解耦

##### 1.3.1 分离型双分支（Differentiable Wavelet模式，V2推荐）

| 架构 | 类名 | 文件 | GUI选项 | 说明 |
|------|------|------|---------|------|
| Dual-Branch CNN V2 | `DualBranchDifferentiableWaveletAutoEncoderV2` | `dual_branch_differentiable_autoencoder_v2.py` | `dual_branch_cnn` | **✅ 推荐**：正确对称架构 |
| Dual-Branch MLP V2 | `DualBranchDifferentiableWaveletMLPAutoEncoderV2` | `dual_branch_differentiable_autoencoder_v2.py` | `dual_branch_mlp` | **✅ 推荐**：正确对称架构 |
| Dual-Branch CNN V1 | `DualBranchDifferentiableWaveletAutoEncoder` | `dual_branch_differentiable_autoencoder.py` | `dual_branch_cnn_v1` | ⚠️ 旧版（架构缺陷，仅向后兼容） |
| Dual-Branch MLP V1 | `DualBranchDifferentiableWaveletMLPAutoEncoder` | `dual_branch_differentiable_autoencoder.py` | `dual_branch_mlp_v1` | ⚠️ 旧版（架构缺陷，仅向后兼容） |

**V2 vs V1 关键区别**:
- ✅ V2: Decoder也是双分支（ll_decoder + hf_decoder），架构对称
- ✅ V2: ll_latent_dim和hf_latent_dim真正生效（V1硬编码为128）
- ✅ V2: ll_ratio参数实际控制latent空间分配（V1无作用）
- ✅ V2: _combine_channels()保证通道顺序正确（V1缺失）
- 详见: `docs/DUAL_BRANCH_V1_VS_V2_COMPARISON.md`

**使用示例**:
```python
from autoencoder.utils.frequency_config import create_autoencoder_system

# 创建V2双分支MLP (推荐)
system = create_autoencoder_system(
    config_name='2freq',
    mode='differentiable_wavelet',
    architecture='dual_branch_mlp',  # V2为默认
    latent_dim=32,
    activation='sin'
)

# ll_ratio控制LL/HF隐空间分配
# latent_dim=32, ll_ratio=0.7 → ll_latent=22, hf_latent=10
```

##### 1.3.2 叠加型双分支（Additive Dual-Branch，新架构⭐）

**核心思想**: 双Decoder分别学习高频和低频特征，输出加权叠加

```
输入 → Encoder → Latent
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
Decoder_HighFreq          Decoder_Smooth
(Sin激活)                 (Tanh/Swish激活)
        ↓                       ↓
  高频特征重建             低频趋势重建
        └───────────┬───────────┘
                    ↓
          输出 = α·高频 + β·低频
```

| 模式 | 架构 | 类名 | 文件 | GUI选项 |
|------|------|------|------|---------|
| Wavelet | Additive CNN | `AdditiveDualBranchWaveletAutoEncoder` | `additive_dual_branch_autoencoder.py` | `additive_dual_branch_cnn` |
| Wavelet | Additive MLP | `AdditiveDualBranchWaveletMLPAutoEncoder` | `additive_dual_branch_mlp.py` | `additive_dual_branch_mlp` |
| Direct | Additive CNN | `AdditiveDualBranchDirectAutoEncoder` | `additive_dual_branch_autoencoder.py` | `additive_dual_branch_cnn` |
| Direct | Additive MLP | `AdditiveDualBranchDirectMLPAutoEncoder` | `additive_dual_branch_mlp.py` | `additive_dual_branch_mlp` |

**优势**:
- ✅ **Sin激活**：学习高频振荡、细节特征
- ✅ **Smooth激活**（Tanh/Swish）：学习低频趋势、整体统计特性
- ✅ **输出叠加**：兼顾高频和低频，提升重建质量
- ✅ **可学习权重**：支持固定权重或可学习权重（`learnable_weights=True`）

**使用示例**:
```python
# 创建叠加型双分支CNN
system = create_autoencoder_system(
    config_name='2freq',
    mode='wavelet',
    architecture='additive_dual_branch_cnn',
    latent_dim=256,
    activation_encoder='relu',     # Encoder激活
    activation_high='sin',          # 高频分支激活
    activation_smooth='tanh',       # 低频分支激活
    learnable_weights=False,        # 是否学习权重
    alpha_high=0.5,                 # 高频权重（固定）
    alpha_smooth=0.5                # 低频权重（固定）
)
```

### 2. 频率配置

```python
# 2freq: 当前标准配置
frequencies = [1.5GHz, 3.0GHz]
input_channels = 8  # 2频率 × 4小波带 (LL, LH, HL, HH)

# 3freq: 扩展配置（包含6GHz）
frequencies = [1.5GHz, 3.0GHz, 6.0GHz]
input_channels = 12  # 3频率 × 4小波带
```

### 3. 三阶段训练流程

**Stage 1**: AutoEncoder预训练（重建损失）
- 目标: 学习RCS数据的压缩表示
- 损失: MSE(重建, 原始)
- Wavelet模式: 小波系数重建
- Direct模式: RCS数据重建

**Stage 2**: 参数映射器训练（映射损失）
- 目标: 建立参数→隐空间的映射
- 损失: MSE(predicted_latent, autoencoder_latent)
- AutoEncoder冻结

**Stage 3**: 端到端微调（重建损失）
- 目标: 联合优化整个系统
- 损失: MSE(最终重建, 原始RCS)
- 全部解冻

---

## ⚙️ 核心文件结构

```
wavelet/
├── gui.py                          # 主GUI（待重构）
├── main.py                         # 命令行入口
├── gui_managers/                   # GUI管理器模块
│   ├── extensions/                 # GUI扩展模块
│   │   ├── gui_autoencoder_extension.py # AutoEncoder扩展
│   │   └── gui_batch_experiment_extension.py # 批量实验扩展
│   ├── managers/                   # 业务逻辑管理器
│   ├── tabs/                       # 界面标签页
│   └── trainers/                   # 训练器封装
├── scripts/                        # 诊断、修复与运行脚本
├── tools/                          # 可视化与教学工具
├── networks/                       # 网络定义与管理
│   ├── network_registry.py         # 网络注册系统
│   └── example_networks.py         # 示例网络
├── CLAUDE.md                       # 本文档
├── README.md                       # 完整项目文档
├── PARAMETERS_REFERENCE.md        # 参数参考文档
├── DESIGN_ANALYSIS.md             # 设计分析文档
│
├── autoencoder/
│   ├── models/                     # 网络定义
│   │   ├── __init__.py            # 导出6个核心网络
│   │   ├── cnn_autoencoder.py     # WaveletAutoEncoder
│   │   ├── direct_autoencoder.py  # DirectAutoEncoder
│   │   ├── mlp_autoencoder.py     # MLP系列
│   │   ├── enhanced_cnn_autoencoder.py  # Enhanced系列
│   │   ├── parameter_mapper.py    # 参数映射器
│   │   ├── MODEL_INVENTORY.md     # 模型清单文档
│   │   └── experimental/          # 实验性模型
│   │
│   ├── utils/                      # 工具模块
│   │   ├── frequency_config.py    # 创建AutoEncoder系统
│   │   ├── correct_wavelet_transform.py  # 小波变换
│   │   ├── data_adapters.py       # 数据预处理
│   │   ├── comparison_system.py   # 网络对比框架
│   │   ├── batch_experiment.py    # 批量实验管理器
│   │   ├── configurable_loss.py   # 可配置损失函数
│   │   └── data_cache.py          # 数据缓存工具
│   │
│   └── training/
│       └── ae_trainer.py          # 训练器
│
├── data_processing/
│   ├── data_loader.py             # 数据加载
│   └── data_preprocessor.py       # 预处理
│
└── batch_experiments/             # 批量实验结果目录
    └── experiment_name_timestamp/
```

---

## 🔧 工作流程规范

### Git Commit规范

**必须遵守**（用户明确要求）：
- ✅ 每次修改都要commit
- ✅ 必须写清楚修改目的
- ✅ 使用约定格式

**Commit格式**：
```
<type>: <简短描述>

修改目的：
- <详细说明为什么做这个修改>
- <解决了什么问题>

<具体修改内容>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Type类型**：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关

**描述规范** (重要！)：
- ✅ 简短且具体描述出错原因："decoder输出缺少逆标准化"
- ✅ 清晰说明技术细节："预测RCS值停留在标准化空间"
- ❌ 避免模糊描述："修复XXX的严重Bug"
- ❌ 避免情绪化用词："严重"、"重大"等

### 代码规范

1. **AutoEncoder模型必须提供统一接口**
   - **所有新模型必须提供 `encoder` 和 `decoder` 属性**
   - 用于训练时的冻结/解冻操作

   ```python
   # ✅ 正确：模型有encoder/decoder属性
   class MyAutoEncoder(nn.Module):
       def __init__(self):
           super().__init__()
           # 方式1: 使用nn.Sequential（推荐CNN/MLP）
           self.encoder = nn.Sequential(...)
           self.decoder = nn.Sequential(...)

           # 方式2: 使用nn.ModuleList（推荐Complex架构）
           self.encoder = nn.ModuleList([self.conv1, self.conv2, ...])
           self.decoder = nn.ModuleList([self.deconv1, self.deconv2, ...])

       def encode(self, x):
           # 编码逻辑
           pass

       def decode(self, latent):
           # 解码逻辑
           pass
   ```

   - **必需方法**: `encode(x)`, `decode(latent)`, `forward(x)`
   - **统一接口原因**:
     - 阶段2训练需要冻结encoder: `for param in model.encoder.parameters(): param.requires_grad = False`
     - 阶段2结束需要解冻encoder
     - 如果缺少统一接口会导致训练失败

2. **不使用softplus/clip机制**
   - 原因: 让错误数据及时暴露，而不是隐藏
   - 如果数据完全错误应该能立即分辨

3. **⚠️ 数据预处理顺序至关重要！**
   - **所有训练阶段必须使用data_adapter进行数据预处理**
   - **关键**: 小波变换必须在原始线性数据上运行，不能在标准化后的数据上！
   - 保证标准化/对数变换的一致性

   ```python
   # ✅ 正确：小波模式先变换再标准化
   data_adapter = self.ae_system['data_adapter']

   if mode == 'wavelet':
       # Step 1: 在原始RCS数据上做小波变换
       wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
       # Step 2: 对小波系数进行标准化
       input_data = data_adapter.adapt_rcs_data(wavelet_coeffs)
   else:
       # Direct模式: 直接标准化RCS
       input_data = data_adapter.adapt_rcs_data(rcs_data)

   # ❌ 错误1：先标准化再小波变换（破坏小波基正交性！）
   adapted_data = data_adapter.adapt_rcs_data(rcs_data)  # ❌
   wavelet_coeffs = wavelet_transform.forward_transform(adapted_data)  # ❌

   # ❌ 错误2：完全不使用标准化
   rcs_tensor = torch.FloatTensor(rcs_data)  # ❌
   ```

   - **为什么重要**:
     - RCS数据范围通常很大（-50~50 dBsm）
     - 不标准化会导致训练不稳定、收敛慢
     - 不同频率可能有不同的数值范围，需要独立标准化

   - **数据预处理选项**（GUI中可配置）:
     - **标准化 (normalize)**: Z-score标准化，每个频率独立进行（强烈推荐）
     - **对数变换 (log_transform)**: sign(x)*log(|x|)，压缩动态范围（可选）

4. **⚠️ decoder输出必须正确逆变换！**
   - **关键原则**: decoder输出在**标准化空间**，任何评估/可视化前必须逆变换到原始RCS空间
   - **训练时**: 损失在标准化空间计算（正确✅）
   - **推理时**: decoder输出必须逆变换（容易漏！❌）

   ```python
   # ✅ 正确：Three Stage评估/可视化
   predicted_latents = parameter_mapper(params)
   predicted_output = autoencoder.decode(predicted_latents)  # 标准化空间

   # 获取data_adapter
   data_adapter = self.ae_system.get('data_adapter', None)

   if mode == 'wavelet':
       # Wavelet: 标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
       if data_adapter:
           predicted_output_np = predicted_output.cpu().numpy()
           predicted_coeffs = data_adapter.inverse_adapt(predicted_output_np)
           predicted_coeffs = torch.FloatTensor(predicted_coeffs).to(device)
       predicted_rcs = wavelet_transform.inverse_transform(predicted_coeffs)
   else:
       # Direct: 标准化RCS → 逆标准化（逆dB + 逆Z-score） → RCS
       if data_adapter:
           predicted_output_np = predicted_output.cpu().numpy()
           predicted_rcs = data_adapter.inverse_adapt(predicted_output_np)
           predicted_rcs = torch.FloatTensor(predicted_rcs).to(device)

   # ❌ 错误1：Wavelet模式只做逆小波，忘记逆标准化
   predicted_rcs = wavelet_transform.inverse_transform(predicted_output)  # ❌

   # ❌ 错误2：Direct模式直接使用decoder输出
   predicted_rcs = predicted_output  # ❌ 停留在标准化空间！
   ```

   - **为什么容易出错**:
     - 训练代码不需要逆变换，容易形成思维定势
     - 小波模式的逆小波变换容易让人误以为已经完成全部逆变换
     - Direct模式更容易忘记，因为没有逆小波这个提示

   - **检查清单**:
     - [ ] 评估函数调用了`data_adapter.inverse_adapt()`？
     - [ ] 可视化函数调用了`data_adapter.inverse_adapt()`？
     - [ ] 统计对比函数调用了`data_adapter.inverse_adapt()`？
     - [ ] 任何显示/保存RCS预测值的地方都逆变换了？

5. **损失计算必须sample-weighted**
   ```python
   # ❌ 错误：batch averaging
   train_loss += loss.item()
   avg_loss = train_loss / num_batches

   # ✅ 正确：sample-weighted averaging
   batch_size = data.size(0)
   train_loss += loss.item() * batch_size
   total_samples += batch_size
   avg_loss = train_loss / total_samples
   ```
   - 原因: 训练集用`drop_last=True`，验证集用`drop_last=False`
   - 最后一个batch可能更小，不能给予相同权重

6. **小波变换统一使用CorrectWaveletTransform**
   - 所有小波操作必须通过`correct_wavelet_transform.py`
   - 不要直接调用pywt（确保一致性）

---

## ⚠️ 已知问题和注意事项

### 1. 训练损失正常现象

**现象**: Train Loss > Val Loss（约10倍）
```
Train Loss: 0.0071817
Val Loss:   0.0005653
```

**原因**: Dropout=0.2
- 训练时: 20%神经元被dropout → 容量降低 → 损失更高
- 验证时: 全部神经元激活 → 满容量 → 损失更低

**结论**: ✅ 这是正常的，不是bug

### 2. 频率配置验证

系统会自动验证模型和数据的频率匹配：
```python
# 如果用3freq模型加载2freq数据会报错
AssertionError: 频率数量不匹配: 模型期望3个频率, 数据有2个频率
```

这是有意设计的保护机制，防止系统性错误。

### 3. 网络注册优化

GUI启动时会注册多个网络，已优化为静默跳过已注册网络，避免重复警告。

### 4. 模型文件组织

- ✅ **正在使用**: `autoencoder/models/*.py` (6个核心文件)
- ⚠️ **实验性**: `autoencoder/models/experimental/` (4个，未使用)
- 详见: `autoencoder/models/MODEL_INVENTORY.md`

---

## 🚀 常用操作

### 创建AutoEncoder系统

```python
from autoencoder.utils.frequency_config import create_autoencoder_system

# Wavelet模式 + 标准CNN
system = create_autoencoder_system(
    config_name='2freq',
    mode='wavelet',
    architecture='cnn',
    latent_dim=256
)

# Direct模式 + MLP
system = create_autoencoder_system(
    config_name='2freq',
    mode='direct',
    architecture='mlp',
    latent_dim=256
)

# Wavelet模式 + Enhanced CNN
system = create_autoencoder_system(
    config_name='3freq',  # 6GHz扩展
    mode='wavelet',
    architecture='enhanced_cnn',
    latent_dim=256
)
```

### 测试系统可用性

```bash
# 测试导入
python -c "from autoencoder.models import WaveletAutoEncoder, DirectAutoEncoder; print('Import OK')"

# 测试系统创建
python -c "from autoencoder.utils.frequency_config import create_autoencoder_system; system = create_autoencoder_system('2freq', 'wavelet', 'cnn', 256); print(f'System OK: {type(system[\"autoencoder\"]).__name__}')"

# 启动GUI
python gui.py
```

### 独立训练脚本

```bash
# 使用debug脚本
python debug_ae_training.py
```

---

## 🔍 调试技巧

### 1. 查看模型参数量

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"AutoEncoder: {count_parameters(autoencoder):,}")
print(f"Parameter Mapper: {count_parameters(param_mapper):,}")
```

### 2. 检查小波变换尺寸

```python
# Wavelet模式
# 输入: [B, 91, 91, 2]
# 小波变换: [B, 49, 49, 8]
# AE输出: [B, 256]

# Direct模式
# 输入: [B, 91, 91, 2]
# AE输出: [B, 256]
```

### 3. 查看训练进度

GUI会自动可视化三阶段训练曲线，保存在`ae_checkpoints/`目录。

---

## 📊 性能对比

| 网络 | 参数量 | 推理速度 | 适用场景 |
|------|--------|---------|---------|
| WaveletAutoEncoder | ~1.5M | 快 | 通用，推荐默认 |
| DirectAutoEncoder | ~2.5M | 中等 | 无小波开销场景 |
| WaveletMLPAutoEncoder | ~3M | 慢 | 参数敏感性分析 |
| DirectMLPAutoEncoder | ~5M | 很慢 | 实验性 |
| EnhancedWaveletAutoEncoder | ~11M | 中等 | 复杂模式，更大感受野 |
| EnhancedDirectAutoEncoder | ~25M | 慢 | Direct模式最强表达力 |
| DeepWaveletAutoEncoder | ~29M | 慢 | Wavelet模式最强表达力 |
| DeepDirectAutoEncoder | ~79M | 很慢 | 最强表达力，计算密集 |

---

## 💡 开发建议

### 添加新架构的流程

如果要添加新的实验性架构（如DeepCNN）:

1. **创建模型文件**（遵循命名规范）
   ```python
   # DeepWaveletAutoEncoder + DeepDirectAutoEncoder
   class DeepWaveletAutoEncoder(nn.Module):
       def __init__(self, latent_dim=256, ...):
           super().__init__()
           # 定义网络层...
           self.conv1 = ...
           self.conv2 = ...

           # ⚠️ 必须：提供统一的encoder/decoder接口
           self.encoder = nn.ModuleList([self.conv1, self.conv2, ...])
           self.decoder = nn.ModuleList([self.deconv1, self.deconv2, ...])

       def encode(self, x):
           # 编码逻辑
           pass

       def decode(self, latent):
           # 解码逻辑
           pass

       def forward(self, x):
           latent = self.encode(x)
           recon = self.decode(latent)
           return recon, latent
   ```

2. **修改frequency_config.py**
   ```python
   if mode == 'wavelet':
       if architecture == 'deep_cnn':
           autoencoder = DeepWaveletAutoEncoder(...)
   ```

3. **更新__init__.py**
   ```python
   from .deep_autoencoder import DeepWaveletAutoEncoder, DeepDirectAutoEncoder
   __all__ = [..., 'DeepWaveletAutoEncoder', 'DeepDirectAutoEncoder']
   ```

4. **充分测试再集成**
   - 先在小数据集验证
   - 对比现有6个网络的性能
   - 记录实验结果

---

## 📌 重要技术细节

### 小波变换尺寸

```
原始RCS: 91×91
小波变换: 49×49 (使用db4小波, symmetric边界)

计算公式:
wavelet_size = (original_size + wavelet_filter_length - 1) // 2
49 = (91 + 7 - 1) // 2
```

### 数据增强

系统支持6GHz频率扩展（3freq模式）:
- 基于1.5GHz和3GHz数据学习外插规律
- 生成合理的6GHz数据估计
- 详见: `autoencoder/utils/frequency_extension.py`

### Dropout率

当前固定为0.2（所有网络）:
- 防止过拟合
- 提高泛化能力
- 导致train_loss > val_loss（正常）

---

## 🎓 学习资源

### 核心文档
- **项目结构**: `PROJECT_STRUCTURE.md` - 功能快速索引（搜索代码必看⭐）
- **项目README**: `README.md` - 完整项目文档
- **本文档**: `CLAUDE.md` - Claude工作上下文（核心参考）
- **参数参考**: `PARAMETERS_REFERENCE.md` - ⚠️ 所有参数命名规范（必看）
- **设计分析**: `DESIGN_ANALYSIS.md` - 架构设计问题分析

### 技术文档
- **技术文档索引**: `docs/README.md` - 所有技术文档导航
- **数据流程**: `docs/DATA_PIPELINE.md` - 数据预处理完整说明
- **架构分析**: `docs/architecture/` - CNN/MLP架构详解

### 模型相关
- **模型清单**: `autoencoder/models/MODEL_INVENTORY.md` - 8个核心网络说明
- **实验性模型**: `autoencoder/models/experimental/README.md` - 4个实验性网络

---

## 🔄 最近更新记录

### 2025-01-18 (下午)

1. **🆕 L-BFGS优化器完整支持** (新功能)
   - **需求**: 用户请求添加L-BFGS二阶优化方法
   - **问题发现**: 优化器配置GUI参数完全未生效（自2025-10-19引入bug）
   - **修复内容**:
     - ✅ **Bug修复**: `training_manager.py:287-289`添加optimizer_type/momentum到配置字典
     - ✅ **优化器支持**: `ae_trainer.py:730-782`重构优化器创建，支持Adam/AdamW/SGD/L-BFGS
     - ✅ **训练循环适配**: 所有三个训练阶段支持L-BFGS闭包机制
     - ✅ **GUI更新**: `gui_autoencoder_extension.py:274`添加lbfgs选项
   - **L-BFGS实现细节**:
     - Stage 1/2: `_train_batch_with_lbfgs()`自动检测优化器并使用闭包
     - Stage 3: 端到端训练内联闭包（联合优化autoencoder+parameter_mapper）
     - Line Search: PyTorch内置strong Wolfe条件，每step 5-20次前向传播
   - **使用方法**:
     ```python
     # GUI配置
     optimizer_type = "lbfgs"
     learning_rate = 1.0  # L-BFGS不需要小学习率
     batch_size = 256  # 建议大批量减少噪声
     # lbfgs_max_iter = 20  # 每step最大迭代（默认）
     # lbfgs_history_size = 100  # 历史梯度数量（默认）
     ```
   - **性能特征**:
     - 速度: 每epoch慢3-8倍（line search开销）
     - 收敛: 需要epoch数减少2-3倍
     - 精度: 通常找到更优解（二阶方法）
     - 适用: Stage 3微调、小数据集（<5000样本）
   - **注意事项**:
     - ⚠️ L-BFGS不支持梯度监控（多次前向传播干扰）
     - ⚠️ 建议batch_size≥256，否则噪声影响line search
     - ⚠️ GPU利用率可能较低（频繁CPU-GPU通信）
   - **影响文件**:
     - `gui_managers/managers/training_manager.py`: 读取优化器配置
     - `gui_managers/trainers/ae_trainer.py`: L-BFGS训练循环
     - `gui_autoencoder_extension.py`: GUI优化器下拉框
   - **Commits**: cc5572b (bug修复), 206352c (L-BFGS实现)

2. **🆕 Dual-Branch V2正确实现** (重大架构修复)
   - **问题**: V1版本DualBranchDifferentiableAutoEncoder存在严重架构缺陷
   - **核心缺陷**:
     - Encoder有双分支，但Decoder是单分支（不对称）
     - ll_latent_dim和hf_latent_dim被计算但未使用（硬编码为128）
     - ll_ratio参数无作用
     - 缺少_combine_channels()导致通道顺序可能错误
   - **V2修复**:
     - ✅ Decoder也实现双分支（ll_decoder + hf_decoder）
     - ✅ ll_latent_dim和hf_latent_dim真正生效
     - ✅ ll_ratio参数实际控制latent空间分配
     - ✅ 添加_combine_channels()保证通道顺序正确
     - ✅ 移除无用fusion层，直接concat
   - **新增文件**:
     - `autoencoder/models/dual_branch_differentiable_autoencoder_v2.py`
     - `docs/DUAL_BRANCH_V1_VS_V2_COMPARISON.md`: 详细对比文档
   - **影响文件**:
     - `autoencoder/models/__init__.py`: 导出V2模型
     - `autoencoder/utils/frequency_config.py`: 支持创建V2
   - **使用方法**:
     ```python
     system = create_autoencoder_system(
         architecture='dual_branch_mlp_v2',  # 或 'dual_branch_cnn_v2'
         latent_dim=32
     )
     ```
   - **验证结果**:
     - MLP V2: 20.2M参数, ll_latent=22, hf_latent=10
     - CNN V2: 2.2M参数, 架构对称
     - 通道顺序测试通过
   - **推荐**: 新项目使用V2，V1仅保留向后兼容
   - **Commits**: ce57f9c
   - **相关文档**:
     - `docs/DUAL_BRANCH_MLP_ANALYSIS.md`: V1问题分析
     - `docs/CONCAT_IMPLEMENTATION_ISSUE.md`: 通道顺序问题
     - `docs/DUAL_BRANCH_CORRECT_IMPLEMENTATION_PLAN.md`: V2实现方案
     - `docs/WAVELET_CHANNEL_ORDER_EXPLAINED.md`: 通道顺序说明

### 2025-01-06

1. **模型保存功能增强** (用户体验改进)
   - **需求**: 用户希望保存模型时能快速识别模型配置
   - **实现**:
     - **预设文件名**: 自动生成包含关键参数的文件名
       - 格式: `{mode}_{architecture}_{activation}_{preprocess}_{timestamp}.pth`
       - 例如: `wavelet_cnn_relu_norm_db_20250106_143022.pth`
       - 预处理标签说明：
         - `norm_db`: 标准化 + dB变换
         - `norm`: 仅标准化
         - `db`: 仅dB变换
         - `raw`: 无预处理
     - **自动生成JSON配置文件**: 保存.pth时同步生成同名_config.json文件
       - 包含完整的模型信息、频率配置、数据预处理设置、训练历史
       - 文件名: `{model_name}_config.json`
   - **优点**:
     - 文件名包含最关键的架构和训练配置（激活函数、预处理方式）
     - JSON配置文件提供完整详细信息（隐空间维度、频率配置等）
     - 两个文件同名（除扩展名），便于一起管理
   - **影响文件**:
     - `gui.py:2870-3003` - 修改save_ae_model函数
   - **JSON配置文件内容**:
     - model_info: mode, architecture, latent_dim, dropout_rate, activation
     - frequency_config: 频率数量、频率标签、频率值
     - data_preprocessing: 小波类型、是否标准化、是否dB变换、统计信息
     - training_info: 训练模式（three_stage/stage1_only）、训练历史
     - save_info: 保存时间、文件名
   - **Commits**: ab41cec (初始实现)

### 2025-01-18

1. **decoder输出缺少逆标准化步骤** (数据后处理Bug修复)
   - **问题**: Three Stage评估和可视化函数中，decoder输出在标准化空间未逆变换
   - **影响**:
     - 评估指标不准确（基于错误的数据空间）
     - 所有可视化图形显示错误的数值范围（-3~+3而非0.00000009~0.5）
     - 残差图、MSE/RMSE/MAE指标无法反映真实误差
   - **根本原因**:
     - 训练时损失在标准化空间计算（正确）
     - 推理时decoder输出直接使用，忘记逆变换回原始RCS空间
     - Wavelet模式只做了逆小波变换，缺少逆标准化
     - Direct模式完全缺少逆dB和逆Z-score变换
   - **修复内容**:
     - `gui.py:2949-2971` - 修复`_evaluate_autoencoder_model`评估函数
     - `gui.py:7571-7602` - 修复`_plot_ae_comparison`对比图函数
     - `gui.py:7392-7416` - 修复`_plot_ae_2d_heatmap` 2D热图函数
     - `gui.py:4172-4193` - 修复`_plot_global_statistics_comparison`统计对比函数
   - **修复方案**:
     ```python
     # Wavelet: decoder输出 → 逆标准化 → 逆小波变换 → RCS
     # Direct: decoder输出 → 逆标准化（逆dB + 逆Z-score） → RCS
     ```
   - **预期效果**:
     - 评估指标准确反映模型性能
     - 可视化显示正确的RCS值范围
     - 残差图真实展示预测误差
   - **代码规范更新**:
     - 添加"decoder输出必须正确逆变换"规范（代码规范#4）
     - 添加检查清单防止类似错误
   - **Commits**: 832ddd8 (评估函数), b39e8a8 (3个可视化函数)

### 2025-01-14

0. **🆕 Stage 1 Only训练模式** (新功能 - 专注重建性能研究)
   - **需求**: 用户要求排除参数映射器干扰，专注研究AutoEncoder重建性能
   - **功能**: 添加仅Stage 1训练模式（stage1_only）
   - **实现内容**:
     - 训练配置界面添加训练模式选择（three_stage / stage1_only）
     - 训练流程自动识别模式，stage1_only只执行AutoEncoder预训练
     - 模型保存/加载包含training_mode标记
     - 评估功能自动适配：
       - stage1_only: 直接测试RCS重建（RCS → Encoder → Decoder → 重建RCS）
       - three_stage: 从参数预测RCS（参数 → ParameterMapper → Decoder → RCS）
   - **影响文件**:
     - `gui_autoencoder_extension.py`: 主GUI添加"仅Stage 1"选项，训练模式UI
     - `gui.py`: _run_three_stage_training_v2支持两种训练模式
     - `gui.py`: save_ae_model/load_ae_model保存/加载training_mode
     - `gui.py`: _evaluate_autoencoder_model适配两种评估方式
   - **使用场景**:
     - AutoEncoder架构对比研究
     - 重建性能调参优化
     - 快速验证模型重建能力
     - 不需要参数映射器的应用场景
   - **用户体验改进**:
     - commit 0cf33fb: 在gui.py旧界面添加选项+中英文映射
     - commit ca5af90: 更新文档记录
     - commit f233635: 在gui_autoencoder_extension.py正确位置添加选项 ✅
     - 主GUI训练模式下拉框添加"仅Stage 1"选项
     - 实现中文选项→英文标识符的自动映射
     - 用户无需打开训练配置对话框即可选择Stage 1 Only模式
     - 训练信息输出添加Stage 1 Only模式说明
   - **重要说明**:
     - 实际使用的界面在`gui_autoencoder_extension.py`
     - `gui.py`中的旧界面已标记废弃，仅作向后兼容
   - **Commits**: 2f470a8 (训练流程), fa1e7f2 (评估适配), f233635 (主GUI修复)

1. **🚨 数据处理顺序Bug修复** (严重Bug⚠️⚠️⚠️)
   - **问题**: 所有三个训练阶段都在**标准化后再做小波变换**！这完全错误！
   - **影响**: 破坏了小波基的正交性，导致小波系数失去物理意义
   - **修复内容**:
     - Stage 1/2/3全部修改为正确顺序: `原始RCS → 小波变换 → 标准化`
     - 添加详细日志输出，清晰显示数据处理步骤
     - 创建DATA_PIPELINE.md完整文档说明数据流程
   - **影响文件**:
     - `gui.py`: 修复_train_autoencoder_stage1_v2 (line 6053-6074)
     - `gui.py`: 修复_train_parameter_mapping_stage2_v2 (line 6231-6251)
     - `gui.py`: 修复_train_end_to_end_stage3_v2 (line 6408-6421)
     - `CLAUDE.md`: 更新代码规范#3，强调数据处理顺序
     - `DATA_PIPELINE.md`: 新建完整数据流程文档
   - **预期效果**: 小波模式性能显著提升，训练更稳定
   - **用户反馈**: "小波变换要在线性的原始数据上运行，请注意不要错误地在小波变换前（如果选择了）进行标准化等操作"

1. **数据标准化集成** (重要修复⚠️)
   - **问题**: 之前的训练**完全没有使用数据标准化**！
   - 修复内容：
     - GUI添加数据预处理配置（标准化+对数变换选项）
     - Stage 1/2/3训练全部集成`RCS_DataAdapter`
     - 保存/加载模型时保存adapter统计信息（mean/std）
     - 默认启用标准化（强烈推荐）
   - 影响文件：
     - `gui_autoencoder_extension.py`: 添加预处理控制器
     - `gui.py`: 三个训练阶段全部使用data_adapter
     - `gui.py`: 保存/加载模型包含adapter统计信息
   - **预期效果**: 训练收敛更快、更稳定、性能明显提升

2. **模型统一接口规范**
   - 为所有模型添加统一的`encoder`/`decoder`属性
   - 修复Enhanced_CNN和Deep_CNN训练时的"no attribute 'encoder'"错误
   - 更新开发规范：所有新模型必须提供统一接口
   - 影响文件：
     - `enhanced_cnn_autoencoder.py`: 添加encoder/decoder ModuleList
     - `deep_autoencoder.py`: 添加encoder/decoder ModuleList
     - `gui.py`: 简化冻结/解冻逻辑
     - `CLAUDE.md`: 添加模型接口规范

2. **网络注册优化** (commit: efba50f)
   - 静默跳过已注册网络，加快启动速度

2. **损失计算修复** (commit: 919f18d)
   - 修复sample-weighted averaging
   - 影响所有三个训练阶段

3. **模式日志修复** (commit: 56b4801)
   - Direct模式显示正确的"RCS数据重建"
   - Wavelet模式显示"小波系数重建"

4. **模型组织优化** (commit: 0a7fb66)
   - 创建MODEL_INVENTORY.md
   - 移动实验性模型到experimental/子目录
   - 明确6个核心网络和4个实验性网络

---

## 🧪 批量实验功能（新增）

**用途**: 自动化对比不同配置的AutoEncoder性能（超参数搜索、架构对比、消融实验）

**核心模块**:
- `autoencoder/utils/batch_experiment.py`: BatchExperimentManager
- `gui_batch_experiment_extension.py`: GUI扩展

**支持对比维度**:
- AE模式（wavelet/direct/differentiable_wavelet）
- 架构类型（CNN/Enhanced_CNN/Deep_CNN/MLP等）
- 激活函数（relu/sin/gelu/swish/tanh/mish等）
- 数据预处理（标准化方法×dB变换）
- 小波类型（db4/db8/haar/bior2.2，仅Wavelet模式）

**自动生成内容**:
- 评估指标：Train/Test MSE/RMSE/MAE
- 对比图表：训练曲线、指标柱状图、雷达图、收敛速度、训练时间等
- 单模型可视化：RCS热图对比、残差分布（训练/测试集各3样本）
- 结果文件：JSON配置、CSV汇总、模型权重

**使用流程**:
1. 在【AutoEncoder】页面配置基准参数
2. 在【批量实验】页面读取配置、选择对比维度
3. 执行批量训练（自动循环所有配置组合）
4. 查看结果目录：`batch_experiments/experiment_name_timestamp/`

**典型场景**: AE模式对比、激活函数对比、架构选择、预处理方案对比、消融实验

详细文档见 `README.md`

---

## ✅ 快速检查清单

开始工作前快速确认：

- [ ] 确认当前频率配置（2freq or 3freq）
- [ ] 确认使用的模式（wavelet or direct）
- [ ] 确认使用的架构（cnn, mlp, or enhanced_cnn）
- [ ] 训练时检查train/val loss比例（10倍内正常）
- [ ] Commit前写清楚修改目的
- [ ] 修改后测试导入和系统创建

---

**维护者**: Claude Code
**项目路径**: `G:\feko_data\wavelet`
**Python环境**: RCS_OP1 (Conda)
- 在本项目中优先使用可微分小波变换
- 在创建新接口或调用老接口时，首先参考PARAMETERS_REFERENCE.md文件
- 在创建新接口或调用老接口时，首先参考PARAMETERS_REFERENCE.md文件