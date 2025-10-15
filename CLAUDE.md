# Claude项目上下文文档

> **最后更新**: 2025-01-14
> **项目**: RCS预测AutoEncoder系统
> **核心技术**: PyTorch + 小波变换 + AutoEncoder

---

## 🎯 项目概述

基于AutoEncoder的雷达散射截面（RCS）预测系统，支持多频率配置（2freq/3freq），使用小波变换增强特征提取，实现参数→RCS的快速预测。

**核心目标**：用AutoEncoder压缩RCS数据，通过参数映射器实现参数空间到隐空间的映射，快速预测任意参数下的RCS。

---

## 📋 核心架构

### 1. AutoEncoder网络（8个核心网络）

**命名规范**: `<Mode><Architecture>AutoEncoder`

| 模式 (Mode) | 架构 (Architecture) | 类名 | 文件 |
|------------|---------------------|------|------|
| Wavelet | 标准CNN (默认) | `WaveletAutoEncoder` | `cnn_autoencoder.py` |
| Wavelet | MLP | `WaveletMLPAutoEncoder` | `mlp_autoencoder.py` |
| Wavelet | Enhanced CNN | `EnhancedWaveletAutoEncoder` | `enhanced_cnn_autoencoder.py` |
| Wavelet | Deep CNN | `DeepWaveletAutoEncoder` | `deep_autoencoder.py` |
| Direct | 标准CNN (默认) | `DirectAutoEncoder` | `direct_autoencoder.py` |
| Direct | MLP | `DirectMLPAutoEncoder` | `mlp_autoencoder.py` |
| Direct | Enhanced CNN | `EnhancedDirectAutoEncoder` | `enhanced_cnn_autoencoder.py` |
| Direct | Deep CNN | `DeepDirectAutoEncoder` | `deep_autoencoder.py` |

**模式说明**:
- **Wavelet模式**: RCS → 小波变换 → [49×49×8] 小波系数 → AutoEncoder
- **Direct模式**: RCS [91×91×2] → 直接输入AutoEncoder（无小波变换）

**架构说明**:
- **标准CNN**: 4层encoder + 4层decoder，平衡性能和速度（推荐默认）
- **MLP**: 5层全连接，适合参数敏感性分析
- **Enhanced CNN**: 多尺度卷积 + 空洞残差 + 通道注意力，更大感受野
- **Deep CNN**: 4层深度卷积 + 双卷积块 + 通道注意力，最强表达力

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
├── gui.py                          # 主GUI（6000+行，核心界面）
├── gui_autoencoder_extension.py   # AutoEncoder GUI扩展
├── gui_training_config.py         # 训练配置对话框
├── wavelet_gui_helper.py          # 小波分析辅助工具
├── main.py                         # 命令行入口
├── CLAUDE.md                       # 本文档
├── README.md                       # 完整项目文档
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
│   │   └── experimental/          # 实验性/废弃模型
│   │       ├── README.md
│   │       ├── correct_cnn_autoencoder.py  # 废弃
│   │       ├── deep_cnn_autoencoder.py     # 5层深度
│   │       ├── efficient_cnn_autoencoder.py # 轻量3层
│   │       └── micro_latent_autoencoder.py  # 微隐空间
│   │
│   ├── utils/                      # 工具模块
│   │   ├── frequency_config.py    # 创建AutoEncoder系统（核心）
│   │   ├── correct_wavelet_transform.py  # 小波变换
│   │   ├── data_adapter.py        # 数据预处理
│   │   └── comparison_system.py   # 网络对比框架
│   │
│   └── training/
│       └── ae_trainer.py          # 训练器（独立训练脚本用）
│
├── data_processing/
│   ├── data_loader.py             # 数据加载
│   └── data_preprocessor.py       # 预处理
│
└── network_system/
    ├── network_interface.py       # 网络接口基类
    └── network_registry.py        # 网络注册系统
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

4. **损失计算必须sample-weighted**
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

3. **小波变换统一使用CorrectWaveletTransform**
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

- **项目README**: `README.md` - 完整项目文档
- **模型清单**: `autoencoder/models/MODEL_INVENTORY.md`
- **实验性模型**: `autoencoder/models/experimental/README.md`
- **改进方案**: `CNN感受野分析与改进方案.md`, `MLP架构详解.md`

---

## 🔄 最近更新记录

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
     - `gui_training_config.py`: 添加训练模式UI和配置管理
     - `gui.py`: _run_three_stage_training_v2支持两种训练模式
     - `gui.py`: save_ae_model/load_ae_model保存/加载training_mode
     - `gui.py`: _evaluate_autoencoder_model适配两种评估方式
     - `gui_autoencoder_extension.py`: 主GUI添加"仅Stage 1"选项 (✅ 正确位置)
     - `gui.py`: 旧AutoEncoder界面同步更新+废弃标记 (向后兼容)
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
