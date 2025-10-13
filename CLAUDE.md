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

### 1. AutoEncoder网络（6个核心网络）

**命名规范**: `<Mode><Architecture>AutoEncoder`

| 模式 (Mode) | 架构 (Architecture) | 类名 | 文件 |
|------------|---------------------|------|------|
| Wavelet | 标准CNN (默认) | `WaveletAutoEncoder` | `cnn_autoencoder.py` |
| Wavelet | MLP | `WaveletMLPAutoEncoder` | `mlp_autoencoder.py` |
| Wavelet | Enhanced CNN | `EnhancedWaveletAutoEncoder` | `enhanced_cnn_autoencoder.py` |
| Direct | 标准CNN (默认) | `DirectAutoEncoder` | `direct_autoencoder.py` |
| Direct | MLP | `DirectMLPAutoEncoder` | `mlp_autoencoder.py` |
| Direct | Enhanced CNN | `EnhancedDirectAutoEncoder` | `enhanced_cnn_autoencoder.py` |

**模式说明**:
- **Wavelet模式**: RCS → 小波变换 → [49×49×8] 小波系数 → AutoEncoder
- **Direct模式**: RCS [91×91×2] → 直接输入AutoEncoder（无小波变换）

**架构说明**:
- **标准CNN**: 4层encoder + 4层decoder，平衡性能和速度
- **MLP**: 5层全连接，适合参数敏感性分析
- **Enhanced CNN**: 多尺度卷积 + 更大感受野，捕捉复杂模式

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

1. **不使用softplus/clip机制**
   - 原因: 让错误数据及时暴露，而不是隐藏
   - 如果数据完全错误应该能立即分辨

2. **损失计算必须sample-weighted**
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
| EnhancedWaveletAutoEncoder | ~2M | 中等 | 复杂模式，需要更大感受野 |
| EnhancedDirectAutoEncoder | ~4M | 慢 | 最强表达力 |

---

## 💡 开发建议

### 添加新架构的流程

如果要添加新的实验性架构（如DeepCNN）:

1. **创建模型文件**（遵循命名规范）
   ```python
   # DeepWaveletAutoEncoder + DeepDirectAutoEncoder
   class DeepWaveletAutoEncoder(nn.Module):
       # 5层深度CNN for wavelet coefficients
       pass
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

1. **网络注册优化** (commit: efba50f)
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
