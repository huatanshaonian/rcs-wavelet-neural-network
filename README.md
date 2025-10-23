# RCS AutoEncoder预测系统

基于AutoEncoder深度压缩的飞行器RCS（雷达散射截面）智能预测系统，通过参数空间到隐空间的映射实现高效RCS数据预测。

## 🚀 项目特色

### 核心架构
- **双模式AutoEncoder系统**:
  - **Wavelet模式**: RCS → 小波变换 → 小波系数压缩 → 256维隐空间
  - **Direct模式**: RCS → 直接CNN压缩 → 256维隐空间
- **三阶段训练流程**:
  - Stage 1: AutoEncoder预训练（学习RCS数据压缩）
  - Stage 2: ParameterMapper训练（建立参数→隐空间映射）
  - Stage 3: 端到端微调（联合优化整个系统）

### 网络架构多样性
提供多种AutoEncoder架构选择：
- **CNN (标准)**: 平衡性能和速度的卷积网络
- **Enhanced_CNN**: 增强感受野，多尺度卷积+空洞残差+通道注意力
- **Deep_CNN**: 深度卷积，双卷积块+通道注意力，最强表达力
- **MLP**: 全连接网络，适合参数敏感性分析

### 智能数据处理
- **Mode-aware自动预处理**: 根据模式自动选择预处理策略
  - Direct模式：自动应用dB变换压缩动态范围（5.4M倍 → 67dB）
  - Wavelet模式：Z-score标准化（保留小波系数37%负值）
- **多频率配置**: 完整支持2频(1.5GHz+3GHz)和3频(1.5GHz+3GHz+6GHz)
- **数据缓存加速**: 智能缓存机制，显著加速数据加载

### 完整工作流
- **现代化GUI**: 基于tkinter的直观操作界面
- **小波分析工具**: 独立小波变换分析和可视化
- **性能对比系统**: Wavelet vs Direct双系统对比分析
- **训练配置管理**: 灵活的训练参数配置和保存

---

## 📁 项目结构

```
wavelet/
├── gui.py                              # 主GUI程序（6000+行）
├── gui_autoencoder_extension.py        # AutoEncoder GUI扩展
├── gui_training_config.py              # 训练配置对话框
├── wavelet_gui_helper.py               # 小波分析辅助工具
├── main.py                             # 命令行入口
├── CLAUDE.md                           # 项目开发指令（重要）
├── README.md                           # 本文档
│
├── autoencoder/                        # AutoEncoder完整系统
│   ├── models/                         # 模型定义
│   │   ├── __init__.py                # 导出核心网络
│   │   ├── cnn_autoencoder.py         # CNN AutoEncoder (Wavelet/Direct)
│   │   ├── direct_autoencoder.py      # Direct CNN AutoEncoder
│   │   ├── mlp_autoencoder.py         # MLP系列（Wavelet/Direct）
│   │   ├── enhanced_cnn_autoencoder.py # Enhanced系列（多尺度）
│   │   ├── deep_autoencoder.py        # Deep CNN系列（最强表达力）
│   │   ├── parameter_mapper.py        # 参数映射器
│   │   ├── MODEL_INVENTORY.md         # 模型清单文档
│   │   └── experimental/              # 实验性模型
│   │       └── README.md
│   ├── training/                       # 训练模块
│   │   └── ae_trainer.py              # 三阶段训练器
│   ├── evaluation/                     # 评估模块
│   │   ├── ae_evaluator.py            # AutoEncoder评估器
│   │   └── reconstruction_metrics.py   # 重建指标计算
│   └── utils/                          # 工具模块
│       ├── frequency_config.py        # 频率配置系统（核心）
│       ├── correct_wavelet_transform.py # 小波变换工具
│       ├── data_adapters.py           # 数据预处理适配器
│       └── comparison_system.py       # 性能对比框架
│
├── data_processing/                    # 数据处理
│   ├── data_loader.py                 # 数据加载
│   ├── data_preprocessor.py           # 预处理
│   └── data_cache.py                  # 缓存管理
│
├── network_system/                     # 网络注册系统
│   ├── network_interface.py           # 网络接口基类
│   └── network_registry.py            # 网络注册中心
│
├── docs/                               # 文档目录
│   ├── architecture/                  # 架构文档
│   │   ├── MLP_Architecture.md       # MLP架构详解
│   │   └── CNN_Receptive_Field_Analysis.md # CNN改进方案
│   ├── DATA_PIPELINE.md              # 数据处理流程
│   ├── autoencoder_development_log.md # 开发日志
│   └── autoencoder_design.md         # 设计文档
│
└── models/                             # 保存的模型文件
    └── ae_checkpoints/                # AutoEncoder检查点
```

---

## 🛠️ 安装和配置

### 1. 环境要求
- Python 3.8+
- CUDA 11.0+ (推荐，GPU加速)
- 内存: 16GB+ (推荐)
- 显存: 8GB+ (推荐)

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 数据准备
确保数据文件位于正确位置：
```
../parameter/
├── parameters_sorted.csv            # 飞行器参数
└── csv_output/                      # RCS数据
    ├── 001_1.5G.csv, 001_3G.csv    # 2频配置
    └── 001_6G.csv, ...              # 3频配置（可选）
```

---

## 🎯 快速开始

### 方式1: GUI界面（推荐）

```bash
python gui.py
```

**操作流程**：

1. **数据管理**标签页
   - 配置数据路径
   - 选择频率配置（2freq/3freq）
   - 加载数据

2. **AutoEncoder**标签页（✅ 主要工作区）
   - **模式选择**: Wavelet / Direct
   - **架构选择**: CNN / Enhanced_CNN / Deep_CNN / MLP
   - **数据预处理**:
     - ✅ 数据标准化（强烈推荐）
     - 📊 dB变换（只读显示，系统自动决定）
   - **创建系统**: 点击"创建当前模式系统"
   - **开始训练**: 配置训练参数后点击"开始训练"
   - **性能对比**: 创建双系统进行Wavelet vs Direct对比

3. **小波分析**标签页
   - 选择模型和频率
   - 运行小波变换分析
   - 查看系数分布和重建效果

### 方式2: Python API

```python
from autoencoder.utils.frequency_config import create_autoencoder_system
import numpy as np

# 1. 创建AutoEncoder系统
system = create_autoencoder_system(
    config_name='2freq',        # 2频或3频
    mode='wavelet',             # wavelet或direct
    architecture='cnn',         # cnn/enhanced_cnn/deep_cnn/mlp
    latent_dim=256,
    normalize=True              # 启用标准化
)

# 2. 获取组件
autoencoder = system['autoencoder']
wavelet_transform = system['wavelet_transform']  # None if mode='direct'
data_adapter = system['data_adapter']           # 数据预处理器
parameter_mapper = system['parameter_mapper']

# 3. 数据预处理（自动mode-aware）
rcs_data = load_rcs_data()  # [N, 91, 91, 2]
if mode == 'wavelet':
    # 先小波变换（在原始数据上）
    wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
    # 再标准化（在小波系数上）
    input_data = data_adapter.adapt_rcs_data(wavelet_coeffs)
else:
    # Direct模式：直接预处理（自动dB+Z-score）
    input_data = data_adapter.adapt_rcs_data(rcs_data)

# 4. 训练AutoEncoder (Stage 1)
autoencoder.train()
# ... 训练代码 ...

# 5. 从参数预测RCS
params = np.array([[1.2, 0.8, 2.1, 1.5, 0.9, 1.8, 2.3, 1.1, 0.7]])
latent = parameter_mapper(torch.FloatTensor(params))
predicted = autoencoder.decode(latent)
# 逆预处理得到RCS
rcs_pred = data_adapter.inverse_adapt(predicted)
if mode == 'wavelet':
    rcs_pred = wavelet_transform.inverse_transform(rcs_pred)
```

---

## 🧠 核心概念

### 1. 双模式架构

#### Wavelet模式 🌊
```
原始RCS [91×91×2]
    ↓ 小波变换 (db4, symmetric)
小波系数 [49×49×8]  # 2freq × 4小波带 (LL/LH/HL/HH)
    ↓ Z-score标准化
    ↓ CNN Encoder
隐空间 [256]
    ↓ CNN Decoder
    ↓ 逆标准化
    ↓ 逆小波变换
重建RCS [91×91×2]
```

**优势**: 小波域稀疏表示，更好的特征压缩

#### Direct模式 🔄
```
原始RCS [91×91×2]
    ↓ dB变换 + Z-score标准化
    ↓ CNN Encoder
隐空间 [256]
    ↓ CNN Decoder
    ↓ 逆标准化 + 逆dB
重建RCS [91×91×2]
```

**优势**: 无小波计算开销，更快推理速度

### 2. Mode-aware数据预处理

**关键创新**：系统根据模式自动选择最优预处理策略

| 模式 | 数据特性 | 自动预处理策略 | 原因 |
|------|---------|--------------|------|
| **Direct** | RCS原始数据<br/>5.4M倍动态范围<br/>(0.00000009~0.5) | dB + Z-score | 压缩巨大动态范围<br/>→ 67dB范围 |
| **Wavelet** | 小波系数<br/>600倍范围<br/>37% negative values | Z-score only | 已压缩<br/>保留符号信息 |

**用户操作**：
- ✅ 勾选"数据标准化"
- 📊 dB变换复选框自动显示（禁用状态，由系统控制）
- 无需手动配置，系统自动应用正确策略

### 3. 三阶段训练

**Stage 1: AutoEncoder预训练**
```python
目标: 学习RCS数据的压缩表示
损失: MSE(重建, 原始)
训练: AutoEncoder全部参数
```

**Stage 2: 参数映射器训练**
```python
目标: 建立参数→隐空间的映射
损失: MSE(predicted_latent, autoencoder_latent)
训练: ParameterMapper参数（AutoEncoder冻结）
```

**Stage 3: 端到端微调**
```python
目标: 联合优化整个系统
损失: MSE(最终重建RCS, 原始RCS)
训练: AutoEncoder + ParameterMapper全部参数
```

---

## 📊 性能对比

### 网络架构对比

| 网络 | 参数量 | 推理速度 | 适用场景 | 感受野 |
|------|--------|---------|---------|--------|
| **WaveletAutoEncoder** | ~10M | 快 | 通用，推荐默认 | 31×31 |
| **DirectAutoEncoder** | ~7.5M | 最快 | 无小波开销 | 31×31 |
| **EnhancedWaveletAutoEncoder** | ~11M | 中等 | 复杂模式，更大感受野 | 67×67 ✨ |
| **EnhancedDirectAutoEncoder** | ~25M | 慢 | Direct模式最强表达力 | 全局 |
| **DeepWaveletAutoEncoder** | ~29M | 慢 | Wavelet模式最强表达力 | 全局 |
| **DeepDirectAutoEncoder** | ~79M | 很慢 | 最强表达力，计算密集 | 全局 |
| **WaveletMLPAutoEncoder** | ~404M | 很慢 | 参数敏感性分析 | 全局 |
| **DirectMLPAutoEncoder** | ~361M | 很慢 | 实验性 | 全局 |

### 模式对比

| 特性 | Wavelet模式 | Direct模式 |
|------|------------|-----------|
| **输入尺寸** | 49×49×8 | 91×91×2 |
| **小波计算** | 需要 | 不需要 |
| **数据压缩** | 小波域 + 网络 | 仅网络 |
| **推理速度** | 中等 | 快 |
| **重建质量** | 通常更好 | 取决于架构 |

---

## 🔧 高级配置

### 训练配置

通过GUI的"训练配置"按钮或训练配置对话框设置：

```python
{
    "batch_size": 8,                    # 批次大小
    "stage1_epochs": 100,               # Stage1训练轮数
    "stage2_epochs": 80,                # Stage2训练轮数
    "stage3_epochs": 50,                # Stage3训练轮数
    "learning_rate": 0.001,             # 初始学习率
    "min_lr": 1e-6,                     # 最小学习率
    "lr_scheduler": "cosine_restart",   # 学习率调度
    "patience_stage1": 20,              # Stage1早停耐心值
    "patience_stage2": 15,
    "patience_stage3": 15,
    "use_custom_loss": False            # 是否使用自定义损失
}
```

### 数据预处理配置

系统自动配置，无需手动设置：
```python
# 通过create_autoencoder_system自动创建
data_adapter = RCS_DataAdapter(
    normalize=True,      # 用户控制
    mode='wavelet',      # 自动根据模式决定dB是否启用
    expected_frequencies=2
)
```

---

## 📈 评估指标

### 重建质量
- **RMSE**: 均方根误差（线性域）
- **R²**: 决定系数
- **Correlation**: 相关系数
- **SSIM**: 结构相似性
- **PSNR**: 峰值信噪比

### 频率分离
- 单频重建质量
- 跨频率一致性

---

## 📝 文档目录

- **[CLAUDE.md](CLAUDE.md)**: 项目开发指令和规范（⭐ 重要）
- **[docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)**: 数据处理流程完整说明
- **[docs/architecture/MLP_Architecture.md](docs/architecture/MLP_Architecture.md)**: MLP架构详细解析
- **[docs/architecture/CNN_Receptive_Field_Analysis.md](docs/architecture/CNN_Receptive_Field_Analysis.md)**: CNN感受野分析与改进
- **[autoencoder/models/MODEL_INVENTORY.md](autoencoder/models/MODEL_INVENTORY.md)**: 模型清单
- **[autoencoder/models/experimental/README.md](autoencoder/models/experimental/README.md)**: 实验性模型说明

---

## 🐛 故障排除

### 常见问题

**Q: 训练loss很高，不收敛**
- 检查数据是否正确加载
- 确认标准化已启用
- 降低学习率或增加warmup

**Q: Direct模式和Wavelet模式该选哪个？**
- 默认推荐Wavelet模式（更好的特征压缩）
- 追求速度选Direct模式
- 可创建双系统进行对比分析

**Q: dB变换选项为什么是禁用的？**
- 这是正确的设计！系统根据模式自动决定
- Direct模式 + 标准化 → 自动启用dB
- Wavelet模式 → 不使用dB（保留小波系数符号）

**Q: 显存不足**
- 减小batch_size
- 使用更小的网络架构（CNN而非Deep_CNN）
- 降低隐空间维度

**Q: 频率配置不匹配错误**
- 确保模型和数据使用相同的频率配置（2freq或3freq）
- 重新创建系统匹配当前数据

---

## 📝 更新日志

### v2.2 (2025-01-15) - Mode-aware数据预处理
- ✅ **智能数据预处理**: Mode-aware自动策略
  - Direct模式自动启用dB变换压缩动态范围
  - Wavelet模式仅Z-score标准化
- ✅ **GUI改进**: dB选项改为只读自动显示
- ✅ **数据流程修复**: 小波变换在原始数据上运行（CRITICAL BUG FIX）
- ✅ **Stage 1 Only模式**: 新增仅训练AutoEncoder的模式
- ✅ **文档整理**: 完整的文档目录结构

### v2.1 (2025-01-14)
- ✅ **Deep CNN架构**: 添加深度卷积网络（4层+双卷积块）
- ✅ **模型统一接口**: 所有模型提供encoder/decoder属性
- ✅ **损失计算修复**: Sample-weighted averaging
- ✅ **模型组织优化**: 实验性模型移至experimental/

### v2.0 (AutoEncoder系统)
- ✅ **完整AutoEncoder系统**: 6+2个核心网络架构
- ✅ **三阶段训练**: Stage1→Stage2→Stage3完整流程
- ✅ **多频率支持**: 2freq/3freq灵活配置
- ✅ **双模式架构**: Wavelet + Direct完整支持
- ✅ **性能对比工具**: 系统级对比分析

---

## 🔮 开发路线

### 近期计划
- [ ] 变分AutoEncoder (VAE)
- [ ] 条件生成模型
- [ ] 注意力机制增强
- [ ] 模型量化和加速

### 长期计划
- [ ] 多GPU训练支持
- [ ] 实时推理优化
- [ ] Web界面开发
- [ ] 迁移学习框架

---

## 📧 许可和联系

**项目**: RCS AutoEncoder Prediction System
**版本**: v2.2
**Python**: 3.8+
**框架**: PyTorch 1.10+

**重要提示**: 使用前请仔细阅读[CLAUDE.md](CLAUDE.md)了解项目规范和开发指令。

---

> 💡 **提示**: 本项目专注于AutoEncoder架构的RCS预测，提供灵活的模式选择和架构配置。建议从默认的Wavelet + CNN配置开始，根据实际需求调整。
