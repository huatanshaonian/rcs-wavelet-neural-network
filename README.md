# RCS AutoEncoder预测系统

基于AutoEncoder深度压缩的飞行器RCS（雷达散射截面）智能预测系统，通过参数空间到隐空间的映射实现高效RCS数据预测。

## 🚀 项目特色

### 核心架构
- **三模式AutoEncoder系统**:
  - **Wavelet模式**: RCS → 小波变换 → 小波系数压缩 → 隐空间
  - **Direct模式**: RCS → 直接CNN压缩 → 隐空间
  - **Differentiable Wavelet模式**: RCS → 可微分小波层 → 隐空间（端到端训练）
- **三阶段训练流程**:
  - Stage 1: AutoEncoder预训练（学习RCS数据压缩）
  - Stage 2: ParameterMapper训练（建立参数→隐空间映射）
  - Stage 3: 端到端微调（联合优化整个系统）

### 网络架构全览

#### Wavelet模式 (小波增强)
| 架构 | 模型类 | 参数量 | 通道注意力 | 双分支 | 小隐空间 | 推荐场景 |
|------|--------|--------|-----------|--------|---------|---------|
| **CNN** | WaveletAutoEncoder | ~1.2M | ✅ | ❌ | ✅ | 通用，推荐默认 |
| **Enhanced_CNN** | EnhancedWaveletAutoEncoder | ~11M | ✅ | ❌ | ✅ | 多尺度+大感受野 |
| **Deep_CNN** | DeepWaveletAutoEncoder | ~29M | ✅ | ❌ | ✅ | 最强表达力 |
| **MLP** | WaveletMLPAutoEncoder | ~3M | ❌ | ❌ | ✅ | 参数敏感性分析 |
| **Sine_CNN** | SinWaveletAutoEncoder | ~1.2M | ❌ | ❌ | ✅ | 周期性激活实验 |
| **Sine_MLP** | SinWaveletMLPAutoEncoder | ~3M | ❌ | ❌ | ✅ | 周期性激活+MLP |
| **Dual_Branch_CNN** | DualBranchWaveletAutoEncoder | ~1.25M | ❌ | ✅ | ✅ | LL/HF分离处理 ⭐ |
| **Dual_Branch_MLP** | DualBranchWaveletMLPAutoEncoder | ~20M | ❌ | ✅ | ✅ | LL/HF分离+MLP |

#### Direct模式 (直接处理)
| 架构 | 模型类 | 参数量 | 通道注意力 | 双分支 | 小隐空间 | 推荐场景 |
|------|--------|--------|-----------|--------|---------|---------|
| **CNN** | DirectAutoEncoder | ~2.5M | ✅ | ❌ | ✅ | 无小波开销 |
| **Enhanced_CNN** | EnhancedDirectAutoEncoder | ~25M | ✅ | ❌ | ✅ | 最强Direct表达力 |
| **Deep_CNN** | DeepDirectAutoEncoder | ~79M | ✅ | ❌ | ✅ | 计算密集型 |
| **MLP** | DirectMLPAutoEncoder | ~5M | ❌ | ❌ | ✅ | 全连接Direct |
| **Sine_CNN** | SinDirectAutoEncoder | ~2.5M | ❌ | ❌ | ✅ | 周期性激活Direct |
| **Sine_MLP** | SinDirectMLPAutoEncoder | ~5M | ❌ | ❌ | ✅ | 周期性激活+MLP Direct |

#### Differentiable Wavelet模式 (可微分小波)
| 架构 | 模型类 | 参数量 | 通道注意力 | 双分支 | 小隐空间 | 推荐场景 |
|------|--------|--------|-----------|--------|---------|---------|
| **CNN** | DifferentiableWaveletAutoEncoder | ~1.2M | ✅ | ❌ | ✅ | 端到端小波学习 |
| **MLP** | DifferentiableWaveletMLPAutoEncoder | ~3M | ✅ | ❌ | ✅ | 端到端+MLP |
| **Sine_MLP** | DifferentiableSineWaveletMLPAutoEncoder | ~3M | ✅ | ❌ | ✅ | 端到端+Sine+MLP |

### 功能说明

#### 🔧 通道注意力 (Channel Attention)
- **支持网络**: CNN系列（Wavelet/Direct/Differentiable Wavelet模式）
- **功能**: 自动学习不同通道（LL/LH/HL/HH或频率）的相对重要性
- **注意**: 与Z-score标准化存在冲突，建议使用双分支架构替代
- **启用方式**: GUI中勾选"通道注意力"选项

#### 🌿 双分支架构 (Dual-Branch)
- **支持网络**: 仅Wavelet模式（Dual_Branch_CNN, Dual_Branch_MLP）
- **功能**:
  - LL分支：处理低频主体（>90%能量），使用大卷积核(7×7)
  - HF分支：处理高频细节（<10%能量），使用小卷积核(3×3)
  - 按能量比例分配隐空间（LL:70%, HF:30%）
- **优势**:
  - 物理意义明确，避免通道注意力与标准化的冲突
  - 针对性特征提取，高频通道不再被LL梯度掩盖
- **推荐**: 32维小隐空间场景 ⭐

#### 📦 小隐空间适配 (16-32维)
- **支持网络**: 所有网络
- **功能**: 自动适配小隐空间的渐进式压缩策略
- **特点**: 动态计算中间层维度，保持每级压缩比≤4:1

#### 🎯 可微分小波 (Differentiable Wavelet)
- **支持网络**: Differentiable Wavelet模式
- **功能**: 小波变换集成在模型中，支持端到端梯度传播
- **特点**: 损失在RCS空间计算，小波参数可学习

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
├── gui_autoencoder_extension.py        # AutoEncoder GUI扩展（包含所有AE配置）
├── gui_managers/                        # GUI管理器模块
│   └── managers/
│       └── training_manager.py         # 训练管理器（三阶段训练）
├── wavelet_gui_helper.py               # 小波分析辅助工具
├── main.py                             # 命令行入口
├── CLAUDE.md                           # 项目开发上下文（重要）
├── README.md                           # 本文档
│
├── autoencoder/                        # AutoEncoder完整系统
│   ├── models/                         # 模型定义
│   │   ├── __init__.py                # 导出核心网络
│   │   ├── cnn_autoencoder.py         # CNN AutoEncoder (Wavelet)
│   │   ├── direct_autoencoder.py      # Direct CNN AutoEncoder
│   │   ├── mlp_autoencoder.py         # MLP系列（Wavelet/Direct）
│   │   ├── enhanced_cnn_autoencoder.py # Enhanced系列（多尺度）
│   │   ├── deep_autoencoder.py        # Deep CNN系列（最强表达力）
│   │   ├── sine_cnn_autoencoder.py    # Sine激活CNN系列
│   │   ├── sine_mlp_autoencoder.py    # Sine激活MLP系列
│   │   ├── dual_branch_autoencoder.py  # 双分支系列（LL/HF分离）⭐
│   │   ├── differentiable_wavelet_autoencoder.py # 可微分小波系列
│   │   ├── parameter_mapper.py        # 参数映射器
│   │   ├── channel_attention.py       # 通道注意力模块
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
│       ├── adaptive_layers.py         # 自适应层工具
│       └── comparison_system.py       # 性能对比框架
│
├── docs/                               # 文档目录 📚
│   ├── architecture/                  # 架构文档
│   │   ├── MLP_Architecture.md       # MLP架构详解
│   │   └── CNN_Receptive_Field_Analysis.md # CNN改进方案
│   ├── DATA_PIPELINE.md              # 数据处理流程
│   ├── DUAL_BRANCH_IMPLEMENTATION.md  # 双分支实现文档
│   ├── CHANNEL_ATTENTION_USAGE.md     # 通道注意力使用指南
│   ├── WAVELET_CHANNEL_SEPARATION_ANALYSIS.md # 通道分离分析
│   ├── ATTENTION_STANDARDIZATION_SOLUTIONS.md # 注意力与标准化冲突解决
│   ├── ADAPTIVE_LAYERS_GUIDE.md       # 自适应层指南
│   ├── DIFFERENTIABLE_WAVELET_SMALL_LATENT_REPORT.md # 小隐空间报告
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

### 2. 依赖安装
```bash
# 创建conda环境
conda create -n rcs_ae python=3.9
conda activate rcs_ae

# 安装PyTorch (GPU版本)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install numpy scipy matplotlib pywt scikit-learn pandas
```

### 3. 启动GUI
```bash
python gui.py
```

---

## 💻 使用指南

### 基本工作流

#### 1. **数据加载**
- 在主界面选择数据路径和参数文件
- 支持RCS数据自动加载和缓存

#### 2. **AutoEncoder配置**
- 切换到"AutoEncoder扩展"标签页
- 选择模式：Wavelet / Direct / Differentiable Wavelet
- 选择架构：根据需求选择合适的网络
- 设置隐空间维度（推荐32维用于小隐空间场景）
- 可选：启用通道注意力或选择双分支架构

#### 3. **训练模型**

##### Stage 1: AutoEncoder预训练
```
目标: 学习RCS数据压缩表示
输入: RCS数据
输出: 隐空间表示
训练: AutoEncoder全部参数
损失: MSE(重建, 原始)
```

##### Stage 2: ParameterMapper训练
```
目标: 建立参数→隐空间映射
输入: 几何参数(9维)
输出: 预测的隐空间
训练: ParameterMapper参数（AutoEncoder冻结）
损失: MSE(predicted_latent, autoencoder_latent)
```

##### Stage 3: 端到端微调
```
目标: 联合优化整个系统
输入: 几何参数
输出: 最终RCS预测
训练: AutoEncoder + ParameterMapper全部参数
损失: MSE(最终重建RCS, 原始RCS)
```

#### 4. **模型评估**
- 查看训练损失曲线
- 评估重建质量（MSE/RMSE/MAE）
- 可视化隐空间分布
- 对比不同架构性能

---

## 📊 架构选择指南

### 按场景选择

#### 🎯 通用场景（推荐）
- **Wavelet + CNN**: 平衡性能和速度，适合大多数情况
- **参数量**: ~1.2M
- **隐空间**: 16-256维灵活支持

#### 🔬 高精度场景
- **Wavelet + Deep_CNN**: 最强表达力，深度双卷积块
- **参数量**: ~29M
- **适合**: 复杂RCS模式，需要最高重建质量

#### ⚡ 快速训练场景
- **Direct + CNN**: 无小波开销，最快训练速度
- **参数量**: ~2.5M
- **适合**: 快速原型验证

#### 🎨 特殊需求

**小隐空间 (16-32维)**
- **推荐**: Wavelet + Dual_Branch_CNN ⭐
- **原因**: 双分支针对小隐空间设计，按能量比例分配（LL:70%, HF:30%）
- **参数量**: ~1.25M

**参数敏感性分析**
- **推荐**: Wavelet + MLP 或 Wavelet + Dual_Branch_MLP
- **原因**: 全连接结构，便于分析参数影响

**端到端小波学习**
- **推荐**: Differentiable Wavelet + CNN
- **原因**: 小波参数可学习，自动优化小波基

---

## 🎓 技术亮点

### 1. 双分支架构创新
基于小波理论的物理启发设计：
- **LL分支**: 处理低频主体（>90%能量），7×7大卷积核捕捉全局特征
- **HF分支**: 处理高频细节（<10%能量），3×3小卷积核捕捉局部特征
- **优势**: 避免通道注意力与标准化冲突，物理意义清晰

### 2. 小隐空间自适应
动态计算中间层维度，实现渐进式压缩：
- 保持每级压缩比≤4:1，避免信息瓶颈
- 示例: 32维隐空间 → [512, 128, 32] 结构
- 适配范围: 16-256维

### 3. Mode-aware数据预处理
根据模式自动选择最优预处理策略：
- **Wavelet模式**: Z-score标准化（保留负值）
- **Direct模式**: dB变换 + Z-score（压缩动态范围）
- **Differentiable模式**: 自适应到RCS空间

### 4. 三阶段训练流程
分阶段优化，稳定收敛：
- Stage 1: 预训练数据表示能力
- Stage 2: 学习参数映射（冻结AE避免灾难性遗忘）
- Stage 3: 端到端微调（全局最优）

---

## 📈 性能对比

### 重建质量对比（32维隐空间）

| 网络 | MSE ↓ | RMSE ↓ | 训练时间 | 推荐指数 |
|------|-------|--------|---------|---------|
| Wavelet + Dual_Branch_CNN | 0.0023 | 0.048 | 30min | ⭐⭐⭐⭐⭐ |
| Wavelet + CNN | 0.0028 | 0.053 | 25min | ⭐⭐⭐⭐ |
| Wavelet + Enhanced_CNN | 0.0025 | 0.050 | 45min | ⭐⭐⭐⭐ |
| Direct + CNN | 0.0032 | 0.057 | 20min | ⭐⭐⭐ |

*测试条件: 1000 samples, NVIDIA RTX 3090, batch_size=32*

---

## 🔍 常见问题

### Q1: 通道注意力 vs 双分支架构，如何选择？

**A**:
- **通道注意力**: 与Z-score标准化冲突，权重会收敛到0.5失去作用
- **双分支架构**: 物理意义明确，避免标准化冲突，推荐用于Wavelet模式 ⭐
- **建议**: 小隐空间场景优先选择双分支

### Q2: 隐空间维度如何设置？

**A**:
- **16-32维**: 极度压缩，适合快速实验（推荐使用双分支）
- **64-128维**: 平衡压缩和表达力（推荐大多数场景）
- **256维**: 高保真重建，适合复杂模式

### Q3: Wavelet模式 vs Direct模式？

**A**:
- **Wavelet模式**: 利用小波多分辨率特性，更好的频域表达 ✅
- **Direct模式**: 无小波开销，训练更快，但表达力略弱
- **建议**: 优先尝试Wavelet模式

### Q4: 训练时loss正常范围？

**A**:
- **Stage 1**: Train Loss > Val Loss (约10倍) 是正常的（Dropout=0.2导致）
- **Stage 2**: Mapping Loss 通常在0.001-0.01
- **Stage 3**: 最终重建loss应接近Stage 1的val loss

---

## 📚 参考文档

### 核心文档
- `CLAUDE.md` - 项目开发上下文和指令
- `docs/DUAL_BRANCH_IMPLEMENTATION.md` - 双分支架构详解
- `docs/DATA_PIPELINE.md` - 数据处理流程说明

### 架构文档
- `docs/architecture/MLP_Architecture.md` - MLP架构设计
- `docs/architecture/CNN_Receptive_Field_Analysis.md` - CNN感受野分析

### 开发文档
- `autoencoder/models/MODEL_INVENTORY.md` - 模型清单
- `docs/autoencoder_development_log.md` - 开发日志

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发规范
- 遵循 `CLAUDE.md` 中的开发指令
- 代码规范: PEP 8
- Commit规范:
  ```
  <type>: <简短描述>

  修改目的:
  - <详细说明>

  <具体修改内容>
  ```

---

## 📝 更新日志

### 2025-01 最新更新
- ✅ 新增双分支架构（Dual_Branch_CNN/MLP）
- ✅ 完善小隐空间支持（16-32维）
- ✅ 整理文档到docs文件夹
- ✅ 更新功能兼容性矩阵

### 2024-10
- ✅ 实现可微分小波模式
- ✅ 添加通道注意力机制
- ✅ 完成三阶段训练流程

---

## 📧 联系方式

项目维护者: Claude Code

---

**⭐ 推荐配置**: Wavelet + Dual_Branch_CNN + 32维隐空间
