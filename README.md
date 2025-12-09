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
| 架构 | 模型类 | 参数量 | 通道注意力 | 双分支 | 小隐空间 | 激活函数 | 推荐场景 |
|------|--------|--------|-----------|--------|---------|---------|---------|
| **CNN** | WaveletAutoEncoder | ~1.2M | ✅ | ❌ | ✅ | 可配置 | 通用，推荐默认 |
| **Enhanced_CNN** | EnhancedWaveletAutoEncoder | ~11M | ✅ | ❌ | ✅ | 可配置 | 多尺度+大感受野 |
| **Deep_CNN** | DeepWaveletAutoEncoder | ~29M | ✅ | ❌ | ✅ | 可配置 | 最强表达力 |
| **MLP** | WaveletMLPAutoEncoder | ~3M | ❌ | ❌ | ✅ | 可配置 | 参数敏感性分析 |
| **Dual_Branch_CNN** | DualBranchWaveletAutoEncoder | ~1.25M | ❌ | ✅ | ✅ | 可配置 | LL/HF分离处理 ⭐ |
| **Dual_Branch_MLP** | DualBranchWaveletMLPAutoEncoder | ~20M | ❌ | ✅ | ✅ | 可配置 | LL/HF分离+MLP |

#### Direct模式 (直接处理)
| 架构 | 模型类 | 参数量 | 通道注意力 | 双分支 | 小隐空间 | 激活函数 | 推荐场景 |
|------|--------|--------|-----------|--------|---------|---------|---------|
| **CNN** | DirectAutoEncoder | ~2.5M | ✅ | ❌ | ✅ | 可配置 | 无小波开销 |
| **Enhanced_CNN** | EnhancedDirectAutoEncoder | ~25M | ✅ | ❌ | ✅ | 可配置 | 最强Direct表达力 |
| **Deep_CNN** | DeepDirectAutoEncoder | ~79M | ✅ | ❌ | ✅ | 可配置 | 计算密集型 |
| **MLP** | DirectMLPAutoEncoder | ~5M | ❌ | ❌ | ✅ | 可配置 | 全连接Direct |

#### Differentiable Wavelet模式 (可微分小波)
| 架构 | 模型类 | 参数量 | 通道注意力 | 双分支 | 小隐空间 | 激活函数 | 推荐场景 |
|------|--------|--------|-----------|--------|---------|---------|---------|
| **CNN** | DifferentiableWaveletAutoEncoder | ~1.2M | ✅ | ❌ | ✅ | 可配置 | 端到端小波学习 |
| **MLP** | DifferentiableWaveletMLPAutoEncoder | ~3M | ✅ | ❌ | ✅ | 可配置 | 端到端+MLP |
| **Dual_Branch_CNN** | DualBranchDifferentiableWaveletAutoEncoder | ~1.25M | ❌ | ✅ | ✅ | 可配置 | 端到端+LL/HF分离 ⭐ |
| **Dual_Branch_MLP** | DualBranchDifferentiableWaveletMLPAutoEncoder | ~3M | ❌ | ✅ | ✅ | 可配置 | 端到端+MLP+分离 |

### 功能说明

#### 🔧 通道注意力 (Channel Attention)
- **支持网络**: CNN系列（Wavelet/Direct/Differentiable Wavelet模式）
- **功能**: 自动学习不同通道（LL/LH/HL/HH或频率）的相对重要性
- **注意**: 与Z-score标准化存在冲突，建议使用双分支架构替代
- **启用方式**: GUI中勾选"通道注意力"选项

#### 🌿 双分支架构 (Dual-Branch)
- **支持网络**: Wavelet模式和Differentiable Wavelet模式（Dual_Branch_CNN, Dual_Branch_MLP）
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

#### 🎨 激活函数参数化
- **支持网络**: 所有网络（12个核心模型）
- **可选激活函数**: relu (默认), sin, gelu, swish, tanh, sigmoid, mish, elu, leaky_relu, prelu
- **使用方式**:
  - **GUI**: 在模型配置中选择"激活函数"下拉框
  - **代码**: `create_autoencoder_system(..., activation='sin')`
- **特点**: 统一接口，无需为每个激活函数维护单独的模型类

#### 🧪 批量对比实验系统
- **用途**: 自动化对比不同配置的模型性能，支持超参数搜索、架构对比、消融实验
- **核心功能**:
  - **参数继承**: 从AE页面一键读取基准配置
  - **多维度对比**: 同时对比AE模式、架构类型、激活函数、预处理方案、小波类型
  - **自动评估**: 在训练集/测试集上计算MSE/RMSE/MAE
  - **丰富可视化**: 6种对比图表 + 单模型热图/残差图
- **支持对比维度**:
  - AE模式：wavelet, direct, differentiable_wavelet
  - 架构类型：CNN, Enhanced_CNN, Deep_CNN, MLP, Dual_Branch_CNN, Dual_Branch_MLP
  - 激活函数：relu, sin, gelu, swish, tanh, mish, elu, leaky_relu, prelu
  - 数据预处理：标准化方法(none/zscore/minmax) × dB变换(开/关)
  - 小波类型：db4, db8, haar, bior2.2（仅Wavelet模式）
- **生成内容**:
  - **对比图表**: 训练曲线、指标柱状图、综合性能雷达图、误差分布、收敛速度、训练时间对比
  - **单模型可视化**: 从训练/测试集各选3样本生成RCS热图对比和残差分布
  - **结果文件**: JSON配置、CSV汇总表、模型权重、评估指标
- **典型应用**:
  - 激活函数性能对比（哪个激活函数最适合RCS预测？）
  - 架构选择（CNN vs MLP vs Enhanced_CNN？）
  - 预处理方案评估（dB变换+标准化效果如何？）
  - 消融实验（逐步添加/移除组件观察性能变化）

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

##### ⭐ 从Stage 1继续训练（v0.3.0新增）

系统支持从Stage 1 Only模型继续完成三阶段训练：

**适用场景**:
- 已有Stage 1模型，想要添加参数映射功能
- 需要在不同数据集上微调已训练的AutoEncoder

**操作步骤**:
1. 在【AutoEncoder】页面加载Stage 1 Only模型（.pth文件）
2. 系统自动检测训练模式，弹出提示框：
   ```
   检测到该模型仅完成了Stage 1训练（AutoEncoder重建）

   是否要继续训练Stage 2（参数映射）和Stage 3（端到端微调）？
   ```
3. 选择"是"：系统自动执行Stage 2和Stage 3训练
4. 训练完成后，模型自动更新为Three Stage模式

**技术细节**:
- Stage 1的AutoEncoder权重保持不变作为初始化
- 自动创建并训练ParameterMapper
- 训练历史包含所有三个阶段的记录
- `training_mode`自动从`stage1_only`更新为`three_stage`

#### 4. **模型评估**
- 查看训练损失曲线
- 评估重建质量（MSE/RMSE/MAE）
- 可视化隐空间分布
- 对比不同架构性能

#### 5. **批量对比实验**（推荐用于超参数搜索）

批量实验系统可以自动对比不同配置的模型性能，节省大量手动调参时间。

##### 启动方式
```bash
python test_batch_experiment.py
```

##### 操作步骤

**Step 1: 配置基准参数**
- 在【AutoEncoder】页面设置所有训练参数（这些参数作为基准）
- 包括：模式、架构、epochs、batch_size、learning_rate、数据预处理等

**Step 2: 读取基准配置**
- 切换到【批量实验】标签页
- 点击 "🔄 从AE页面读取配置" 按钮
- 确认基准配置显示正确

**Step 3: 选择对比维度**
- 勾选要对比的维度（可多选）：
  - ☑️ **AE模式**：对比wavelet vs direct vs differentiable_wavelet
  - ☑️ **架构类型**：对比CNN vs Enhanced_CNN vs Deep_CNN vs MLP等
  - ☑️ **激活函数**：对比relu vs sin vs gelu vs swish vs mish等
  - ☑️ **数据预处理**：对比标准化方法(zscore/minmax/none) × dB变换(开/关)
  - ☑️ **小波类型**：对比db4 vs db8 vs haar vs bior2.2（仅Wavelet模式）

- 在每个维度下勾选要测试的具体值
- 点击 "🔢 计算实验数量" 查看将要运行的实验数量

**示例配置**:
```
基准配置: mode=wavelet, latent_dim=256, epochs=(50,30,20), batch_size=8
对比维度:
  ☑ 激活函数: ☑relu ☑sin ☑gelu
  → 将运行3个实验
```

**Step 4: 执行批量训练**
- 点击 "▶ 开始批量训练"
- 系统自动循环训练所有配置组合
- 实时显示进度：当前实验ID、总体进度、训练日志

**Step 5: 查看结果**
- 训练完成后，点击 "📊 查看结果" 打开实验目录
- 或手动打开：`batch_experiments/experiment_name_timestamp/`

##### 结果文件说明

实验完成后会生成完整的结果文件夹：

```
batch_experiments/activation_comparison_20250107_143000/
├── experiment_config.json          # 实验总配置
├── results_summary.csv             # 结果汇总表（可用Excel打开）
├── detailed_results.json           # 详细JSON结果
│
├── comparison_plots/               # 对比图表（6张）
│   ├── loss_curves.png            # 三阶段训练曲线对比
│   ├── metrics_bar.png            # MSE/RMSE/MAE柱状图
│   ├── radar_chart.png            # 综合性能雷达图
│   ├── error_distribution_box.png # 测试误差分布
│   ├── convergence_comparison.png # 收敛速度对比
│   └── training_time_comparison.png # 训练时间对比
│
├── models/                         # 所有模型文件
│   ├── relu_model.pth             # 模型权重
│   ├── relu_model_config.json     # 模型配置
│   ├── relu_evaluation.json       # 评估结果（Train/Test MSE/RMSE/MAE）
│   ├── sin_model.pth
│   ├── sin_model_config.json
│   └── ...
│
├── visualizations/                 # 单模型可视化（每模型6张图）
│   ├── relu/                      # 每个模型一个子目录
│   │   ├── relu_train_sample0_heatmap.png   # 训练集样本0热图
│   │   ├── relu_train_sample0_residual.png  # 训练集样本0残差
│   │   ├── relu_train_sample1_heatmap.png
│   │   ├── relu_train_sample1_residual.png
│   │   ├── relu_train_sample2_heatmap.png
│   │   ├── relu_train_sample2_residual.png
│   │   ├── relu_test_sample0_heatmap.png    # 测试集样本0热图
│   │   ├── relu_test_sample0_residual.png   # 测试集样本0残差
│   │   ├── relu_test_sample1_heatmap.png
│   │   └── ...
│   ├── sin/
│   │   └── ...
│   └── gelu/
│       └── ...
│
└── training_logs/                  # 训练日志和进度图 ⭐新增
    ├── relu_training_progress.png  # 每个模型的训练进度曲线
    ├── relu_wavelet_coeffs_comparison.png  # 小波系数对比（仅Wavelet模式）
    ├── sin_training_progress.png
    ├── sin_wavelet_coeffs_comparison.png
    └── batch_experiment.log        # 完整训练日志
```

**⭐ v0.3.0 新增可视化**:
- **训练进度图**: 每个模型自动生成三阶段训练曲线（Train/Val Loss对比）
- **小波系数对比图**: Wavelet模式下自动生成小波系数重建对比（LL/LH/HL/HH四通道）
- **模型ID显示**: 所有可视化图表自动标注实验序号，便于识别

##### 典型应用场景

**场景1: 激活函数性能对比**
```
目标: 找出最适合RCS预测的激活函数
配置:
  - 基准: wavelet + cnn + 256维隐空间
  - 对比维度: 激活函数
  - 测试值: relu, sin, gelu, swish, mish
结果: 查看metrics_bar.png和loss_curves.png决策
```

**场景2: 架构选择**
```
目标: 对比CNN vs MLP vs Enhanced_CNN
配置:
  - 基准: wavelet + relu激活 + zscore标准化
  - 对比维度: 架构类型
  - 测试值: CNN, MLP, Enhanced_CNN, Deep_CNN
结果: 根据性能和训练时间权衡选择
```

**场景3: 预处理方案评估**
```
目标: 验证dB变换和标准化方法的效果
配置:
  - 基准: wavelet + cnn + relu
  - 对比维度: 数据预处理
  - 测试值: none, zscore, minmax, zscore+db, minmax+db
结果: 评估预处理对收敛速度和最终性能的影响
```

**场景4: AE模式对比**
```
目标: 对比小波变换vs直接输入vs可微分小波的性能
配置:
  - 基准: cnn + relu激活 + zscore标准化 + 256维隐空间
  - 对比维度: AE模式
  - 测试值: wavelet, direct, differentiable_wavelet
结果: 评估不同输入方式对模型性能和训练效率的影响
```

**场景5: 消融实验**
```
目标: 研究通道注意力、双分支的作用
配置:
  - 基准: wavelet + 32维隐空间
  - 对比维度: 架构类型
  - 测试值: CNN, Dual_Branch_CNN
结果: 验证双分支在小隐空间场景的优势
```

##### 注意事项

⚠️ **训练时长**: 批量实验可能需要较长时间，建议：
- 先用少量epochs（如10,10,10）测试配置是否正确
- 确认无误后再用正式epochs（如50,30,20）

⚠️ **存储空间**: 每个模型约200-500MB（含权重和可视化），注意磁盘空间

⚠️ **数据集划分**: 系统自动按80/20划分训练集和测试集

⚠️ **中途停止**: 当前版本不支持优雅停止，建议等待当前实验完成

💡 **提示**: 批量实验结束后，可以将最佳模型的.pth文件复制到主目录，在【AutoEncoder】页面加载使用

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

### v0.4.0 (2025-01-18) 🔥

**🚀 重大功能**
- ✅ **继续训练功能**: 支持从已训练模型继续训练，区分"开始训练"和"继续训练"按钮
  - 加载模型后保留权重副本，可使用扩充数据集继续训练
  - 自动检测数据集变化并提示adapter统计信息重新计算
  - 训练历史保留并追加新记录
- ✅ **GUI配置自动恢复**: 加载模型时自动恢复所有配置到GUI界面
  - 自动恢复：模式、架构、激活函数、隐空间维度、学习率、epochs等
  - 用户可在恢复的基础上修改参数（如降低学习率）再继续训练
- ✅ **Dual-Branch V2架构**: 修复V1架构缺陷，实现正确的对称双分支设计
  - Encoder和Decoder都采用双分支（ll_branch + hf_branch）
  - ll_ratio参数真正控制隐空间分配（V1中硬编码为128）
  - 新增V2版本：`DualBranchDifferentiableWaveletAutoEncoderV2`和`DualBranchDifferentiableWaveletMLPAutoEncoderV2`

**✨ 新功能**
- ✅ **设计参数标准化**: 支持Z-score标准化设计参数，提升映射器泛化能力
- ✅ **隐空间统计信息**: 训练后自动打印隐空间维度分布统计
- ✅ **梯度监控集成**: AE训练进度可视化集成梯度监控曲线
- ✅ **RCS分布对比**: 新增RCS数值分布直方图对比功能
- ✅ **ReconstructionMetrics**: 集成完整的重建质量评估指标体系

**🔧 重要修复**
- ✅ **训练未保存最佳模型**: 修复AutoEncoder训练过程中未保存最佳模型的严重Bug
- ✅ **Stage 1训练Bug**: 修复Stage 1训练中parameter_mapper未定义的错误
- ✅ **decoder逆变换缺失**: 修复Three Stage评估和可视化中decoder输出未逆标准化的严重Bug
- ✅ **数据处理顺序**: 修复小波变换在标准化之后执行的严重错误（应在之前）
- ✅ **旧模型兼容性**: 修复加载旧模型时normalization_method推断错误

**🎨 用户体验改进**
- ✅ **训练模式简化**: 自动根据模型类型设置训练模式，移除加载时弹窗
- ✅ **GUI错误处理**: 修复数据加载时wavelet_model_selection属性缺失错误
- ✅ **状态栏增强**: 显示创建网络的完整参数信息
- ✅ **关闭确认**: 添加关闭程序时的确认对话框，防止误操作

**📚 文档更新**
- ✅ 新增`DUAL_BRANCH_V1_VS_V2_COMPARISON.md` - V1和V2架构详细对比
- ✅ 更新`CLAUDE.md` - 添加decoder逆变换规范和继续训练说明
- ✅ 更新`PARAMETERS_REFERENCE.md` - 所有参数命名规范
- ✅ 新增`DESIGN_ANALYSIS.md` - 架构设计问题分析

### v0.3.0 (2025-01-10) 🎉

**🚀 新功能**
- ✅ **从Stage 1继续训练**: 支持加载Stage 1 Only模型，自动继续完成Stage 2和Stage 3训练
- ✅ **批量实验训练进度图**: 批量实验为每个模型自动生成三阶段训练曲线
- ✅ **批量实验小波系数对比图**: Wavelet模式下自动生成小波系数重建对比可视化
- ✅ **模型ID标注**: 所有批量实验可视化图表自动标注实验序号

**🔧 代码优化**
- ✅ **统一绘图接口**: 提取`plot_ae_training_progress()`和`plot_wavelet_coefficients_comparison()`到`autoencoder/utils/plotting.py`
- ✅ **统一评估接口**: 重构批量实验评估流程，消除~200行重复代码
- ✅ **统一模型I/O**: 标准化模型保存/加载接口，修复键名不一致问题
- ✅ **完善批量实验日志**: 包含批量实验管理器的所有输出到单一日志文件

**📚 文档改进**
- ✅ 更新README批量实验可视化说明
- ✅ 添加从Stage 1继续训练使用指南
- ✅ 更新.gitignore忽略临时实验结果

**🐛 Bug修复**
- ✅ 修复批量实验模型ID显示为"N/A"的问题
- ✅ 修复批量实验模型保存键名不一致导致无法加载

### v0.2.0 (2025-01)
- ✅ 新增双分支架构（Dual_Branch_CNN/MLP）
- ✅ 完善小隐空间支持（16-32维）
- ✅ 整理文档到docs文件夹
- ✅ 更新功能兼容性矩阵
- ✅ 实现批量对比实验系统

### v0.1.1 (2024-10)
- ✅ 实现可微分小波模式
- ✅ 添加通道注意力机制

### v0.1.0 (2024-10)
- ✅ 完成三阶段训练流程
- ✅ 实现Wavelet和Direct两种模式
- ✅ 基础GUI界面

---

## 📧 联系方式

项目维护者: Claude Code

---

**⭐ 推荐配置**: Wavelet + Dual_Branch_CNN + 32维隐空间
